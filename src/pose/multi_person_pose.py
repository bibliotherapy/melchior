"""Multi-person 2D pose estimation using MMPose RTMPose.

Detects all persons in each frame across three camera views (front/left/right).
Outputs per-frame 2D keypoints for all detected persons, then assigns them
to identity masks (child, caregiver) produced by SAM2.

Pipeline:
    1. RTMDet-m detects person bounding boxes
    2. RTMPose-L estimates 17 COCO keypoints per detected person
    3. Person identifier assigns skeletons to SAM2 masks (or height fallback)
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# COCO 17 keypoint names for reference
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def _load_mmpose_inferencer(det_model, pose_model, device):
    """Load MMPose inferencer with specified models."""
    try:
        from mmpose.apis import MMPoseInferencer
        return MMPoseInferencer(
            pose2d=pose_model,
            det_model=det_model,
            device=device,
        )
    except ImportError:
        raise ImportError(
            "mmpose is required. Install with: "
            "pip install -U openmim && mim install mmpose mmdet"
        )


class MultiPersonPoseExtractor:
    """Extracts 2D keypoints for all detected persons per frame.

    Uses RTMDet for person detection and RTMPose for keypoint estimation.
    """

    def __init__(self, det_model="rtmdet-m", pose_model="rtmpose-l",
                 device="cuda:0", batch_size=16, confidence_threshold=0.3):
        """
        Args:
            det_model: Person detector model name.
            pose_model: Pose estimation model name.
            device: Torch device.
            batch_size: Batch size for inference.
            confidence_threshold: Minimum keypoint confidence.
        """
        self.det_model = det_model
        self.pose_model = pose_model
        self.device = device
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self._inferencer = None

    def _ensure_model(self):
        if self._inferencer is None:
            logger.info("Loading MMPose: det=%s, pose=%s", self.det_model, self.pose_model)
            self._inferencer = _load_mmpose_inferencer(
                self.det_model, self.pose_model, self.device
            )

    def extract_frame(self, frame):
        """Extract keypoints for all persons in a single frame.

        Args:
            frame: (H, W, 3) BGR image array.

        Returns:
            List of (17, 3) arrays, one per detected person.
            Each row is (x, y, confidence).
        """
        self._ensure_model()

        results = next(self._inferencer(frame, return_vis=False))
        predictions = results.get("predictions", [[]])[0]

        skeletons = []
        for pred in predictions:
            keypoints = np.array(pred["keypoints"], dtype=np.float32)
            scores = np.array(pred["keypoint_scores"], dtype=np.float32)
            skeleton = np.column_stack([keypoints, scores])  # (17, 3)
            skeletons.append(skeleton)

        return skeletons

    def extract_video(self, video_path, person_masks=None, sample_rate=1):
        """Extract identified keypoints for all frames in a video.

        Args:
            video_path: Path to video file.
            person_masks: Optional dict mapping identity -> (T, H, W) bool array
                from SAM2. Used for mask-guided skeleton assignment.
            sample_rate: Process every Nth frame (1 = all frames).

        Returns:
            Dict mapping identity name -> (T, 17, 3) array of keypoints.
            T is the number of processed frames.
        """
        from .person_identifier import identify_persons

        self._ensure_model()
        video_path = Path(video_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = (total_frames + sample_rate - 1) // sample_rate

        # Determine identity names
        if person_masks is not None:
            identity_names = list(person_masks.keys())
        else:
            identity_names = ["child", "caregiver"]

        # Pre-allocate output arrays
        results = {name: np.zeros((processed_frames, 17, 3), dtype=np.float32)
                   for name in identity_names}

        frame_idx = 0
        out_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue

            # Detect all persons
            all_skeletons = self.extract_frame(frame)

            # Get masks for this frame (use frame_idx, not out_idx, since
            # mask arrays are indexed by original video frame number)
            frame_masks = None
            if person_masks is not None:
                frame_masks = {}
                for name, mask_array in person_masks.items():
                    if name in ("child", "caregiver") and frame_idx < mask_array.shape[0]:
                        frame_masks[name] = mask_array[frame_idx]

            # Assign skeletons to identities
            assignments = identify_persons(
                all_skeletons,
                person_masks=frame_masks,
                conf_threshold=self.confidence_threshold,
            )

            # Store results
            for name in identity_names:
                skeleton = assignments.get(name)
                if skeleton is not None:
                    results[name][out_idx] = skeleton

            out_idx += 1
            frame_idx += 1

            if out_idx % 100 == 0:
                logger.debug("Processed %d/%d frames", out_idx, processed_frames)

        cap.release()

        # Trim to actual processed count
        for name in identity_names:
            results[name] = results[name][:out_idx]

        # Interpolate missing detections
        for name in identity_names:
            results[name] = _interpolate_missing(results[name])

        logger.info(
            "Extracted %d frames from %s: identities=%s",
            out_idx, video_path.name, list(results.keys())
        )
        return results


def _interpolate_missing(keypoints_seq):
    """Linearly interpolate frames where detection failed (all zeros).

    Args:
        keypoints_seq: (T, 17, 3) array.

    Returns:
        (T, 17, 3) array with missing frames interpolated.
    """
    T = keypoints_seq.shape[0]
    if T < 2:
        return keypoints_seq

    # Find frames with detections (non-zero)
    has_detection = np.any(keypoints_seq[:, :, :2] != 0, axis=(1, 2))

    if has_detection.sum() < 2:
        return keypoints_seq

    detected_indices = np.where(has_detection)[0]

    for joint in range(17):
        for coord in range(3):
            values = keypoints_seq[:, joint, coord]
            valid_mask = has_detection & (values != 0) if coord < 2 else has_detection
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) < 2:
                continue
            invalid_indices = np.where(~valid_mask)[0]
            if len(invalid_indices) == 0:
                continue
            # Only interpolate between first and last valid frames
            interp_mask = (invalid_indices >= valid_indices[0]) & \
                          (invalid_indices <= valid_indices[-1])
            if interp_mask.sum() == 0:
                continue
            interp_indices = invalid_indices[interp_mask]
            keypoints_seq[interp_indices, joint, coord] = np.interp(
                interp_indices, valid_indices, values[valid_indices]
            )

    return keypoints_seq
