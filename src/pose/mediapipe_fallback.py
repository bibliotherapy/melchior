"""MediaPipe-based pose estimation fallback for environments without MMPose.

Provides the same interface as MultiPersonPoseExtractor but uses MediaPipe
Pose, which works on CPU/Mac without CUDA. Detects a single person per
frame and assigns it as 'child'. Caregiver keypoints are set to zeros.

MediaPipe landmarks are mapped to COCO 17-joint format.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe landmark index -> COCO 17-joint index mapping
_MP_TO_COCO = {
    0: 0,    # nose -> nose
    2: 1,    # left eye -> left eye
    5: 2,    # right eye -> right eye
    7: 3,    # left ear -> left ear
    8: 4,    # right ear -> right ear
    11: 5,   # left shoulder -> left shoulder
    12: 6,   # right shoulder -> right shoulder
    13: 7,   # left elbow -> left elbow
    14: 8,   # right elbow -> right elbow
    15: 9,   # left wrist -> left wrist
    16: 10,  # right wrist -> right wrist
    23: 11,  # left hip -> left hip
    24: 12,  # right hip -> right hip
    25: 13,  # left knee -> left knee
    26: 14,  # right knee -> right knee
    27: 15,  # left ankle -> left ankle
    28: 16,  # right ankle -> right ankle
}


class MediaPipePoseExtractor:
    """Single-person pose extractor using MediaPipe Pose.

    Drop-in replacement for MultiPersonPoseExtractor on platforms
    where MMPose is not available.
    """

    def __init__(self, confidence_threshold=0.3, **kwargs):
        self.confidence_threshold = confidence_threshold
        self._pose = None

    def _ensure_model(self):
        if self._pose is None:
            import mediapipe as mp
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def extract_frame(self, frame):
        """Extract COCO 17-joint keypoints from a single frame.

        Returns:
            List of (17, 3) arrays. At most one person detected.
        """
        self._ensure_model()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)

        if not result.pose_landmarks:
            return []

        h, w = frame.shape[:2]
        landmarks = result.pose_landmarks.landmark

        coco_kp = np.zeros((17, 3), dtype=np.float32)
        for mp_idx, coco_idx in _MP_TO_COCO.items():
            lm = landmarks[mp_idx]
            coco_kp[coco_idx] = [lm.x * w, lm.y * h, lm.visibility]

        return [coco_kp]

    def extract_video(self, video_path, person_masks=None, sample_rate=1):
        """Extract keypoints for all frames, assigning to 'child'.

        Returns:
            Dict: {'child': (T, 17, 3), 'caregiver': (T, 17, 3)}
        """
        self._ensure_model()
        video_path = Path(video_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = (total_frames + sample_rate - 1) // sample_rate

        child_kp = np.zeros((processed_frames, 17, 3), dtype=np.float32)
        caregiver_kp = np.zeros((processed_frames, 17, 3), dtype=np.float32)

        frame_idx = 0
        out_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue

            skeletons = self.extract_frame(frame)
            if skeletons:
                child_kp[out_idx] = skeletons[0]

            out_idx += 1
            frame_idx += 1

            if out_idx % 100 == 0:
                logger.debug("Processed %d/%d frames", out_idx, processed_frames)

        cap.release()

        child_kp = child_kp[:out_idx]
        caregiver_kp = caregiver_kp[:out_idx]

        # Interpolate missing detections
        child_kp = _interpolate_missing(child_kp)

        logger.info(
            "MediaPipe extracted %d frames from %s (single-person mode)",
            out_idx, video_path.name,
        )
        return {"child": child_kp, "caregiver": caregiver_kp}


def _interpolate_missing(keypoints_seq):
    """Linearly interpolate frames where detection failed (all zeros)."""
    T = keypoints_seq.shape[0]
    if T < 2:
        return keypoints_seq

    has_detection = np.any(keypoints_seq[:, :, :2] != 0, axis=(1, 2))
    if has_detection.sum() < 2:
        return keypoints_seq

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
            interp_mask = (
                (invalid_indices >= valid_indices[0])
                & (invalid_indices <= valid_indices[-1])
            )
            if interp_mask.sum() == 0:
                continue
            interp_indices = invalid_indices[interp_mask]
            keypoints_seq[interp_indices, joint, coord] = np.interp(
                interp_indices, valid_indices, values[valid_indices],
            )

    return keypoints_seq
