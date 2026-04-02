"""SAM2 video object tracking for child, caregiver, and walker identification.

Uses Meta's Segment Anything Model 2 (SAM2) to propagate manual first-frame
annotations through all frames of a video clip. Point prompts on the first
frame initialize object masks, which SAM2 then tracks automatically.

Typical usage:
    tracker = SAM2VideoTracker(model_cfg="sam2_hiera_l", checkpoint="checkpoints/sam2_hiera_large.pt")
    tracker.initialize_from_points(
        video_path="data/raw_synced/kku_w_01_FV.mp4",
        frame_idx=0,
        object_points={"child": (320, 400), "caregiver": (150, 300), "walker": (350, 500)},
    )
    masks = tracker.propagate()
    tracker.save_masks("data/processed/masks/kku_w_01_FV/")
"""

import json
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


def _load_sam2_predictor(model_cfg, checkpoint, device):
    """Load SAM2 video predictor, handling different SAM2 package versions."""
    try:
        from sam2.build_sam import build_sam2_video_predictor
        return build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    except ImportError:
        from segment_anything_2.build_sam import build_sam2_video_predictor
        return build_sam2_video_predictor(model_cfg, checkpoint, device=device)


class SAM2VideoTracker:
    """Tracks manually-specified objects through video using SAM2.

    Objects are initialized via point prompts on a single frame, then
    propagated bidirectionally through the entire clip.
    """

    def __init__(self, model_cfg="sam2_hiera_l", checkpoint="checkpoints/sam2_hiera_large.pt",
                 device="cuda:0"):
        """
        Args:
            model_cfg: SAM2 model configuration name.
            checkpoint: Path to SAM2 model checkpoint.
            device: Torch device string.
        """
        if "cuda" in device and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        self.device = device
        self.model_cfg = model_cfg
        self.checkpoint = checkpoint
        self._predictor = None
        self._masks = {}
        self._video_path = None
        self._frame_count = 0
        self._frame_size = (0, 0)

    def _ensure_predictor(self):
        if self._predictor is None:
            logger.info("Loading SAM2 model: %s", self.model_cfg)
            self._predictor = _load_sam2_predictor(
                self.model_cfg, self.checkpoint, self.device
            )

    def _read_video_info(self, video_path):
        """Read video metadata (frame count, dimensions)."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_size = (h, w)
        cap.release()
        return self._frame_count, self._frame_size

    def _extract_frames_to_dir(self, video_path, tmp_dir):
        """Extract video frames to a temporary directory as JPEG files.

        SAM2 video predictor expects a directory of frame images.
        """
        tmp_dir = Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = tmp_dir / f"{idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            idx += 1
        cap.release()
        return idx

    def initialize_from_points(self, video_path, frame_idx, object_points):
        """Initialize object tracking from point prompts on a single frame.

        Args:
            video_path: Path to video file.
            frame_idx: Frame index for initial annotation (usually 0).
            object_points: Dict mapping object names to (x, y) pixel coordinates.
                Example: {"child": (320, 400), "caregiver": (150, 300)}
                Values can be a single (x, y) tuple or a list of (x, y) tuples
                for multiple point prompts per object.
        """
        self._ensure_predictor()
        self._video_path = str(video_path)
        self._read_video_info(video_path)
        self._object_points = {}
        self._frame_idx = frame_idx

        for name, points in object_points.items():
            if points is None:
                continue
            if isinstance(points, tuple) and len(points) == 2 and not isinstance(points[0], tuple):
                points = [points]
            self._object_points[name] = points

        logger.info(
            "Initialized tracking for %d objects on frame %d: %s",
            len(self._object_points), frame_idx, list(self._object_points.keys())
        )

    def propagate(self, tmp_frame_dir=None):
        """Run SAM2 video propagation and return per-object masks.

        Args:
            tmp_frame_dir: Directory to store extracted frames. If None,
                uses a temp directory alongside the video file.

        Returns:
            Dict mapping object_name -> np.ndarray of shape (T, H, W), dtype bool.
        """
        if not self._object_points:
            raise RuntimeError("No objects initialized. Call initialize_from_points first.")

        video_path = Path(self._video_path)
        if tmp_frame_dir is None:
            tmp_frame_dir = video_path.parent / f".sam2_frames_{video_path.stem}"

        tmp_frame_dir = Path(tmp_frame_dir)
        num_frames = self._extract_frames_to_dir(video_path, tmp_frame_dir)
        logger.info("Extracted %d frames to %s", num_frames, tmp_frame_dir)

        with torch.inference_mode():
            state = self._predictor.init_state(video_path=str(tmp_frame_dir))

            # Add point prompts for each object
            obj_id_to_name = {}
            for obj_idx, (name, points) in enumerate(self._object_points.items(), start=1):
                point_coords = np.array(points, dtype=np.float32)
                point_labels = np.ones(len(points), dtype=np.int32)

                _, _, _ = self._predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=self._frame_idx,
                    obj_id=obj_idx,
                    points=point_coords,
                    labels=point_labels,
                )
                obj_id_to_name[obj_idx] = name

            # Propagate through all frames
            all_masks = {name: np.zeros((num_frames, *self._frame_size), dtype=bool)
                         for name in self._object_points}

            for frame_idx, obj_ids, masks in self._predictor.propagate_in_video(state):
                for obj_id, mask in zip(obj_ids, masks):
                    name = obj_id_to_name.get(obj_id)
                    if name is not None:
                        mask_np = (mask[0] > 0.0).cpu().numpy()
                        if mask_np.shape != self._frame_size:
                            mask_np = cv2.resize(
                                mask_np.astype(np.uint8),
                                (self._frame_size[1], self._frame_size[0]),
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(bool)
                        all_masks[name][frame_idx] = mask_np

            self._predictor.reset_state(state)

        # Clean up temp frames
        if tmp_frame_dir.exists():
            shutil.rmtree(tmp_frame_dir)

        self._masks = all_masks
        logger.info("Propagation complete: %s", {k: v.shape for k, v in all_masks.items()})
        return all_masks

    def save_masks(self, output_dir):
        """Save propagated masks to compressed .npz files.

        Args:
            output_dir: Directory to save masks. Creates one .npz per object:
                child.npz, caregiver.npz, walker.npz
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, mask_array in self._masks.items():
            out_path = output_dir / f"{name}.npz"
            np.savez_compressed(str(out_path), masks=mask_array)
            logger.info("Saved %s mask: %s (shape=%s)", name, out_path, mask_array.shape)

        # Save metadata
        meta = {
            "video_path": self._video_path,
            "frame_count": self._frame_count,
            "frame_size": list(self._frame_size),
            "objects": list(self._masks.keys()),
            "annotation_frame_idx": self._frame_idx,
            "object_points": {
                name: [list(p) for p in pts]
                for name, pts in self._object_points.items()
            },
        }
        meta_path = output_dir / "tracking_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def load_masks(mask_dir):
        """Load pre-computed masks from disk.

        Args:
            mask_dir: Directory containing .npz mask files.

        Returns:
            Dict mapping object_name -> np.ndarray of shape (T, H, W), dtype bool.
        """
        mask_dir = Path(mask_dir)
        masks = {}
        for npz_path in sorted(mask_dir.glob("*.npz")):
            name = npz_path.stem
            data = np.load(str(npz_path))
            masks[name] = data["masks"].astype(bool)
        return masks

    @staticmethod
    def load_metadata(mask_dir):
        """Load tracking metadata from disk."""
        meta_path = Path(mask_dir) / "tracking_meta.json"
        with open(meta_path) as f:
            return json.load(f)
