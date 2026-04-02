"""Batch 2D pose extraction across all triplets.

Runs multi-person RTMPose on every video in raw_synced/,
uses SAM2 masks for person identification when available,
saves per-frame identified 2D keypoints to skeleton_2d/.

Usage:
    python scripts/01_extract_2d_pose.py
    python scripts/01_extract_2d_pose.py --config configs/default.yaml
    python scripts/01_extract_2d_pose.py --patient kku
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pose.multi_person_pose import MultiPersonPoseExtractor
from src.tracking.sam2_tracker import SAM2VideoTracker


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def discover_videos(raw_dir):
    """Find all video files in the raw synced directory."""
    raw_dir = Path(raw_dir)
    videos = []
    for ext in [".mp4", ".avi", ".mov"]:
        videos.extend(raw_dir.rglob(f"*{ext}"))
    videos.sort(key=lambda p: p.stem)
    return videos


def load_masks_for_clip(mask_dir, clip_id):
    """Load SAM2 masks for a clip if they exist.

    Returns:
        Dict of person masks (child, caregiver) or None.
    """
    clip_mask_dir = Path(mask_dir) / clip_id
    if not clip_mask_dir.exists():
        return None

    masks = SAM2VideoTracker.load_masks(clip_mask_dir)

    # Only return person masks (child, caregiver), not walker
    person_masks = {}
    for name in ["child", "caregiver"]:
        if name in masks:
            person_masks[name] = masks[name]

    return person_masks if person_masks else None


def save_keypoints(output_dir, clip_id, keypoints_dict):
    """Save identified keypoints to .npz file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{clip_id}.npz"

    save_data = {}
    for name, kp_array in keypoints_dict.items():
        save_data[name] = kp_array

    np.savez_compressed(str(out_path), **save_data)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Batch 2D pose extraction")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--patient", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    raw_dir = config.get("raw_synced_dir", "data/raw_synced")
    output_dir = config.get("skeleton_2d_dir", "data/skeleton_2d")
    mask_dir = config.get("tracking", {}).get("mask_output_dir", "data/processed/masks")

    pose_cfg = config.get("pose", {})
    extractor = MultiPersonPoseExtractor(
        det_model=pose_cfg.get("detector", "rtmdet-m"),
        pose_model=pose_cfg.get("model", "rtmpose-l"),
        device=pose_cfg.get("device", "cuda:0"),
        batch_size=pose_cfg.get("batch_size", 16),
        confidence_threshold=pose_cfg.get("confidence_threshold", 0.3),
    )

    videos = discover_videos(raw_dir)
    if args.patient:
        videos = [v for v in videos if v.stem.startswith(args.patient)]

    logger.info("Found %d videos to process", len(videos))

    processed = 0
    mask_guided = 0
    height_fallback = 0

    for i, video_path in enumerate(videos):
        clip_id = video_path.stem
        out_path = Path(output_dir) / f"{clip_id}.npz"

        if out_path.exists() and not args.overwrite:
            continue

        logger.info("[%d/%d] Processing: %s", i + 1, len(videos), clip_id)
        start = time.time()

        # Load SAM2 masks if available
        person_masks = load_masks_for_clip(mask_dir, clip_id)
        if person_masks is not None:
            mask_guided += 1
        else:
            height_fallback += 1

        try:
            keypoints = extractor.extract_video(
                video_path=video_path,
                person_masks=person_masks,
            )
            save_keypoints(output_dir, clip_id, keypoints)
            processed += 1

            elapsed = time.time() - start
            method = "mask-guided" if person_masks else "height-fallback"
            logger.info(
                "[%d/%d] Done %s (%.1fs, %s)",
                i + 1, len(videos), clip_id, elapsed, method
            )

        except Exception as e:
            logger.error("[%d/%d] Failed %s: %s", i + 1, len(videos), clip_id, e)

    logger.info(
        "Complete: %d processed (%d mask-guided, %d height-fallback)",
        processed, mask_guided, height_fallback
    )


if __name__ == "__main__":
    main()
