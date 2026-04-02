"""Batch SAM2 mask propagation for all annotated clips.

Reads first-frame annotations from sam2_annotations.json, runs SAM2 video
propagation for each clip, and saves per-object binary masks to disk.

This script must run BEFORE the main pipeline (scripts/01-06).

Usage:
    python scripts/00_propagate_masks.py
    python scripts/00_propagate_masks.py --config configs/default.yaml
    python scripts/00_propagate_masks.py --patient kku  # single patient
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.sam2_tracker import SAM2VideoTracker
from src.utils.naming import clip_id_to_patient


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_annotations(annotations_path):
    with open(annotations_path) as f:
        return json.load(f)


def get_object_points(annotation):
    """Extract object point prompts from annotation entry."""
    points = {}
    for obj in ["child", "caregiver", "walker"]:
        val = annotation.get(obj)
        if val is not None:
            points[obj] = tuple(val)
    return points


def main():
    parser = argparse.ArgumentParser(description="Batch SAM2 mask propagation")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to config YAML")
    parser.add_argument("--annotations", default="data/metadata/sam2_annotations.json",
                        help="Path to first-frame annotations")
    parser.add_argument("--patient", type=str, default=None,
                        help="Process only clips for this patient")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing masks")
    args = parser.parse_args()

    config = load_config(args.config)
    tracking_cfg = config.get("tracking", {})

    annotations_path = Path(args.annotations)
    if not annotations_path.exists():
        logger.error("Annotations not found: %s", annotations_path)
        logger.error("Run scripts/annotate_first_frame.py first.")
        sys.exit(1)

    annotations = load_annotations(annotations_path)
    logger.info("Loaded %d clip annotations", len(annotations))

    # Filter by patient if specified
    if args.patient:
        annotations = {
            k: v for k, v in annotations.items()
            if clip_id_to_patient(k) == args.patient
        }
        logger.info("Filtered to %d clips for patient: %s", len(annotations), args.patient)

    # Initialize SAM2 tracker
    model_cfg = tracking_cfg.get("model", "sam2_hiera_l")
    checkpoint = tracking_cfg.get("checkpoint", "checkpoints/sam2_hiera_large.pt")
    device = tracking_cfg.get("device", "cuda:0")
    mask_output_dir = Path(tracking_cfg.get("mask_output_dir", "data/processed/masks"))

    tracker = SAM2VideoTracker(
        model_cfg=model_cfg,
        checkpoint=checkpoint,
        device=device,
    )

    data_root = Path(config.get("raw_synced_dir", "data/raw_synced"))

    processed = 0
    skipped = 0
    errors = 0
    total = len(annotations)

    for i, (clip_id, ann) in enumerate(annotations.items()):
        clip_mask_dir = mask_output_dir / clip_id

        # Skip if already processed
        if clip_mask_dir.exists() and not args.overwrite:
            meta_path = clip_mask_dir / "tracking_meta.json"
            if meta_path.exists():
                skipped += 1
                continue

        # Resolve video path
        video_rel = ann.get("video_path", "")
        video_path = data_root / video_rel if video_rel else data_root / f"{clip_id}.mp4"
        if not video_path.exists():
            # Try common extensions
            for ext in [".mp4", ".avi", ".mov"]:
                candidate = data_root / f"{clip_id}{ext}"
                if candidate.exists():
                    video_path = candidate
                    break

        if not video_path.exists():
            logger.warning("[%d/%d] Video not found for %s, skipping", i + 1, total, clip_id)
            errors += 1
            continue

        # Get object points
        object_points = get_object_points(ann)
        if not object_points:
            logger.warning("[%d/%d] No object points for %s, skipping", i + 1, total, clip_id)
            errors += 1
            continue

        frame_idx = ann.get("frame_idx", 0)

        logger.info(
            "[%d/%d] Processing %s: objects=%s",
            i + 1, total, clip_id, list(object_points.keys())
        )

        try:
            start_time = time.time()

            tracker.initialize_from_points(
                video_path=video_path,
                frame_idx=frame_idx,
                object_points=object_points,
            )
            tracker.propagate()
            tracker.save_masks(clip_mask_dir)

            elapsed = time.time() - start_time
            processed += 1
            logger.info(
                "[%d/%d] Done %s in %.1fs (%d processed, %d skipped, %d errors)",
                i + 1, total, clip_id, elapsed, processed, skipped, errors
            )

        except Exception as e:
            logger.error("[%d/%d] Failed %s: %s", i + 1, total, clip_id, e)
            errors += 1

    logger.info(
        "Batch propagation complete: %d processed, %d skipped, %d errors out of %d total",
        processed, skipped, errors, total
    )


if __name__ == "__main__":
    main()
