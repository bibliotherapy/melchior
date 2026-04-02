"""Batch 3D triangulation for all clips.

Triangulates patient and caregiver 3D skeletons from multi-view 2D
detections using per-patient camera calibration. Saves (T, 17, 3) arrays.

Usage:
    python scripts/03_triangulate_3d.py
    python scripts/03_triangulate_3d.py --config configs/default.yaml
    python scripts/03_triangulate_3d.py --patient kku
"""

import argparse
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

from src.calibration.pose_calibration import load_calibration
from src.triangulation.triangulate_3d import SkeletonTriangulator
from src.utils.naming import clip_id_to_patient, clip_id_to_triplet_base, clip_id_to_view


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def discover_triplets(skeleton_2d_dir):
    """Discover and group 2D skeleton files into triplets.

    Returns:
        List of (triplet_base, {view: clip_id}) sorted by triplet_base.
    """
    skeleton_2d_dir = Path(skeleton_2d_dir)
    if not skeleton_2d_dir.exists():
        return []

    triplets = {}
    for npz_path in sorted(skeleton_2d_dir.glob("*.npz")):
        clip_id = npz_path.stem
        view = clip_id_to_view(clip_id)
        if view is None:
            continue
        triplet_base = clip_id_to_triplet_base(clip_id)

        if triplet_base not in triplets:
            triplets[triplet_base] = {}
        triplets[triplet_base][view] = clip_id

    return sorted(triplets.items())


def main():
    parser = argparse.ArgumentParser(description="Batch 3D triangulation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--patient", type=str, default=None,
                        help="Process only clips for this patient")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    skeleton_2d_dir = Path(config.get("skeleton_2d_dir", "data/skeleton_2d"))
    skeleton_3d_dir = Path(config.get("skeleton_3d_dir", "data/skeleton_3d"))
    calibration_dir = Path(config.get("calibration_dir", "data/calibration"))
    skeleton_3d_dir.mkdir(parents=True, exist_ok=True)

    triangulator = SkeletonTriangulator(config.get("triangulation", {}))

    all_triplets = discover_triplets(skeleton_2d_dir)
    if args.patient:
        all_triplets = [
            (tb, views) for tb, views in all_triplets
            if clip_id_to_patient(tb) == args.patient
        ]

    if not all_triplets:
        logger.warning(
            "No triplets found in %s. Run scripts/01_extract_2d_pose.py first.",
            skeleton_2d_dir,
        )
        sys.exit(0)

    logger.info("Found %d triplets to triangulate", len(all_triplets))

    processed = 0
    skipped = 0
    errors = 0

    # Cache loaded calibrations per patient
    cal_cache = {}

    for i, (triplet_base, views) in enumerate(all_triplets):
        out_path = skeleton_3d_dir / f"{triplet_base}.npz"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        patient_id = clip_id_to_patient(triplet_base)

        # Load calibration (cached per patient)
        if patient_id not in cal_cache:
            cal_path = calibration_dir / f"{patient_id}.npz"
            if not cal_path.exists():
                logger.warning(
                    "[%d/%d] No calibration for %s, skipping %s",
                    i + 1, len(all_triplets), patient_id, triplet_base,
                )
                errors += 1
                continue
            cal_cache[patient_id] = load_calibration(str(cal_path))

        calibration = cal_cache[patient_id]

        # Check minimum views
        if len(views) < 2:
            logger.warning(
                "[%d/%d] Only %d view(s) for %s, need >= 2",
                i + 1, len(all_triplets), len(views), triplet_base,
            )
            errors += 1
            continue

        logger.info(
            "[%d/%d] Triangulating %s: views=%s",
            i + 1, len(all_triplets), triplet_base, sorted(views.keys()),
        )

        start = time.time()

        try:
            # Load 2D keypoints for all views
            child_kp = {}
            caregiver_kp = {}
            for view, clip_id in views.items():
                data = np.load(str(skeleton_2d_dir / f"{clip_id}.npz"))
                child_kp[view] = data["child"]
                if "caregiver" in data:
                    caregiver_kp[view] = data["caregiver"]
                else:
                    caregiver_kp[view] = np.zeros_like(data["child"])

            # Triangulate
            result = triangulator.triangulate_clip(
                child_kp, caregiver_kp, calibration,
            )

            np.savez_compressed(
                str(out_path),
                child=result["child"],
                caregiver=result["caregiver"],
            )

            elapsed = time.time() - start
            T = result["child"].shape[0]
            logger.info(
                "[%d/%d] Done %s: %d frames in %.1fs",
                i + 1, len(all_triplets), triplet_base, T, elapsed,
            )
            processed += 1

        except Exception as e:
            logger.error(
                "[%d/%d] Failed %s: %s", i + 1, len(all_triplets), triplet_base, e,
            )
            errors += 1

    logger.info(
        "Triangulation complete: %d processed, %d skipped, %d errors out of %d total",
        processed, skipped, errors, len(all_triplets),
    )


if __name__ == "__main__":
    main()
