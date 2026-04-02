"""Batch camera calibration per patient.

Uses matched human pose keypoints across three views to estimate
camera extrinsic parameters via Human Pose as Calibration Pattern
(Takahashi et al. [5]).

Usage:
    python scripts/02_calibrate_cameras.py
    python scripts/02_calibrate_cameras.py --config configs/default.yaml
    python scripts/02_calibrate_cameras.py --patient kku
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.pose_calibration import (
    CalibrationError,
    PoseCalibrator,
    flatten_calibration,
)
from src.utils.naming import clip_id_to_patient, clip_id_to_triplet_base, clip_id_to_view


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def discover_triplets(skeleton_2d_dir):
    """Discover and group 2D skeleton files into patient -> triplet -> views.

    Returns:
        {patient_id: {triplet_base: {view: clip_id}}}
    """
    skeleton_2d_dir = Path(skeleton_2d_dir)
    if not skeleton_2d_dir.exists():
        return {}

    patients = {}
    for npz_path in sorted(skeleton_2d_dir.glob("*.npz")):
        clip_id = npz_path.stem
        view = clip_id_to_view(clip_id)
        if view is None:
            continue
        triplet_base = clip_id_to_triplet_base(clip_id)
        patient_id = clip_id_to_patient(clip_id)

        if patient_id not in patients:
            patients[patient_id] = {}
        if triplet_base not in patients[patient_id]:
            patients[patient_id][triplet_base] = {}
        patients[patient_id][triplet_base][view] = clip_id

    # Warn about incomplete triplets
    for pid, triplets in patients.items():
        for tb, views in triplets.items():
            if len(views) < 3:
                missing = {"FV", "LV", "RV"} - set(views.keys())
                logger.debug("Incomplete triplet %s: missing %s", tb, missing)

    return patients


def main():
    parser = argparse.ArgumentParser(description="Batch camera calibration per patient")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--patient", type=str, default=None,
                        help="Calibrate a single patient")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    skeleton_2d_dir = Path(config.get("skeleton_2d_dir", "data/skeleton_2d"))
    calibration_dir = Path(config.get("calibration_dir", "data/calibration"))
    calibration_dir.mkdir(parents=True, exist_ok=True)

    calibrator = PoseCalibrator(config.get("calibration", {}))
    triplet_map = discover_triplets(skeleton_2d_dir)

    if not triplet_map:
        logger.warning(
            "No 2D keypoints found in %s. Run scripts/01_extract_2d_pose.py first.",
            skeleton_2d_dir,
        )
        sys.exit(0)

    patients = sorted(triplet_map.keys())
    if args.patient:
        patients = [p for p in patients if p == args.patient]
        if not patients:
            logger.error("Patient %s not found in 2D keypoints", args.patient)
            sys.exit(1)

    logger.info("Calibrating %d patients", len(patients))

    calibrated = 0
    failed = 0

    for i, patient_id in enumerate(patients):
        out_path = calibration_dir / f"{patient_id}.npz"
        if out_path.exists() and not args.overwrite:
            logger.debug("Skipping %s (already calibrated)", patient_id)
            continue

        triplets = triplet_map[patient_id]
        num_triplets = len(triplets)
        views_available = set()
        for tb, views in triplets.items():
            views_available.update(views.keys())

        logger.info(
            "[%d/%d] Calibrating %s: %d triplets, views=%s",
            i + 1, len(patients), patient_id, num_triplets, sorted(views_available),
        )

        # Load child keypoints for all triplets
        patient_kp = {}
        for triplet_base, views in triplets.items():
            patient_kp[triplet_base] = {}
            for view, clip_id in views.items():
                data = np.load(str(skeleton_2d_dir / f"{clip_id}.npz"))
                patient_kp[triplet_base][view] = data["child"]

        try:
            calibration = calibrator.calibrate_patient(patient_kp)
            flat = flatten_calibration(calibration)
            np.savez_compressed(str(out_path), **flat)

            quality = calibration.get("quality", {})
            mean_err = np.mean(list(quality.values())) if quality else -1
            logger.info(
                "[%d/%d] Saved %s: views=%s, mean_reproj=%.2f px",
                i + 1, len(patients), out_path.name,
                calibration["calibrated_views"], mean_err,
            )
            calibrated += 1

        except CalibrationError as e:
            logger.error("[%d/%d] Failed %s: %s", i + 1, len(patients), patient_id, e)
            failed += 1

    logger.info(
        "Calibration complete: %d calibrated, %d failed out of %d patients",
        calibrated, failed, len(patients),
    )


if __name__ == "__main__":
    main()
