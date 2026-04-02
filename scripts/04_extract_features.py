"""Batch feature extraction for all clips.

Extracts Layer 1 (skeleton features), Layer 2 (interaction features),
Layer 3 (context vector), and walker spatial features for every clip.
Saves to features/.

Usage:
    python scripts/04_extract_features.py
    python scripts/04_extract_features.py --config configs/default.yaml
    python scripts/04_extract_features.py --patient kku
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

from src.features.context_vector import ContextVectorEncoder
from src.features.movement_quality import extract_movement_quality_features
from src.features.skeleton_features import extract_skeleton_features
from src.features.walker_features import extract_walker_features
from src.tracking.sam2_tracker import SAM2VideoTracker
from src.utils.naming import clip_id_to_movement, clip_id_to_patient


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_3d_skeleton(skeleton_3d_dir, clip_id):
    """Load triangulated 3D skeleton for a clip.

    Returns:
        Dict with 'child' and 'caregiver' arrays of shape (T, 17, 3).
    """
    path = Path(skeleton_3d_dir) / f"{clip_id}.npz"
    if not path.exists():
        return None
    data = np.load(str(path))
    return {k: data[k] for k in data.files}


def load_2d_keypoints(skeleton_2d_dir, clip_id):
    """Load 2D keypoints for a clip (pixel coordinates).

    Walker spatial features require 2D pixel-space keypoints to match
    SAM2 masks, not 3D triangulated coordinates.

    If clip_id is a triplet base (no view suffix), loads the FV (front
    view) file since walker features are best observed from the front.

    Returns:
        Dict with 'child' and 'caregiver' arrays of shape (T, 17, 3).
    """
    path = Path(skeleton_2d_dir) / f"{clip_id}.npz"
    if not path.exists():
        fv_path = Path(skeleton_2d_dir) / f"{clip_id}_FV.npz"
        if fv_path.exists():
            path = fv_path
        else:
            return None
    data = np.load(str(path))
    return {k: data[k] for k in data.files}


def load_walker_masks(mask_dir, clip_id):
    """Load walker mask for a clip if available.

    Falls back to FV (front view) mask if clip_id is a triplet base.
    """
    clip_mask_dir = Path(mask_dir) / clip_id
    if not clip_mask_dir.exists():
        fv_mask_dir = Path(mask_dir) / f"{clip_id}_FV"
        if fv_mask_dir.exists():
            clip_mask_dir = fv_mask_dir
        else:
            return None
    masks = SAM2VideoTracker.load_masks(clip_mask_dir)
    return masks.get("walker")


def save_features(output_dir, clip_id, features_dict):
    """Save extracted features to .npz file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{clip_id}.npz"
    np.savez_compressed(str(out_path), **features_dict)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Batch feature extraction")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--patient", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    skeleton_3d_dir = config.get("skeleton_3d_dir", "data/skeleton_3d")
    skeleton_2d_dir = config.get("skeleton_2d_dir", "data/skeleton_2d")
    features_dir = config.get("features_dir", "data/features")
    mask_dir = config.get("tracking", {}).get("mask_output_dir", "data/processed/masks")
    metadata_dir = config.get("metadata_dir", "data/metadata")
    features_cfg = config.get("features", {})
    fps = features_cfg.get("fps", 30)
    walker_threshold = features_cfg.get("walker_proximity_threshold_px", 30)

    # Initialize context vector encoder
    ann_path = Path(metadata_dir) / "assistive_annotations.json"
    lab_path = Path(metadata_dir) / "labels.json"
    context_encoder = ContextVectorEncoder(str(ann_path), str(lab_path))

    # Find all 3D skeleton files
    skeleton_files = sorted(Path(skeleton_3d_dir).glob("*.npz"))
    if args.patient:
        skeleton_files = [f for f in skeleton_files if f.stem.startswith(args.patient)]

    logger.info("Found %d clips to process", len(skeleton_files))

    processed = 0
    for i, skel_path in enumerate(skeleton_files):
        clip_id = skel_path.stem
        out_path = Path(features_dir) / f"{clip_id}.npz"

        if out_path.exists() and not args.overwrite:
            continue

        logger.info("[%d/%d] Extracting features: %s", i + 1, len(skeleton_files), clip_id)

        # Load 3D skeletons
        skeletons = load_3d_skeleton(skeleton_3d_dir, clip_id)
        if skeletons is None:
            logger.warning("No 3D skeleton for %s, skipping", clip_id)
            continue

        child_kp_3d = skeletons.get("child", np.zeros((1, 17, 3)))
        caregiver_kp_3d = skeletons.get("caregiver", np.zeros_like(child_kp_3d))

        # Load 2D keypoints for walker spatial features (pixel-space to match SAM2 masks)
        kp_2d = load_2d_keypoints(skeleton_2d_dir, clip_id)
        if kp_2d is not None:
            child_kp_2d = kp_2d.get("child", np.zeros_like(child_kp_3d))
            caregiver_kp_2d = kp_2d.get("caregiver", np.zeros_like(child_kp_3d))
        else:
            child_kp_2d = np.zeros_like(child_kp_3d)
            caregiver_kp_2d = np.zeros_like(child_kp_3d)

        patient_id = clip_id_to_patient(clip_id)

        # Layer 1: Skeleton features from 3D keypoints
        layer1_features = extract_skeleton_features(child_kp_3d, features_cfg)

        # Layer 1 extended: Movement quality features (per-clip summary)
        movement_type = clip_id_to_movement(clip_id)
        movement_quality_features = extract_movement_quality_features(
            child_kp_3d, movement_type, features_cfg
        )

        # Layer 2: Interaction features from 3D keypoints (placeholder)
        # TODO: implement interaction_features.extract_interaction_features(child_kp_3d, caregiver_kp_3d)
        layer2_features = np.zeros((child_kp_3d.shape[0], 10), dtype=np.float32)

        # Layer 3: Context vector
        try:
            context_vec = context_encoder.encode(patient_id)
        except KeyError:
            logger.warning("No annotation for patient %s, using zero vector", patient_id)
            context_vec = np.zeros(18, dtype=np.float32)

        # Walker spatial features (2D pixel-space keypoints + SAM2 masks)
        walker_masks = load_walker_masks(mask_dir, clip_id)
        walker_feats = extract_walker_features(
            child_keypoints=child_kp_2d,
            walker_masks=walker_masks,
            caregiver_keypoints=caregiver_kp_2d,
            proximity_threshold_px=walker_threshold,
            fps=fps,
        )

        # Save all features
        save_features(features_dir, clip_id, {
            "skeleton_features": layer1_features,
            "movement_quality_features": movement_quality_features,
            "interaction_features": layer2_features,
            "context_vector": context_vec,
            "walker_features": walker_feats,
        })

        processed += 1

    logger.info("Feature extraction complete: %d clips processed", processed)


if __name__ == "__main__":
    main()
