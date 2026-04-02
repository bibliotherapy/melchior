"""Visual verification of 2D pose estimation results.

Generates overlaid videos showing detected skeletons and SAM2 masks
on sample clips for human review of identity assignment quality.

Usage:
    python scripts/verify_2d_pose.py
    python scripts/verify_2d_pose.py --config configs/default.yaml
    python scripts/verify_2d_pose.py --patient kku
    python scripts/verify_2d_pose.py --clip kku_w_01_FV
    python scripts/verify_2d_pose.py --level 3 --samples 3
    python scripts/verify_2d_pose.py --all
    python scripts/verify_2d_pose.py --video-dir data/raw
    python scripts/verify_2d_pose.py --no-masks
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.sam2_tracker import SAM2VideoTracker
from src.utils.naming import clip_id_to_patient
from src.utils.visualization import (
    draw_verification_frame,
    write_verification_video,
)


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_labels(metadata_dir):
    """Load patient GMFCS labels.

    Returns:
        Dict mapping patient_id -> gmfcs_level, or empty dict if unavailable.
    """
    labels_path = Path(metadata_dir) / "labels.json"
    if not labels_path.exists():
        logger.warning("labels.json not found at %s", labels_path)
        return {}
    with open(labels_path) as f:
        data = json.load(f)
    return {p["patient_id"]: p["gmfcs_level"] for p in data["patients"]}


def discover_keypoints(skeleton_2d_dir):
    """Find all available 2D keypoint files.

    Returns:
        List of clip_id strings (stems of .npz files).
    """
    skeleton_2d_dir = Path(skeleton_2d_dir)
    if not skeleton_2d_dir.exists():
        return []
    return sorted(p.stem for p in skeleton_2d_dir.glob("*.npz"))


def select_clips(available_clips, labels, mask_dir, patient=None, clip=None,
                 level=None, samples=2, process_all=False):
    """Select clips for verification review.

    Args:
        available_clips: List of all available clip_id strings.
        labels: Dict patient_id -> gmfcs_level.
        mask_dir: Path to mask directory (to detect mask-guided clips).
        patient: Filter to single patient.
        clip: Filter to single clip.
        level: Filter to single GMFCS level.
        samples: Number of clips per level in auto mode.
        process_all: Process all clips.

    Returns:
        List of (clip_id, gmfcs_level) tuples.
    """
    if clip:
        matched = [c for c in available_clips if c == clip]
        if not matched:
            logger.warning("Clip %s not found in available keypoints", clip)
            return []
        pid = clip_id_to_patient(clip)
        lvl = labels.get(pid, 0)
        return [(clip, lvl)]

    # Map clips to levels
    clips_by_level = {i: [] for i in range(1, 6)}
    unleveled = []

    for cid in available_clips:
        pid = clip_id_to_patient(cid)
        if patient and pid != patient:
            continue
        lvl = labels.get(pid)
        if lvl is not None:
            clips_by_level[lvl].append(cid)
        else:
            unleveled.append(cid)

    if process_all or patient:
        result = []
        for lvl in range(1, 6):
            for cid in clips_by_level[lvl]:
                result.append((cid, lvl))
        for cid in unleveled:
            result.append((cid, 0))
        return result

    # Auto-select: pick `samples` per level
    target_levels = [level] if level else range(1, 6)
    result = []
    mask_dir = Path(mask_dir)

    for lvl in target_levels:
        candidates = clips_by_level.get(lvl, [])
        if not candidates:
            continue

        # Split by mask availability
        has_mask = [c for c in candidates if (mask_dir / c).exists()]
        no_mask = [c for c in candidates if not (mask_dir / c).exists()]

        selected = []

        # Prefer front view walk clips for L3 (walker confusion check)
        if lvl == 3:
            walk_fv = [c for c in has_mask if "_w_" in c and c.endswith("FV")]
            if walk_fv:
                selected.append(walk_fv[0])

        # Pick one mask-guided clip
        if len(selected) < samples and has_mask:
            for c in has_mask:
                if c not in selected:
                    selected.append(c)
                    break

        # Pick one height-fallback clip
        if len(selected) < samples and no_mask:
            selected.append(no_mask[0])

        # Fill remaining from any available
        for c in candidates:
            if len(selected) >= samples:
                break
            if c not in selected:
                selected.append(c)

        result.extend((cid, lvl) for cid in selected)

    return result


def find_video_for_clip(clip_id, video_dirs):
    """Locate the source video file for a clip ID.

    Args:
        clip_id: Clip identifier string.
        video_dirs: List of directories to search, in priority order.

    Returns:
        Path to video file, or None if not found.
    """
    for vdir in video_dirs:
        vdir = Path(vdir)
        if not vdir.exists():
            continue
        for ext in [".mp4", ".MP4", ".avi", ".mov", ".MOV"]:
            candidate = vdir / f"{clip_id}{ext}"
            if candidate.exists():
                return candidate
        # Search recursively
        for ext in [".mp4", ".MP4", ".avi", ".mov", ".MOV"]:
            matches = list(vdir.rglob(f"{clip_id}{ext}"))
            if matches:
                return matches[0]
    return None


def load_clip_keypoints(skeleton_2d_dir, clip_id):
    """Load 2D keypoints for a clip.

    Returns:
        Dict mapping identity -> (T, 17, 3) array.
    """
    path = Path(skeleton_2d_dir) / f"{clip_id}.npz"
    data = np.load(str(path))
    return {name: data[name] for name in data.files}


def load_clip_masks(mask_dir, clip_id):
    """Load SAM2 masks for a clip if available.

    Returns:
        Dict mapping identity -> (T, H, W) bool array, or None.
    """
    clip_mask_dir = Path(mask_dir) / clip_id
    if not clip_mask_dir.exists():
        return None
    try:
        return SAM2VideoTracker.load_masks(clip_mask_dir)
    except Exception as e:
        logger.warning("Failed to load masks for %s: %s", clip_id, e)
        return None


def compute_clip_stats(keypoints_dict, conf_threshold=0.3):
    """Compute summary statistics for a clip.

    Returns:
        Dict with detection_rate and mean_confidence per identity.
    """
    stats = {}
    for identity, kp in keypoints_dict.items():
        T = kp.shape[0]
        has_detection = np.any(kp[:, :, :2] != 0, axis=(1, 2))
        det_rate = float(has_detection.sum()) / T if T > 0 else 0.0

        valid_conf = kp[has_detection, :, 2] if has_detection.any() else np.array([])
        mean_conf = float(valid_conf.mean()) if len(valid_conf) > 0 else 0.0

        stats[identity] = {
            "detection_rate": det_rate,
            "mean_confidence": mean_conf,
        }
    return stats


def process_clip(clip_id, video_path, skeleton_2d_dir, mask_dir, output_path,
                 config, no_masks=False):
    """Generate verification video for a single clip.

    Returns:
        Dict with summary stats, or None on failure.
    """
    vis_cfg = config.get("visualization", {})
    conf_threshold = vis_cfg.get("confidence_threshold", 0.3)
    mask_alpha = vis_cfg.get("mask_alpha", 0.3)

    # Load keypoints
    keypoints_dict = load_clip_keypoints(skeleton_2d_dir, clip_id)
    T_kp = max(kp.shape[0] for kp in keypoints_dict.values())

    # Load masks
    masks_all = None
    if not no_masks:
        masks_all = load_clip_masks(mask_dir, clip_id)
    has_masks = masks_all is not None

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    T = min(total_frames, T_kp)

    def frame_generator():
        frame_idx = 0
        while frame_idx < T:
            ret, frame = cap.read()
            if not ret:
                break

            # Get keypoints for this frame
            frame_kp = {}
            for identity, kp_seq in keypoints_dict.items():
                if frame_idx < kp_seq.shape[0]:
                    frame_kp[identity] = kp_seq[frame_idx]

            # Get masks for this frame
            frame_masks = None
            if masks_all:
                frame_masks = {}
                for identity, mask_seq in masks_all.items():
                    if frame_idx < mask_seq.shape[0]:
                        frame_masks[identity] = mask_seq[frame_idx]

            yield draw_verification_frame(
                frame, frame_kp, frame_masks,
                frame_idx=frame_idx, total_frames=T, clip_id=clip_id,
                conf_threshold=conf_threshold, mask_alpha=mask_alpha,
            )
            frame_idx += 1

    result_path = write_verification_video(output_path, frame_generator(), fps=fps)
    cap.release()

    if result_path is None:
        return None

    stats = compute_clip_stats(keypoints_dict, conf_threshold)
    stats["method"] = "mask-guided" if has_masks else "height-fallback"
    return stats


def print_summary(results, output_dir):
    """Print and save summary table.

    Args:
        results: List of (clip_id, level, stats) tuples.
        output_dir: Directory to save summary.txt.
    """
    header = (
        f"{'Level':>5} | {'Clip ID':<25} | {'Method':<15} | "
        f"{'Child Det%':>10} | {'CG Det%':>8} | {'Mean Conf':>9}"
    )
    sep = "-" * len(header)

    lines = ["=== 2D Pose Verification Summary ===", "", header, sep]

    for clip_id, level, stats in results:
        child = stats.get("child", {})
        cg = stats.get("caregiver", {})
        method = stats.get("method", "unknown")

        child_det = f"{child.get('detection_rate', 0) * 100:.1f}%"
        cg_det = f"{cg.get('detection_rate', 0) * 100:.1f}%" if cg else "N/A"
        mean_conf = child.get("mean_confidence", 0)

        line = (
            f"  L{level:d}  | {clip_id:<25} | {method:<15} | "
            f"{child_det:>10} | {cg_det:>8} | {mean_conf:>9.3f}"
        )
        lines.append(line)

    lines.append(sep)
    lines.append(f"\nOutput directory: {output_dir}")

    text = "\n".join(lines)
    print(text)

    summary_path = Path(output_dir) / "summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(text + "\n")
    logger.info("Summary saved to %s", summary_path)


def main():
    parser = argparse.ArgumentParser(
        description="Visual verification of 2D pose estimation results"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--patient", type=str, default=None,
                        help="Process all clips for a single patient")
    parser.add_argument("--clip", type=str, default=None,
                        help="Process a single clip by ID")
    parser.add_argument("--level", type=int, default=None, choices=[1, 2, 3, 4, 5],
                        help="Process samples from a specific GMFCS level")
    parser.add_argument("--samples", type=int, default=2,
                        help="Number of sample clips per GMFCS level (default: 2)")
    parser.add_argument("--all", action="store_true",
                        help="Process ALL clips")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="Override video directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--no-masks", action="store_true",
                        help="Skip mask overlay")
    args = parser.parse_args()

    config = load_config(args.config)

    skeleton_2d_dir = config.get("skeleton_2d_dir", "data/skeleton_2d")
    mask_dir = config.get("tracking", {}).get("mask_output_dir", "data/processed/masks")
    metadata_dir = config.get("metadata_dir", "data/metadata")
    raw_synced_dir = config.get("raw_synced_dir", "data/raw_synced")
    vis_cfg = config.get("visualization", {})
    output_dir = args.output_dir or vis_cfg.get("output_dir", "outputs/verification_2d")

    # Build video search directories
    video_dirs = [raw_synced_dir]
    if args.video_dir:
        video_dirs.insert(0, args.video_dir)

    # Discover available keypoints
    available_clips = discover_keypoints(skeleton_2d_dir)
    if not available_clips:
        logger.warning(
            "No 2D keypoints found in %s. "
            "Run scripts/01_extract_2d_pose.py first to generate keypoints.",
            skeleton_2d_dir,
        )
        sys.exit(0)

    logger.info("Found %d clips with 2D keypoints", len(available_clips))

    # Load labels
    labels = load_labels(metadata_dir)

    # Select clips
    selected = select_clips(
        available_clips, labels, mask_dir,
        patient=args.patient, clip=args.clip, level=args.level,
        samples=args.samples, process_all=args.all,
    )

    if not selected:
        logger.warning("No clips selected for verification")
        sys.exit(0)

    logger.info("Selected %d clips for verification", len(selected))

    # Process clips
    results = []
    for i, (clip_id, level) in enumerate(selected):
        logger.info("[%d/%d] Verifying: %s (L%d)", i + 1, len(selected), clip_id, level)

        video_path = find_video_for_clip(clip_id, video_dirs)
        if video_path is None:
            logger.warning("Video not found for %s, skipping", clip_id)
            continue

        level_dir = Path(output_dir) / f"L{level}"
        out_path = level_dir / f"{clip_id}_verify.mp4"

        stats = process_clip(
            clip_id, video_path, skeleton_2d_dir, mask_dir, out_path,
            config, no_masks=args.no_masks,
        )

        if stats is not None:
            results.append((clip_id, level, stats))

    if results:
        print_summary(results, output_dir)
    else:
        logger.warning("No clips were successfully processed")


if __name__ == "__main__":
    main()
