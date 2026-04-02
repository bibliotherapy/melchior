"""Visual verification of 3D triangulation results.

Generates 3D skeleton plots, bone length consistency charts,
and reprojection error analysis for human review.

Usage:
    python scripts/verify_3d_triangulation.py
    python scripts/verify_3d_triangulation.py --patient kku
    python scripts/verify_3d_triangulation.py --clip kku_w_01
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.pose_calibration import load_calibration
from src.triangulation.triangulate_3d import COCO_BONES, SkeletonTriangulator
from src.utils.naming import clip_id_to_patient, clip_id_to_view


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_labels(metadata_dir):
    labels_path = Path(metadata_dir) / "labels.json"
    if not labels_path.exists():
        return {}
    with open(labels_path) as f:
        data = json.load(f)
    return {p["patient_id"]: p["gmfcs_level"] for p in data["patients"]}


BONE_NAMES = {
    (5, 7): "L_upper_arm", (7, 9): "L_forearm",
    (6, 8): "R_upper_arm", (8, 10): "R_forearm",
    (5, 11): "L_torso", (6, 12): "R_torso",
    (11, 13): "L_thigh", (13, 15): "L_shin",
    (12, 14): "R_thigh", (14, 16): "R_shin",
    (5, 6): "shoulder_w", (11, 12): "hip_w",
}


def compute_bone_lengths(skeleton_3d):
    """Compute bone lengths across all frames.

    Returns:
        {(j1, j2): (T,) lengths} for each bone.
    """
    lengths = {}
    for j1, j2 in COCO_BONES:
        diff = skeleton_3d[:, j1] - skeleton_3d[:, j2]
        bone_len = np.linalg.norm(diff, axis=1)
        # NaN where either joint is missing
        missing = np.isnan(skeleton_3d[:, j1, 0]) | np.isnan(skeleton_3d[:, j2, 0])
        zero = (skeleton_3d[:, j1, 0] == 0) & (skeleton_3d[:, j2, 0] == 0)
        bone_len[missing | zero] = np.nan
        lengths[(j1, j2)] = bone_len
    return lengths


def plot_3d_skeleton_frame(skeleton_3d, frame_idx, title, save_path):
    """Plot a single 3D skeleton frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils.visualization import draw_3d_skeleton

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    draw_3d_skeleton(ax, skeleton_3d[frame_idx], identity="child")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    fig.savefig(str(save_path), dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_bone_consistency(bone_lengths, clip_id, save_path):
    """Plot bone lengths across frames with consistency bounds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_bones = len(bone_lengths)
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    for idx, ((j1, j2), lengths) in enumerate(bone_lengths.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        valid = ~np.isnan(lengths)
        if valid.sum() > 0:
            median = np.nanmedian(lengths)
            ax.plot(np.where(valid)[0], lengths[valid], "b-", linewidth=0.5)
            ax.axhline(median, color="g", linestyle="--", linewidth=1)
            ax.axhline(median * 1.3, color="r", linestyle=":", linewidth=0.5)
            ax.axhline(median * 0.7, color="r", linestyle=":", linewidth=0.5)
        name = BONE_NAMES.get((j1, j2), f"{j1}-{j2}")
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)

    for idx in range(n_bones, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Bone Length Consistency: {clip_id}", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=100, bbox_inches="tight")
    plt.close(fig)


def compute_reproj_errors(skeleton_3d, skeleton_2d_dir, triplet_base,
                          calibration):
    """Compute reprojection errors for a clip."""
    triangulator = SkeletonTriangulator()
    kp_per_view = {}
    for view in calibration.get("calibrated_views", []):
        clip_id = f"{triplet_base}_{view}"
        path = Path(skeleton_2d_dir) / f"{clip_id}.npz"
        if path.exists():
            data = np.load(str(path))
            kp_per_view[view] = data["child"]
    if not kp_per_view:
        return None
    return triangulator.compute_reprojection_error(skeleton_3d, kp_per_view, calibration)


def process_clip(triplet_base, skeleton_3d_dir, skeleton_2d_dir,
                 calibration_dir, output_dir, level):
    """Generate verification outputs for one clip."""
    # Load 3D skeleton
    path_3d = Path(skeleton_3d_dir) / f"{triplet_base}.npz"
    data = np.load(str(path_3d))
    child_3d = data["child"]

    clip_dir = Path(output_dir) / f"L{level}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    T = child_3d.shape[0]
    stats = {"triplet_base": triplet_base, "level": level, "frames": T}

    # 3D skeleton plot (mid-frame)
    mid = T // 2
    plot_3d_skeleton_frame(
        child_3d, mid,
        f"{triplet_base} frame {mid}/{T}",
        clip_dir / f"{triplet_base}_3d.png",
    )

    # Bone length consistency
    bone_lengths = compute_bone_lengths(child_3d)
    plot_bone_consistency(bone_lengths, triplet_base, clip_dir / f"{triplet_base}_bones.png")

    # Bone length stats
    bone_cv = []
    for (j1, j2), lengths in bone_lengths.items():
        valid = lengths[~np.isnan(lengths)]
        if len(valid) > 0:
            cv = float(np.std(valid) / (np.mean(valid) + 1e-12))
            bone_cv.append(cv)
    stats["mean_bone_cv"] = float(np.mean(bone_cv)) if bone_cv else float("nan")

    # Reprojection error
    patient_id = clip_id_to_patient(triplet_base)
    cal_path = Path(calibration_dir) / f"{patient_id}.npz"
    if cal_path.exists():
        calibration = load_calibration(str(cal_path))
        reproj = compute_reproj_errors(
            child_3d, skeleton_2d_dir, triplet_base, calibration,
        )
        if reproj:
            all_errors = []
            for view, errs in reproj.items():
                valid = errs[~np.isnan(errs)]
                all_errors.extend(valid.tolist())
            stats["mean_reproj_px"] = float(np.mean(all_errors)) if all_errors else float("nan")
        else:
            stats["mean_reproj_px"] = float("nan")
    else:
        stats["mean_reproj_px"] = float("nan")

    return stats


def print_summary(results, output_dir):
    """Print and save summary."""
    header = (
        f"{'Level':>5} | {'Clip':>20} | {'Frames':>6} | "
        f"{'Bone CV':>8} | {'Reproj (px)':>11} | {'Status':>6}"
    )
    sep = "-" * len(header)
    lines = ["=== 3D Triangulation Verification ===", "", header, sep]

    for r in results:
        cv = r["mean_bone_cv"]
        reproj = r["mean_reproj_px"]
        status = "PASS" if (not np.isnan(reproj) and reproj < 10) else "CHECK"

        lines.append(
            f"  L{r['level']}  | {r['triplet_base']:>20} | {r['frames']:>6} | "
            f"{cv:>8.4f} | {reproj:>11.2f} | {status:>6}"
        )

    lines.append(sep)
    lines.append(f"\nOutput: {output_dir}")
    text = "\n".join(lines)
    print(text)

    summary_path = Path(output_dir) / "summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(text + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Verify 3D triangulation results"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--patient", type=str, default=None)
    parser.add_argument("--clip", type=str, default=None,
                        help="Single triplet base (e.g., kku_w_01)")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir", default="outputs/verification_3d")
    args = parser.parse_args()

    config = load_config(args.config)
    skeleton_3d_dir = Path(config.get("skeleton_3d_dir", "data/skeleton_3d"))
    skeleton_2d_dir = Path(config.get("skeleton_2d_dir", "data/skeleton_2d"))
    calibration_dir = Path(config.get("calibration_dir", "data/calibration"))
    metadata_dir = config.get("metadata_dir", "data/metadata")

    # Discover 3D skeletons
    skel_files = sorted(skeleton_3d_dir.glob("*.npz"))
    if not skel_files:
        logger.warning("No 3D skeletons in %s. Run 03_triangulate_3d.py first.", skeleton_3d_dir)
        sys.exit(0)

    labels = load_labels(metadata_dir)

    # Select clips
    clips = []
    for f in skel_files:
        tb = f.stem
        pid = clip_id_to_patient(tb)
        level = labels.get(pid, 0)
        if args.clip and tb != args.clip:
            continue
        if args.patient and pid != args.patient:
            continue
        clips.append((tb, level))

    if not args.all and not args.clip and not args.patient:
        # Sample per level
        by_level = {}
        for tb, level in clips:
            by_level.setdefault(level, []).append(tb)
        clips = []
        for level in sorted(by_level):
            for tb in by_level[level][:args.samples]:
                clips.append((tb, level))

    logger.info("Verifying %d clips", len(clips))

    results = []
    for i, (tb, level) in enumerate(clips):
        logger.info("[%d/%d] %s (L%d)", i + 1, len(clips), tb, level)
        try:
            stats = process_clip(
                tb, skeleton_3d_dir, skeleton_2d_dir,
                calibration_dir, args.output_dir, level,
            )
            results.append(stats)
        except Exception as e:
            logger.error("Failed %s: %s", tb, e)

    if results:
        print_summary(results, args.output_dir)


if __name__ == "__main__":
    main()
