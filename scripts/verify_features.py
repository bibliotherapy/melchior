"""Feature distribution verification (Step 4.5).

Loads extracted features, groups by GMFCS level, runs sanity checks,
and generates box plots for each feature. Validates that feature
distributions match clinical expectations.

Usage:
    python scripts/verify_features.py
    python scripts/verify_features.py --config configs/default.yaml
    python scripts/verify_features.py --movement w
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server
import matplotlib.pyplot as plt
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.skeleton_features import get_feature_names as get_skeleton_feature_names
from src.features.interaction_features import get_interaction_feature_names
from src.features.movement_quality import get_movement_quality_feature_names
from src.utils.naming import clip_id_to_movement, clip_id_to_patient

# GMFCS level colors for consistent visualization
LEVEL_COLORS = {1: "#2196F3", 2: "#4CAF50", 3: "#FF9800", 4: "#F44336", 5: "#9C27B0"}
LEVEL_LABELS = {1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5"}


# ── Data loading ─────────────────────────────────────────────────────────

def load_labels(labels_path):
    """Load patient_id -> gmfcs_level mapping."""
    with open(labels_path) as f:
        data = json.load(f)
    return {p["patient_id"]: p["gmfcs_level"] for p in data["patients"]}


def load_annotations(ann_path):
    """Load per-patient device annotations."""
    with open(ann_path) as f:
        data = json.load(f)
    return {p["patient_id"]: p for p in data["patients"]}


def load_all_features(features_dir):
    """Load all extracted feature files.

    Returns:
        List of dicts, each with keys:
            clip_id, patient_id, movement_type, gmfcs_level,
            skeleton_features, interaction_features,
            movement_quality_features, context_vector, walker_features
    """
    features_dir = Path(features_dir)
    if not features_dir.exists():
        return []

    files = sorted(features_dir.glob("*.npz"))
    records = []
    for f in files:
        clip_id = f.stem
        data = np.load(str(f))
        records.append({
            "clip_id": clip_id,
            "patient_id": clip_id_to_patient(clip_id),
            "movement_type": clip_id_to_movement(clip_id),
            **{k: data[k] for k in data.files},
        })
    return records


# ── Sanity checks ────────────────────────────────────────────────────────

def check_wfi_by_level(records, labels):
    """Check 1: WFI distribution by level for walk clips.

    Expected: L1/L2 low (~0.1-0.3), L3 high (~0.7-0.9).
    """
    logger.info("=== Check 1: WFI Distribution by Level (Walk) ===")
    walk_records = [r for r in records if r["movement_type"] == "w"]
    if not walk_records:
        logger.warning("  No walk clips found, skipping WFI check")
        return {}

    level_wfi = {}
    for r in walk_records:
        level = labels.get(r["patient_id"])
        if level is None:
            continue
        skel = r.get("skeleton_features")
        if skel is None:
            continue
        # WFI = mean of columns 0 (left) and 1 (right)
        mean_wfi = np.mean(skel[:, :2])
        level_wfi.setdefault(level, []).append(mean_wfi)

    for lvl in sorted(level_wfi):
        vals = level_wfi[lvl]
        logger.info("  L%d: mean=%.3f, std=%.3f, n=%d",
                     lvl, np.mean(vals), np.std(vals), len(vals))

    return level_wfi


def check_asa_by_level(records, labels):
    """Check 2: ASA distribution by level for walk clips.

    Expected: L1 highest, decreasing through L2, L3 near zero.
    """
    logger.info("=== Check 2: ASA Distribution by Level (Walk) ===")
    walk_records = [r for r in records if r["movement_type"] == "w"]
    if not walk_records:
        logger.warning("  No walk clips found, skipping ASA check")
        return {}

    level_asa = {}
    for r in walk_records:
        level = labels.get(r["patient_id"])
        if level is None:
            continue
        skel = r.get("skeleton_features")
        if skel is None:
            continue
        # ASA = mean of columns 2 (left) and 3 (right)
        mean_asa = np.mean(skel[:, 2:4])
        level_asa.setdefault(level, []).append(mean_asa)

    for lvl in sorted(level_asa):
        vals = level_asa[lvl]
        logger.info("  L%d: mean=%.3f, std=%.3f, n=%d",
                     lvl, np.mean(vals), np.std(vals), len(vals))

    return level_asa


def check_arom_by_afo(records, labels, annotations):
    """Check 3: AROM for known AFO users vs non-AFO.

    Expected: AFO users have notably lower AROM.
    """
    logger.info("=== Check 3: AROM by AFO Status ===")

    afo_patients = set()
    non_afo_patients = set()
    for pid, ann in annotations.items():
        devices = ann.get("devices", {})
        if devices.get("afo", False):
            afo_patients.add(pid)
        else:
            non_afo_patients.add(pid)

    if not afo_patients:
        logger.warning("  No AFO users annotated, skipping AROM check")
        return {}

    walk_records = [r for r in records if r["movement_type"] == "w"]
    afo_vals, non_afo_vals = [], []
    for r in walk_records:
        skel = r.get("skeleton_features")
        if skel is None:
            continue
        # AROM = mean of columns 4 (left) and 5 (right)
        mean_arom = np.mean(skel[:, 4:6])
        pid = r["patient_id"]
        if pid in afo_patients:
            afo_vals.append(mean_arom)
        elif pid in non_afo_patients:
            non_afo_vals.append(mean_arom)

    result = {}
    if afo_vals:
        logger.info("  AFO users: mean=%.3f, std=%.3f, n=%d",
                     np.mean(afo_vals), np.std(afo_vals), len(afo_vals))
        result["afo"] = afo_vals
    if non_afo_vals:
        logger.info("  Non-AFO:   mean=%.3f, std=%.3f, n=%d",
                     np.mean(non_afo_vals), np.std(non_afo_vals), len(non_afo_vals))
        result["non_afo"] = non_afo_vals

    return result


def check_independence_score(records, labels):
    """Check 4: Independence score for side rolling, L4 vs L5.

    Expected: L4 ~0.7-1.0, L5 ~0.0-0.3.
    """
    logger.info("=== Check 4: Independence Score (Side Rolling, L4 vs L5) ===")
    sr_records = [r for r in records if r["movement_type"] == "sr"]
    if not sr_records:
        logger.warning("  No side rolling clips found, skipping independence check")
        return {}

    level_scores = {}
    for r in sr_records:
        level = labels.get(r["patient_id"])
        if level not in (4, 5):
            continue
        inter = r.get("interaction_features")
        if inter is None:
            continue
        # Independence score = column 5
        mean_score = np.mean(inter[:, 5])
        level_scores.setdefault(level, []).append(mean_score)

    for lvl in sorted(level_scores):
        vals = level_scores[lvl]
        logger.info("  L%d: mean=%.3f, std=%.3f, n=%d",
                     lvl, np.mean(vals), np.std(vals), len(vals))

    return level_scores


def check_contact_duration(records, labels):
    """Check 5: Contact duration for L4 vs L5 movements.

    Expected: L5 significantly higher than L4.
    """
    logger.info("=== Check 5: Contact Duration Ratio (L4 vs L5) ===")

    level_contact = {}
    for r in records:
        level = labels.get(r["patient_id"])
        if level not in (4, 5):
            continue
        inter = r.get("interaction_features")
        if inter is None:
            continue
        # Contact duration ratio = column 3, take the last frame (cumulative)
        final_ratio = inter[-1, 3] if inter.shape[0] > 0 else 0.0
        level_contact.setdefault(level, []).append(final_ratio)

    for lvl in sorted(level_contact):
        vals = level_contact[lvl]
        logger.info("  L%d: mean=%.3f, std=%.3f, n=%d",
                     lvl, np.mean(vals), np.std(vals), len(vals))

    return level_contact


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_boxplot_by_level(level_data, title, ylabel, output_path, expected=None):
    """Generate a box plot grouped by GMFCS level.

    Args:
        level_data: dict mapping GMFCS level (int) -> list of values.
        title: plot title.
        ylabel: y-axis label.
        output_path: where to save the plot.
        expected: optional dict mapping level -> (expected_low, expected_high)
            for reference bands.
    """
    if not level_data:
        return

    levels = sorted(level_data.keys())
    data = [level_data[lvl] for lvl in levels]
    labels = [f"L{lvl}\n(n={len(level_data[lvl])})" for lvl in levels]
    colors = [LEVEL_COLORS.get(lvl, "#999") for lvl in levels]

    fig, ax = plt.subplots(figsize=(max(6, len(levels) * 1.5), 5))

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", linewidth=1.5))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Add individual data points (jittered)
    for i, (lvl_data, lvl) in enumerate(zip(data, levels)):
        if len(lvl_data) > 0:
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(lvl_data))
            ax.scatter(np.full(len(lvl_data), i + 1) + jitter, lvl_data,
                       color=LEVEL_COLORS.get(lvl, "#999"), alpha=0.5,
                       s=20, zorder=3)

    # Add expected range bands
    if expected:
        for lvl, (lo, hi) in expected.items():
            if lvl in levels:
                idx = levels.index(lvl) + 1
                ax.axhspan(lo, hi, xmin=(idx - 0.8) / (len(levels) + 1),
                           xmax=(idx + 0.2) / (len(levels) + 1),
                           alpha=0.1, color="green", zorder=0)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("GMFCS Level")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", output_path)


def plot_all_skeleton_features(records, labels, output_dir):
    """Generate box plots for all 15 skeleton features across GMFCS levels."""
    logger.info("=== Generating skeleton feature distributions ===")
    feature_names = get_skeleton_feature_names()

    for col, name in enumerate(feature_names):
        level_data = {}
        for r in records:
            level = labels.get(r["patient_id"])
            if level is None:
                continue
            skel = r.get("skeleton_features")
            if skel is None:
                continue
            mean_val = np.mean(skel[:, col])
            level_data.setdefault(level, []).append(mean_val)

        if level_data:
            plot_boxplot_by_level(
                level_data,
                title=f"Skeleton Feature: {name}",
                ylabel=f"Mean {name}",
                output_path=output_dir / f"skeleton_{col:02d}_{name}.png",
            )


def plot_all_interaction_features(records, labels, output_dir):
    """Generate box plots for all 10 interaction features across GMFCS levels."""
    logger.info("=== Generating interaction feature distributions ===")
    feature_names = get_interaction_feature_names()

    for col, name in enumerate(feature_names):
        level_data = {}
        for r in records:
            level = labels.get(r["patient_id"])
            if level is None:
                continue
            inter = r.get("interaction_features")
            if inter is None:
                continue
            mean_val = np.mean(inter[:, col])
            level_data.setdefault(level, []).append(mean_val)

        if level_data:
            plot_boxplot_by_level(
                level_data,
                title=f"Interaction Feature: {name}",
                ylabel=f"Mean {name}",
                output_path=output_dir / f"interaction_{col:02d}_{name}.png",
            )


def plot_movement_quality_by_type(records, labels, output_dir):
    """Generate box plots for movement quality features, grouped by movement type."""
    logger.info("=== Generating movement quality feature distributions ===")

    for movement in ["w", "cr", "c_s", "s_c", "sr"]:
        mv_records = [r for r in records if r["movement_type"] == movement]
        if not mv_records:
            continue

        feature_names = get_movement_quality_feature_names(movement)

        for col, name in enumerate(feature_names):
            if name.startswith("_pad_"):
                continue  # skip padding features

            level_data = {}
            for r in mv_records:
                level = labels.get(r["patient_id"])
                if level is None:
                    continue
                mq = r.get("movement_quality_features")
                if mq is None:
                    continue
                level_data.setdefault(level, []).append(float(mq[col]))

            if level_data:
                plot_boxplot_by_level(
                    level_data,
                    title=f"Movement Quality: {name}",
                    ylabel=name,
                    output_path=output_dir / f"mq_{movement}_{col:02d}_{name}.png",
                )


def plot_walker_features(records, labels, output_dir):
    """Generate box plots for walker spatial features across GMFCS levels."""
    logger.info("=== Generating walker feature distributions ===")
    walker_names = [
        "hand_to_walker_distance",
        "walker_engagement_ratio",
        "walker_lateral_offset",
        "walker_height_ratio",
        "walker_presence_score",
    ]

    for col, name in enumerate(walker_names):
        level_data = {}
        for r in records:
            level = labels.get(r["patient_id"])
            if level is None:
                continue
            wf = r.get("walker_features")
            if wf is None or col >= len(wf):
                continue
            level_data.setdefault(level, []).append(float(wf[col]))

        if level_data:
            plot_boxplot_by_level(
                level_data,
                title=f"Walker Feature: {name}",
                ylabel=name,
                output_path=output_dir / f"walker_{col:02d}_{name}.png",
            )


# ── Summary report ───────────────────────────────────────────────────────

def write_summary(output_dir, records, labels, check_results):
    """Write a text summary of all checks."""
    summary_path = output_dir / "verification_summary.txt"

    lines = [
        "Feature Verification Summary",
        "=" * 40,
        f"Total clips: {len(records)}",
        "",
    ]

    # Clips per level
    level_counts = {}
    for r in records:
        lvl = labels.get(r["patient_id"])
        if lvl:
            level_counts[lvl] = level_counts.get(lvl, 0) + 1
    for lvl in sorted(level_counts):
        lines.append(f"  L{lvl}: {level_counts[lvl]} clips")

    # Movement distribution
    lines.append("")
    mv_counts = {}
    for r in records:
        mt = r["movement_type"] or "unknown"
        mv_counts[mt] = mv_counts.get(mt, 0) + 1
    for mt in sorted(mv_counts):
        lines.append(f"  {mt}: {mv_counts[mt]} clips")

    lines.append("")
    lines.append("Sanity Check Results")
    lines.append("-" * 40)

    for name, data in check_results.items():
        lines.append(f"\n{name}:")
        if not data:
            lines.append("  No data available")
            continue
        for key in sorted(data):
            vals = data[key]
            if vals:
                lines.append(f"  {key}: mean={np.mean(vals):.3f}, "
                             f"std={np.std(vals):.3f}, n={len(vals)}")

    summary_text = "\n".join(lines)
    summary_path.write_text(summary_text)
    logger.info("Summary written to %s", summary_path)
    print("\n" + summary_text)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Feature distribution verification")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--movement", type=str, default=None,
                        help="Filter to specific movement type (w, cr, c_s, s_c, sr)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    features_dir = config.get("features_dir", "data/features")
    metadata_dir = config.get("metadata_dir", "data/metadata")
    output_dir = Path("outputs/feature_distributions")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    labels = load_labels(Path(metadata_dir) / "labels.json")
    annotations = load_annotations(Path(metadata_dir) / "assistive_annotations.json")

    # Load all feature files
    records = load_all_features(features_dir)
    if not records:
        logger.error("No feature files found in %s. Run 04_extract_features.py first.", features_dir)
        sys.exit(1)

    # Filter by movement type if specified
    if args.movement:
        records = [r for r in records if r["movement_type"] == args.movement]
        logger.info("Filtered to %d %s clips", len(records), args.movement)

    logger.info("Loaded %d feature files from %s", len(records), features_dir)

    # Assign GMFCS levels
    for r in records:
        r["gmfcs_level"] = labels.get(r["patient_id"])

    # Run sanity checks
    check_results = {}
    check_results["WFI by Level (Walk)"] = check_wfi_by_level(records, labels)
    check_results["ASA by Level (Walk)"] = check_asa_by_level(records, labels)
    check_results["AROM by AFO Status"] = check_arom_by_afo(records, labels, annotations)
    check_results["Independence Score (SR, L4/L5)"] = check_independence_score(records, labels)
    check_results["Contact Duration (L4/L5)"] = check_contact_duration(records, labels)

    # Generate plots for all feature types
    plot_all_skeleton_features(records, labels, output_dir)
    plot_all_interaction_features(records, labels, output_dir)
    plot_movement_quality_by_type(records, labels, output_dir)
    plot_walker_features(records, labels, output_dir)

    # Write summary
    write_summary(output_dir, records, labels, check_results)

    logger.info("Feature verification complete. Plots saved to %s", output_dir)


if __name__ == "__main__":
    main()
