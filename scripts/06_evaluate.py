"""Evaluation entry point for hierarchical GMFCS classifier.

Loads training results, runs per-stage and end-to-end evaluation,
generates confusion matrices, training curves, ablation comparisons,
and feature importance analysis.

Usage:
    python scripts/06_evaluate.py
    python scripts/06_evaluate.py --config configs/default.yaml
    python scripts/06_evaluate.py --results-dir outputs/models
    python scripts/06_evaluate.py --skip-importance
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.evaluation import (
    GMFCSEvaluator,
    generate_report,
    plot_ablation_comparison,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_training_curves,
)


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


def load_training_results(results_path):
    """Load training_results.json."""
    with open(results_path) as f:
        return json.load(f)


def find_ablation_results(results_dir):
    """Find all training result files for ablation comparison.

    Looks for training_results*.json files.

    Returns:
        dict mapping config_name -> results dict.
    """
    results_dir = Path(results_dir)
    ablation = {}

    for f in sorted(results_dir.glob("training_results*.json")):
        name = f.stem.replace("training_results", "").strip("_") or "full_model"
        with open(f) as fh:
            ablation[name] = json.load(fh)

    return ablation


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate hierarchical GMFCS classifier"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--results-dir", default="outputs/models",
                        help="Directory with training_results.json and model files")
    parser.add_argument("--output-dir", default="outputs/evaluation",
                        help="Directory to save evaluation outputs")
    parser.add_argument("--skip-importance", action="store_true",
                        help="Skip permutation feature importance (slow)")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip ablation study comparison")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_dir = config.get("metadata_dir", "data/metadata")

    # Load metadata
    labels = load_labels(Path(metadata_dir) / "labels.json")
    annotations = load_annotations(Path(metadata_dir) / "assistive_annotations.json")

    # Load training results
    results_path = results_dir / "training_results.json"
    if not results_path.exists():
        logger.error("No training results found at %s", results_path)
        logger.error("Run scripts/05_train.py first.")
        sys.exit(1)

    all_results = load_training_results(results_path)
    logger.info("Loaded training results from %s", results_path)

    # Initialize evaluator
    evaluator = GMFCSEvaluator(labels, annotations)

    # ── Per-stage evaluation ──
    logger.info("=" * 50)
    logger.info("Per-Stage Evaluation")
    logger.info("=" * 50)

    per_stage_metrics = {}
    for stage in ["stage1", "stage2a", "stage2b"]:
        stage_data = all_results.get(stage, {})
        if not stage_data.get("folds"):
            logger.warning("No folds for %s, skipping", stage)
            continue

        metrics = evaluator.evaluate_per_stage(stage_data, stage)
        per_stage_metrics[stage] = metrics
        logger.info("%s accuracy: %.1f%% (n=%d)",
                     stage, metrics["accuracy"] * 100, metrics["n_samples"])

        # Confusion matrix plot
        if metrics.get("confusion_matrix"):
            import numpy as np
            plot_confusion_matrix(
                np.array(metrics["confusion_matrix"]),
                metrics["class_names"],
                f"Confusion Matrix — {stage}",
                output_dir / f"confusion_matrix_{stage}.png",
            )

        # Training curves
        plot_training_curves(
            stage_data, stage, output_dir / f"training_curves_{stage}.png"
        )

    # Save per-stage metrics
    with open(output_dir / "per_stage_metrics.json", "w") as f:
        json.dump(per_stage_metrics, f, indent=2)

    # ── End-to-end evaluation ──
    logger.info("=" * 50)
    logger.info("End-to-End Evaluation")
    logger.info("=" * 50)

    e2e_metrics = evaluator.evaluate_end_to_end(all_results)
    logger.info("End-to-end accuracy: %.1f%% (%s 80%% target)",
                 e2e_metrics["accuracy"] * 100,
                 "MEETS" if e2e_metrics.get("target_80_met") else "BELOW")

    # End-to-end confusion matrix
    if e2e_metrics.get("confusion_matrix"):
        import numpy as np
        plot_confusion_matrix(
            np.array(e2e_metrics["confusion_matrix"]),
            e2e_metrics["class_names"],
            "End-to-End Confusion Matrix (5-class GMFCS)",
            output_dir / "confusion_matrix_end_to_end.png",
        )

    with open(output_dir / "end_to_end_metrics.json", "w") as f:
        json.dump({k: v for k, v in e2e_metrics.items()
                   if k not in ("predicted_levels", "true_levels", "clip_ids")},
                  f, indent=2)

    # ── Generate report ──
    generate_report(per_stage_metrics, e2e_metrics,
                    output_dir / "evaluation_report.txt")

    # ── Ablation comparison ──
    if not args.skip_ablation:
        logger.info("=" * 50)
        logger.info("Ablation Study")
        logger.info("=" * 50)

        ablation_files = find_ablation_results(results_dir)
        if len(ablation_files) > 1:
            ablation_summary = {}
            for cfg_name, cfg_results in ablation_files.items():
                cfg_eval = GMFCSEvaluator(labels, annotations)
                accs = {}
                for stage in ["stage1", "stage2a", "stage2b"]:
                    stage_data = cfg_results.get(stage, {})
                    if stage_data.get("folds"):
                        m = cfg_eval.evaluate_per_stage(stage_data, stage)
                        accs[f"{stage}_acc"] = m["accuracy"]
                e2e = cfg_eval.evaluate_end_to_end(cfg_results)
                accs["end_to_end_acc"] = e2e["accuracy"]
                ablation_summary[cfg_name] = accs
                logger.info("  %s: e2e=%.1f%%", cfg_name,
                             accs["end_to_end_acc"] * 100)

            plot_ablation_comparison(
                ablation_summary, output_dir / "ablation_comparison.png"
            )

            with open(output_dir / "ablation_results.json", "w") as f:
                json.dump(ablation_summary, f, indent=2)
        else:
            logger.info("Only one result file found, skipping ablation comparison")

    # ── Feature importance ──
    if not args.skip_importance:
        logger.info("=" * 50)
        logger.info("Feature Importance Analysis")
        logger.info("=" * 50)

        import torch
        from src.model.lite_stgcn import LiteSTGCN
        from src.model.classifier import MultiStreamClassifier
        from src.model.dataset import GMFCSDataset, _load_ambulatory_status
        from src.utils.evaluation import compute_permutation_importance

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_cfg = config.get("model", {})
        features_dir = config.get("features_dir", "data/features")
        skeleton_3d_dir = config.get("skeleton_3d_dir", "data/skeleton_3d")

        labels_path = str(Path(metadata_dir) / "labels.json")
        ann_path = str(Path(metadata_dir) / "assistive_annotations.json")
        ambulatory_status = _load_ambulatory_status(labels_path, ann_path)

        stage_args = {"stage1": 1, "stage2a": "2a", "stage2b": "2b"}

        for stage in ["stage1", "stage2a", "stage2b"]:
            # Find best fold
            stage_data = all_results.get(stage, {})
            folds = stage_data.get("folds", [])
            if not folds:
                continue

            best_fold = max(folds, key=lambda f: f["test_acc"])
            fold_idx = best_fold["fold_idx"]
            fold_dir = results_dir / stage / f"fold_{fold_idx}"

            stgcn_path = fold_dir / "stgcn.pt"
            clf_path = fold_dir / "classifier.pt"
            if not stgcn_path.exists() or not clf_path.exists():
                logger.warning("Model files not found for %s fold %d, skipping",
                               stage, fold_idx)
                continue

            # Load models
            stgcn = LiteSTGCN.from_config(model_cfg).to(device)
            stgcn.load_state_dict(
                torch.load(str(stgcn_path), map_location=device, weights_only=True)
            )

            n_classes = {"stage1": 2, "stage2a": 3, "stage2b": 2}[stage]
            fusion_cfg = model_cfg.get("fusion", {})
            classifier = MultiStreamClassifier(
                stgcn_dim=stgcn.output_dim,
                skeleton_feature_dim=model_cfg.get("skeleton_feature_dim", 15),
                interaction_feature_dim=model_cfg.get("interaction_feature_dim", 10),
                context_vector_dim=model_cfg.get("context_vector_dim", 18),
                walker_feature_dim=model_cfg.get("walker_feature_dim", 5),
                hidden_dim=fusion_cfg.get("hidden_dim", 64),
                dropout=fusion_cfg.get("dropout", 0.3),
                num_classes=n_classes,
                hierarchical=True,
            ).to(device)
            classifier.load_state_dict(
                torch.load(str(clf_path), map_location=device, weights_only=True)
            )

            # Create test dataset from best fold's test clips
            test_clips = best_fold["test_clip_ids"]
            test_ds = GMFCSDataset(
                test_clips, features_dir, skeleton_3d_dir, labels,
                stage=stage, max_seq_len=150,
                ambulatory_status=ambulatory_status,
            )

            logger.info("Computing permutation importance for %s (fold %d, %d clips)...",
                         stage, fold_idx, len(test_clips))
            importance = compute_permutation_importance(
                stgcn, classifier, test_ds, stage_args[stage], device, n_repeats=3
            )

            plot_feature_importance(
                importance, stage, output_dir / f"feature_importance_{stage}.png"
            )

            with open(output_dir / f"feature_importance_{stage}.json", "w") as f:
                json.dump(importance, f, indent=2)

    logger.info("=" * 50)
    logger.info("Evaluation complete. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
