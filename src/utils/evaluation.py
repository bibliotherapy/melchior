"""Evaluation utilities for hierarchical GMFCS classification.

Provides:
  - Per-stage metrics (confusion matrix, precision, recall, F1)
  - End-to-end hierarchical prediction resolution (Stage 1 → Stage 2 → GMFCS level)
  - Confusion matrix and training curve plotting
  - Permutation feature importance analysis
  - Ablation study comparison
  - Classification report generation
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Consistent GMFCS level colors (shared with verify_features.py)
LEVEL_COLORS = {1: "#2196F3", 2: "#4CAF50", 3: "#FF9800", 4: "#F44336", 5: "#9C27B0"}

STAGE_CLASS_NAMES = {
    "stage1": ["Ambulatory", "Non-ambulatory"],
    "stage2a": ["L1", "L2", "L3-L4"],
    "stage2b": ["L3-L4", "L5"],
}


# ── Metrics ──────────────────────────────────────────────────────────────

def confusion_matrix(y_true, y_pred, n_classes=None):
    """Compute confusion matrix.

    Returns:
        (n_classes, n_classes) numpy array. cm[i][j] = count of true=i predicted=j.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if n_classes is None:
        n_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def per_class_metrics(cm):
    """Compute precision, recall, F1 per class from confusion matrix.

    Returns:
        List of dicts, one per class: {precision, recall, f1, support}.
    """
    n_classes = cm.shape[0]
    metrics = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        support = cm[c, :].sum()
        metrics.append({
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        })
    return metrics


# ── Hierarchical resolution ──────────────────────────────────────────────

def resolve_hierarchical_predictions(stage1_results, stage2a_results,
                                      stage2b_results, annotations):
    """Combine hierarchical stage predictions into final GMFCS levels.

    For each clip that appeared in any test fold:
    1. Stage 1: ambulatory (0) or non-ambulatory (1)
    2. If ambulatory → Stage 2A: L1(0), L2(1), L3-L4(2)
    3. If non-ambulatory → Stage 2B: L3-L4(0), L5(1)
    4. L3-L4 merged → resolve via walker_used annotation:
       walker=True → L3, walker=False → L4

    Args:
        stage1_results: stage result dict with 'folds' list.
        stage2a_results: stage result dict for ambulatory branch.
        stage2b_results: stage result dict for non-ambulatory branch.
        annotations: dict mapping patient_id -> annotation dict.

    Returns:
        (predicted_levels, true_levels, clip_ids): lists of GMFCS levels 1-5.
    """
    from src.utils.naming import clip_id_to_patient

    # Build per-clip prediction maps from each stage's test folds
    s1_preds = {}  # clip_id -> stage1 pred
    s2a_preds = {}  # clip_id -> stage2a pred
    s2b_preds = {}  # clip_id -> stage2b pred

    for fold in stage1_results.get("folds", []):
        for cid, pred in zip(fold["test_clip_ids"], fold["test_preds"]):
            s1_preds[cid] = pred

    for fold in stage2a_results.get("folds", []):
        for cid, pred in zip(fold["test_clip_ids"], fold["test_preds"]):
            s2a_preds[cid] = pred

    for fold in stage2b_results.get("folds", []):
        for cid, pred in zip(fold["test_clip_ids"], fold["test_preds"]):
            s2b_preds[cid] = pred

    # Walker lookup for L3/L4 resolution
    def _has_walker(patient_id):
        ann = annotations.get(patient_id, {})
        devices = ann.get("devices", {})
        return devices.get("walker", False)

    predicted = []
    true_levels = []
    clip_ids_out = []

    # Use stage1 clips as the base (all clips appear in stage1)
    for fold in stage1_results.get("folds", []):
        for cid, s1_pred, s1_label in zip(
            fold["test_clip_ids"], fold["test_preds"], fold["test_labels"]
        ):
            pid = clip_id_to_patient(cid)

            if s1_pred == 0:  # Ambulatory
                s2a_pred = s2a_preds.get(cid)
                if s2a_pred is not None:
                    if s2a_pred == 0:
                        gmfcs = 1
                    elif s2a_pred == 1:
                        gmfcs = 2
                    else:  # 2 = L3-L4
                        gmfcs = 3 if _has_walker(pid) else 4
                else:
                    # Clip wasn't in stage2a (maybe non-ambulatory patient
                    # misclassified as ambulatory) — use L1 default
                    gmfcs = 1
            else:  # Non-ambulatory
                s2b_pred = s2b_preds.get(cid)
                if s2b_pred is not None:
                    if s2b_pred == 0:  # L3-L4
                        gmfcs = 3 if _has_walker(pid) else 4
                    else:  # L5
                        gmfcs = 5
                else:
                    # Clip wasn't in stage2b — use L5 default
                    gmfcs = 5

            predicted.append(gmfcs)
            clip_ids_out.append(cid)

    return predicted, clip_ids_out


# ── GMFCSEvaluator ───────────────────────────────────────────────────────

class GMFCSEvaluator:
    """Evaluation engine for hierarchical GMFCS classification."""

    def __init__(self, labels, annotations=None):
        """
        Args:
            labels: dict mapping patient_id -> gmfcs_level (1-5).
            annotations: dict mapping patient_id -> annotation dict.
        """
        self.labels = labels
        self.annotations = annotations or {}

    def evaluate_per_stage(self, stage_results, stage_name):
        """Compute per-stage metrics from CV fold results.

        Args:
            stage_results: dict with 'folds' list from training_results.json.
            stage_name: 'stage1', 'stage2a', or 'stage2b'.

        Returns:
            dict with cm, per_class, accuracy, macro_f1, n_samples.
        """
        all_preds = []
        all_labels = []
        for fold in stage_results.get("folds", []):
            all_preds.extend(fold["test_preds"])
            all_labels.extend(fold["test_labels"])

        if not all_preds:
            return {"accuracy": 0.0, "n_samples": 0}

        class_names = STAGE_CLASS_NAMES.get(stage_name, [])
        n_classes = len(class_names) if class_names else (max(all_labels) + 1)
        cm = confusion_matrix(all_labels, all_preds, n_classes)
        cls_metrics = per_class_metrics(cm)

        accuracy = np.trace(cm) / max(cm.sum(), 1)
        macro_f1 = np.mean([m["f1"] for m in cls_metrics]) if cls_metrics else 0.0

        return {
            "stage_name": stage_name,
            "class_names": class_names,
            "confusion_matrix": cm.tolist(),
            "per_class": cls_metrics,
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "n_samples": len(all_preds),
        }

    def evaluate_end_to_end(self, all_results):
        """Combine hierarchical predictions into 5-class GMFCS evaluation.

        Args:
            all_results: dict with 'stage1', 'stage2a', 'stage2b' keys.

        Returns:
            dict with cm, per_class, accuracy, macro_f1 for 5-class GMFCS.
        """
        from src.utils.naming import clip_id_to_patient

        predicted, clip_ids = resolve_hierarchical_predictions(
            all_results.get("stage1", {}),
            all_results.get("stage2a", {}),
            all_results.get("stage2b", {}),
            self.annotations,
        )

        # Look up true GMFCS levels
        true_levels = []
        for cid in clip_ids:
            pid = clip_id_to_patient(cid)
            true_levels.append(self.labels.get(pid, 0))

        if not predicted:
            return {"accuracy": 0.0, "n_samples": 0}

        # GMFCS levels are 1-5, map to 0-4 for confusion matrix
        pred_idx = [p - 1 for p in predicted]
        true_idx = [t - 1 for t in true_levels]
        class_names = ["L1", "L2", "L3", "L4", "L5"]

        cm = confusion_matrix(true_idx, pred_idx, n_classes=5)
        cls_metrics = per_class_metrics(cm)

        accuracy = np.trace(cm) / max(cm.sum(), 1)
        macro_f1 = np.mean([m["f1"] for m in cls_metrics]) if cls_metrics else 0.0

        target_met = accuracy >= 0.80

        return {
            "class_names": class_names,
            "confusion_matrix": cm.tolist(),
            "per_class": cls_metrics,
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "n_samples": len(predicted),
            "target_80_met": target_met,
            "predicted_levels": predicted,
            "true_levels": true_levels,
            "clip_ids": clip_ids,
        }


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, class_names, title, output_path):
    """Plot confusion matrix as annotated heatmap.

    Args:
        cm: (n, n) numpy array or list-of-lists.
        class_names: list of class label strings.
        title: plot title.
        output_path: path to save PNG.
    """
    cm = np.asarray(cm)
    n = len(class_names)

    fig, ax = plt.subplots(figsize=(max(5, n * 1.2), max(4, n * 1.0)))

    # Normalize for color mapping
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)

    # Annotate cells with count and percentage
    for i in range(n):
        for j in range(n):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{count}\n({pct:.0f}%)",
                    ha="center", va="center", color=color, fontsize=10)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=13, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Recall")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_training_curves(stage_results, stage_name, output_path):
    """Plot train/val loss and val accuracy curves across folds.

    Args:
        stage_results: dict with 'folds' list, each fold has 'history'.
        stage_name: for title.
        output_path: path to save PNG.
    """
    folds = stage_results.get("folds", [])
    if not folds:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for fold in folds:
        history = fold.get("history", {})
        fidx = fold["fold_idx"]
        epochs = range(1, len(history.get("train_loss", [])) + 1)

        if "train_loss" in history:
            ax1.plot(epochs, history["train_loss"], alpha=0.4,
                     label=f"Fold {fidx} train")
        if "val_loss" in history:
            ax1.plot(epochs, history["val_loss"], "--", alpha=0.7,
                     label=f"Fold {fidx} val")

        if "val_acc" in history:
            ax2.plot(epochs, history["val_acc"], alpha=0.7,
                     label=f"Fold {fidx}")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{stage_name} — Loss Curves")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{stage_name} — Validation Accuracy")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_ablation_comparison(ablation_results, output_path):
    """Bar chart comparing accuracy across ablation configurations.

    Args:
        ablation_results: dict mapping config_name -> {stage1_acc, stage2a_acc,
            stage2b_acc, end_to_end_acc}.
        output_path: path to save PNG.
    """
    if not ablation_results:
        return

    configs = list(ablation_results.keys())
    stages = ["stage1", "stage2a", "stage2b", "end_to_end"]
    stage_labels = ["Stage 1", "Stage 2A", "Stage 2B", "End-to-End"]

    n_configs = len(configs)
    n_stages = len(stages)
    x = np.arange(n_stages)
    width = 0.8 / n_configs

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    for i, cfg in enumerate(configs):
        vals = [ablation_results[cfg].get(f"{s}_acc", 0) for s in stages]
        offset = (i - n_configs / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=cfg,
                      color=colors[i % len(colors)], alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels)
    ax.set_ylabel("Accuracy")
    ax.set_title("Ablation Study: Accuracy by Configuration", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.axhline(0.80, color="red", linestyle="--", alpha=0.5, label="80% target")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_feature_importance(importance_dict, stage_name, output_path, top_k=15):
    """Horizontal bar chart of feature importance scores.

    Args:
        importance_dict: dict mapping feature_name -> importance_score.
        stage_name: for title.
        output_path: path to save PNG.
        top_k: show only top-k features.
    """
    if not importance_dict:
        return

    # Sort by importance
    sorted_feats = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    top = sorted_feats[:top_k]
    names = [f[0] for f in reversed(top)]
    scores = [f[1] for f in reversed(top)]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
    colors = ["#F44336" if s > 0 else "#9E9E9E" for s in scores]
    ax.barh(names, scores, color=colors, alpha=0.8)
    ax.set_xlabel("Accuracy Drop (Permutation Importance)")
    ax.set_title(f"{stage_name} — Feature Importance (Top {top_k})", fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


# ── Permutation importance ───────────────────────────────────────────────

def compute_permutation_importance(stgcn, classifier, dataset, stage_arg,
                                    device, n_repeats=5):
    """Compute permutation importance for all feature streams.

    Shuffles one feature column at a time and measures accuracy drop.

    Args:
        stgcn: LiteSTGCN model (eval mode).
        classifier: MultiStreamClassifier model (eval mode).
        dataset: GMFCSDataset instance.
        stage_arg: classifier stage argument (1, '2a', '2b').
        device: torch device.
        n_repeats: number of shuffle repetitions per feature.

    Returns:
        dict mapping feature_name -> mean accuracy drop.
    """
    import torch
    from torch.utils.data import DataLoader
    from src.model.train import collate_fn

    stgcn.eval()
    classifier.eval()

    loader = DataLoader(dataset, batch_size=16, shuffle=False,
                        collate_fn=collate_fn)

    # Baseline accuracy
    baseline_acc = _eval_accuracy(stgcn, classifier, loader, stage_arg, device)
    logger.info("  Baseline accuracy: %.3f", baseline_acc)

    importance = {}

    # Feature stream definitions: (batch_key, n_cols, name_prefix, is_temporal)
    from src.features.skeleton_features import get_feature_names as get_skel_names
    from src.features.interaction_features import get_interaction_feature_names

    skel_names = get_skel_names()
    inter_names = get_interaction_feature_names()

    streams = []
    # Skeleton features: permute each column individually
    for col, name in enumerate(skel_names):
        streams.append(("skeleton_features", col, f"skel_{name}", True))
    # Interaction features: permute each column individually
    for col, name in enumerate(inter_names):
        streams.append(("interaction_features", col, f"inter_{name}", True))
    # Context vector: permute each dimension
    ctx_names = [f"ctx_{i}" for i in range(18)]
    for col in range(18):
        streams.append(("context_vector", col, ctx_names[col], False))
    # Walker features: permute each dimension
    walker_names = ["walker_dist", "walker_engage", "walker_lat",
                    "walker_height", "walker_presence"]
    for col, name in enumerate(walker_names):
        streams.append(("walker_features", col, name, False))

    for batch_key, col, feat_name, is_temporal in streams:
        drops = []
        for _ in range(n_repeats):
            acc = _eval_accuracy_permuted(
                stgcn, classifier, loader, stage_arg, device,
                batch_key, col, is_temporal,
            )
            drops.append(baseline_acc - acc)
        importance[feat_name] = float(np.mean(drops))

    return importance


@torch.no_grad()
def _eval_accuracy(stgcn, classifier, loader, stage_arg, device):
    """Compute accuracy on a data loader."""
    import torch
    correct = 0
    total = 0
    for batch in loader:
        skeleton = batch["skeleton"].to(device)
        skel_feat = batch["skeleton_features"].to(device)
        inter_feat = batch["interaction_features"].to(device)
        ctx_vec = batch["context_vector"].to(device)
        walker = batch["walker_features"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        emb = stgcn(skeleton, mask=mask)
        logits = classifier(emb, skel_feat, inter_feat, ctx_vec, walker,
                            stage=stage_arg)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def _eval_accuracy_permuted(stgcn, classifier, loader, stage_arg, device,
                             batch_key, col, is_temporal):
    """Evaluate accuracy with one feature column shuffled."""
    import torch
    correct = 0
    total = 0
    for batch in loader:
        skeleton = batch["skeleton"].to(device)
        skel_feat = batch["skeleton_features"].clone().to(device)
        inter_feat = batch["interaction_features"].clone().to(device)
        ctx_vec = batch["context_vector"].clone().to(device)
        walker = batch["walker_features"].clone().to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        # Shuffle the specified column across batch
        target_map = {
            "skeleton_features": skel_feat,
            "interaction_features": inter_feat,
            "context_vector": ctx_vec,
            "walker_features": walker,
        }
        tensor = target_map[batch_key]
        B = tensor.shape[0]
        perm = torch.randperm(B, device=device)

        if is_temporal:
            # (B, T, D) — shuffle column across batch
            tensor[:, :, col] = tensor[perm, :, col]
        else:
            # (B, D) — shuffle column across batch
            tensor[:, col] = tensor[perm, col]

        emb = stgcn(skeleton, mask=mask)
        logits = classifier(emb, skel_feat, inter_feat, ctx_vec, walker,
                            stage=stage_arg)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


# ── Report generation ────────────────────────────────────────────────────

def generate_report(per_stage_metrics, end_to_end_metrics, output_path):
    """Generate human-readable evaluation report.

    Args:
        per_stage_metrics: dict mapping stage_name -> metrics dict.
        end_to_end_metrics: end-to-end 5-class metrics dict.
        output_path: path to save text report.
    """
    lines = [
        "GMFCS Classification — Evaluation Report",
        "=" * 50,
        "",
    ]

    # Per-stage results
    for stage in ["stage1", "stage2a", "stage2b"]:
        m = per_stage_metrics.get(stage, {})
        lines.append(f"--- {stage.upper()} ---")
        lines.append(f"  Accuracy: {m.get('accuracy', 0):.1%}")
        lines.append(f"  Macro F1: {m.get('macro_f1', 0):.3f}")
        lines.append(f"  Samples:  {m.get('n_samples', 0)}")

        class_names = m.get("class_names", [])
        per_class = m.get("per_class", [])
        if class_names and per_class:
            lines.append(f"  {'Class':<15} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}")
            for name, mc in zip(class_names, per_class):
                lines.append(
                    f"  {name:<15} {mc['precision']:>6.3f} "
                    f"{mc['recall']:>6.3f} {mc['f1']:>6.3f} "
                    f"{mc['support']:>5d}"
                )
        lines.append("")

    # End-to-end
    lines.append("--- END-TO-END (5-CLASS GMFCS) ---")
    e2e = end_to_end_metrics
    acc = e2e.get("accuracy", 0)
    lines.append(f"  Overall Accuracy: {acc:.1%}")
    lines.append(f"  Macro F1:         {e2e.get('macro_f1', 0):.3f}")
    lines.append(f"  Target (80%):     {'MET' if e2e.get('target_80_met') else 'NOT MET'}")
    lines.append(f"  Samples:          {e2e.get('n_samples', 0)}")

    class_names = e2e.get("class_names", [])
    per_class = e2e.get("per_class", [])
    if class_names and per_class:
        lines.append(f"  {'Level':<15} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}")
        for name, mc in zip(class_names, per_class):
            lines.append(
                f"  {name:<15} {mc['precision']:>6.3f} "
                f"{mc['recall']:>6.3f} {mc['f1']:>6.3f} "
                f"{mc['support']:>5d}"
            )
    lines.append("")

    report_text = "\n".join(lines)
    Path(output_path).write_text(report_text)
    logger.info("Report saved to %s", output_path)
    print("\n" + report_text)
    return report_text
