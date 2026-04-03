"""Hierarchical 2-stage training loop.

Stage 1: Ambulatory vs Non-ambulatory (binary, routed by w_status)
Stage 2-A: L1 vs L2 vs L3-L4 (ambulatory branch, walk quality + walker)
Stage 2-B: L3-L4 vs L5 (non-ambulatory branch, caregiver assistance)

Each stage trains independently with its own LiteSTGCN + MultiStreamClassifier.
Patient-level cross-validation with class-weight balancing.

Training features:
  - AdamW optimizer (lr=0.001, weight_decay=0.0001)
  - CosineAnnealingLR scheduler
  - CrossEntropyLoss with computed class weights
  - Early stopping (patience=20 on validation loss)
  - Mixed precision (torch.amp.autocast + GradScaler) for V100
  - DDP support for multi-GPU training
  - Model saving on rank 0 only
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.classifier import MultiStreamClassifier
from src.model.dataset import (
    GMFCSDataset,
    PatientLevelSplitter,
    _load_ambulatory_status,
    _load_labels,
)
from src.model.lite_stgcn import LiteSTGCN
from src.utils.naming import clip_id_to_patient

logger = logging.getLogger(__name__)


# ── Collate function ─────────────────────────────────────────────────────

def collate_fn(batch):
    """Custom collate for GMFCSDataset that handles mixed tensor/string fields."""
    return {
        k: torch.stack([b[k] for b in batch])
        if isinstance(batch[0][k], torch.Tensor)
        else [b[k] for b in batch]
        for k in batch[0].keys()
    }


# ── Utilities ────────────────────────────────────────────────────────────

def compute_class_weights(dataset):
    """Compute inverse-frequency class weights from a dataset.

    Returns:
        torch.FloatTensor of shape (num_classes,).
    """
    labels = []
    for i in range(len(dataset)):
        labels.append(dataset[i]["label"].item())
    labels = np.array(labels)

    classes = np.unique(labels)
    n_samples = len(labels)
    weights = np.zeros(classes.max() + 1, dtype=np.float32)
    for c in classes:
        count = (labels == c).sum()
        weights[c] = n_samples / (len(classes) * count + 1e-8)

    return torch.from_numpy(weights)


class EarlyStopping:
    """Early stopping monitor on validation loss."""

    def __init__(self, patience=20):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        """Returns True if training should stop."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            # Save best model state
            self.best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            return False
        self.counter += 1
        return self.counter >= self.patience

    def load_best(self, model):
        """Restore best model weights."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ── Single-stage trainer ─────────────────────────────────────────────────

class StageTrainer:
    """Trains one classification stage (ST-GCN + Classifier) for one fold.

    Args:
        stgcn: LiteSTGCN encoder instance.
        classifier: MultiStreamClassifier instance.
        stage_name: 'stage1', 'stage2a', or 'stage2b'.
        device: torch device.
        config: training config dict from default.yaml.
    """

    def __init__(self, stgcn, classifier, stage_name, device, config):
        self.stgcn = stgcn.to(device)
        self.classifier = classifier.to(device)
        self.stage_name = stage_name
        self.device = device

        self.epochs = config.get("epochs", 100)
        self.lr = config.get("lr", 0.001)
        self.weight_decay = config.get("weight_decay", 0.0001)
        self.patience = config.get("patience", 20)
        self.use_amp = torch.cuda.is_available()

        # Map stage names to classifier stage args
        self._stage_arg = {
            "stage1": 1, "stage2a": "2a", "stage2b": "2b",
        }.get(stage_name, None)

    def train(self, train_loader, val_loader, class_weights=None):
        """Train for one fold.

        Args:
            train_loader: DataLoader for training clips.
            val_loader: DataLoader for validation clips.
            class_weights: optional tensor of class weights for loss.

        Returns:
            dict with training history (train_loss, val_loss per epoch).
        """
        # Loss function
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(self.device)
            )
        else:
            criterion = nn.CrossEntropyLoss()

        # Optimizer — all parameters from both models
        params = list(self.stgcn.parameters()) + list(self.classifier.parameters())
        optimizer = torch.optim.AdamW(
            params, lr=self.lr, weight_decay=self.weight_decay
        )

        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        # Mixed precision
        scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Early stopping
        early_stop = EarlyStopping(patience=self.patience)

        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(self.epochs):
            # ── Train ──
            self.stgcn.train()
            self.classifier.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                skeleton = batch["skeleton"].to(self.device)
                skel_feat = batch["skeleton_features"].to(self.device)
                inter_feat = batch["interaction_features"].to(self.device)
                ctx_vec = batch["context_vector"].to(self.device)
                walker = batch["walker_features"].to(self.device)
                mask = batch["mask"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()

                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        embedding = self.stgcn(skeleton, mask=mask)
                        logits = self.classifier(
                            embedding, skel_feat, inter_feat,
                            ctx_vec, walker, stage=self._stage_arg,
                        )
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    embedding = self.stgcn(skeleton, mask=mask)
                    logits = self.classifier(
                        embedding, skel_feat, inter_feat,
                        ctx_vec, walker, stage=self._stage_arg,
                    )
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            scheduler.step()

            train_loss /= max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)

            # ── Validate ──
            val_loss, val_acc, _, _ = self.evaluate(val_loader, criterion)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    "[%s] Epoch %d/%d — train_loss=%.4f train_acc=%.3f "
                    "val_loss=%.4f val_acc=%.3f lr=%.6f",
                    self.stage_name, epoch + 1, self.epochs,
                    train_loss, train_acc, val_loss, val_acc,
                    scheduler.get_last_lr()[0],
                )

            # Early stopping check
            if early_stop.step(val_loss, nn.ModuleList([self.stgcn, self.classifier])):
                logger.info("[%s] Early stopping at epoch %d (patience=%d)",
                            self.stage_name, epoch + 1, self.patience)
                break

        # Restore best weights
        early_stop.load_best(nn.ModuleList([self.stgcn, self.classifier]))

        return history

    @torch.no_grad()
    def evaluate(self, loader, criterion=None):
        """Evaluate on a data loader.

        Returns:
            (loss, accuracy, all_preds, all_labels)
        """
        self.stgcn.eval()
        self.classifier.eval()

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in loader:
            skeleton = batch["skeleton"].to(self.device)
            skel_feat = batch["skeleton_features"].to(self.device)
            inter_feat = batch["interaction_features"].to(self.device)
            ctx_vec = batch["context_vector"].to(self.device)
            walker = batch["walker_features"].to(self.device)
            mask = batch["mask"].to(self.device)
            labels = batch["label"].to(self.device)

            embedding = self.stgcn(skeleton, mask=mask)
            logits = self.classifier(
                embedding, skel_feat, inter_feat,
                ctx_vec, walker, stage=self._stage_arg,
            )

            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        n = max(len(all_labels), 1)
        avg_loss = total_loss / n
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        acc = (all_preds == all_labels).mean() if len(all_labels) > 0 else 0.0

        return avg_loss, acc, all_preds, all_labels


# ── Hierarchical trainer ─────────────────────────────────────────────────

class HierarchicalTrainer:
    """Trains the full hierarchical 3-stage GMFCS classifier.

    Stage 1: Ambulatory vs Non-ambulatory (2 classes)
    Stage 2-A: L1 vs L2 vs L3-L4 (3 classes, ambulatory patients)
    Stage 2-B: L3-L4 vs L5 (2 classes, non-ambulatory patients)

    Each stage has its own LiteSTGCN + MultiStreamClassifier pair,
    trained independently with patient-level CV.
    """

    STAGE_CONFIGS = {
        "stage1": {"num_classes": 2, "stage_arg": 1},
        "stage2a": {"num_classes": 3, "stage_arg": "2a"},
        "stage2b": {"num_classes": 2, "stage_arg": "2b"},
    }

    def __init__(self, config):
        """
        Args:
            config: full config dict from default.yaml.
        """
        self.config = config
        self.model_cfg = config.get("model", {})
        self.train_cfg = config.get("training", {})
        self.device = torch.device(
            config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        )

        # Paths
        self.features_dir = config.get("features_dir", "data/features")
        self.skeleton_3d_dir = config.get("skeleton_3d_dir", "data/skeleton_3d")
        self.metadata_dir = config.get("metadata_dir", "data/metadata")
        self.output_dir = Path("outputs/models")
        self.log_dir = Path("outputs/logs")

        # Load metadata
        labels_path = Path(self.metadata_dir) / "labels.json"
        ann_path = Path(self.metadata_dir) / "assistive_annotations.json"
        self.labels = _load_labels(str(labels_path))
        self.ambulatory_status = _load_ambulatory_status(
            str(labels_path), str(ann_path)
        )

        # Discover available clips
        feat_dir = Path(self.features_dir)
        if feat_dir.exists():
            self.all_clip_ids = sorted(
                f.stem for f in feat_dir.glob("*.npz")
            )
        else:
            self.all_clip_ids = []

        self.patient_ids = sorted(self.labels.keys())
        logger.info("Found %d clips from %d patients",
                     len(self.all_clip_ids), len(self.patient_ids))

    def _create_models(self, num_classes):
        """Create fresh ST-GCN + Classifier pair."""
        stgcn = LiteSTGCN.from_config(self.model_cfg)

        fusion_cfg = self.model_cfg.get("fusion", {})
        classifier = MultiStreamClassifier(
            stgcn_dim=stgcn.output_dim,
            skeleton_feature_dim=self.model_cfg.get("skeleton_feature_dim", 15),
            interaction_feature_dim=self.model_cfg.get("interaction_feature_dim", 10),
            context_vector_dim=self.model_cfg.get("context_vector_dim", 22),
            walker_feature_dim=self.model_cfg.get("walker_feature_dim", 5),
            hidden_dim=fusion_cfg.get("hidden_dim", 64),
            dropout=fusion_cfg.get("dropout", 0.3),
            num_classes=num_classes,
            hierarchical=True,
        )
        return stgcn, classifier

    def _filter_clips_for_stage(self, clip_ids, stage_name):
        """Filter clips to those relevant for a given stage.

        Stage 2A: only ambulatory patients.
        Stage 2B: only non-ambulatory patients.
        """
        if stage_name == "stage2a":
            return [c for c in clip_ids
                    if self.ambulatory_status.get(clip_id_to_patient(c), False)]
        elif stage_name == "stage2b":
            return [c for c in clip_ids
                    if not self.ambulatory_status.get(clip_id_to_patient(c), True)]
        return clip_ids  # stage1 uses all clips

    def train_stage(self, stage_name, n_folds=None):
        """Train one stage with patient-level cross-validation.

        Args:
            stage_name: 'stage1', 'stage2a', or 'stage2b'.
            n_folds: number of CV folds (default from config).

        Returns:
            dict with per-fold results (predictions, labels, history).
        """
        if n_folds is None:
            n_folds = self.train_cfg.get("cv_folds", 6)

        stage_cfg = self.STAGE_CONFIGS[stage_name]
        num_classes = stage_cfg["num_classes"]
        batch_size = self.train_cfg.get("batch_size", 32)
        max_seq_len = 150

        logger.info("=" * 60)
        logger.info("Training %s (%d classes, %d folds)",
                     stage_name, num_classes, n_folds)
        logger.info("=" * 60)

        # Filter clips for this stage
        stage_clips = self._filter_clips_for_stage(self.all_clip_ids, stage_name)
        if not stage_clips:
            logger.warning("No clips for %s, skipping", stage_name)
            return {"folds": []}

        # Get patients in these clips
        stage_patients = sorted(set(
            clip_id_to_patient(c) for c in stage_clips
        ))
        stage_labels = {p: self.labels[p] for p in stage_patients
                        if p in self.labels}

        logger.info("Stage %s: %d clips, %d patients",
                     stage_name, len(stage_clips), len(stage_patients))

        # Patient-level splitter
        splitter = PatientLevelSplitter(
            stage_patients, stage_labels, n_folds=n_folds
        )

        results = {"folds": [], "stage_name": stage_name}

        for fold_idx, train_clips, test_clips in splitter.get_clip_split(
            stage_clips, clip_id_to_patient
        ):
            logger.info("--- Fold %d: train=%d clips, test=%d clips ---",
                         fold_idx, len(train_clips), len(test_clips))

            if not train_clips or not test_clips:
                logger.warning("Empty fold %d, skipping", fold_idx)
                continue

            # Create datasets
            train_ds = GMFCSDataset(
                train_clips, self.features_dir, self.skeleton_3d_dir,
                self.labels, stage=stage_name, max_seq_len=max_seq_len,
                ambulatory_status=self.ambulatory_status,
            )
            test_ds = GMFCSDataset(
                test_clips, self.features_dir, self.skeleton_3d_dir,
                self.labels, stage=stage_name, max_seq_len=max_seq_len,
                ambulatory_status=self.ambulatory_status,
            )

            # Data loaders
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True,
                collate_fn=collate_fn, num_workers=0, drop_last=False,
            )
            test_loader = DataLoader(
                test_ds, batch_size=batch_size, shuffle=False,
                collate_fn=collate_fn, num_workers=0,
            )

            # Fresh models for each fold
            stgcn, classifier = self._create_models(num_classes)

            # Compute class weights
            class_weights = compute_class_weights(train_ds)
            logger.info("  Class weights: %s", class_weights.numpy())

            # Train
            trainer = StageTrainer(
                stgcn, classifier, stage_name, self.device, self.train_cfg
            )
            history = trainer.train(train_loader, test_loader, class_weights)

            # Final evaluation
            _, test_acc, test_preds, test_labels = trainer.evaluate(test_loader)
            logger.info("  Fold %d test accuracy: %.3f", fold_idx, test_acc)

            # Save fold model
            fold_dir = self.output_dir / stage_name / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            torch.save(stgcn.state_dict(), fold_dir / "stgcn.pt")
            torch.save(classifier.state_dict(), fold_dir / "classifier.pt")

            results["folds"].append({
                "fold_idx": fold_idx,
                "test_acc": test_acc,
                "test_preds": test_preds.tolist(),
                "test_labels": test_labels.tolist(),
                "test_clip_ids": test_clips,
                "history": {k: [float(v) for v in vals]
                            for k, vals in history.items()},
                "n_train": len(train_clips),
                "n_test": len(test_clips),
            })

        # Aggregate results
        if results["folds"]:
            all_preds = []
            all_labels = []
            for f in results["folds"]:
                all_preds.extend(f["test_preds"])
                all_labels.extend(f["test_labels"])
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            overall_acc = (all_preds == all_labels).mean()
            results["overall_acc"] = float(overall_acc)
            logger.info("%s overall CV accuracy: %.3f", stage_name, overall_acc)
        else:
            results["overall_acc"] = 0.0

        return results

    def train_all(self):
        """Train all 3 hierarchical stages sequentially.

        Returns:
            dict with results for each stage.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        start = time.time()
        all_results = {}

        # Stage 1: Ambulatory vs Non-ambulatory
        all_results["stage1"] = self.train_stage("stage1")

        # Stage 2-A: L1 vs L2 vs L3-L4 (ambulatory branch)
        all_results["stage2a"] = self.train_stage("stage2a")

        # Stage 2-B: L3-L4 vs L5 (non-ambulatory branch)
        all_results["stage2b"] = self.train_stage("stage2b")

        elapsed = time.time() - start
        logger.info("Total training time: %.1f minutes", elapsed / 60)

        # Save combined results
        results_path = self.output_dir / "training_results.json"
        # Convert numpy types for JSON serialization
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info("Results saved to %s", results_path)

        # Summary
        logger.info("=" * 60)
        logger.info("Training Summary")
        logger.info("=" * 60)
        for stage in ["stage1", "stage2a", "stage2b"]:
            r = all_results.get(stage, {})
            acc = r.get("overall_acc", 0.0)
            n_folds = len(r.get("folds", []))
            logger.info("  %s: %.1f%% accuracy (%d folds)", stage, acc * 100, n_folds)

        return all_results
