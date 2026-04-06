"""Data loading and batching for multi-stream GMFCS classification.

Handles patient-level splits, triplet grouping, and synchronized loading
of skeleton sequences, features, and context vectors.

GMFCSDataset loads from npz feature files and returns:
  - skeleton:             (T, 17, 3) raw 3D skeleton, padded/truncated
  - skeleton_features:    (T, 15) Layer 1 device-proxy features
  - interaction_features: (T, 10) Layer 2 caregiver interaction features
  - context_vector:       (18,) Layer 3 per-clip metadata
  - walker_features:      (5,) walker-skeleton spatial features
  - movement_quality:     (6,) per-clip movement quality features
  - mask:                 (T,) 1.0 for real frames, 0.0 for padding
  - label:                int, stage-dependent class index

PatientLevelSplitter provides stratified patient-level cross-validation
with no data leakage between folds.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ── Label mapping for hierarchical stages ────────────────────────────────

# Stage 1: Ambulatory (0) vs Non-ambulatory (1)
# Routing by actual walking ability (w_status), not GMFCS level.
# Ambulatory: L1(all), L2(all), L3(ly,mkj,pjw), L4(hdi,jrh)
# Non-ambulatory: L3(kku), L4(lsa), L5(all)

# Stage 2A: Ambulatory branch — L1(0) vs L2(1) vs L3-L4(2)
STAGE2A_LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 2}

# Stage 2B: Non-ambulatory branch — L3-L4(0) vs L5(1)
STAGE2B_LABEL_MAP = {3: 0, 4: 0, 5: 1}

# Flat 5-class: L1(0) .. L5(4)
FLAT_LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}


def _load_labels(labels_path):
    """Load patient_id -> gmfcs_level mapping from labels.json."""
    with open(labels_path) as f:
        data = json.load(f)
    return {p["patient_id"]: p["gmfcs_level"] for p in data["patients"]}


def _load_ambulatory_status(labels_path, annotations_path):
    """Determine ambulatory status per patient from w_status in annotations.

    Ambulatory = patient has performed walk (w_status > 0 or assistance < 1.0
    for walk, or GMFCS L1/L2 which always walk).

    Returns:
        dict: patient_id -> bool (True = ambulatory).
    """
    labels = _load_labels(labels_path)

    # Default: L1/L2 always ambulatory, L5 always non-ambulatory
    status = {}
    for pid, level in labels.items():
        if level <= 2:
            status[pid] = True
        elif level == 5:
            status[pid] = False
        else:
            # L3/L4: check annotations for walk ability
            status[pid] = False  # default non-ambulatory

    # Override from annotations if available
    ann_path = Path(annotations_path)
    if ann_path.exists():
        with open(ann_path) as f:
            ann_data = json.load(f)
        for p in ann_data.get("patients", []):
            pid = p["patient_id"]
            level = labels.get(pid)
            if level in (3, 4):
                walk_info = p.get("per_movement", {}).get("walk", {})
                # If assistance < 1.0, patient can walk with some help
                assistance = walk_info.get("assistance", 0.0)
                device = walk_info.get("device", "none")
                if assistance > 0 or device != "none":
                    status[pid] = True

    return status


# ── Dataset ──────────────────────────────────────────────────────────────

class GMFCSDataset(Dataset):
    """PyTorch dataset for multi-stream GMFCS classification.

    Loads pre-extracted features from npz files and returns all streams
    needed by MultiStreamClassifier.

    Args:
        clip_ids: list of clip identifiers (npz file stems).
        features_dir: path to directory with {clip_id}.npz files.
        skeleton_3d_dir: path to directory with 3D skeleton npz files.
        labels: dict mapping patient_id -> gmfcs_level.
        stage: 'stage1', 'stage2a', 'stage2b', or 'flat'.
        max_seq_len: pad/truncate temporal sequences to this length.
            Default 150 (5 sec at 30fps).
        ambulatory_status: dict mapping patient_id -> bool. Required
            for stage1 label assignment.
    """

    def __init__(self, clip_ids, features_dir, skeleton_3d_dir, labels,
                 stage="flat", max_seq_len=150, ambulatory_status=None):
        self.clip_ids = list(clip_ids)
        self.features_dir = Path(features_dir)
        self.skeleton_3d_dir = Path(skeleton_3d_dir)
        self.labels = labels
        self.stage = stage
        self.max_seq_len = max_seq_len
        self.ambulatory_status = ambulatory_status or {}

        # Import here to avoid circular dependency at module level
        from src.utils.naming import clip_id_to_patient
        self._clip_id_to_patient = clip_id_to_patient

        # Build label map based on stage
        if stage == "stage1":
            self._label_fn = self._stage1_label
        elif stage == "stage2a":
            self._label_fn = self._stage2a_label
        elif stage == "stage2b":
            self._label_fn = self._stage2b_label
        else:
            self._label_fn = self._flat_label

    def _stage1_label(self, patient_id):
        """0 = ambulatory, 1 = non-ambulatory."""
        return 0 if self.ambulatory_status.get(patient_id, False) else 1

    def _stage2a_label(self, patient_id):
        """0=L1, 1=L2, 2=L3-L4 (ambulatory branch)."""
        level = self.labels.get(patient_id, 1)
        return STAGE2A_LABEL_MAP.get(level, 0)

    def _stage2b_label(self, patient_id):
        """0=L3-L4, 1=L5 (non-ambulatory branch)."""
        level = self.labels.get(patient_id, 5)
        return STAGE2B_LABEL_MAP.get(level, 1)

    def _flat_label(self, patient_id):
        """0-4 for GMFCS L1-L5."""
        level = self.labels.get(patient_id, 1)
        return FLAT_LABEL_MAP.get(level, 0)

    def _pad_or_truncate(self, arr, target_len):
        """Pad with zeros or truncate temporal sequence to target_len.

        Args:
            arr: (T, ...) array.
            target_len: desired T.

        Returns:
            Padded/truncated array of shape (target_len, ...).
            Mask of shape (target_len,) — 1 for real, 0 for pad.
        """
        T = arr.shape[0]
        rest_shape = arr.shape[1:]

        if T >= target_len:
            return arr[:target_len], np.ones(target_len, dtype=np.float32)

        padded = np.zeros((target_len,) + rest_shape, dtype=arr.dtype)
        padded[:T] = arr
        mask = np.zeros(target_len, dtype=np.float32)
        mask[:T] = 1.0
        return padded, mask

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, idx):
        clip_id = self.clip_ids[idx]
        patient_id = self._clip_id_to_patient(clip_id)

        # Load pre-extracted features
        feat_path = self.features_dir / f"{clip_id}.npz"
        feat_data = np.load(str(feat_path))

        skeleton_feats = feat_data.get("skeleton_features",
                                        np.zeros((1, 15), dtype=np.float32))
        interaction_feats = feat_data.get("interaction_features",
                                           np.zeros((1, 10), dtype=np.float32))
        context_vec = feat_data.get("context_vector",
                                     np.zeros(18, dtype=np.float32))
        walker_feats = feat_data.get("walker_features",
                                      np.zeros(5, dtype=np.float32))
        mq_feats = feat_data.get("movement_quality_features",
                                   np.zeros(6, dtype=np.float32))

        # Load raw 3D skeleton for ST-GCN input
        skel_path = self.skeleton_3d_dir / f"{clip_id}.npz"
        if skel_path.exists():
            skel_data = np.load(str(skel_path))
            skeleton_3d = skel_data.get("child",
                                         np.zeros((1, 17, 3), dtype=np.float32))
        else:
            T = skeleton_feats.shape[0]
            skeleton_3d = np.zeros((T, 17, 3), dtype=np.float32)

        # Pad/truncate temporal sequences to fixed length
        skeleton_3d, mask = self._pad_or_truncate(skeleton_3d, self.max_seq_len)
        skeleton_feats, _ = self._pad_or_truncate(skeleton_feats, self.max_seq_len)
        interaction_feats, _ = self._pad_or_truncate(interaction_feats,
                                                      self.max_seq_len)

        # Label
        label = self._label_fn(patient_id)

        return {
            "skeleton": torch.from_numpy(skeleton_3d).float(),
            "skeleton_features": torch.from_numpy(skeleton_feats).float(),
            "interaction_features": torch.from_numpy(interaction_feats).float(),
            "context_vector": torch.from_numpy(context_vec).float(),
            "walker_features": torch.from_numpy(walker_feats).float(),
            "movement_quality": torch.from_numpy(mq_feats).float(),
            "mask": torch.from_numpy(mask).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "clip_id": clip_id,
            "patient_id": patient_id,
        }


# ── Patient-level splitter ───────────────────────────────────────────────

class PatientLevelSplitter:
    """Stratified patient-level cross-validation splitter.

    CRITICAL: All clips from one patient are in the SAME fold.
    No data leakage between train and test.

    Stratification ensures each fold has representation from every
    GMFCS level when possible.

    Args:
        patient_ids: list of patient identifiers.
        gmfcs_levels: dict mapping patient_id -> gmfcs_level.
        n_folds: number of CV folds (default 6 for 24 patients = 4/fold).
        seed: random seed for reproducible splits.
    """

    def __init__(self, patient_ids, gmfcs_levels, n_folds=6, seed=42):
        self.patient_ids = sorted(set(patient_ids))
        self.gmfcs_levels = gmfcs_levels
        self.n_folds = n_folds
        self.seed = seed
        self.folds = self._create_folds()

    def _create_folds(self):
        """Create stratified patient-level folds.

        Groups patients by GMFCS level, then distributes across folds
        round-robin to ensure each fold has balanced representation.
        """
        rng = np.random.RandomState(self.seed)

        # Group patients by level
        level_groups = defaultdict(list)
        for pid in self.patient_ids:
            level = self.gmfcs_levels.get(pid, 0)
            level_groups[level].append(pid)

        # Shuffle within each level group
        for level in level_groups:
            rng.shuffle(level_groups[level])

        # Round-robin assignment across folds
        folds = [[] for _ in range(self.n_folds)]
        for level in sorted(level_groups.keys()):
            patients = level_groups[level]
            for i, pid in enumerate(patients):
                folds[i % self.n_folds].append(pid)

        return folds

    def split(self):
        """Generate train/test patient ID sets for each fold.

        Yields:
            (fold_idx, train_patients, test_patients) tuples.
        """
        for fold_idx in range(self.n_folds):
            test_patients = set(self.folds[fold_idx])
            train_patients = set()
            for i in range(self.n_folds):
                if i != fold_idx:
                    train_patients.update(self.folds[i])
            yield fold_idx, train_patients, test_patients

    def get_clip_split(self, clip_ids, clip_to_patient_fn):
        """Split clip IDs into train/test for each fold.

        Args:
            clip_ids: list of all clip identifiers.
            clip_to_patient_fn: function mapping clip_id -> patient_id.

        Yields:
            (fold_idx, train_clip_ids, test_clip_ids) tuples.
        """
        for fold_idx, train_patients, test_patients in self.split():
            train_clips = [c for c in clip_ids
                           if clip_to_patient_fn(c) in train_patients]
            test_clips = [c for c in clip_ids
                          if clip_to_patient_fn(c) in test_patients]
            yield fold_idx, train_clips, test_clips

    def __repr__(self):
        lines = [f"PatientLevelSplitter(n_folds={self.n_folds}, "
                 f"n_patients={len(self.patient_ids)})"]
        for i, fold in enumerate(self.folds):
            levels = [self.gmfcs_levels.get(p, "?") for p in fold]
            lines.append(f"  Fold {i}: {fold} (levels: {levels})")
        return "\n".join(lines)
