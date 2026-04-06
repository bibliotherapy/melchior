"""Layer 3: Per-clip assistive context vector encoder.

Encodes an 18D per-clip context vector from assistive_annotations.json.
The vector is split into a patient-invariant zone (same for all clips
from a patient) and a clip-specific zone (varies by movement type).

18D vector layout:

  Patient-Invariant Zone [0-10] — "Who is this patient?"
  [0]  sex                   # 0=female, 1=male
  [1]  age_normalized        # age_months / 72 (0-1)
  [2]  w_status              # 0=cannot, 0.5=performed, 1.0=not_needed
  [3]  cr_status             # same encoding
  [4]  c_s_status            # floor sit-to-stand
  [5]  s_c_status            # floor stand-to-sit
  [6]  sr_status             # side rolling
  [7]  cc_s_status           # chair sit-to-stand
  [8]  s_cc_status           # chair stand-to-sit
  [9]  overall_assistance    # FIM 0-6 normalized → 0-1
  [10] device_count          # 0-3 normalized → 0-1

  Clip-Specific Zone [11-17] — "What's happening in THIS clip?"
  [11] clip_walker           # 0/1 — walker used in THIS movement
  [12] clip_walker_type      # 0/0.5/1.0 (none/anterior/posterior)
  [13] clip_afo              # 0/1 — AFO worn during THIS movement
  [14] clip_afo_laterality   # 0/0.5/1.0 (none/unilateral/bilateral)
  [15] clip_acrylic_stand    # 0/1 — acrylic stand used
  [16] clip_surface          # 0=floor, 0.5=n/a, 1.0=chair
  [17] clip_assistance       # 0.0-1.0 independence for THIS movement
"""

import json
import numpy as np
from pathlib import Path


# Movement keys used across the pipeline (floor + chair variants)
MOVEMENTS = [
    "walk", "crawl", "seated_to_standing", "standing_to_seated", "side_rolling",
    "chair_seated_to_standing", "standing_to_chair_seated",
]

# Short keys for the base 5 movement status portion of the vector [2-6]
MOVEMENT_STATUS_KEYS = ["walk", "crawl", "seated_to_standing", "standing_to_seated", "side_rolling"]

# Chair movement status keys at [7-8]
CHAIR_MOVEMENT_STATUS_KEYS = ["chair_seated_to_standing", "standing_to_chair_seated"]

# Walker type encoding
WALKER_TYPE_MAP = {"none": 0, "anterior": 1, "posterior": 2, "body_support": 2}

# AFO laterality encoding
AFO_LATERALITY_MAP = {
    "none": 0,
    "unilateral_left": 1,
    "unilateral_right": 1,
    "bilateral": 2,
}

# GMFCS levels where inability to perform a movement means "cannot" (0) vs "not needed" (2)
# L1-L2: ambulatory, most movements not needed if absent
# L4-L5: non-ambulatory, movements absent because they cannot perform them
AMBULATORY_LEVELS = {1, 2, 3}

# Movement code (from clip_id) → annotation key (in per_movement)
MOVEMENT_CODE_MAP = {
    "w": "walk",
    "cr": "crawl",
    "c_s": "seated_to_standing",
    "s_c": "standing_to_seated",
    "sr": "side_rolling",
    "cc_s": "chair_seated_to_standing",
    "s_cc": "standing_to_chair_seated",
}

# Device string → (walker, afo, acrylic_stand) decomposition
DEVICE_DECOMPOSITION = {
    "none":           (0, 0, 0),
    "walker":         (1, 0, 0),
    "afo_only":       (0, 1, 0),
    "acrylic_stand":  (0, 0, 1),
    "walker_and_afo": (1, 1, 0),
}

# Movement code → surface type (floor=0, n/a=0.5, chair=1.0)
SURFACE_MAP = {
    "c_s": 0.0, "s_c": 0.0,      # floor transitions
    "cc_s": 1.0, "s_cc": 1.0,    # chair transitions
}


class ContextVectorEncoder:
    """Encodes per-clip assistive context into an 18D normalized vector."""

    def __init__(self, annotations_path, labels_path):
        """
        Args:
            annotations_path: Path to assistive_annotations.json
            labels_path: Path to labels.json (sex, age_months)
        """
        self.annotations_path = Path(annotations_path)
        self.labels_path = Path(labels_path)

        with open(self.annotations_path) as f:
            ann_data = json.load(f)
        self.annotations = {p["patient_id"]: p for p in ann_data["patients"]}

        with open(self.labels_path) as f:
            lab_data = json.load(f)
        self.labels = {p["patient_id"]: p for p in lab_data["patients"]}

    def _get_movement_status(self, patient_ann, movement_key):
        """Derive movement status: 0=cannot, 1=performed, 2=not_needed.

        If the movement key exists in per_movement, the patient performed it (1).
        If absent, infer from GMFCS level:
          - Ambulatory (L1-L3): absent movement is "not needed" (2)
          - Non-ambulatory (L4-L5): absent movement is "cannot perform" (0)
        """
        if movement_key in patient_ann.get("per_movement", {}):
            return 1  # performed
        gmfcs = patient_ann["gmfcs_level"]
        if gmfcs in AMBULATORY_LEVELS:
            return 2  # not needed (too mild)
        return 0  # cannot perform (too severe)

    def _safe_walker_type(self, walker_type_str):
        """Handle TODO strings in walker_type field."""
        if isinstance(walker_type_str, str) and walker_type_str.startswith("TODO"):
            return 0  # unknown, treat as none until annotated
        return WALKER_TYPE_MAP.get(walker_type_str, 0)

    def _safe_afo_laterality(self, laterality_str):
        """Handle TODO strings in afo_laterality field."""
        if isinstance(laterality_str, str) and laterality_str.startswith("TODO"):
            return 0
        return AFO_LATERALITY_MAP.get(laterality_str, 0)

    def _get_assistance(self, patient_ann, movement_key):
        """Get assistance level for a movement. Returns 0.0 if not performed."""
        movement_data = patient_ann.get("per_movement", {}).get(movement_key)
        if movement_data is None:
            return 0.0  # not performed = dependent
        return float(movement_data.get("assistance", 0.0))

    def _count_devices(self, devices):
        """Count number of distinct assistive devices used (0-3)."""
        count = 0
        if devices.get("walker", False):
            count += 1
        if devices.get("afo", False):
            count += 1
        if devices.get("acrylic_stand", False):
            count += 1
        return count

    def encode(self, patient_id, movement_type=None):
        """Encode a single clip into an 18D normalized vector.

        Args:
            patient_id: Patient identifier string.
            movement_type: Short movement code (e.g. "w", "cr", "cc_s").
                If None, clip-specific zone is filled with zeros.

        Returns:
            np.ndarray of shape (18,) with all values in [0, 1].
        """
        ann = self.annotations[patient_id]
        lab = self.labels.get(patient_id, {})
        devices = ann.get("devices", {})

        # ── Patient-Invariant Zone [0-10] ──────────────────────────────

        # [0] sex: 0=female, 1=male, 0.5 if unknown (-1)
        sex = lab.get("sex", -1)
        sex_val = 0.5 if sex == -1 else float(sex)

        # [1] age_normalized: age_months / 72, 0.5 if unknown
        age = lab.get("age_months", -1)
        age_val = 0.5 if age == -1 else min(float(age) / 72.0, 1.0)

        # [2-6] base movement statuses (normalized: 0/0.5/1.0 for 0/1/2)
        statuses = []
        for mv_key in MOVEMENT_STATUS_KEYS:
            status = self._get_movement_status(ann, mv_key)
            statuses.append(status / 2.0)  # normalize 0-2 to 0-1

        # [7-8] chair movement statuses
        chair_statuses = []
        for mv_key in CHAIR_MOVEMENT_STATUS_KEYS:
            status = self._get_movement_status(ann, mv_key)
            chair_statuses.append(status / 2.0)

        # [9] overall_assistance: FIM-like 0-6 normalized to 0-1
        overall = ann.get("overall_assistance", 0)
        overall_val = float(overall) / 6.0

        # [10] device_count: 0-3 normalized to 0-1
        device_count = self._count_devices(devices)
        device_count_val = device_count / 3.0

        # ── Clip-Specific Zone [11-17] ─────────────────────────────────

        ann_key = MOVEMENT_CODE_MAP.get(movement_type) if movement_type else None
        per_movement = ann.get("per_movement", {})

        if ann_key is None:
            # movement_type not provided or not recognized — zero out clip zone
            clip_walker = 0.0
            clip_walker_type = 0.0
            clip_afo = 0.0
            clip_afo_lat = 0.0
            clip_acrylic = 0.0
            clip_surface = 0.5
            clip_assistance = 0.0
        else:
            mv_data = per_movement.get(ann_key)

            if mv_data is not None:
                # Per-movement annotation exists — decompose device string
                device_str = mv_data.get("device", "none")
                if isinstance(device_str, str) and device_str.startswith("TODO"):
                    device_str = "none"
                walker_flag, afo_flag, acrylic_flag = DEVICE_DECOMPOSITION.get(
                    device_str, (0, 0, 0)
                )

                clip_walker = float(walker_flag)
                clip_walker_type = (
                    self._safe_walker_type(devices.get("walker_type", "none")) / 2.0
                    if walker_flag else 0.0
                )
                clip_afo = float(afo_flag)
                clip_afo_lat = (
                    self._safe_afo_laterality(devices.get("afo_laterality", "none")) / 2.0
                    if afo_flag else 0.0
                )
                clip_acrylic = float(acrylic_flag)
                clip_surface = SURFACE_MAP.get(movement_type, 0.5)
                clip_assistance = float(mv_data.get("assistance", 0.0))
            else:
                # Movement not in per_movement — fall back to patient-level defaults
                clip_walker = 1.0 if devices.get("walker", False) else 0.0
                clip_walker_type = (
                    self._safe_walker_type(devices.get("walker_type", "none")) / 2.0
                    if devices.get("walker", False) else 0.0
                )
                clip_afo = 1.0 if devices.get("afo", False) else 0.0
                clip_afo_lat = (
                    self._safe_afo_laterality(devices.get("afo_laterality", "none")) / 2.0
                    if devices.get("afo", False) else 0.0
                )
                clip_acrylic = 1.0 if devices.get("acrylic_stand", False) else 0.0
                clip_surface = SURFACE_MAP.get(movement_type, 0.5)
                clip_assistance = overall_val  # fall back to overall

        # ── Assemble 18D vector ────────────────────────────────────────

        vector = np.array([
            sex_val,            # [0]
            age_val,            # [1]
            statuses[0],        # [2]  w_status
            statuses[1],        # [3]  cr_status
            statuses[2],        # [4]  c_s_status
            statuses[3],        # [5]  s_c_status
            statuses[4],        # [6]  sr_status
            chair_statuses[0],  # [7]  cc_s_status
            chair_statuses[1],  # [8]  s_cc_status
            overall_val,        # [9]  overall_assistance
            device_count_val,   # [10] device_count
            clip_walker,        # [11]
            clip_walker_type,   # [12]
            clip_afo,           # [13]
            clip_afo_lat,       # [14]
            clip_acrylic,       # [15]
            clip_surface,       # [16]
            clip_assistance,    # [17]
        ], dtype=np.float32)

        return vector

    def encode_all(self):
        """Encode all patients with movement_type=None (patient-invariant zone only).

        Deprecated: Use encode(patient_id, movement_type) for per-clip encoding.

        Returns:
            dict mapping patient_id -> np.ndarray of shape (18,)
        """
        return {pid: self.encode(pid, movement_type=None) for pid in self.annotations}

    def get_field_names(self):
        """Return human-readable names for each of the 18 dimensions."""
        return [
            "sex", "age_normalized",
            "w_status", "cr_status", "c_s_status", "s_c_status", "sr_status",
            "cc_s_status", "s_cc_status",
            "overall_assistance", "device_count",
            "clip_walker", "clip_walker_type",
            "clip_afo", "clip_afo_laterality",
            "clip_acrylic_stand", "clip_surface", "clip_assistance",
        ]

    def print_summary(self, patient_id, movement_type=None):
        """Print a readable summary of a clip's context vector."""
        vec = self.encode(patient_id, movement_type)
        ann = self.annotations[patient_id]
        names = self.get_field_names()
        print(f"\n{'='*60}")
        mv_str = f" | Movement: {movement_type}" if movement_type else ""
        print(f"Patient: {patient_id}  |  GMFCS Level: {ann['gmfcs_level']}{mv_str}")
        print(f"{'='*60}")
        for i, (name, val) in enumerate(zip(names, vec)):
            print(f"  [{i:2d}] {name:<28s} = {val:.3f}")
        print(f"{'='*60}")


def verify_context_vectors(annotations_path, labels_path):
    """Verification: test per-clip encoding with different movements.

    Checks:
      - Same patient + different movements → different vectors
      - Shape is (18,), values in [0, 1]
      - L1/L3/L5 patients produce expected patterns
    """
    encoder = ContextVectorEncoder(annotations_path, labels_path)

    # Per-clip verification: same patient, different movements
    print("=== Per-clip verification (patient kku, L3 walker user) ===")
    encoder.print_summary("kku", "w")    # walk — walker=1
    encoder.print_summary("kku", "cr")   # crawl — walker=0
    encoder.print_summary("kku", "c_s")  # floor sit-to-stand — surface=0

    # Cross-level check
    print("\n=== Cross-level verification ===")
    encoder.print_summary("jyh", "w")    # L1, no devices
    encoder.print_summary("ajy", "sr")   # L5, low assistance

    # movement_type=None fallback
    print("\n=== movement_type=None fallback ===")
    encoder.print_summary("kku", None)

    # Verify shape and value range
    vec_walk = encoder.encode("kku", "w")
    vec_crawl = encoder.encode("kku", "cr")
    assert vec_walk.shape == (18,), f"Expected (18,), got {vec_walk.shape}"
    assert vec_walk.min() >= 0.0, f"Vector has negative values"
    assert vec_walk.max() <= 1.0, f"Vector exceeds 1.0"

    # Same patient, different movements → different vectors
    assert not np.array_equal(vec_walk, vec_crawl), \
        "Walk and crawl vectors should differ for walker user"
    # Patient-invariant zone should be identical
    assert np.array_equal(vec_walk[:11], vec_crawl[:11]), \
        "Patient-invariant zone should be identical across movements"

    # Encode all (deprecated path)
    all_vectors = encoder.encode_all()
    stacked = np.stack(list(all_vectors.values()))
    print(f"\nAll patients encoded: {len(all_vectors)} patients, vector shape: {stacked.shape}")
    print(f"Value range: [{stacked.min():.3f}, {stacked.max():.3f}]")

    print("\n✓ All verifications passed.")
    return all_vectors


if __name__ == "__main__":
    import sys

    base = Path(__file__).resolve().parent.parent.parent / "data" / "metadata"
    ann_path = base / "assistive_annotations.json"
    lab_path = base / "labels.json"

    if not ann_path.exists():
        print(f"ERROR: {ann_path} not found")
        sys.exit(1)
    if not lab_path.exists():
        print(f"ERROR: {lab_path} not found")
        sys.exit(1)

    verify_context_vectors(str(ann_path), str(lab_path))
