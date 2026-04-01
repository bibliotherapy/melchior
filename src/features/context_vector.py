"""Layer 3: Extended metadata / assistive context vector encoder.

Encodes the 18D per-patient context vector from assistive_annotations.json,
extending the original 7D metadata (sex, age, movement statuses) with
walker type, AFO presence, and assistance levels per movement.

18D vector layout:
  [0]  sex                    # 0=female, 1=male
  [1]  age_normalized         # age_months / 72 (0-1)
  [2]  w_status               # 0=cannot, 1=performed, 2=not_needed
  [3]  cr_status              # same encoding
  [4]  c_s_status             # same encoding
  [5]  s_c_status             # same encoding
  [6]  sr_status              # same encoding
  [7]  walker_used            # 0/1
  [8]  walker_type            # 0=none, 1=anterior, 2=posterior
  [9]  afo_present            # 0/1
  [10] afo_laterality         # 0=none, 1=unilateral, 2=bilateral
  [11] support_surface_used   # 0/1 (acrylic stand)
  [12] caregiver_assist_walk  # 0.0-1.0 independence scale
  [13] caregiver_assist_c_s   # 0.0-1.0
  [14] caregiver_assist_crawl # 0.0-1.0
  [15] caregiver_assist_sr    # 0.0-1.0
  [16] overall_assistance     # 0-6 FIM-like, normalized to 0-1
  [17] device_count           # 0-3, normalized to 0-1
"""

import json
import numpy as np
from pathlib import Path


# Movement keys used across the pipeline
MOVEMENTS = ["walk", "crawl", "seated_to_standing", "standing_to_seated", "side_rolling"]

# Short keys for the 7D status portion of the vector
MOVEMENT_STATUS_KEYS = ["walk", "crawl", "seated_to_standing", "standing_to_seated", "side_rolling"]

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


class ContextVectorEncoder:
    """Encodes per-patient assistive context into an 18D normalized vector."""

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

    def encode(self, patient_id):
        """Encode a single patient into an 18D normalized vector.

        Args:
            patient_id: Patient identifier string.

        Returns:
            np.ndarray of shape (18,) with all values in [0, 1].
        """
        ann = self.annotations[patient_id]
        lab = self.labels.get(patient_id, {})
        devices = ann.get("devices", {})

        # [0] sex: 0=female, 1=male, 0.5 if unknown (-1)
        sex = lab.get("sex", -1)
        sex_val = 0.5 if sex == -1 else float(sex)

        # [1] age_normalized: age_months / 72, 0.5 if unknown
        age = lab.get("age_months", -1)
        age_val = 0.5 if age == -1 else min(float(age) / 72.0, 1.0)

        # [2-6] movement statuses (normalized: 0/0.5/1.0 for 0/1/2)
        statuses = []
        for mv_key in MOVEMENT_STATUS_KEYS:
            status = self._get_movement_status(ann, mv_key)
            statuses.append(status / 2.0)  # normalize 0-2 to 0-1

        # [7] walker_used
        walker_used = 1.0 if devices.get("walker", False) else 0.0

        # [8] walker_type (normalized: 0/0.5/1.0 for 0/1/2)
        walker_type = self._safe_walker_type(devices.get("walker_type", "none"))
        walker_type_val = walker_type / 2.0

        # [9] afo_present
        afo_present = 1.0 if devices.get("afo", False) else 0.0

        # [10] afo_laterality (normalized: 0/0.5/1.0 for 0/1/2)
        afo_lat = self._safe_afo_laterality(devices.get("afo_laterality", "none"))
        afo_lat_val = afo_lat / 2.0

        # [11] support_surface_used
        support_surface = 1.0 if devices.get("acrylic_stand", False) else 0.0

        # [12-15] per-movement caregiver assistance (already 0.0-1.0)
        assist_walk = self._get_assistance(ann, "walk")
        assist_c_s = self._get_assistance(ann, "seated_to_standing")
        assist_crawl = self._get_assistance(ann, "crawl")
        assist_sr = self._get_assistance(ann, "side_rolling")

        # [16] overall_assistance: FIM-like 0-6 normalized to 0-1
        overall = ann.get("overall_assistance", 0)
        overall_val = float(overall) / 6.0

        # [17] device_count: 0-3 normalized to 0-1
        device_count = self._count_devices(devices)
        device_count_val = device_count / 3.0

        vector = np.array([
            sex_val,            # [0]
            age_val,            # [1]
            statuses[0],        # [2] w_status
            statuses[1],        # [3] cr_status
            statuses[2],        # [4] c_s_status
            statuses[3],        # [5] s_c_status
            statuses[4],        # [6] sr_status
            walker_used,        # [7]
            walker_type_val,    # [8]
            afo_present,        # [9]
            afo_lat_val,        # [10]
            support_surface,    # [11]
            assist_walk,        # [12]
            assist_c_s,         # [13]
            assist_crawl,       # [14]
            assist_sr,          # [15]
            overall_val,        # [16]
            device_count_val,   # [17]
        ], dtype=np.float32)

        return vector

    def encode_all(self):
        """Encode all patients.

        Returns:
            dict mapping patient_id -> np.ndarray of shape (18,)
        """
        return {pid: self.encode(pid) for pid in self.annotations}

    def get_field_names(self):
        """Return human-readable names for each of the 18 dimensions."""
        return [
            "sex", "age_normalized",
            "w_status", "cr_status", "c_s_status", "s_c_status", "sr_status",
            "walker_used", "walker_type", "afo_present", "afo_laterality",
            "support_surface_used",
            "assist_walk", "assist_seated_to_standing",
            "assist_crawl", "assist_side_rolling",
            "overall_assistance", "device_count",
        ]

    def print_summary(self, patient_id):
        """Print a readable summary of a patient's context vector."""
        vec = self.encode(patient_id)
        ann = self.annotations[patient_id]
        names = self.get_field_names()
        print(f"\n{'='*60}")
        print(f"Patient: {patient_id}  |  GMFCS Level: {ann['gmfcs_level']}")
        print(f"{'='*60}")
        for i, (name, val) in enumerate(zip(names, vec)):
            print(f"  [{i:2d}] {name:<28s} = {val:.3f}")
        print(f"{'='*60}")


def verify_context_vectors(annotations_path, labels_path):
    """Verification function: print vectors for one L1, one L3, one L5 patient.

    Expected patterns:
      L1: all assistance = 1.0, no devices, overall_assistance = 1.0
      L3: walker_used = 1, walk assistance = 1.0, overall_assistance ~ 0.83
      L5: all assistance low (0.0-0.2), no devices, overall_assistance ~ 0.17
    """
    encoder = ContextVectorEncoder(annotations_path, labels_path)

    # Pick representative patients
    l1_patient = "jyh"
    l3_patient = "kku"
    l5_patient = "ajy"

    for pid in [l1_patient, l3_patient, l5_patient]:
        encoder.print_summary(pid)

    # Encode all and print shape summary
    all_vectors = encoder.encode_all()
    stacked = np.stack(list(all_vectors.values()))
    print(f"\nAll patients encoded: {len(all_vectors)} patients, vector shape: {stacked.shape}")
    print(f"Value range: [{stacked.min():.3f}, {stacked.max():.3f}]")

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
