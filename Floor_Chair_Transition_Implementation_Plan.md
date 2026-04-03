# Floor/Chair Transition Sub-Types: Design & Implementation Plan

## Context

**Problem:** The GMFCS classification pipeline processes only 5 movement types, silently dropping 362 chair-based transition clips (`chair_seated_to_standing`: 221, `standing_to_chair_seated`: 141) — 22% of potentially usable data. The raw video data already labels these separately, but the pipeline never ingests them.

**Why this matters for accuracy:**
- 54.2% of patients are ages 4-6, exactly where the GMFCS-E&R explicitly describes *different* floor vs chair transfer capabilities per level (L1: no hand support; L2: needs surface to push; L3: needs pelvic/trunk support; L4: needs adult assistance)
- Chair transfer independence is a key discriminator for L2/L3/L4 — the hardest levels to distinguish
- Floor-to-stand and chair-to-stand have different biomechanics (starting height, ROM, muscle activation), so mixing them or ignoring one loses signal

**Outcome:** Add `cc_s` (chair sit-to-stand) and `s_cc` (chair stand-to-sit) as distinct movement types, with surface-adapted feature extraction, expanded context vector (18D→22D), and +22% more training data.

**Design spec:** `docs/superpowers/specs/2026-04-03-floor-chair-transition-subtype-design.md`

---

## Design Summary

### 1. Movement Type System (5 → 7)

| Code | Full Name | Clips | Status |
|---|---|---:|---|
| `w` | walk | 640 | Existing |
| `cr` | crawl | 363 | Existing |
| `c_s` | seated_to_standing (floor) | 304 | Existing |
| `s_c` | standing_to_seated (floor) | 212 | Existing |
| `sr` | side_rolling | 128 | Existing |
| **`cc_s`** | **chair_seated_to_standing** | **221** | **New** |
| **`s_cc`** | **standing_to_chair_seated** | **141** | **New** |

**Excluded:** `seated_to_chair_seated` (11 clips, 96% L3-exclusive — shortcut learning risk).

### 2. Feature Extraction: Surface-Parameterized

Add `surface="floor"|"chair"` parameter to existing `SitToStandQualityFeatures` and `StandToSitQualityFeatures`:

| Feature | Floor Behavior | Chair Adaptation |
|---|---|---|
| S1: Transition duration | Normalize by 5.0s | Normalize by 3.0s (shorter ROM from elevated start) |
| S3: Hand-to-ground contact | 5th-percentile wrist height = "ground" | Initial pelvis height = chair seat surface |
| D4: Hand support frequency | 5th-percentile wrist height | Final pelvis height = target chair surface |
| S2, S4, S5, S6, D1-D3 | Unchanged | Unchanged (surface-agnostic biomechanics) |

Dispatcher maps: `cc_s → (c_s, "chair")`, `s_cc → (s_c, "chair")`.

### 3. Context Vector: 18D → 22D

```
Existing [0-17] unchanged
[18] cc_s_status             # 0=cannot, 1=performed, 2=not_needed
[19] s_cc_status             # same encoding
[20] caregiver_assist_cc_s   # 0.0-1.0 independence scale
[21] caregiver_assist_s_cc   # 0.0-1.0
```

### 4. Model Dimension Cascade

- Stream D (context passthrough): 18 → 22
- Fusion concat: ~201D → ~205D
- `classifier.py` context_dim default update
- `configs/default.yaml` dimension update

---

## Implementation Steps

### Step 0: Save Design Spec
- Write design doc to `docs/superpowers/specs/2026-04-03-floor-chair-transition-subtype-design.md`
- Save this plan to project root as `Floor_Chair_Transition_Implementation_Plan.md`

### Step 1: Update Naming Utilities
**File:** `src/utils/naming.py`
- Add `"cc_s"`, `"s_cc"` to `compound_codes` in `clip_id_to_movement()`
- Rewrite `clip_id_to_patient()` to check compound codes *before* single-char `"s"` / `"c"` match (current code has `{"w", "cr", "c", "s", "sr"}` which would incorrectly match the `s` in `cc_s`)
- Add test: `clip_id_to_movement("hdi_cc_s_01_FV")` == `"cc_s"`, `clip_id_to_patient("hdi_cc_s_01_FV")` == `"hdi"`

### Step 2: Update Feature Extraction
**File:** `src/features/movement_quality.py`
- Add `surface="floor"` parameter to `SitToStandQualityFeatures.compute()` (line 275)
- In S1 (line 314): `norm_duration = 5.0 if surface == "floor" else 3.0`
- In S3 (lines 332-340): if surface == "chair", use initial pelvis height (`hip_height_norm[:n_start]` median) as reference instead of 5th-percentile wrist height
- Add `surface="floor"` parameter to `StandToSitQualityFeatures.compute()` (line 623)
- In D4 (lines 704-712): if surface == "chair", use final pelvis height as reference
- Add `CHAIR_MAP = {"cc_s": ("c_s", "chair"), "s_cc": ("s_c", "chair")}` before dispatcher
- Update `extract_movement_quality_features()` (line 728) to resolve chair map before dispatch
- Update `MOVEMENT_CLASSES` (line 55): add `"cc_s"`, `"s_cc"`
- Update `get_movement_quality_feature_names()` with chair-prefixed names

### Step 3: Expand Context Vector
**File:** `src/features/context_vector.py`
- Add `"chair_seated_to_standing"`, `"standing_to_chair_seated"` to `MOVEMENTS` (line 34)
- Add to `MOVEMENT_STATUS_KEYS` (line 37)
- In `encode()` (line 121): extend vector assembly with indices [18]-[21]
- Update `get_field_names()` (line 211)
- Update module docstring (lines 1-26) with new 22D layout
- Change all references from "18D" to "22D"

### Step 4: Update Annotation Schema & Data
**File:** `data/metadata/annotation_schema.json`
- Add `"chair_seated_to_standing"` and `"standing_to_chair_seated"` to `per_movement.properties`
- Same `$ref: "#/definitions/movement_annotation"` (device, assistance, notes)

**File:** `data/metadata/assistive_annotations.json`
- Add chair movement annotations for patients who performed them
- This requires reviewing which patients have chair clips (likely concentrated in L1-L4)
- Fill device and assistance fields based on clinical records

### Step 5: Prepare Chair Clip Data
This is the largest manual step:
1. Map raw chair video files to clip IDs using the `cc_s`/`s_cc` naming convention
2. Annotate first frames for SAM2 tracking (`data/metadata/sam2_annotations.json`)
3. Run processing pipeline: SAM2 → pose → calibration → triangulation → feature extraction
4. Verify output npz files contain correct features

**Note:** The 221+141 chair clips are in raw video form with Korean filenames. A mapping script may be needed to convert raw filenames to standardized clip IDs.

### Step 6: Update Model & Training
**File:** `src/model/classifier.py`
- Update `context_dim` parameter default: 18 → 22
- Verify fusion dimension calculation auto-adjusts (should be sum of all stream dims)

**File:** `src/model/dataset.py`
- Verify `GMFCSDataset.__getitem__()` handles new movement codes (it loads from npz, movement-agnostic)
- Verify label mapping still works (patient-level, movement-agnostic)

**File:** `configs/default.yaml`
- Update `context_vector_dim: 22`
- Add `cc_s`, `s_cc` to movement_types list if one exists

**File:** `src/model/train.py`
- Verify `_filter_clips_for_stage()` works with new clip IDs (filters by patient, not movement)

### Step 7: Update Evaluation
**File:** `scripts/06_evaluate.py`, `src/utils/evaluation.py`
- Update movement type lists for per-movement accuracy metrics
- Add chair transition accuracy reporting
- Compare floor vs chair feature distributions for sanity check

---

## Critical Files Summary

| File | Change |
|---|---|
| `src/utils/naming.py` | Add cc_s/s_cc parsing |
| `src/features/movement_quality.py` | Surface parameter, chair map, dispatcher |
| `src/features/context_vector.py` | 18D → 22D expansion |
| `src/model/classifier.py` | context_dim default update |
| `configs/default.yaml` | Dimension and movement list updates |
| `data/metadata/annotation_schema.json` | Add chair movement definitions |
| `data/metadata/assistive_annotations.json` | Add per-patient chair annotations |
| `data/metadata/sam2_annotations.json` | Add first-frame annotations for chair clips |

---

## Verification

### Unit Tests
- `clip_id_to_movement("hdi_cc_s_01_FV")` == `"cc_s"`
- `clip_id_to_patient("hdi_cc_s_01_FV")` == `"hdi"`
- `clip_id_to_patient("hdi_s_cc_01_FV")` == `"hdi"`
- Feature extraction returns 6 features for `cc_s` (chair c_s), 4 for `s_cc` (chair s_c)
- Context vector encoder returns shape `(22,)` for all patients
- Chair S3 feature uses pelvis height reference (not wrist percentile)

### Integration
- Process one chair clip through full pipeline (tracking → pose → calibration → triangulation → features)
- Verify output npz contains valid `movement_quality_features`, `skeleton_features`, etc.
- Load in `GMFCSDataset`, verify tensor shapes match model expectations

### Training Validation
- Run 5-fold patient-level CV with 7-movement dataset
- Compare overall accuracy vs baseline 5-movement model
- Check per-level accuracy improvements, especially L2/L3/L4
- Verify no overfitting from added data (validation loss should not increase)
- Feature importance analysis: do chair-specific features contribute to L3 discrimination?
