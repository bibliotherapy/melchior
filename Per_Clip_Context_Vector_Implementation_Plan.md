# Per-Clip Context Vector Implementation Plan

## Context

The context vector is currently per-patient (same 22D vector for all clips of a patient). This creates a fundamental data mismatch: a walk clip using a walker gets the same metadata as a crawl clip using no device. The new design produces an 18D per-clip vector with movement-specific device/assistance context.

**Design spec:** `docs/superpowers/specs/2026-04-03-per-clip-context-vector-design.md`

## Implementation Steps

### Step 1: Restructure ContextVectorEncoder
**File:** `src/features/context_vector.py`

- Change signature: `encode(patient_id)` → `encode(patient_id, movement_type)`
- Add movement-code-to-annotation-key mapping
- Add device string decomposition (e.g., `"walker_and_afo"` → walker=1, afo=1)
- Implement patient-invariant zone (indices 0-10): unchanged logic
- Implement clip-specific zone (indices 11-17): read from `per_movement[key]`
- Add surface encoding: c_s/s_c→0 (floor), cc_s/s_cc→1 (chair), others→0.5
- Update docstring, `get_field_names()`, `encode_all()`
- `encode_all()` may need to become `encode_all(movement_types_by_patient)` or be removed (it's per-patient only)

### Step 2: Update Feature Extraction Script
**File:** `scripts/04_extract_features.py`

- Pass `movement_type` to `context_encoder.encode(patient_id, movement_type)` at line ~170
- The `movement_type` is already extracted via `clip_id_to_movement(clip_id)` — just pass it through

### Step 3: Update Dimension Cascade (22D → 18D)
**Files:**
- `src/model/classifier.py`: `context_vector_dim=18` default, update all docstrings/comments
- `src/model/dataset.py`: `np.zeros(18)` fallback, update docstring
- `src/model/train.py`: fallback to 18
- `scripts/06_evaluate.py`: fallback to 18
- `configs/default.yaml`: `context_vector_dim: 18`
- `CLAUDE.md`: 18D references, ~205D → ~201D fusion

### Step 4: Update Annotation Data
**File:** `data/metadata/assistive_annotations.json`

- Verify all patients have per-movement device/assistance entries for movements they performed
- Add chair movement entries (chair_seated_to_standing, standing_to_chair_seated)
- Ensure `device` field correctly reflects per-movement reality (not just patient-level defaults)

### Step 5: Re-Extract All Features
- Re-run `scripts/04_extract_features.py` on ALL existing clips
- Each clip's `.npz` now gets a movement-specific 18D context vector instead of the shared 22D
- Verify output: same patient, different movements → different context_vector in npz

### Step 6: Verify and Test
- Unit tests for encoder: same patient + different movements → different vectors
- Model forward pass with 18D: no dimension errors
- Run 5-fold CV and compare accuracy: per-clip encoding vs old per-patient encoding

## Verification Checklist
- [ ] `encode("kku", "w")` → clip_walker=1
- [ ] `encode("kku", "cr")` → clip_walker=0
- [ ] `encode("kku", "c_s")` → clip_acrylic=1, clip_surface=0
- [ ] Vector shape is (18,) for all calls
- [ ] All existing clips re-extracted with new vectors
- [ ] Model trains without dimension errors
- [ ] Accuracy comparison: per-clip vs per-patient baseline
