# Per-Clip Context Vector Design

**Date:** 2026-04-03
**Status:** Approved
**Scope:** Restructure context vector from per-patient to per-clip encoding

## Problem

The context vector is currently **per-patient**: `encode(patient_id)` produces one identical 22D vector that is broadcast to every clip from that patient. This creates a fundamental mismatch:

- A walk clip using a **walker** gets the same metadata as a crawl clip using **no device**
- A floor sit-to-stand clip gets the same `acrylic_stand=0` as a chair sit-to-stand — even when the acrylic stand IS used for the floor version
- All 7 per-movement assistance values are broadcast to every clip, but only 1 is relevant

The annotation JSON (`assistive_annotations.json`) already stores per-movement device/assistance data, but `ContextVectorEncoder.encode()` ignores which movement the clip belongs to.

## Design

### New Signature

```python
# OLD: per-patient (same vector for all clips)
context_encoder.encode(patient_id)

# NEW: per-clip (movement-specific device/assistance)
context_encoder.encode(patient_id, movement_type)
```

### 18D Vector Layout

#### Patient-Invariant Zone (11D) — "Who is this patient?"

Same for all clips from a patient. Captures the capability profile.

```
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
```

#### Clip-Specific Zone (7D) — "What's happening in THIS clip?"

Varies per clip based on which movement is being performed. Read from `per_movement[movement_key]` in annotations.

```
[11] clip_walker           # 0/1 — walker used in THIS movement
[12] clip_walker_type      # 0/0.5/1.0 (none/anterior/posterior)
[13] clip_afo              # 0/1 — AFO worn during THIS movement
[14] clip_afo_laterality   # 0/0.5/1.0 (none/unilateral/bilateral)
[15] clip_acrylic_stand    # 0/1 — acrylic stand used
[16] clip_surface          # 0=floor, 0.5=n/a, 1.0=chair
[17] clip_assistance       # 0.0-1.0 independence for THIS movement
```

### Clip-Specific Zone Encoding Rules

Given a `per_movement` entry with `device` field:

| `device` value | `[11]` walker | `[13]` afo | `[15]` acrylic |
|---|---|---|---|
| `"none"` | 0 | 0 | 0 |
| `"walker"` | 1 | 0 | 0 |
| `"afo_only"` | 0 | 1 | 0 |
| `"acrylic_stand"` | 0 | 0 | 1 |
| `"walker_and_afo"` | 1 | 1 | 0 |

For `[12] clip_walker_type`: read from patient-level `devices.walker_type`, but only when `clip_walker=1`. Otherwise 0.

For `[14] clip_afo_laterality`: read from patient-level `devices.afo_laterality`, but only when `clip_afo=1`. Otherwise 0.

For `[16] clip_surface`:
- `movement_type in ("c_s", "s_c")` → `0.0` (floor)
- `movement_type in ("cc_s", "s_cc")` → `1.0` (chair)
- all other movements → `0.5` (not applicable)

For `[17] clip_assistance`: read from `per_movement[movement_key].assistance`.

### Movement Type to Annotation Key Mapping

| Movement Code | Annotation Key |
|---|---|
| `w` | `walk` |
| `cr` | `crawl` |
| `c_s` | `seated_to_standing` |
| `s_c` | `standing_to_seated` |
| `sr` | `side_rolling` |
| `cc_s` | `chair_seated_to_standing` |
| `s_cc` | `standing_to_chair_seated` |

### Example: Patient `kku` (L3, walker user)

| Clip | `[11]` walk | `[12]` type | `[13]` afo | `[15]` acryl | `[16]` surf | `[17]` assist |
|---|---|---|---|---|---|---|
| `kku_w_01` (walk) | **1** | 0.5 | 0 | 0 | 0.5 | 1.0 |
| `kku_cr_02` (crawl) | **0** | 0 | 0 | 0 | 0.5 | 1.0 |
| `kku_c_s_01` (floor c_s) | **0** | 0 | 0 | **1** | **0** | 0.8 |
| `kku_cc_s_01` (chair c_s) | **0** | 0 | 0 | 0 | **1** | 0.6 |

Previously all four clips received the identical 22D vector with `walker_used=1`.

## Dimension Change: 22D → 18D

| Metric | Old (22D) | New (18D) | Reason |
|---|---|---|---|
| Per-movement assistance values | 7 (all broadcast) | 1 (correct one) | Removed 6 irrelevant values |
| Device flags | 5 (patient-level) | 5 (clip-specific) | Same count, more precise |
| Surface type | 0 | 1 | New: floor/chair per clip |
| Movement statuses | 7 | 7 | Unchanged (capability profile) |
| Overall summary | 2 | 2 | Unchanged |
| **Total** | **22** | **18** | -4D less noise, +1D surface |

## Files Affected

| File | Change |
|---|---|
| `src/features/context_vector.py` | Restructure encoder: `encode(patient_id, movement_type)`, 18D output |
| `scripts/04_extract_features.py` | Pass `movement_type` to encoder |
| `src/model/classifier.py` | `context_vector_dim=18` |
| `src/model/dataset.py` | Fallback `np.zeros(18)` |
| `src/model/train.py` | Fallback `context_vector_dim` to 18 |
| `scripts/06_evaluate.py` | Fallback `context_vector_dim` to 18 |
| `configs/default.yaml` | `context_vector_dim: 18` |
| `CLAUDE.md` | Update 22D → 18D references |
| `data/metadata/assistive_annotations.json` | Add per-movement device details for chair movements |

## Why GMFCS Level Is NOT in the Vector

The GMFCS level is the **prediction target** (label). Including it as input would be data leakage — the model would just read the answer from its input. During real-world inference, the GMFCS level is unknown (that's what the model predicts). The context vector contains only information a clinician would know *before* the GMFCS assessment.

## Verification

1. `encode("kku", "w")` → walker=1, assist=1.0
2. `encode("kku", "cr")` → walker=0, assist=1.0
3. `encode("kku", "c_s")` → acrylic=1, surface=0, assist=0.8
4. Same patient, different clips → different vectors (clip-specific zone differs)
5. Same patient, same movement → same vector
6. Model forward pass with 18D context vector: no dimension errors
7. Re-extract all clip features with new encoder
8. 5-fold CV: compare accuracy with old per-patient vs new per-clip encoding
