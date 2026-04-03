# Floor/Chair Transition Sub-Type Design

**Date:** 2026-04-03
**Status:** Approved
**Scope:** Movement type system expansion for GMFCS classification

## Problem

The GMFCS classification pipeline processes only 5 movement types, dropping 362 chair-based transition clips (`chair_seated_to_standing`: 221, `standing_to_chair_seated`: 141) — 22% of usable data. The raw data already labels these separately but the pipeline never ingests them.

### Why This Matters

- **Clinical**: GMFCS-E&R ages 4-6 (54.2% of patients) explicitly describes different floor vs chair transfer capabilities per level
- **Biomechanical**: Floor-to-stand and chair-to-stand have different starting heights, ROM, and muscle activation patterns
- **Discriminative**: Chair transfer independence is key for L2/L3/L4 distinction — the hardest levels to separate

## Design

### Movement Type System (5 → 7)

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

### Clip ID Convention

Follows existing pattern `{patient}_{movement}_{num}_{view}`:
- `hdi_cc_s_01_FV` — patient hdi, chair sit-to-stand, clip 01, front view
- `hdi_s_cc_01_FV` — patient hdi, chair stand-to-sit, clip 01, front view

### Feature Extraction: Surface-Parameterized

Extend existing `SitToStandQualityFeatures` and `StandToSitQualityFeatures` with `surface="floor"|"chair"`:

| Feature | Floor Behavior | Chair Adaptation |
|---|---|---|
| S1: Transition duration | Normalize by 5.0s | Normalize by 3.0s (shorter ROM) |
| S3: Hand-to-ground contact | 5th-pct wrist height = "ground" | Initial pelvis height = chair surface |
| D4: Hand support frequency | 5th-pct wrist height | Final pelvis height = target surface |
| S2, S4, S5, S6, D1-D3 | Unchanged | Unchanged (surface-agnostic) |

Dispatcher resolves chair codes to base class + surface flag:
- `cc_s` → `(c_s, surface="chair")`
- `s_cc` → `(s_c, surface="chair")`

### Context Vector: 18D → 22D

```
Existing [0-17] unchanged
[18] cc_s_status             # 0=cannot, 1=performed, 2=not_needed
[19] s_cc_status             # same encoding
[20] caregiver_assist_cc_s   # 0.0-1.0 independence scale
[21] caregiver_assist_s_cc   # 0.0-1.0
```

### Model Dimension Cascade

- Stream D (context passthrough): 18 → 22
- Fusion concat: ~201D → ~205D
- Update `classifier.py` context_dim, `configs/default.yaml`

## Clinical Basis (GMFCS-E&R Ages 4-6)

| Level | Floor Transfer | Chair Transfer |
|---|---|---|
| L1 | Without hand support, without objects | Gets in/out of chair without hand support |
| L2 | Often requires stable surface to push/pull | Sits in chair, both hands free; needs surface to push |
| L3 | — | Regular chair, may need pelvic/trunk support, uses arms |
| L4 | — | Adaptive seating needed, adult assistance required |
| L5 | Cannot | Cannot |

The ability to perform floor-to-stand vs only chair-to-stand, and the quality of each, are distinct assessment criteria.

## Data Impact

| Metric | Before | After |
|---|---:|---:|
| Movement types | 5 | 7 |
| Usable clips | ~1,647 | ~2,009 |
| Context vector dim | 18 | 22 |
| Fusion dimension | ~201 | ~205 |
