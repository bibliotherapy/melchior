# Plan: SAM2-Guided Multi-Object Tracking Pipeline

## Context

The current Melchior pipeline design assumes fully automatic person identification via a height-ratio heuristic (min_height_ratio: 1.3) and skeletal proxy features for walker/AFO detection. Through analysis, we identified critical problems:

1. **Automatic person ID fails at the critical boundary**: L4/L5 patients receive physical assistance from caregivers, causing body overlap and skeleton merging — exactly when correct person identification matters most
2. **Skeletal proxy features are confounded with CP itself**: WFI (wrist fixation) produces identical signatures for "gripping walker" and "being held by caregiver." CP-related spasticity mimics device-usage patterns
3. **The user's prior experience** building a similar pipeline confirmed that automatic child segmentation was unreliable even with good models
4. **With only 24 patients, every annotation must be as accurate as possible** — there's no room for systematic detection errors

**Decision**: Replace automatic detection with SAM2-guided tracking for **child, caregiver, and walker**. Keep skeletal proxies + context vector for AFO and acrylic stand (not visually trackable).

---

## Design Summary

### What SAM2 Handles (manual first-frame → tracking)
- **Child**: Marked in every clip → reliable skeleton assignment
- **Caregiver**: Marked in clips with caregiver present → reliable interaction features
- **Walker**: Marked in clips where walker is visible → direct hand-to-walker spatial features that resolve L3/L4 ambiguity

### What Stays Skeleton/Metadata-Based
- **AFO**: Context vector per-patient flag + AROM as supplementary signal (too small/hidden for SAM2)
- **Acrylic stand**: Context vector per-patient flag + SPC feature (transparent, not trackable)
- **All Layer 1 skeleton features**: Kept as supplementary (WFI, ASA, AROM, SPC, CoM sway, etc.)
- **All Layer 2 interaction features**: Movement independence score, contact proximity, velocity correlation
- **Layer 3 context vector**: 18D metadata — already implemented, unchanged
- **Model architecture**: Lite ST-GCN + multi-stream fusion (with ~5 new walker-skeleton features in Stream B)
- **Hierarchical 2-stage classification**: Unchanged
- **Camera calibration + 3D triangulation**: Unchanged

### New Walker-Skeleton Spatial Features
| Feature | Definition | L3/L4 Signal |
|---------|-----------|-------------|
| hand_to_walker_dist | Min distance from wrist keypoints to walker mask edge | L3: small (gripping), L4: large |
| walker_engagement_ratio | Fraction of frames with hand near walker | L3: ~0.8+, L4: ~0.1 |
| walker_velocity | Optical flow within walker mask | L3: moves with child, L4: stationary |
| walker_child_velocity_corr | Correlation of walker and child CoM velocity | L3: high, L4: uncorrelated |
| support_source_ratio | hand-to-walker vs hand-to-caregiver distance ratio | L3: closer to walker, L4: closer to caregiver |

### Annotation Effort
| Target | Clips | Est. Time |
|--------|-------|-----------|
| Child | ~3,175 | ~8.8 hours |
| Caregiver | ~1,300 (L3-L5) | ~2.9 hours |
| Walker | ~200 (L3 walking) | ~0.3 hours |
| **Total** | | **~12 hours (one-time)** |

---

## Revised Pipeline Architecture

```
Video → SAM2 (manual first-frame init: child + caregiver + walker)
      → Per-frame masks (child_mask, caregiver_mask, walker_mask)
      → RTMPose multi-person detection on full frame
      → Match detected skeletons to person masks (IoU overlap)
      → Identified: child_skeleton, caregiver_skeleton
      → Walker mask → walker-skeleton spatial features
      → 3D triangulation (dual skeleton from 3 camera views)
      → Feature extraction (Layer 1 skeleton + Layer 2 interaction + NEW walker-spatial + Layer 3 context)
      → Multi-stream fusion → Hierarchical classification
```

---

## Implementation Steps

### Step 1: Write Design Spec
- Save the full design spec to `docs/superpowers/specs/2026-04-02-sam2-guided-tracking-design.md`
- Also save standalone plan to project root per CLAUDE.md conventions
- Commit

### Step 2: SAM2 Integration Module (`src/tracking/sam2_tracker.py`)
- Install SAM2 (segment-anything-2) dependency
- Implement `SAM2VideoTracker` class:
  - `__init__(model_cfg, checkpoint_path, device)` — load SAM2 model
  - `initialize_from_points(video_path, frame_idx, object_points: dict)` — accept {object_name: (x, y)} click coordinates
  - `propagate(video_path) -> dict[str, np.ndarray]` — return per-object binary masks for all frames, shape (T, H, W)
  - `save_masks(output_dir)` — serialize masks to disk (compressed .npz per clip)
  - `load_masks(mask_dir) -> dict[str, np.ndarray]` — load pre-computed masks
- Config additions to `configs/default.yaml`:
  ```yaml
  tracking:
    model: sam2_hiera_large
    checkpoint: checkpoints/sam2_hiera_large.pt
    device: cuda:0
    mask_output_dir: data/processed/masks/
  ```
- **Critical files**: `src/tracking/__init__.py`, `src/tracking/sam2_tracker.py`

### Step 3: Annotation Tool (`scripts/annotate_first_frame.py`)
- Build a simple annotation script using OpenCV GUI:
  - Display first frame of each clip
  - User clicks on child → point prompt for SAM2
  - User clicks on caregiver (if present) → second point prompt
  - User clicks on walker (if present) → third point prompt
  - Save annotation as JSON: `{clip_id: {child: (x,y), caregiver: (x,y)|null, walker: (x,y)|null, frame_idx: 0}}`
  - Support resuming (skip already-annotated clips)
  - Support correcting (re-annotate specific clips)
- Output: `data/metadata/sam2_annotations.json`
- **Critical files**: `scripts/annotate_first_frame.py`

### Step 4: Batch SAM2 Propagation (`scripts/00_propagate_masks.py`)
- New script (runs before existing 01-06 pipeline):
  - Load annotations from `data/metadata/sam2_annotations.json`
  - For each clip: run SAM2 propagation → save masks to `data/processed/masks/{patient}_{movement}_{num}_{view}/`
  - Each mask dir contains: `child.npz`, `caregiver.npz` (optional), `walker.npz` (optional)
  - GPU: Single GPU (cuda:0), batch processing
- **Critical files**: `scripts/00_propagate_masks.py`

### Step 5: Modify Pose Extraction (`src/pose/multi_person_pose.py`)
- Implement multi-person 2D pose extraction using MMPose RTMPose
- Add mask-guided skeleton assignment:
  - Run RTMPose → get all detected person keypoints
  - For each detected skeleton, compute IoU of its bounding box with child_mask and caregiver_mask
  - Assign skeleton to the mask with highest IoU
  - If no skeleton matches a mask (detection failure), interpolate from adjacent frames
- Remove height-ratio heuristic from `src/pose/person_identifier.py` (replace with mask-based assignment)
- **Critical files**: `src/pose/multi_person_pose.py`, `src/pose/person_identifier.py`

### Step 6: Walker Spatial Features (`src/features/walker_features.py`)
- New module for walker-skeleton spatial features:
  - `compute_hand_to_walker_distance(wrist_keypoints, walker_mask)` → per-frame distance
  - `compute_walker_engagement_ratio(distances, threshold)` → fraction of engaged frames
  - `compute_walker_velocity(walker_masks)` → optical flow within mask region
  - `compute_walker_child_velocity_correlation(walker_vel, child_com_vel)` → correlation
  - `compute_support_source_ratio(wrist_kp, walker_mask, caregiver_kp)` → device vs human support
  - `extract_all_walker_features(clip_data) -> np.ndarray` → concatenated ~5D feature vector
- Returns zero vector when no walker is present (graceful degradation)
- **Critical files**: `src/features/__init__.py`, `src/features/walker_features.py`

### Step 7: Update Multi-Stream Classifier (`src/model/classifier.py`)
- Add walker-skeleton features to Stream B (extend from ~30D to ~35D with walker features)
- OR: Create Stream E for walker features (~5D) as separate input → update fusion layer dimension
- Decision: Add to Stream B (simpler, walker features are skeleton-derived)
- Update fusion input dimension: ~196D → ~201D (5 new walker features via mean+std pooling → 10D addition... but some features are already per-clip, not per-frame)
- **Critical files**: `src/model/classifier.py`

### Step 8: Update Pipeline Scripts
- Update `scripts/01_extract_2d_pose.py` to load masks and use mask-guided assignment
- Update `scripts/04_extract_features.py` to include walker feature extraction
- Update `configs/default.yaml` with tracking config section
- **Critical files**: `scripts/01_extract_2d_pose.py`, `scripts/04_extract_features.py`, `configs/default.yaml`

### Step 9: Update Source Layout in CLAUDE.md
- Add `src/tracking/` to source layout description
- Add `scripts/00_propagate_masks.py` to pipeline scripts
- Update architecture description to reflect SAM2-guided approach

---

## Verification

1. **Annotation tool test**: Run on 3-5 sample clips, verify click → mask → propagation works end-to-end
2. **Mask quality check**: Visualize propagated masks on sample clips (child, caregiver, walker) — check for drift, identity swaps
3. **Skeleton assignment test**: Compare mask-guided vs height-based assignment on clips where both child and caregiver are clearly visible — verify mask-guided is more consistent
4. **Walker feature sanity check**: Compute walker-skeleton features on known L3 (walker user) and L4 (no walker) clips — verify features discriminate correctly
5. **Integration test**: Run full pipeline (scripts 00-04) on a subset of patients — verify all outputs are produced without errors
6. **End-to-end**: Train model with and without walker-skeleton features — compare L3 vs L4 classification accuracy

---

## Dependencies
- `segment-anything-2` (SAM2) — Meta's video segmentation model
- SAM2 model checkpoint (`sam2_hiera_large.pt`)
- OpenCV GUI (`cv2.imshow`) for annotation tool
- Existing dependencies: MMPose, PyTorch, OpenCV
