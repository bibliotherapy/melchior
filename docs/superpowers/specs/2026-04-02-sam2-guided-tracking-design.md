# SAM2-Guided Multi-Object Tracking Pipeline Design

**Date**: 2026-04-02
**Status**: Approved
**Supersedes**: Automatic height-ratio person identification (from 2026-03-31 spec, Section 4.2)

## Problem

The original pipeline assumed fully automatic person identification via a height-ratio heuristic (`min_height_ratio: 1.3`) and skeletal proxy features for walker/AFO detection. Analysis revealed critical failures:

1. **Height heuristic fails during physical assistance** (L4/L5) when bodies overlap and skeletons merge — exactly when correct person ID matters most for computing the Movement Independence Score.
2. **Skeletal proxy features (WFI, ASA) are confounded with CP itself**: spasticity and motor impairment produce identical wrist fixation and arm swing patterns as device usage.
3. **WFI cannot distinguish "gripping walker" from "held by caregiver"** — the exact distinction that defines the L3/L4 boundary.
4. **Prior experience** building a similar pipeline confirmed automatic child segmentation was unreliable even with state-of-the-art models.

## Decision

Replace automatic detection with **SAM2-guided tracking** for three targets:
- **Child** (all clips) — reliable skeleton assignment
- **Caregiver** (clips with caregiver present) — reliable interaction features
- **Walker** (clips where walker is visible) — direct hand-to-walker spatial features

Keep skeletal proxies + context vector for **AFO** (too small/hidden) and **acrylic stand** (transparent).

## Architecture

```
Video -> SAM2 (manual first-frame init: child + caregiver + walker)
      -> Per-frame masks (child_mask, caregiver_mask, walker_mask)
      -> RTMPose multi-person detection on full frame
      -> Match detected skeletons to person masks (IoU overlap)
      -> Identified: child_skeleton, caregiver_skeleton
      -> Walker mask -> walker-skeleton spatial features
      -> 3D triangulation (dual skeleton from 3 camera views)
      -> Feature extraction (Layers 1-3 + walker-spatial)
      -> Multi-stream fusion -> Hierarchical classification
```

### Annotation Workflow

1. Display first frame of each clip (OpenCV GUI)
2. User clicks on child -> SAM2 point prompt -> child mask
3. User clicks on caregiver (if present) -> caregiver mask
4. User clicks on walker (if present) -> walker mask
5. SAM2 propagates masks through all frames
6. Annotations saved as JSON; masks saved as compressed .npz

**Edge cases:**
- Caregiver enters mid-clip: annotate on first frame where they appear (SAM2 supports multi-frame prompting)
- Walker leaves frame: SAM2 handles object exit/re-entry
- Multiple caregivers: annotate primary caregiver only

**Estimated annotation effort:** ~12 hours total (one-time, also serves as data quality review)

### Mask-Guided Skeleton Assignment

Replaces height-ratio heuristic:

```python
def assign_skeletons_to_masks(detected_skeletons, child_mask, caregiver_mask):
    for skeleton in detected_skeletons:
        bbox = skeleton_to_bbox(skeleton)
        child_iou = compute_mask_bbox_iou(child_mask, bbox)
        caregiver_iou = compute_mask_bbox_iou(caregiver_mask, bbox)
        if child_iou > caregiver_iou:
            skeleton.identity = "child"
        else:
            skeleton.identity = "caregiver"
```

### New Walker-Skeleton Spatial Features

| Feature | Definition | L3/L4 Signal |
|---------|-----------|-------------|
| `hand_to_walker_dist` | Min distance from wrist keypoints to walker mask edge | L3: small (gripping), L4: large |
| `walker_engagement_ratio` | Fraction of frames with hand near walker | L3: ~0.8+, L4: ~0.1 |
| `walker_velocity` | Optical flow within walker mask region | L3: moves with child, L4: stationary |
| `walker_child_velocity_corr` | Correlation of walker and child CoM velocity | L3: high, L4: uncorrelated |
| `support_source_ratio` | hand-to-walker vs hand-to-caregiver distance | L3: closer to walker, L4: closer to caregiver |

### What Stays the Same

- **Layer 1 skeleton features** (WFI, ASA, AROM, SPC, CoM sway) — kept as supplementary
- **Layer 2 interaction features** (independence score, contact proximity, velocity correlation) — unchanged but now with correctly identified skeletons
- **Layer 3 context vector** (18D metadata) — already implemented, unchanged
- **AFO detection**: context vector flag + AROM proxy (not visually trackable)
- **Acrylic stand detection**: context vector flag + SPC feature (transparent)
- **Model architecture**: Lite ST-GCN + multi-stream fusion (Stream B extended with walker features)
- **Hierarchical 2-stage classification**: unchanged
- **Camera calibration + 3D triangulation**: unchanged

## New Source Layout

```
src/tracking/              # SAM2 video object tracking
  sam2_tracker.py          # SAM2VideoTracker class
scripts/annotate_first_frame.py  # Manual first-frame annotation tool
scripts/00_propagate_masks.py    # Batch SAM2 mask propagation
src/features/walker_features.py  # Walker-skeleton spatial features
```

## Configuration Additions

```yaml
tracking:
  model: sam2_hiera_large
  checkpoint: checkpoints/sam2_hiera_large.pt
  device: cuda:0
  mask_output_dir: data/processed/masks/

features:
  walker_proximity_threshold_px: 30  # pixels, hand-to-walker "contact" threshold
```

## Dependencies

- `segment-anything-2` (SAM2) — Meta's video segmentation model
- SAM2 model checkpoint (`sam2_hiera_large.pt`)
