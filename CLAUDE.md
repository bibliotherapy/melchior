# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Melchior** — GMFCS Level Classification AI for children with cerebral palsy (ages 6 and under). Classifies GMFCS Levels 1-5 from multi-view video using 3D skeletal modeling, assistive device inference, and caregiver interaction analysis.

- 24 patients, ~3,175 clips, 3 camera viewpoints (GoPro front / iPhone left / Galaxy right), 30fps
- Target: 80%+ classification accuracy

### Server / GPU Environment

| Spec | Value |
|---|---|
| GPUs | 2x NVIDIA Tesla V100-DGXS-32GB |
| VRAM per GPU | 32GB |
| CUDA | 12.4 |
| Driver | 550.144.03 |

**GPU usage policy for all code in this project:**

- **Training:** Use `torch.nn.DataParallel` (DP) or `torch.nn.parallel.DistributedDataParallel` (DDP) across both GPUs. Prefer DDP with `torchrun` for training scripts. Default world size = 2.
- **Inference / pose estimation:** Run on a single GPU (`cuda:0`). The second GPU can run parallel experiments or hyperparameter sweeps.
- **Device selection:** Always use `torch.cuda.is_available()` checks. Never hardcode device indices without a config fallback. Use `configs/default.yaml` to set `device: cuda` and `num_gpus: 2`.
- **DDP launch command:** `torchrun --nproc_per_node=2 scripts/05_train.py`
- **Mixed precision:** Use `torch.amp.autocast('cuda')` + `GradScaler` for faster training and lower VRAM usage on V100.
- **Batch size:** Scale per-GPU batch size so total batch = per_gpu_batch * num_gpus. Default per-GPU batch: 16 (total 32).
- **Model saving:** Always save on rank 0 only when using DDP. Use `model.module.state_dict()` to unwrap DDP wrapper.

## Key Documentation

| File | Purpose |
|---|---|
| `GMFCS_Classification_Comprehensive_Report_EN.md` | Full technical report: dataset, pipeline, movement analysis, quality descriptors |
| `Assistive_Device_Integration_Implementation_Plan.md` | 6-phase implementation plan with code structure and pseudocode |
| `docs/superpowers/specs/2026-03-31-assistive-device-integration-design.md` | Design spec: three-layer architecture for device/caregiver encoding |
| `docs/superpowers/specs/2026-04-02-sam2-guided-tracking-design.md` | Design spec: SAM2-guided person/walker tracking (supersedes auto height-ratio) |
| `gmfcs.md` | GMFCS-E&R clinical classification reference |
| `data/metadata/assistive_annotations.json` | Per-patient device/assistance annotations (partially filled, has TODOs) |
| `data/metadata/sam2_annotations.json` | Per-clip first-frame point annotations for SAM2 tracking |

## Architecture

### Three-Layer Feature Architecture

1. **Layer 1 — Enhanced Patient Skeleton:** 3D triangulated skeleton `(T, 17, 3)` + ~15 derived features per frame (wrist fixation index, arm swing amplitude, ankle ROM, CoM sway, etc.) that proxy for assistive device usage without object detection.

2. **Layer 2 — Caregiver Interaction Skeleton:** Multi-person pose estimation detects both patient and caregiver. Interaction features (~10/frame): contact proximity, velocity correlation, movement independence score. The independence score is THE key signal for L4 vs L5 distinction.

3. **Layer 3 — Assistive Context Vector:** 18D metadata vector extending the original 7D (sex, age, movement status) with walker type, AFO presence, assistance levels per movement.

### Classification Pipeline

```
SAM2 manual first-frame annotation → Mask propagation (child, caregiver, walker)
→ Multi-person 2D pose (MMPose RTMPose) → Mask-guided skeleton assignment
→ Camera calibration (Human Pose as Calibration Pattern)
→ 3D triangulation (OpenCV, dual skeleton) → Feature extraction (Layers 1-3 + walker-spatial)
→ Hierarchical 2-stage classification:
    Stage 1: Ambulatory vs Non-ambulatory (binary)
    Stage 2-A: L1 vs L2 vs L3 (walk quality, walker usage)
    Stage 2-B: L4 vs L5 (caregiver assistance, side rolling independence)
```

### Model: Lite ST-GCN + Multi-Stream Fusion

- Stream A: Lite ST-GCN (3 layers, 64ch, ~200K params) on raw skeleton → 128D
- Stream B: Temporal mean+std pooling on skeleton features → 30D
- Stream C: Temporal mean+std pooling on interaction features → 20D
- Stream D: Context vector passthrough → 18D
- Stream E: Walker-skeleton spatial features → 5D
- Fusion: concat (~201D) → MLP(64) → classification head

## Source Layout

```
src/tracking/          # SAM2 video object tracking for person/walker identification
src/pose/              # Multi-person 2D pose extraction + mask-guided person identification
src/calibration/       # Camera calibration via human pose correspondence
src/triangulation/     # 3D triangulation for patient and caregiver
src/features/          # Layer 1 skeleton features, Layer 2 interaction features, Layer 3 context vector, walker spatial features
src/model/             # Lite ST-GCN, multi-stream classifier, hierarchical training
src/utils/             # Visualization, evaluation
scripts/annotate_first_frame.py  # Manual first-frame annotation tool for SAM2
scripts/00-06_*.py     # Batch processing entry points (sequential pipeline)
configs/default.yaml   # All hyperparameters and paths
```

## Data Conventions

- **Patient-level splits only** — all clips from one patient in the same fold. No cross-patient leakage.
- **Triplet = 3 synchronized viewpoints** of one movement: `{patient}_{movement}_{num}_{FV|LV|RV}.mp4`
- **5 movements:** walk (w), crawl (cr), seated_to_standing (c_s), standing_to_seated (s_c), side_rolling (sr)
- **Excluded:** static seated, run, jump (cause shortcut learning)
- **data/ is gitignored** — raw videos and processed outputs stay local

## GMFCS Level Discrimination Keys

- **Stage 1 (ambulatory vs non-ambulatory):** Routed by actual walking ability (w_status), not GMFCS level. L3/L4 patients can appear in either branch.
- **Stage 2-A (ambulatory branch):**
  - **L1 vs L2:** Gait speed, symmetry, balance (subtle, same movement pattern)
  - **L2 vs L3-L4:** Walker usage — walker-skeleton spatial features (hand-to-walker distance, engagement ratio) + wrist fixation index + arm swing amplitude
- **Stage 2-B (non-ambulatory branch):**
  - **L3-L4 vs L5:** Movement independence score during side rolling (self-initiated vs caregiver-driven)
- **L3 vs L4 (within merged class):** Device-assisted (walker engagement → L3) vs human-assisted (caregiver support_source_ratio → L4) mobility

## Auto-Backup

`.claude/auto-backup.sh` automatically commits and pushes documentation changes on file edits. Excludes `data/` and `.claude/` directories.

## Plans

All implementation plans must be saved as standalone `.md` files in the project root directory, not only in Claude's internal `.claude/plans/` folder.
