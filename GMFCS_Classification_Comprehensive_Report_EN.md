# GMFCS Level Classification AI Model — Comprehensive Report

**Date**: 2026-03-20 (updated)
**Objective**: Achieve 80%+ inference accuracy for video-based GMFCS Level Classification of children with cerebral palsy aged 6 and under

---

## 1. Executive Summary

### Project Overview

This project aims to build an AI-based system that automatically classifies GMFCS Levels (1–5) using multi-view videos of 24 children with cerebral palsy (CP), aged 6 and under, filmed at the Samsung Seoul Hospital Pediatric Physical Therapy Room. The model currently training on the server (V100 GPU × 2) has achieved an Overall Accuracy of 69.28% at Epoch 6/20, which falls short of the 80% target.

### Key Conclusions

1. **Discard the existing model architecture and redesign from scratch.** The current segmented video-based single-view model cannot overcome data limitations.
2. **Integrate 3 viewpoints (front/left/right) videos as a single data unit**, perform 3D Skeleton Triangulation, and use this as model input. Multi-view triangulation has been reported to reduce recognition error by over 50% [2].
3. **Adopt a 2-stage Hierarchical Classification** strategy: primary classification by ambulatory status → secondary detailed Level classification. This aligns with the GMFCS-ER clinical assessment framework [10].
4. **Compose the dataset with 4 key movements** (seated_to_standing, crawl, walk, standing_to_seated), with side_rolling used as a supplementary movement for non-ambulatory group differentiation.
5. **Exclude static seated (sitting still)** — confirmed by clinical feedback from pediatric rehabilitation specialists that Level differentiation is impossible from static sitting videos alone.

---

## 2. Raw Video → Data Sample Editing Guide

### 2.1 Mandatory Pre-Editing Step: Time Synchronization

Raw videos were recorded simultaneously with 3 cameras per patient (front GoPro / left iPhone / right Galaxy), but recording start times differ. **You must use Final Cut Pro's "Clip Synchronization" feature (audio waveform-based) to time-align the 3 videos before making movement-specific cuts.** For visual-based synchronization, the latest research VisualSync [7] achieved synchronization error below 50ms.

```
[Required Workflow]
Raw_FV.MP4 ─┐
Raw_LV.MOV ─┼→ Final Cut Pro audio sync → Simultaneous 3-view cuts at identical timepoints → Save triplet
Raw_RV.mp4 ─┘
```

### 2.2 Frame Rate Unification

| Camera | Original FPS | Conversion |
|--------|-------------|------------|
| GoPro (front) | ~60fps | → **30fps** |
| iPhone (left) | ~60fps | → **30fps** |
| Galaxy (right) | 30fps | Keep as-is |

Since the Galaxy records at 30fps, downsample the others to 30fps. Setting the project to 30fps in Final Cut Pro applies this automatically.

### 2.3 Key Movements and Clip Conditions

| Priority | Movement | Clip Length | Cutting Criteria | Notes |
|---------|----------|-------------|------------------|-------|
| **1st** | **seated_to_standing** | 3–6 sec | From seated position start rising ~ 2 sec after fully standing | 5/5 Levels, 19 patients, transition quality differs dramatically by Level |
| **2nd** | **crawl** | 5–8 sec | Start crawling ~ 3+ repetitions | 5/5 Levels, 15 patients, key for L3-L5 differentiation |
| **3rd** | **walk** | 5–8 sec | Start walking ~ include direction change, one-way or round-trip | L1-L4, 16 patients, L5 inability itself is informative |
| **4th** | **standing_to_seated** | 3–6 sec | Entire process from standing to sitting | 5/5 Levels, 15 patients, transition movement |
| Supplementary | **side_rolling** | 5–8 sec | 2–3 side rolls | L4-L5 only, supplementary for non-ambulatory group differentiation |
| ~~Excluded~~ | ~~seated (static sitting)~~ | — | — | Even specialists cannot differentiate Levels |
| ~~Excluded~~ | ~~run, jump~~ | — | — | L1-L2 exclusive, causes shortcut learning |

### 2.4 Single Clip Principles

- **One clip = one movement type**: Walking with a direction change can be in the same clip (walk is walk). Walking then sitting must be separate clips (walk ≠ standing_to_seated).
- **One clip = 3 simultaneous viewpoint cuts**: FV/LV/RV must be cut at the same time interval.
- **Frame count consistency is mandatory**: Duration difference between the 3 clips must be within ±0.1 seconds.

### 2.5 Occlusion Handling Criteria

| Situation | Decision | Reason |
|-----------|----------|--------|
| Temporary occlusion (1–2 sec) in 1 viewpoint | **Include** | 3D reconstruction possible from remaining 2 viewpoints |
| Prolonged occlusion (over half) in 1 viewpoint | **Include but mark** | Automatic processing adjusts weights |
| Simultaneous occlusion in 2 viewpoints | **Exclude that segment** | 3D impossible with only 1 viewpoint |
| All 3 viewpoints occluded | **Exclude that segment** | No information available |

### 2.6 Filename and Directory Structure

```
CP_dataset/
├── raw_synced/
│   ├── ajy/
│   │   ├── ajy_seated_to_standing_01_FV.mp4
│   │   ├── ajy_seated_to_standing_01_LV.mp4
│   │   ├── ajy_seated_to_standing_01_RV.mp4    ← These 3 form one triplet
│   │   ├── ajy_crawl_01_FV.mp4
│   │   ├── ajy_crawl_01_LV.mp4
│   │   ├── ajy_crawl_01_RV.mp4
│   │   └── ...
│   └── ...
├── skeleton_2d/          # 2D pose estimation results
├── calibration/          # Per-patient camera parameters
├── skeleton_3d/          # 3D triangulation results (.npy)
└── metadata/
    ├── labels.json       # GMFCS labels
    └── triplets.json     # Triplet mapping info
```

**Naming convention**: `{patientID}_{movement}_{number}_{viewpoint}.mp4`
- `{patientID}_{movement}_{number}` = triplet identifier (shared across 3 viewpoints)
- `_{viewpoint}` = FV / LV / RV

---

## 3. Dataset Status

### 3.1 Patient Information (24 patients)

| Item | Value |
|------|-------|
| Total patients | 24 |
| Age | Mean 47.2 months (23–72 months) |
| Sex | Male 14, Female 10 |
| CP type | Spastic diplegia 13 (54.2%), Hemiplegia 5, Quadriplegia 3, Dyskinetic 2, Other 1 |
| Physician/caregiver GMFCS agreement rate | 100% (all 24 patients in agreement) |

### 3.2 GMFCS Level Distribution

| GMFCS | Patients | Clips | Proportion | Mean clips per patient |
|-------|----------|-------|------------|----------------------|
| Level 1 | 6 | 1,099 | 34.6% | 183.2 |
| Level 2 | 5 | 783 | 24.7% | 156.6 |
| Level 3 | 4 | 595 | 18.7% | 148.8 |
| Level 4 | 3 | 360 | 11.3% | 120.0 |
| Level 5 | 6 | 338 | 10.6% | 56.3 |
| **Total** | **24** | **3,175** | **100%** | — |

### 3.3 Class Imbalance Severity

- Level 1 (1,099) vs Level 5 (338): **3.25x** difference
- Level 4: Only **3 patients** (hdi, jrh, lsa)
- hdu (Level 5): Only **6 clips**
- kra (Level 1): **292 clips** → approximately **49x** difference compared to hdu

### 3.4 Complete Triplet (FV+LV+RV) Status

| Item | Count |
|------|-------|
| Complete triplets | 868 |
| Incomplete | 362 |
| Triplet coverage | 2,604 / 3,175 clips (82.0%) |

---

## 4. Recording Environment

### 4.1 Camera Placement (Per Recording Setup Document)

| Position | Equipment | Angle | Height | Distance |
|----------|-----------|-------|--------|----------|
| Front (FV) | GoPro | 0° | 90cm | 250cm |
| Left (LV) | iPhone (12 mini, etc.) | 45° | 90cm | 250cm |
| Right (RV) | Samsung Galaxy | 30° (changed from original 45°) | 90cm | 250cm |

**Note**: In practice, research assistants did not strictly adhere to these conditions, resulting in significant variation across patients.

### 4.2 Camera Specifications

| Camera | Resolution | FPS | Codec | Notes |
|--------|-----------|-----|-------|-------|
| GoPro (front) | 1920×1080 | ~60fps | H.264/H.265 | Wide-angle distortion (barrel distortion) correction needed |
| iPhone (left) | 1920×1080 | ~60fps | H.264 | Some vertical recordings (rotate=90) |
| Galaxy (right) | 1920×1080 or 1080×2320 | **30fps** | HEVC | Portrait mode videos → excluded from study |

### 4.3 Recording Location

Samsung Seoul Hospital Pediatric Physical Therapy Room (single fixed space). Green mat placed on the floor, rehabilitation equipment present in surroundings. The recording setup was based on Kim et al. [11]'s K-DST multi-camera pediatric developmental assessment protocol.

### 4.4 Calibration Data

**No calibration files available.** Neither checkerboard recordings nor camera intrinsic/extrinsic parameter files exist. However, using the "Human Pose as Calibration Pattern" technique [5], the child's joints visible in the video can serve as correspondence points for post-hoc calibration. Liu et al. [4] and Pätzold et al. [6] also demonstrated the effectiveness of automatic calibration using human keypoints.

---

## 5. Current Model Performance (Interim Results)

### 5.1 Training Environment

| Item | Specification |
|------|--------------|
| Server | Samsung Seoul Hospital |
| GPU | NVIDIA Tesla V100-DGXS-32GB × 2 |
| CUDA | 12.4 |
| Disk | 1.5TB available |
| Input data | SAM2 Segmented Video |
| Training status | Epoch 6/20, running in tmux main session |

### 5.2 Validation Results (Epoch 6/20, Val Loss: 1.3092)

| GMFCS Level | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Level I | 0.67 | 0.98 | 0.80 | 118 |
| Level II | 0.88 | 0.77 | 0.82 | 90 |
| Level III | 0.69 | 0.64 | 0.66 | 152 |
| Level IV | 1.00 | **0.20** | 0.33 | 30 |
| Level V | 0.52 | 0.48 | 0.50 | 82 |

| Overall Metric | Precision | Recall | F1-Score |
|----------------|-----------|--------|----------|
| Overall Accuracy | — | — | **69.28%** |
| Macro Avg | 0.75 | 0.61 | 0.62 |
| Weighted Avg | 0.71 | 0.69 | 0.68 |

### 5.3 Analysis of Current Model Issues

- **Level IV**: Recall 0.20 → only 6 out of 30 correctly classified, remaining 24 misclassified (severe under-detection)
- **Level V**: Overall poor performance (F1 0.50), movement patterns limited to seated/side_rolling
- **Level I**: Recall 0.98 but Precision 0.67 → tendency to misclassify other Levels as Level I
- **Level III**: Largest support at 152 but F1 0.66 → ambiguity of intermediate Levels

---

## 6. Movement Analysis

### 6.1 Movement-Level Coverage (Actual Data)

| Movement | L1 | L2 | L3 | L4 | L5 | Levels | Patients |
|----------|----|----|----|----|----|----|----------|
| **seated** | 5 | 5 | 4 | 3 | 4 | 5/5 | 20 |
| **seated_to_standing** | 5 | 5 | 3 | 2 | 3 | 5/5 | 19 |
| **standing_to_seated** | 3 | 3 | 3 | 2 | 1 | 5/5 | 15 |
| **crawl** | 1 | 2 | 2 | 3 | 2 | 5/5 | 15 |
| **walk** | 4 | 5 | 3 | 2 | 0 | 4/5 | 16 |
| side_rolling | 0 | 0 | 0 | 2 | 5 | 2/5 | 7 |
| run | 5 | 1 | 0 | 0 | 0 | 2/5 | 6 |
| jump | 4 | 2 | 0 | 0 | 0 | 2/5 | 6 |

### 6.2 Key Movement Triplet Counts

| Movement | L1 | L2 | L3 | L4 | L5 | Total |
|----------|----|----|----|----|------|-------|
| seated | 81 | 52 | 37 | 43 | 47 | 260 |
| seated_to_standing | 22 | 24 | 15 | 17 | 4 | 82 |
| crawl | 6 | 5 | 37 | 37 | 15 | 100 |
| walk | 83 | 60 | 31 | 12 | 0 | 186 |
| standing_to_seated | 13 | 17 | 9 | 14 | 2 | 55 |
| side_rolling | 0 | 0 | 0 | 4 | 37 | 41 |

### 6.3 Movement Exclusion Rationale and Decisions

**Excluded (Causes Shortcut Learning)**:

| Movement | Exclusion Reason |
|----------|-----------------|
| side_rolling | 95% are L5, causes "side_rolling=L5" shortcut learning (however, used as supplementary for non-ambulatory group) |
| run | 76% are L1, causes "running=mild" learning |
| jump | 83% are L1, causes "jumping=mild" learning |
| seated (static) | **Even pediatric rehabilitation specialists cannot differentiate Levels** — L1–L4 all sit stably |
| seated_to_chair_seated | 96% exclusive to L3 |
| prone_to_seated | 95% exclusive to L4, only 2 patients |
| Other rare movements | walk_to_chair, supine_to_prone, crawl_to_seated, roll_side — all exclusive to 1 patient |

### 6.4 Actual Performable Movements for Level 4, 5 Patients

**Level 4:**
- hdi: crawl(32), seated(27), walk(25), seated_to_standing(15), standing_to_seated(15), side_rolling(6)
- jrh: crawl(75), seated(42), seated_to_standing(39), standing_to_seated(30), walk(21)
- lsa: seated(15), crawl(9), side_rolling(6) — **cannot walk, cannot perform seated_to_standing**

**Level 5:**
- ajy: seated(50), side_rolling(6) — cannot walk/crawl
- hdu: side_rolling(6) — extremely limited data
- kcw: side_rolling(41) — only has side_rolling
- kri: crawl(51), seated(15), seated_to_standing(3) — **L5 patient capable of crawling**
- oms: seated(45), side_rolling(36), seated_to_standing(3)
- pjo: side_rolling(27), seated(23), crawl(12), seated_to_standing(6), standing_to_seated(6)

---

## 7. Hierarchical Classification Strategy Based on Ambulatory Status

### 7.1 Walking Ability Data Review

| Level | Patients | Can Walk | Cannot Walk |
|-------|----------|----------|-------------|
| L1 | 6 | **All 6** | 0 |
| L2 | 5 | **All 5** | 0 |
| L3 | 4 | ly, mkj, pjw (3) | **kku** (1) |
| L4 | 3 | hdi, jrh (2) | **lsa** (1) |
| L5 | 6 | 0 | **All 6** |

**Can Walk: 16 / Cannot Walk: 8**

### 7.2 Two-Stage Hierarchical Classification Structure

```
                       All Patients (24)
                            │
                  ━━━━━━━━━━┿━━━━━━━━━━
                  │                    │
            [Stage 1]            [Stage 1]
           Ambulatory           Non-ambulatory
          Can Walk Group        Cannot Walk Group
         (16, ~2,600 clips)    (8, ~560 clips)
                  │                    │
            ┌─────┼─────┐        ┌─────┼─────┐
            │     │     │        │           │
         [Stage 2-A]          [Stage 2-B]
         L1   L2   L3(L4)    L3(L4)     L5

         Differentiated by     Differentiated by
         walk + s2s quality    crawl + side_rolling quality
```

### 7.3 Movements Used Per Stage

| Stage | Classification Goal | Movements Used | Differentiation Points |
|-------|-------------------|----------------|----------------------|
| Stage 1 | Ambulatory vs Non-ambulatory | seated_to_standing, crawl | Transition quality, independent performance ability |
| Stage 2-A | L1 vs L2 vs L3-L4 | walk, seated_to_standing | Gait speed/symmetry, transition stability |
| Stage 2-B | L3-L4 vs L5 | crawl, side_rolling, seated_to_standing | Crawling ability, mode of locomotion |

### 7.4 Advantages of This Strategy

1. **Identical to clinical assessment process**: Physicians also assess in the order "Can they walk?" → "How well do they walk?" [10]
2. **Mitigates class imbalance**: Splits from 5-class at once → binary + 3-class in two steps
3. **Optimal movements per stage**: Only the most discriminative movements used at each stage
4. **80% feasibility**: With 90% at Stage 1 and 85% at Stage 2, overall ~76%; adding 3D Triangulation + multi-view can achieve 80%+

---

## 8. 3D Skeleton Triangulation Strategy

### 8.1 Why 3D?

The current model uses single-view 2D video as input. This results in depth ambiguity, making it impossible to determine the actual 3D position of joints. Triangulating 3 viewpoints enables:
- Recovery of actual 3D coordinates (x, y, z) of joints
- Resolution of single-view occlusion problems (supplemented by other views)
- Direct provision of view-invariant features
- Skarimva [2] reported 50%+ reduction in recognition error with multi-view triangulation. Dual-Camera CP Gait Analysis [3] also confirmed improved accuracy of CP gait analysis through 3D reconstruction

### 8.2 Automatic Calibration Pipeline

Since no calibration files exist, we use the "Human Pose as Calibration Pattern" technique proposed by Takahashi et al. [5]:

```
[Step 1] Time Synchronization
    → Final Cut Pro's audio-based automatic synchronization (visual-based synchronization such as VisualSync [7] also possible)

[Step 2] 2D Pose Estimation (per viewpoint)
    → Extract child's 2D joint coordinates using MediaPipe/OpenPose

[Step 3] Camera Intrinsic Estimation
    → Use known focal lengths for GoPro/iPhone/Galaxy models
    → GoPro wide-angle distortion correction

[Step 4] Camera Extrinsic Automatic Estimation
    → Use 2D joint coordinates from synchronized frames as correspondence points
    → Fundamental Matrix → Essential Matrix → Rotation/Translation recovery [4][6]
    → Independent estimation per patient (recording session), automatically resolving recording condition variations
    → Latest neural-based calibration such as SteerPose [12] can also be applied

[Step 5] 3D Triangulation
    → Recover 3D joint coordinates using OpenCV triangulatePoints
    → Output: (T, 17, 3) — frames × joints × xyz
```

### 8.3 Resolving Recording Condition Variations

Since research assistants did not strictly follow the recording setup (250cm, 90cm, angles), camera positions may vary significantly across patients. However, Human Pose as Calibration Pattern [5] estimates camera parameters from **intrinsic characteristics of the video itself**, without depending on the recording setup document values. Liu et al. [4] demonstrated that this approach outperforms traditional checkerboard calibration, and Pätzold et al. [6] demonstrated online calibration converging within minutes. Since estimation is performed independently per patient, **this is actually more accurate than using fixed values**.

---

## 9. Review of Multi-view Score Fusion

### 9.1 Originally Considered Method

```
P_final = (P_FV + P_LV + P_RV) / 3
GMFCS_predicted = argmax(P_final)
```

### 9.2 Limitations: When Individual View Performance Is Low, the Average Is Also Low

This method provides correction only when the 3 views make **different types of errors** [8]. In a situation like the current model where Level IV is barely predicted (Recall 0.20), FV, LV, and RV are all likely to err in the same direction. Score Fusion is a "finishing strategy" that extracts the last 2–3% from an already good model, not a method to rescue a fundamentally underperforming model.

→ Therefore, **an architecture that learns multi-view information jointly from the beginning within the model** (3D Triangulation → 3D skeleton input) is needed.

---

## 10. Data Sample Length and Composition Review

### 10.1 Four Strategies for Clip Length

| Strategy | Recommendation | Reason |
|----------|---------------|--------|
| Clip duplication (looping) to double length | **Not recommended** | No new information, temporal discontinuity, overfitting to spurious correlation |
| Temporal concatenation of 2 clips of same movement | **Conditionally recommended** | Only for same movement type; learns consistency/variability of repetitive patterns |
| Physical synthesis of 3+ clips | **Not recommended** | Clip-level Aggregation (embedding voting) is more effective |
| **3 viewpoints as one unit** | **Strongly recommended** | Fundamental increase in information through 3D Triangulation |

### 10.2 Current Clip Statistics

| Item | Value |
|------|-------|
| Overall mean length | 8.0 sec (median 7.3 sec) |
| Range | 0.9 sec – 20.1 sec |
| Walk mean | 5.4 sec |
| Crawl mean | 5.0 sec |
| Seated mean | 8.9 sec |
| Seated_to_standing mean | 6.3 sec |
| Side_rolling mean | 12.0 sec |

---

## 11. Prior Research (Zhao et al., IEEE TNSRE 2024)

### 11.1 Overview

Zhao et al. [1] performed GMFCS I–IV classification using STGCN + metric learning (triplet loss + consistency loss). MIT team, using the Kidzinski public dataset. The STGCN backbone was pre-trained on the NTU RGB+D [9] dataset.

### 11.2 Key Figures

| Item | Value |
|------|-------|
| Dataset | 861 patients, 1,450 videos |
| Target age | Mean 11 years (s.d. 5.9) |
| GMFCS range | Level I–IV (Level V excluded) |
| Pose Estimation | OpenPose |
| Encoder | STGCN (transfer learning from NTU RGB+D 120 [9]) |
| End-to-end accuracy | 76.6% |
| Metric learning + confidence 0.95 | **88%** |
| Cohen's Kappa | κlw = 0.733 |
| Train/Val/Test split | Patient-level 7:1:2 |

### 11.3 Comparison with This Study

| Item | Zhao et al. | This Study |
|------|-------------|------------|
| Patients | 861 | **24** |
| Target age | Mean 11 years | **6 and under** |
| GMFCS Level | I–IV | **I–V** |
| Camera views | Single | **3 (FV/LV/RV)** |
| Input | 2D skeleton | **3D skeleton (target)** |
| Movements | Walking/running | **Multiple movements** |

---

## 12. Server Environment

| Item | Specification |
|------|--------------|
| GPU | NVIDIA Tesla V100-DGXS-32GB × 2 (currently mostly idle) |
| CUDA | 12.4 |
| Driver | 550.144.03 |
| Total disk | 1.8TB |
| Available disk | 1.5TB (currently 158GB used, 10%) |
| Server access | http://14.63.89.203:28888/lab |

---

## 13. Summary of Key Issues and Resolution Strategies

| Issue | Cause | Resolution Strategy |
|-------|-------|-------------------|
| Absolute sample shortage (24 patients) | Extremely small compared to 861 in prior research [1] | Maximize information through 3D Triangulation [2][3], reduce per-stage difficulty via hierarchical classification |
| Class imbalance | L1(1,099) vs L5(338), L4(only 3 patients) | Two-stage classification: binary → sub-classification, class weight adjustment |
| Movement type confounding | "side_rolling=L5" shortcut | Use only 4 key movements, retain only movements spanning multiple Levels |
| No movement common to L1–5 | L5 cannot walk, L1 doesn't need side_rolling | Hierarchical classification: "Can they do it?" → "How well do they do it?" |
| Static seated lacks discriminative power | Even specialists cannot differentiate | Exclude static seated segments, focus on dynamic transition movements |
| Patient-level data leakage | Same patient's clips spanning train/test | Must use patient-level split |
| Single-view limitations | Depth ambiguity, occlusion | 3D Triangulation [2][3] + Human Pose Calibration [4][5][6] |

---

## 14. Per-Patient Movement Performance Status Based on Raw Videos (Measured Data)

Files from the raw video directory (`CP_videos_cut_original/data/raw/`) were exhaustively surveyed to calculate exact clip counts for 24 patients × 5 key movements. Exact string matching was applied to avoid confusion with similarly named movements such as `chair_seated_to_standing`.

### 14.1 L1 Patients (6) Movement Performance

| Movement | jyh | kdu | kra | kto | orj | phm | Performance Rate | Total Clips |
|----------|-----|-----|-----|-----|-----|-----|-----------------|------------|
| **walk (w)** | 23 | 80 | 85 | 98 | 14 | 29 | **6/6** | **329** |
| **seated_to_standing (c_s)** | 6 | 24 | 3 | 33 | 15 | 14 | **6/6** | **95** |
| **standing_to_seated (s_c)** | 2 | 22 | ✗ | 19 | 15 | 4 | 5/6 | 62 |
| crawl (cr) | 14 | ✗ | ✗ | 21 | 8 | 2 | 4/6 | 45 |
| side_rolling (sr) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 0/6 | 0 |

### 14.2 L2 Patients (5) Movement Performance

| Movement | hja | jeu | jji | jty | lrl | Performance Rate | Total Clips |
|----------|-----|-----|-----|-----|-----|-----------------|------------|
| **walk (w)** | 35 | 45 | 24 | 27 | 78 | **5/5** | **209** |
| **seated_to_standing (c_s)** | 21 | 8 | 21 | 6 | 35 | **5/5** | **91** |
| **standing_to_seated (s_c)** | 21 | 2 | 13 | ✗ | 33 | 4/5 | 69 |
| crawl (cr) | 3 | 14 | 2 | ✗ | ✗ | 3/5 | 19 |
| side_rolling (sr) | ✗ | ✗ | ✗ | ✗ | ✗ | 0/5 | 0 |

### 14.3 L3 Patients (4) Movement Performance

| Movement | kku | ly | mkj | pjw | Performance Rate | Total Clips |
|----------|-----|-----|-----|-----|-----------------|------------|
| **crawl (cr)** | 91 | 27 | ✗ | 2 | **3/4** | **120** |
| **walk (w)** | ✗ | 11 | 53 | 34 | **3/4** | **98** |
| **seated_to_standing (c_s)** | 13 | 15 | ✗ | 24 | **3/4** | **52** |
| **standing_to_seated (s_c)** | 7 | 8 | ✗ | 15 | 3/4 | 30 |
| side_rolling (sr) | ✗ | ✗ | ✗ | ✗ | 0/4 | 0 |

### 14.4 L4 Patients (3) Movement Performance

| Movement | hdi | jrh | lsa | Performance Rate | Total Clips |
|----------|-----|-----|-----|-----------------|------------|
| **crawl (cr)** | 32 | 75 | 9 | **3/3 (100%)** | **116** |
| **walk (w)** | 25 | 21 | ✗ | 2/3 | 46 |
| **seated_to_standing (c_s)** | 15 | 39 | ✗ | 2/3 | 54 |
| **standing_to_seated (s_c)** | 15 | 30 | ✗ | 2/3 | 45 |
| **side_rolling (sr)** | 6 | ✗ | 6 | 2/3 | 12 |

### 14.5 L5 Patients (6) Movement Performance

| Movement | ajy | hdu | kcw | kri | oms | pjo | Performance Rate | Total Clips |
|----------|-----|-----|-----|-----|-----|-----|-----------------|------------|
| **side_rolling (sr)** | 6 | 6 | 41 | ✗ | 36 | 27 | **5/6 (83%)** | **116** |
| crawl (cr) | ✗ | ✗ | ✗ | 51 | ✗ | 12 | 2/6 | 63 |
| seated_to_standing (c_s) | ✗ | ✗ | ✗ | 3 | 3 | 6 | 3/6 | 12 |
| standing_to_seated (s_c) | ✗ | ✗ | ✗ | ✗ | ✗ | 6 | 1/6 | 6 |
| walk (w) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 0/6 | 0 |

---

## 15. In-Depth Analysis of Discriminative Movements Between GMFCS Levels

### 15.1 L4 vs L5 Discrimination Analysis

L4 and L5 both belong to the Non-ambulatory group, and differentiation between these two most severe levels is needed.

**L4∩L5 Overlap by Movement:**

| Movement | L4 Patients | L5 Patients | Both? | Discriminative Power |
|----------|------------|------------|-------|---------------------|
| **crawl (cr)** | 3/3 (100%) | 2/6 (33%) | △ Partial | ⭕ **Crawling ability itself is the L4→L5 branching signal** |
| **side_rolling (sr)** | 2/3 (67%) | 5/6 (83%) | △ Partial | ⭕ Qualitative differences in rolling (symmetry, speed, spontaneity) |
| **seated_to_standing (c_s)** | 2/3 (67%) | 3/6 (50%) | △ Partial | ⭕ Qualitative differences in transition ability |
| standing_to_seated (s_c) | 2/3 (67%) | 1/6 (17%) | ❌ Minimal | — |
| walk (w) | 2/3 (67%) | 0/6 (0%) | ❌ L5 cannot | — |

**Discrimination strategy:** Crawl as 1st priority (all L4 capable vs most L5 incapable), side_rolling as 2nd priority (qualitative differences), seated_to_standing as 3rd priority (supplementary).

**Note:** Side_rolling alone is insufficient — L4's jrh has no side_rolling data, and L5's kri has no side_rolling data.

### 15.2 L1 vs L2 vs L3 Discrimination Analysis

L1, L2, and L3 belong to the Ambulatory group, differentiated by qualitative differences in gait.

**3-Level Overlap by Movement:**

| Movement | L1 Rate | L2 Rate | L3 Rate | All 3 Levels? | Discriminative Power |
|----------|---------|---------|---------|---------------|---------------------|
| **walk (w)** | 6/6 (100%) | 5/5 (100%) | 3/4 (75%) | ⭕ **Top priority** | L1 fast and stable → L2 slightly asymmetric → L3 slow, unstable, assistive device |
| **seated_to_standing (c_s)** | 6/6 (100%) | 5/5 (100%) | 3/4 (75%) | ⭕ **Top priority** | L1 immediate independent → L2 slight delay → L3 hand support/assistance needed |
| **standing_to_seated (s_c)** | 5/6 (83%) | 4/5 (80%) | 3/4 (75%) | △ Good | Differences in descent control (supplementary) |
| crawl (cr) | 4/6 (67%) | 3/5 (60%) | 3/4 (75%) | △ Partial | L1·L2 don't need it (they walk instead), L3 uses it as primary locomotion |
| side_rolling (sr) | 0/6 | 0/5 | 0/4 | ❌ Not applicable | — |

**Key finding:** L1 vs L2 have nearly identical movement performance patterns, so they can only be distinguished by **qualitative video differences (gait speed, symmetry, stability)**. Metadata alone cannot separate L1 and L2.

### 15.3 L3 vs L4 Discrimination Analysis — Key Area of Interest for Supervising Professor

In GMFCS, the L3-L4 boundary is the **branching point between "independent ambulation vs dependence on assisted mobility"** [10], which the supervising professor identified as the most critical distinction.

**L3∩L4 Overlap by Movement:**

| Movement | L3 Performance | L4 Performance | Both? | Discriminative Power | Total Clips |
|----------|---------------|---------------|-------|---------------------|-------------|
| **crawl (cr)** | 3/4 (75%) | **3/3 (100%)** | ⭕ | ⭕⭕⭕ **Highest** | 236 |
| **walk (w)** | 3/4 (75%) | 2/3 (67%) | ⭕ | ⭕⭕ High | 144 |
| **seated_to_standing (c_s)** | 3/4 (75%) | 2/3 (67%) | ⭕ | ⭕⭕ High | 106 |
| **standing_to_seated (s_c)** | 3/4 (75%) | 2/3 (67%) | △ | ⭕ Supplementary | 75 |
| side_rolling (sr) | 0/4 | 2/3 | ❌ No L3 data | — | — |

**Why crawl is the key movement for L3 vs L4 discrimination:**

1. **Highest coverage**: L3 3/4 (120 clips) + L4 3/3 (116 clips) = 236 clips, highest performance rate on both sides
2. **Stark qualitative differences**: L3 shows reciprocal crawl (alternating limb pattern, rhythmical) / L4 shows incomplete alternation, closer to belly crawling, asymmetric
3. **Clip duration differences**: L3 mean 3.4–5.6 sec vs L4 mean 4.5–8.5 sec → L4 significantly slower
4. **Includes lsa (L4)**: lsa, who cannot walk, performed 9 crawl clips → includes patients where walk-based comparison is impossible

**Difficulties in L3 vs L4 discrimination:**

- **lsa (L4, 23 months)**: Cannot perform walk, seated_to_standing, or standing_to_seated. Functional profile effectively close to L5
- **mkj (L3, 60 months)**: No crawl, seated_to_standing, or standing_to_seated data. Only performed chair-related movements and walk
- **hdi·jrh (L4)**: Can walk (hdi 25 clips, jrh 21 clips) → ambiguous boundary with L3

### 15.4 GMFCS-E&R Based Movement Quality Descriptors

Analysis to this point has focused on **"which movements can be performed"** (possible/impossible). However, for cases like L1 vs L2 or L3 vs L4 where the same movements are performed but Levels differ, **qualitative differences in movement execution** are the only classification cues. Based on the GMFCS-E&R [10] Age Band Descriptions and Distinctions Between Levels, we summarize qualitative differences per key movement for the 3 age bands corresponding to our target age range (23–72 months).

#### 15.4.1 Walk (Gait) Quality Differences — L1 vs L2 vs L3 vs L4

| Quality Characteristic | L1 | L2 | L3 | L4 |
|----------------------|-----|-----|-----|-----|
| **Gait independence** | Independent walking without assistive devices, including stairs | Indoor walking without assistive devices possible but limited on long distances/uneven surfaces | Hand-held mobility device (walker) required | Adult assistance + walker, short distances only |
| **Speed/rhythm** | Fast and consistent cadence, running/jumping ability emerging | Slow, irregular cadence, cannot run/jump | Markedly slow, rhythm dependent on assistive device | Extremely slow, prefers wheelchair over walking |
| **Symmetry** | Bilateral symmetric swing phase | Slight asymmetry (hemiplegic pattern possible) | Marked asymmetry, compensatory trunk lateral flexion | Severe asymmetry, bilateral functional impairment |
| **Balance** | Stable, maintained during direction changes | Balance loss on long distances/uneven surfaces | Cannot maintain balance without assistive device | Unstable weight bearing, extremely limited balance |
| **Stair walking** | Independent stair walking emerging (4–6 years) | Handrail required | Adult assistance required | Not possible |
| **Upper limb role** | Free (can carry objects) | Intermittent use for balance assistance | Used for grasping assistive device, no free upper limbs | Assistive device + adult assistance |

**Skeleton-based measurable characteristics:**
- **Cadence (stride cycle):** Calculated from periodic x-axis displacement of ankle joint → L1 > L2 > L3 > L4
- **Gait symmetry index:** Left/right ankle swing duration ratio → L1 ≈ 1.0, L2 < 1.0, L3 ≪ 1.0
- **Trunk lateral sway:** Amplitude of lateral oscillation of pelvis/spine → markedly increased in L3, L4
- **Step width:** Distance between bilateral ankles → wider base of support in more severe cases
- **Upper limb freedom:** Range of free motion of wrist coordinates → L1 is free, L3+ is fixed (grasping assistive device)
- **Head stability:** Vertical/lateral oscillation of head joint → increasingly unstable in more severe cases

#### 15.4.2 Seated-to-Standing (Sit-to-Stand Transition) Quality Differences — All Levels

| Quality Characteristic | L1 | L2 | L3 | L4 | L5 |
|----------------------|-----|-----|-----|-----|-----|
| **Transition duration** | Immediate (<1 sec) | Slight delay (1–2 sec) | Slow (2–4 sec), may pause midway | Very slow (>4 sec), adult assistance or support surface required | Cannot perform independently (very limited attempts by rare few) |
| **Hand use** | Rises without using hands | Occasionally places hands on floor or knees | Hand placement on stable surface (chair, floor) required | Adult assistance or fixed object required | — |
| **Trunk control** | Consistent upright transition | Slight forward lean then correction | Marked forward lean, compensatory trunk flexion | Insufficient trunk control, significant swaying | — |
| **Lower limb symmetry** | Bilateral simultaneous symmetric | Slight asymmetry | Asymmetric, one limb leading | Severe asymmetry, dependent on one limb | — |
| **Balance recovery** | Immediately stable after standing | Slight sway after standing | Unstable for several seconds after standing, grasps assistive device | Cannot maintain independence even after standing | — |

**Skeleton-based measurable characteristics:**
- **Transition duration:** Time for hip joint to reach standing height from seated height → L1 < L2 < L3 < L4
- **Trunk anterior tilt:** Maximum forward tilt of spine-pelvis angle → marked in L3, L4
- **Hand-to-ground contact:** Frequency of wrist joint reaching floor level → increases from L2
- **CoM (Center of Mass) trajectory smoothness:** Jerk (3rd derivative) of pelvis trajectory → L1 is smooth, L3+ irregular
- **Bilateral knee extension symmetry:** Time lag in left/right knee angle change → L1 simultaneous, L3+ asymmetric
- **Post-transition sway:** AP/ML oscillation of pelvis for 2 sec after reaching standing → greater in more severe cases

#### 15.4.3 Crawl (Crawling) Quality Differences — L3 vs L4 vs L5

| Quality Characteristic | L3 | L4 | L5 |
|----------------------|-----|-----|-----|
| **Reciprocal pattern** | ⭕ Alternating limb pattern (right hand + left knee → left hand + right knee) | △ Incomplete alternating pattern, tendency toward homolateral (same-side) limb use | ❌ Cannot alternate, belly crawling only or unable to perform |
| **Speed/rhythm** | Rhythmical with consistent speed | Irregular, frequent stops and restarts | Extremely slow, insufficient propulsion |
| **Trunk elevation** | Trunk sufficiently elevated from floor (hands-and-knees) | Trunk close to floor, frequent elbow crawling (commando crawl) | Cannot elevate trunk, only belly crawl possible |
| **Trunk rotation** | Minimal trunk rotation, stable | Excessive trunk rotation, body twisting | — |
| **Distance covered** | Used as primary locomotion, can cover considerable distance | Limited to short distances only | Minimal distance or unable to move |
| **Lower limb involvement** | Both lower limbs actively participate in alternation | Lower limbs drag, upper-limb-dominant locomotion | Lower limbs minimally involved |

**Skeleton-based measurable characteristics:**
- **Reciprocal pattern index:** Proportion of simultaneous contralateral hand-knee pair advancement → L3 ≈ 1.0, L4 < 0.7, L5 ≈ 0
- **Trunk elevation:** Mean pelvis-to-ground height → L3 > L4 ≫ L5 (belly crawl)
- **Crawl velocity:** Horizontal movement speed of pelvis → L3 > L4 > L5
- **Cycle regularity:** Coefficient of variation (CV) of hand/knee advancement cycle → L3 low (regular), L4 high (irregular)
- **Trunk roll amplitude:** Lateral roll angle of spine → excessive in L4, minimal in L3
- **Upper vs lower limb contribution ratio:** Upper limb displacement / lower limb displacement → higher upper limb ratio in L4 (upper-limb-dominant)

#### 15.4.4 Standing-to-Seated (Stand-to-Sit Transition) Quality Differences — L1 vs L2 vs L3 vs L4

| Quality Characteristic | L1 | L2 | L3 | L4 |
|----------------------|-----|-----|-----|-----|
| **Descent control** | Smooth and controlled descent | Slightly unstable but independent | Hand support or grasping assistive device required | Adult assistance required, sits down "as if falling" |
| **Descent speed** | Consistent speed | Slightly abrupt descent segments | Irregular, may pause midway | Uncontrolled, abrupt descent |
| **Upper limb use** | Not needed | Intermittent balance assistance | Essential (chair armrest, floor support) | Essential + adult assistance |
| **Landing impact** | Soft landing | Slight impact | Somewhat rough landing | Drops down, high impact |

**Skeleton-based measurable characteristics:**
- **Descent velocity profile:** Time profile of pelvis z-axis velocity → L1 is constant, L4 shows abrupt acceleration
- **Controlled deceleration:** Pelvis deceleration rate just before sitting → L1 high (controlled), L4 low (uncontrolled)
- **Impact jerk at landing:** Vertical jerk of pelvis at moment of sitting → L1 minimal, L4 maximal
- **Hand support detection:** Timing/frequency of wrist contacting support surface → increases from L2

#### 15.4.5 Side Rolling Quality Differences — L4 vs L5

| Quality Characteristic | L4 | L5 |
|----------------------|-----|-----|
| **Spontaneity** | Self-initiated | Initiated with adult assistance or self-initiated but very slow |
| **Symmetry** | Can roll in both directions | Only one direction possible or incomplete in both |
| **Speed** | Relatively fast and fluid | Very slow, pauses during rolling |
| **Trunk-limb dissociation** | Segmental rotation of trunk and limbs | Log-roll (trunk and limbs rotate as one unit) |
| **Repeatability** | Can roll 2–3 consecutive times | Fatigued after 1 roll or difficulty re-attempting |
| **Upper limb involvement** | Arms assist with propulsion, free arm movement | Arms trapped against body or minimal propulsive contribution |

**Skeleton-based measurable characteristics:**
- **Segmental rotation ratio:** Time lag between shoulder roll onset and hip roll onset → L4 has dissociated rotation (lag > 0.3 sec), L5 is log-roll (lag ≈ 0)
- **Rolling velocity:** Time for 1 complete roll → L4 < L5
- **Bilateral symmetry:** Left→right vs right→left rolling speed ratio → L4 ≈ 1.0, L5 biased
- **Inter-roll recovery time:** Rest time between consecutive rolls → L4 < L5
- **Arm range of motion during roll:** Shoulder abduction/flexion range → L4 > L5

### 15.5 Comprehensive Distinctions Between Levels by Age Band (GMFCS-E&R Distinctions)

Our patient cohort (23–72 months) spans 3 age bands of the GMFCS-E&R. Since observable functions differ for the same Level across age bands, we summarize key distinctions per age band to enable the model to utilize age information.

#### Age Band 1: Under 2 Years (Before 2nd Birthday) — This dataset: lsa (L4, 23 months)

| Distinction | L1 | L2 | L3 | L4 | L5 |
|-------------|-----|-----|-----|-----|-----|
| **Floor sitting** | Sits with both hands free | Uses both hands but may need hand support for balance | Maintains floor sitting only with trunk support | Head control present but trunk support needed for sitting | Cannot maintain antigravity postures |
| **Floor mobility** | Hands-and-knees crawling possible | Belly crawling or hands-and-knees | Rolling and belly crawling | Rolling only (supine→prone) | Cannot roll without adult assistance |
| **Standing/walking** | Pulls to stand using furniture → independent walking at 18–24 months | Pulls to stand, cruising attempts possible | — | — | — |

#### Age Band 2: 2–4 Years (Between 2nd and 4th Birthday) — This dataset: kdu(L1,26mo), phm(L1,31mo), kto(L1,34mo), orj(L1,35mo), jeu(L2,33mo), kku(L3,28mo), oms(L5,29mo)

| Distinction | L1 | L2 | L3 | L4 | L5 |
|-------------|-----|-----|-----|-----|-----|
| **Floor sitting** | Both hands free | Both hands free but difficulty with balance | Frequent "W-sitting", assumes sitting with adult assistance | Maintains sitting when placed but needs both hands for alignment/balance | Generally cannot maintain antigravity postures |
| **Locomotion method** | Walking is preferred locomotion | Walking with assistive device preferred | Belly crawling/hands-and-knees (non-reciprocal), walker for short indoor distances | Rolling, belly crawling, non-reciprocal hands-and-knees (short indoor distances) | No independent locomotion, transported |
| **Walking** | Independent walking without assistive device | Prefers assistive device | Walker + adult assistance (steering/turning) | — | — |

#### Age Band 3: 4–6 Years (Between 4th and 6th Birthday) — Majority of patients in this dataset

| Distinction | L1 | L2 | L3 | L4 | L5 |
|-------------|-----|-----|-----|-----|-----|
| **Chair sit-to-stand** | Rises without hand support | Pushes or pulls up from stable surface | Pushes or pulls up with arms (stable surface required) | Adult assistance or stable surface required | Cannot perform independently, adaptive equipment needed |
| **Indoor walking** | Independent walking | Walking without assistive device | Walking with hand-held assistive device | Walker + adult supervision, difficulty with turning/uneven surfaces | Cannot walk independently |
| **Outdoor walking** | Independent walking + stairs | Short distances only, wheeled mobility for long distances | Frequently transported | Transported, powered wheelchair possible | Transported, powered wheelchair with extensive adaptations partially possible |
| **Running/jumping** | Emerging | Not possible | — | — | — |
| **Stairs** | Independent | Handrail required | Adult assistance | Not possible | Not possible |

#### 15.5.1 Implications for Model Training

1. **The key to distinguishing L1 vs L2 is "running/jumping ability" and "stair independence."** However, since this dataset excludes run/jump, **continuous qualitative indicators such as gait speed, cadence regularity, bilateral symmetry, and balance maintenance during long-distance walking** must be used as substitutes.

2. **L2 vs L3 distinction is the most clear-cut:** L2 can walk without assistive devices after age 4, L3 requires assistive devices. In 3D skeleton data, **the pattern of upper limbs grasping an assistive device** (wrist coordinates fixed at hip height) is a strong signal for L3.

3. **Key to L3 vs L4 distinction:**
   - L3: "Rises independently from stable surface, walks indoors with walker"
   - L4: "Difficulty rising without adult assistance, walker + adult supervision, fails on turns/uneven surfaces"
   - → Skeleton features: Frequency of **upper limb reaching toward external support (person/object)** during seated_to_standing, **balance loss events during direction changes** in walk

4. **Importance of age information:** Even within L2, the motor patterns of a 33-month-old (jeu) and a 72-month-old (jty) differ. At 2–4 years, L1 is the period when "walking becomes the preferred locomotion," while at 4–6 years, L1 is the period of "running/jumping emergence." **The age_normalized in the metadata_vector reflects this difference, allowing the model to learn within-Level functional variation by age.**

5. **"Inability to perform" itself is classification information:** GMFCS-E&R specifies what each Level "cannot do" (L2: cannot run/jump / L5: no independent locomotion). The metadata_vector in this project (w_status, cr_status, etc.) directly encodes this information, and **the absence of specific movement clips itself contributes to Level inference.**

---

## 16. Comprehensive Key Movement Summary for Hierarchical Classification

### 16.1 Key Movement Assignments by Classification Stage

| Classification Stage | Classification Goal | Priority | Key Movement | Differentiation Point |
|---------------------|-------------------|----------|-------------|---------------------|
| **Stage 1** | Ambulatory vs Non-ambulatory | 1st | **walk (w)** | Walking ability itself provides primary classification |
| | | 2nd | **seated_to_standing (c_s)** | Supplementary through transition independence |
| **Stage 2-A** | **L1 vs L2 vs L3** | 1st | **walk (w)** | L1 fast and stable → L2 slightly asymmetric → L3 slow, unstable, assistive device |
| | | 2nd | **seated_to_standing (c_s)** | L1 immediate independent → L2 slight delay → L3 hand support/assistance needed |
| | | 3rd | **standing_to_seated (s_c)** | Differences in descent control |
| **L3 vs L4** | **L3 vs L4 (key)** | 1st | **crawl (cr)** | L3 reciprocal crawl → L4 incomplete, belly crawling, asymmetric |
| | | 2nd | **walk (w)** | L3 independent with assistive device → L4 needs others' assistance or cannot |
| | | 3rd | **seated_to_standing (c_s)** | L3 rises independently (hand support) → L4 needs others' assistance |
| **Stage 2-B** | **L4 vs L5** | 1st | **crawl (cr)** | All L4 capable vs most L5 incapable |
| | | 2nd | **side_rolling (sr)** | Qualitative differences in rolling symmetry, speed, spontaneity |
| | | 3rd | **seated_to_standing (c_s)** | Qualitative differences in transition ability |

### 16.2 Movement Role Summary

| Movement | Abbrev. | Classification Stages Involved | Overall Role |
|----------|---------|-------------------------------|-------------|
| **walk** | w | Stage 1 + Stage 2-A + L3vsL4 | **Most versatile** — Primary classification by walking ability, L1/L2/L3 differentiation by gait quality, L3/L4 independence differentiation |
| **crawl** | cr | L3vsL4 + Stage 2-B | **Core for L3–L5 differentiation** — Both qualitative differences (L3vsL4) and ability/inability (L4vsL5) |
| **seated_to_standing** | c_s | Stage 2-A + L3vsL4 + Stage 2-B | **Supplementary across all stages** — Provides supplementary discriminative power in every classification |
| **standing_to_seated** | s_c | Stage 2-A | Supplementary L1/L2/L3 differentiation |
| **side_rolling** | sr | Stage 2-B | Dedicated L4 vs L5 differentiation |

### 16.3 Video Editing Priority by Movement

| Priority | Movement | Target Patients | Editing Goal |
|---------|----------|----------------|-------------|
| ⭐⭐⭐ | **walk (w)** | L1–L4 ambulatory patients (16) | Edit triplets for all walk segments |
| ⭐⭐⭐ | **crawl (cr)** | L3–L5 capable patients | Edit triplets for all crawl segments |
| ⭐⭐ | **seated_to_standing (c_s)** | L1–L5 capable patients | Edit triplets for all transition segments |
| ⭐⭐ | **standing_to_seated (s_c)** | L1–L4 capable patients | Edit triplets where possible |
| ⭐ | **side_rolling (sr)** | L4–L5 (7 patients) | For non-ambulatory L4/L5 group differentiation |

**Note:** Crawl data from L1·L2 patients is not included in training. For mildly affected children, crawling is "too easy to need" and is not a valid target for movement quality comparison.

---

## 17. Movement Performance Checklist and Metadata Vector

### 17.1 Per-Patient 5 Key Movement Performance Checklist

Three states are assigned for each movement:
- ✅ **Performed**: Video clips of that movement exist
- 🔵 **Not Needed** (Too Easy): Movement is unnecessary due to mild severity (e.g., crawl/side_rolling for L1·L2)
- ❌ **Cannot Perform**: Movement cannot be performed due to severity

| Patient | GMFCS | Sex | Age | w | cr | c_s | s_c | sr |
|---------|-------|-----|-----|---|-----|-----|-----|-----|
| jyh | L1 | M | 58mo | ✅ Performed(23) | 🔵 Not Needed | ✅ Performed(6) | ✅ Performed(2) | 🔵 Not Needed |
| kdu | L1 | M | 26mo | ✅ Performed(80) | 🔵 Not Needed | ✅ Performed(24) | ✅ Performed(22) | 🔵 Not Needed |
| kra | L1 | F | 62mo | ✅ Performed(85) | 🔵 Not Needed | ✅ Performed(3) | ⚠️ No Data | 🔵 Not Needed |
| kto | L1 | M | 34mo | ✅ Performed(98) | 🔵 Not Needed | ✅ Performed(33) | ✅ Performed(19) | 🔵 Not Needed |
| orj | L1 | F | 35mo | ✅ Performed(14) | 🔵 Not Needed | ✅ Performed(15) | ✅ Performed(15) | 🔵 Not Needed |
| phm | L1 | M | 31mo | ✅ Performed(29) | 🔵 Not Needed | ✅ Performed(14) | ✅ Performed(4) | 🔵 Not Needed |
| hja | L2 | F | 50mo | ✅ Performed(35) | 🔵 Not Needed | ✅ Performed(21) | ✅ Performed(21) | 🔵 Not Needed |
| jeu | L2 | F | 33mo | ✅ Performed(45) | 🔵 Not Needed | ✅ Performed(8) | ✅ Performed(2) | 🔵 Not Needed |
| jji | L2 | F | 46mo | ✅ Performed(24) | 🔵 Not Needed | ✅ Performed(21) | ✅ Performed(13) | 🔵 Not Needed |
| jty | L2 | M | 72mo | ✅ Performed(27) | 🔵 Not Needed | ✅ Performed(6) | ⚠️ No Data | 🔵 Not Needed |
| lrl | L2 | F | 51mo | ✅ Performed(78) | 🔵 Not Needed | ✅ Performed(35) | ✅ Performed(33) | 🔵 Not Needed |
| kku | L3 | M | 28mo | ❌ Cannot Perform | ✅ Performed(91) | ✅ Performed(13) | ✅ Performed(7) | 🔵 Not Needed |
| ly | L3 | M | 38mo | ✅ Performed(11) | ✅ Performed(27) | ✅ Performed(15) | ✅ Performed(8) | 🔵 Not Needed |
| mkj | L3 | M | 60mo | ✅ Performed(53) | ⚠️ No Data | ⚠️ No Data | ⚠️ No Data | 🔵 Not Needed |
| pjw | L3 | M | 51mo | ✅ Performed(34) | ✅ Performed(2) | ✅ Performed(24) | ✅ Performed(15) | 🔵 Not Needed |
| hdi | L4 | F | 62mo | ✅ Performed(25) | ✅ Performed(32) | ✅ Performed(15) | ✅ Performed(15) | ✅ Performed(6) |
| jrh | L4 | M | 52mo | ✅ Performed(21) | ✅ Performed(75) | ✅ Performed(39) | ✅ Performed(30) | ⚠️ No Data |
| lsa | L4 | F | 23mo | ❌ Cannot Perform | ✅ Performed(9) | ❌ Cannot Perform | ❌ Cannot Perform | ✅ Performed(6) |
| ajy | L5 | F | 44mo | ❌ Cannot Perform | ❌ Cannot Perform | ❌ Cannot Perform | ❌ Cannot Perform | ✅ Performed(6) |
| hdu | L5 | M | 64mo | ❌ Cannot Perform | ❌ Cannot Perform | ❌ Cannot Perform | ❌ Cannot Perform | ✅ Performed(6) |
| kcw | L5 | F | 56mo | ❌ Cannot Perform | ❌ Cannot Perform | ❌ Cannot Perform | ❌ Cannot Perform | ✅ Performed(41) |
| kri | L5 | M | 62mo | ❌ Cannot Perform | ✅ Performed(51) | ✅ Performed(3) | ❌ Cannot Perform | ⚠️ No Data |
| oms | L5 | M | 29mo | ❌ Cannot Perform | ❌ Cannot Perform | ✅ Performed(3) | ❌ Cannot Perform | ✅ Performed(36) |
| pjo | L5 | M | 65mo | ❌ Cannot Perform | ✅ Performed(12) | ✅ Performed(6) | ✅ Performed(6) | ✅ Performed(27) |

### 17.2 Movement Performance Pattern Signatures by GMFCS Level

| GMFCS | w | cr | c_s | s_c | sr | Pattern Characteristics |
|-------|---|-----|-----|-----|-----|----------------------|
| **L1** | ✅ | 🔵 | ✅ | ✅ | 🔵 | Everything possible, crawl/rolling not needed |
| **L2** | ✅ | 🔵 | ✅ | ✅ | 🔵 | Identical pattern to L1 → **differentiated by video quality only** |
| **L3** | ✅/❌ | ✅ | ✅ | ✅ | 🔵 | Walking begins to fail, crawl becomes primary |
| **L4** | ✅/❌ | ✅ | ✅/❌ | ✅/❌ | ✅ | Overall functional decline, rolling needed |
| **L5** | ❌ | ❌/✅ few | ❌/✅ rare | ❌ | ✅ | Most movements impossible, only rolling possible |

**Key finding:** L1 and L2 have completely identical movement performance patterns, making differentiation impossible through metadata alone. In contrast, for L3 and below, "what can be done" itself provides powerful classification information.

### 17.3 Metadata Vector Design

To utilize the movement performance checklist as supplementary input information for the model, a per-patient metadata vector is defined. This vector is provided to the model alongside 3D skeleton data **at all stages: training, validation, and inference.**

```
metadata_vector = [
    sex,              # 0=female, 1=male
    age_normalized,   # age in months / 72 (0–1 normalization)
    w_status,         # 0=cannot perform, 1=performed, 2=not needed   (walk)
    cr_status,        # 0=cannot perform, 1=performed, 2=not needed   (crawl)
    c_s_status,       # 0=cannot perform, 1=performed, 2=not needed   (seated_to_standing)
    s_c_status,       # 0=cannot perform, 1=performed, 2=not needed   (standing_to_seated)
    sr_status,        # 0=cannot perform, 1=performed, 2=not needed   (side_rolling)
]
# → 7-dimensional vector, assigned identically per patient across all clips
```

**Encoding rules:**

| Status | Code | Meaning |
|--------|------|---------|
| 0 (Cannot Perform) | ❌ | Cannot perform the movement due to severity |
| 1 (Performed) | ✅ | Video clips of the movement exist and are used as training data |
| 2 (Not Needed) | 🔵 | Movement is unnecessary due to mild severity (can do it but clinically meaningless) |

**Application at inference stage:** Before recording a new patient, the caregiver/therapist fills out a simple checklist:
- "Can this child walk?" → w_status
- "Can they crawl (belly crawl/hands-and-knees)?" → cr_status
- "Can they rise from the floor independently?" → c_s_status
- "Can they sit down from standing independently?" → s_c_status
- "Can they do side rolling?" → sr_status

This approach requires no additional recording, making it **highly practical**, and integrates naturally with the clinical GMFCS pre-assessment procedure [10].

---

## 18. References

[1] P. Zhao, M. Alencastre-Miranda, Z. Shen, C. O'Neill, D. Whiteman, J. Gervas-Arruga, and H. I. Krebs, "Computer Vision for Gait Assessment in Cerebral Palsy: Metric Learning and Confidence Estimation," *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, vol. 32, pp. 2336–2345, 2024. DOI: 10.1109/TNSRE.2024.3416159
→ https://ieeexplore.ieee.org/document/10560023/

[2] Skarimva, "Skeleton-based Action Recognition is a Multi-view Application," *ICASSP 2025 / arXiv:2602.23231*, 2025.
→ https://arxiv.org/html/2602.23231v1

[3] "Enhancing Cerebral Palsy Gait Analysis with 3D Computer Vision: A Dual-Camera Approach," *10th International Conference on Control, Decision and Information Technologies (CoDIT)*, IEEE, 2024.
→ https://ieeexplore.ieee.org/document/10708137/

[4] K. Liu, L. Chen, L. Xie, J. Yin, S. Gan, Y. Yan, and E. Yin, "Auto calibration of multi-camera system for human pose estimation," *IET Computer Vision*, vol. 16, no. 8, pp. 660–673, 2022. DOI: 10.1049/cvi2.12130
→ https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cvi2.12130

[5] K. Takahashi, D. Mikami, M. Isogawa, and H. Kimata, "Human Pose as Calibration Pattern; 3D Human Pose Estimation with Multiple Unsynchronized and Uncalibrated Cameras," *IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2018.
→ https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w34/Takahashi_Human_Pose_As_CVPR_2018_paper.pdf

[6] M. Pätzold, S. Bultmann, and S. Behnke, "Online Marker-Free Extrinsic Camera Calibration Using Person Keypoint Detections," *DAGM German Conference on Pattern Recognition (GCPR)*, Lecture Notes in Computer Science, vol. 13485, Springer, 2022.
→ https://arxiv.org/abs/2209.07393

[7] S. Liu, D. Y. Yao, S. Gupta, and S. Wang, "VisualSync: Multi-Camera Synchronization via Cross-View Object Motion," *NeurIPS 2025 / arXiv:2512.02017*, 2025.
→ https://arxiv.org/abs/2512.02017

[8] R. Kavi, V. Kulathumani, F. Rohit, and V. Kecojevic, "Multiview fusion for activity recognition using deep neural networks," *Journal of Electronic Imaging*, vol. 25, no. 4, 043010, 2016. DOI: 10.1117/1.JEI.25.4.043010
→ https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-25/issue-4/043010/Multiview-fusion-for-activity-recognition-using-deep-neural-networks/10.1117/1.JEI.25.4.043010.short

[9] A. Shahroudy, J. Liu, T.-T. Ng, and G. Wang, "NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis," *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016.
→ https://github.com/shahroudy/NTURGB-D

[10] R. Palisano, P. Rosenbaum, D. Bartlett, and M. Livingston, "GMFCS – E&R: Gross Motor Function Classification System – Expanded and Revised," CanChild Centre for Childhood Disability Research, McMaster University, 2007.
→ https://canchild.ca/resources/42-gmfcs-e-r/

[11] "Multiview child motor development dataset for AI-driven assessment of child development," *GigaScience*, vol. 12, giad039, Oxford Academic, 2023. DOI: 10.1093/gigascience/giad039
→ https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giad039/7181060

[12] SteerPose: "Simultaneous Extrinsic Camera Calibration and Matching from Articulation," *arXiv:2506.01691*, 2025.
→ https://arxiv.org/abs/2506.01691

---

*Generated: 2026-03-20 | Updated: 2026-03-31 (Sections 15.4–15.5 added: GMFCS-E&R based movement quality descriptors, comprehensive distinctions between levels by age band, skeleton-based measurable characteristic mapping)*
