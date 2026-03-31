# Assistive Device & Caregiver Assistance Integration into Skeletal Modeling Pipeline

**Date:** 2026-03-31
**Status:** Approved
**Project:** GMFCS Classification System for Cerebral Palsy (Melchior)

---

## 1. Problem Statement

The GMFCS classification system distinguishes levels based heavily on:
- Use of assistive devices (walkers, AFOs, support surfaces)
- Level of caregiver physical assistance
- Independence of movement

The planned 3D skeletal pipeline (MediaPipe/OpenPose -> 3D triangulation) captures only the patient's 17 body joints as `(T, 17, 3)`. This discards all information about walkers, ankle-foot orthoses, acrylic support stands, and caregiver assistance — making it impossible to capture the key criteria that differentiate GMFCS levels.

### What must be captured

| GMFCS Distinction | Clinical Criterion | Information Lost in Skeleton-Only |
|---|---|---|
| L1 vs L2 | L2 has balance limitations, may need railing | Subtle — partially visible in skeleton sway |
| L2 vs L3 | L3 requires hand-held mobility device (walker) | **Walker completely invisible** |
| L3 vs L4 | L4 needs adult assistance + device for short distances | **Caregiver assistance invisible** |
| L4 vs L5 | L5 requires extensive physical assistance for rolling | **Caregiver physically moving patient invisible** |

---

## 2. Solution: Dual Skeleton + Interaction Features + Context Encoding

### Architecture Overview: Three Layers

```
Layer 1: Enhanced Patient Skeleton
  - Standard 3D triangulation (T, 17, 3)
  - + Derived skeletal features that proxy for device usage (~15 features/frame)

Layer 2: Caregiver Interaction Skeleton
  - Multi-person pose estimation to detect caregiver
  - 3D triangulate caregiver separately
  - Compute interaction features (~10 features/frame)

Layer 3: Assistive Context Vector
  - One-time manual annotation of 24 patients
  - Extended metadata vector (7D -> ~18D)
```

### Key Design Decisions

1. **No object detection for devices** — transparent acrylic stands are invisible to detectors, and only 24 patients makes fine-tuning unreliable. Instead, device usage is inferred from skeletal patterns + explicit metadata.
2. **Dual skeleton for caregiver** — multi-person pose estimation captures the caregiver's body, enabling computation of interaction features (contact, force direction, independence score).
3. **Lightweight classification model** — with only 24 patients (~3,175 clips), a lite ST-GCN (2-3 layers, 64 channels) prevents overfitting. Handcrafted features carry the domain knowledge.
4. **Extended metadata vector** — one-time annotation of 24 patients adds device type and assistance level as explicit inputs.

---

## 3. Layer 1: Enhanced Patient Skeleton

### 3.1 Standard Pipeline (unchanged from report)

```
2D Pose Estimation (per viewpoint, multi-person capable)
    -> Camera Intrinsic Estimation
    -> Camera Extrinsic Estimation (Human Pose as Calibration Pattern)
    -> 3D Triangulation (OpenCV triangulatePoints)
    -> Output: (T, 17, 3) per clip
```

### 3.2 Derived Skeletal Features

Computed per frame from the 3D skeleton. These serve as proxy indicators for device usage without needing to detect the devices visually.

#### 3.2.1 Wrist Fixation Index (WFI)
**Detects:** Hands gripping walker or acrylic stand

```python
def wrist_fixation_index(wrist_positions, window_size=15):
    """
    Low variance in wrist position = gripping something.
    Computed over sliding window.
    Returns: 0.0 (free movement) to 1.0 (completely fixed)
    """
    std = rolling_std(wrist_positions, window_size)  # (T, 3)
    wfi = 1.0 - (norm(std) / max_expected_std)
    return clip(wfi, 0, 1)  # per frame scalar
```

- **Walker signature during walking:** Both wrists have high WFI at hip height
- **Acrylic stand signature during sit-to-stand:** Both wrists have high WFI at stand-top height, arms directed downward (not forward), while torso rises

#### 3.2.2 Arm Swing Amplitude (ASA)
**Detects:** Walker-assisted walking (low swing) vs free walking (high swing)

```python
def arm_swing_amplitude(wrist_positions, gait_cycles):
    """
    Amplitude of wrist oscillation in anteroposterior direction per gait cycle.
    L1: high swing. L3: near-zero swing (hands on walker).
    """
    for cycle in gait_cycles:
        amplitude = max(wrist_ap[cycle]) - min(wrist_ap[cycle])
    return mean(amplitudes)
```

#### 3.2.3 Ankle Range of Motion (AROM)
**Detects:** AFO (ankle-foot orthosis) presence

```python
def ankle_rom(knee_pos, ankle_pos, toe_pos):
    """
    AFOs restrict dorsiflexion/plantarflexion.
    Reduced AROM = AFO is constraining the joint.
    """
    ankle_angles = compute_angle(knee_pos, ankle_pos, toe_pos)  # per frame
    return max(ankle_angles) - min(ankle_angles)
```

#### 3.2.4 Upper Limb Freedom Score (ULFS)
**Detects:** Overall arm constraint level (free vs device-bound)

```python
def upper_limb_freedom(wrist_vel, shoulder_rom, symmetry):
    """
    Composite score: wrist velocity * shoulder ROM * bilateral symmetry.
    L1: high (free arms). L3: low (grasping walker).
    """
    return mean_velocity(wrist_vel) * shoulder_rom * symmetry
```

#### 3.2.5 Center of Mass (CoM) Trajectory Features
**Detects:** Balance and stability

```python
def com_features(hip, shoulder, head):
    """
    Proxy CoM from major joint positions.
    """
    com = 0.5 * hip + 0.3 * shoulder + 0.2 * head  # weighted average
    sway = lateral_variance(com)       # balance metric
    smoothness = jerk_metric(com)      # movement quality (lower = smoother)
    return sway, smoothness
```

#### 3.2.6 Support Point Convergence (SPC)
**Detects:** Acrylic stand usage during sit-to-stand

```python
def support_point_convergence(left_wrist, right_wrist, hip_height):
    """
    During sit-to-stand: if both wrists converge to a fixed point
    while torso rises, patient is using a support surface.
    
    Key: wrists at a fixed height, arms directed downward (not forward).
    """
    wrist_height_stability = std(mean(left_wrist.z, right_wrist.z))
    torso_rising = derivative(hip_height) > 0
    convergence = wrist_height_stability < threshold AND torso_rising
    return convergence
```

#### 3.2.7 Gait Symmetry Index (GSI)
**Detects:** Walking quality and asymmetry

```python
def gait_symmetry(step_lengths_left, step_lengths_right):
    """
    |left - right| / mean. Higher = more asymmetric = lower function.
    """
    return abs(mean(left) - mean(right)) / mean(left + right)
```

### 3.3 Output Format

```
patient_skeleton:   (T, 17, 3)   # raw 3D joint coordinates
skeleton_features:  (T, ~15)     # derived features per frame
```

Features per frame (~15 total):
- WFI_left, WFI_right (2)
- ASA_left, ASA_right (2)
- AROM_left, AROM_right (2)
- ULFS (1)
- CoM_sway, CoM_smoothness (2)
- SPC (1)
- GSI (1)
- Hand_height_left, Hand_height_right relative to hip (2)
- Bilateral_wrist_distance (1)
- Torso_vertical_velocity (1)

---

## 4. Layer 2: Caregiver Interaction Skeleton

### 4.1 Multi-Person Pose Estimation

**Current plan:** MediaPipe BlazePose (single-person)
**Required change:** Switch to multi-person-capable estimator

**Recommended: MMPose with RTMPose backbone**
- State-of-the-art accuracy with good inference speed
- Native multi-person support via top-down or bottom-up approaches
- Well-maintained library with extensive documentation
- Compatible with V100 GPUs and CUDA 12.4

**Alternative:** YOLOv8 for person detection -> crop each person -> single-person pose estimation per crop. This is simpler but may have lower accuracy for overlapping people.

### 4.2 Patient vs Caregiver Identification

Patients are children aged 2-6 years. Caregivers are adults. The height difference is substantial and reliable.

```python
def identify_patient_caregiver(detected_skeletons):
    """
    Identify patient (child) vs caregiver (adult) by skeleton height.
    """
    heights = [skeleton_bounding_box_height(s) for s in detected_skeletons]
    patient_idx = argmin(heights)  # smallest = child
    caregiver_idxs = [i for i in range(len(heights)) if i != patient_idx]
    return patient_idx, caregiver_idxs
```

**Fallback confirmation:**
- Patient is typically centered in the camera frame
- Temporal consistency: track person identity across frames via height + position
- In the rare case of two children, use position (patient is the one performing the movement)

### 4.3 Caregiver 3D Triangulation

Same pipeline as patient skeleton:
1. Detect caregiver in each of 3 camera views
2. Match caregiver identity across views (height + relative position to patient)
3. Triangulate caregiver joints to 3D
4. Output: `(T, 17, 3)` for caregiver — **zero-padded when no caregiver is present**

### 4.4 Interaction Features

Computed per frame from the paired 3D skeletons (patient + caregiver).

#### 4.4.1 Caregiver Presence
```python
caregiver_present = 1 if caregiver_skeleton_detected else 0  # binary per frame
```

#### 4.4.2 Minimum Hand-Body Distance
```python
def min_hand_body_distance(caregiver_wrists, patient_joints):
    """
    Minimum Euclidean distance between caregiver wrists and any patient joint.
    Low value = physical contact / close support.
    """
    distances = [euclidean(cw, pj) for cw in caregiver_wrists for pj in patient_joints]
    return min(distances)
```

#### 4.4.3 Contact Point Count
```python
def contact_point_count(caregiver_joints, patient_joints, threshold=0.15):
    """
    Number of patient joints where a caregiver joint is within contact threshold.
    More contact points = more extensive physical support.
    """
    count = 0
    for pj in patient_joints:
        for cj in caregiver_joints:
            if euclidean(cj, pj) < threshold:
                count += 1
                break
    return count
```

#### 4.4.4 Contact Duration Ratio
```python
contact_ratio = frames_with_any_contact / total_frames
sustained_contact = longest_continuous_contact_streak / total_frames
```

#### 4.4.5 Velocity Correlation (Critical for L4 vs L5)
```python
def velocity_correlation(caregiver_hand_vel, patient_torso_vel):
    """
    Correlation between caregiver hand movement and patient body movement.
    High correlation = caregiver is DRIVING the patient's movement.
    Low correlation = patient moving independently (caregiver just watching/guarding).
    """
    return pearson_correlation(caregiver_hand_vel, patient_torso_vel)
```

#### 4.4.6 Movement Independence Score (THE key L4/L5 feature)
```python
def movement_independence_score(patient_joint_velocities, caregiver_contact_forces):
    """
    Estimates what fraction of movement is self-generated vs externally applied.
    
    For side rolling:
      - L4: patient generates rotational force (score ~0.7-1.0)
      - L5: caregiver drives the rotation (score ~0.0-0.3)
    
    Approximation: ratio of patient velocity during non-contact vs contact phases.
    If patient moves at same speed regardless of contact -> independent.
    If patient only moves when contacted -> dependent.
    """
    vel_during_contact = mean(patient_vel[contact_frames])
    vel_during_no_contact = mean(patient_vel[no_contact_frames])
    
    if vel_during_contact + vel_during_no_contact == 0:
        return 0.0  # no movement at all
    
    # If patient moves equally during contact and non-contact -> independent
    # If patient only moves during contact -> dependent
    independence = vel_during_no_contact / max(vel_during_contact, vel_during_no_contact)
    return independence
```

#### 4.4.7 Contact Body Region
```python
def contact_body_region(caregiver_joints, patient_joints, threshold):
    """
    Encode which patient body regions are in contact with caregiver.
    Returns: [head_contact, trunk_contact, upper_limb_contact, lower_limb_contact]
    """
    regions = {
        'head': [0],           # head joint index
        'trunk': [1, 2, 5, 8], # spine, shoulders, hips
        'upper_limb': [3,4,6,7], # elbows, wrists
        'lower_limb': [9,10,11,12,13,14,15,16] # knees, ankles, feet
    }
    contact = []
    for region_name, joint_idxs in regions.items():
        in_contact = any(
            euclidean(cj, patient_joints[pj]) < threshold
            for pj in joint_idxs for cj in caregiver_joints
        )
        contact.append(float(in_contact))
    return contact  # [0/1, 0/1, 0/1, 0/1]
```

#### 4.4.8 Assistance Continuity
```python
assistance_continuity = longest_unbroken_contact_streak / total_frames
```

#### 4.4.9 Relative Position
```python
def relative_position(caregiver_hip, patient_hip):
    """
    Caregiver position relative to patient:
    - Behind (rolling assist)
    - Beside (walking support)
    - In front (facing, guiding)
    Encoded as angle in horizontal plane.
    """
    direction = caregiver_hip[:2] - patient_hip[:2]
    angle = atan2(direction[1], direction[0])
    return angle  # radians, relative to patient facing direction
```

### 4.5 Output Format

```
caregiver_present:       (T, 1)
interaction_features:    (T, ~10)
```

Features per frame (~10 total):
- caregiver_present (1)
- min_hand_body_distance (1)
- contact_point_count (1)
- contact_duration_ratio (1) — computed as running ratio
- velocity_correlation (1)
- movement_independence_score (1)
- contact_body_region (4) — head/trunk/upper/lower

Note: assistance_continuity and relative_position are clip-level summary features, added to the context vector or computed after temporal pooling.

---

## 5. Layer 3: Assistive Context Vector

### 5.1 Manual Annotation Schema

One-time annotation for each of the 24 patients. Per-movement granularity.

```json
{
    "patient_id": "kku",
    "gmfcs_level": 3,
    "devices": {
        "walker": true,
        "walker_type": "posterior",
        "afo": true,
        "afo_laterality": "bilateral",
        "acrylic_stand": true
    },
    "per_movement": {
        "walk": {"device": "walker", "assistance": "independent"},
        "seated_to_standing": {"device": "acrylic_stand", "assistance": "standby"},
        "standing_to_seated": {"device": "acrylic_stand", "assistance": "standby"},
        "crawl": {"device": "none", "assistance": "independent"},
        "side_rolling": {"device": "none", "assistance": "independent"}
    },
    "overall_assistance": "independent_with_device"
}
```

**Estimated annotation effort:** ~5-10 minutes per patient = ~2-4 hours total for 24 patients.

### 5.2 Extended Metadata Vector

Original 7D vector preserved. 11 new fields added.

```
Index  Field                    Type     Range          Source
-----  -----                    ----     -----          ------
0      sex                      binary   0/1            Original
1      age_normalized           float    0-1            Original
2      w_status                 ordinal  0/1/2          Original (cannot/performed/not_needed)
3      cr_status                ordinal  0/1/2          Original
4      c_s_status               ordinal  0/1/2          Original
5      s_c_status               ordinal  0/1/2          Original
6      sr_status                ordinal  0/1/2          Original
--- NEW FIELDS ---
7      walker_used              binary   0/1            Manual annotation
8      walker_type              ordinal  0/1/2          none/anterior/posterior
9      afo_present              binary   0/1            Manual annotation
10     afo_laterality           ordinal  0/1/2          none/unilateral/bilateral
11     support_surface_used     binary   0/1            Manual annotation (acrylic stand)
12     caregiver_assist_walk    float    0-1            Independence scale for walking
13     caregiver_assist_c_s     float    0-1            Independence scale for sit-to-stand
14     caregiver_assist_crawl   float    0-1            Independence scale for crawling
15     caregiver_assist_sr      float    0-1            Independence scale for side rolling
16     overall_assistance       ordinal  0-6            FIM-like scale
17     device_count             int      0-3            Total devices used
```

**Total: 18-dimensional context vector**

### 5.3 Independence Scale (for fields 12-15)

```
0.0 = Dependent (caregiver performs the movement)
0.2 = Maximal assist (patient contributes <25% effort)
0.4 = Moderate assist (patient contributes 25-50%)
0.6 = Minimal assist (patient contributes 50-75%)
0.8 = Contact guard / standby (caregiver present, minimal physical contact)
1.0 = Independent (no caregiver assistance)
```

### 5.4 FIM-like Overall Assistance Scale (field 16)

```
0 = Total dependence
1 = Maximal assistance
2 = Moderate assistance
3 = Minimal contact assistance
4 = Supervision/standby
5 = Modified independence (uses device but no human help)
6 = Complete independence
```

---

## 6. Classification Architecture

### 6.1 Data-Appropriate Design

**Constraint:** 24 patients, ~3,175 clips. Patient-level cross-validation means ~4-5 test patients per fold. Complex deep models will overfit.

**Principle:** Handcrafted features (Layers 1-3) carry the domain knowledge from GMFCS criteria. The neural network captures additional patterns from raw skeleton sequences.

### 6.2 Architecture

```
Input:
  patient_skeleton:   (T, 17, 3)     # 3D joint coordinates
  skeleton_features:  (T, ~15)       # derived device/movement features
  interaction_feats:  (T, ~10)       # caregiver interaction features
  context_vector:     (18,)          # extended metadata

Processing:
  Stream A: Lite ST-GCN (2-3 layers, 64 channels)
    Input: patient_skeleton (T, 17, 3)
    Output: temporal representation -> global pooling -> (128,)

  Stream B: Temporal Pooling (no learnable params)
    Input: skeleton_features (T, ~15)
    Output: mean + std pooling -> (30,)

  Stream C: Temporal Pooling (no learnable params)
    Input: interaction_feats (T, ~10)
    Output: mean + std pooling -> (20,)

  Stream D: Context Vector
    Input: (18,)
    Output: (18,) passed through

Fusion:
  Concatenate [A, B, C, D] -> (~196D)
  MLP: 196 -> 64 -> Dropout(0.3) -> C classes

Output per stage:
  Stage 1: C=2 (ambulatory vs non-ambulatory)
  Stage 2-A: C=3 (L1 vs L2 vs L3)
  Stage 2-B: C=2 (L4 vs L5)
```

### 6.3 ST-GCN Configuration

```python
stgcn_config = {
    'num_layers': 3,
    'channels': [3, 64, 64, 128],    # input_dim -> hidden -> hidden -> output
    'kernel_size': 9,                  # temporal convolution kernel
    'num_joints': 17,
    'graph': 'coco_17_joint',         # standard 17-joint graph topology
    'dropout': 0.3,
    'global_pool': 'mean',            # temporal global average pooling
}
```

**Estimated parameters:** ~200K (very lightweight for V100)

### 6.4 Per-Stage Feature Importance

| Stage | Primary Signals | Secondary Signals |
|---|---|---|
| **Stage 1** (Ambulatory?) | walker_used, overall_assistance, cr_status | WFI, contact_duration_ratio |
| **Stage 2-A** (L1/L2/L3) | WFI during walking, ASA, walker_used | CoM_sway, GSI, ULFS |
| **Stage 2-B** (L4/L5) | **movement_independence_score**, velocity_correlation | contact_body_region, caregiver_assist_sr, AROM |

### 6.5 How Each GMFCS Distinction Is Captured

| Distinction | GMFCS Clinical Criteria | Feature(s) That Capture It |
|---|---|---|
| L1 vs L2 | L2: balance limitations, may need railing for stairs | ASA (L2 lower), CoM_sway (L2 higher), ULFS (L2 occasionally reaching) |
| L2 vs L3 | L3: needs hand-held mobility device indoors | WFI (L3 high=gripping walker), walker_used=1, ASA (L3 near zero) |
| L3 vs L4 | L4: needs adult assistance + device for short distances only | caregiver_present, contact_duration during walking, caregiver_assist_walk |
| L4 vs L5 | L5: extensive physical assist, cannot self-roll | **movement_independence_score** (L4 ~0.7-1.0, L5 ~0.0-0.3), velocity_correlation |

### 6.6 Resource Estimates (on 2x V100-DGXS-32GB server)

| Component | VRAM Usage | Time Estimate |
|---|---|---|
| MMPose RTMPose multi-person inference (all clips) | ~4GB on 1 GPU | ~2-4 hours (one-time) |
| 3D triangulation (OpenCV, CPU-bound) | N/A | ~1 hour (one-time) |
| Feature extraction (Layers 1-3) | ~1GB | ~30 min (one-time) |
| Lite ST-GCN training (per stage, per fold) | ~2-4GB on 1 GPU | ~15-30 min |
| **Total active VRAM during training** | **<8GB** | Fits on 1x V100 |

Second V100 remains free for parallel experiments or hyperparameter search.

---

## 7. Processing Pipeline Summary

```
Step 1: Multi-Person 2D Pose Estimation
  Tool: MMPose RTMPose (top-down)
  Input: Raw video clips (3 viewpoints per clip)
  Output: 2D joint coordinates for ALL persons per frame per viewpoint
  Key: Identify patient vs caregiver by skeleton height

Step 2: Camera Calibration
  Method: Human Pose as Calibration Pattern (existing plan)
  Input: Patient 2D joints from synchronized frames
  Output: Per-patient camera intrinsic + extrinsic parameters

Step 3: 3D Triangulation (Patient)
  Tool: OpenCV triangulatePoints
  Input: Patient 2D joints from 3 views + camera params
  Output: (T, 17, 3) patient 3D skeleton

Step 4: 3D Triangulation (Caregiver) [when present]
  Tool: OpenCV triangulatePoints
  Input: Caregiver 2D joints from 3 views + camera params
  Output: (T, 17, 3) caregiver 3D skeleton (zero-padded when absent)

Step 5: Feature Extraction (Layer 1)
  Input: Patient 3D skeleton
  Output: (T, ~15) skeleton-derived features

Step 6: Interaction Feature Extraction (Layer 2)
  Input: Patient 3D skeleton + Caregiver 3D skeleton
  Output: (T, ~10) interaction features

Step 7: Context Vector Construction (Layer 3)
  Input: Manual annotations JSON
  Output: (18,) per-patient context vector

Step 8: Hierarchical Classification Training
  Input: All features concatenated
  Model: Lite ST-GCN + temporal pooling + MLP
  Output: GMFCS Level 1-5
```

---

## 8. Verification Plan

### 8.1 Multi-Person Pose Estimation Verification
- Visually inspect detected skeletons on 5-10 sample clips
- Confirm patient vs caregiver identification accuracy
- Check that caregiver is correctly tracked across frames

### 8.2 Feature Extraction Verification
- For known walker users (L3 patients): verify high WFI, low ASA
- For known AFO users: verify reduced AROM
- For acrylic stand users during sit-to-stand: verify SPC detection
- For caregiver-assisted L5 patients: verify low movement_independence_score

### 8.3 Classification Verification
- Patient-level leave-one-out or k-fold cross-validation
- Per-stage confusion matrices
- Special attention to L3/L4 boundary and L4/L5 boundary (the critical distinctions)
- Ablation study: compare accuracy WITH vs WITHOUT the new features (Layers 1-3) to quantify the contribution of assistive device/caregiver encoding

### 8.4 Feature Importance Analysis
- SHAP values or permutation importance to validate that the expected features drive the expected stage decisions
- Verify that movement_independence_score is the top feature for Stage 2-B (L4/L5)
- Verify that WFI and walker_used are top features for Stage 2-A (L2/L3 boundary)
