# Comprehensive Implementation Plan: Assistive Device & Caregiver Integration

## Context

**Problem:** The 3D skeletal pipeline (2D pose → triangulation → classification) captures only the patient's 17 body joints as `(T, 17, 3)`. All information about walkers, AFOs, acrylic support stands, and caregiver physical assistance is lost. Per GMFCS-E&R criteria, these are the primary signals distinguishing L2/L3, L3/L4, and L4/L5 boundaries.

**Solution:** Three-layer architecture:
- **Layer 1:** Enhanced Patient Skeleton — standard 3D skeleton + derived features that proxy for device usage
- **Layer 2:** Caregiver Interaction Skeleton — dual-skeleton from multi-person pose estimation + interaction features
- **Layer 3:** Assistive Context Vector — manual annotation of 24 patients extending the 7D metadata to ~18D

**Design Specs:**
- `docs/superpowers/specs/2026-03-31-assistive-device-integration-design.md` — Three-layer architecture
- `docs/superpowers/specs/2026-04-02-sam2-guided-tracking-design.md` — SAM2-guided person/walker tracking (supersedes auto height-ratio)

**Hardware:** 2x NVIDIA Tesla V100-DGXS-32GB, CUDA 12.4, 1.5TB disk  
**Data:** 24 patients, ~3,175 clips, 3 viewpoints (GoPro front / iPhone left / Galaxy right), 30fps

---

## Phase 0: Project Setup & Environment

### Step 0.1: Create project directory structure
```
CP/
├── src/
│   ├── __init__.py
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── sam2_tracker.py            # SAM2 video object tracking (child/caregiver/walker)
│   ├── pose/
│   │   ├── __init__.py
│   │   ├── multi_person_pose.py      # Multi-person 2D pose estimation + mask-guided assignment
│   │   └── person_identifier.py       # Mask-guided person ID (SAM2 primary, height-ratio fallback)
│   ├── calibration/
│   │   ├── __init__.py
│   │   └── pose_calibration.py        # Human Pose as Calibration Pattern
│   ├── triangulation/
│   │   ├── __init__.py
│   │   └── triangulate_3d.py          # 3D triangulation (patient + caregiver)
│   ├── features/
│   │   ├── __init__.py
│   │   ├── skeleton_features.py       # Layer 1: device-proxy skeletal features
│   │   ├── interaction_features.py    # Layer 2: caregiver interaction features
│   │   ├── walker_features.py         # Walker-skeleton spatial features (L3/L4 discrimination)
│   │   ├── movement_quality.py        # GMFCS-E&R movement quality descriptors
│   │   └── context_vector.py          # Layer 3: extended metadata encoder
│   ├── model/
│   │   ├── __init__.py
│   │   ├── dataset.py                 # Data loading and batching
│   │   ├── lite_stgcn.py             # Lightweight ST-GCN encoder
│   │   ├── classifier.py             # Multi-stream fusion (5 streams) + classification head
│   │   └── train.py                   # Hierarchical 2-stage training loop
│   └── utils/
│       ├── __init__.py
│       ├── naming.py                  # Clip ID parsing utilities (patient, view, movement)
│       ├── visualization.py           # Skeleton overlay, feature distribution plots
│       └── evaluation.py              # Cross-validation, ablation, SHAP analysis
├── data/
│   ├── raw/                           # Existing raw video files
│   ├── raw_synced/                    # Time-synchronized, clipped triplets (to be created)
│   ├── processed/
│   │   └── masks/                     # SAM2 propagated masks per clip
│   ├── skeleton_2d/                   # 2D pose estimation output (identified: child/caregiver)
│   ├── calibration/                   # Per-patient camera parameters
│   ├── skeleton_3d/
│   │   ├── patient/                   # Patient 3D skeletons (.npy)
│   │   └── caregiver/                 # Caregiver 3D skeletons (.npy)
│   ├── features/                      # Extracted features (.npy)
│   └── metadata/
│       ├── labels.json                # GMFCS labels (existing)
│       ├── triplets.json              # Triplet mapping (existing)
│       ├── annotation_schema.json     # Annotation schema (created)
│       ├── assistive_annotations.json # Device/assistance annotations (created, needs filling)
│       └── sam2_annotations.json      # Per-clip first-frame point annotations for SAM2
├── configs/
│   └── default.yaml                   # All hyperparameters and paths
├── scripts/
│   ├── annotate_first_frame.py        # Manual first-frame annotation tool for SAM2
│   ├── 00_propagate_masks.py          # Batch SAM2 mask propagation
│   ├── 01_extract_2d_pose.py          # Batch 2D pose extraction (mask-guided)
│   ├── 02_calibrate_cameras.py        # Batch camera calibration
│   ├── 03_triangulate_3d.py           # Batch 3D triangulation
│   ├── 04_extract_features.py         # Batch feature extraction (incl. walker spatial)
│   ├── 05_train.py                    # Training entry point
│   └── 06_evaluate.py                 # Evaluation entry point
└── docs/
    └── superpowers/specs/             # Design specifications
```

### Step 0.2: Install dependencies on V100 server

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install mmcv mmpose mmdet     # MMPose for multi-person pose estimation
pip install segment-anything-2    # SAM2 for video object tracking
pip install opencv-python-headless numpy scipy
pip install pyyaml tqdm matplotlib seaborn
pip install shap                   # For feature importance analysis
```

**Download SAM2 checkpoint:**
```bash
mkdir -p checkpoints/
wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

**Verify:**
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
# Expected: True 2
python -c "from mmpose.apis import MMPoseInferencer; print('MMPose OK')"
python -c "from sam2.build_sam import build_sam2_video_predictor; print('SAM2 OK')"
```

### Step 0.3: Create configuration file

File: `configs/default.yaml`

```yaml
# Paths
data_root: ./data
raw_synced_dir: ./data/raw_synced
skeleton_2d_dir: ./data/skeleton_2d
calibration_dir: ./data/calibration
skeleton_3d_dir: ./data/skeleton_3d
features_dir: ./data/features
metadata_dir: ./data/metadata

# SAM2 video object tracking
tracking:
  model: sam2_hiera_l
  checkpoint: checkpoints/sam2_hiera_large.pt
  device: cuda:0
  mask_output_dir: data/processed/masks/
  annotations_path: data/metadata/sam2_annotations.json

# Pose estimation
pose:
  model: rtmpose-l  # RTMPose-Large (COCO 17-joint)
  detector: rtmdet-m  # Person detector
  device: cuda:0
  batch_size: 16
  confidence_threshold: 0.3

# Person identification (mask-guided primary, height-ratio fallback)
person_id:
  method: mask_guided  # SAM2 masks primary, height_ratio fallback
  min_height_ratio: 1.3  # fallback: adult must be >= 1.3x child height
  fallback: center_position  # use center-of-frame if still ambiguous

# Camera calibration
calibration:
  method: human_pose_calibration  # Takahashi et al. [5]
  min_visible_joints: 8  # minimum joints visible for calibration
  ransac_threshold: 5.0  # pixels
  num_calibration_frames: 50  # frames used for calibration per patient

# 3D triangulation
triangulation:
  method: dlt  # Direct Linear Transform via OpenCV
  min_views: 2  # minimum camera views for triangulation
  reprojection_threshold: 15.0  # pixels, for outlier rejection
  bone_length_filter: true  # reject implausible bone lengths

# Feature extraction
features:
  wfi_window_size: 15  # frames for Wrist Fixation Index sliding window
  contact_threshold_m: 0.15  # meters, caregiver-patient contact threshold
  walker_proximity_threshold_px: 30  # pixels, hand-to-walker "contact" threshold
  fps: 30

# Model
model:
  stgcn:
    num_layers: 3
    channels: [3, 64, 64, 128]
    temporal_kernel: 9
    dropout: 0.3
    num_joints: 17
  fusion:
    hidden_dim: 64
    dropout: 0.3
  skeleton_feature_dim: 15  # Layer 1
  interaction_feature_dim: 10  # Layer 2
  context_vector_dim: 18  # Layer 3
  walker_feature_dim: 5  # Walker-skeleton spatial features

# Training
training:
  epochs: 100
  batch_size: 32
  lr: 0.001
  weight_decay: 0.0001
  scheduler: cosine
  patience: 20  # early stopping
  cv_folds: 6  # patient-level cross-validation (leave-4-out)
  class_weight: balanced

# GPU
device: cuda
num_gpus: 2
```

---

## Phase 1: Manual Annotation (Layer 3 Context Vector + SAM2 First-Frame Points)

**Why first:** Zero code dependency for context vector. SAM2 annotation requires only the annotation tool (already implemented). Both are manual tasks that produce ground truth for all downstream processing.

### Step 1.1: Review and complete assistive_annotations.json

**Already created:** `data/metadata/assistive_annotations.json` with all 24 patients pre-filled with defaults.

**Action required (manual, by you):** Review each patient's video clips and update:
1. `devices.walker` / `devices.walker_type` — Is a walker visible? Anterior or posterior?
2. `devices.afo` / `devices.afo_laterality` — Is the child wearing AFOs? One side or both?
3. `devices.acrylic_stand` — Is a transparent acrylic support stand used?
4. `per_movement.*.device` — Which device is used during each movement?
5. `per_movement.*.assistance` — Independence scale (0.0=dependent to 1.0=independent)
6. `overall_assistance` — FIM-like scale (0-6)

**Estimated effort:** ~5-10 minutes per patient = ~2-4 hours for all 24.

**Critical patients to annotate carefully:**
- **L3 patients (kku, ly, mkj, pjw):** Likely use walkers during walking. Check if acrylic stand is used for sit-to-stand.
- **L4 patients (hdi, jrh, lsa):** Check caregiver assistance level during walking and transitions. lsa (23 months) cannot walk — verify what assistance is provided for crawl/rolling.
- **L5 patients (ajy, hdu, kcw, kri, oms, pjo):** Caregiver physically assisting side rolling is the KEY annotation. Quantify how much the caregiver is doing vs the child.

### Step 1.2: Implement context vector encoder

File: `src/features/context_vector.py`

**Input:** `assistive_annotations.json` + original `labels.json` metadata  
**Output:** 18-dimensional vector per patient-movement pair

**Fields (18D):**
```
[0]  sex                    # 0=female, 1=male
[1]  age_normalized         # age_months / 72 (0-1)
[2]  w_status               # 0=cannot, 1=performed, 2=not_needed
[3]  cr_status              # same encoding
[4]  c_s_status             # same encoding
[5]  s_c_status             # same encoding
[6]  sr_status              # same encoding
[7]  walker_used            # 0/1 (from annotation)
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
```

**Implementation details:**
- Load both JSON files
- For each patient-movement pair, construct the 18D vector
- Movement-specific assistance field is filled from the matching movement entry
- If a movement wasn't performed, the per-movement assistance defaults to 0.0 (dependent)
- Normalize all fields to [0, 1] range for model input

**Verification:** Print context vectors for one L1, one L3, and one L5 patient. Confirm:
- L1: all assistance = 1.0, no devices, overall_assistance = 1.0
- L3: walker_used = 1, walk assistance = 1.0 (independent with device), overall_assistance ≈ 0.83
- L5: all assistance low (0.0-0.2), no devices, overall_assistance ≈ 0.17

### Step 1.3: SAM2 first-frame annotation (child, caregiver, walker)

**Tool:** `scripts/annotate_first_frame.py` (already implemented)

**Why SAM2 instead of automatic detection:** Analysis showed that automatic height-ratio person identification fails during physical assistance (L4/L5), produces identical wrist fixation signatures for "gripping walker" vs "held by caregiver" (L3/L4 confound), and was confirmed unreliable in prior project experience. Manual first-frame annotation with SAM2 tracking provides 100% correct person identification initialization. See `docs/superpowers/specs/2026-04-02-sam2-guided-tracking-design.md` for full rationale.

**Action required (manual, by you):** For each video clip, open the annotation tool and click on:
1. **Child** (required, every clip) — click anywhere on the child's body in the first frame
2. **Caregiver** (optional, when present) — click on the caregiver. Press S to skip if no caregiver visible.
3. **Walker** (optional, when visible) — click on the walker. Press S to skip if no walker.

**Usage:**
```bash
python scripts/annotate_first_frame.py --data-root ./data/raw_synced --resume
```

**Estimated effort:**
| Target | Clips | Est. Time |
|--------|-------|-----------|
| Child | ~3,175 | ~8.8 hours |
| Caregiver | ~1,300 (L3-L5 clips) | ~2.9 hours |
| Walker | ~200 (L3 walking clips) | ~0.3 hours |
| **Total** | | **~12 hours (one-time)** |

**Tips:**
- The tool supports `--resume` to skip already-annotated clips (auto-saves after each clip)
- Use `--re-annotate kku_w_01_FV` to fix a specific clip
- This annotation also serves as a data quality review — flag any problematic clips

**Output:** `data/metadata/sam2_annotations.json`

### Step 1.4: Batch SAM2 mask propagation

**Script:** `scripts/00_propagate_masks.py` (already implemented)

After first-frame annotations are complete, propagate masks through all frames:

```bash
python scripts/00_propagate_masks.py --config configs/default.yaml
# Or per-patient:
python scripts/00_propagate_masks.py --patient kku
```

**Output:** `data/processed/masks/{clip_id}/` containing:
- `child.npz` — (T, H, W) boolean mask array
- `caregiver.npz` — (T, H, W) if caregiver was annotated
- `walker.npz` — (T, H, W) if walker was annotated
- `tracking_meta.json` — annotation coordinates and video metadata

**Estimated runtime:** ~3,175 clips at ~2-5 seconds per clip on V100 = ~2-4 hours.

**Verification:**
- Visualize propagated masks on 2-3 sample clips per GMFCS level
- Check for mask drift, identity swaps, or loss of tracking
- Specifically verify L4/L5 clips where caregiver physically contacts child

---

## Phase 2: Multi-Person 2D Pose Estimation

**Goal:** Detect all persons in every frame using MMPose RTMPose, then assign detected skeletons to identity masks from SAM2 (child, caregiver). Falls back to height-ratio heuristic when masks are unavailable.

### Step 2.1: Set up MMPose with RTMPose

**Model selection:** RTMPose-Large with COCO 17-joint topology
- Pretrained on COCO + AIC + CrowdPose
- Top-down approach: RTMDet-m person detector → RTMPose-L pose estimator
- ~70+ AP on COCO val, real-time inference on V100

**Download:**
```bash
mim download mmpose --config rtmpose-l_8xb256-420e_coco-256x192 --dest checkpoints/
mim download mmdet --config rtmdet_m_8xb32-300e_coco --dest checkpoints/
```

### Step 2.2: Implement multi-person 2D pose extraction with mask-guided assignment

File: `src/pose/multi_person_pose.py` **(implemented)**

**Class: `MultiPersonPoseExtractor`**

```python
class MultiPersonPoseExtractor:
    def __init__(self, det_model, pose_model, device, batch_size, confidence_threshold):
        # Initialize MMPose inferencer with RTMPose-L + RTMDet-m
        # CUDA availability check with CPU fallback
        
    def extract_frame(self, frame):
        """Extract keypoints for all detected persons in a single frame.
        Returns: List of (17, 3) arrays."""
        
    def extract_video(self, video_path, person_masks=None, sample_rate=1):
        """Extract identified keypoints for all frames.
        
        If person_masks provided (from SAM2): uses mask-guided assignment
        If person_masks is None: falls back to height-ratio heuristic
        
        Returns: Dict mapping identity -> (T, 17, 3) array.
        """
```

**Output format per video:**
```
{clip_id}.npz:
    child: (T, 17, 3) — identified child keypoints (x, y, conf)
    caregiver: (T, 17, 3) — identified caregiver keypoints (if present)
```

**Handling edge cases:**
- Frames with 0 persons detected: linearly interpolated from adjacent valid frames
- Frames with 3+ persons: assigned by mask overlap (SAM2) or kept top-2 by consistency (fallback)
- Missing frames use `_interpolate_missing()` for smooth temporal sequences

### Step 2.3: Implement mask-guided person identification

File: `src/pose/person_identifier.py` **(implemented)**

**Primary method: SAM2 mask-guided assignment**

```python
def assign_skeletons_mask_guided(detected_skeletons, person_masks, conf_threshold=0.3):
    """Assign detected skeletons to SAM2 identity masks.
    
    Scoring: 0.7 * keypoint_overlap + 0.3 * bbox_IoU
    Assignment: greedy best-match (sufficient for 2 persons)
    
    Returns: Dict mapping identity name -> (17, 3) skeleton or None.
    """
```

**Fallback: height-ratio heuristic** (when SAM2 masks unavailable)

```python
def assign_skeletons_height_ratio(detected_skeletons, min_height_ratio=1.3):
    """Child = shortest skeleton. Caregiver = tallest if >= 1.3x child height."""
```

**Top-level function:**

```python
def identify_persons(detected_skeletons, person_masks=None, min_height_ratio=1.3):
    """Mask-guided if masks available, else height-ratio fallback."""
```

### Step 2.4: Batch processing script

File: `scripts/01_extract_2d_pose.py` **(implemented)**

```python
"""
Process all video clips:
1. For each clip:
   a. Load SAM2 person masks (child, caregiver) if available
   b. Run multi-person RTMPose pose extraction
   c. Assign skeletons to identity masks (or height-ratio fallback)
   d. Save identified 2D keypoints as .npz
   
Output: data/skeleton_2d/{clip_id}.npz (contains 'child' and 'caregiver' arrays)
"""
```

**Usage:**
```bash
python scripts/01_extract_2d_pose.py --config configs/default.yaml
python scripts/01_extract_2d_pose.py --patient kku
```

**Estimated runtime:** ~3,175 clips at ~10 fps inference = ~3-4 hours on 1x V100.

Logs report how many clips used mask-guided vs height-ratio fallback.

### Step 2.5: Visual verification

File: `src/utils/visualization.py` **(implemented)** — Skeleton overlay library
File: `scripts/verify_2d_pose.py` **(implemented)** — Verification script

**Functions in `visualization.py`:**
- `draw_skeleton()` — COCO 17-joint skeleton overlay (limbs + colored keypoints)
- `draw_mask_overlay()` — Semi-transparent SAM2 mask overlay
- `draw_verification_frame()` — Composite frame with skeletons, masks, and HUD
- `write_verification_video()` — Generator-based MP4 output via OpenCV

**Usage:**
```bash
python scripts/verify_2d_pose.py                          # 2 samples per level
python scripts/verify_2d_pose.py --patient kku             # all clips for patient
python scripts/verify_2d_pose.py --clip kku_w_01_FV        # single clip
python scripts/verify_2d_pose.py --level 3 --samples 3     # 3 L3 samples
python scripts/verify_2d_pose.py --all                     # every clip
python scripts/verify_2d_pose.py --video-dir data/raw      # override video source
python scripts/verify_2d_pose.py --no-masks                # skip mask overlay
```

**Output:** `outputs/verification_2d/L{1-5}/{clip_id}_verify.mp4` + `summary.txt`

**Specifically verify:**
- **Mask-guided clips:** Child skeleton correctly follows child mask throughout entire clip
- **L3 patient + walker:** Child skeleton is on the child, not confused with walker
- **L4/L5 patient + caregiver:** Both skeletons maintain correct identity during physical contact
- **Height-fallback clips:** Check if any clips without masks have incorrect assignment

---

## Phase 3: Camera Calibration & 3D Triangulation

### Step 3.1: Camera calibration (Human Pose as Calibration Pattern)

File: `src/calibration/pose_calibration.py` **(implemented)**

**Method:** Takahashi et al. [5] — uses the child's visible 2D joints across views as correspondence points, then estimates camera parameters.

**Class: `PoseCalibrator`**

```python
class PoseCalibrator:
    def __init__(self, config):
        # Known camera intrinsics (focal length, principal point)
        # for GoPro, iPhone 12 mini, Galaxy
        
    def get_intrinsics(self, camera_type):
        """
        Return camera intrinsic matrix K for known camera models.
        GoPro: wide-angle, needs distortion correction (k1, k2, p1, p2)
        iPhone 12 mini: ~26mm equivalent, minimal distortion
        Galaxy: standard, minimal distortion
        """
        
    def calibrate_extrinsics(self, patient_keypoints_per_view, intrinsics_per_view):
        """
        Estimate camera extrinsic parameters (R, t) for each view pair.
        
        Method:
        1. Collect 2D patient joint correspondences across views
           (only use joints visible in both views with confidence > threshold)
        2. Compute Fundamental Matrix (F) using 8-point algorithm + RANSAC
        3. Decompose to Essential Matrix: E = K2^T @ F @ K1
        4. Recover R, t from E using cv2.recoverPose
        5. Refine with bundle adjustment (optional, via scipy.optimize)
        
        Args:
            patient_keypoints_per_view: dict of {view_name: (T, 17, 3)} 
            intrinsics_per_view: dict of {view_name: K_matrix}
            
        Returns:
            extrinsics: dict of {view_name: (R, t)} relative to reference view (FV)
        """
        
    def undistort_gopro(self, keypoints, K, dist_coeffs):
        """Correct GoPro barrel distortion on 2D keypoints."""
```

**Per-patient calibration:** Camera extrinsics are estimated independently per patient because recording conditions (camera placement, angles) varied across sessions.

**GoPro distortion correction:** GoPro has barrel distortion (wide-angle lens). Must apply `cv2.undistortPoints()` before calibration and triangulation.

**Camera intrinsics (approximate, to be refined):**
```python
# GoPro Hero (wide-angle, 1920x1080)
K_gopro = np.array([[960, 0, 960], [0, 960, 540], [0, 0, 1]])  # ~2mm focal length equiv
dist_gopro = np.array([-0.25, 0.06, 0, 0])  # barrel distortion

# iPhone 12 mini (1920x1080, ~26mm equiv)
K_iphone = np.array([[1400, 0, 960], [0, 1400, 540], [0, 0, 1]])
dist_iphone = np.array([0, 0, 0, 0])  # negligible

# Galaxy (1920x1080)
K_galaxy = np.array([[1350, 0, 960], [0, 1350, 540], [0, 0, 1]])
dist_galaxy = np.array([0, 0, 0, 0])
```
NOTE: These are initial approximations. The actual intrinsics should be refined using EXIF data from the video files or calibrated during the process.

### Step 3.2: 3D triangulation

File: `src/triangulation/triangulate_3d.py`

**Class: `SkeletonTriangulator`**

```python
class SkeletonTriangulator:
    def __init__(self, config):
        # Triangulation parameters
        
    def triangulate_person(self, keypoints_per_view, projection_matrices):
        """
        Triangulate a single person's 3D skeleton from multi-view 2D keypoints.
        
        For each joint, for each frame:
        1. Collect 2D coordinates from all views where confidence > threshold
        2. If >= 2 views available: cv2.triangulatePoints(P1, P2, pts1, pts2)
        3. If 3 views available: use all pairs and average (or RANSAC best pair)
        4. If < 2 views: mark as missing, interpolate from adjacent frames
        
        Args:
            keypoints_per_view: {view: (T, 17, 3)} — x, y, confidence
            projection_matrices: {view: P_3x4} — P = K @ [R|t]
            
        Returns:
            skeleton_3d: (T, 17, 3) — x, y, z in world coordinates
            confidence_3d: (T, 17) — triangulation confidence per joint
        """
        
    def validate_skeleton(self, skeleton_3d):
        """
        Post-processing:
        1. Bone length consistency: reject frames where bone lengths deviate > 30% from median
        2. Joint angle plausibility: reject physiologically impossible angles
        3. Temporal smoothing: Savitzky-Golay filter for jitter reduction
        4. Interpolate missing joints from adjacent frames
        """
        
    def triangulate_clip(self, patient_kp_per_view, caregiver_kp_per_view, 
                          projection_matrices):
        """
        Triangulate both patient and caregiver for a single clip.
        
        Returns:
            patient_3d: (T, 17, 3)
            caregiver_3d: (T, 17, 3) or zeros if no caregiver
            caregiver_present: (T,) boolean
        """
```

### Step 3.3: Batch processing script

File: `scripts/02_calibrate_cameras.py` + `scripts/03_triangulate_3d.py`

**Calibration script:**
```python
"""
For each patient:
1. Load 2D patient keypoints from all triplets of that patient
2. Select best calibration frames (highest joint visibility across all 3 views)
3. Estimate extrinsics using PoseCalibrator
4. Save calibration results to data/calibration/{patient_id}.npz
"""
```

**Triangulation script:**
```python
"""
For each triplet:
1. Load calibration for this patient
2. Load 2D keypoints (patient + caregiver) for all 3 viewpoints
3. Build projection matrices P = K @ [R|t] per view
4. Triangulate patient -> save data/skeleton_3d/patient/{clip_id}.npy
5. Triangulate caregiver (if present) -> save data/skeleton_3d/caregiver/{clip_id}.npy
"""
```

**Output format:**
```
data/skeleton_3d/patient/{patient_id}_{movement}_{number}.npy
    Shape: (T, 17, 3) — float32, xyz in meters

data/skeleton_3d/caregiver/{patient_id}_{movement}_{number}.npy
    Shape: (T, 17, 3) — float32, xyz in meters (zeros if no caregiver)
```

### Step 3.4: Verification

- Visualize 3D skeletons (matplotlib 3D scatter) for 2-3 clips per GMFCS level
- Check bone length consistency across frames (should be near-constant for rigid skeleton)
- Check that caregiver skeleton is anatomically plausible when present
- Compute reprojection error: project 3D joints back to 2D, measure pixel distance from original 2D detections. Target: < 10 pixels mean error.

---

## Phase 4: Feature Extraction

### Step 4.1: Layer 1 — Skeleton-derived features for device usage detection

File: `src/features/skeleton_features.py`

**Class: `SkeletonFeatureExtractor`**

Computes ~15 features per frame from the patient's 3D skeleton. These features serve as proxy indicators for assistive device usage and movement quality.

**Joint index convention (COCO 17-joint):**
```
0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
```

**Feature list with computation details:**

#### F1-F2: Wrist Fixation Index (WFI) — left and right
**Detects:** Hands gripping walker or acrylic stand
```python
def wrist_fixation_index(wrist_pos_3d, window=15):
    # wrist_pos_3d: (T, 3)
    # Compute rolling standard deviation of 3D position
    rolling_std = rolling_window_std(wrist_pos_3d, window)  # (T, 3)
    wfi = 1.0 - np.clip(np.linalg.norm(rolling_std, axis=-1) / MAX_STD, 0, 1)
    return wfi  # (T,) — 0=free movement, 1=completely fixed
```
- **Walker during walking:** Both WFI_left and WFI_right high (~0.8-1.0), wrists at hip height
- **Acrylic stand during sit-to-stand:** Both WFI high, wrists at stand-top height, arms directed downward

#### F3-F4: Arm Swing Amplitude (ASA) — left and right
**Detects:** Walker-assisted (low) vs free walking (high)
```python
def arm_swing_amplitude(wrist_pos_3d, pelvis_forward_direction):
    # Project wrist onto anteroposterior axis
    wrist_ap = project_onto_axis(wrist_pos_3d, pelvis_forward_direction)
    # Compute peak-to-peak amplitude per gait cycle (or fixed window)
    amplitude = rolling_peak_to_peak(wrist_ap, window=60)  # 2 sec at 30fps
    return amplitude  # (T,)
```

#### F5-F6: Ankle Range of Motion (AROM) — left and right
**Detects:** AFO presence (restricted ROM)
```python
def ankle_rom(knee_3d, ankle_3d, toe_proxy_3d):
    # toe_proxy approximated from ankle direction
    angles = compute_joint_angle(knee_3d, ankle_3d, toe_proxy_3d)  # (T,)
    # Rolling ROM over 1-second window
    rom = rolling_max(angles, 30) - rolling_min(angles, 30)
    return rom  # (T,) — low ROM = AFO constraining joint
```
NOTE: COCO 17-joint doesn't include toes. Approximate toe position using ankle position + foot direction estimated from ankle-knee vector.

#### F7: Upper Limb Freedom Score (ULFS)
**Detects:** Overall arm constraint (free vs device-bound)
```python
def upper_limb_freedom(wrist_vel, shoulder_rom, bilateral_symmetry):
    # Composite: mean wrist velocity * shoulder ROM * bilateral symmetry
    return mean_velocity * shoulder_range * symmetry  # (T,)
```

#### F8-F9: Center of Mass (CoM) Sway and Smoothness
**Detects:** Balance and stability
```python
def com_features(hip_l, hip_r, shoulder_l, shoulder_r, head):
    com = 0.4 * (hip_l + hip_r)/2 + 0.4 * (shoulder_l + shoulder_r)/2 + 0.2 * head
    sway = rolling_lateral_variance(com, window=30)  # (T,)
    jerk = np.diff(com, n=3, axis=0)  # third derivative
    smoothness = 1.0 / (1.0 + np.linalg.norm(jerk, axis=-1))  # (T-3,) padded
    return sway, smoothness
```

#### F10: Support Point Convergence (SPC)
**Detects:** Acrylic stand usage during sit-to-stand
```python
def support_point_convergence(left_wrist, right_wrist, hip_height):
    wrist_mean_height = (left_wrist[:, 2] + right_wrist[:, 2]) / 2  # z-axis
    wrist_height_stability = rolling_std(wrist_mean_height, window=15)
    hip_rising = np.gradient(hip_height) > 0.001  # torso is rising
    spc = (wrist_height_stability < THRESHOLD) & hip_rising
    return spc.astype(float)  # (T,) — 1.0 when using support surface
```
**Key behavior:** Patient grips top edge of acrylic stand with arms directed downward (not forward). Wrists stay at a fixed height while hip rises.

#### F11: Gait Symmetry Index (GSI)
**Detects:** Walking quality and asymmetry
```python
def gait_symmetry(left_ankle, right_ankle):
    # Compute step length from ankle displacement per half-cycle
    # GSI = |left_step - right_step| / mean_step
    # L1 ≈ 0 (symmetric), L3+ >> 0 (asymmetric)
```

#### F12-F13: Wrist height relative to hip — left and right
**Detects:** Hand position pattern (gripping height indicator)
```python
def relative_hand_height(wrist_3d, hip_3d):
    return wrist_3d[:, 2] - hip_3d[:, 2]  # (T,) — positive = above hip
```
- Walker grip: wrists near hip height (relative ≈ 0)
- Acrylic stand during sit-to-stand: wrists initially above hip, then at/below hip as patient rises

#### F14: Bilateral wrist distance
**Detects:** Two-handed grip pattern (walker has ~50cm handle width)
```python
def bilateral_wrist_distance(left_wrist, right_wrist):
    return np.linalg.norm(left_wrist - right_wrist, axis=-1)  # (T,)
```

#### F15: Torso vertical velocity
**Detects:** Transition dynamics (sit-to-stand speed, descent control)
```python
def torso_vertical_velocity(hip_center):
    return np.gradient(hip_center[:, 2]) * fps  # (T,) m/s
```

**Output:** `(T, 15)` tensor per clip, saved as `.npy`

### Step 4.2: Layer 1 — Extended movement quality features

File: `src/features/movement_quality.py`

These features are derived from the GMFCS-E&R movement quality descriptors (Report Section 15.4). They are movement-type-specific features computed from the 3D skeleton.

**Walk-specific features (from Report 15.4.1):**
```python
class WalkQualityFeatures:
    def compute(self, skeleton_3d):
        return {
            'cadence': ...,               # stride cycles per second (ankle periodicity)
            'gait_symmetry': ...,         # L/R ankle swing duration ratio
            'trunk_lateral_sway': ...,    # pelvis lateral oscillation amplitude
            'step_width': ...,            # bilateral ankle distance during stance
            'head_stability': ...,        # head vertical+lateral oscillation
        }
```

**Seated-to-standing features (from Report 15.4.2):**
```python
class SitToStandQualityFeatures:
    def compute(self, skeleton_3d):
        return {
            'transition_duration': ...,    # time for hip to reach standing height
            'trunk_anterior_tilt': ...,    # max forward tilt of spine-pelvis angle
            'hand_to_ground_contact': ..., # freq of wrist reaching floor level
            'com_trajectory_jerk': ...,    # pelvis trajectory jerk (smoothness)
            'knee_extension_symmetry': ...,# L/R knee angle change time lag
            'post_transition_sway': ...,   # AP/ML oscillation 2sec after standing
        }
```

**Crawl-specific features (from Report 15.4.3):**
```python
class CrawlQualityFeatures:
    def compute(self, skeleton_3d):
        return {
            'reciprocal_pattern_index': ...,  # contralateral hand-knee pair ratio
            'trunk_elevation': ...,            # pelvis-to-ground height
            'crawl_velocity': ...,             # horizontal pelvis speed
            'cycle_regularity': ...,           # CV of hand/knee advancement cycle
            'trunk_roll_amplitude': ...,       # lateral roll angle of spine
            'upper_lower_limb_ratio': ...,     # upper/lower displacement ratio
        }
```

**Side-rolling features (from Report 15.4.5):**
```python
class SideRollingQualityFeatures:
    def compute(self, skeleton_3d):
        return {
            'segmental_rotation_ratio': ..., # shoulder-hip roll onset lag
            'rolling_velocity': ...,          # time per complete roll
            'bilateral_symmetry': ...,        # L→R vs R→L speed ratio
            'inter_roll_recovery': ...,       # rest time between rolls
            'arm_rom_during_roll': ...,       # shoulder abduction/flexion range
        }
```

**Standing-to-seated features (from Report 15.4.4):**
```python
class StandToSitQualityFeatures:
    def compute(self, skeleton_3d):
        return {
            'descent_velocity_profile': ..., # pelvis z-velocity profile
            'controlled_deceleration': ...,  # deceleration rate before sitting
            'impact_jerk': ...,              # vertical jerk at sitting moment
            'hand_support_frequency': ...,   # wrist contacting support surface
        }
```

**These are computed per clip (not per frame) and produce a fixed-length feature vector per movement type.** The model receives these as additional input alongside the per-frame features.

### Step 4.3: Layer 2 — Caregiver interaction features

File: `src/features/interaction_features.py`

**Class: `InteractionFeatureExtractor`**

Computes ~10 features per frame from the paired patient + caregiver 3D skeletons.

**Feature list:**

#### I1: Caregiver present
```python
caregiver_present = (caregiver_skeleton != 0).any(axis=(1,2)).astype(float)  # (T,)
```

#### I2: Minimum hand-body distance
```python
def min_hand_body_distance(caregiver_wrists, patient_all_joints):
    # caregiver_wrists: (T, 2, 3) — left and right wrist
    # patient_all_joints: (T, 17, 3)
    distances = cdist(caregiver_wrists, patient_all_joints)  # per frame
    return distances.min(axis=(1,2))  # (T,)
```

#### I3: Contact point count
```python
def contact_point_count(caregiver_joints, patient_joints, threshold=0.15):
    # Count patient joints within threshold distance of any caregiver joint
    # (T,) integer — 0 to 17
```

#### I4: Contact duration ratio (running)
```python
def contact_duration_ratio(contact_flags):
    # Cumulative ratio of frames with any contact up to current frame
    return np.cumsum(contact_flags) / np.arange(1, len(contact_flags)+1)
```

#### I5: Velocity correlation (CRITICAL for L4/L5)
```python
def velocity_correlation(caregiver_hand_vel, patient_torso_vel, window=30):
    """
    Pearson correlation between caregiver hand velocity and patient torso velocity.
    High correlation = caregiver is DRIVING the movement.
    
    Computed over rolling window.
    """
    # caregiver_hand_vel: (T, 3) — mean of left+right wrist velocity
    # patient_torso_vel: (T, 3) — hip center velocity
    corr = rolling_correlation(caregiver_hand_vel, patient_torso_vel, window)
    return corr  # (T,) — -1 to 1, high positive = caregiver driving movement
```

#### I6: Movement independence score (THE key L4/L5 feature)
```python
def movement_independence_score(patient_vel, contact_flags):
    """
    Ratio of patient's own movement during non-contact vs contact phases.
    
    If patient moves at same speed regardless of contact -> independent (score ~1.0)
    If patient only moves when being contacted -> dependent (score ~0.0)
    
    L4 side rolling: score ~0.7-1.0 (self-initiates)
    L5 side rolling: score ~0.0-0.3 (caregiver drives rotation)
    """
    vel_magnitude = np.linalg.norm(patient_vel, axis=-1)  # (T,)
    
    contact_mask = contact_flags.astype(bool)
    if contact_mask.sum() == 0 or (~contact_mask).sum() == 0:
        return np.ones(len(patient_vel))  # fully independent or no contact info
    
    vel_no_contact = vel_magnitude[~contact_mask].mean()
    vel_contact = vel_magnitude[contact_mask].mean()
    
    if vel_contact == 0:
        return np.ones(len(patient_vel))  # no movement = independent (static)
    
    score = vel_no_contact / max(vel_contact, vel_no_contact)
    return np.full(len(patient_vel), score)  # clip-level score broadcast to frames
```

#### I7-I10: Contact body region (4 binary channels)
```python
def contact_body_region(caregiver_joints, patient_joints, threshold):
    """
    Which patient body regions are in contact with caregiver.
    Returns (T, 4): [head, trunk, upper_limb, lower_limb]
    """
    regions = {
        'head': [0, 1, 2, 3, 4],       # nose, eyes, ears
        'trunk': [5, 6, 11, 12],        # shoulders, hips
        'upper_limb': [7, 8, 9, 10],    # elbows, wrists
        'lower_limb': [13, 14, 15, 16]  # knees, ankles
    }
    # Per frame, per region: 1 if any joint in region is within threshold of caregiver
```

**Output:** `(T, 10)` tensor per clip

**When no caregiver is present:** All interaction features are zero. This itself is informative (L1/L2/L3 should have no caregiver during most movements).

### Step 4.4: Batch processing script

File: `scripts/04_extract_features.py`

```python
"""
For each clip:
1. Load patient 3D skeleton from data/skeleton_3d/patient/
2. Load caregiver 3D skeleton from data/skeleton_3d/caregiver/ (if exists)
3. Determine movement type from filename
4. Compute Layer 1 skeleton features -> (T, 15)
5. Compute movement quality features -> (M,) fixed-length per movement type
6. Compute Layer 2 interaction features -> (T, 10)
7. Load Layer 3 context vector -> (18,)
8. Save all features to data/features/{clip_id}.npz
"""
```

### Step 4.5: Feature verification

**Sanity checks (run after batch feature extraction):**

1. **WFI distribution by level:**
   - Compute mean WFI across all walking clips, grouped by GMFCS level
   - Expected: L1/L2 low (~0.1-0.3), L3 high (~0.7-0.9)

2. **ASA distribution by level:**
   - Expected: L1 highest, decreasing through L2, L3 near zero

3. **AROM for known AFO users:**
   - If annotations indicate AFO: AROM should be notably lower than non-AFO patients

4. **Independence score for side rolling:**
   - L4 patients: ~0.7-1.0
   - L5 patients: ~0.0-0.3

5. **Contact duration for L5 movements:**
   - Should be significantly higher than L4

Generate distribution plots (box plots per GMFCS level) for each feature. Save to `outputs/feature_distributions/`.

---

## Phase 5: Classification Model

### Step 5.1: Dataset and data loading

File: `src/model/dataset.py`

**Class: `GMFCSDataset(torch.utils.data.Dataset)`**

```python
class GMFCSDataset:
    def __init__(self, clip_ids, features_dir, metadata_dir, 
                 max_seq_len=150, stage='stage1'):
        """
        Args:
            clip_ids: list of clip identifiers
            max_seq_len: pad/truncate to fixed length (150 frames = 5 sec at 30fps)
            stage: 'stage1' (binary), 'stage2a' (L1/L2/L3), 'stage2b' (L4/L5)
        """
        
    def __getitem__(self, idx):
        """
        Returns:
            patient_skeleton: (T, 17, 3) — padded/truncated to max_seq_len
            skeleton_features: (T, 15)
            interaction_features: (T, 10)
            context_vector: (18,)
            label: int (0/1 for stage1, 0/1/2 for stage2a, 0/1 for stage2b)
            movement_quality: (M,) — movement-specific quality features
            mask: (T,) — 1 for real frames, 0 for padding
        """
```

**Patient-level data split:**
```python
class PatientLevelSplitter:
    def __init__(self, patient_ids, gmfcs_levels, n_folds=6):
        """
        Stratified patient-level cross-validation.
        Each fold: ~4 patients in test set, ~20 in train.
        Stratified by GMFCS level to ensure each fold has representation.
        
        CRITICAL: All clips from one patient are in the SAME fold.
        No data leakage between train and test.
        """
```

### Step 5.2: Lite ST-GCN encoder

File: `src/model/lite_stgcn.py`

```python
class SpatialGraphConv(nn.Module):
    """Single spatial graph convolution layer."""
    def __init__(self, in_channels, out_channels, adjacency_matrix):
        # Standard spatial GCN: A @ X @ W
        
class TemporalConv(nn.Module):
    """Temporal convolution along the time axis."""
    def __init__(self, in_channels, out_channels, kernel_size=9):
        # 1D convolution along time dimension
        
class STGCNBlock(nn.Module):
    """One ST-GCN block = SpatialGraphConv + TemporalConv + residual + BN + ReLU."""
    
class LiteSTGCN(nn.Module):
    """
    Lightweight ST-GCN encoder.
    
    Config:
        num_layers: 3
        channels: [3, 64, 64, 128]
        temporal_kernel: 9
        num_joints: 17
        dropout: 0.3
        
    Forward:
        Input: (B, T, 17, 3)
        -> Reshape to (B, 3, T, 17)  [channel-first for conv]
        -> 3x STGCNBlock
        -> Global average pooling over time and joints
        -> Output: (B, 128)
        
    Estimated parameters: ~200K
    """
```

**Graph adjacency matrix (COCO 17-joint):**
```python
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]
# Build normalized adjacency: A_hat = D^(-1/2) @ (A + I) @ D^(-1/2)
```

### Step 5.3: Multi-stream fusion classifier

File: `src/model/classifier.py`

```python
class MultiStreamClassifier(nn.Module):
    """
    Multi-stream architecture (implemented in src/model/classifier.py):
    
    Stream A: LiteSTGCN on patient skeleton -> (B, 128) 
    Stream B: Temporal mean+std pooling on skeleton features -> (B, 30)
    Stream C: Temporal mean+std pooling on interaction features -> (B, 20)
    Stream D: Context vector passthrough -> (B, 18)
    Stream E: Walker-skeleton spatial features -> (B, 5)
    
    Fusion: concat all -> MLP(201 -> 64 -> dropout -> num_classes)
    
    Supports hierarchical mode with separate heads:
        Stage 1: Ambulatory vs Non-ambulatory (2 classes, routed by w_status)
        Stage 2-A: L1 vs L2 vs L3-L4 (3 classes, ambulatory branch)
        Stage 2-B: L3-L4 vs L5 (2 classes, non-ambulatory branch)
    """
    
    def __init__(self, stgcn_dim=128, skeleton_feature_dim=15,
                 interaction_feature_dim=10, context_vector_dim=18,
                 walker_feature_dim=5, hidden_dim=64, dropout=0.3,
                 num_classes=5, hierarchical=True):
        ...

    @staticmethod
    def temporal_pool(features_seq):
        """Mean + std temporal pooling. (T, D) -> (2*D,)"""
        ...
        
    def forward(self, stgcn_embedding, skeleton_features, interaction_features,
                context_vector, walker_features, stage=None):
        # Stream B/C: temporal pool
        # Fuse A(128) + B(30) + C(20) + D(18) + E(5) = 201D
        # Route to appropriate stage head
        ...
```

**Key change from original plan:** Stream E is now **walker-skeleton spatial features (5D)** computed from SAM2 walker masks + child keypoints, not movement quality features. Walker features provide direct L3/L4 discrimination via hand-to-walker distance, engagement ratio, and support source ratio. Movement quality descriptors are folded into Layer 1 skeleton features.

### Step 5.4: Hierarchical 2-stage training

File: `src/model/train.py`

```python
class HierarchicalTrainer:
    """
    Stage 1: Binary classifier — ambulatory vs non-ambulatory
        - Routing by actual walking ability (w_status from metadata), NOT by GMFCS level.
        - Ambulatory: L1(6) + L2(5) + L3(3: ly,mkj,pjw) + L4(2: hdi,jrh) = 16 patients
        - Non-ambulatory: L3(1: kku) + L4(1: lsa) + L5(6) = 8 patients
        - Uses: seated_to_standing, crawl (transition quality, independent performance ability)
    
    Stage 2-A: 3-class classifier — L1 vs L2 vs L3-L4 (on ambulatory subset)
        - L3 and L4 merged: both can walk (L3 with device, L4 with caregiver assistance)
        - Uses: walk, seated_to_standing, standing_to_seated
        - Key features: WFI, ASA, walker_engagement, support_source_ratio
    
    Stage 2-B: 2-class classifier — L3-L4 vs L5 (on non-ambulatory subset)
        - L3 and L4 merged (only 2 patients total: kku L3, lsa L4)
        - Uses: crawl, side_rolling, seated_to_standing
        - Key features: movement_independence_score, velocity_correlation
    
    Note: L3/L4 are merged in both branches because the ambulatory/non-ambulatory
    split does not cleanly map to GMFCS levels. Final L3 vs L4 resolution relies
    on device features (walker engagement → L3, caregiver assistance → L4).
    
    Each stage trained independently with its own model instance.
    Patient-level cross-validation.
    Class-weight balancing per stage.
    """
    
    def train_stage(self, stage_name, dataset, num_classes, n_folds=6):
        """
        Train one stage with patient-level CV.
        
        For each fold:
            1. Split patients into train/test
            2. Create train and test datasets
            3. Compute class weights from training set
            4. Train GMFCSClassifier with CE loss + class weights
            5. Early stopping on validation loss (patience=20)
            6. Record test predictions
        
        Returns: per-fold confusion matrices, overall metrics
        """
        
    def train_all(self):
        """
        Sequential training of all stages:
        1. Train Stage 1
        2. For Stage 2-A: filter to ambulatory patients, train
        3. For Stage 2-B: filter to non-ambulatory patients, train
        4. Combine: Stage 1 routing + Stage 2 predictions = final GMFCS level
        """
```

**Training hyperparameters:**
- Optimizer: AdamW (lr=0.001, weight_decay=0.0001)
- Scheduler: CosineAnnealingLR (T_max=100)
- Loss: CrossEntropyLoss with class_weight='balanced'
- Early stopping: patience=20 epochs on validation loss
- Batch size: 32
- Max epochs: 100

**Estimated training time per stage per fold:** ~15-30 minutes on 1x V100  
**Total training time:** 3 stages x 6 folds x ~20 min = ~6 hours

### Step 5.5: Training script

File: `scripts/05_train.py`

```python
"""
Entry point for training:
1. Load configuration from configs/default.yaml
2. Initialize HierarchicalTrainer
3. Train all 3 stages with patient-level CV
4. Save trained models and results to outputs/models/
5. Log training curves to outputs/logs/
"""
```

---

## Phase 6: Evaluation & Verification

### Step 6.1: Per-stage evaluation

File: `src/utils/evaluation.py`

```python
class GMFCSEvaluator:
    def evaluate_per_stage(self, predictions, labels):
        """
        Per-stage confusion matrix, precision, recall, F1.
        Focus metrics:
        - Stage 1: ambulatory vs non-ambulatory accuracy
        - Stage 2-A: L2 vs L3 boundary (walker usage distinction)
        - Stage 2-B: L4 vs L5 boundary (caregiver assistance distinction)
        """
        
    def evaluate_end_to_end(self, stage1_pred, stage2a_pred, stage2b_pred, true_labels):
        """
        Combine hierarchical predictions into final GMFCS level.
        Compute overall 5-class accuracy, macro F1, per-level metrics.
        Compare to target: 80%+ accuracy.
        """
```

### Step 6.2: Ablation study

**4 model configurations to compare:**

| Model | Skeleton | Layer 1 (device proxy) | Layer 2 (caregiver) | Layer 3 (metadata) |
|---|---|---|---|---|
| Baseline | (T,17,3) | - | - | 7D original |
| + Device Features | (T,17,3) | (T,15) | - | 7D original |
| + Caregiver | (T,17,3) | (T,15) | (T,10) | 7D original |
| **Full Model** | (T,17,3) | (T,15) | (T,10) | **18D extended** |

Train each configuration with identical CV splits. Compare accuracy at each stage.

**Expected outcome:**
- Baseline → Full Model: significant improvement on L2/L3, L3/L4, L4/L5 boundaries
- Layer 2 (caregiver) provides largest boost for Stage 2-B (L4/L5)
- Layer 1 (device proxy) provides largest boost for Stage 2-A (L2/L3 boundary)

### Step 6.3: Feature importance analysis

```python
def feature_importance_analysis(model, test_data):
    """
    SHAP values or permutation importance per classification stage.
    
    Verify:
    - Stage 2-A: WFI, ASA, walker_used are top features for L2/L3
    - Stage 2-B: movement_independence_score, velocity_correlation are top for L4/L5
    - Stage 1: walker_used, overall_assistance, w_status are top
    """
```

### Step 6.4: Evaluation script

File: `scripts/06_evaluate.py`

```python
"""
1. Load trained models from outputs/models/
2. Run per-stage evaluation
3. Run end-to-end evaluation
4. Run ablation comparison
5. Run feature importance analysis
6. Generate all reports and plots to outputs/evaluation/
"""
```

---

## Phase Summary & Dependencies

```
Phase 0: Setup ──────────────────────────────────────┐
                                                      │
Phase 1: Manual Annotation (no code dependency) ──────┤
                                                      │
Phase 2: Multi-Person 2D Pose ── depends on Phase 0 ─┤
                                                      │
Phase 3: Calibration + 3D ──── depends on Phase 2 ───┤
                                                      │
Phase 4: Feature Extraction ── depends on Phase 3 + 1 ┤
                                                      │
Phase 5: Classification ────── depends on Phase 4 ────┤
                                                      │
Phase 6: Evaluation ─────────  depends on Phase 5 ────┘
```

**Phase 1 (manual annotation) can run in parallel with Phases 0-3.**

## Verification Milestones

| After Phase | Verification | Pass Criteria |
|---|---|---|
| Phase 2 | Visual overlay of detected skeletons | Patient/caregiver correctly identified in >95% of frames |
| Phase 3 | Reprojection error | < 10 pixels mean across all clips |
| Phase 3 | Bone length consistency | < 5% coefficient of variation per patient |
| Phase 4 | WFI distribution by GMFCS level | Clear separation: L1/L2 low, L3 high |
| Phase 4 | Independence score for side rolling | L4 > 0.5, L5 < 0.4 |
| Phase 5 | Stage 2-B (L4/L5) accuracy | > 80% with full model |
| Phase 6 | End-to-end 5-class accuracy | Target: 80%+ |
| Phase 6 | Ablation: Full > Baseline | Statistically significant improvement |
