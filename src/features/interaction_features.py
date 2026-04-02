"""Layer 2: Caregiver interaction features.

Computes 10 per-frame features from paired patient-caregiver 3D skeletons:
contact proximity, velocity correlation, movement independence score, and
contact body region indicators.

The movement independence score is THE key signal for L4 vs L5 distinction.

Features (10D per frame):
  [0]  caregiver_present        - Binary: 1 if caregiver skeleton detected
  [1]  min_hand_body_distance   - Closest caregiver wrist to any patient joint
  [2]  contact_point_count      - Fraction of patient joints in contact range
  [3]  contact_duration_ratio   - Running ratio of frames with any contact
  [4]  velocity_correlation     - Pearson r of caregiver hand & patient torso speed
  [5]  movement_independence    - Patient velocity ratio: non-contact / contact phases
  [6]  contact_region_head      - Binary: caregiver near patient head/face
  [7]  contact_region_trunk     - Binary: caregiver near patient trunk
  [8]  contact_region_upper     - Binary: caregiver near patient arms
  [9]  contact_region_lower     - Binary: caregiver near patient legs

All distance-based features are normalized by patient torso length to handle
the scale ambiguity from cv2.recoverPose.

When no caregiver is present (all-zero skeleton), all features are zero.
This is itself informative: L1/L2/L3 should have no caregiver during most
movements.
"""

import logging

import numpy as np

from src.features.skeleton_features import (
    DEFAULT_FPS,
    LEFT_ANKLE,
    LEFT_EAR,
    LEFT_ELBOW,
    LEFT_EYE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_EAR,
    RIGHT_ELBOW,
    RIGHT_EYE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    _torso_length,
    _valid_frame_mask,
)

logger = logging.getLogger(__name__)

NUM_INTERACTION_FEATURES = 10

# Body region definitions for contact region features (I7-I10)
_REGION_HEAD = [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]
_REGION_TRUNK = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
_REGION_UPPER_LIMB = [LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]
_REGION_LOWER_LIMB = [LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]

# Default contact threshold in torso-length units.
# Config provides contact_threshold_m (default 0.15m). We convert to
# torso-length-relative units: 0.15m / ~0.35m child torso ≈ 0.4.
# Since 3D coordinates have scale ambiguity, we work in relative units.
DEFAULT_CONTACT_THRESHOLD_REL = 0.4
_ASSUMED_TORSO_M = 0.35  # approximate child torso length in meters

# Rolling window for velocity correlation (frames)
DEFAULT_CORR_WINDOW = 30


# ── Helpers ──────────────────────────────────────────────────────────────

def _caregiver_present_mask(caregiver_3d):
    """Detect frames where caregiver skeleton has non-zero joints.

    Args:
        caregiver_3d: (T, 17, 3).

    Returns:
        (T,) boolean mask. True if any joint is non-zero.
    """
    return np.any(np.abs(caregiver_3d) > 1e-6, axis=(1, 2))


def _full_distance_matrix(caregiver_3d, child_3d):
    """Compute per-frame pairwise distance matrix between all joint pairs.

    Vectorized: O(T * 17 * 17) with numpy broadcasting (no Python loops).

    Args:
        caregiver_3d: (T, 17, 3).
        child_3d: (T, 17, 3).

    Returns:
        (T, 17, 17) float array. dist_matrix[t, cj, pj] = distance between
        caregiver joint cj and patient joint pj at frame t.
        Missing joints (zero vectors) get distance set to inf.
    """
    # (T, 17_cg, 1, 3) - (T, 1, 17_pt, 3) → (T, 17, 17, 3)
    diff = caregiver_3d[:, :, None, :] - child_3d[:, None, :, :]
    dist = np.linalg.norm(diff, axis=-1)  # (T, 17, 17)

    # Mask out missing joints (zero vectors) by setting distance to inf
    cg_missing = np.linalg.norm(caregiver_3d, axis=2) < 1e-6  # (T, 17)
    pt_missing = np.linalg.norm(child_3d, axis=2) < 1e-6       # (T, 17)
    # Expand masks: cg_missing[:, :, None] for caregiver dim, pt_missing[:, None, :] for patient dim
    dist[cg_missing[:, :, None].repeat(17, axis=2)] = np.inf
    dist[pt_missing[:, None, :].repeat(17, axis=1)] = np.inf

    return dist


def _rolling_correlation(signal_a, signal_b, window):
    """Rolling Pearson correlation between two 1D signals.

    Args:
        signal_a: (T,) array.
        signal_b: (T,) array.
        window: int, window size in frames.

    Returns:
        (T,) array of correlation values in [-1, 1].
    """
    T = len(signal_a)
    result = np.zeros(T, dtype=np.float64)

    for t in range(T):
        start = max(0, t - window + 1)
        a_seg = signal_a[start:t + 1]
        b_seg = signal_b[start:t + 1]

        if len(a_seg) < 3:
            continue

        std_a = np.std(a_seg)
        std_b = np.std(b_seg)

        if std_a < 1e-8 or std_b < 1e-8:
            # No variance — correlation undefined, default to 0
            continue

        mean_a = np.mean(a_seg)
        mean_b = np.mean(b_seg)
        cov = np.mean((a_seg - mean_a) * (b_seg - mean_b))
        result[t] = cov / (std_a * std_b)

    return np.clip(result, -1.0, 1.0)


# ── Feature computation ─────────────────────────────────────────────────

def _compute_caregiver_present(caregiver_3d):
    """I1: Binary caregiver presence indicator.

    Returns:
        (T,) float array. 1.0 if caregiver detected, 0.0 otherwise.
    """
    return _caregiver_present_mask(caregiver_3d).astype(np.float64)


def _compute_min_hand_body_distance(dist_matrix, torso_len, cg_present):
    """I2: Hand-body proximity — inverse of min caregiver-wrist-to-patient distance.

    Transformed: proximity = 1 / (1 + normalized_dist).
    Higher = closer contact. Bounded in [0, 1].

    Returns:
        (T,) float array in [0, 1].
    """
    # Caregiver wrists (indices 9, 10) to all patient joints
    wrist_dists = dist_matrix[:, [LEFT_WRIST, RIGHT_WRIST], :]  # (T, 2, 17)
    min_dist = np.min(wrist_dists, axis=(1, 2))  # (T,)
    # Replace inf with large value
    min_dist[np.isinf(min_dist)] = 1e6

    # Normalize by torso length and invert
    norm_dist = min_dist / (torso_len + 1e-8)
    result = 1.0 / (1.0 + norm_dist)

    result[~cg_present] = 0.0
    return result


def _compute_contact_point_count(dist_matrix, torso_len, cg_present, contact_thresh):
    """I3: Fraction of patient joints within contact range of any caregiver joint.

    Args:
        dist_matrix: (T, 17, 17) pairwise distances.
        contact_thresh: threshold in torso-length units.

    Returns:
        (T,) float array in [0, 1]. Normalized by 17 (total joints).
    """
    T = dist_matrix.shape[0]
    # Per-patient-joint minimum distance to any caregiver joint: (T, 17)
    min_dist_per_patient = np.min(dist_matrix, axis=1)  # min over caregiver joints

    # Threshold: per-frame, in absolute coordinates
    thresh_abs = contact_thresh * torso_len  # (T,)

    # Count patient joints in contact: min_dist < threshold
    in_contact = min_dist_per_patient < thresh_abs[:, None]  # (T, 17)
    count = in_contact.sum(axis=1).astype(np.float64)  # (T,)

    result = count / 17.0
    result[~cg_present] = 0.0
    return result


def _compute_contact_flags(dist_matrix, torso_len, cg_present, contact_thresh):
    """Compute per-frame binary contact flag (any joint pair in contact).

    Vectorized using the pre-computed distance matrix.

    Returns:
        (T,) boolean array.
    """
    T = dist_matrix.shape[0]
    # Minimum distance across all joint pairs per frame
    min_dist = np.min(dist_matrix.reshape(T, -1), axis=1)  # (T,)
    thresh_abs = contact_thresh * torso_len  # (T,)
    flags = (min_dist < thresh_abs) & cg_present
    return flags


def _compute_contact_duration_ratio(contact_flags):
    """I4: Running cumulative ratio of frames with any contact.

    Returns:
        (T,) float array in [0, 1].
    """
    T = len(contact_flags)
    if T == 0:
        return np.zeros(0, dtype=np.float64)

    cumsum = np.cumsum(contact_flags.astype(np.float64))
    frame_idx = np.arange(1, T + 1, dtype=np.float64)
    return cumsum / frame_idx


def _compute_velocity_correlation(caregiver_3d, child_3d, cg_present,
                                   torso_len, fps, window):
    """I5: Rolling Pearson correlation between caregiver hand speed and patient torso speed.

    High positive correlation = caregiver is DRIVING the movement.
    Near zero = independent movement or no interaction.

    Returns:
        (T,) float array in [-1, 1].
    """
    T = child_3d.shape[0]
    result = np.zeros(T, dtype=np.float64)

    if T < 3:
        return result

    # Caregiver hand velocity: mean of both wrists
    cg_hand = (caregiver_3d[:, LEFT_WRIST] + caregiver_3d[:, RIGHT_WRIST]) / 2
    cg_hand_vel = np.linalg.norm(np.gradient(cg_hand, axis=0), axis=1) * fps
    cg_hand_vel /= (torso_len + 1e-8)  # normalize

    # Patient torso velocity: hip center
    patient_hip = (child_3d[:, LEFT_HIP] + child_3d[:, RIGHT_HIP]) / 2
    patient_vel = np.linalg.norm(np.gradient(patient_hip, axis=0), axis=1) * fps
    patient_vel /= (torso_len + 1e-8)  # normalize

    # Zero out caregiver velocity when not present
    cg_hand_vel[~cg_present] = 0.0

    result = _rolling_correlation(cg_hand_vel, patient_vel, window)
    result[~cg_present] = 0.0

    return result


def _compute_movement_independence(child_3d, contact_flags, cg_present,
                                    torso_len, fps):
    """I6: Movement independence score — THE key L4/L5 feature.

    Ratio of patient's own movement during non-contact vs contact phases.
    Broadcast as a clip-level score to all frames.

    If patient moves at same speed regardless of contact -> independent (~1.0)
    If patient only moves when being contacted -> dependent (~0.0)

    L4 side rolling: ~0.7-1.0 (self-initiates)
    L5 side rolling: ~0.0-0.3 (caregiver drives rotation)

    Returns:
        (T,) float array in [0, 1].
    """
    T = child_3d.shape[0]
    result = np.ones(T, dtype=np.float64)  # default: fully independent

    if T < 2 or not cg_present.any():
        return result  # no caregiver = independent

    # Patient velocity magnitude (hip center)
    patient_hip = (child_3d[:, LEFT_HIP] + child_3d[:, RIGHT_HIP]) / 2
    vel = np.linalg.norm(np.gradient(patient_hip, axis=0), axis=1) * fps
    vel /= (torso_len + 1e-8)

    contact_mask = contact_flags & cg_present
    no_contact_mask = ~contact_flags & cg_present  # only compare when caregiver is around

    if contact_mask.sum() == 0 or no_contact_mask.sum() == 0:
        return result  # can't compute ratio without both phases

    vel_contact = vel[contact_mask].mean()
    vel_no_contact = vel[no_contact_mask].mean()

    if vel_contact < 1e-8 and vel_no_contact < 1e-8:
        return result  # no movement at all = independent (static)

    denominator = max(vel_contact, vel_no_contact)
    score = vel_no_contact / (denominator + 1e-8)
    score = np.clip(score, 0.0, 1.0)

    return np.full(T, score, dtype=np.float64)


def _compute_contact_body_region(caregiver_3d, child_3d, torso_len,
                                  cg_present, contact_thresh):
    """I7-I10: Which patient body regions are in contact with caregiver.

    Returns (T, 4): [head, trunk, upper_limb, lower_limb].
    Each channel is binary (1.0 if any joint in region is within threshold).
    """
    T = child_3d.shape[0]
    result = np.zeros((T, 4), dtype=np.float64)

    regions = [_REGION_HEAD, _REGION_TRUNK, _REGION_UPPER_LIMB, _REGION_LOWER_LIMB]

    for t in range(T):
        if not cg_present[t]:
            continue

        thresh_abs = contact_thresh * torso_len[t]

        # Active caregiver joints for this frame
        cg_active = []
        for j in range(17):
            if np.linalg.norm(caregiver_3d[t, j]) > 1e-6:
                cg_active.append(j)

        if not cg_active:
            continue

        for r_idx, region_joints in enumerate(regions):
            for pj in region_joints:
                if np.linalg.norm(child_3d[t, pj]) < 1e-6:
                    continue
                for cj in cg_active:
                    dist = np.linalg.norm(child_3d[t, pj] - caregiver_3d[t, cj])
                    if dist < thresh_abs:
                        result[t, r_idx] = 1.0
                        break
                if result[t, r_idx] > 0:
                    break  # region already flagged

    return result


# ── Main entry point ─────────────────────────────────────────────────────

def extract_interaction_features(child_3d, caregiver_3d, config=None):
    """Extract 10 per-frame interaction features from paired skeletons.

    All distance-based features are normalized by patient torso length to
    handle scale ambiguity from cv2.recoverPose.

    When no caregiver is present (all-zero skeleton), all features are zero.

    Args:
        child_3d: (T, 17, 3) float32, patient 3D skeleton.
        caregiver_3d: (T, 17, 3) float32, caregiver 3D skeleton.
            Zeros for absent caregiver.
        config: dict with optional keys:
            - 'fps': int (default 30)
            - 'contact_threshold_m': float (default 0.15, meters).
              Converted to torso-length-relative units internally.

    Returns:
        np.ndarray of shape (T, 10), dtype float32.
    """
    config = config or {}
    fps = config.get("fps", DEFAULT_FPS)
    # Read meters from config, convert to torso-length-relative units
    threshold_m = config.get("contact_threshold_m", 0.15)
    contact_thresh = threshold_m / _ASSUMED_TORSO_M

    T = child_3d.shape[0]
    zero_features = np.zeros((T, NUM_INTERACTION_FEATURES), dtype=np.float32)

    if T < 2:
        return zero_features

    # Check if child skeleton is valid
    if np.all(np.abs(child_3d) < 1e-6):
        logger.warning("All-zero child skeleton, returning zero interaction features")
        return zero_features

    # Check if caregiver is present at all
    cg_present = _caregiver_present_mask(caregiver_3d)
    if not cg_present.any():
        # No caregiver in any frame — all zeros (informative for L1/L2/L3)
        return zero_features

    # Patient torso length for normalization
    child_valid = _valid_frame_mask(child_3d)
    if not child_valid.any():
        return zero_features
    torso_len = _torso_length(child_3d, child_valid)

    # Pre-compute contact flags (used by multiple features)
    contact_flags = _compute_contact_flags(
        caregiver_3d, child_3d, torso_len, cg_present, contact_thresh
    )

    # Compute all 10 features
    f0 = _compute_caregiver_present(caregiver_3d)
    f1 = _compute_min_hand_body_distance(caregiver_3d, child_3d, torso_len, cg_present)
    f2 = _compute_contact_point_count(
        caregiver_3d, child_3d, torso_len, cg_present, contact_thresh
    )
    f3 = _compute_contact_duration_ratio(contact_flags)
    f4 = _compute_velocity_correlation(
        caregiver_3d, child_3d, cg_present, torso_len, fps, DEFAULT_CORR_WINDOW
    )
    f5 = _compute_movement_independence(
        child_3d, contact_flags, cg_present, torso_len, fps
    )
    f6_f9 = _compute_contact_body_region(
        caregiver_3d, child_3d, torso_len, cg_present, contact_thresh
    )

    # Stack: (T, 10)
    features = np.column_stack([
        f0,           # I1: caregiver_present
        f1,           # I2: min_hand_body_distance (proximity)
        f2,           # I3: contact_point_count
        f3,           # I4: contact_duration_ratio
        f4,           # I5: velocity_correlation
        f5,           # I6: movement_independence_score
        f6_f9[:, 0],  # I7: contact_region_head
        f6_f9[:, 1],  # I8: contact_region_trunk
        f6_f9[:, 2],  # I9: contact_region_upper_limb
        f6_f9[:, 3],  # I10: contact_region_lower_limb
    ])

    # NaN safety
    features = np.nan_to_num(features, nan=0.0)
    return features.astype(np.float32)


def get_interaction_feature_names():
    """Return human-readable names for the 10 interaction features."""
    return [
        "caregiver_present",
        "min_hand_body_distance",
        "contact_point_count",
        "contact_duration_ratio",
        "velocity_correlation",
        "movement_independence_score",
        "contact_region_head",
        "contact_region_trunk",
        "contact_region_upper_limb",
        "contact_region_lower_limb",
    ]
