"""Layer 1: Device-proxy skeletal features.

Computes 15 per-frame features from the patient's 3D skeleton that serve
as proxy indicators for assistive device usage and movement quality.

Features (15D per frame):
  [0]  wfi_left                  - Wrist Fixation Index, left hand
  [1]  wfi_right                 - Wrist Fixation Index, right hand
  [2]  arm_swing_amplitude_left  - Arm swing in anteroposterior direction
  [3]  arm_swing_amplitude_right - Arm swing in anteroposterior direction
  [4]  ankle_rom_left            - Ankle range of motion (AFO detection)
  [5]  ankle_rom_right           - Ankle range of motion (AFO detection)
  [6]  upper_limb_freedom_score  - Composite arm constraint score
  [7]  com_sway                  - Center of mass lateral sway
  [8]  com_smoothness            - CoM jerk-based smoothness
  [9]  support_point_convergence - Acrylic stand usage indicator
  [10] gait_symmetry_index       - Left/right step asymmetry
  [11] wrist_height_left         - Wrist height relative to hip
  [12] wrist_height_right        - Wrist height relative to hip
  [13] bilateral_wrist_distance  - Distance between wrists
  [14] torso_vertical_velocity   - Hip center vertical velocity

All distance-based features are normalized by torso length to handle
the scale ambiguity from cv2.recoverPose. Angle-based features are
inherently scale-invariant.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

NUM_SKELETON_FEATURES = 15

# COCO 17-joint keypoint indices
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# Default parameters
DEFAULT_FPS = 30
DEFAULT_WFI_WINDOW = 15        # 0.5s at 30fps
DEFAULT_ASA_WINDOW = 60        # 2.0s at 30fps
DEFAULT_ROM_WINDOW = 30        # 1.0s at 30fps
DEFAULT_COM_WINDOW = 30        # 1.0s at 30fps
DEFAULT_SPC_STD_THRESH = 0.05  # relative to torso length
DEFAULT_SPC_GRAD_THRESH = 0.001
DEFAULT_WFI_MAX_STD = 1.0      # 1 torso-length of std is max


# ── Helpers ──────────────────────────────────────────────────────────────

def _valid_frame_mask(skeleton_3d):
    """Detect frames with enough non-zero joints for feature computation.

    A frame is valid if all four torso joints (both hips, both shoulders)
    are non-zero.

    Args:
        skeleton_3d: (T, 17, 3).

    Returns:
        (T,) boolean mask.
    """
    torso_joints = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    valid = np.ones(skeleton_3d.shape[0], dtype=bool)
    for j in torso_joints:
        valid &= np.linalg.norm(skeleton_3d[:, j], axis=1) > 1e-6
    return valid


def _torso_length(skeleton_3d, valid_mask):
    """Compute per-frame torso length for scale normalization.

    torso_length = norm(shoulder_center - hip_center)

    Invalid frames are filled with the median of valid frames.

    Args:
        skeleton_3d: (T, 17, 3).
        valid_mask: (T,) boolean.

    Returns:
        (T,) float64. Guaranteed > 0 (fallback to 1.0).
    """
    shoulder_c = (skeleton_3d[:, LEFT_SHOULDER] + skeleton_3d[:, RIGHT_SHOULDER]) / 2
    hip_c = (skeleton_3d[:, LEFT_HIP] + skeleton_3d[:, RIGHT_HIP]) / 2
    tlen = np.linalg.norm(shoulder_c - hip_c, axis=1)

    if valid_mask.any():
        median_len = np.median(tlen[valid_mask])
        if median_len < 1e-6:
            median_len = 1.0
    else:
        median_len = 1.0

    tlen[~valid_mask] = median_len
    tlen[tlen < 1e-6] = median_len
    return tlen


def _rolling_std(signal, window):
    """Rolling standard deviation with graceful edge handling.

    Args:
        signal: (T,) or (T, D) array.
        window: window size in frames.

    Returns:
        Same shape as signal.
    """
    T = signal.shape[0]
    if T == 0:
        return signal.copy()

    result = np.zeros_like(signal, dtype=np.float64)
    for t in range(T):
        start = max(0, t - window + 1)
        seg = signal[start:t + 1]
        if len(seg) > 1:
            result[t] = np.std(seg, axis=0)
    return result


def _rolling_range(signal, window):
    """Rolling peak-to-peak (max - min) with edge handling.

    Args:
        signal: (T,) array.
        window: window size in frames.

    Returns:
        (T,) array.
    """
    T = len(signal)
    if T == 0:
        return signal.copy()

    result = np.zeros(T, dtype=np.float64)
    for t in range(T):
        start = max(0, t - window + 1)
        seg = signal[start:t + 1]
        result[t] = seg.max() - seg.min()
    return result


def _estimate_forward_direction(skeleton_3d, valid_mask):
    """Estimate anteroposterior (forward) axis per frame.

    Forward = cross(up, lateral) where:
      up = shoulder_center - hip_center (body vertical)
      lateral = right_shoulder - left_shoulder

    Args:
        skeleton_3d: (T, 17, 3).
        valid_mask: (T,) boolean.

    Returns:
        (T, 3) unit vectors. Invalid frames filled from nearest valid.
    """
    T = skeleton_3d.shape[0]
    shoulder_c = (skeleton_3d[:, LEFT_SHOULDER] + skeleton_3d[:, RIGHT_SHOULDER]) / 2
    hip_c = (skeleton_3d[:, LEFT_HIP] + skeleton_3d[:, RIGHT_HIP]) / 2
    up = shoulder_c - hip_c
    lateral = skeleton_3d[:, RIGHT_SHOULDER] - skeleton_3d[:, LEFT_SHOULDER]

    forward = np.cross(up, lateral)
    norms = np.linalg.norm(forward, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    forward = forward / norms

    # Fill invalid frames with nearest valid
    if valid_mask.any():
        valid_idx = np.where(valid_mask)[0]
        for t in range(T):
            if not valid_mask[t]:
                nearest = valid_idx[np.argmin(np.abs(valid_idx - t))]
                forward[t] = forward[nearest]

    return forward


def _project_onto_axis(positions, axis):
    """Dot product projection of positions onto axis vectors.

    Args:
        positions: (T, 3).
        axis: (T, 3) unit vectors.

    Returns:
        (T,) scalar projections.
    """
    return np.sum(positions * axis, axis=1)


def _approximate_toe_position(skeleton_3d):
    """Approximate toe positions from ankle and knee.

    COCO 17-joint has no toe keypoint. Estimate as:
      toe = ankle + 0.5 * (ankle - knee)

    Args:
        skeleton_3d: (T, 17, 3).

    Returns:
        left_toe: (T, 3), right_toe: (T, 3).
    """
    l_knee = skeleton_3d[:, LEFT_KNEE]
    l_ankle = skeleton_3d[:, LEFT_ANKLE]
    r_knee = skeleton_3d[:, RIGHT_KNEE]
    r_ankle = skeleton_3d[:, RIGHT_ANKLE]

    left_toe = l_ankle + 0.5 * (l_ankle - l_knee)
    right_toe = r_ankle + 0.5 * (r_ankle - r_knee)
    return left_toe, right_toe


def _joint_angle(p1, p_vertex, p2):
    """Compute angle at vertex formed by segments p1-vertex and vertex-p2.

    Args:
        p1, p_vertex, p2: (T, 3) arrays.

    Returns:
        (T,) angles in radians.
    """
    v1 = p1 - p_vertex
    v2 = p2 - p_vertex
    dot = np.sum(v1 * v2, axis=1)
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    cos_angle = dot / (n1 * n2 + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def _head_position(skeleton_3d):
    """Compute head position from available head joints.

    Uses average of nose, eyes, ears. Falls back to shoulder center.

    Args:
        skeleton_3d: (T, 17, 3).

    Returns:
        (T, 3) head positions.
    """
    head_joints = [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]
    T = skeleton_3d.shape[0]
    head = np.zeros((T, 3), dtype=skeleton_3d.dtype)

    for t in range(T):
        pts = []
        for j in head_joints:
            if np.linalg.norm(skeleton_3d[t, j]) > 1e-6:
                pts.append(skeleton_3d[t, j])
        if pts:
            head[t] = np.mean(pts, axis=0)
        else:
            # Fallback to shoulder center
            head[t] = (skeleton_3d[t, LEFT_SHOULDER] + skeleton_3d[t, RIGHT_SHOULDER]) / 2

    return head


# ── Feature computation ──────────────────────────────────────────────────

def _compute_wfi(skeleton_3d, torso_len, valid, window, max_std):
    """F1-F2: Wrist Fixation Index, left and right.

    Measures wrist stability relative to the body frame (hip center).
    High WFI (~1) = wrist is fixed relative to body (gripping walker/stand).
    Low WFI (~0) = wrist moves freely relative to body.
    """
    T = skeleton_3d.shape[0]
    hip_c = (skeleton_3d[:, LEFT_HIP] + skeleton_3d[:, RIGHT_HIP]) / 2
    result = np.zeros((T, 2), dtype=np.float64)

    for i, joint in enumerate([LEFT_WRIST, RIGHT_WRIST]):
        # Wrist position relative to hip center (body-relative)
        wrist_rel = skeleton_3d[:, joint] - hip_c
        # Scale-normalize by torso length
        wrist_rel_norm = wrist_rel / torso_len[:, None]
        # Rolling std of body-relative normalized position
        rstd = _rolling_std(wrist_rel_norm, window)  # (T, 3)
        std_mag = np.linalg.norm(rstd, axis=1)       # (T,)
        wfi = 1.0 - np.clip(std_mag / max_std, 0, 1)
        result[:, i] = wfi

    result[~valid] = 0.0
    return result


def _compute_asa(skeleton_3d, forward_dir, torso_len, valid, window):
    """F3-F4: Arm Swing Amplitude, left and right.

    Low ASA = walker-assisted walking. High = free gait.
    """
    T = skeleton_3d.shape[0]
    result = np.zeros((T, 2), dtype=np.float64)

    for i, joint in enumerate([LEFT_WRIST, RIGHT_WRIST]):
        wrist = skeleton_3d[:, joint]
        ap = _project_onto_axis(wrist, forward_dir)
        ap_norm = ap / torso_len
        amp = _rolling_range(ap_norm, window)
        # Normalize: 2 torso-lengths of swing is extreme
        result[:, i] = np.clip(amp / 2.0, 0, 1)

    result[~valid] = 0.0
    return result


def _compute_ankle_rom(skeleton_3d, valid, window):
    """F5-F6: Ankle Range of Motion, left and right.

    Low ROM = AFO restricting joint. Scale-invariant (angle).
    """
    T = skeleton_3d.shape[0]
    result = np.zeros((T, 2), dtype=np.float64)
    left_toe, right_toe = _approximate_toe_position(skeleton_3d)

    for i, (knee_j, ankle_j, toe) in enumerate([
        (LEFT_KNEE, LEFT_ANKLE, left_toe),
        (RIGHT_KNEE, RIGHT_ANKLE, right_toe),
    ]):
        knee = skeleton_3d[:, knee_j]
        ankle = skeleton_3d[:, ankle_j]
        angles = _joint_angle(knee, ankle, toe)
        rom = _rolling_range(angles, window)
        # Normalize: pi/3 (60 deg) is typical max ROM
        result[:, i] = np.clip(rom / (np.pi / 3), 0, 1)

    result[~valid] = 0.0
    return result


def _compute_ulfs(skeleton_3d, torso_len, valid, fps):
    """F7: Upper Limb Freedom Score.

    Composite: wrist_velocity × shoulder_ROM × bilateral_symmetry.
    """
    T = skeleton_3d.shape[0]
    result = np.zeros(T, dtype=np.float64)

    if T < 2:
        return result

    # Wrist velocities (normalized by torso length and fps)
    l_vel = np.linalg.norm(np.diff(skeleton_3d[:, LEFT_WRIST], axis=0), axis=1)
    r_vel = np.linalg.norm(np.diff(skeleton_3d[:, RIGHT_WRIST], axis=0), axis=1)
    l_vel_norm = l_vel / torso_len[1:]
    r_vel_norm = r_vel / torso_len[1:]

    # Mean velocity (clipped, normalized)
    mean_vel = (l_vel_norm + r_vel_norm) / 2
    mean_vel_score = np.clip(mean_vel / 0.3, 0, 1)  # 0.3 torso-len/frame is fast

    # Shoulder ROM: angle(elbow, shoulder, hip) rolling range
    l_angle = _joint_angle(
        skeleton_3d[1:, LEFT_ELBOW], skeleton_3d[1:, LEFT_SHOULDER],
        skeleton_3d[1:, LEFT_HIP],
    )
    r_angle = _joint_angle(
        skeleton_3d[1:, RIGHT_ELBOW], skeleton_3d[1:, RIGHT_SHOULDER],
        skeleton_3d[1:, RIGHT_HIP],
    )
    shoulder_rom = (_rolling_range(l_angle, 60) + _rolling_range(r_angle, 60)) / 2
    shoulder_rom_score = np.clip(shoulder_rom / np.pi, 0, 1)

    # Bilateral symmetry
    symmetry = 1.0 - np.abs(l_vel_norm - r_vel_norm) / (l_vel_norm + r_vel_norm + 1e-8)
    symmetry = np.clip(symmetry, 0, 1)

    # Composite (T-1,)
    ulfs = mean_vel_score * shoulder_rom_score * symmetry

    # Pad to (T,)
    result[1:] = ulfs
    result[0] = result[1] if T > 1 else 0.0
    result[~valid] = 0.0
    return result


def _compute_com_sway(skeleton_3d, torso_len, valid, window):
    """F8: Center of Mass lateral sway."""
    T = skeleton_3d.shape[0]
    shoulder_c = (skeleton_3d[:, LEFT_SHOULDER] + skeleton_3d[:, RIGHT_SHOULDER]) / 2
    hip_c = (skeleton_3d[:, LEFT_HIP] + skeleton_3d[:, RIGHT_HIP]) / 2
    head = _head_position(skeleton_3d)
    com = 0.4 * hip_c + 0.4 * shoulder_c + 0.2 * head

    # Lateral axis
    lateral = skeleton_3d[:, RIGHT_SHOULDER] - skeleton_3d[:, LEFT_SHOULDER]
    lat_norm = np.linalg.norm(lateral, axis=1, keepdims=True)
    lat_norm[lat_norm < 1e-8] = 1.0
    lateral_dir = lateral / lat_norm

    # Project CoM onto lateral axis, normalize by torso length
    lateral_com = _project_onto_axis(com, lateral_dir)
    lateral_com_norm = lateral_com / torso_len

    # Rolling variance
    sway_std = _rolling_std(lateral_com_norm, window)
    sway = sway_std ** 2
    # Normalize: variance of 0.1 is quite large
    result = np.clip(sway / 0.1, 0, 1)
    result[~valid] = 0.0
    return result


def _compute_com_smoothness(skeleton_3d, torso_len, valid, fps):
    """F9: CoM smoothness via jerk (third derivative)."""
    T = skeleton_3d.shape[0]
    shoulder_c = (skeleton_3d[:, LEFT_SHOULDER] + skeleton_3d[:, RIGHT_SHOULDER]) / 2
    hip_c = (skeleton_3d[:, LEFT_HIP] + skeleton_3d[:, RIGHT_HIP]) / 2
    head = _head_position(skeleton_3d)
    com = 0.4 * hip_c + 0.4 * shoulder_c + 0.2 * head

    # Normalize by torso length
    com_norm = com / torso_len[:, None]

    # Jerk = third derivative
    vel = np.gradient(com_norm, axis=0) * fps
    acc = np.gradient(vel, axis=0) * fps
    jerk = np.gradient(acc, axis=0) * fps
    jerk_mag = np.linalg.norm(jerk, axis=1)

    result = 1.0 / (1.0 + jerk_mag)
    result[~valid] = 0.0
    return result


def _compute_spc(skeleton_3d, torso_len, valid, fps, std_thresh, grad_thresh):
    """F10: Support Point Convergence (acrylic stand usage)."""
    T = skeleton_3d.shape[0]
    shoulder_c = (skeleton_3d[:, LEFT_SHOULDER] + skeleton_3d[:, RIGHT_SHOULDER]) / 2
    hip_c = (skeleton_3d[:, LEFT_HIP] + skeleton_3d[:, RIGHT_HIP]) / 2

    # Body vertical axis
    up = shoulder_c - hip_c
    up_norm = np.linalg.norm(up, axis=1, keepdims=True)
    up_norm[up_norm < 1e-8] = 1.0
    up_dir = up / up_norm

    # Wrist mean height relative to hip (projected on body vertical)
    wrist_mean = (skeleton_3d[:, LEFT_WRIST] + skeleton_3d[:, RIGHT_WRIST]) / 2
    wrist_rel = wrist_mean - hip_c
    wrist_height = _project_onto_axis(wrist_rel, up_dir)
    wrist_height_norm = wrist_height / torso_len

    # Wrist height stability
    wrist_stability = _rolling_std(wrist_height_norm, 15)

    # Hip rising: gradient of hip vertical position
    hip_vertical = _project_onto_axis(hip_c, up_dir)
    hip_grad = np.gradient(hip_vertical) * fps
    hip_grad_norm = hip_grad / torso_len

    # SPC = stable wrists AND rising hip
    result = np.where(
        (wrist_stability < std_thresh) & (hip_grad_norm > grad_thresh),
        1.0, 0.0,
    )
    result[~valid] = 0.0
    return result


def _compute_gsi(skeleton_3d, torso_len, valid):
    """F11: Gait Symmetry Index."""
    T = skeleton_3d.shape[0]
    result = np.zeros(T, dtype=np.float64)

    if T < 2:
        return result

    # Ankle displacement per frame, normalized
    l_disp = np.linalg.norm(np.diff(skeleton_3d[:, LEFT_ANKLE], axis=0), axis=1)
    r_disp = np.linalg.norm(np.diff(skeleton_3d[:, RIGHT_ANKLE], axis=0), axis=1)
    l_norm = l_disp / torso_len[1:]
    r_norm = r_disp / torso_len[1:]

    # Rolling sum over 30 frames as step-length proxy
    window = 30
    l_sum = np.convolve(l_norm, np.ones(window), mode="same")
    r_sum = np.convolve(r_norm, np.ones(window), mode="same")

    mean_step = (l_sum + r_sum) / 2 + 1e-8
    gsi = np.abs(l_sum - r_sum) / mean_step

    result[1:] = np.clip(gsi, 0, 1)
    result[0] = result[1] if T > 1 else 0.0
    result[~valid] = 0.0
    return result


def _compute_wrist_height(skeleton_3d, torso_len, valid):
    """F12-F13: Wrist height relative to hip, left and right."""
    T = skeleton_3d.shape[0]
    shoulder_c = (skeleton_3d[:, LEFT_SHOULDER] + skeleton_3d[:, RIGHT_SHOULDER]) / 2
    hip_c = (skeleton_3d[:, LEFT_HIP] + skeleton_3d[:, RIGHT_HIP]) / 2

    up = shoulder_c - hip_c
    up_norm = np.linalg.norm(up, axis=1, keepdims=True)
    up_norm[up_norm < 1e-8] = 1.0
    up_dir = up / up_norm

    result = np.zeros((T, 2), dtype=np.float64)
    for i, joint in enumerate([LEFT_WRIST, RIGHT_WRIST]):
        wrist_rel = skeleton_3d[:, joint] - hip_c
        height = _project_onto_axis(wrist_rel, up_dir)
        result[:, i] = np.clip(height / torso_len, -2.0, 2.0)

    result[~valid] = 0.0
    return result


def _compute_bilateral_wrist_distance(skeleton_3d, torso_len, valid):
    """F14: Distance between left and right wrists."""
    dist = np.linalg.norm(
        skeleton_3d[:, LEFT_WRIST] - skeleton_3d[:, RIGHT_WRIST], axis=1,
    )
    result = np.clip(dist / torso_len / 3.0, 0, 1)
    result[~valid] = 0.0
    return result


def _compute_torso_velocity(skeleton_3d, torso_len, valid, fps):
    """F15: Torso vertical velocity."""
    T = skeleton_3d.shape[0]
    shoulder_c = (skeleton_3d[:, LEFT_SHOULDER] + skeleton_3d[:, RIGHT_SHOULDER]) / 2
    hip_c = (skeleton_3d[:, LEFT_HIP] + skeleton_3d[:, RIGHT_HIP]) / 2

    up = shoulder_c - hip_c
    up_norm = np.linalg.norm(up, axis=1, keepdims=True)
    up_norm[up_norm < 1e-8] = 1.0
    up_dir = up / up_norm

    hip_vertical = _project_onto_axis(hip_c, up_dir)
    hip_vel = np.gradient(hip_vertical) * fps
    result = np.clip(hip_vel / torso_len / 2.0, -1.0, 1.0)
    result[~valid] = 0.0
    return result


# ── Main entry point ─────────────────────────────────────────────────────

def extract_skeleton_features(skeleton_3d, config=None):
    """Extract 15 per-frame skeleton features from a 3D skeleton sequence.

    All distance-based features are normalized by torso length to handle
    scale ambiguity from cv2.recoverPose.

    Args:
        skeleton_3d: (T, 17, 3) float32, xyz in world coordinates.
            Zeros for missing/failed joints.
        config: dict with optional keys:
            - 'wfi_window_size': int (default 15)
            - 'fps': int (default 30)

    Returns:
        np.ndarray of shape (T, 15), dtype float32.
        Returns all zeros for degenerate input.
    """
    config = config or {}
    fps = config.get("fps", DEFAULT_FPS)
    wfi_window = config.get("wfi_window_size", DEFAULT_WFI_WINDOW)

    T = skeleton_3d.shape[0]
    zero_features = np.zeros((T, NUM_SKELETON_FEATURES), dtype=np.float32)

    if T < 2:
        return zero_features

    # Check for degenerate input
    if np.all(np.abs(skeleton_3d) < 1e-6):
        logger.warning("All-zero skeleton, returning zero features")
        return zero_features

    # Shared intermediates
    valid = _valid_frame_mask(skeleton_3d)
    if not valid.any():
        logger.warning("No valid frames in skeleton, returning zero features")
        return zero_features

    torso_len = _torso_length(skeleton_3d, valid)
    forward_dir = _estimate_forward_direction(skeleton_3d, valid)

    # Compute all features
    wfi = _compute_wfi(skeleton_3d, torso_len, valid, wfi_window, DEFAULT_WFI_MAX_STD)
    asa = _compute_asa(skeleton_3d, forward_dir, torso_len, valid, DEFAULT_ASA_WINDOW)
    arom = _compute_ankle_rom(skeleton_3d, valid, DEFAULT_ROM_WINDOW)
    ulfs = _compute_ulfs(skeleton_3d, torso_len, valid, fps)
    sway = _compute_com_sway(skeleton_3d, torso_len, valid, DEFAULT_COM_WINDOW)
    smooth = _compute_com_smoothness(skeleton_3d, torso_len, valid, fps)
    spc = _compute_spc(skeleton_3d, torso_len, valid, fps,
                       DEFAULT_SPC_STD_THRESH, DEFAULT_SPC_GRAD_THRESH)
    gsi = _compute_gsi(skeleton_3d, torso_len, valid)
    wh = _compute_wrist_height(skeleton_3d, torso_len, valid)
    bwd = _compute_bilateral_wrist_distance(skeleton_3d, torso_len, valid)
    tvv = _compute_torso_velocity(skeleton_3d, torso_len, valid, fps)

    # Stack: (T, 15)
    features = np.column_stack([
        wfi[:, 0],    # F1: wfi_left
        wfi[:, 1],    # F2: wfi_right
        asa[:, 0],    # F3: arm_swing_left
        asa[:, 1],    # F4: arm_swing_right
        arom[:, 0],   # F5: ankle_rom_left
        arom[:, 1],   # F6: ankle_rom_right
        ulfs,         # F7: upper_limb_freedom
        sway,         # F8: com_sway
        smooth,       # F9: com_smoothness
        spc,          # F10: support_point_convergence
        gsi,          # F11: gait_symmetry_index
        wh[:, 0],     # F12: wrist_height_left
        wh[:, 1],     # F13: wrist_height_right
        bwd,          # F14: bilateral_wrist_distance
        tvv,          # F15: torso_vertical_velocity
    ])

    # Safety: replace any remaining NaN
    features = np.nan_to_num(features, nan=0.0)

    return features.astype(np.float32)


def get_feature_names():
    """Return human-readable names for the 15 skeleton features."""
    return [
        "wfi_left",
        "wfi_right",
        "arm_swing_amplitude_left",
        "arm_swing_amplitude_right",
        "ankle_rom_left",
        "ankle_rom_right",
        "upper_limb_freedom_score",
        "com_sway",
        "com_smoothness",
        "support_point_convergence",
        "gait_symmetry_index",
        "wrist_height_left",
        "wrist_height_right",
        "bilateral_wrist_distance",
        "torso_vertical_velocity",
    ]
