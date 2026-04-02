"""Layer 1 extended: GMFCS-E&R movement quality descriptors.

Extracts clinically-meaningful per-clip movement quality features aligned
with GMFCS-E&R assessment criteria (Report Section 15.4) for each of the
5 movement types.

Features are per-clip summaries (not per-frame), padded to a uniform 6D
output regardless of movement type.

Movement types and feature counts:
  walk (w):             5 features [cadence, gait_symmetry, trunk_lateral_sway,
                                    step_width, head_stability]
  sit-to-stand (c_s):   6 features [transition_duration, trunk_anterior_tilt,
                                    hand_ground_contact, com_jerk_smoothness,
                                    knee_extension_symmetry, post_transition_sway]
  crawl (cr):           6 features [reciprocal_pattern_index, trunk_elevation,
                                    crawl_velocity, cycle_regularity,
                                    trunk_roll_amplitude, upper_lower_limb_ratio]
  side-rolling (sr):    5 features [segmental_rotation_ratio, rolling_velocity,
                                    bilateral_symmetry, inter_roll_recovery,
                                    arm_rom_during_roll]
  stand-to-sit (s_c):   4 features [descent_velocity_consistency, controlled_deceleration,
                                    impact_softness, hand_support_frequency]
"""

import logging

import numpy as np

from src.features.skeleton_features import (
    DEFAULT_FPS,
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    _estimate_forward_direction,
    _head_position,
    _joint_angle,
    _project_onto_axis,
    _torso_length,
    _valid_frame_mask,
)

logger = logging.getLogger(__name__)

NUM_MOVEMENT_QUALITY_FEATURES = 6
MOVEMENT_CLASSES = {"w", "cr", "c_s", "s_c", "sr"}


# ── Shared helpers ──────────────────────────────────────────────────────

def _hip_center(skeleton_3d):
    """Midpoint of both hips. (T, 3)."""
    return (skeleton_3d[:, LEFT_HIP] + skeleton_3d[:, RIGHT_HIP]) / 2


def _shoulder_center(skeleton_3d):
    """Midpoint of both shoulders. (T, 3)."""
    return (skeleton_3d[:, LEFT_SHOULDER] + skeleton_3d[:, RIGHT_SHOULDER]) / 2


def _lateral_direction(skeleton_3d, valid):
    """Unit vector from left to right shoulder per frame. (T, 3)."""
    lat = skeleton_3d[:, RIGHT_SHOULDER] - skeleton_3d[:, LEFT_SHOULDER]
    norms = np.linalg.norm(lat, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    lat_dir = lat / norms
    # Fill invalid frames from nearest valid
    if valid.any():
        valid_idx = np.where(valid)[0]
        for t in range(len(lat_dir)):
            if not valid[t]:
                nearest = valid_idx[np.argmin(np.abs(valid_idx - t))]
                lat_dir[t] = lat_dir[nearest]
    return lat_dir


def _body_vertical(skeleton_3d, valid):
    """Normalized shoulder_center - hip_center per frame. (T, 3)."""
    up = _shoulder_center(skeleton_3d) - _hip_center(skeleton_3d)
    norms = np.linalg.norm(up, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    up_dir = up / norms
    if valid.any():
        valid_idx = np.where(valid)[0]
        for t in range(len(up_dir)):
            if not valid[t]:
                nearest = valid_idx[np.argmin(np.abs(valid_idx - t))]
                up_dir[t] = up_dir[nearest]
    return up_dir


def _pelvis_height_signal(skeleton_3d, valid, torso_len):
    """Hip center projected onto body vertical, normalized by torso length. (T,)."""
    hip_c = _hip_center(skeleton_3d)
    up_dir = _body_vertical(skeleton_3d, valid)
    height = _project_onto_axis(hip_c, up_dir)
    height_norm = height / torso_len
    return height_norm


def _detect_peaks(signal, min_distance=5):
    """Simple peak detection via local maxima.

    Args:
        signal: (N,) array.
        min_distance: minimum frames between peaks.

    Returns:
        Array of peak indices.
    """
    if len(signal) < 3:
        return np.array([], dtype=int)

    # Find all local maxima
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            peaks.append(i)

    if not peaks:
        return np.array([], dtype=int)

    # Enforce min_distance: keep tallest peak in each window
    peaks = np.array(peaks)
    filtered = [peaks[0]]
    for p in peaks[1:]:
        if p - filtered[-1] >= min_distance:
            filtered.append(p)
        elif signal[p] > signal[filtered[-1]]:
            filtered[-1] = p
    return np.array(filtered, dtype=int)


def _zero_crossings(signal):
    """Find indices where signal crosses zero (sign changes).

    Returns:
        Array of indices immediately before each zero crossing.
    """
    signs = np.sign(signal)
    # Replace exact zeros with sign of previous non-zero
    for i in range(len(signs)):
        if signs[i] == 0 and i > 0:
            signs[i] = signs[i - 1]
    diffs = np.diff(signs)
    return np.where(diffs != 0)[0]


def _com_position(skeleton_3d):
    """Approximate center of mass. (T, 3).

    CoM = 0.4 * hip_center + 0.4 * shoulder_center + 0.2 * head.
    """
    hip_c = _hip_center(skeleton_3d)
    shoulder_c = _shoulder_center(skeleton_3d)
    head = _head_position(skeleton_3d)
    return 0.4 * hip_c + 0.4 * shoulder_c + 0.2 * head


# ── Walk Quality Features ───────────────────────────────────────────────

class WalkQualityFeatures:
    """Walk (gait) quality features — 5 per clip (Report 15.4.1).

    Discriminates L1 vs L2 (subtle gait quality) and L2 vs L3 (device-assisted).
    """

    NUM_FEATURES = 5

    @staticmethod
    def compute(skeleton_3d, valid, torso_len, forward_dir, lat_dir, fps):
        """Compute 5 walk quality features.

        Returns:
            (5,) float64 array.
        """
        T = skeleton_3d.shape[0]
        features = np.zeros(5, dtype=np.float64)

        # ── W1: Cadence (stride cycles per second) ──
        l_ankle = skeleton_3d[:, LEFT_ANKLE]
        ankle_fwd = _project_onto_axis(l_ankle, forward_dir)
        # Detrend: subtract linear fit to remove drift
        t_axis = np.arange(T, dtype=np.float64)
        if T > 1:
            coeffs = np.polyfit(t_axis, ankle_fwd, 1)
            ankle_detrended = ankle_fwd - np.polyval(coeffs, t_axis)
        else:
            ankle_detrended = ankle_fwd

        # Autocorrelation-based cadence detection
        sig = ankle_detrended - np.mean(ankle_detrended)
        norm_factor = np.sum(sig ** 2)
        if norm_factor > 1e-10 and T > int(fps * 0.6):
            autocorr = np.correlate(sig, sig, mode="full")
            autocorr = autocorr[T - 1:]  # positive lags only
            autocorr = autocorr / (norm_factor + 1e-10)

            min_lag = max(int(fps * 0.3), 1)
            max_lag = min(int(fps * 3.0), T - 1)
            if max_lag > min_lag:
                search = autocorr[min_lag:max_lag]
                peak_idx = np.argmax(search) + min_lag
                if autocorr[peak_idx] > 0.1:  # meaningful periodicity
                    stride_period = peak_idx / fps
                    cadence = 1.0 / stride_period if stride_period > 0 else 0.0
                    features[0] = np.clip(cadence / 2.5, 0, 1)

        # ── W2: Gait symmetry (L/R swing duration ratio) ──
        l_ankle_fwd = _project_onto_axis(skeleton_3d[:, LEFT_ANKLE], forward_dir)
        r_ankle_fwd = _project_onto_axis(skeleton_3d[:, RIGHT_ANKLE], forward_dir)
        # Velocity-based: forward velocity of each ankle
        l_vel = np.gradient(l_ankle_fwd)
        r_vel = np.gradient(r_ankle_fwd)
        # Swing phases: forward velocity > 0
        l_crossings = _zero_crossings(l_vel)
        r_crossings = _zero_crossings(r_vel)

        if len(l_crossings) >= 2 and len(r_crossings) >= 2:
            l_durations = np.diff(l_crossings).astype(float)
            r_durations = np.diff(r_crossings).astype(float)
            l_mean = np.mean(l_durations)
            r_mean = np.mean(r_durations)
            if max(l_mean, r_mean) > 0:
                features[1] = min(l_mean, r_mean) / (max(l_mean, r_mean) + 1e-8)

        # ── W3: Trunk lateral sway ──
        hip_c = _hip_center(skeleton_3d)
        hip_lateral = _project_onto_axis(hip_c, lat_dir)
        hip_lat_norm = hip_lateral / torso_len
        sway_range = np.ptp(hip_lat_norm[valid]) if valid.any() else 0.0
        features[2] = np.clip(sway_range / 0.5, 0, 1)

        # ── W4: Step width (bilateral ankle lateral distance) ──
        ankle_diff = skeleton_3d[:, LEFT_ANKLE] - skeleton_3d[:, RIGHT_ANKLE]
        lateral_dist = np.abs(_project_onto_axis(ankle_diff, lat_dir))
        lateral_dist_norm = lateral_dist / torso_len
        if valid.any():
            features[3] = np.clip(np.median(lateral_dist_norm[valid]) / 2.0, 0, 1)

        # ── W5: Head stability ──
        head = _head_position(skeleton_3d)
        up_dir = _body_vertical(skeleton_3d, valid)
        head_vert = _project_onto_axis(head, up_dir) / torso_len
        head_lat = _project_onto_axis(head, lat_dir) / torso_len
        if valid.sum() > 1:
            std_v = np.std(head_vert[valid])
            std_l = np.std(head_lat[valid])
            instability = np.sqrt(std_v ** 2 + std_l ** 2)
            features[4] = 1.0 - np.clip(instability / 0.3, 0, 1)

        return features


# ── Sit-to-Stand Quality Features ───────────────────────────────────────

class SitToStandQualityFeatures:
    """Seated-to-standing transition features — 6 per clip (Report 15.4.2).

    Supplementary discriminative power across all classification stages.
    """

    NUM_FEATURES = 6

    @staticmethod
    def compute(skeleton_3d, valid, torso_len, forward_dir, lat_dir, fps):
        """Compute 6 sit-to-stand quality features.

        Returns:
            (6,) float64 array.
        """
        T = skeleton_3d.shape[0]
        features = np.zeros(6, dtype=np.float64)

        # Pelvis height signal for transition detection
        hip_c = _hip_center(skeleton_3d)
        up_dir = _body_vertical(skeleton_3d, valid)
        hip_height_norm = _pelvis_height_signal(skeleton_3d, valid, torso_len)

        # Detect transition boundaries
        n_start = max(1, int(T * 0.1))
        n_end = max(1, int(T * 0.1))
        h_start = np.median(hip_height_norm[:n_start])
        h_end = np.median(hip_height_norm[-n_end:])

        if h_end <= h_start + 0.05:
            # No clear sit-to-stand transition detected
            return features

        # Rising phase indices
        threshold_low = h_start + 0.1 * (h_end - h_start)
        threshold_high = h_start + 0.8 * (h_end - h_start)
        below_low = np.where(hip_height_norm <= threshold_low)[0]
        above_high = np.where(hip_height_norm >= threshold_high)[0]

        t_start = below_low[-1] if len(below_low) > 0 else 0
        t_standing = above_high[0] if len(above_high) > 0 else T - 1

        rising_mask = np.zeros(T, dtype=bool)
        rising_mask[t_start:t_standing + 1] = True
        rising_mask &= valid

        # ── S1: Transition duration ──
        duration_sec = (t_standing - t_start) / fps
        features[0] = np.clip(duration_sec / 5.0, 0, 1)

        # ── S2: Trunk anterior tilt ──
        spine_vec = _shoulder_center(skeleton_3d) - hip_c
        fwd_proj = _project_onto_axis(spine_vec, forward_dir)
        vert_proj = _project_onto_axis(spine_vec, up_dir)
        tilt_angle = np.arctan2(np.abs(fwd_proj), np.abs(vert_proj) + 1e-8)
        if rising_mask.any():
            max_tilt = np.max(tilt_angle[rising_mask])
            features[1] = np.clip(max_tilt / (np.pi / 3), 0, 1)

        # ── S3: Hand-to-ground contact frequency ──
        l_wrist_h = _project_onto_axis(
            skeleton_3d[:, LEFT_WRIST] - hip_c, up_dir
        ) / torso_len
        r_wrist_h = _project_onto_axis(
            skeleton_3d[:, RIGHT_WRIST] - hip_c, up_dir
        ) / torso_len
        # "Ground" = lowest wrist position in clip
        all_wrist_h = np.concatenate([l_wrist_h[valid], r_wrist_h[valid]])
        if len(all_wrist_h) > 0:
            ground_level = np.percentile(all_wrist_h, 5)
            contact = (
                (l_wrist_h < ground_level + 0.2)
                | (r_wrist_h < ground_level + 0.2)
            ) & valid
            features[2] = contact.sum() / (valid.sum() + 1e-8)

        # ── S4: CoM trajectory jerk (smoothness) ──
        com = _com_position(skeleton_3d)
        com_norm = com / torso_len[:, None]
        vel = np.gradient(com_norm, axis=0) * fps
        acc = np.gradient(vel, axis=0) * fps
        jerk = np.gradient(acc, axis=0) * fps
        jerk_mag = np.linalg.norm(jerk, axis=1)
        if valid.any():
            mean_jerk = np.mean(jerk_mag[valid])
            features[3] = 1.0 / (1.0 + mean_jerk)

        # ── S5: Knee extension symmetry ──
        l_knee_angle = _joint_angle(
            skeleton_3d[:, LEFT_HIP],
            skeleton_3d[:, LEFT_KNEE],
            skeleton_3d[:, LEFT_ANKLE],
        )
        r_knee_angle = _joint_angle(
            skeleton_3d[:, RIGHT_HIP],
            skeleton_3d[:, RIGHT_KNEE],
            skeleton_3d[:, RIGHT_ANKLE],
        )
        if rising_mask.any():
            rising_idx = np.where(rising_mask)[0]
            l_ext_t = rising_idx[np.argmax(l_knee_angle[rising_mask])]
            r_ext_t = rising_idx[np.argmax(r_knee_angle[rising_mask])]
            lag_sec = abs(l_ext_t - r_ext_t) / fps
            features[4] = 1.0 - np.clip(lag_sec / 1.0, 0, 1)

        # ── S6: Post-transition sway ──
        post_window = int(2.0 * fps)  # 2 seconds
        post_start = t_standing
        post_end = min(T, post_start + post_window)
        post_mask = np.zeros(T, dtype=bool)
        post_mask[post_start:post_end] = True
        post_mask &= valid

        if post_mask.sum() > 5:
            hip_fwd = _project_onto_axis(hip_c, forward_dir) / torso_len
            hip_lat = _project_onto_axis(hip_c, lat_dir) / torso_len
            std_ap = np.std(hip_fwd[post_mask])
            std_ml = np.std(hip_lat[post_mask])
            sway = np.sqrt(std_ap ** 2 + std_ml ** 2)
            features[5] = np.clip(sway / 0.15, 0, 1)

        return features


# ── Crawl Quality Features ──────────────────────────────────────────────

class CrawlQualityFeatures:
    """Crawl quality features — 6 per clip (Report 15.4.3).

    Core for L3 vs L4 vs L5 differentiation (highest discriminative power).
    """

    NUM_FEATURES = 6

    @staticmethod
    def compute(skeleton_3d, valid, torso_len, forward_dir, lat_dir, fps):
        """Compute 6 crawl quality features.

        Returns:
            (6,) float64 array.
        """
        T = skeleton_3d.shape[0]
        features = np.zeros(6, dtype=np.float64)

        # ── C1: Reciprocal pattern index ──
        # Forward velocity of four limb endpoints
        joints = [RIGHT_WRIST, LEFT_WRIST, RIGHT_KNEE, LEFT_KNEE]
        fwd_vel = {}
        for j in joints:
            fwd_proj = _project_onto_axis(skeleton_3d[:, j], forward_dir)
            fwd_vel[j] = np.gradient(fwd_proj) * fps

        # Advancing threshold: small forward motion
        if valid.any():
            med_torso = np.median(torso_len[valid])
        else:
            med_torso = 1.0
        adv_thresh = 0.01 * med_torso * fps

        r_hand_adv = fwd_vel[RIGHT_WRIST] > adv_thresh
        l_hand_adv = fwd_vel[LEFT_WRIST] > adv_thresh
        r_knee_adv = fwd_vel[RIGHT_KNEE] > adv_thresh
        l_knee_adv = fwd_vel[LEFT_KNEE] > adv_thresh

        # Contralateral pattern: R_hand+L_knee OR L_hand+R_knee
        contralateral = (r_hand_adv & l_knee_adv) | (l_hand_adv & r_knee_adv)
        # Any limb active
        active = r_hand_adv | l_hand_adv | r_knee_adv | l_knee_adv
        active_valid = active & valid
        contralateral_valid = contralateral & valid

        if active_valid.sum() > 0:
            features[0] = np.clip(
                contralateral_valid.sum() / (active_valid.sum() + 1e-8), 0, 1
            )

        # ── C2: Trunk elevation ──
        hip_c = _hip_center(skeleton_3d)
        up_dir = _body_vertical(skeleton_3d, valid)
        hip_height = _project_onto_axis(hip_c, up_dir) / torso_len
        if valid.any():
            features[1] = np.clip(np.mean(hip_height[valid]) / 1.5, 0, 1)

        # ── C3: Crawl velocity (horizontal pelvis speed) ──
        if T > 1:
            hip_disp = np.diff(hip_c, axis=0)  # (T-1, 3)
            # Project onto horizontal plane (forward + lateral)
            fwd_comp = np.sum(hip_disp * forward_dir[1:], axis=1)
            lat_comp = np.sum(hip_disp * lat_dir[1:], axis=1)
            horiz_speed = np.sqrt(fwd_comp ** 2 + lat_comp ** 2)
            horiz_speed_norm = horiz_speed / torso_len[1:]
            valid_speed = valid[1:] & valid[:-1]
            if valid_speed.any():
                mean_speed = np.mean(horiz_speed_norm[valid_speed]) * fps
                features[2] = np.clip(mean_speed / 0.1, 0, 1)

        # ── C4: Cycle regularity (CV of advancement cycles) ──
        r_wrist_fwd = _project_onto_axis(skeleton_3d[:, RIGHT_WRIST], forward_dir)
        peaks = _detect_peaks(r_wrist_fwd, min_distance=int(fps * 0.3))
        if len(peaks) >= 3:
            cycle_durations = np.diff(peaks).astype(float)
            cv = np.std(cycle_durations) / (np.mean(cycle_durations) + 1e-8)
            features[3] = 1.0 - np.clip(cv / 1.0, 0, 1)
        elif len(peaks) >= 2:
            features[3] = 0.5  # Indeterminate with only 1 cycle

        # ── C5: Trunk roll amplitude ──
        shoulder_vec = skeleton_3d[:, RIGHT_SHOULDER] - skeleton_3d[:, LEFT_SHOULDER]
        hip_vec = skeleton_3d[:, RIGHT_HIP] - skeleton_3d[:, LEFT_HIP]
        # Roll angle: angle between shoulder and hip planes
        # Project both onto frontal plane (perpendicular to forward)
        s_lat = _project_onto_axis(shoulder_vec, lat_dir)
        s_vert = _project_onto_axis(shoulder_vec, up_dir)
        h_lat = _project_onto_axis(hip_vec, lat_dir)
        h_vert = _project_onto_axis(hip_vec, up_dir)
        s_angle = np.arctan2(s_vert, s_lat + 1e-8)
        h_angle = np.arctan2(h_vert, h_lat + 1e-8)
        roll_diff = s_angle - h_angle
        if valid.any():
            amplitude = np.ptp(roll_diff[valid])
            features[4] = np.clip(amplitude / (np.pi / 4), 0, 1)

        # ── C6: Upper-lower limb contribution ratio ──
        if T > 1:
            upper_disp = (
                np.sum(np.linalg.norm(np.diff(skeleton_3d[:, LEFT_WRIST], axis=0), axis=1))
                + np.sum(np.linalg.norm(np.diff(skeleton_3d[:, RIGHT_WRIST], axis=0), axis=1))
            )
            lower_disp = (
                np.sum(np.linalg.norm(np.diff(skeleton_3d[:, LEFT_KNEE], axis=0), axis=1))
                + np.sum(np.linalg.norm(np.diff(skeleton_3d[:, RIGHT_KNEE], axis=0), axis=1))
            )
            total = upper_disp + lower_disp + 1e-8
            features[5] = upper_disp / total

        return features


# ── Side Rolling Quality Features ───────────────────────────────────────

class SideRollingQualityFeatures:
    """Side rolling quality features — 5 per clip (Report 15.4.5).

    Dedicated for L4 vs L5 differentiation (segmental vs log-roll).
    """

    NUM_FEATURES = 5

    @staticmethod
    def compute(skeleton_3d, valid, torso_len, forward_dir, lat_dir, fps):
        """Compute 5 side rolling quality features.

        Returns:
            (5,) float64 array.
        """
        T = skeleton_3d.shape[0]
        features = np.zeros(5, dtype=np.float64)

        # Orientation angles for shoulder and hip planes
        shoulder_vec = skeleton_3d[:, RIGHT_SHOULDER] - skeleton_3d[:, LEFT_SHOULDER]
        hip_vec = skeleton_3d[:, RIGHT_HIP] - skeleton_3d[:, LEFT_HIP]
        shoulder_angle = np.arctan2(shoulder_vec[:, 1], shoulder_vec[:, 0])
        hip_angle = np.arctan2(hip_vec[:, 1], hip_vec[:, 0])
        shoulder_angle = np.unwrap(shoulder_angle)
        hip_angle = np.unwrap(hip_angle)

        # Angular velocities
        shoulder_omega = np.abs(np.gradient(shoulder_angle)) * fps
        hip_omega = np.abs(np.gradient(hip_angle)) * fps

        rotation_thresh = 0.5  # rad/s

        # ── R1: Segmental rotation ratio (shoulder-hip onset lag) ──
        shoulder_rotating = shoulder_omega > rotation_thresh
        hip_rotating = hip_omega > rotation_thresh

        # Find onset frames (transitions from not-rotating to rotating)
        s_onsets = np.where(np.diff(shoulder_rotating.astype(int)) == 1)[0]
        h_onsets = np.where(np.diff(hip_rotating.astype(int)) == 1)[0]

        if len(s_onsets) > 0 and len(h_onsets) > 0:
            lags = []
            for s_t in s_onsets:
                if len(h_onsets) > 0:
                    nearest_h = h_onsets[np.argmin(np.abs(h_onsets - s_t))]
                    lags.append(abs(s_t - nearest_h) / fps)
            if lags:
                mean_lag = np.mean(lags)
                features[0] = np.clip(mean_lag / 0.5, 0, 1)

        # ── R2: Rolling velocity (rolls per second) ──
        clip_duration = T / fps
        total_angle_change = np.abs(shoulder_angle[-1] - shoulder_angle[0])
        num_rolls = total_angle_change / np.pi  # each roll ≈ pi radians
        if clip_duration > 0:
            velocity = num_rolls / clip_duration
            features[1] = np.clip(velocity / 1.0, 0, 1)

        # ── R3: Bilateral symmetry (L-to-R vs R-to-L) ──
        signed_omega = np.gradient(shoulder_angle) * fps
        pos_mask = signed_omega > rotation_thresh  # L→R episodes
        neg_mask = signed_omega < -rotation_thresh  # R→L episodes
        speed_pos = np.mean(np.abs(signed_omega[pos_mask])) if pos_mask.any() else 0
        speed_neg = np.mean(np.abs(signed_omega[neg_mask])) if neg_mask.any() else 0
        max_speed = max(speed_pos, speed_neg)
        if max_speed > 0:
            features[2] = min(speed_pos, speed_neg) / (max_speed + 1e-8)

        # ── R4: Inter-roll recovery time ──
        rolling = shoulder_omega > rotation_thresh
        if rolling.any() and not rolling.all():
            rest_mask = ~rolling & valid
            # Find contiguous rest segments
            rest_changes = np.diff(rest_mask.astype(int))
            rest_starts = np.where(rest_changes == 1)[0] + 1
            rest_ends = np.where(rest_changes == -1)[0] + 1
            # Handle edge cases
            if rest_mask[0]:
                rest_starts = np.concatenate([[0], rest_starts])
            if rest_mask[-1]:
                rest_ends = np.concatenate([rest_ends, [T]])

            n_segments = min(len(rest_starts), len(rest_ends))
            if n_segments > 0:
                durations = (rest_ends[:n_segments] - rest_starts[:n_segments]) / fps
                features[3] = np.clip(np.mean(durations) / 3.0, 0, 1)

        # ── R5: Arm ROM during roll ──
        l_shoulder_angle = _joint_angle(
            skeleton_3d[:, LEFT_ELBOW],
            skeleton_3d[:, LEFT_SHOULDER],
            skeleton_3d[:, LEFT_HIP],
        )
        r_shoulder_angle = _joint_angle(
            skeleton_3d[:, RIGHT_ELBOW],
            skeleton_3d[:, RIGHT_SHOULDER],
            skeleton_3d[:, RIGHT_HIP],
        )
        if valid.any():
            l_rom = np.ptp(l_shoulder_angle[valid])
            r_rom = np.ptp(r_shoulder_angle[valid])
            rom = (l_rom + r_rom) / 2
            features[4] = np.clip(rom / np.pi, 0, 1)

        return features


# ── Stand-to-Sit Quality Features ───────────────────────────────────────

class StandToSitQualityFeatures:
    """Standing-to-seated transition features — 4 per clip (Report 15.4.4).

    Supplementary for L1/L2/L3 differentiation via descent control.
    """

    NUM_FEATURES = 4

    @staticmethod
    def compute(skeleton_3d, valid, torso_len, forward_dir, lat_dir, fps):
        """Compute 4 stand-to-sit quality features.

        Returns:
            (4,) float64 array.
        """
        T = skeleton_3d.shape[0]
        features = np.zeros(4, dtype=np.float64)

        # Pelvis height signal
        hip_c = _hip_center(skeleton_3d)
        up_dir = _body_vertical(skeleton_3d, valid)
        hip_height_norm = _pelvis_height_signal(skeleton_3d, valid, torso_len)

        # Detect descent phase
        n_start = max(1, int(T * 0.1))
        n_end = max(1, int(T * 0.1))
        h_start = np.median(hip_height_norm[:n_start])
        h_end = np.median(hip_height_norm[-n_end:])

        if h_start <= h_end + 0.05:
            # No clear descent detected
            return features

        # Descent phase: from high to low
        threshold_high = h_start - 0.1 * (h_start - h_end)
        threshold_low = h_end + 0.2 * (h_start - h_end)
        below_high = np.where(hip_height_norm <= threshold_high)[0]
        above_low = np.where(hip_height_norm >= threshold_low)[0]

        t_descent_start = below_high[0] if len(below_high) > 0 else 0
        t_descent_end = above_low[-1] if len(above_low) > 0 else T - 1

        descent_mask = np.zeros(T, dtype=bool)
        descent_mask[t_descent_start:t_descent_end + 1] = True
        descent_mask &= valid

        if descent_mask.sum() < 3:
            return features

        # Vertical velocity during descent
        hip_vel = np.gradient(hip_height_norm) * fps

        # ── D1: Descent velocity consistency ──
        vel_descent = hip_vel[descent_mask]
        mean_vel = np.mean(vel_descent)
        std_vel = np.std(vel_descent)
        if abs(mean_vel) > 1e-8:
            consistency = 1.0 - np.clip(std_vel / (abs(mean_vel) + 1e-8), 0, 2) / 2
            features[0] = np.clip(consistency, 0, 1)

        # ── D2: Controlled deceleration ──
        descent_indices = np.where(descent_mask)[0]
        n_descent = len(descent_indices)
        last_20pct = descent_indices[int(n_descent * 0.8):]
        if len(last_20pct) > 1:
            vel_end = hip_vel[last_20pct]
            acc_end = np.gradient(vel_end) * fps
            # Deceleration = positive acceleration when velocity is negative (slowing down)
            mean_decel = np.mean(np.maximum(acc_end, 0))
            med_torso = np.median(torso_len[valid]) if valid.any() else 1.0
            features[1] = np.clip(mean_decel / (med_torso * fps * 0.01 + 1e-8), 0, 1)

        # ── D3: Impact softness (inverse of peak jerk at sitting) ──
        jerk = np.gradient(np.gradient(hip_vel)) * fps * fps
        # Find sitting moment (lowest pelvis point near end of descent)
        sitting_frame = descent_indices[-1] if len(descent_indices) > 0 else T - 1
        jerk_window = max(0, sitting_frame - 5), min(T, sitting_frame + 6)
        if jerk_window[1] > jerk_window[0]:
            peak_jerk = np.max(np.abs(jerk[jerk_window[0]:jerk_window[1]]))
            med_torso = np.median(torso_len[valid]) if valid.any() else 1.0
            norm_jerk = peak_jerk / (med_torso * fps ** 3 + 1e-8)
            features[2] = 1.0 / (1.0 + norm_jerk * 100)

        # ── D4: Hand support frequency ──
        l_wrist_h = _project_onto_axis(
            skeleton_3d[:, LEFT_WRIST] - hip_c, up_dir
        ) / torso_len
        r_wrist_h = _project_onto_axis(
            skeleton_3d[:, RIGHT_WRIST] - hip_c, up_dir
        ) / torso_len
        # Support = wrists near or below hip level during descent
        all_wrist_h = np.concatenate([l_wrist_h[valid], r_wrist_h[valid]])
        if len(all_wrist_h) > 0:
            ground_level = np.percentile(all_wrist_h, 5)
            contact = (
                (l_wrist_h < ground_level + 0.2)
                | (r_wrist_h < ground_level + 0.2)
            ) & valid
            features[3] = contact.sum() / (valid.sum() + 1e-8)

        return features


# ── Dispatcher & entry point ────────────────────────────────────────────

_MOVEMENT_DISPATCHERS = {
    "w": WalkQualityFeatures,
    "c_s": SitToStandQualityFeatures,
    "cr": CrawlQualityFeatures,
    "sr": SideRollingQualityFeatures,
    "s_c": StandToSitQualityFeatures,
}


def extract_movement_quality_features(skeleton_3d, movement_type, config=None):
    """Extract per-clip movement quality features.

    Args:
        skeleton_3d: (T, 17, 3) float32, xyz in world coordinates.
        movement_type: str — 'w', 'cr', 'c_s', 's_c', 'sr', or None.
        config: dict with optional 'fps' key.

    Returns:
        np.ndarray of shape (NUM_MOVEMENT_QUALITY_FEATURES,), dtype float32.
        Padded with trailing zeros for movement types with fewer than 6 features.
    """
    config = config or {}
    fps = config.get("fps", DEFAULT_FPS)

    zero_features = np.zeros(NUM_MOVEMENT_QUALITY_FEATURES, dtype=np.float32)

    # Validate movement type
    if movement_type not in _MOVEMENT_DISPATCHERS:
        if movement_type is not None:
            logger.warning("Unknown movement type '%s', returning zeros", movement_type)
        return zero_features

    T = skeleton_3d.shape[0]
    if T < 2:
        return zero_features

    if np.all(np.abs(skeleton_3d) < 1e-6):
        logger.warning("All-zero skeleton, returning zero movement quality features")
        return zero_features

    valid = _valid_frame_mask(skeleton_3d)
    if not valid.any():
        logger.warning("No valid frames, returning zero movement quality features")
        return zero_features

    # Shared intermediates
    torso_len = _torso_length(skeleton_3d, valid)
    forward_dir = _estimate_forward_direction(skeleton_3d, valid)
    lat_dir = _lateral_direction(skeleton_3d, valid)

    # Dispatch to movement-specific class
    cls = _MOVEMENT_DISPATCHERS[movement_type]
    raw_features = cls.compute(skeleton_3d, valid, torso_len, forward_dir, lat_dir, fps)

    # Pad to uniform dimension
    result = zero_features.copy()
    n = min(len(raw_features), NUM_MOVEMENT_QUALITY_FEATURES)
    result[:n] = raw_features[:n]

    # NaN safety
    result = np.nan_to_num(result, nan=0.0)
    return result.astype(np.float32)


def get_movement_quality_feature_names(movement_type):
    """Return human-readable names for the features of a given movement type.

    Args:
        movement_type: str — 'w', 'cr', 'c_s', 's_c', 'sr'.

    Returns:
        List of strings. Length = NUM_MOVEMENT_QUALITY_FEATURES (padded names).
    """
    names_map = {
        "w": [
            "walk_cadence",
            "walk_gait_symmetry",
            "walk_trunk_lateral_sway",
            "walk_step_width",
            "walk_head_stability",
        ],
        "c_s": [
            "sts_transition_duration",
            "sts_trunk_anterior_tilt",
            "sts_hand_ground_contact",
            "sts_com_jerk_smoothness",
            "sts_knee_extension_symmetry",
            "sts_post_transition_sway",
        ],
        "cr": [
            "crawl_reciprocal_pattern_index",
            "crawl_trunk_elevation",
            "crawl_velocity",
            "crawl_cycle_regularity",
            "crawl_trunk_roll_amplitude",
            "crawl_upper_lower_limb_ratio",
        ],
        "sr": [
            "roll_segmental_rotation_ratio",
            "roll_rolling_velocity",
            "roll_bilateral_symmetry",
            "roll_inter_roll_recovery",
            "roll_arm_rom",
        ],
        "s_c": [
            "stsi_descent_velocity_consistency",
            "stsi_controlled_deceleration",
            "stsi_impact_softness",
            "stsi_hand_support_frequency",
        ],
    }
    raw = names_map.get(movement_type, [])
    # Pad to uniform length
    padded = raw + [f"_pad_{i}" for i in range(NUM_MOVEMENT_QUALITY_FEATURES - len(raw))]
    return padded[:NUM_MOVEMENT_QUALITY_FEATURES]
