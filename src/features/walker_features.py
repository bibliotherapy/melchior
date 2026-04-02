"""Walker-skeleton spatial features for L3 vs L4 discrimination.

Computes features from the spatial relationship between the child's skeleton
keypoints and the walker's SAM2 mask. These features directly measure
device-assisted vs human-assisted mobility — the core L3/L4 clinical question.

Features (5D per clip):
  [0] hand_to_walker_dist_mean     - Mean min wrist-to-walker-mask distance
  [1] walker_engagement_ratio      - Fraction of frames with hand near walker
  [2] walker_velocity_mean         - Mean walker mask centroid velocity
  [3] walker_child_velocity_corr   - Correlation of walker and child CoM velocity
  [4] support_source_ratio         - hand-to-walker vs hand-to-caregiver distance ratio

All features are normalized to [0, 1] range where possible.
Returns zero vector when no walker mask is available.
"""

import logging

import cv2
import numpy as np
from scipy import ndimage
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

NUM_WALKER_FEATURES = 5

# COCO keypoint indices
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6


def _mask_centroid(mask):
    """Compute centroid of a binary mask.

    Returns:
        (cx, cy) tuple, or None if mask is empty.
    """
    if not mask.any():
        return None
    cy, cx = ndimage.center_of_mass(mask)
    return (float(cx), float(cy))


def _mask_edge_distance(mask, point_x, point_y):
    """Compute minimum distance from a point to the nearest mask edge pixel.

    Args:
        mask: (H, W) binary mask.
        point_x, point_y: Query point coordinates.

    Returns:
        Minimum Euclidean distance in pixels. Returns float('inf') if mask is empty.
    """
    if not mask.any():
        return float("inf")

    # Find mask contour pixels
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return float("inf")

    # Concatenate all contour points
    all_points = np.vstack(contours).squeeze()
    if all_points.ndim == 1:
        all_points = all_points.reshape(1, 2)

    # Compute distances
    dists = np.sqrt((all_points[:, 0] - point_x) ** 2 +
                    (all_points[:, 1] - point_y) ** 2)
    return float(dists.min())


def _point_in_mask(mask, point_x, point_y):
    """Check if a point falls inside a mask."""
    h, w = mask.shape
    px, py = int(round(point_x)), int(round(point_y))
    if 0 <= px < w and 0 <= py < h:
        return bool(mask[py, px])
    return False


def compute_hand_to_walker_distance(wrist_keypoints, walker_mask):
    """Compute minimum distance from either wrist to walker mask edge.

    If a wrist is INSIDE the mask (gripping the walker), distance is 0.

    Args:
        wrist_keypoints: (2, 3) array for [left_wrist, right_wrist] with (x, y, conf).
        walker_mask: (H, W) binary mask of the walker.

    Returns:
        Minimum distance in pixels (0 if wrist is inside mask).
    """
    min_dist = float("inf")

    for wrist in wrist_keypoints:
        x, y, conf = wrist
        if conf < 0.3:
            continue
        if _point_in_mask(walker_mask, x, y):
            return 0.0
        dist = _mask_edge_distance(walker_mask, x, y)
        min_dist = min(min_dist, dist)

    return min_dist


def compute_walker_engagement_ratio(distances, threshold_px=30):
    """Compute fraction of frames where hand is near walker.

    Args:
        distances: (T,) array of per-frame hand-to-walker distances.
        threshold_px: Distance threshold for "engaged" (in pixels).

    Returns:
        Ratio in [0, 1]. Higher = more engagement with walker (L3 signal).
    """
    valid = np.isfinite(distances)
    if valid.sum() == 0:
        return 0.0
    engaged = (distances[valid] <= threshold_px).sum()
    return float(engaged) / float(valid.sum())


def compute_walker_velocity(walker_masks, fps=30):
    """Compute walker centroid velocity across frames.

    Args:
        walker_masks: (T, H, W) binary mask array.
        fps: Video frame rate.

    Returns:
        (T-1,) array of centroid velocities in pixels/frame.
    """
    T = walker_masks.shape[0]
    centroids = []
    for t in range(T):
        c = _mask_centroid(walker_masks[t])
        centroids.append(c)

    velocities = np.zeros(max(T - 1, 1))
    for t in range(T - 1):
        if centroids[t] is not None and centroids[t + 1] is not None:
            dx = centroids[t + 1][0] - centroids[t][0]
            dy = centroids[t + 1][1] - centroids[t][1]
            velocities[t] = np.sqrt(dx ** 2 + dy ** 2)

    return velocities


def compute_child_com_velocity(child_keypoints):
    """Compute child's center-of-mass velocity from skeleton.

    CoM approximated as midpoint of hips and shoulders.

    Args:
        child_keypoints: (T, 17, 3) array.

    Returns:
        (T-1,) array of CoM velocities in pixels/frame.
    """
    T = child_keypoints.shape[0]

    # CoM from hip and shoulder midpoints
    com_indices = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    com_positions = np.zeros((T, 2))

    for t in range(T):
        valid_points = []
        for idx in com_indices:
            if child_keypoints[t, idx, 2] > 0.3:
                valid_points.append(child_keypoints[t, idx, :2])
        if valid_points:
            com_positions[t] = np.mean(valid_points, axis=0)

    velocities = np.zeros(max(T - 1, 1))
    for t in range(T - 1):
        dx = com_positions[t + 1, 0] - com_positions[t, 0]
        dy = com_positions[t + 1, 1] - com_positions[t, 1]
        velocities[t] = np.sqrt(dx ** 2 + dy ** 2)

    return velocities


def compute_walker_child_velocity_correlation(walker_vel, child_vel):
    """Pearson correlation between walker and child velocities.

    High correlation = child is pushing the walker (L3).
    Low correlation = walker is stationary or moved by others (L4).

    Args:
        walker_vel: (T-1,) walker centroid velocities.
        child_vel: (T-1,) child CoM velocities.

    Returns:
        Correlation coefficient in [-1, 1], or 0 if insufficient data.
    """
    min_len = min(len(walker_vel), len(child_vel))
    if min_len < 5:
        return 0.0

    wv = walker_vel[:min_len]
    cv = child_vel[:min_len]

    # Skip if either has zero variance
    if np.std(wv) < 1e-6 or np.std(cv) < 1e-6:
        return 0.0

    corr, _ = pearsonr(wv, cv)
    return float(corr) if np.isfinite(corr) else 0.0


def compute_support_source_ratio(wrist_keypoints_seq, walker_masks, caregiver_keypoints_seq):
    """Compute ratio of hand-to-walker vs hand-to-caregiver distance.

    Values < 0.5 = closer to walker (L3 signal)
    Values > 0.5 = closer to caregiver (L4 signal)
    Value = 0.5 = equidistant or no data

    Args:
        wrist_keypoints_seq: (T, 2, 3) wrist keypoints per frame.
        walker_masks: (T, H, W) walker masks.
        caregiver_keypoints_seq: (T, 17, 3) caregiver skeleton per frame.
            Can be all zeros if no caregiver present.

    Returns:
        Mean support source ratio in [0, 1].
    """
    T = wrist_keypoints_seq.shape[0]
    ratios = []

    for t in range(T):
        wrists = wrist_keypoints_seq[t]
        walker_mask = walker_masks[t]
        caregiver_kp = caregiver_keypoints_seq[t]

        # Hand-to-walker distance
        d_walker = compute_hand_to_walker_distance(wrists, walker_mask)

        # Hand-to-caregiver distance (min distance to caregiver's hands)
        d_caregiver = float("inf")
        for wrist_idx in [LEFT_WRIST, RIGHT_WRIST]:
            if caregiver_kp[wrist_idx, 2] > 0.3:
                for child_wrist in wrists:
                    if child_wrist[2] > 0.3:
                        dx = child_wrist[0] - caregiver_kp[wrist_idx, 0]
                        dy = child_wrist[1] - caregiver_kp[wrist_idx, 1]
                        d = np.sqrt(dx ** 2 + dy ** 2)
                        d_caregiver = min(d_caregiver, d)

        # Compute ratio
        if np.isfinite(d_walker) and np.isfinite(d_caregiver):
            total = d_walker + d_caregiver
            if total > 0:
                ratios.append(d_walker / total)
        elif np.isfinite(d_walker):
            ratios.append(0.0 if d_walker < 30 else 0.5)
        elif np.isfinite(d_caregiver):
            ratios.append(1.0 if d_caregiver < 30 else 0.5)

    if not ratios:
        return 0.5
    return float(np.mean(ratios))


def extract_walker_features(child_keypoints, walker_masks, caregiver_keypoints=None,
                            proximity_threshold_px=30, fps=30):
    """Extract all walker-skeleton spatial features for a clip.

    Args:
        child_keypoints: (T, 17, 3) child skeleton keypoints.
        walker_masks: (T, H, W) walker binary masks, or None.
        caregiver_keypoints: (T, 17, 3) caregiver skeleton, or None.
        proximity_threshold_px: Pixel threshold for walker engagement.
        fps: Video frame rate.

    Returns:
        np.ndarray of shape (5,) with walker features.
        Returns zeros if walker_masks is None or all empty.
    """
    zero_features = np.zeros(NUM_WALKER_FEATURES, dtype=np.float32)

    # No walker present
    if walker_masks is None:
        return zero_features

    T = walker_masks.shape[0]

    # Check if walker is visible in any frame
    walker_visible = np.array([walker_masks[t].any() for t in range(T)])
    if not walker_visible.any():
        return zero_features

    # Ensure matching frame counts
    T = min(T, child_keypoints.shape[0])

    # Extract wrist keypoints: (T, 2, 3)
    wrist_indices = [LEFT_WRIST, RIGHT_WRIST]
    wrists = child_keypoints[:T, wrist_indices, :]

    # [0] Hand-to-walker distance (mean, normalized by image diagonal)
    h, w = walker_masks.shape[1], walker_masks.shape[2]
    diag = np.sqrt(h ** 2 + w ** 2)
    distances = np.array([
        compute_hand_to_walker_distance(wrists[t], walker_masks[t])
        for t in range(T)
    ])
    distances = np.clip(distances, 0, diag)
    valid_dists = distances[np.isfinite(distances)]
    dist_mean = float(np.mean(valid_dists)) / diag if len(valid_dists) > 0 else 1.0

    # [1] Walker engagement ratio
    engagement = compute_walker_engagement_ratio(distances, threshold_px=proximity_threshold_px)

    # [2] Walker velocity (mean, normalized)
    walker_vel = compute_walker_velocity(walker_masks[:T], fps=fps)
    vel_mean = float(np.mean(walker_vel)) / diag if len(walker_vel) > 0 else 0.0
    vel_mean = min(vel_mean, 1.0)

    # [3] Walker-child velocity correlation (shifted to [0, 1])
    child_vel = compute_child_com_velocity(child_keypoints[:T])
    vel_corr = compute_walker_child_velocity_correlation(walker_vel, child_vel)
    vel_corr_norm = (vel_corr + 1.0) / 2.0  # [-1, 1] -> [0, 1]

    # [4] Support source ratio
    if caregiver_keypoints is not None:
        caregiver_kp = caregiver_keypoints[:T]
    else:
        caregiver_kp = np.zeros_like(child_keypoints[:T])
    support_ratio = compute_support_source_ratio(wrists, walker_masks[:T], caregiver_kp)

    features = np.array([
        dist_mean,
        engagement,
        vel_mean,
        vel_corr_norm,
        support_ratio,
    ], dtype=np.float32)

    return features


def get_feature_names():
    """Return human-readable names for walker features."""
    return [
        "hand_to_walker_dist_mean",
        "walker_engagement_ratio",
        "walker_velocity_mean",
        "walker_child_velocity_corr",
        "support_source_ratio",
    ]
