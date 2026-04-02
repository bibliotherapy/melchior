"""3D triangulation for patient and caregiver skeletons.

Uses Direct Linear Transform (DLT) via OpenCV triangulatePoints to
reconstruct 3D joint positions from multi-view 2D detections.
Produces (T, 17, 3) arrays for both patient and caregiver.

Post-processing includes bone length consistency filtering,
temporal smoothing (Savitzky-Golay), and missing joint interpolation.
"""

import logging
from itertools import combinations

import cv2
import numpy as np
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

# COCO skeleton bones for length validation
COCO_BONES = [
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (5, 11), (6, 12),    # torso sides
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
    (5, 6), (11, 12),    # shoulder/hip width
]

CONF_THRESHOLD = 0.3


def build_projection_matrices(calibration):
    """Build projection matrices from calibration data.

    Args:
        calibration: output from PoseCalibrator.calibrate_patient()
            or load_calibration().

    Returns:
        Dict {view: P_3x4} for each calibrated view.
    """
    proj = {}
    for view in calibration.get("calibrated_views", []):
        P = calibration[view].get("P")
        if P is not None:
            proj[view] = P
    return proj


class SkeletonTriangulator:
    """Triangulate 3D skeletons from multi-view 2D detections."""

    def __init__(self, config=None):
        config = config or {}
        self.min_views = config.get("min_views", 2)
        self.reproj_threshold = config.get("reprojection_threshold", 15.0)
        self.bone_length_filter = config.get("bone_length_filter", True)

    def _sync_frame_counts(self, kp_per_view):
        """Truncate all views to the minimum frame count.

        Args:
            kp_per_view: {view: (T_i, 17, 3)}

        Returns:
            {view: (T_min, 17, 3)} truncated.
        """
        if not kp_per_view:
            return kp_per_view

        T_min = min(kp.shape[0] for kp in kp_per_view.values())
        T_max = max(kp.shape[0] for kp in kp_per_view.values())

        if T_max > T_min * 1.1:
            logger.warning(
                "Frame count mismatch: min=%d, max=%d (%.1f%% difference)",
                T_min, T_max, (T_max - T_min) / T_min * 100,
            )

        return {v: kp[:T_min] for v, kp in kp_per_view.items()}

    def _triangulate_joint_pair(self, pt1, pt2, P1, P2):
        """Triangulate one joint from two views.

        Args:
            pt1: (2,) point in view 1 (pixel coordinates).
            pt2: (2,) point in view 2 (pixel coordinates).
            P1, P2: (3, 4) projection matrices.

        Returns:
            point_3d: (3,) world coordinates.
            reproj_error: mean pixel error across both views.
        """
        pts1 = pt1.reshape(2, 1).astype(np.float64)
        pts2 = pt2.reshape(2, 1).astype(np.float64)

        X_homo = cv2.triangulatePoints(P1, P2, pts1, pts2)
        X = X_homo[:3] / X_homo[3]
        point_3d = X.flatten()

        # Reprojection error
        X_h = np.append(point_3d, 1.0)
        proj1 = P1 @ X_h
        proj1 = proj1[:2] / proj1[2]
        proj2 = P2 @ X_h
        proj2 = proj2[:2] / proj2[2]

        err1 = np.linalg.norm(proj1 - pt1)
        err2 = np.linalg.norm(proj2 - pt2)
        return point_3d, (err1 + err2) / 2

    def _triangulate_joint_multiview(self, points_per_view, proj_matrices):
        """Triangulate a joint from all available views.

        If 3 views: triangulate all pairs, pick best reprojection error.
        If 2 views: direct triangulation.

        Args:
            points_per_view: {view: (2,)} pixel coordinates.
            proj_matrices: {view: (3, 4)}.

        Returns:
            (point_3d, reproj_error) or (None, inf).
        """
        views = list(points_per_view.keys())
        if len(views) < 2:
            return None, float("inf")

        best_point = None
        best_error = float("inf")

        for v1, v2 in combinations(views, 2):
            pt1 = points_per_view[v1]
            pt2 = points_per_view[v2]
            P1 = proj_matrices[v1]
            P2 = proj_matrices[v2]

            point_3d, error = self._triangulate_joint_pair(pt1, pt2, P1, P2)
            if error < best_error:
                best_error = error
                best_point = point_3d

        if best_error > self.reproj_threshold:
            return None, best_error

        return best_point, best_error

    def triangulate_person(self, kp_per_view, proj_matrices, calibration):
        """Triangulate full sequence for one person.

        Args:
            kp_per_view: {view: (T, 17, 3)} per-view 2D detections.
            proj_matrices: {view: (3, 4)} projection matrices.
            calibration: calibration dict (for undistortion).

        Returns:
            skeleton_3d: (T, 17, 3) with NaN for missing joints.
            confidence_3d: (T, 17) triangulation confidence.
        """
        kp_per_view = self._sync_frame_counts(kp_per_view)
        views = [v for v in kp_per_view if v in proj_matrices]

        if len(views) < 2:
            T = max(kp.shape[0] for kp in kp_per_view.values()) if kp_per_view else 1
            return np.full((T, 17, 3), np.nan), np.zeros((T, 17))

        # Undistort keypoints per view (critical for GoPro barrel distortion).
        # Convert from distorted pixels to undistorted pixels using K and dist.
        kp_undistorted = {}
        for v in views:
            kp_orig = kp_per_view[v]
            T_v, n_joints = kp_orig.shape[0], kp_orig.shape[1]
            K = calibration[v]["K"]
            dist = calibration[v]["dist"]
            kp_ud = kp_orig.copy()
            if np.any(dist != 0):
                for t in range(T_v):
                    pts = kp_orig[t, :, :2].reshape(-1, 1, 2).astype(np.float64)
                    # Undistort to normalized, then reproject to undistorted pixels
                    pts_norm = cv2.undistortPoints(pts, K, dist)
                    pts_ud = cv2.undistortPoints(pts, K, dist, P=K)
                    kp_ud[t, :, :2] = pts_ud.reshape(-1, 2)
            kp_undistorted[v] = kp_ud

        T = kp_undistorted[views[0]].shape[0]
        skeleton_3d = np.full((T, 17, 3), np.nan)
        confidence_3d = np.zeros((T, 17))

        for t in range(T):
            for j in range(17):
                # Collect views where this joint is confident
                pts = {}
                confs = []
                for v in views:
                    conf = kp_undistorted[v][t, j, 2]
                    if conf > CONF_THRESHOLD:
                        pts[v] = kp_undistorted[v][t, j, :2]
                        confs.append(conf)

                if len(pts) >= 2:
                    point_3d, error = self._triangulate_joint_multiview(
                        pts, proj_matrices,
                    )
                    if point_3d is not None:
                        skeleton_3d[t, j] = point_3d
                        confidence_3d[t, j] = min(confs)

        return skeleton_3d, confidence_3d

    def validate_skeleton(self, skeleton_3d, confidence_3d):
        """Post-process triangulated skeleton.

        Steps:
            1. Bone length filter (reject > 30% deviation from median)
            2. Interpolate missing (NaN) joints
            3. Temporal smoothing (Savitzky-Golay)

        Args:
            skeleton_3d: (T, 17, 3) with NaN for missing.
            confidence_3d: (T, 17).

        Returns:
            Cleaned (T, 17, 3) with NaN replaced.
        """
        cleaned = skeleton_3d.copy()

        if self.bone_length_filter:
            valid_mask = self._bone_length_filter(cleaned)
            cleaned[~valid_mask] = np.nan

        cleaned = self._interpolate_missing(cleaned)
        cleaned = self._temporal_smooth(cleaned)

        # Replace remaining NaN with zeros
        cleaned = np.nan_to_num(cleaned, nan=0.0)
        return cleaned

    def _bone_length_filter(self, skeleton_3d):
        """Flag joints in frames with implausible bone lengths.

        Returns:
            valid_mask: (T, 17) boolean.
        """
        T = skeleton_3d.shape[0]
        valid_mask = np.ones((T, 17), dtype=bool)

        for j1, j2 in COCO_BONES:
            # Compute bone length per frame
            diff = skeleton_3d[:, j1] - skeleton_3d[:, j2]
            lengths = np.linalg.norm(diff, axis=1)

            # Skip if too many NaN
            valid = ~np.isnan(lengths)
            if valid.sum() < 5:
                continue

            median_len = np.nanmedian(lengths[valid])
            if median_len < 1e-6:
                continue

            # Flag frames with > 30% deviation
            deviation = np.abs(lengths - median_len) / median_len
            bad = valid & (deviation > 0.3)
            valid_mask[bad, j1] = False
            valid_mask[bad, j2] = False

        return valid_mask

    def _interpolate_missing(self, skeleton_3d, max_gap=10):
        """Linearly interpolate missing (NaN) joints along time axis.

        Args:
            skeleton_3d: (T, 17, 3) with NaN for missing.
            max_gap: maximum consecutive NaN frames to interpolate.

        Returns:
            (T, 17, 3) with gaps <= max_gap filled.
        """
        T = skeleton_3d.shape[0]
        if T < 2:
            return skeleton_3d

        result = skeleton_3d.copy()

        for j in range(17):
            for axis in range(3):
                values = result[:, j, axis]
                valid = ~np.isnan(values)
                valid_idx = np.where(valid)[0]

                if len(valid_idx) < 2:
                    continue

                invalid_idx = np.where(~valid)[0]
                # Only interpolate within the valid range
                in_range = (invalid_idx >= valid_idx[0]) & (invalid_idx <= valid_idx[-1])
                to_interp = invalid_idx[in_range]

                if len(to_interp) == 0:
                    continue

                # Check gap lengths
                for idx in to_interp:
                    # Find nearest valid frames
                    left = valid_idx[valid_idx < idx]
                    right = valid_idx[valid_idx > idx]
                    if len(left) > 0 and len(right) > 0:
                        gap = right[0] - left[-1]
                        if gap <= max_gap:
                            # Linear interpolation
                            t_left, t_right = left[-1], right[0]
                            alpha = (idx - t_left) / (t_right - t_left)
                            result[idx, j, axis] = (
                                values[t_left] * (1 - alpha) + values[t_right] * alpha
                            )

        return result

    def _temporal_smooth(self, skeleton_3d, window=7, polyorder=2):
        """Apply Savitzky-Golay filter per joint per axis.

        Only smooths non-NaN segments.

        Args:
            skeleton_3d: (T, 17, 3).
            window: filter window length (must be odd).
            polyorder: polynomial order.

        Returns:
            Smoothed (T, 17, 3).
        """
        T = skeleton_3d.shape[0]
        if T < window:
            return skeleton_3d

        result = skeleton_3d.copy()

        for j in range(17):
            for axis in range(3):
                values = result[:, j, axis]
                valid = ~np.isnan(values)

                if valid.sum() < window:
                    continue

                # Find contiguous valid segments
                segments = self._find_segments(valid)
                for start, end in segments:
                    seg_len = end - start
                    if seg_len >= window:
                        result[start:end, j, axis] = savgol_filter(
                            values[start:end], window, polyorder,
                        )

        return result

    @staticmethod
    def _find_segments(mask):
        """Find contiguous True segments in a boolean mask.

        Returns:
            List of (start, end) tuples.
        """
        segments = []
        in_seg = False
        start = 0
        for i, val in enumerate(mask):
            if val and not in_seg:
                start = i
                in_seg = True
            elif not val and in_seg:
                segments.append((start, i))
                in_seg = False
        if in_seg:
            segments.append((start, len(mask)))
        return segments

    def compute_reprojection_error(self, skeleton_3d, kp_per_view, calibration):
        """Project 3D joints back to 2D and measure pixel error.

        Args:
            skeleton_3d: (T, 17, 3) triangulated skeleton.
            kp_per_view: {view: (T, 17, 3)} original 2D detections.
            calibration: calibration dict with P matrices.

        Returns:
            {view: (T, 17) pixel errors} for each calibrated view.
        """
        proj_matrices = build_projection_matrices(calibration)
        T = skeleton_3d.shape[0]
        errors = {}

        for view, P in proj_matrices.items():
            if view not in kp_per_view:
                continue

            kp_2d = kp_per_view[view]
            T_v = min(T, kp_2d.shape[0])
            view_errors = np.full((T, 17), np.nan)

            for t in range(T_v):
                for j in range(17):
                    if np.any(np.isnan(skeleton_3d[t, j])) or np.all(skeleton_3d[t, j] == 0):
                        continue
                    if kp_2d[t, j, 2] < CONF_THRESHOLD:
                        continue

                    X_h = np.append(skeleton_3d[t, j], 1.0)
                    proj = P @ X_h
                    proj_2d = proj[:2] / proj[2]
                    orig_2d = kp_2d[t, j, :2]
                    view_errors[t, j] = np.linalg.norm(proj_2d - orig_2d)

            errors[view] = view_errors

        return errors

    def triangulate_clip(self, child_kp_per_view, caregiver_kp_per_view,
                         calibration):
        """Triangulate both child and caregiver for a clip.

        Args:
            child_kp_per_view: {view: (T, 17, 3)}.
            caregiver_kp_per_view: {view: (T, 17, 3)}.
            calibration: calibration dict.

        Returns:
            Dict with 'child' and 'caregiver' (T, 17, 3) arrays.
        """
        proj_matrices = build_projection_matrices(calibration)

        # Triangulate child
        child_3d, child_conf = self.triangulate_person(
            child_kp_per_view, proj_matrices, calibration,
        )
        child_3d = self.validate_skeleton(child_3d, child_conf)

        # Triangulate caregiver (skip if all zeros)
        caregiver_present = False
        for view_kp in caregiver_kp_per_view.values():
            if np.any(view_kp[:, :, 2] > CONF_THRESHOLD):
                caregiver_present = True
                break

        if caregiver_present:
            cg_3d, cg_conf = self.triangulate_person(
                caregiver_kp_per_view, proj_matrices, calibration,
            )
            cg_3d = self.validate_skeleton(cg_3d, cg_conf)
        else:
            T = child_3d.shape[0]
            cg_3d = np.zeros((T, 17, 3), dtype=np.float32)

        return {
            "child": child_3d.astype(np.float32),
            "caregiver": cg_3d.astype(np.float32),
        }
