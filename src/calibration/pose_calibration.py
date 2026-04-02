"""Camera calibration via Human Pose as Calibration Pattern.

Implements Takahashi et al. [5]: uses matched human pose keypoints across
three unsynchronized camera views to estimate extrinsic camera parameters.

Camera setup (fixed for entire dataset):
    FV (Front View): GoPro Hero — wide-angle, barrel distortion
    LV (Left View):  iPhone 12 mini — ~26mm equiv, minimal distortion
    RV (Right View): Samsung Galaxy — minimal distortion

Calibration is per-patient because camera placement varies between sessions.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Approximate camera intrinsics (1920x1080 resolution)
# These are initial estimates; refine with EXIF data if available.
CAMERA_INTRINSICS = {
    "FV": {
        "K": np.array([[960, 0, 960], [0, 960, 540], [0, 0, 1]], dtype=np.float64),
        "dist": np.array([-0.25, 0.06, 0, 0], dtype=np.float64),
        "name": "GoPro Hero",
    },
    "LV": {
        "K": np.array([[1400, 0, 960], [0, 1400, 540], [0, 0, 1]], dtype=np.float64),
        "dist": np.array([0, 0, 0, 0], dtype=np.float64),
        "name": "iPhone 12 mini",
    },
    "RV": {
        "K": np.array([[1350, 0, 960], [0, 1350, 540], [0, 0, 1]], dtype=np.float64),
        "dist": np.array([0, 0, 0, 0], dtype=np.float64),
        "name": "Samsung Galaxy",
    },
}


class CalibrationError(Exception):
    """Raised when calibration cannot proceed."""
    pass


class PoseCalibrator:
    """Estimate camera extrinsics from matched human pose correspondences."""

    def __init__(self, config=None):
        config = config or {}
        self.min_visible_joints = config.get("min_visible_joints", 8)
        self.ransac_threshold = config.get("ransac_threshold", 5.0)
        self.num_calibration_frames = config.get("num_calibration_frames", 50)
        self.conf_threshold = 0.3

    def get_intrinsics(self, view):
        """Return (K, dist_coeffs) for a camera view.

        Args:
            view: 'FV', 'LV', or 'RV'.

        Returns:
            Tuple of (K_3x3, dist_4).
        """
        cam = CAMERA_INTRINSICS[view]
        return cam["K"].copy(), cam["dist"].copy()

    def undistort_keypoints(self, keypoints, view):
        """Undistort 2D keypoints using camera parameters.

        Critical for FV (GoPro) which has barrel distortion.

        Args:
            keypoints: (N, 2) pixel coordinates.
            view: camera view name.

        Returns:
            (N, 2) undistorted normalized coordinates.
        """
        K, dist = self.get_intrinsics(view)
        pts = keypoints.reshape(-1, 1, 2).astype(np.float64)
        undistorted = cv2.undistortPoints(pts, K, dist)
        return undistorted.reshape(-1, 2)

    def _collect_correspondences(self, kp_view1, kp_view2):
        """Collect matched joint correspondences across two views.

        Args:
            kp_view1: (T, 17, 3) keypoints from view 1 (x, y, conf).
            kp_view2: (T, 17, 3) keypoints from view 2 (x, y, conf).

        Returns:
            pts1: (M, 2) matched points in view 1.
            pts2: (M, 2) matched points in view 2.
        """
        T = min(kp_view1.shape[0], kp_view2.shape[0])
        pts1, pts2 = [], []

        for t in range(T):
            for j in range(17):
                c1 = kp_view1[t, j, 2]
                c2 = kp_view2[t, j, 2]
                if c1 > self.conf_threshold and c2 > self.conf_threshold:
                    pts1.append(kp_view1[t, j, :2])
                    pts2.append(kp_view2[t, j, :2])

        if not pts1:
            return np.empty((0, 2)), np.empty((0, 2))
        return np.array(pts1), np.array(pts2)

    def _select_calibration_frames(self, kp_per_view, num_frames=None):
        """Select frames with highest joint visibility across all views.

        Args:
            kp_per_view: {view: (T, 17, 3)} child keypoints.
            num_frames: target count (default from config).

        Returns:
            List of selected frame indices.
        """
        if num_frames is None:
            num_frames = self.num_calibration_frames

        views = list(kp_per_view.keys())
        T = min(kp_per_view[v].shape[0] for v in views)

        # Score each frame: minimum visible joints across all views
        scores = np.zeros(T)
        for t in range(T):
            min_visible = 17
            for v in views:
                visible = (kp_per_view[v][t, :, 2] > self.conf_threshold).sum()
                min_visible = min(min_visible, visible)
            scores[t] = min_visible

        # Sort by score descending
        ranked = np.argsort(scores)[::-1]

        # Select top frames, ensuring >= 5 frame spacing
        selected = []
        for idx in ranked:
            if len(selected) >= num_frames:
                break
            if scores[idx] < self.min_visible_joints:
                break
            if all(abs(idx - s) >= 5 for s in selected):
                selected.append(int(idx))

        return sorted(selected)

    def calibrate_pair(self, kp_view1, kp_view2, view1, view2):
        """Estimate extrinsics between two views.

        Steps:
            1. Collect correspondences (joints visible in both views)
            2. Undistort points
            3. Fundamental matrix via RANSAC
            4. Essential matrix: E = K2.T @ F @ K1
            5. Recover R, t from E

        Args:
            kp_view1: (T, 17, 3) keypoints from view 1.
            kp_view2: (T, 17, 3) keypoints from view 2.
            view1, view2: view names for intrinsic lookup.

        Returns:
            Dict with R, t, inlier_count, reprojection_error.
            None if calibration fails.
        """
        pts1, pts2 = self._collect_correspondences(kp_view1, kp_view2)

        if len(pts1) < 15:
            logger.warning(
                "Only %d correspondences for %s-%s (need >= 15)",
                len(pts1), view1, view2,
            )
            return None

        K1, _ = self.get_intrinsics(view1)
        K2, _ = self.get_intrinsics(view2)

        # Undistort to normalized coordinates
        pts1_norm = self.undistort_keypoints(pts1, view1)
        pts2_norm = self.undistort_keypoints(pts2, view2)

        # Convert back to pixel coordinates for findFundamentalMat
        pts1_px = pts1.astype(np.float64)
        pts2_px = pts2.astype(np.float64)

        # Fundamental matrix
        F, mask = cv2.findFundamentalMat(
            pts1_px, pts2_px, cv2.FM_RANSAC, self.ransac_threshold,
        )
        if F is None or F.shape != (3, 3):
            logger.warning("Fundamental matrix estimation failed for %s-%s", view1, view2)
            return None

        inlier_mask = mask.ravel().astype(bool)
        inlier_count = int(inlier_mask.sum())
        if inlier_count < 10:
            logger.warning("Only %d inliers for %s-%s", inlier_count, view1, view2)
            return None

        # Essential matrix
        E = K2.T @ F @ K1

        # Recover R, t using inlier points (normalized coordinates)
        pts1_inlier = pts1_norm[inlier_mask]
        pts2_inlier = pts2_norm[inlier_mask]

        # recoverPose expects points in normalized coordinates when K=I
        K_eye = np.eye(3, dtype=np.float64)
        _, R, t, pose_mask = cv2.recoverPose(E, pts1_inlier, pts2_inlier, K_eye)

        # Compute reprojection error (approximate)
        reproj_error = self._compute_reproj_error(
            pts1_px[inlier_mask], pts2_px[inlier_mask], F,
        )

        logger.info(
            "  %s-%s: %d inliers / %d total, reproj=%.2f px",
            view1, view2, inlier_count, len(pts1), reproj_error,
        )

        return {
            "R": R,
            "t": t,
            "F": F,
            "E": E,
            "inlier_count": inlier_count,
            "total_correspondences": len(pts1),
            "reprojection_error": reproj_error,
        }

    @staticmethod
    def _compute_reproj_error(pts1, pts2, F):
        """Compute mean Sampson epipolar error."""
        n = len(pts1)
        if n == 0:
            return float("inf")
        pts1_h = np.hstack([pts1, np.ones((n, 1))])
        pts2_h = np.hstack([pts2, np.ones((n, 1))])
        # Epipolar lines
        l2 = (F @ pts1_h.T).T  # lines in view 2
        l1 = (F.T @ pts2_h.T).T  # lines in view 1
        # Sampson distance
        d2 = np.abs(np.sum(pts2_h * l2, axis=1))
        d1 = np.abs(np.sum(pts1_h * l1, axis=1))
        denom2 = l2[:, 0] ** 2 + l2[:, 1] ** 2
        denom1 = l1[:, 0] ** 2 + l1[:, 1] ** 2
        err = d2 / np.sqrt(denom2 + 1e-12) + d1 / np.sqrt(denom1 + 1e-12)
        return float(np.mean(err) / 2)

    def calibrate_patient(self, patient_kp_per_view):
        """Calibrate all camera pairs for a patient.

        FV is the reference frame (R=I, t=0).

        Args:
            patient_kp_per_view: {triplet_base: {view: (T, 17, 3)}}
                Multiple triplets from the same patient.

        Returns:
            Calibration dict with per-view K, dist, R, t, P matrices.

        Raises:
            CalibrationError if calibration fails.
        """
        # Pool child keypoints across all triplets per view
        pooled = self._pool_patient_keypoints(patient_kp_per_view)
        available_views = list(pooled.keys())

        if len(available_views) < 2:
            raise CalibrationError(
                f"Need >= 2 views, got {len(available_views)}: {available_views}"
            )

        # Build result with FV as reference
        result = {}
        quality = {}

        for view in ["FV", "LV", "RV"]:
            K, dist = self.get_intrinsics(view)
            result[view] = {"K": K, "dist": dist, "R": None, "t": None, "P": None}

        # FV is reference
        result["FV"]["R"] = np.eye(3, dtype=np.float64)
        result["FV"]["t"] = np.zeros((3, 1), dtype=np.float64)
        K_fv = result["FV"]["K"]
        result["FV"]["P"] = K_fv @ np.hstack([np.eye(3), np.zeros((3, 1))])

        # Calibrate FV-LV and FV-RV
        ref_view = "FV"
        for other_view in ["LV", "RV"]:
            if ref_view not in pooled or other_view not in pooled:
                logger.warning("Missing %s or %s keypoints, skipping pair",
                               ref_view, other_view)
                continue

            pair_result = self.calibrate_pair(
                pooled[ref_view], pooled[other_view], ref_view, other_view,
            )
            if pair_result is None:
                logger.warning("Calibration failed for %s-%s", ref_view, other_view)
                continue

            R, t = pair_result["R"], pair_result["t"]
            K_other = result[other_view]["K"]
            P = K_other @ np.hstack([R, t])

            result[other_view]["R"] = R
            result[other_view]["t"] = t
            result[other_view]["P"] = P
            quality[f"{ref_view}_{other_view}"] = pair_result["reprojection_error"]

        # Check at least one pair succeeded
        calibrated_views = [v for v in ["FV", "LV", "RV"] if result[v]["P"] is not None]
        if len(calibrated_views) < 2:
            raise CalibrationError(
                f"Only {len(calibrated_views)} views calibrated: {calibrated_views}"
            )

        result["quality"] = quality
        result["calibrated_views"] = calibrated_views
        return result

    @staticmethod
    def _pool_patient_keypoints(patient_kp_per_view):
        """Pool child keypoints across all triplets for each view.

        Args:
            patient_kp_per_view: {triplet_base: {view: (T, 17, 3)}}

        Returns:
            {view: (T_total, 17, 3)} concatenated along time.
        """
        pooled = {}
        for triplet_base, views in patient_kp_per_view.items():
            for view, kp in views.items():
                if view not in pooled:
                    pooled[view] = []
                pooled[view].append(kp)

        return {v: np.concatenate(kps, axis=0) for v, kps in pooled.items()}


def flatten_calibration(calibration):
    """Flatten calibration dict for np.savez_compressed.

    Args:
        calibration: output from PoseCalibrator.calibrate_patient().

    Returns:
        Flat dict suitable for np.savez_compressed.
    """
    flat = {}
    for view in ["FV", "LV", "RV"]:
        vd = calibration[view]
        flat[f"{view}_K"] = vd["K"]
        flat[f"{view}_dist"] = vd["dist"]
        if vd["R"] is not None:
            flat[f"{view}_R"] = vd["R"]
            flat[f"{view}_t"] = vd["t"]
            flat[f"{view}_P"] = vd["P"]

    # Quality metrics
    quality = calibration.get("quality", {})
    for key, val in quality.items():
        flat[f"quality_{key}"] = np.array([val])

    # Calibrated views list
    cal_views = calibration.get("calibrated_views", [])
    flat["calibrated_views"] = np.array(cal_views, dtype="U2")

    return flat


def load_calibration(path):
    """Load calibration from saved .npz file.

    Args:
        path: Path to calibration .npz file.

    Returns:
        Calibration dict with same structure as calibrate_patient() output.
    """
    data = np.load(str(path), allow_pickle=False)

    result = {}
    for view in ["FV", "LV", "RV"]:
        result[view] = {
            "K": data[f"{view}_K"],
            "dist": data[f"{view}_dist"],
            "R": data.get(f"{view}_R"),
            "t": data.get(f"{view}_t"),
            "P": data.get(f"{view}_P"),
        }

    # Quality
    quality = {}
    for key in data.files:
        if key.startswith("quality_") and key != "quality_":
            pair_name = key[len("quality_"):]
            quality[pair_name] = float(data[key][0])
    result["quality"] = quality

    # Calibrated views
    if "calibrated_views" in data:
        result["calibrated_views"] = list(data["calibrated_views"])
    else:
        result["calibrated_views"] = [
            v for v in ["FV", "LV", "RV"] if result[v]["P"] is not None
        ]

    return result
