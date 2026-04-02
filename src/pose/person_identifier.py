"""Mask-guided person identification from multi-person pose detections.

Assigns detected skeletons to tracked identity masks (child, caregiver)
produced by SAM2. Falls back to height-ratio heuristic when masks are
unavailable.

Primary method: Compute IoU between each skeleton's bounding box and each
person mask. Assign skeleton to the mask with highest overlap.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO 17-joint keypoint indices
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def keypoints_to_bbox(keypoints, confidence=None, conf_threshold=0.3, padding=10):
    """Convert keypoints to bounding box.

    Args:
        keypoints: (17, 2) or (17, 3) array of keypoints.
        confidence: Optional (17,) array of confidence scores.
            If keypoints is (17, 3), the third column is used as confidence.
        conf_threshold: Minimum confidence to include a keypoint.
        padding: Pixel padding around the bounding box.

    Returns:
        (x1, y1, x2, y2) bounding box, or None if too few valid keypoints.
    """
    if keypoints.shape[-1] == 3:
        confidence = keypoints[:, 2]
        keypoints = keypoints[:, :2]

    if confidence is not None:
        valid = confidence > conf_threshold
        keypoints = keypoints[valid]

    if len(keypoints) < 3:
        return None

    x1 = keypoints[:, 0].min() - padding
    y1 = keypoints[:, 1].min() - padding
    x2 = keypoints[:, 0].max() + padding
    y2 = keypoints[:, 1].max() + padding
    return (int(x1), int(y1), int(x2), int(y2))


def compute_mask_bbox_iou(mask, bbox):
    """Compute IoU between a binary mask and a bounding box.

    Args:
        mask: (H, W) binary mask.
        bbox: (x1, y1, x2, y2) bounding box.

    Returns:
        IoU score (0-1).
    """
    h, w = mask.shape
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    # Mask pixels within bbox
    mask_roi = mask[y1:y2, x1:x2]
    mask_area = mask.sum()
    bbox_area = (x2 - x1) * (y2 - y1)
    intersection = mask_roi.sum()

    union = mask_area + bbox_area - intersection
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def compute_mask_keypoint_overlap(mask, keypoints, confidence=None, conf_threshold=0.3):
    """Compute fraction of keypoints that fall within a mask.

    More robust than bbox IoU when persons overlap.

    Args:
        mask: (H, W) binary mask.
        keypoints: (17, 2) or (17, 3) array.
        confidence: Optional (17,) confidence array.
        conf_threshold: Minimum confidence.

    Returns:
        Fraction of valid keypoints inside the mask (0-1).
    """
    if keypoints.shape[-1] == 3:
        confidence = keypoints[:, 2]
        keypoints = keypoints[:, :2]

    h, w = mask.shape
    count_in = 0
    count_valid = 0

    for i in range(len(keypoints)):
        if confidence is not None and confidence[i] < conf_threshold:
            continue
        x, y = int(round(keypoints[i, 0])), int(round(keypoints[i, 1]))
        if 0 <= x < w and 0 <= y < h:
            count_valid += 1
            if mask[y, x]:
                count_in += 1

    if count_valid == 0:
        return 0.0
    return count_in / count_valid


def assign_skeletons_mask_guided(detected_skeletons, person_masks, conf_threshold=0.3):
    """Assign detected skeletons to person identity masks.

    Uses keypoint overlap as primary metric, falls back to bbox IoU.

    Args:
        detected_skeletons: List of (17, 3) arrays (x, y, confidence).
        person_masks: Dict mapping identity name -> (H, W) binary mask.
            Example: {"child": child_mask, "caregiver": caregiver_mask}

    Returns:
        Dict mapping identity name -> (17, 3) skeleton array.
        Unmatched identities get None.
    """
    if not detected_skeletons or not person_masks:
        return {name: None for name in person_masks}

    identity_names = list(person_masks.keys())
    n_skeletons = len(detected_skeletons)
    n_identities = len(identity_names)

    # Compute overlap scores: (n_skeletons, n_identities)
    scores = np.zeros((n_skeletons, n_identities))
    for si, skeleton in enumerate(detected_skeletons):
        for ii, name in enumerate(identity_names):
            mask = person_masks[name]
            # Primary: keypoint overlap
            kp_score = compute_mask_keypoint_overlap(
                mask, skeleton, conf_threshold=conf_threshold
            )
            # Secondary: bbox IoU
            bbox = keypoints_to_bbox(skeleton, conf_threshold=conf_threshold)
            bbox_score = compute_mask_bbox_iou(mask, bbox) if bbox else 0.0
            # Combined score (keypoint overlap is more reliable)
            scores[si, ii] = 0.7 * kp_score + 0.3 * bbox_score

    # Greedy assignment: match highest scores first
    assignments = {}
    used_skeletons = set()
    used_identities = set()

    flat_order = np.argsort(scores.ravel())[::-1]
    for flat_idx in flat_order:
        si = flat_idx // n_identities
        ii = flat_idx % n_identities
        if si in used_skeletons or ii in used_identities:
            continue
        if scores[si, ii] < 0.05:
            continue
        name = identity_names[ii]
        assignments[name] = detected_skeletons[si]
        used_skeletons.add(si)
        used_identities.add(ii)

    # Fill unmatched identities with None
    for name in identity_names:
        if name not in assignments:
            assignments[name] = None

    return assignments


def assign_skeletons_height_ratio(detected_skeletons, min_height_ratio=1.3):
    """Fallback: assign skeletons by height ratio (child = shortest).

    Used when SAM2 masks are not available.

    Args:
        detected_skeletons: List of (17, 3) arrays.
        min_height_ratio: Minimum adult/child height ratio.

    Returns:
        Dict with "child" and optionally "caregiver" skeleton arrays.
    """
    if not detected_skeletons:
        return {"child": None, "caregiver": None}

    if len(detected_skeletons) == 1:
        return {"child": detected_skeletons[0], "caregiver": None}

    # Compute skeleton heights (top of head to ankle)
    heights = []
    for skeleton in detected_skeletons:
        kp = skeleton[:, :2]
        conf = skeleton[:, 2]
        valid = conf > 0.3
        if valid.sum() < 3:
            heights.append(0)
            continue
        y_coords = kp[valid, 1]
        heights.append(y_coords.max() - y_coords.min())

    heights = np.array(heights)
    sorted_idx = np.argsort(heights)

    child_idx = sorted_idx[0]
    result = {"child": detected_skeletons[child_idx], "caregiver": None}

    if len(detected_skeletons) >= 2:
        caregiver_idx = sorted_idx[-1]
        if heights[caregiver_idx] > 0 and heights[child_idx] > 0:
            ratio = heights[caregiver_idx] / heights[child_idx]
            if ratio >= min_height_ratio:
                result["caregiver"] = detected_skeletons[caregiver_idx]
            else:
                logger.debug(
                    "Height ratio %.2f < %.2f, caregiver assignment uncertain",
                    ratio, min_height_ratio
                )
                result["caregiver"] = detected_skeletons[caregiver_idx]

    return result


def identify_persons(detected_skeletons, person_masks=None, min_height_ratio=1.3,
                     conf_threshold=0.3):
    """Top-level person identification: mask-guided with height fallback.

    Args:
        detected_skeletons: List of (17, 3) arrays (x, y, confidence).
        person_masks: Optional dict of identity masks from SAM2.
            If provided, uses mask-guided assignment.
            If None, falls back to height-ratio heuristic.
        min_height_ratio: Fallback height ratio threshold.
        conf_threshold: Keypoint confidence threshold.

    Returns:
        Dict mapping identity name -> (17, 3) skeleton or None.
    """
    if person_masks is not None and len(person_masks) > 0:
        return assign_skeletons_mask_guided(
            detected_skeletons, person_masks, conf_threshold=conf_threshold
        )
    else:
        return assign_skeletons_height_ratio(
            detected_skeletons, min_height_ratio=min_height_ratio
        )
