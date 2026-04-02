"""Visualization utilities.

Skeleton overlay on video frames, mask overlay, and verification
video generation for 2D pose estimation quality review.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO 17-joint skeleton limb connections
COCO_SKELETON = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Arms
    (5, 7), (7, 9), (6, 8), (8, 10),
    # Torso
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Legs
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# BGR colors matching annotate_first_frame.py
IDENTITY_COLORS = {
    "child": (0, 255, 0),       # green
    "caregiver": (0, 165, 255), # orange (BGR)
    "walker": (255, 0, 255),    # magenta
}

DEFAULT_KP_RADIUS = 4
DEFAULT_LIMB_THICKNESS = 2
DEFAULT_MASK_ALPHA = 0.3


def draw_skeleton(frame, keypoints, identity="child", color=None,
                  conf_threshold=0.3, kp_radius=DEFAULT_KP_RADIUS,
                  limb_thickness=DEFAULT_LIMB_THICKNESS):
    """Draw a COCO 17-joint skeleton on a frame.

    Args:
        frame: (H, W, 3) BGR image. Modified in-place.
        keypoints: (17, 3) array with (x, y, confidence).
        identity: Name for color lookup ("child", "caregiver").
        color: Override color (BGR tuple). If None, uses IDENTITY_COLORS.
        conf_threshold: Skip keypoints below this score.
        kp_radius: Radius for keypoint circles.
        limb_thickness: Line thickness for limb connections.

    Returns:
        The modified frame.
    """
    if keypoints is None:
        return frame

    if color is None:
        color = IDENTITY_COLORS.get(identity, (200, 200, 200))

    # Draw limb connections
    for (i, j) in COCO_SKELETON:
        if keypoints[i, 2] < conf_threshold or keypoints[j, 2] < conf_threshold:
            continue
        pt1 = (int(round(keypoints[i, 0])), int(round(keypoints[i, 1])))
        pt2 = (int(round(keypoints[j, 0])), int(round(keypoints[j, 1])))
        cv2.line(frame, pt1, pt2, color, limb_thickness, cv2.LINE_AA)

    # Draw keypoint circles: colored border + white fill
    for k in range(17):
        if keypoints[k, 2] < conf_threshold:
            continue
        pt = (int(round(keypoints[k, 0])), int(round(keypoints[k, 1])))
        cv2.circle(frame, pt, kp_radius + 1, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, max(kp_radius - 1, 1), (255, 255, 255), -1, cv2.LINE_AA)

    return frame


def draw_mask_overlay(frame, mask, identity="child", color=None,
                      alpha=DEFAULT_MASK_ALPHA):
    """Overlay a semi-transparent colored mask on a frame.

    Args:
        frame: (H, W, 3) BGR image. Modified in-place.
        mask: (H, W) boolean or uint8 mask.
        identity: Name for color lookup.
        color: Override color (BGR). If None, uses IDENTITY_COLORS.
        alpha: Transparency (0=invisible, 1=opaque).

    Returns:
        The modified frame.
    """
    if mask is None or not mask.any():
        return frame

    if color is None:
        color = IDENTITY_COLORS.get(identity, (200, 200, 200))

    mask_bool = mask.astype(bool)
    frame[mask_bool] = (
        np.array(color, dtype=np.float32) * alpha
        + frame[mask_bool].astype(np.float32) * (1 - alpha)
    ).astype(np.uint8)

    return frame


def draw_verification_frame(frame, keypoints_dict, masks_dict=None,
                            frame_idx=0, total_frames=0, clip_id="",
                            conf_threshold=0.3, mask_alpha=DEFAULT_MASK_ALPHA):
    """Draw a complete verification frame with skeletons, masks, and HUD.

    Args:
        frame: (H, W, 3) BGR image (will be copied).
        keypoints_dict: Mapping identity -> (17, 3) keypoints for this frame.
        masks_dict: Optional mapping identity -> (H, W) mask for this frame.
        frame_idx: Current frame number.
        total_frames: Total frames in clip.
        clip_id: Clip identifier string.
        conf_threshold: Keypoint confidence threshold.
        mask_alpha: Mask overlay transparency.

    Returns:
        New annotated frame (copy of input).
    """
    out = frame.copy()

    # Draw masks first (behind skeletons)
    if masks_dict:
        for identity in ["walker", "caregiver", "child"]:
            if identity in masks_dict:
                draw_mask_overlay(out, masks_dict[identity], identity,
                                  alpha=mask_alpha)

    # Draw skeletons
    for identity in ["caregiver", "child"]:
        if identity in keypoints_dict and keypoints_dict[identity] is not None:
            draw_skeleton(out, keypoints_dict[identity], identity,
                          conf_threshold=conf_threshold)

    # HUD bar
    h, w = out.shape[:2]
    bar_h = 32
    cv2.rectangle(out, (0, 0), (w, bar_h), (0, 0, 0), -1)

    # Frame info
    info = f"{clip_id}  Frame {frame_idx}/{total_frames}"
    cv2.putText(out, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)

    # Identity legend (right side)
    legend_x = max(0, w - 300)
    for identity, color in [("child", IDENTITY_COLORS["child"]),
                            ("caregiver", IDENTITY_COLORS["caregiver"])]:
        if identity in keypoints_dict and keypoints_dict[identity] is not None:
            cv2.circle(out, (legend_x, 16), 6, color, -1)
            cv2.putText(out, identity, (legend_x + 12, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            legend_x += 110

    return out


def write_verification_video(output_path, frame_generator, fps=30.0,
                             codec="mp4v"):
    """Write frames from a generator to an MP4 video file.

    Args:
        output_path: Output .mp4 file path.
        frame_generator: Iterator yielding (H, W, 3) BGR frames.
        fps: Output video frame rate.
        codec: FourCC codec string.

    Returns:
        Path to the written video file, or None if no frames were written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    count = 0

    try:
        for frame in frame_generator:
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
                if not writer.isOpened():
                    logger.error("Failed to open video writer: %s", output_path)
                    return None
            writer.write(frame)
            count += 1
    finally:
        if writer is not None:
            writer.release()

    if count == 0:
        return None

    logger.info("Wrote %d frames to %s", count, output_path)
    return output_path


def create_side_by_side(left, right, labels=("Original", "Overlay")):
    """Create a side-by-side comparison frame.

    Args:
        left: (H, W, 3) BGR image.
        right: (H, W, 3) BGR image (resized to match left height).
        labels: Text labels for each side.

    Returns:
        (H, W*2, 3) combined frame.
    """
    h, w = left.shape[:2]
    if right.shape[:2] != (h, w):
        right = cv2.resize(right, (w, h))

    combined = np.hstack([left, right])

    # Labels
    cv2.putText(combined, labels[0], (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(combined, labels[1], (w + 8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return combined
