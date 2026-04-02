"""Manual first-frame annotation tool for SAM2 tracking.

Opens each video clip, displays the first frame, and lets the user click
on the child, caregiver (optional), and walker (optional) to create
point prompts for SAM2 video propagation.

Usage:
    python scripts/annotate_first_frame.py --data-root ./data/raw_synced
    python scripts/annotate_first_frame.py --data-root ./data/raw_synced --resume
    python scripts/annotate_first_frame.py --data-root ./data/raw_synced --clip kku_w_01_FV

Controls:
    Left click  : Place point for current object
    N           : Skip to next object (child -> caregiver -> walker)
    S           : Skip caregiver/walker (mark as absent)
    R           : Reset all points for this clip
    Enter       : Confirm and save annotation for this clip
    Q           : Quit (progress is auto-saved)
    Esc         : Quit without saving current clip
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ANNOTATION_OUTPUT = "data/metadata/sam2_annotations.json"

OBJECT_ORDER = ["child", "caregiver", "walker"]
OBJECT_COLORS = {
    "child": (0, 255, 0),       # green
    "caregiver": (255, 165, 0),  # orange (BGR)
    "walker": (255, 0, 255),     # magenta
}
OBJECT_REQUIRED = {"child": True, "caregiver": False, "walker": False}


class AnnotationState:
    """Tracks annotation state for a single clip."""

    def __init__(self):
        self.points = {}
        self.current_object_idx = 0
        self.done = False
        self.skipped_objects = set()

    @property
    def current_object(self):
        if self.current_object_idx < len(OBJECT_ORDER):
            return OBJECT_ORDER[self.current_object_idx]
        return None

    def add_point(self, x, y):
        obj = self.current_object
        if obj is not None:
            self.points[obj] = (int(x), int(y))

    def next_object(self):
        self.current_object_idx += 1

    def skip_current(self):
        obj = self.current_object
        if obj and not OBJECT_REQUIRED.get(obj, False):
            self.skipped_objects.add(obj)
            self.next_object()
            return True
        return False

    def reset(self):
        self.points.clear()
        self.current_object_idx = 0
        self.skipped_objects.clear()

    def is_complete(self):
        for obj in OBJECT_ORDER:
            if OBJECT_REQUIRED.get(obj, False) and obj not in self.points:
                return False
        return True

    def to_dict(self):
        result = {"frame_idx": 0}
        for obj in OBJECT_ORDER:
            if obj in self.points:
                result[obj] = list(self.points[obj])
            elif obj in self.skipped_objects:
                result[obj] = None
        return result


def draw_annotation_frame(frame, state):
    """Draw current annotation state on frame."""
    display = frame.copy()
    h, w = display.shape[:2]

    # Draw existing points
    for obj_name, point in state.points.items():
        color = OBJECT_COLORS.get(obj_name, (255, 255, 255))
        cv2.circle(display, point, 8, color, -1)
        cv2.circle(display, point, 10, color, 2)
        cv2.putText(display, obj_name, (point[0] + 12, point[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw current prompt
    current = state.current_object
    if current is not None:
        required = OBJECT_REQUIRED.get(current, False)
        prompt = f"Click on: {current}"
        if not required:
            prompt += "  (press S to skip)"
        color = OBJECT_COLORS.get(current, (255, 255, 255))
        cv2.rectangle(display, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(display, prompt, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        cv2.rectangle(display, (0, 0), (w, 40), (0, 80, 0), -1)
        cv2.putText(display, "All objects marked. Press ENTER to confirm, R to reset.",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw instructions at bottom
    instructions = "N=next | S=skip | R=reset | Enter=confirm | Q=quit"
    cv2.rectangle(display, (0, h - 30), (w, h), (0, 0, 0), -1)
    cv2.putText(display, instructions, (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return display


def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks for point placement."""
    state = param
    if event == cv2.EVENT_LBUTTONDOWN and state.current_object is not None:
        state.add_point(x, y)
        state.next_object()


def annotate_clip(video_path, window_name="Annotate"):
    """Run annotation GUI for a single clip.

    Returns:
        dict with annotation data, or None if skipped/quit.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open: %s", video_path)
        return None

    ret, frame = cap.read()
    cap.release()
    if not ret:
        logger.warning("Cannot read first frame: %s", video_path)
        return None

    state = AnnotationState()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback, state)

    quit_requested = False

    while True:
        display = draw_annotation_frame(frame, state)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            quit_requested = True
            break
        elif key == 27:  # Esc
            return None
        elif key == ord("r"):
            state.reset()
        elif key == ord("s"):
            if not state.skip_current():
                logger.info("Cannot skip required object: %s", state.current_object)
        elif key == ord("n"):
            state.next_object()
        elif key == 13:  # Enter
            if state.is_complete():
                return state.to_dict()
            else:
                logger.info("Annotation incomplete. Mark all required objects first.")

    if quit_requested:
        return "QUIT"
    return None


def discover_clips(data_root):
    """Find all video clips in the data directory."""
    data_root = Path(data_root)
    extensions = {".mp4", ".avi", ".mov", ".mkv"}
    clips = []
    for ext in extensions:
        clips.extend(data_root.rglob(f"*{ext}"))
    clips.sort(key=lambda p: p.stem)
    return clips


def load_existing_annotations(output_path):
    """Load existing annotations for resume support."""
    output_path = Path(output_path)
    if output_path.exists():
        with open(output_path) as f:
            return json.load(f)
    return {}


def save_annotations(annotations, output_path):
    """Save annotations to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Annotate first frames for SAM2 tracking")
    parser.add_argument("--data-root", required=True, help="Root directory with video clips")
    parser.add_argument("--output", default=ANNOTATION_OUTPUT,
                        help="Output annotation JSON path")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-annotated clips")
    parser.add_argument("--clip", type=str, default=None,
                        help="Annotate a specific clip (stem name, e.g., kku_w_01_FV)")
    parser.add_argument("--re-annotate", type=str, default=None,
                        help="Re-annotate a specific clip (overwrite existing)")
    args = parser.parse_args()

    clips = discover_clips(args.data_root)
    if not clips:
        logger.error("No video clips found in %s", args.data_root)
        sys.exit(1)
    logger.info("Found %d video clips", len(clips))

    annotations = load_existing_annotations(args.output)

    # Filter clips
    if args.clip:
        clips = [c for c in clips if c.stem == args.clip]
        if not clips:
            logger.error("Clip not found: %s", args.clip)
            sys.exit(1)
    elif args.re_annotate:
        clips = [c for c in clips if c.stem == args.re_annotate]
        if not clips:
            logger.error("Clip not found: %s", args.re_annotate)
            sys.exit(1)
    elif args.resume:
        clips = [c for c in clips if c.stem not in annotations]
        logger.info("Resuming: %d clips remaining", len(clips))

    annotated_count = 0
    for i, clip_path in enumerate(clips):
        clip_id = clip_path.stem
        logger.info("[%d/%d] Annotating: %s", i + 1, len(clips), clip_id)

        result = annotate_clip(clip_path)

        if result == "QUIT":
            logger.info("Quit requested. Saving progress...")
            break
        elif result is not None:
            annotations[clip_id] = result
            annotations[clip_id]["video_path"] = str(clip_path.relative_to(args.data_root))
            annotated_count += 1
            save_annotations(annotations, args.output)
            logger.info("Saved annotation for %s (%d total)", clip_id, len(annotations))
        else:
            logger.info("Skipped: %s", clip_id)

    cv2.destroyAllWindows()
    save_annotations(annotations, args.output)
    logger.info("Done. Annotated %d clips this session. Total: %d", annotated_count, len(annotations))


if __name__ == "__main__":
    main()
