#!/usr/bin/env python3
"""Terminal-based clip annotation tool for per-clip GMFCS metadata.

Plays front-view video in an OpenCV window while you type numbered
choices in the terminal. Annotates per triplet (3 views share one
annotation). Fast: ~7 keystrokes per clip.

Output: data/metadata/clip_annotations.json
Also syncs to: data/metadata/assistive_annotations.json

Usage:
    python scripts/annotate_clips.py
    python scripts/annotate_clips.py --clips-dir data/video_clips
"""

import argparse
import json
import sys
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── File parsing ────────────────────────────────────────────────────────

VIEW_CODES = {"f": "front", "l": "left", "r": "right"}

MOVEMENT_OPTIONS = [
    ("w",   "Walk"),
    ("cr",  "Crawl"),
    ("c_s", "Sit → Stand"),
    ("s_c", "Stand → Sit"),
    ("sr",  "Side Roll"),
]

SURFACE_OPTIONS = [("floor", "Floor"), ("chair", "Chair")]

AFO_OPTIONS = [
    ("bilateral",        "Bilateral"),
    ("unilateral_left",  "Uni-L"),
    ("unilateral_right", "Uni-R"),
    ("none",             "None"),
]

YESNO = [(True, "Yes"), (False, "No")]

FIM_LABELS = {
    0: "Dependent", 1: "MaxAssist", 2: "ModAssist", 3: "MinAssist",
    4: "ContactGuard", 5: "Supervision/Device", 6: "Independent",
}

# Movement + surface → final movement code
SURFACE_RESOLVE = {
    ("c_s", "floor"): "c_s",   ("c_s", "chair"): "cc_s",
    ("s_c", "floor"): "s_c",   ("s_c", "chair"): "s_cc",
}


def parse_clip_filename(filepath):
    """Parse {patient}_{view}_{movement}_{number}.{ext}."""
    stem = Path(filepath).stem
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    patient, view, number = parts[0], parts[1], parts[-1]
    movement = "_".join(parts[2:-1])
    if view not in VIEW_CODES:
        return None
    return patient, view, movement, number


def discover_triplets(clips_dir):
    """Group video files into triplets by (patient, movement, number)."""
    clips_dir = Path(clips_dir)
    groups = defaultdict(lambda: {"clips": {}})

    for ext in ("*.mov", "*.mp4", "*.avi"):
        for path in sorted(clips_dir.rglob(ext)):
            parsed = parse_clip_filename(path)
            if parsed is None:
                continue
            patient, view, movement, number = parsed
            key = (patient, movement, number)
            groups[key]["patient_id"] = patient
            groups[key]["movement"] = movement
            groups[key]["number"] = number
            groups[key]["clips"][view] = str(path)

    triplets = []
    for key in sorted(groups.keys()):
        g = groups[key]
        g["triplet_id"] = f"{g['patient_id']}_{g['movement']}_{g['number']}"
        triplets.append(g)
    return triplets


# ── Video player (background thread) ───────────────────────────────────

class VideoPlayer:
    """Plays a video in an OpenCV window on a background thread."""

    def __init__(self):
        self._stop = threading.Event()
        self._thread = None
        self._path = None

    def play(self, video_path, window_name="Clip Viewer"):
        """Start playing video (loops until stop() is called)."""
        self.stop()
        self._stop.clear()
        self._path = video_path
        self._thread = threading.Thread(
            target=self._loop, args=(video_path, window_name), daemon=True
        )
        self._thread.start()

    def _loop(self, path, window_name):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = max(1, int(1000 / fps))

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            # Resize for display if too large
            h, w = frame.shape[:2]
            if w > 960:
                scale = 960 / w
                frame = cv2.resize(frame, (960, int(h * scale)))
            try:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(delay) & 0xFF == ord("q"):
                    break
            except Exception:
                break

        cap.release()
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# ── Terminal prompt helpers ─────────────────────────────────────────────

def prompt_choice(label, options, default_idx=None):
    """Prompt user to pick from numbered options.

    Args:
        label: Field name to display.
        options: List of (value, display_label) tuples.
        default_idx: 1-based index of default (Enter to accept).

    Returns:
        Selected value.
    """
    print(f"\n  {label}:")
    for i, (val, disp) in enumerate(options, 1):
        marker = " *" if i == default_idx else ""
        print(f"    {i}) {disp}{marker}")

    if default_idx:
        prompt_str = f"  Enter [default={default_idx}]: "
    else:
        prompt_str = f"  Enter (1-{len(options)}): "

    while True:
        raw = input(prompt_str).strip()
        if raw == "" and default_idx is not None:
            return options[default_idx - 1][0]
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
        except ValueError:
            pass
        print(f"    → Please enter a number 1-{len(options)}")


def prompt_fim():
    """Prompt for FIM score (0-6)."""
    print(f"\n  FIM Score:")
    print(f"    0=Dependent  1=MaxAssist  2=ModAssist  3=MinAssist")
    print(f"    4=ContactGuard  5=Supervision/Device  6=Independent")
    while True:
        raw = input("  Enter (0-6): ").strip()
        try:
            val = int(raw)
            if 0 <= val <= 6:
                return val
        except ValueError:
            pass
        print("    → Please enter a number 0-6")


# ── Save / sync ────────────────────────────────────────────────────────

def save_annotations(output_path, annotations, total_triplets):
    """Write annotations to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "created_at": datetime.now().isoformat(),
        "total_triplets": total_triplets,
        "annotated": len(annotations),
        "triplets": list(annotations.values()),
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def sync_to_assistive_annotations(annotations):
    """Update assistive_annotations.json from clip annotations."""
    ann_path = PROJECT_ROOT / "data" / "metadata" / "assistive_annotations.json"
    if not ann_path.exists():
        return

    with open(ann_path) as f:
        aa = json.load(f)

    patients = {p["patient_id"]: p for p in aa["patients"]}

    mv_key_map = {
        "w": "walk", "cr": "crawl",
        "c_s": "seated_to_standing", "s_c": "standing_to_seated",
        "sr": "side_rolling",
        "cc_s": "chair_seated_to_standing", "s_cc": "standing_to_chair_seated",
    }

    for ann in annotations.values():
        pid = ann["patient_id"]
        if pid not in patients:
            continue

        patient = patients[pid]
        mv_key = mv_key_map.get(ann["movement"])
        if mv_key is None:
            continue

        # Build device string
        has_walker = ann.get("walker", False)
        has_afo = ann.get("afo", "none") != "none"
        has_acrylic = ann.get("acrylic_stand", False)

        if has_walker and has_afo:
            device_str = "walker_and_afo"
        elif has_walker:
            device_str = "walker"
        elif has_afo:
            device_str = "afo_only"
        elif has_acrylic:
            device_str = "acrylic_stand"
        else:
            device_str = "none"

        if "per_movement" not in patient:
            patient["per_movement"] = {}
        patient["per_movement"][mv_key] = {
            "device": device_str,
            "assistance": round(ann.get("fim", 0) / 6.0, 3),
            "notes": "",
        }

        # Update patient-level devices
        patient_anns = [a for a in annotations.values() if a["patient_id"] == pid]
        patient["devices"]["walker"] = any(a.get("walker") for a in patient_anns)
        patient["devices"]["walker_type"] = (
            "anterior" if patient["devices"]["walker"] else "none"
        )
        patient["devices"]["afo"] = any(
            a.get("afo", "none") != "none" for a in patient_anns
        )
        afo_vals = [a["afo"] for a in patient_anns if a.get("afo", "none") != "none"]
        patient["devices"]["afo_laterality"] = afo_vals[0] if afo_vals else "none"
        patient["devices"]["acrylic_stand"] = any(
            a.get("acrylic_stand") for a in patient_anns
        )

        fim_vals = [a["fim"] for a in patient_anns if "fim" in a]
        if fim_vals:
            patient["overall_assistance"] = max(fim_vals)

    aa["patients"] = list(patients.values())
    with open(ann_path, "w") as f:
        json.dump(aa, f, indent=2, ensure_ascii=False)
    print(f"  Synced to {ann_path}")


# ── Confirmation helpers ───────────────────────────────────────────────

def _label_lookup(options):
    """Build {value: display_label} from options list."""
    return {v: d for v, d in options}


_MOVEMENT_LABELS = _label_lookup(MOVEMENT_OPTIONS)
_AFO_LABELS = _label_lookup(AFO_OPTIONS)


def format_annotation_summary(ann):
    """Return a human-readable summary of an annotation dict."""
    mv = ann["movement"]
    mv_label = _MOVEMENT_LABELS.get(mv, mv)
    if mv in ("cc_s", "s_cc"):
        mv_label = _MOVEMENT_LABELS.get(mv[:3] if mv == "s_cc" else mv[1:], mv)
        mv_label += " (Chair)"
    elif mv in ("c_s", "s_c"):
        mv_label += " (Floor)"

    afo_label = _AFO_LABELS.get(ann["afo"], ann["afo"])
    fim = ann["fim"]
    fim_label = f"{fim} ({FIM_LABELS.get(fim, '?')})"

    lines = [
        f"  Movement:   {mv_label}",
        f"  AFO:        {afo_label}",
        f"  Walker:     {'Yes' if ann['walker'] else 'No'}",
        f"  Acrylic:    {'Yes' if ann['acrylic_stand'] else 'No'}",
        f"  Caregiver:  {'Yes' if ann['caregiver_assistance'] else 'No'}",
        f"  FIM:        {fim_label}",
    ]

    width = max(len(l) for l in lines) + 2
    border = "─" * (width - 2)
    summary = f"\n  ┌─ Summary {border[10:]}┐\n"
    for l in lines:
        summary += f"  │{l}{' ' * (width - len(l) - 1)}│\n"
    summary += f"  └{border}┘"
    return summary


def confirm_annotation(ann):
    """Show summary and ask user to confirm. Returns True if confirmed."""
    print(format_annotation_summary(ann))
    while True:
        raw = input("\n  Correct? (Y/n): ").strip().lower()
        if raw in ("", "y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("    → Please enter Y or N")


# ── Main annotation loop ───────────────────────────────────────────────

def annotate_triplet(triplet, player):
    """Annotate one triplet interactively. Returns annotation dict or None."""
    tid = triplet["triplet_id"]
    pid = triplet["patient_id"]
    mv_from_file = triplet["movement"]

    # Start video playback
    video_path = triplet["clips"].get("f") or next(iter(triplet["clips"].values()))
    player.play(video_path, window_name=f"{tid}")

    print(f"\n{'='*55}")
    print(f"  Patient: {pid}  |  Triplet: {tid}")
    print(f"  Views: {', '.join(sorted(triplet['clips'].keys()))}")
    print(f"  (Video playing — press q in video window to close)")
    print(f"{'='*55}")

    # --- Movement ---
    # Find default index from filename
    mv_codes = [opt[0] for opt in MOVEMENT_OPTIONS]
    default_mv = None
    if mv_from_file in mv_codes:
        default_mv = mv_codes.index(mv_from_file) + 1
    elif mv_from_file == "cc_s":
        default_mv = mv_codes.index("c_s") + 1
    elif mv_from_file == "s_cc":
        default_mv = mv_codes.index("s_c") + 1

    movement = prompt_choice("Movement", MOVEMENT_OPTIONS, default_idx=default_mv)

    # --- Surface (only for sit/stand) ---
    surface = None
    if movement in ("c_s", "s_c"):
        default_surf = None
        if mv_from_file in ("cc_s", "s_cc"):
            default_surf = 2  # chair
        surface = prompt_choice("Surface", SURFACE_OPTIONS, default_idx=default_surf)
        final_movement = SURFACE_RESOLVE.get((movement, surface), movement)
    else:
        final_movement = movement

    # --- AFO ---
    afo = prompt_choice("AFO", AFO_OPTIONS)

    # --- Walker ---
    walker = prompt_choice("Walker", YESNO)

    # --- Acrylic Stand ---
    acrylic = prompt_choice("Acrylic Stand", YESNO)

    # --- Caregiver Assistance ---
    caregiver = prompt_choice("Caregiver Assist", YESNO)

    # --- FIM ---
    fim = prompt_fim()

    player.stop()

    # Build annotation
    ann = {
        "triplet_id": tid,
        "patient_id": pid,
        "movement": final_movement,
        "afo": afo,
        "walker": walker,
        "acrylic_stand": acrylic,
        "caregiver_assistance": caregiver,
        "fim": fim,
        "clips": list(triplet["clips"].keys()),
        "annotated_at": datetime.now().isoformat(),
    }
    if final_movement in ("c_s", "s_c"):
        ann["surface"] = "floor"
    elif final_movement in ("cc_s", "s_cc"):
        ann["surface"] = "chair"

    return ann


def main():
    parser = argparse.ArgumentParser(description="Terminal clip annotator")
    parser.add_argument("--clips-dir", default="data/video_clips")
    parser.add_argument("--output", default="data/metadata/clip_annotations.json")
    args = parser.parse_args()

    clips_dir = PROJECT_ROOT / args.clips_dir
    output_path = PROJECT_ROOT / args.output

    if not clips_dir.exists():
        print(f"ERROR: {clips_dir} not found")
        sys.exit(1)

    triplets = discover_triplets(clips_dir)
    if not triplets:
        print(f"ERROR: No video clips found in {clips_dir}")
        sys.exit(1)

    # Load existing annotations
    annotations = {}
    if output_path.exists():
        with open(output_path) as f:
            data = json.load(f)
        annotations = {a["triplet_id"]: a for a in data.get("triplets", [])}

    total = len(triplets)
    done = len(annotations)

    print(f"\n  GMFCS Clip Annotator")
    print(f"  Clips: {clips_dir}")
    print(f"  Found: {total} triplets, {done} already annotated")
    print(f"  Output: {output_path}")
    print(f"  Type number + Enter for each field. Ctrl+C to quit.\n")

    player = VideoPlayer()

    try:
        for i, triplet in enumerate(triplets):
            tid = triplet["triplet_id"]

            # Skip already annotated
            if tid in annotations:
                print(f"  [{i+1}/{total}] {tid} — already done, skipping")
                continue

            ann = annotate_triplet(triplet, player)
            if ann:
                annotations[tid] = ann
                save_annotations(output_path, annotations, total)
                done_now = len(annotations)
                print(f"\n  ✓ Saved! ({done_now}/{total} done)")

    except KeyboardInterrupt:
        print("\n\n  Interrupted.")
    finally:
        player.stop()
        if annotations:
            save_annotations(output_path, annotations, total)
            sync_to_assistive_annotations(annotations)
        print(f"\n  Final: {len(annotations)}/{total} triplets annotated.")
        print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
