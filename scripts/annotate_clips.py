#!/usr/bin/env python3
"""Click-to-annotate GUI for per-clip GMFCS metadata.

Plays front-view video clips and records movement type, assistive devices,
and assistance levels via mouse clicks only. Annotates per triplet (one
annotation covers all 3 views: front/left/right).

Output: data/metadata/clip_annotations.json
Also syncs back to: data/metadata/assistive_annotations.json

Usage:
    python scripts/annotate_clips.py
    python scripts/annotate_clips.py --clips-dir data/video_clips
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError:
    print("ERROR: tkinter not available. Install python3-tk.")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Constants ───────────────────────────────────────────────────────────

VIEW_CODES = {"f": "front", "l": "left", "r": "right"}
FRONT_VIEW = "f"

MOVEMENT_LABELS = {
    "w": "Walk",
    "cr": "Crawl",
    "c_s": "Sit → Stand",
    "s_c": "Stand → Sit",
    "sr": "Side Roll",
}

# After resolving surface, these are the final movement codes
SURFACE_MOVEMENTS = {
    ("c_s", "floor"): "c_s",
    ("c_s", "chair"): "cc_s",
    ("s_c", "floor"): "s_c",
    ("s_c", "chair"): "s_cc",
}

AFO_OPTIONS = ["Bilateral", "Uni-L", "Uni-R", "None"]
FIM_OPTIONS = [0, 1, 2, 3, 4, 5, 6]

# Colors
COLOR_BG = "#2b2b2b"
COLOR_PANEL = "#363636"
COLOR_BTN = "#4a4a4a"
COLOR_BTN_TEXT = "#ffffff"
COLOR_SELECTED = "#2196F3"
COLOR_SELECTED_TEXT = "#ffffff"
COLOR_LABEL = "#b0b0b0"
COLOR_HEADER = "#1a1a1a"
COLOR_SAVE = "#4CAF50"
COLOR_DONE = "#66BB6A"
COLOR_SKIP = "#FF9800"

# Video display size
VIDEO_W, VIDEO_H = 768, 432

# ── File parsing ────────────────────────────────────────────────────────


def parse_clip_filename(filepath):
    """Parse video clip filename into components.

    Format: {patient}_{view}_{movement}_{number}.{ext}
    Examples:
        jyh_f_c_s_01.mov → ('jyh', 'f', 'c_s', '01')
        pjw_l_w_02.mov   → ('pjw', 'l', 'w', '02')
    """
    stem = Path(filepath).stem
    parts = stem.split("_")
    if len(parts) < 4:
        return None

    patient = parts[0]
    view = parts[1]
    number = parts[-1]
    movement = "_".join(parts[2:-1])

    if view not in VIEW_CODES:
        return None

    return patient, view, movement, number


def discover_triplets(clips_dir):
    """Find all triplets (grouped by patient+movement+number).

    Returns list of dicts sorted by patient, then movement, then number:
        [{
            'triplet_id': 'jyh_c_s_01',
            'patient_id': 'jyh',
            'movement': 'c_s',
            'number': '01',
            'clips': {'f': path, 'l': path, 'r': path},
        }, ...]
    """
    clips_dir = Path(clips_dir)
    groups = defaultdict(lambda: {"clips": {}})

    for video_path in sorted(clips_dir.rglob("*.mov")):
        parsed = parse_clip_filename(video_path)
        if parsed is None:
            continue
        patient, view, movement, number = parsed
        key = (patient, movement, number)
        groups[key]["patient_id"] = patient
        groups[key]["movement"] = movement
        groups[key]["number"] = number
        groups[key]["clips"][view] = str(video_path)

    # Also handle .mp4
    for video_path in sorted(clips_dir.rglob("*.mp4")):
        parsed = parse_clip_filename(video_path)
        if parsed is None:
            continue
        patient, view, movement, number = parsed
        key = (patient, movement, number)
        groups[key]["patient_id"] = patient
        groups[key]["movement"] = movement
        groups[key]["number"] = number
        groups[key]["clips"][view] = str(video_path)

    triplets = []
    for key in sorted(groups.keys()):
        g = groups[key]
        g["triplet_id"] = f"{g['patient_id']}_{g['movement']}_{g['number']}"
        triplets.append(g)

    return triplets


# ── GUI ─────────────────────────────────────────────────────────────────


class ClipAnnotator:
    """Tkinter-based click annotation tool."""

    def __init__(self, clips_dir, output_path):
        self.clips_dir = Path(clips_dir)
        self.output_path = Path(output_path)

        # Discover triplets
        self.triplets = discover_triplets(self.clips_dir)
        if not self.triplets:
            print("ERROR: No video clips found in", self.clips_dir)
            sys.exit(1)

        # Load existing annotations
        self.annotations = {}
        if self.output_path.exists():
            with open(self.output_path) as f:
                data = json.load(f)
            self.annotations = {a["triplet_id"]: a for a in data.get("triplets", [])}

        # State
        self.current_idx = self._find_first_unannotated()
        self.cap = None
        self.playing = False
        self.current_selection = {}

        # Build GUI
        self.root = tk.Tk()
        self.root.title("GMFCS Clip Annotator")
        self.root.configure(bg=COLOR_BG)
        self.root.resizable(False, False)

        self._build_gui()
        self._load_triplet(self.current_idx)

    def _find_first_unannotated(self):
        """Find the first triplet without annotation."""
        for i, t in enumerate(self.triplets):
            if t["triplet_id"] not in self.annotations:
                return i
        return 0

    def _build_gui(self):
        # ── Header ──
        header = tk.Frame(self.root, bg=COLOR_HEADER, height=40)
        header.pack(fill="x")
        header.pack_propagate(False)

        self.lbl_progress = tk.Label(
            header, text="", fg="#ffffff", bg=COLOR_HEADER,
            font=("Helvetica", 13, "bold"),
        )
        self.lbl_progress.pack(side="left", padx=15)

        self.lbl_patient = tk.Label(
            header, text="", fg=COLOR_SELECTED, bg=COLOR_HEADER,
            font=("Helvetica", 13, "bold"),
        )
        self.lbl_patient.pack(side="right", padx=15)

        # ── Video area ──
        video_frame = tk.Frame(self.root, bg="#000000")
        video_frame.pack(padx=10, pady=(10, 5))

        self.video_label = tk.Label(
            video_frame, bg="#000000", width=VIDEO_W, height=VIDEO_H,
        )
        self.video_label.pack()

        # Video controls
        ctrl_frame = tk.Frame(self.root, bg=COLOR_BG)
        ctrl_frame.pack(pady=(0, 5))

        self.btn_replay = tk.Button(
            ctrl_frame, text="▶ Replay", command=self._replay,
            bg=COLOR_BTN, fg=COLOR_BTN_TEXT, font=("Helvetica", 11),
            width=10, relief="flat", cursor="hand2",
        )
        self.btn_replay.pack(side="left", padx=5)

        self.lbl_clip_info = tk.Label(
            ctrl_frame, text="", fg=COLOR_LABEL, bg=COLOR_BG,
            font=("Helvetica", 11),
        )
        self.lbl_clip_info.pack(side="left", padx=15)

        # ── Annotation panel ──
        panel = tk.Frame(self.root, bg=COLOR_BG)
        panel.pack(fill="x", padx=10, pady=5)

        self.btn_groups = {}

        # Movement
        self.movement_frame = self._add_button_row(
            panel, "Movement", "movement",
            list(MOVEMENT_LABELS.values()),
            list(MOVEMENT_LABELS.keys()),
        )

        # Surface (conditional — hidden by default)
        self.surface_frame = self._add_button_row(
            panel, "Surface", "surface",
            ["Floor", "Chair"],
            ["floor", "chair"],
        )

        # AFO
        self._add_button_row(
            panel, "AFO", "afo",
            AFO_OPTIONS,
            ["bilateral", "unilateral_left", "unilateral_right", "none"],
        )

        # Walker
        self._add_button_row(
            panel, "Walker", "walker",
            ["Yes", "No"],
            [True, False],
        )

        # Acrylic Stand
        self._add_button_row(
            panel, "Acrylic Stand", "acrylic_stand",
            ["Yes", "No"],
            [True, False],
        )

        # Caregiver Assistance
        self._add_button_row(
            panel, "Caregiver Assist", "caregiver_assistance",
            ["Yes", "No"],
            [True, False],
        )

        # FIM
        self._add_button_row(
            panel, "FIM Score", "fim",
            [str(i) for i in FIM_OPTIONS],
            FIM_OPTIONS,
        )

        # ── Navigation ──
        nav = tk.Frame(self.root, bg=COLOR_BG)
        nav.pack(fill="x", padx=10, pady=(10, 15))

        self.btn_prev = tk.Button(
            nav, text="◀  Prev", command=self._nav_prev,
            bg=COLOR_BTN, fg=COLOR_BTN_TEXT, font=("Helvetica", 12, "bold"),
            width=10, height=1, relief="flat", cursor="hand2",
        )
        self.btn_prev.pack(side="left", padx=5)

        self.btn_skip = tk.Button(
            nav, text="Skip  ▶▶", command=self._nav_skip,
            bg=COLOR_SKIP, fg="#ffffff", font=("Helvetica", 12, "bold"),
            width=10, height=1, relief="flat", cursor="hand2",
        )
        self.btn_skip.pack(side="left", padx=5)

        self.btn_save = tk.Button(
            nav, text="Save & Next  ▶", command=self._save_and_next,
            bg=COLOR_SAVE, fg="#ffffff", font=("Helvetica", 14, "bold"),
            width=20, height=1, relief="flat", cursor="hand2",
        )
        self.btn_save.pack(side="right", padx=5)

        # Keyboard shortcuts
        self.root.bind("<Return>", lambda e: self._save_and_next())
        self.root.bind("<Right>", lambda e: self._nav_skip())
        self.root.bind("<Left>", lambda e: self._nav_prev())
        self.root.bind("<space>", lambda e: self._replay())

    def _add_button_row(self, parent, label_text, field_name, display_labels, values):
        """Add a labeled row of toggle buttons."""
        frame = tk.Frame(parent, bg=COLOR_BG)
        frame.pack(fill="x", pady=3)

        lbl = tk.Label(
            frame, text=label_text, fg=COLOR_LABEL, bg=COLOR_BG,
            font=("Helvetica", 11, "bold"), width=16, anchor="w",
        )
        lbl.pack(side="left", padx=(0, 10))

        buttons = []
        for disp, val in zip(display_labels, values):
            btn = tk.Button(
                frame, text=disp,
                bg=COLOR_BTN, fg=COLOR_BTN_TEXT,
                activebackground=COLOR_SELECTED,
                font=("Helvetica", 11), relief="flat",
                width=max(8, len(disp) + 2), height=1,
                cursor="hand2",
                command=lambda f=field_name, v=val: self._on_select(f, v),
            )
            btn.pack(side="left", padx=2)
            buttons.append((btn, val))

        self.btn_groups[field_name] = buttons
        return frame

    def _on_select(self, field_name, value):
        """Handle button selection — highlight selected, deselect others."""
        self.current_selection[field_name] = value

        for btn, val in self.btn_groups[field_name]:
            if val == value:
                btn.configure(bg=COLOR_SELECTED, fg=COLOR_SELECTED_TEXT)
            else:
                btn.configure(bg=COLOR_BTN, fg=COLOR_BTN_TEXT)

        # Show/hide surface row based on movement
        if field_name == "movement":
            if value in ("c_s", "s_c"):
                self.surface_frame.pack(fill="x", pady=3,
                                         after=self.movement_frame)
            else:
                self.surface_frame.pack_forget()
                self.current_selection.pop("surface", None)

    def _update_surface_visibility(self):
        """Show surface row only for sit/stand movements."""
        mv = self.current_selection.get("movement")
        if mv in ("c_s", "s_c"):
            self.surface_frame.pack(fill="x", pady=3)
        else:
            self.surface_frame.pack_forget()

    def _clear_selections(self):
        """Reset all button highlights."""
        self.current_selection = {}
        for field_name, buttons in self.btn_groups.items():
            for btn, val in buttons:
                btn.configure(bg=COLOR_BTN, fg=COLOR_BTN_TEXT)
        self.surface_frame.pack_forget()

    def _apply_selection(self, field_name, value):
        """Programmatically select a value (for pre-filling)."""
        self._on_select(field_name, value)

    def _load_triplet(self, idx):
        """Load a triplet and start video playback."""
        if idx < 0 or idx >= len(self.triplets):
            return

        self.current_idx = idx
        triplet = self.triplets[idx]

        # Update header
        annotated = sum(1 for t in self.triplets if t["triplet_id"] in self.annotations)
        total = len(self.triplets)
        self.lbl_progress.configure(
            text=f"Clip {idx + 1} / {total}  ({annotated} annotated)"
        )
        self.lbl_patient.configure(
            text=f"Patient: {triplet['patient_id']}  |  {triplet['triplet_id']}"
        )

        # Show clip info
        views = ", ".join(sorted(triplet["clips"].keys()))
        self.lbl_clip_info.configure(text=f"Views: {views}")

        # Check if already annotated
        existing = self.annotations.get(triplet["triplet_id"])

        # Clear and pre-fill
        self._clear_selections()

        if existing:
            # Restore saved annotation
            self._prefill_from_annotation(existing)
            self.btn_save.configure(text="Update & Next  ▶")
        else:
            # Pre-fill movement from filename
            mv = triplet["movement"]
            if mv in MOVEMENT_LABELS:
                self._apply_selection("movement", mv)
            elif mv in ("cc_s", "s_cc"):
                # Chair variant — pre-fill base movement + chair surface
                base = "c_s" if mv == "cc_s" else "s_c"
                self._apply_selection("movement", base)
                self._apply_selection("surface", "chair")
            self.btn_save.configure(text="Save & Next  ▶")

        # Update nav buttons
        self.btn_prev.configure(state="normal" if idx > 0 else "disabled")

        # Start video
        self._start_video(triplet)

    def _prefill_from_annotation(self, ann):
        """Fill buttons from a saved annotation."""
        mv = ann.get("movement", "")
        # Determine base movement and surface
        if mv in ("cc_s", "s_cc"):
            base = "c_s" if mv == "cc_s" else "s_c"
            self._apply_selection("movement", base)
            self._apply_selection("surface", "chair")
        elif mv in ("c_s", "s_c"):
            self._apply_selection("movement", mv)
            self._apply_selection("surface", "floor")
        elif mv in MOVEMENT_LABELS:
            self._apply_selection("movement", mv)

        if "afo" in ann:
            self._apply_selection("afo", ann["afo"])
        if "walker" in ann:
            self._apply_selection("walker", ann["walker"])
        if "acrylic_stand" in ann:
            self._apply_selection("acrylic_stand", ann["acrylic_stand"])
        if "caregiver_assistance" in ann:
            self._apply_selection("caregiver_assistance", ann["caregiver_assistance"])
        if "fim" in ann:
            self._apply_selection("fim", ann["fim"])

    def _start_video(self, triplet):
        """Start playing the front-view video."""
        self.playing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        # Prefer front view, fall back to any available
        video_path = triplet["clips"].get(FRONT_VIEW)
        if video_path is None:
            video_path = next(iter(triplet["clips"].values()), None)
        if video_path is None:
            return

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            return

        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.playing = True
        self._play_frame()

    def _play_frame(self):
        """Read and display one video frame."""
        if not self.playing or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            # Loop back to start
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))
            img = Image.fromarray(frame)
            self.photo = ImageTk.PhotoImage(img)
            self.video_label.configure(image=self.photo)

        delay = max(1, int(1000 / self.video_fps))
        self.root.after(delay, self._play_frame)

    def _replay(self):
        """Restart video from beginning."""
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if not self.playing:
                self.playing = True
                self._play_frame()

    def _resolve_movement(self):
        """Resolve base movement + surface into final movement code."""
        mv = self.current_selection.get("movement")
        if mv is None:
            return None

        if mv in ("c_s", "s_c"):
            surface = self.current_selection.get("surface")
            if surface is None:
                return None  # surface required
            return SURFACE_MOVEMENTS.get((mv, surface), mv)
        return mv

    def _validate(self):
        """Check all required fields are filled."""
        missing = []
        if "movement" not in self.current_selection:
            missing.append("Movement")
        mv = self.current_selection.get("movement")
        if mv in ("c_s", "s_c") and "surface" not in self.current_selection:
            missing.append("Surface")
        if "afo" not in self.current_selection:
            missing.append("AFO")
        if "walker" not in self.current_selection:
            missing.append("Walker")
        if "acrylic_stand" not in self.current_selection:
            missing.append("Acrylic Stand")
        if "caregiver_assistance" not in self.current_selection:
            missing.append("Caregiver Assist")
        if "fim" not in self.current_selection:
            missing.append("FIM Score")
        return missing

    def _save_and_next(self):
        """Validate, save current annotation, advance to next."""
        missing = self._validate()
        if missing:
            messagebox.showwarning(
                "Missing Fields",
                "Please select: " + ", ".join(missing),
            )
            return

        triplet = self.triplets[self.current_idx]
        final_movement = self._resolve_movement()

        # Build annotation record
        ann = {
            "triplet_id": triplet["triplet_id"],
            "patient_id": triplet["patient_id"],
            "movement": final_movement,
            "afo": self.current_selection["afo"],
            "walker": self.current_selection["walker"],
            "acrylic_stand": self.current_selection["acrylic_stand"],
            "caregiver_assistance": self.current_selection["caregiver_assistance"],
            "fim": self.current_selection["fim"],
            "clips": list(triplet["clips"].keys()),
            "annotated_at": datetime.now().isoformat(),
        }

        # Add surface only if relevant
        if final_movement in ("c_s", "s_c"):
            ann["surface"] = "floor"
        elif final_movement in ("cc_s", "s_cc"):
            ann["surface"] = "chair"

        self.annotations[triplet["triplet_id"]] = ann
        self._save_to_disk()

        # Advance
        if self.current_idx + 1 < len(self.triplets):
            self._load_triplet(self.current_idx + 1)
        else:
            annotated = len(self.annotations)
            total = len(self.triplets)
            messagebox.showinfo(
                "Complete",
                f"All clips processed!\n{annotated}/{total} annotated.",
            )

    def _nav_prev(self):
        if self.current_idx > 0:
            self._load_triplet(self.current_idx - 1)

    def _nav_skip(self):
        if self.current_idx + 1 < len(self.triplets):
            self._load_triplet(self.current_idx + 1)

    def _save_to_disk(self):
        """Write annotations to JSON file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "created_at": datetime.now().isoformat(),
            "total_triplets": len(self.triplets),
            "annotated": len(self.annotations),
            "triplets": list(self.annotations.values()),
        }

        with open(self.output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _sync_to_assistive_annotations(self):
        """Sync clip annotations back to assistive_annotations.json."""
        ann_path = PROJECT_ROOT / "data" / "metadata" / "assistive_annotations.json"
        if not ann_path.exists():
            return

        with open(ann_path) as f:
            aa = json.load(f)

        patients = {p["patient_id"]: p for p in aa["patients"]}

        for ann in self.annotations.values():
            pid = ann["patient_id"]
            if pid not in patients:
                continue

            patient = patients[pid]
            mv = ann["movement"]

            # Map movement code to annotation key
            mv_key_map = {
                "w": "walk", "cr": "crawl",
                "c_s": "seated_to_standing", "s_c": "standing_to_seated",
                "sr": "side_rolling",
                "cc_s": "chair_seated_to_standing", "s_cc": "standing_to_chair_seated",
            }
            mv_key = mv_key_map.get(mv)
            if mv_key is None:
                continue

            # Build device string
            devices_used = []
            if ann["walker"]:
                devices_used.append("walker")
            if ann["afo"] != "none":
                devices_used.append("afo")
            if ann["acrylic_stand"]:
                devices_used.append("acrylic_stand")

            if not devices_used:
                device_str = "none"
            elif devices_used == ["walker"]:
                device_str = "walker"
            elif devices_used == ["afo"]:
                device_str = "afo_only"
            elif devices_used == ["acrylic_stand"]:
                device_str = "acrylic_stand"
            elif set(devices_used) == {"walker", "afo"}:
                device_str = "walker_and_afo"
            else:
                device_str = "none"

            # FIM → assistance (0-6 → 0.0-1.0)
            assistance = ann["fim"] / 6.0

            # Update per_movement
            if "per_movement" not in patient:
                patient["per_movement"] = {}
            patient["per_movement"][mv_key] = {
                "device": device_str,
                "assistance": round(assistance, 3),
                "notes": "",
            }

            # Update patient-level devices
            patient["devices"]["walker"] = any(
                a["walker"] for a in self.annotations.values()
                if a["patient_id"] == pid
            )
            if patient["devices"]["walker"]:
                patient["devices"]["walker_type"] = "anterior"
            else:
                patient["devices"]["walker_type"] = "none"

            patient["devices"]["afo"] = any(
                a["afo"] != "none" for a in self.annotations.values()
                if a["patient_id"] == pid
            )
            afo_vals = [
                a["afo"] for a in self.annotations.values()
                if a["patient_id"] == pid and a["afo"] != "none"
            ]
            if afo_vals:
                patient["devices"]["afo_laterality"] = afo_vals[0]
            else:
                patient["devices"]["afo_laterality"] = "none"

            patient["devices"]["acrylic_stand"] = any(
                a["acrylic_stand"] for a in self.annotations.values()
                if a["patient_id"] == pid
            )

            # Update overall FIM (max across movements for this patient)
            fim_vals = [
                a["fim"] for a in self.annotations.values()
                if a["patient_id"] == pid
            ]
            if fim_vals:
                patient["overall_assistance"] = max(fim_vals)

        aa["patients"] = list(patients.values())
        with open(ann_path, "w") as f:
            json.dump(aa, f, indent=2, ensure_ascii=False)

        print(f"Synced annotations to {ann_path}")

    def run(self):
        """Start the annotation GUI."""
        print(f"Found {len(self.triplets)} triplets to annotate")
        print(f"Already annotated: {len(self.annotations)}")
        print("Keyboard: Enter=Save, Space=Replay, ←/→=Navigate")

        def on_close():
            self.playing = False
            if self.cap:
                self.cap.release()
            if self.annotations:
                self._sync_to_assistive_annotations()
            self.root.destroy()

        self.root.protocol("WM_DELETE_WINDOW", on_close)
        self.root.mainloop()


# ── Main ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Click-to-annotate GMFCS clips")
    parser.add_argument(
        "--clips-dir", default="data/video_clips",
        help="Directory containing patient video clips",
    )
    parser.add_argument(
        "--output", default="data/metadata/clip_annotations.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    clips_dir = PROJECT_ROOT / args.clips_dir
    output_path = PROJECT_ROOT / args.output

    if not clips_dir.exists():
        print(f"ERROR: {clips_dir} not found")
        sys.exit(1)

    app = ClipAnnotator(clips_dir, output_path)
    app.run()


if __name__ == "__main__":
    main()
