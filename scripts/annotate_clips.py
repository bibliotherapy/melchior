#!/usr/bin/env python3
"""Click-to-annotate web GUI for per-clip GMFCS metadata.

Opens a local web page where you can watch video clips and record
movement type, assistive devices, and assistance levels with mouse
clicks. Annotates per triplet (3 views share one annotation).

Output: data/metadata/clip_annotations.json
Also syncs to: data/metadata/assistive_annotations.json

Usage:
    python scripts/annotate_clips.py
    python scripts/annotate_clips.py --clips-dir data/video_clips
    python scripts/annotate_clips.py --port 8899
"""

import argparse
import json
import mimetypes
import sys
import webbrowser
from collections import defaultdict
from datetime import datetime
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── File parsing ────────────────────────────────────────────────────────

VIEW_CODES = {"f": "front", "l": "left", "r": "right"}


def parse_clip_filename(filepath):
    """Parse {patient}_{view}_{movement}_{number}.{ext}."""
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
            groups[key]["clips"][view] = str(path.relative_to(clips_dir))

    triplets = []
    for key in sorted(groups.keys()):
        g = groups[key]
        g["triplet_id"] = f"{g['patient_id']}_{g['movement']}_{g['number']}"
        triplets.append(g)
    return triplets


# ── Sync to assistive_annotations.json ──────────────────────────────────


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

        # Update patient-level devices from all this patient's annotations
        patient_anns = [a for a in annotations.values() if a["patient_id"] == pid]
        patient["devices"]["walker"] = any(a.get("walker") for a in patient_anns)
        patient["devices"]["walker_type"] = "anterior" if patient["devices"]["walker"] else "none"
        patient["devices"]["afo"] = any(a.get("afo", "none") != "none" for a in patient_anns)
        afo_vals = [a["afo"] for a in patient_anns if a.get("afo", "none") != "none"]
        patient["devices"]["afo_laterality"] = afo_vals[0] if afo_vals else "none"
        patient["devices"]["acrylic_stand"] = any(a.get("acrylic_stand") for a in patient_anns)

        fim_vals = [a["fim"] for a in patient_anns if "fim" in a]
        if fim_vals:
            patient["overall_assistance"] = max(fim_vals)

    aa["patients"] = list(patients.values())
    with open(ann_path, "w") as f:
        json.dump(aa, f, indent=2, ensure_ascii=False)
    print(f"  Synced to {ann_path}")


# ── HTML page ───────────────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>GMFCS Clip Annotator</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #1a1a2e; color: #eee; }

.header { background: #16213e; padding: 12px 20px; display: flex;
           justify-content: space-between; align-items: center; }
.header .progress { font-size: 15px; color: #a0a0a0; }
.header .patient { font-size: 16px; font-weight: 700; color: #4fc3f7; }

.main { display: flex; gap: 16px; padding: 16px; max-width: 1200px; margin: 0 auto; }

.video-panel { flex: 1; min-width: 0; }
video { width: 100%; border-radius: 8px; background: #000; }
.video-controls { margin-top: 8px; display: flex; gap: 8px; align-items: center; }
.video-controls button { padding: 6px 16px; border: none; border-radius: 6px;
    background: #2a2a4a; color: #fff; cursor: pointer; font-size: 13px; }
.video-controls button:hover { background: #3a3a5a; }
.clip-info { color: #888; font-size: 13px; margin-left: 12px; }

.ann-panel { width: 380px; flex-shrink: 0; }

.field-group { margin-bottom: 12px; }
.field-label { font-size: 12px; font-weight: 600; color: #888;
               text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
.field-btns { display: flex; flex-wrap: wrap; gap: 6px; }

.field-btns button { padding: 10px 16px; border: 2px solid #333; border-radius: 8px;
    background: #2a2a4a; color: #ccc; cursor: pointer; font-size: 14px;
    font-weight: 500; transition: all 0.1s; min-width: 60px; text-align: center; }
.field-btns button:hover { border-color: #4fc3f7; color: #fff; }
.field-btns button.selected { background: #1565c0; border-color: #42a5f5;
    color: #fff; font-weight: 700; }

.field-btns button.fim { min-width: 44px; padding: 10px 12px; }

.surface-row { display: none; }
.surface-row.visible { display: block; }

.nav { display: flex; gap: 10px; margin-top: 16px; }
.nav button { flex: 1; padding: 14px; border: none; border-radius: 8px;
    font-size: 15px; font-weight: 700; cursor: pointer; transition: all 0.15s; }
.btn-prev { background: #333; color: #aaa; }
.btn-prev:hover { background: #444; }
.btn-skip { background: #e65100; color: #fff; }
.btn-skip:hover { background: #f57c00; }
.btn-save { background: #2e7d32; color: #fff; }
.btn-save:hover { background: #43a047; }
.btn-save:disabled { background: #333; color: #555; cursor: not-allowed; }

.done-badge { display: inline-block; background: #2e7d32; color: #fff;
    padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-left: 8px; }

.keyboard-hint { color: #555; font-size: 11px; text-align: center; margin-top: 12px; }
</style>
</head>
<body>

<div class="header">
    <span class="progress" id="progress"></span>
    <span class="patient" id="patient-info"></span>
</div>

<div class="main">
    <div class="video-panel">
        <video id="video" controls autoplay loop muted></video>
        <div class="video-controls">
            <button onclick="replayVideo()">&#9654; Replay</button>
            <button onclick="toggleMute()">&#128264; Sound</button>
            <span class="clip-info" id="clip-info"></span>
        </div>
    </div>

    <div class="ann-panel">
        <div class="field-group">
            <div class="field-label">Movement</div>
            <div class="field-btns" id="grp-movement">
                <button data-val="w" onclick="sel('movement','w',this)">Walk</button>
                <button data-val="cr" onclick="sel('movement','cr',this)">Crawl</button>
                <button data-val="c_s" onclick="sel('movement','c_s',this)">Sit&#8594;Stand</button>
                <button data-val="s_c" onclick="sel('movement','s_c',this)">Stand&#8594;Sit</button>
                <button data-val="sr" onclick="sel('movement','sr',this)">Side Roll</button>
            </div>
        </div>

        <div class="field-group surface-row" id="surface-row">
            <div class="field-label">Surface</div>
            <div class="field-btns" id="grp-surface">
                <button data-val="floor" onclick="sel('surface','floor',this)">Floor</button>
                <button data-val="chair" onclick="sel('surface','chair',this)">Chair</button>
            </div>
        </div>

        <div class="field-group">
            <div class="field-label">AFO (Ankle-Foot Orthosis)</div>
            <div class="field-btns" id="grp-afo">
                <button data-val="bilateral" onclick="sel('afo','bilateral',this)">Bilateral</button>
                <button data-val="unilateral_left" onclick="sel('afo','unilateral_left',this)">Uni-L</button>
                <button data-val="unilateral_right" onclick="sel('afo','unilateral_right',this)">Uni-R</button>
                <button data-val="none" onclick="sel('afo','none',this)">None</button>
            </div>
        </div>

        <div class="field-group">
            <div class="field-label">Walker</div>
            <div class="field-btns" id="grp-walker">
                <button data-val="true" onclick="sel('walker',true,this)">Yes</button>
                <button data-val="false" onclick="sel('walker',false,this)">No</button>
            </div>
        </div>

        <div class="field-group">
            <div class="field-label">Acrylic Stand</div>
            <div class="field-btns" id="grp-acrylic_stand">
                <button data-val="true" onclick="sel('acrylic_stand',true,this)">Yes</button>
                <button data-val="false" onclick="sel('acrylic_stand',false,this)">No</button>
            </div>
        </div>

        <div class="field-group">
            <div class="field-label">Caregiver Assistance</div>
            <div class="field-btns" id="grp-caregiver_assistance">
                <button data-val="true" onclick="sel('caregiver_assistance',true,this)">Yes</button>
                <button data-val="false" onclick="sel('caregiver_assistance',false,this)">No</button>
            </div>
        </div>

        <div class="field-group">
            <div class="field-label">FIM Score</div>
            <div class="field-btns" id="grp-fim">
                <button class="fim" data-val="0" onclick="sel('fim',0,this)">0</button>
                <button class="fim" data-val="1" onclick="sel('fim',1,this)">1</button>
                <button class="fim" data-val="2" onclick="sel('fim',2,this)">2</button>
                <button class="fim" data-val="3" onclick="sel('fim',3,this)">3</button>
                <button class="fim" data-val="4" onclick="sel('fim',4,this)">4</button>
                <button class="fim" data-val="5" onclick="sel('fim',5,this)">5</button>
                <button class="fim" data-val="6" onclick="sel('fim',6,this)">6</button>
            </div>
        </div>

        <div class="nav">
            <button class="btn-prev" onclick="navPrev()">&#9664; Prev</button>
            <button class="btn-skip" onclick="navSkip()">Skip &#9654;&#9654;</button>
            <button class="btn-save" id="btn-save" onclick="saveAndNext()">Save &amp; Next &#9654;</button>
        </div>

        <div class="keyboard-hint">
            Keyboard: Enter = Save &amp; Next &nbsp;|&nbsp; &#8592;/&#8594; = Prev/Skip &nbsp;|&nbsp; Space = Replay
        </div>
    </div>
</div>

<script>
let triplets = [];
let annotations = {};
let currentIdx = 0;
let currentSelection = {};

const SURFACE_MOVEMENTS = {
    'c_s_floor': 'c_s', 'c_s_chair': 'cc_s',
    's_c_floor': 's_c', 's_c_chair': 's_cc'
};

async function init() {
    const resp = await fetch('/api/data');
    const data = await resp.json();
    triplets = data.triplets;
    annotations = data.annotations || {};
    // Start at first unannotated
    currentIdx = triplets.findIndex(t => !(t.triplet_id in annotations));
    if (currentIdx < 0) currentIdx = 0;
    loadTriplet(currentIdx);
}

function loadTriplet(idx) {
    if (idx < 0 || idx >= triplets.length) return;
    currentIdx = idx;
    currentSelection = {};
    const t = triplets[idx];

    // Header
    const annotated = triplets.filter(t => t.triplet_id in annotations).length;
    document.getElementById('progress').textContent =
        `Clip ${idx+1} / ${triplets.length}  (${annotated} annotated)`;
    const badge = (t.triplet_id in annotations) ? '<span class="done-badge">DONE</span>' : '';
    document.getElementById('patient-info').innerHTML =
        `Patient: ${t.patient_id}  |  ${t.triplet_id}${badge}`;

    // Video — use front view
    const videoPath = t.clips['f'] || Object.values(t.clips)[0];
    const video = document.getElementById('video');
    video.src = '/video/' + videoPath;
    video.load();
    video.play();

    // Clip info
    document.getElementById('clip-info').textContent =
        'Views: ' + Object.keys(t.clips).sort().join(', ');

    // Clear all selections
    document.querySelectorAll('.field-btns button').forEach(b => b.classList.remove('selected'));
    document.getElementById('surface-row').classList.remove('visible');

    // Pre-fill from existing annotation or filename
    const existing = annotations[t.triplet_id];
    if (existing) {
        prefill(existing);
        document.getElementById('btn-save').textContent = 'Update & Next ▶';
    } else {
        // Pre-fill movement from filename
        let mv = t.movement;
        if (mv === 'cc_s') { selQuiet('movement','c_s'); selQuiet('surface','chair'); }
        else if (mv === 's_cc') { selQuiet('movement','s_c'); selQuiet('surface','chair'); }
        else if (['w','cr','c_s','s_c','sr'].includes(mv)) { selQuiet('movement', mv); }
        document.getElementById('btn-save').textContent = 'Save & Next ▶';
    }
    updateSurfaceVisibility();
}

function prefill(ann) {
    let mv = ann.movement;
    if (mv === 'cc_s') { selQuiet('movement','c_s'); selQuiet('surface','chair'); }
    else if (mv === 's_cc') { selQuiet('movement','s_c'); selQuiet('surface','chair'); }
    else if (mv === 'c_s') { selQuiet('movement','c_s'); selQuiet('surface','floor'); }
    else if (mv === 's_c') { selQuiet('movement','s_c'); selQuiet('surface','floor'); }
    else { selQuiet('movement', mv); }

    if (ann.afo !== undefined) selQuiet('afo', ann.afo);
    if (ann.walker !== undefined) selQuiet('walker', ann.walker);
    if (ann.acrylic_stand !== undefined) selQuiet('acrylic_stand', ann.acrylic_stand);
    if (ann.caregiver_assistance !== undefined) selQuiet('caregiver_assistance', ann.caregiver_assistance);
    if (ann.fim !== undefined) selQuiet('fim', ann.fim);
    updateSurfaceVisibility();
}

function sel(field, value, btn) {
    currentSelection[field] = value;
    // Highlight
    const grp = document.getElementById('grp-' + field);
    grp.querySelectorAll('button').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    if (field === 'movement') {
        updateSurfaceVisibility();
        if (!['c_s','s_c'].includes(value)) delete currentSelection.surface;
    }
}

function selQuiet(field, value) {
    currentSelection[field] = value;
    const grp = document.getElementById('grp-' + field);
    if (!grp) return;
    grp.querySelectorAll('button').forEach(b => {
        let bval = b.dataset.val;
        // Handle bool/number comparison
        if (bval === 'true') bval = true;
        else if (bval === 'false') bval = false;
        else if (!isNaN(bval) && bval !== '') bval = Number(bval);
        b.classList.toggle('selected', bval === value);
    });
}

function updateSurfaceVisibility() {
    const mv = currentSelection.movement;
    const row = document.getElementById('surface-row');
    if (mv === 'c_s' || mv === 's_c') {
        row.classList.add('visible');
    } else {
        row.classList.remove('visible');
    }
}

function replayVideo() {
    const v = document.getElementById('video');
    v.currentTime = 0; v.play();
}

function toggleMute() {
    const v = document.getElementById('video');
    v.muted = !v.muted;
}

function resolveMovement() {
    const mv = currentSelection.movement;
    if (!mv) return null;
    if (mv === 'c_s' || mv === 's_c') {
        const surface = currentSelection.surface;
        if (!surface) return null;
        return SURFACE_MOVEMENTS[mv + '_' + surface] || mv;
    }
    return mv;
}

function validate() {
    const missing = [];
    if (!currentSelection.movement) missing.push('Movement');
    if (['c_s','s_c'].includes(currentSelection.movement) && !currentSelection.surface)
        missing.push('Surface');
    if (currentSelection.afo === undefined) missing.push('AFO');
    if (currentSelection.walker === undefined) missing.push('Walker');
    if (currentSelection.acrylic_stand === undefined) missing.push('Acrylic Stand');
    if (currentSelection.caregiver_assistance === undefined) missing.push('Caregiver Assist');
    if (currentSelection.fim === undefined) missing.push('FIM Score');
    return missing;
}

async function saveAndNext() {
    const missing = validate();
    if (missing.length > 0) {
        alert('Please select: ' + missing.join(', '));
        return;
    }

    const t = triplets[currentIdx];
    const finalMv = resolveMovement();

    const ann = {
        triplet_id: t.triplet_id,
        patient_id: t.patient_id,
        movement: finalMv,
        afo: currentSelection.afo,
        walker: currentSelection.walker,
        acrylic_stand: currentSelection.acrylic_stand,
        caregiver_assistance: currentSelection.caregiver_assistance,
        fim: currentSelection.fim,
        clips: Object.keys(t.clips),
        annotated_at: new Date().toISOString()
    };

    if (['c_s','s_c'].includes(finalMv)) ann.surface = 'floor';
    else if (['cc_s','s_cc'].includes(finalMv)) ann.surface = 'chair';

    // Save to server
    await fetch('/api/save', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(ann)
    });

    annotations[t.triplet_id] = ann;

    // Next
    if (currentIdx + 1 < triplets.length) {
        loadTriplet(currentIdx + 1);
    } else {
        const done = Object.keys(annotations).length;
        alert(`All clips processed! ${done}/${triplets.length} annotated.`);
    }
}

function navPrev() { if (currentIdx > 0) loadTriplet(currentIdx - 1); }
function navSkip() { if (currentIdx + 1 < triplets.length) loadTriplet(currentIdx + 1); }

// Keyboard shortcuts
document.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); saveAndNext(); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); navSkip(); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); navPrev(); }
    else if (e.key === ' ') { e.preventDefault(); replayVideo(); }
});

init();
</script>
</body>
</html>
"""


# ── HTTP Server ─────────────────────────────────────────────────────────


class AnnotationHandler(BaseHTTPRequestHandler):
    """HTTP handler for annotation GUI."""

    clips_dir = None
    output_path = None
    triplets = []
    annotations = {}

    def log_message(self, format, *args):
        """Suppress default access logs."""
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._send_html()
        elif path == "/api/data":
            self._send_json({
                "triplets": self.triplets,
                "annotations": self.annotations,
            })
        elif path.startswith("/video/"):
            self._send_video(path[7:])  # strip /video/
        else:
            self._send_404()

    def do_POST(self):
        if self.path == "/api/save":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            ann = json.loads(body)
            tid = ann["triplet_id"]
            self.__class__.annotations[tid] = ann
            self._save_to_disk()
            self._send_json({"ok": True, "saved": tid})
        else:
            self._send_404()

    def _send_html(self):
        data = HTML_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _send_video(self, rel_path):
        video_path = Path(self.clips_dir) / rel_path
        if not video_path.exists():
            self._send_404()
            return

        mime, _ = mimetypes.guess_type(str(video_path))
        if mime is None:
            mime = "video/quicktime"

        file_size = video_path.stat().st_size

        # Handle Range requests for video seeking
        range_header = self.headers.get("Range")
        if range_header:
            start, end = self._parse_range(range_header, file_size)
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", length)
            self.send_header("Content-Type", mime)
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(video_path, "rb") as f:
                f.seek(start)
                self.wfile.write(f.read(length))
        else:
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", file_size)
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(video_path, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

    def _parse_range(self, range_header, file_size):
        """Parse Range: bytes=start-end header."""
        range_str = range_header.replace("bytes=", "")
        parts = range_str.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1
        end = min(end, file_size - 1)
        return start, end

    def _send_404(self):
        self.send_response(404)
        self.end_headers()

    def _save_to_disk(self):
        output_path = Path(self.__class__.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "created_at": datetime.now().isoformat(),
            "total_triplets": len(self.__class__.triplets),
            "annotated": len(self.__class__.annotations),
            "triplets": list(self.__class__.annotations.values()),
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# ── Main ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Click-to-annotate GMFCS clips")
    parser.add_argument("--clips-dir", default="data/video_clips")
    parser.add_argument("--output", default="data/metadata/clip_annotations.json")
    parser.add_argument("--port", type=int, default=8899)
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

    # Configure handler
    AnnotationHandler.clips_dir = str(clips_dir)
    AnnotationHandler.output_path = str(output_path)
    AnnotationHandler.triplets = triplets
    AnnotationHandler.annotations = annotations

    annotated = len(annotations)
    total = len(triplets)

    print(f"GMFCS Clip Annotator")
    print(f"  Clips directory: {clips_dir}")
    print(f"  Triplets found:  {total}")
    print(f"  Already done:    {annotated}")
    print(f"  Output:          {output_path}")
    print()
    print(f"  Opening http://localhost:{args.port} ...")
    print(f"  Press Ctrl+C to stop\n")

    webbrowser.open(f"http://localhost:{args.port}")

    server = HTTPServer(("localhost", args.port), AnnotationHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nStopping server...")
        if annotations:
            sync_to_assistive_annotations(AnnotationHandler.annotations)
        print(f"Done. {len(AnnotationHandler.annotations)}/{total} triplets annotated.")
        server.server_close()


if __name__ == "__main__":
    main()
