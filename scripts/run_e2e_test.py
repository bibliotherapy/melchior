"""End-to-end pipeline test with 7 video clips.

Runs the full Melchior pipeline from 2D pose extraction through
training and inference on a small test set.

Skips SAM2 mask propagation (step 00) — uses height-ratio fallback
for person identification.

Usage:
    python scripts/run_e2e_test.py
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATIENTS = ["hja", "jrh", "jyh", "kcw", "pjw"]
VAL_PATIENTS = ["ly"]
TEST_PATIENTS = ["mkj"]

CONFIG = "configs/e2e_test.yaml"
OUTPUT_DIR = "outputs/e2e_test/models"


def run_step(name, cmd):
    """Run a pipeline step and return (success, elapsed_seconds)."""
    print(f"\n{'=' * 60}")
    print(f"  STEP: {name}")
    print(f"{'=' * 60}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  FAILED: {name} (exit code {result.returncode}, {elapsed:.1f}s)")
        return False, elapsed

    print(f"\n  PASSED: {name} ({elapsed:.1f}s)")
    return True, elapsed


def main():
    py = sys.executable
    results = []

    steps = [
        (
            "01 — 2D Pose Extraction",
            [py, "scripts/01_extract_2d_pose.py",
             "--config", CONFIG, "--overwrite"],
        ),
        (
            "02 — Camera Calibration",
            [py, "scripts/02_calibrate_cameras.py",
             "--config", CONFIG, "--overwrite"],
        ),
        (
            "03 — 3D Triangulation",
            [py, "scripts/03_triangulate_3d.py",
             "--config", CONFIG, "--overwrite"],
        ),
        (
            "04 — Feature Extraction",
            [py, "scripts/04_extract_features.py",
             "--config", CONFIG, "--overwrite"],
        ),
        (
            "05 — Training (fixed split)",
            [py, "scripts/05_train.py",
             "--config", CONFIG, "--quick",
             "--train-patients"] + TRAIN_PATIENTS +
            ["--val-patients"] + VAL_PATIENTS +
            ["--output-dir", OUTPUT_DIR],
        ),
        (
            "07 — Inference (mkj)",
            [py, "scripts/07_infer.py",
             "--config", CONFIG,
             "--model-dir", OUTPUT_DIR,
             "--patient"] + TEST_PATIENTS,
        ),
    ]

    total_start = time.time()

    for name, cmd in steps:
        ok, elapsed = run_step(name, cmd)
        results.append((name, ok, elapsed))
        if not ok:
            print(f"\n  Pipeline stopped at: {name}")
            break

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'=' * 60}")
    print("  E2E PIPELINE TEST SUMMARY")
    print(f"{'=' * 60}")
    for name, ok, elapsed in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name} ({elapsed:.1f}s)")
    print(f"\n  Total time: {total_elapsed:.1f}s")

    all_passed = all(ok for _, ok, _ in results)
    if all_passed:
        print("  Result: ALL STEPS PASSED")
    else:
        print("  Result: PIPELINE FAILED")
    print(f"{'=' * 60}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
