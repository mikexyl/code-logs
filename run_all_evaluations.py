#!/usr/bin/env python3
"""
Run global LM+Huber optimization and ATE evaluation on all datasets/variants,
then collect results into a single CSV table.

Usage:
    python3 run_all_evaluations.py [--skip_optimize] [--skip_evaluate] [--output ate_results.csv]
"""

import argparse
import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from utils.io import is_robot_dir

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

BASE = Path(__file__).parent

# Each entry: (experiment_dir, gt_exp_name or None)
# For variant experiments, evaluate.py is run on each variant subdir.
# For flat experiments, evaluate.py is run on the experiment dir directly.
DATASETS: list[tuple[str, str | None]] = [
    # flat (robots directly in exp dir)
    ("g123",   "g123"),
    ("g156",   "g156"),
    # variant experiments
    ("a5678",  "a5678"),
    ("campus", "campus"),
    ("g2345",  "g2345"),
    ("gate",   "gate"),
]


def _is_robot_dir(d: Path) -> bool:
    return (d / "distributed").is_dir() or (d / "dpgo").is_dir()


def discover_variants(exp_dir: Path) -> list[Path]:
    """Return variant subdirs (contain robot dirs), excluding generated lm_optimized dirs."""
    return [
        d for d in sorted(exp_dir.iterdir())
        if d.is_dir()
        and d.name != "lm_optimized"
        and any(_is_robot_dir(s) for s in d.iterdir() if s.is_dir())
    ]


def has_g2o_files(folder: Path) -> bool:
    return bool(list(folder.glob("*/dpgo/bpsam_robot_*.g2o")))


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], label: str) -> bool:
    print(f"\n>>> {label}")
    print(f"    {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"    [FAILED] exit code {result.returncode}")
        return False
    return True


def optimize_variant(variant_dir: Path, skip: bool) -> bool:
    """Run merge_optimize_g2o.py on variant_dir if lm_optimized doesn't exist yet."""
    lm_dir = variant_dir / "lm_optimized"
    if skip or (lm_dir.exists() and any(lm_dir.glob("*/dpgo/Robot *.tum"))):
        print(f"    [skip optimize] {variant_dir.relative_to(BASE)}/lm_optimized already exists")
        return True
    if not has_g2o_files(variant_dir):
        print(f"    [skip optimize] no bpsam_robot_*.g2o in {variant_dir.relative_to(BASE)}")
        return False
    return run(
        ["pixi", "run", "python3", "merge_optimize_g2o.py", str(variant_dir)],
        f"Optimize {variant_dir.relative_to(BASE)}",
    )


def evaluate_variant(variant_dir: Path, gt_exp_name: str | None, skip: bool) -> bool:
    """Run evaluate.py on variant_dir."""
    ape_zip     = variant_dir / "evo_ape.zip"
    lm_ape_zip  = variant_dir / "lm_optimized" / "evo_ape.zip"
    if skip and ape_zip.exists() and lm_ape_zip.exists():
        print(f"    [skip evaluate] both evo_ape.zip files exist")
        return True
    cmd = ["pixi", "run", "python3", "evaluate.py", str(variant_dir),
           "--gt_folder", "ground_truth"]
    if gt_exp_name:
        cmd += ["--gt_exp_name", gt_exp_name]
    return run(cmd, f"Evaluate {variant_dir.relative_to(BASE)}")


# ---------------------------------------------------------------------------
# Read ATE stats
# ---------------------------------------------------------------------------

def read_ape_stats(zip_path: Path) -> dict | None:
    if not zip_path.exists():
        return None
    try:
        with zipfile.ZipFile(zip_path) as z:
            return json.loads(z.read("stats.json"))
    except Exception as e:
        print(f"    [warn] could not read {zip_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Collect results
# ---------------------------------------------------------------------------

def collect_rows(variant_dir: Path, exp_name: str, variant_name: str) -> list[dict]:
    rows = []
    for method, zip_path in [
        ("dpgo",         variant_dir / "evo_ape.zip"),
        ("lm_optimized", variant_dir / "lm_optimized" / "evo_ape.zip"),
    ]:
        stats = read_ape_stats(zip_path)
        if stats is None:
            continue
        rows.append({
            "experiment": exp_name,
            "variant":    variant_name,
            "method":     method,
            "ate_rmse":   round(stats["rmse"],   4),
            "ate_mean":   round(stats["mean"],   4),
            "ate_median": round(stats["median"], 4),
            "ate_std":    round(stats["std"],    4),
            "ate_min":    round(stats["min"],    4),
            "ate_max":    round(stats["max"],    4),
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_optimize", action="store_true",
                        help="Skip optimization if lm_optimized/ already exists.")
    parser.add_argument("--skip_evaluate", action="store_true",
                        help="Skip evo evaluation if evo_ape.zip already exists.")
    parser.add_argument("--output", "-o", default="ate_results.csv",
                        help="Output CSV path (default: ate_results.csv).")
    args = parser.parse_args()

    all_rows: list[dict] = []

    for exp_rel, gt_exp_name in DATASETS:
        exp_dir = BASE / exp_rel
        if not exp_dir.is_dir():
            print(f"\n[skip] {exp_rel} — directory not found")
            continue

        variants = discover_variants(exp_dir)
        targets: list[tuple[Path, str]] = []  # (dir_to_run, variant_label)

        if variants:
            for v in variants:
                targets.append((v, v.name))
        else:
            # Flat structure: experiment dir itself is the "variant"
            targets.append((exp_dir, "—"))

        for variant_dir, variant_label in targets:
            print(f"\n{'='*60}")
            print(f"  {exp_rel} / {variant_label}")
            print(f"{'='*60}")

            optimize_variant(variant_dir, args.skip_optimize)
            evaluate_variant(variant_dir, gt_exp_name, args.skip_evaluate)

            rows = collect_rows(variant_dir, exp_rel, variant_label)
            all_rows.extend(rows)
            for r in rows:
                print(f"  {r['method']:14s}  ATE RMSE={r['ate_rmse']:.3f} m  mean={r['ate_mean']:.3f} m")

    # Write CSV
    out = BASE / args.output
    fields = ["experiment", "variant", "method",
              "ate_rmse", "ate_mean", "ate_median", "ate_std", "ate_min", "ate_max"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{'='*60}")
    print(f"Saved {len(all_rows)} rows → {out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
