#!/usr/bin/env python3
"""
Run MESA baselines (DGS, MB-ADMM, Geodesic-MESA) on per-robot g2o files.

Pipeline:
  1. Auto-discover per-robot g2o files in the experiment folder (same logic as
     merge_optimize_g2o.py: */dpgo/bpsam_robot_*.g2o).
  2. Merge them into a single g2o (de-duped vertices, all edges).
  3. Invoke g2o-2-mr-jrl inside the mesa-baselines Docker image to produce a
     .jrl dataset file.
  4. Run each method (dgs, mbadmm, geodesic-mesa) via run-dist-batch in Docker.
  5. Results land in <experiment_folder>/mesa_baselines/<variant>/<method>/.

Usage:
    python3 run_mesa_baselines.py <experiment_folder> [--methods dgs mbadmm geodesic-mesa]
    python3 run_mesa_baselines.py g2345/ns-as
    python3 run_mesa_baselines.py g2345               # discovers all variants
"""

import argparse
import subprocess
import sys
from pathlib import Path

DOCKER_IMAGE = "mesa-baselines"
RUNNER = "/mesa_ws/mesa/build/experiments/run-dist-batch"

DEFAULT_METHODS = ["dgs", "asapp", "geodesic-mesa"]


# ---------------------------------------------------------------------------
# Discovery (same pattern as merge_optimize_g2o.py)
# ---------------------------------------------------------------------------

def find_g2o_files(folder: Path) -> list[Path]:
    files = sorted(folder.glob("*/dpgo/bpsam_robot_*.g2o"))
    if not files:
        files = sorted(folder.glob("*.g2o"))
    return files


def find_variants(folder: Path) -> list[Path]:
    """Return sub-folders that look like experiment variants (have per-robot g2o files).
    A variant must contain */dpgo/bpsam_robot_*.g2o (multi-robot layout).
    Single merged .g2o folders (like lm_optimized) are excluded."""
    variants = []
    for sub in sorted(folder.iterdir()):
        if not sub.is_dir():
            continue
        # Only count as variant if it has the per-robot pattern
        if list(sub.glob("*/dpgo/bpsam_robot_*.g2o")):
            variants.append(sub)
    return variants


# ---------------------------------------------------------------------------
# Convert g2o → JRL using our Python converter
# ---------------------------------------------------------------------------

def convert_to_jrl(g2o_files: list[Path], out_path: Path, name: str) -> bool:
    """Call g2o_to_jrl.py to produce a JRL file. Returns True on success."""
    script = Path(__file__).parent / "g2o_to_jrl.py"
    cmd = [sys.executable, str(script)] + [str(p) for p in g2o_files] + ["-o", str(out_path), "-n", name]
    print(f"  $", " ".join(cmd))
    return subprocess.run(cmd).returncode == 0


# ---------------------------------------------------------------------------
# Docker runner
# ---------------------------------------------------------------------------

def docker_run(host_dir: Path, cmd: list[str]) -> int:
    """Run cmd inside the mesa-baselines Docker container with host_dir mounted."""
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{host_dir.resolve()}:/data",
        DOCKER_IMAGE,
    ] + cmd
    print("  $", " ".join(docker_cmd))
    return subprocess.run(docker_cmd).returncode


# ---------------------------------------------------------------------------
# Per-variant pipeline
# ---------------------------------------------------------------------------

def run_variant(folder: Path, methods: list[str]) -> None:
    g2o_files = find_g2o_files(folder)
    if not g2o_files:
        print(f"  No g2o files found in {folder}, skipping.")
        return

    out_dir = folder / "mesa_baselines"
    out_dir.mkdir(exist_ok=True)

    # Step 1+2 — convert per-robot g2o files to JRL using g2o_to_jrl.py
    jrl_file = out_dir / "dataset.jrl"
    dataset_name = folder.name
    print(f"\n[1/2] Converting {len(g2o_files)} g2o files → {jrl_file}")
    if not convert_to_jrl(g2o_files, jrl_file, dataset_name):
        print("  Conversion failed, skipping methods.")
        return

    # Step 2 — run each method
    print(f"\n[2/2] Running methods: {methods}")
    for method in methods:
        method_out = f"mesa_baselines/{method}"
        (folder / method_out).mkdir(exist_ok=True)
        print(f"\n  --- {method} ---")
        rc = docker_run(folder, [
            RUNNER,
            "-i", "/data/mesa_baselines/dataset.jrl",
            "-m", method,
            "-o", f"/data/{method_out}",
            "--is3d",
        ])
        if rc != 0:
            print(f"  {method} failed (rc={rc})")
        else:
            results = sorted((folder / method_out).glob("**/final_results*"))
            print(f"  {method} done → {[str(r.relative_to(folder)) for r in results]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run MESA baselines on per-robot g2o files.")
    parser.add_argument("folder", type=Path, help="Experiment folder or variant sub-folder")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                        metavar="METHOD", help=f"Methods to run (default: {DEFAULT_METHODS})")
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.exists():
        print(f"Error: {folder} does not exist", file=sys.stderr)
        sys.exit(1)

    # Check docker image exists
    rc = subprocess.run(["docker", "image", "inspect", DOCKER_IMAGE],
                        capture_output=True).returncode
    if rc != 0:
        print(f"Error: Docker image '{DOCKER_IMAGE}' not found. "
              "Run: docker build -t mesa-baselines third_party/mesa/",
              file=sys.stderr)
        sys.exit(1)

    # Discover variants or treat folder directly
    variants = find_variants(folder)
    if variants:
        print(f"Found {len(variants)} variant(s) in {folder}:")
        for v in variants:
            print(f"  {v.name}")
        for variant in variants:
            print(f"\n=== Variant: {variant.name} ===")
            run_variant(variant, args.methods)
    else:
        # Single folder
        print(f"=== Running on: {folder} ===")
        run_variant(folder, args.methods)

    print("\nDone.")


if __name__ == "__main__":
    main()
