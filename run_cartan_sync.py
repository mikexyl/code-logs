#!/usr/bin/env python3
"""
Run CartanSync global optimizer on a multi-robot experiment folder and write
per-robot TUM trajectories to <folder>/cartan_sync_optimized/.

Pipeline:
  1. Load per-robot g2o files, merge into a single factor graph.
  2. Write a normalized merged g2o (info matrices scaled to max diagonal ~500)
     so that CartanSync's Laplacian solver is well-conditioned.
  3. Call the cartan_sync binary.
  4. Parse the output VERTEX_SE3:QUAT lines.
  5. Write per-robot TUM files with original timestamps from Robot N.tum.

Usage:
    python3 run_cartan_sync.py <experiment_folder> [--out_dir cartan_sync_optimized]
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import gtsam
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
BINARY = SCRIPT_DIR / "build" / "cartan_sync"
LIB_DIR = SCRIPT_DIR / "build" / "third_party" / "cartan-sync" / "C++" / "lib"

# Information matrix normalization target (max diagonal value).
# CartanSync's Laplacian solver is well-conditioned when tau/kappa ~ 100-500.
INFO_NORM_TARGET = 500.0


# ---------------------------------------------------------------------------
# Discovery (mirrors merge_optimize_g2o.py)
# ---------------------------------------------------------------------------

def find_g2o_files(folder: Path) -> list[Path]:
    files = sorted(folder.glob("*/dpgo/bpsam_robot_*.g2o"))
    if not files:
        files = sorted(folder.glob("*.g2o"))
    return files


def find_tum_for_g2o(g2o_path: Path) -> Path | None:
    stem = g2o_path.stem
    parts = stem.split("_")
    if parts[-1].isdigit():
        robot_n = parts[-1]
        tum = g2o_path.parent / f"Robot {robot_n}.tum"
        if tum.exists():
            return tum
    return None


# ---------------------------------------------------------------------------
# Load + merge (same logic as merge_optimize_g2o.py)
# ---------------------------------------------------------------------------

def load_and_merge(g2o_files: list[Path]):
    """Return (graph, values, chr_to_tum, chr_to_sorted_keys)."""
    combined_graph = gtsam.NonlinearFactorGraph()
    combined_values = gtsam.Values()
    chr_to_tum: dict[int, Path] = {}
    chr_to_sorted_keys: dict[int, list[tuple[int, int]]] = {}

    for path in g2o_files:
        print(f"  {path}")
        graph, values = gtsam.readG2o(str(path), True)
        print(f"    {values.size()} poses, {graph.size()} factors")

        for i in range(graph.size()):
            combined_graph.push_back(graph.at(i))
        for key in values.keys():
            if not combined_values.exists(key):
                combined_values.insert(key, values.atPose3(key))
                rc = gtsam.symbolChr(key)
                chr_to_sorted_keys.setdefault(rc, []).append(
                    (gtsam.symbolIndex(key), key)
                )

        if values.size() > 0:
            sample_key = list(values.keys())[0]
            robot_chr = gtsam.symbolChr(sample_key)
            tum_path = find_tum_for_g2o(path)
            if tum_path is not None:
                chr_to_tum[robot_chr] = tum_path
                print(f"    timestamps: {tum_path}")
            else:
                print(f"    (no co-located Robot N.tum found)")

    print(f"  Total: {combined_values.size()} poses, {combined_graph.size()} factors")
    return combined_graph, combined_values, chr_to_tum, chr_to_sorted_keys


# ---------------------------------------------------------------------------
# Write normalized g2o for CartanSync
# ---------------------------------------------------------------------------

def write_normalized_g2o(
    graph: gtsam.NonlinearFactorGraph,
    values: gtsam.Values,
    key_map: dict[int, int],
    out_path: Path,
) -> None:
    """
    Write a g2o file with:
      - Vertex IDs from key_map (0-based, required by CartanSync)
      - Information matrices normalized so max diagonal <= INFO_NORM_TARGET
        (avoids ill-conditioned Laplacian in CartanSync's linear solver)
    """
    with open(out_path, "w") as f:
        # Vertices
        for orig_key, new_id in sorted(key_map.items(), key=lambda kv: kv[1]):
            pose: gtsam.Pose3 = values.atPose3(orig_key)
            t = pose.translation()
            q = pose.rotation().toQuaternion()
            f.write(
                f"VERTEX_SE3:QUAT {new_id}"
                f" {t[0]:.10g} {t[1]:.10g} {t[2]:.10g}"
                f" {q.x():.10g} {q.y():.10g} {q.z():.10g} {q.w():.10g}\n"
            )

        # Edges
        for i in range(graph.size()):
            factor = graph.at(i)
            if not isinstance(factor, gtsam.BetweenFactorPose3):
                continue
            k1, k2 = factor.keys()[0], factor.keys()[1]
            if k1 not in key_map or k2 not in key_map:
                continue
            measured: gtsam.Pose3 = factor.measured()
            t = measured.translation()
            q = measured.rotation().toQuaternion()

            # Build GTSAM information matrix (rotation, translation order)
            R_chol = factor.noiseModel().R()
            info = R_chol.T @ R_chol
            # Convert to g2o order: (translation[0:3], rotation[3:6])
            info_g2o = np.zeros((6, 6))
            info_g2o[0:3, 0:3] = info[3:6, 3:6]
            info_g2o[3:6, 3:6] = info[0:3, 0:3]
            info_g2o[0:3, 3:6] = info[3:6, 0:3]
            info_g2o[3:6, 0:3] = info[0:3, 3:6]

            # Normalize: scale so max diagonal == INFO_NORM_TARGET
            max_diag = np.max(np.diag(info_g2o))
            if max_diag > 0:
                info_g2o *= INFO_NORM_TARGET / max_diag

            upper = [info_g2o[r, c] for r in range(6) for c in range(r, 6)]
            f.write(
                f"EDGE_SE3:QUAT {key_map[k1]} {key_map[k2]}"
                f" {t[0]:.10g} {t[1]:.10g} {t[2]:.10g}"
                f" {q.x():.10g} {q.y():.10g} {q.z():.10g} {q.w():.10g}"
                f" {' '.join(f'{v:.10g}' for v in upper)}\n"
            )


# ---------------------------------------------------------------------------
# Run CartanSync binary
# ---------------------------------------------------------------------------

def run_cartan_sync(g2o_path: Path, result_path: Path) -> bool:
    if not BINARY.exists():
        print(f"Error: cartan_sync binary not found at {BINARY}", file=sys.stderr)
        print("Run:  pixi run build", file=sys.stderr)
        return False

    env = os.environ.copy()
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{LIB_DIR}:{existing_ld}" if existing_ld else str(LIB_DIR)

    proc = subprocess.run(
        [str(BINARY), str(g2o_path), str(result_path)],
        env=env, capture_output=False, text=True,
    )
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# Parse cartan_sync output g2o
# ---------------------------------------------------------------------------

def parse_result_g2o(result_path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return {vertex_id: (xyz, qxyzw)} from VERTEX_SE3:QUAT lines."""
    poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    with open(result_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "VERTEX_SE3:QUAT":
                continue
            vid = int(parts[1])
            xyz = np.array([float(x) for x in parts[2:5]])
            qxyzw = np.array([float(x) for x in parts[5:9]])
            poses[vid] = (xyz, qxyzw)
    return poses


# ---------------------------------------------------------------------------
# Write per-robot TUM files
# ---------------------------------------------------------------------------

def write_tum_files(
    result_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    chr_to_tum: dict[int, Path],
    chr_to_sorted_keys: dict[int, list[tuple[int, int]]],
    key_map: dict[int, int],
    g2o_files: list[Path],
    out_dir: Path,
) -> None:
    for g2o_path in g2o_files:
        _, values_single = gtsam.readG2o(str(g2o_path), True)
        if values_single.size() == 0:
            continue
        sample_key = list(values_single.keys())[0]
        robot_chr = gtsam.symbolChr(sample_key)

        tum_src = chr_to_tum.get(robot_chr)
        if tum_src is None:
            print(f"  Skipping robot '{chr(robot_chr)}': no source timestamps")
            continue

        # Read source timestamps
        timestamps = []
        with open(tum_src) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    timestamps.append(line.split()[0])
        if not timestamps:
            continue

        # Sorted keys for this robot
        sorted_keys = sorted(chr_to_sorted_keys.get(robot_chr, []))
        n = min(len(sorted_keys), len(timestamps))

        if len(sorted_keys) != len(timestamps):
            print(
                f"  Warning: robot '{chr(robot_chr)}' has {len(sorted_keys)} poses "
                f"but {len(timestamps)} timestamps — using min({n})"
            )

        # Output path mirrors variant layout
        stem = g2o_path.stem
        robot_n = stem.split("_")[-1]
        robot_dir_name = g2o_path.parent.parent.name
        tum_out = out_dir / robot_dir_name / "dpgo" / f"Robot {robot_n}.tum"
        tum_out.parent.mkdir(parents=True, exist_ok=True)

        with open(tum_out, "w") as f:
            for i in range(n):
                key = sorted_keys[i][1]
                vid = key_map[key]
                if vid not in result_poses:
                    continue
                xyz, qxyzw = result_poses[vid]
                f.write(
                    f"{timestamps[i]}"
                    f" {xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f}"
                    f" {qxyzw[0]:.9f} {qxyzw[1]:.9f} {qxyzw[2]:.9f} {qxyzw[3]:.9f}\n"
                )

        print(f"  Wrote {n} poses → {tum_out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CartanSync on per-robot g2o files and write TUM trajectories."
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="Experiment folder (auto-discovers bpsam_robot_*.g2o) or explicit .g2o files.",
    )
    parser.add_argument(
        "--out_dir", "-o", type=Path, default=None,
        help="Output directory (default: <folder>/cartan_sync_optimized/).",
    )
    args = parser.parse_args()

    inputs = [Path(p) for p in args.inputs]
    if len(inputs) == 1 and inputs[0].is_dir():
        folder = inputs[0]
        g2o_files = find_g2o_files(folder)
        if not g2o_files:
            print(f"No g2o files found under {folder}", file=sys.stderr)
            sys.exit(1)
        out_dir = args.out_dir or (folder / "cartan_sync_optimized")
    else:
        g2o_files = [p for p in inputs if p.suffix == ".g2o"]
        if not g2o_files:
            print("No .g2o files provided.", file=sys.stderr)
            sys.exit(1)
        out_dir = args.out_dir or Path("cartan_sync_optimized")

    print(f"=== Loading {len(g2o_files)} file(s) ===")
    graph, values, chr_to_tum, chr_to_sorted_keys = load_and_merge(g2o_files)

    if values.size() == 0:
        print("Error: no poses loaded.", file=sys.stderr)
        sys.exit(1)

    # Build 0-based key map
    sorted_keys = sorted(
        values.keys(),
        key=lambda k: (gtsam.symbolChr(k), gtsam.symbolIndex(k)),
    )
    key_map: dict[int, int] = {k: i for i, k in enumerate(sorted_keys)}
    print(f"\nKey remapping: {len(key_map)} poses → IDs 0..{len(key_map) - 1}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write normalized g2o for CartanSync
    normalized_g2o = out_dir / "merged_normalized.g2o"
    print(f"\n=== Writing normalized g2o → {normalized_g2o} ===")
    write_normalized_g2o(graph, values, key_map, normalized_g2o)

    # Run CartanSync
    result_g2o = out_dir / "cartan_result.g2o"
    print(f"\n=== Running CartanSync ===")
    ok = run_cartan_sync(normalized_g2o, result_g2o)
    if not ok:
        print("CartanSync failed.", file=sys.stderr)
        sys.exit(1)

    # Parse result
    result_poses = parse_result_g2o(result_g2o)
    print(f"  Parsed {len(result_poses)} poses from {result_g2o}")

    # Write TUM files
    if chr_to_tum:
        print(f"\n=== Writing per-robot TUM files → {out_dir}/ ===")
        write_tum_files(
            result_poses, chr_to_tum, chr_to_sorted_keys, key_map, g2o_files, out_dir
        )

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
