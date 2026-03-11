#!/usr/bin/env python3
"""
Merge per-robot g2o graphs into a single global graph, optimize with
Levenberg-Marquardt, and write:
  - merged_optimized.g2o  (sequential vertex IDs 1, 2, 3, ...)
  - <robot_dir>/dpgo/Robot N.tum  (optimized poses with original timestamps)

The output directory mirrors the variant structure so that evaluate.py
auto-discovers it as a variant and includes it in ATE comparison.

Usage:
    # Auto-discover per-robot g2o files in an experiment folder:
    python3 merge_optimize_g2o.py <experiment_folder> [--out_dir lm_optimized]

    # Explicit list of g2o files (no TUM output):
    python3 merge_optimize_g2o.py file1.g2o file2.g2o ... [--out_dir .]
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import gtsam
from gtsam import (
    LevenbergMarquardtOptimizer,
    LevenbergMarquardtParams,
    NonlinearFactorGraph,
    Pose3,
    PriorFactorPose3,
    Values,
)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_g2o_files(folder: Path) -> list[Path]:
    """Return per-robot bpsam_robot_*.g2o files sorted by path (robot index)."""
    files = sorted(folder.glob("*/dpgo/bpsam_robot_*.g2o"))
    if not files:
        files = sorted(folder.glob("*.g2o"))
    return files


def find_tum_for_g2o(g2o_path: Path) -> Path | None:
    """Return the Robot N.tum co-located with bpsam_robot_N.g2o, if it exists."""
    # bpsam_robot_N.g2o → Robot N.tum in the same dpgo/ dir
    stem = g2o_path.stem  # e.g. "bpsam_robot_0"
    parts = stem.split("_")
    if parts[-1].isdigit():
        robot_n = parts[-1]
        tum = g2o_path.parent / f"Robot {robot_n}.tum"
        if tum.exists():
            return tum
    return None


# ---------------------------------------------------------------------------
# Load + merge
# ---------------------------------------------------------------------------

def load_and_merge(
    g2o_files: list[Path],
) -> tuple[NonlinearFactorGraph, Values, dict[int, Path]]:
    """
    Load each g2o file, merge into one combined graph + values.
    Returns (graph, values, robot_chr → tum_path) where robot_chr is the
    GTSAM Symbol character byte for that file's poses.
    """
    combined_graph = NonlinearFactorGraph()
    combined_values = Values()
    chr_to_tum: dict[int, Path] = {}

    for path in g2o_files:
        print(f"  {path}")
        graph, values = gtsam.readG2o(str(path), True)
        print(f"    {values.size()} poses, {graph.size()} factors")

        for i in range(graph.size()):
            combined_graph.push_back(graph.at(i))
        for key in values.keys():
            if not combined_values.exists(key):
                combined_values.insert(key, values.atPose3(key))

        # Identify which robot chr this file's keys belong to
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
    return combined_graph, combined_values, chr_to_tum


# ---------------------------------------------------------------------------
# Optimize
# ---------------------------------------------------------------------------

def _wrap_huber(factor: gtsam.BetweenFactorPose3, huber_k: float) -> gtsam.BetweenFactorPose3:
    """Return a copy of factor with its noise model wrapped in a Huber kernel."""
    huber = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(huber_k),
        factor.noiseModel(),
    )
    return gtsam.BetweenFactorPose3(factor.keys()[0], factor.keys()[1], factor.measured(), huber)


def build_robust_graph(
    graph: NonlinearFactorGraph,
    huber_k: float,
) -> NonlinearFactorGraph:
    """
    Return a new graph where inter-robot loop closure edges are wrapped with a
    Huber robust kernel (threshold huber_k) and intra-robot edges are kept as-is.
    """
    robust_graph = NonlinearFactorGraph()
    n_intra, n_inter = 0, 0
    for i in range(graph.size()):
        factor = graph.at(i)
        if not isinstance(factor, gtsam.BetweenFactorPose3):
            robust_graph.push_back(factor)
            continue
        k1, k2 = factor.keys()[0], factor.keys()[1]
        if gtsam.symbolChr(k1) != gtsam.symbolChr(k2):
            robust_graph.push_back(_wrap_huber(factor, huber_k))
            n_inter += 1
        else:
            robust_graph.push_back(factor)
            n_intra += 1
    print(f"  Intra-robot edges: {n_intra} (unchanged)")
    print(f"  Inter-robot edges: {n_inter} (Huber k={huber_k})")
    return robust_graph


def optimize(graph: NonlinearFactorGraph, initial: Values, huber_k: float) -> Values:
    """
    Add a prior on the first pose to fix gauge freedom, wrap inter-robot edges
    with a Huber robust kernel, then run Levenberg-Marquardt.
    """
    robust_graph = build_robust_graph(graph, huber_k)

    first_key = sorted(initial.keys())[0]
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-6))
    robust_graph.push_back(
        PriorFactorPose3(first_key, initial.atPose3(first_key), prior_noise)
    )

    print(f"  Initial error: {robust_graph.error(initial):.6f}")
    params = LevenbergMarquardtParams()
    params.setVerbosity("SUMMARY")
    result = LevenbergMarquardtOptimizer(robust_graph, initial, params).optimize()
    print(f"  Final error:   {robust_graph.error(result):.6f}")
    return result


# ---------------------------------------------------------------------------
# Write g2o with sequential IDs
# ---------------------------------------------------------------------------

def _info_gtsam_to_g2o(R: np.ndarray) -> np.ndarray:
    """
    Convert GTSAM Cholesky R to g2o information matrix.
    GTSAM ordering: (rotation[0:3], translation[3:6]).
    g2o ordering:   (translation[0:3], rotation[3:6]).
    """
    info = R.T @ R
    out = np.zeros((6, 6))
    out[0:3, 0:3] = info[3:6, 3:6]
    out[3:6, 3:6] = info[0:3, 0:3]
    out[0:3, 3:6] = info[3:6, 0:3]
    out[3:6, 0:3] = info[0:3, 3:6]
    return out


def write_g2o(
    graph: NonlinearFactorGraph,
    values: Values,
    key_map: dict[int, int],
    out_path: Path,
) -> None:
    """Write a g2o file with vertex IDs remapped via key_map."""
    with open(out_path, "w") as f:
        for orig_key, new_id in sorted(key_map.items(), key=lambda kv: kv[1]):
            pose: Pose3 = values.atPose3(orig_key)
            t = pose.translation()
            q = pose.rotation().toQuaternion()
            f.write(
                f"VERTEX_SE3:QUAT {new_id}"
                f" {t[0]:.10g} {t[1]:.10g} {t[2]:.10g}"
                f" {q.x():.10g} {q.y():.10g} {q.z():.10g} {q.w():.10g}\n"
            )

        for i in range(graph.size()):
            factor = graph.at(i)
            if not isinstance(factor, gtsam.BetweenFactorPose3):
                continue
            k1, k2 = factor.keys()[0], factor.keys()[1]
            if k1 not in key_map or k2 not in key_map:
                continue
            measured: Pose3 = factor.measured()
            t = measured.translation()
            q = measured.rotation().toQuaternion()
            info_g2o = _info_gtsam_to_g2o(factor.noiseModel().R())
            upper = [info_g2o[r, c] for r in range(6) for c in range(r, 6)]
            f.write(
                f"EDGE_SE3:QUAT {key_map[k1]} {key_map[k2]}"
                f" {t[0]:.10g} {t[1]:.10g} {t[2]:.10g}"
                f" {q.x():.10g} {q.y():.10g} {q.z():.10g} {q.w():.10g}"
                f" {' '.join(f'{v:.10g}' for v in upper)}\n"
            )


# ---------------------------------------------------------------------------
# Write per-robot TUM files
# ---------------------------------------------------------------------------

def _read_tum_timestamps(tum_path: Path) -> list[str]:
    """Return the timestamp field (first column) for each non-comment line."""
    timestamps = []
    with open(tum_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                timestamps.append(line.split()[0])
    return timestamps


def write_tum_files(
    optimized: Values,
    chr_to_tum: dict[int, Path],
    g2o_files: list[Path],
    out_dir: Path,
) -> None:
    """
    For each robot, pair optimized poses with original timestamps and write
    a TUM file in <out_dir>/<robot_dir>/dpgo/Robot N.tum.

    The output directory structure mirrors the variant layout so that
    evaluate.py auto-discovers it as a variant.
    """
    for g2o_path in g2o_files:
        # Identify robot chr from the loaded values
        _, values_single = gtsam.readG2o(str(g2o_path), True)
        if values_single.size() == 0:
            continue
        sample_key = list(values_single.keys())[0]
        robot_chr = gtsam.symbolChr(sample_key)

        tum_src = chr_to_tum.get(robot_chr)
        if tum_src is None:
            print(f"  Skipping TUM output for robot '{chr(robot_chr)}': no source timestamps")
            continue

        timestamps = _read_tum_timestamps(tum_src)

        # Collect optimized poses for this robot, sorted by symbol index
        robot_keys = sorted(
            [k for k in optimized.keys() if gtsam.symbolChr(k) == robot_chr],
            key=lambda k: gtsam.symbolIndex(k),
        )

        if len(robot_keys) != len(timestamps):
            print(
                f"  Warning: robot '{chr(robot_chr)}' has {len(robot_keys)} optimized poses "
                f"but {len(timestamps)} timestamps in {tum_src} — using min"
            )

        n = min(len(robot_keys), len(timestamps))

        # Mirror the robot dir name: g2o is at <robot_dir>/dpgo/bpsam_robot_N.g2o
        robot_dir_name = g2o_path.parent.parent.name  # e.g. "g2"
        stem = g2o_path.stem  # e.g. "bpsam_robot_0"
        robot_n = stem.split("_")[-1]           # e.g. "0"
        tum_out = out_dir / robot_dir_name / "dpgo" / f"Robot {robot_n}.tum"
        tum_out.parent.mkdir(parents=True, exist_ok=True)

        with open(tum_out, "w") as f:
            for i in range(n):
                pose: Pose3 = optimized.atPose3(robot_keys[i])
                t = pose.translation()
                q = pose.rotation().toQuaternion()
                f.write(
                    f"{timestamps[i]}"
                    f" {t[0]:.9f} {t[1]:.9f} {t[2]:.9f}"
                    f" {q.x():.9f} {q.y():.9f} {q.z():.9f} {q.w():.9f}\n"
                )

        print(f"  Wrote {n} poses → {tum_out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-robot g2o files, optimize with LM, and write TUM trajectories."
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="Experiment folder (auto-discovers bpsam_robot_*.g2o) or explicit .g2o files.",
    )
    parser.add_argument(
        "--huber_k", type=float, default=1.345,
        help="Huber loss threshold for inter-robot loop closure edges (default: 1.345).",
    )
    parser.add_argument(
        "--out_dir", "-o", type=Path, default=None,
        help=(
            "Output directory (default: <folder>/lm_optimized/ for folder input). "
            "Will contain merged_optimized.g2o and per-robot <robot_dir>/dpgo/Robot N.tum."
        ),
    )
    args = parser.parse_args()

    inputs = [Path(p) for p in args.inputs]
    if len(inputs) == 1 and inputs[0].is_dir():
        folder = inputs[0]
        g2o_files = find_g2o_files(folder)
        if not g2o_files:
            print(f"No g2o files found under {folder}", file=sys.stderr)
            sys.exit(1)
        out_dir = args.out_dir or (folder / "lm_optimized")
    else:
        g2o_files = [p for p in inputs if p.suffix == ".g2o"]
        if not g2o_files:
            print("No .g2o files provided.", file=sys.stderr)
            sys.exit(1)
        out_dir = args.out_dir or Path("lm_optimized")

    print(f"=== Loading {len(g2o_files)} file(s) ===")
    graph, values, chr_to_tum = load_and_merge(g2o_files)

    if values.size() == 0:
        print("Error: no poses loaded.", file=sys.stderr)
        sys.exit(1)

    # Build key map: GTSAM Symbol key → sequential int starting at 1
    sorted_keys = sorted(
        values.keys(),
        key=lambda k: (gtsam.symbolChr(k), gtsam.symbolIndex(k)),
    )
    key_map: dict[int, int] = {k: i + 1 for i, k in enumerate(sorted_keys)}

    per_robot: dict[str, int] = defaultdict(int)
    for k in sorted_keys:
        per_robot[chr(gtsam.symbolChr(k))] += 1
    print(f"\nKey remapping: {len(key_map)} poses → IDs 1..{len(key_map)}")
    for robot_chr, count in sorted(per_robot.items()):
        print(f"  Robot '{robot_chr}': {count} poses")

    print("\n=== Optimizing (Levenberg-Marquardt + Huber) ===")
    optimized = optimize(graph, values, args.huber_k)

    out_dir.mkdir(parents=True, exist_ok=True)

    g2o_out = out_dir / "merged_optimized.g2o"
    print(f"\n=== Writing {g2o_out} ===")
    write_g2o(graph, optimized, key_map, g2o_out)
    print(f"  {len(key_map)} vertices, {graph.size()} edges")

    if chr_to_tum:
        print(f"\n=== Writing per-robot TUM files → {out_dir}/ ===")
        write_tum_files(optimized, chr_to_tum, g2o_files, out_dir)

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
