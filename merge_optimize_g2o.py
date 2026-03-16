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
    GncLMOptimizer,
    GncLMParams,
    GncLossType,
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
    """Return per-robot g2o files (one per robot, latest snapshot).

    Old layout: */dpgo/bpsam_robot_<id>.g2o
    New layout: */dpgo/cbs_robot_<id>_<ts>.g2o  — pick latest per robot ID
    """
    files = sorted(folder.glob("*/dpgo/bpsam_robot_*.g2o"))
    if files:
        return files

    # New layout: pick latest snapshot per robot ID
    all_new = sorted(folder.glob("*/dpgo/cbs_robot_*.g2o"))
    if all_new:
        latest: dict[int, tuple[int, Path]] = {}
        for p in all_new:
            parts = p.stem.split("_")
            try:
                robot_id, ts = int(parts[2]), int(parts[3])
            except (IndexError, ValueError):
                continue
            if robot_id not in latest or ts > latest[robot_id][0]:
                latest[robot_id] = (ts, p)
        return [latest[rid][1] for rid in sorted(latest)]

    return sorted(folder.glob("*.g2o"))


def find_tum_for_g2o(g2o_path: Path) -> Path | None:
    """Return the TUM file co-located with a per-robot g2o, if it exists.

    Old layout: bpsam_robot_N.g2o → Robot N.tum
    New layout: cbs_robot_N_<ts>.g2o → latest non-empty Robot N_<ts>.tum
    """
    stem = g2o_path.stem
    parts = stem.split("_")
    dpgo = g2o_path.parent

    if stem.startswith("bpsam_robot_") and parts[-1].isdigit():
        robot_n = parts[-1]
        tum = dpgo / f"Robot {robot_n}.tum"
        return tum if tum.exists() else None

    if stem.startswith("cbs_robot_") and len(parts) >= 4:
        robot_n = parts[2]
        candidates = sorted(
            [p for p in dpgo.glob(f"Robot {robot_n}_*.tum") if p.stat().st_size > 0],
            key=lambda p: int(p.stem.split("_")[-1])
        )
        return candidates[-1] if candidates else None

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
# Initialize poses via rotation averaging
# ---------------------------------------------------------------------------

def initialize_poses(graph: NonlinearFactorGraph, odometry_values: Values) -> Values:
    """
    Use InitializePose3 (chordal rotation averaging) to get a good initial
    estimate from the full graph (odometry + inter-robot loops).  This is
    critical for GNC: raw odometry initial values yield inter-robot residuals
    of O(1e4), causing GNC-TLS to reject all loops as outliers.  After
    InitializePose3 the residuals drop to O(1), enabling correct inlier/outlier
    classification.

    Falls back to odometry_values if initialization fails (e.g. underconstrained graph).
    """
    print("  Running InitializePose3 (chordal, full pose)...")
    try:
        # Pin the first pose to fix gauge freedom, then run full pose initialization
        graph_with_prior = NonlinearFactorGraph(graph)
        first_key = sorted(odometry_values.keys())[0]
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-6))
        graph_with_prior.push_back(
            PriorFactorPose3(first_key, odometry_values.atPose3(first_key), prior_noise)
        )
        initialized = gtsam.InitializePose3.initialize(graph_with_prior, odometry_values, False)
    except RuntimeError as e:
        print(f"  InitializePose3 failed ({e}); falling back to odometry initial values")
        return odometry_values

    # Diagnostics: inter-robot residuals after init
    inter_errors = []
    for i in range(graph.size()):
        fac = graph.at(i)
        if not isinstance(fac, gtsam.BetweenFactorPose3):
            continue
        k1, k2 = fac.keys()[0], fac.keys()[1]
        if gtsam.symbolChr(k1) != gtsam.symbolChr(k2):
            if initialized.exists(k1) and initialized.exists(k2):
                inter_errors.append(fac.error(initialized))
    if inter_errors:
        arr = np.array(inter_errors)
        print(
            f"  Inter-robot errors after init: "
            f"median={np.median(arr):.2f}  mean={arr.mean():.2f}  max={arr.max():.2f}"
        )
    return initialized


# ---------------------------------------------------------------------------
# Optimize
# ---------------------------------------------------------------------------

def optimize(graph: NonlinearFactorGraph, initial: Values) -> Values:
    """
    Add a prior on the first pose to fix gauge freedom, then run GNC-LM.
    Intra-robot odometry edges are marked as known inliers so GNC only
    applies its reweighting to inter-robot loop closure edges.
    """
    graph_with_prior = NonlinearFactorGraph(graph)
    first_key = sorted(initial.keys())[0]
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-6))
    graph_with_prior.push_back(
        PriorFactorPose3(first_key, initial.atPose3(first_key), prior_noise)
    )

    # Mark all intra-robot edges as known inliers
    known_inliers: list[int] = []
    n_intra, n_inter = 0, 0
    for i in range(graph_with_prior.size()):
        factor = graph_with_prior.at(i)
        if not isinstance(factor, gtsam.BetweenFactorPose3):
            known_inliers.append(i)   # prior factor — always inlier
            continue
        k1, k2 = factor.keys()[0], factor.keys()[1]
        if gtsam.symbolChr(k1) == gtsam.symbolChr(k2):
            known_inliers.append(i)
            n_intra += 1
        else:
            n_inter += 1
    print(f"  Intra-robot edges: {n_intra} (known inliers)")
    print(f"  Inter-robot edges: {n_inter} (GNC-reweighted)")

    lm_params = LevenbergMarquardtParams()
    lm_params.setVerbosity("SILENT")
    params = GncLMParams(lm_params)
    params.setLossType(GncLossType.GM)
    params.setKnownInliers(known_inliers)
    params.setVerbosityGNC(GncLMParams.Verbosity.SUMMARY)

    print(f"  Initial error: {graph_with_prior.error(initial):.6f}")
    result = GncLMOptimizer(graph_with_prior, initial, params).optimize()
    print(f"  Final error:   {graph_with_prior.error(result):.6f}")
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
        stem = g2o_path.stem  # e.g. "bpsam_robot_0" or "cbs_robot_0_<ts>"
        parts = stem.split("_")
        # For bpsam_robot_N: robot_n is last token
        # For cbs_robot_N_<ts>: robot_n is parts[2]
        if stem.startswith("cbs_robot_") and len(parts) >= 4:
            robot_n = parts[2]
        else:
            robot_n = parts[-1]                 # e.g. "0"
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

    print("\n=== Initializing poses (rotation averaging) ===")
    initial = initialize_poses(graph, values)

    print("\n=== Optimizing (GNC-LM) ===")
    optimized = optimize(graph, initial)

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
