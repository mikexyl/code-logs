#!/usr/bin/env python3
"""
Extract inter-robot ground-truth loop closures from GT trajectory files.

For every pair of robots, any two poses whose translation distance is within
--dist (metres) and relative rotation is within an angle threshold (degrees)
are recorded as a loop closure.  Only inter-robot pairs are considered.

By default the script runs for three angle thresholds (10°, 20°, 30°) and
saves a separate CSV and plot for each, plus a combined stats file.

Usage:
    python extract_gt_loops.py ground_truth/campus
    python extract_gt_loops.py ground_truth/campus --dist 2.0
    python extract_gt_loops.py ground_truth/campus --angles 5 15 30

Output per threshold (tag = "angle<N>"):
    <gt_dir>/gt_loops_angle<N>.csv
    <gt_dir>/gt_loops_viz_angle<N>.pdf / .png
Combined:
    <gt_dir>/gt_loops_stats.txt

CSV columns:
    robot_i, timestamp_i_ns, robot_j, timestamp_j_ns,
    tx, ty, tz, qx, qy, qz, qw
  where (tx,ty,tz,qx,qy,qz,qw) is the relative pose T_{i←j}.
"""

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from utils.io import load_gt_trajectory
from utils.plot import IEEE_RC, save_fig

ROBOT_COLORS = ["#A8C4E0", "#F4C08A", "#A3D4B0", "#E8A0A0", "#C4B8D8", "#C8B09A"]


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def downsample_1hz(
    timestamps: np.ndarray,
    positions: np.ndarray,
    rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep at most one pose per second (first pose in each 1-second bin)."""
    if len(timestamps) == 0:
        return timestamps, positions, rotations
    bins = timestamps // 1_000_000_000          # nanoseconds → second index
    _, first_in_bin = np.unique(bins, return_index=True)
    return timestamps[first_in_bin], positions[first_in_bin], rotations[first_in_bin]


def load_robots(gt_dir: Path) -> dict[str, tuple]:
    """Load and downsample all robot GT files in gt_dir to 1 Hz.

    Supports .csv (comma-separated, ns timestamps) and .txt (TUM, s timestamps).
    """
    robots: dict[str, tuple] = {}
    files = sorted(gt_dir.glob("*.csv")) + sorted(gt_dir.glob("*.txt"))
    for p in sorted(files):
        ts, pos, rot = load_gt_trajectory(p)
        if len(ts) == 0:
            print(f"  Warning: no poses loaded from {p.name}")
            continue
        ts, pos, rot = downsample_1hz(ts, pos, rot)
        robots[p.stem] = (ts, pos, rot)
        print(f"  {p.stem}: {len(ts)} poses (1 Hz)")
    return robots


def find_loops(
    robots: dict[str, tuple],
    dist_thresh: float,
    angle_thresh_rad: float,
) -> list[dict]:
    """
    Find all inter-robot loop-closure pairs in pre-loaded robot data.

    Returns a list of dicts with keys:
        robot_i, timestamp_i_ns, robot_j, timestamp_j_ns,
        tx, ty, tz, qx, qy, qz, qw
    """
    loops: list[dict] = []
    robot_names = sorted(robots.keys())

    for name_i, name_j in itertools.combinations(robot_names, 2):
        ts_i, pos_i, rot_i = robots[name_i]
        ts_j, pos_j, rot_j = robots[name_j]

        # Pre-compute Rotation objects for the full trajectory (batch).
        R_all_i = Rotation.from_quat(rot_i)  # (N_i,)
        R_all_j = Rotation.from_quat(rot_j)  # (N_j,)

        # Spatial lookup: KD-tree gives candidates within translation threshold.
        tree_j = cKDTree(pos_j)
        candidate_lists = tree_j.query_ball_point(pos_i, r=dist_thresh)

        n_loops_pair = 0
        for idx_i, candidates in enumerate(candidate_lists):
            if not candidates:
                continue
            cands = np.asarray(candidates, dtype=np.intp)

            # Vectorised rotation check: compute all relative rotations at once.
            R_rel_batch = R_all_i[idx_i].inv() * R_all_j[cands]
            angles = R_rel_batch.magnitude()          # (K,) radians
            valid = cands[angles <= angle_thresh_rad]

            if valid.size == 0:
                continue

            # Vectorised relative-pose computation for all passing candidates.
            R_i_inv = R_all_i[idx_i].inv()
            pos_j_valid = pos_j[valid]                # (V, 3)
            t_rel_batch = R_i_inv.apply(pos_j_valid - pos_i[idx_i])   # (V, 3)
            q_rel_batch = (R_i_inv * R_all_j[valid]).as_quat()        # (V, 4) xyzw

            for k, idx_j in enumerate(valid):
                loops.append({
                    "robot_i":        name_i,
                    "timestamp_i_ns": int(ts_i[idx_i]),
                    "robot_j":        name_j,
                    "timestamp_j_ns": int(ts_j[idx_j]),
                    "tx": float(t_rel_batch[k, 0]),
                    "ty": float(t_rel_batch[k, 1]),
                    "tz": float(t_rel_batch[k, 2]),
                    "qx": float(q_rel_batch[k, 0]),
                    "qy": float(q_rel_batch[k, 1]),
                    "qz": float(q_rel_batch[k, 2]),
                    "qw": float(q_rel_batch[k, 3]),
                })
            n_loops_pair += valid.size

        print(f"  {name_i} ↔ {name_j}: {n_loops_pair} loop closures")

    return loops


def compute_stats(loops: list[dict]) -> dict:
    """Compute per-pair counts and aggregate translation/rotation statistics."""
    per_pair: dict[tuple[str, str], int] = {}
    for r in loops:
        key = (r["robot_i"], r["robot_j"])
        per_pair[key] = per_pair.get(key, 0) + 1

    stats: dict = {"total": len(loops), "per_pair": per_pair}

    if loops:
        t_arr = np.array([[r["tx"], r["ty"], r["tz"]] for r in loops])
        t_dists = np.linalg.norm(t_arr, axis=1)
        q_arr = np.array([[r["qx"], r["qy"], r["qz"], r["qw"]] for r in loops])
        rot_angles_deg = np.degrees(Rotation.from_quat(q_arr).magnitude())
        stats["t_mean"] = float(np.mean(t_dists))
        stats["t_std"]  = float(np.std(t_dists))
        stats["a_mean"] = float(np.mean(rot_angles_deg))
        stats["a_std"]  = float(np.std(rot_angles_deg))

    return stats


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_loops(
    gt_dir: Path,
    loops_csv: Path,
    robots: dict[str, tuple],
    tag: str = "",
    subsample: int = 50,
) -> None:
    """
    Plot GT trajectories (XY) with subsampled loop-closure red lines.

    Args:
        gt_dir     : folder with per-robot GT CSV files (full trajectories for display)
        loops_csv  : path to the CSV produced by find_loops
        robots     : pre-loaded downsampled robot data (for timestamp→position lookup)
        tag        : label appended to output filenames and the plot title
        subsample  : show 1-in-N loop closure pairs (avoids overplotting)
    """
    robot_names = sorted(robots.keys())
    color_map = {name: ROBOT_COLORS[i % len(ROBOT_COLORS)]
                 for i, name in enumerate(robot_names)}

    # Timestamp → world position from downsampled data (matches loop CSV timestamps).
    ts_to_pos: dict[str, dict[int, np.ndarray]] = {}
    for name, (ts, pos, _) in robots.items():
        ts_to_pos[name] = {int(t): pos[i] for i, t in enumerate(ts)}

    # Load full (non-downsampled) trajectories for display.
    full_traj: dict[str, np.ndarray] = {}
    for p in sorted(gt_dir.glob("*.csv")) + sorted(gt_dir.glob("*.txt")):
        _, pos, _ = load_gt_trajectory(p)
        if len(pos):
            full_traj[p.stem] = pos

    # Collect subsampled endpoint pairs for loop-closure lines.
    loop_lines: list[tuple[np.ndarray, np.ndarray]] = []
    with open(loops_csv) as f:
        reader = csv.DictReader(f)
        for k, row in enumerate(reader):
            if k % subsample != 0:
                continue
            pi_w = ts_to_pos.get(row["robot_i"], {}).get(int(row["timestamp_i_ns"]))
            pj_w = ts_to_pos.get(row["robot_j"], {}).get(int(row["timestamp_j_ns"]))
            if pi_w is None or pj_w is None:
                continue
            loop_lines.append((pi_w, pj_w))

    plt.rcParams.update(IEEE_RC)
    fig, ax = plt.subplots()

    # Draw trajectories first (lower zorder).
    for name, pos in full_traj.items():
        c = color_map.get(name, ROBOT_COLORS[0])
        ax.plot(pos[:, 0], pos[:, 1], color=c, linewidth=1.0,
                label=name, zorder=3)
        ax.plot(pos[0, 0], pos[0, 1], "o", color=c, markersize=3, zorder=4)

    # Draw loop-closure lines on top.
    lc_added = False
    for pi_w, pj_w in loop_lines:
        ax.plot([pi_w[0], pj_w[0]], [pi_w[1], pj_w[1]],
                color="#CC2222", linewidth=1.0, alpha=0.6, zorder=5,
                label="loop closure" if not lc_added else None)
        lc_added = True

    total = _count_csv(loops_csv)
    title = f"GT Loop Closures ({tag}) — 1/{subsample} shown, total {total}"
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="best", markerscale=4, fontsize=5)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.3)
    ax.set_title(title, fontsize=6)
    plt.tight_layout()

    stem = f"gt_loops_viz_{tag}" if tag else "gt_loops_viz"
    save_fig(fig, gt_dir / stem)
    plt.close(fig)
    print(f"  ({len(loop_lines)} loop closure lines plotted)")


def _count_csv(path: Path) -> int:
    """Count data rows in a CSV (excluding header)."""
    with open(path) as f:
        return sum(1 for line in f if line.strip() and not line.startswith("robot_i"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract inter-robot GT loop closures from ground-truth CSV files."
    )
    parser.add_argument("gt_dir", type=Path,
                        help="Folder containing per-robot GT CSV files (e.g. ground_truth/campus)")
    parser.add_argument("--dist",   type=float, default=1.0,
                        help="Max translation distance in metres (default: 1.0)")
    parser.add_argument("--angles", type=float, nargs="+", default=[10.0, 20.0, 30.0],
                        help="Rotation thresholds in degrees to sweep (default: 10 20 30)")
    parser.add_argument("--plot", action="store_true",
                        help="Visualize trajectories and loop closure lines for each threshold")
    parser.add_argument("--subsample", type=int, default=50,
                        help="Show 1-in-N loop closure lines in the plot (default: 50)")
    args = parser.parse_args()

    gt_dir: Path = args.gt_dir.resolve()
    if not gt_dir.exists():
        print(f"Error: {gt_dir} does not exist.")
        raise SystemExit(1)

    print(f"Experiment : {gt_dir.name}")
    print(f"dist ≤ {args.dist} m  |  angle thresholds: {args.angles}°")
    print(f"Loading GT files from {gt_dir} ...")
    robots = load_robots(gt_dir)

    if len(robots) < 2:
        print("Need at least two robots.")
        raise SystemExit(1)

    stats_path = gt_dir / "gt_loops_stats.txt"
    fieldnames = ["robot_i", "timestamp_i_ns", "robot_j", "timestamp_j_ns",
                  "tx", "ty", "tz", "qx", "qy", "qz", "qw"]

    with open(stats_path, "w") as stats_f:
        stats_f.write(f"GT Loop Closure Statistics — {gt_dir.name}\n")
        stats_f.write(f"dist threshold : {args.dist} m\n")
        stats_f.write("=" * 60 + "\n\n")

        for angle_deg in args.angles:
            tag = f"angle{int(angle_deg)}"
            angle_rad = np.deg2rad(angle_deg)

            print(f"\n--- angle ≤ {angle_deg}° ---")
            loops = find_loops(robots, args.dist, angle_rad)

            # Save CSV
            csv_path = gt_dir / f"gt_loops_{tag}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(loops)
            print(f"  CSV → {csv_path}")

            # Compute and write stats
            st = compute_stats(loops)
            stats_f.write(f"angle <= {angle_deg:.0f}°\n")
            stats_f.write(f"  Total loops : {st['total']}\n")
            for (ri, rj), cnt in sorted(st["per_pair"].items()):
                stats_f.write(f"  {ri} <-> {rj} : {cnt}\n")
            if st["total"] > 0:
                stats_f.write(f"  Translation : mean {st['t_mean']:.3f} m, "
                              f"std {st['t_std']:.3f} m\n")
                stats_f.write(f"  Rotation    : mean {st['a_mean']:.2f} deg, "
                              f"std {st['a_std']:.2f} deg\n")
            stats_f.write("\n")

            # Plot
            if args.plot:
                print(f"  Generating visualization ({tag}) ...")
                visualize_loops(gt_dir, csv_path, robots,
                                tag=tag, subsample=args.subsample)

    print(f"\nStats → {stats_path}")


if __name__ == "__main__":
    main()
