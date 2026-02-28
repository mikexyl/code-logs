#!/usr/bin/env python3
"""
Extract inter-robot ground-truth loop closures from GT trajectory files.

For every pair of robots, any two poses whose translation distance is within
--dist (metres) and relative rotation is within --angle (degrees) are recorded
as a loop closure.  Only inter-robot pairs are considered.

Usage:
    python extract_gt_loops.py ground_truth/campus
    python extract_gt_loops.py ground_truth/campus --dist 2.0 --angle 45
    python extract_gt_loops.py ground_truth/campus --out my_loops.csv

Output CSV columns:
    robot_i, timestamp_i_ns, robot_j, timestamp_j_ns,
    tx, ty, tz, qx, qy, qz, qw
  where (tx,ty,tz,qx,qy,qz,qw) is the relative pose T_{i←j}
  (pose of j expressed in the frame of i).
"""

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


IEEE_RC = {
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.figsize': (3.5, 3.5),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.5,
    'lines.linewidth': 0.8,
    'patch.linewidth': 0.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}

ROBOT_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_gt(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a ground-truth CSV.

    Returns:
        timestamps : (N,)  int64 nanoseconds
        positions  : (N,3) float64 xyz
        rotations  : (N,4) float64 xyzw quaternion
    """
    timestamps, positions, rotations = [], [], []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 8:
                continue
            try:
                ts  = int(float(parts[0]))
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                qw, qx, qy, qz = (float(parts[4]), float(parts[5]),
                                   float(parts[6]), float(parts[7]))
                timestamps.append(ts)
                positions.append([x, y, z])
                rotations.append([qx, qy, qz, qw])   # xyzw
            except ValueError:
                continue
    return (np.array(timestamps, dtype=np.int64),
            np.array(positions,  dtype=np.float64),
            np.array(rotations,  dtype=np.float64))


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def relative_pose(pos_i: np.ndarray, rot_i: np.ndarray,
                  pos_j: np.ndarray, rot_j: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute T_{i←j}: pose of j in the frame of i.

    Returns:
        t_rel : (3,)  translation
        q_rel : (4,)  xyzw quaternion
    """
    R_i = Rotation.from_quat(rot_i)   # world←i
    R_j = Rotation.from_quat(rot_j)   # world←j
    R_rel = R_i.inv() * R_j           # i←j
    t_rel = R_i.inv().apply(pos_j - pos_i)
    return t_rel, R_rel.as_quat()     # xyzw


def extract_loops(
    gt_dir: Path,
    dist_thresh: float,
    angle_thresh_rad: float,
) -> list[dict]:
    """
    Find all inter-robot loop-closure pairs across all robot GT files in gt_dir.

    Returns a list of dicts with keys:
        robot_i, timestamp_i_ns, robot_j, timestamp_j_ns,
        tx, ty, tz, qx, qy, qz, qw
    """
    # Load all robots
    robots: dict[str, tuple] = {}
    for p in sorted(gt_dir.glob("*.csv")):
        ts, pos, rot = load_gt(p)
        if len(ts) == 0:
            print(f"  Warning: no poses loaded from {p.name}")
            continue
        robots[p.stem] = (ts, pos, rot)
        print(f"  {p.stem}: {len(ts)} poses")

    if len(robots) < 2:
        print("Need at least two robots.")
        return []

    loops = []
    robot_names = sorted(robots.keys())

    for name_i, name_j in itertools.combinations(robot_names, 2):
        ts_i, pos_i, rot_i = robots[name_i]
        ts_j, pos_j, rot_j = robots[name_j]

        # Spatial lookup: find all pairs within translation threshold
        tree_j = cKDTree(pos_j)
        candidate_lists = tree_j.query_ball_point(pos_i, r=dist_thresh)

        n_loops_pair = 0
        for idx_i, candidates in enumerate(candidate_lists):
            if not candidates:
                continue
            R_i = Rotation.from_quat(rot_i[idx_i])
            for idx_j in candidates:
                # Rotation check
                R_j  = Rotation.from_quat(rot_j[idx_j])
                R_rel = R_i.inv() * R_j
                angle = R_rel.magnitude()   # radians
                if angle > angle_thresh_rad:
                    continue

                t_rel, q_rel = relative_pose(
                    pos_i[idx_i], rot_i[idx_i],
                    pos_j[idx_j], rot_j[idx_j],
                )
                loops.append({
                    "robot_i":        name_i,
                    "timestamp_i_ns": int(ts_i[idx_i]),
                    "robot_j":        name_j,
                    "timestamp_j_ns": int(ts_j[idx_j]),
                    "tx": float(t_rel[0]),
                    "ty": float(t_rel[1]),
                    "tz": float(t_rel[2]),
                    "qx": float(q_rel[0]),
                    "qy": float(q_rel[1]),
                    "qz": float(q_rel[2]),
                    "qw": float(q_rel[3]),
                })
                n_loops_pair += 1

        print(f"  {name_i} ↔ {name_j}: {n_loops_pair} loop closures")

    return loops


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_loops(gt_dir: Path, loops_csv: Path, subsample: int = 50) -> None:
    """
    Plot GT trajectories (XY) with subsampled loop-closure scatter and connecting lines.

    Args:
        gt_dir     : folder with per-robot GT CSV files
        loops_csv  : path to the gt_loop_closures.csv produced by extract_loops
        subsample  : show 1-in-N loop closure pairs (avoids overplotting)
    """
    # Load robot trajectories
    robots: dict[str, np.ndarray] = {}
    for p in sorted(gt_dir.glob("*.csv")):
        ts, pos, _ = load_gt(p)
        if len(ts) == 0:
            continue
        robots[p.stem] = pos
    robot_names = sorted(robots.keys())
    color_map = {name: ROBOT_COLORS[i % len(ROBOT_COLORS)]
                 for i, name in enumerate(robot_names)}

    # Build timestamp → position maps for world coordinate lookup
    ts_to_pos: dict[str, dict[int, np.ndarray]] = {}
    for p in sorted(gt_dir.glob("*.csv")):
        ts, pos, _ = load_gt(p)
        ts_to_pos[p.stem] = {int(t): pos[i] for i, t in enumerate(ts)}

    # Collect subsampled pairs with world positions
    # Group by robot pair to scatter-plot with midpoint colour
    pair_midpoints: dict[tuple[str, str], list[np.ndarray]] = {}
    n_found = 0
    with open(loops_csv) as f:
        reader = csv.DictReader(f)
        for k, row in enumerate(reader):
            if k % subsample != 0:
                continue
            ri = row["robot_i"]
            rj = row["robot_j"]
            ti = int(row["timestamp_i_ns"])
            tj = int(row["timestamp_j_ns"])
            pi_w = ts_to_pos.get(ri, {}).get(ti)
            pj_w = ts_to_pos.get(rj, {}).get(tj)
            if pi_w is None or pj_w is None:
                continue
            pair_key = (ri, rj)
            if pair_key not in pair_midpoints:
                pair_midpoints[pair_key] = []
            pair_midpoints[pair_key].append((pi_w + pj_w) / 2.0)
            n_found += 1

    plt.rcParams.update(IEEE_RC)
    fig, ax = plt.subplots()

    # Draw trajectories (behind loop closures)
    for name, pos in robots.items():
        c = color_map[name]
        ax.plot(pos[:, 0], pos[:, 1], color=c, linewidth=1.0,
                label=name, zorder=3)
        ax.plot(pos[0, 0], pos[0, 1], "o", color=c, markersize=3, zorder=4)

    # Scatter loop closure midpoints, one colour per robot pair
    pair_colors = ["#1a1a1a", "#e6194b", "#3cb44b", "#4363d8",
                   "#f58231", "#911eb4", "#42d4f4", "#f032e6",
                   "#bfef45", "#fabed4", "#469990", "#dcbeff",
                   "#9a6324", "#fffac8", "#800000"]
    for ci, ((ri, rj), mids) in enumerate(sorted(pair_midpoints.items())):
        pts = np.array(mids)
        c = pair_colors[ci % len(pair_colors)]
        ax.scatter(pts[:, 0], pts[:, 1], s=0.5, color=c, alpha=0.5, zorder=2,
                   label=f"{ri}↔{rj}")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="best", markerscale=4, fontsize=5)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.3)
    total = _count_csv(loops_csv)
    ax.set_title(f"GT Loop Closures — midpoints (1/{subsample} shown, total {total})",
                 fontsize=6)
    plt.tight_layout()

    for suffix in (".pdf", ".png"):
        out = gt_dir / f"gt_loop_closures_viz{suffix}"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"Saved to {out}")
    plt.close(fig)
    print(f"  ({n_found} loop closure midpoints plotted)")


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
    parser.add_argument("--dist",  type=float, default=1.0,
                        help="Max translation distance in metres (default: 1.0)")
    parser.add_argument("--angle", type=float, default=30.0,
                        help="Max relative rotation in degrees (default: 30)")
    parser.add_argument("--out",   type=Path,  default=None,
                        help="Output CSV path (default: <gt_dir>/gt_loop_closures.csv)")
    parser.add_argument("--plot", action="store_true",
                        help="Visualize trajectories and (subsampled) loop closure lines")
    parser.add_argument("--subsample", type=int, default=50,
                        help="Show 1-in-N loop closure lines in the plot (default: 50)")
    args = parser.parse_args()

    gt_dir: Path = args.gt_dir.resolve()
    if not gt_dir.exists():
        print(f"Error: {gt_dir} does not exist.")
        raise SystemExit(1)

    angle_rad = np.deg2rad(args.angle)
    out_path  = args.out or gt_dir / "gt_loop_closures.csv"

    print(f"Experiment : {gt_dir.name}")
    print(f"Thresholds : dist ≤ {args.dist} m,  angle ≤ {args.angle}°")
    print(f"Loading GT files from {gt_dir} ...")

    loops = extract_loops(gt_dir, args.dist, angle_rad)

    print(f"\nTotal loop closures: {len(loops)}")

    fieldnames = ["robot_i", "timestamp_i_ns", "robot_j", "timestamp_j_ns",
                  "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(loops)

    print(f"Saved to {out_path}")

    if args.plot:
        print("\nGenerating visualization...")
        visualize_loops(gt_dir, out_path, subsample=args.subsample)


if __name__ == "__main__":
    main()
