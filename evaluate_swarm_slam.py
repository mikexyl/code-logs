#!/usr/bin/env python3
"""
Evaluate ATE for Swarm-SLAM baseline results against ground truth, and write
standard output files so that comparison scripts (evaluate_loops_recall.py,
plot_algebraic_connectivity.py, plot_scalability.py, etc.) can include
Swarm-SLAM automatically.

Swarm-SLAM stores per-robot snapshot directories. Each snapshot contains:
  - optimized_global_pose_graph.g2o  (full multi-robot pose graph)
  - pose_timestampsN.csv             (vertex_id -> sec/nanosec for robot N)
  - log.csv                          (communication and loop statistics)

Vertex ID encoding: (ord(letter) << 48) | pose_index
  where letter = chr(ord('A') + robot_id), robot_id in {0,1,2,3,4,5}

What this script writes
-----------------------
  <swarm_dir>/evo_ape.zip / evo_ape.txt / evo.pdf
  <swarm_dir>/<robot_name>.tum                          (per-robot TUM trajectories)
  <swarm_dir>/<robot_name>/distributed/
      kimera_distributed_keyframes.csv                  (pose_index -> timestamp_ns)
      loop_closures.csv   (robot0 only — all inter-robot edges from the global g2o)
  <baseline_root>/<exp>-Swarm-SLAM_bandwidth.npy        (total comm bandwidth)

After running this script, re-run the full analysis pipeline:
  pixi run python3 evaluate_loops_recall.py <exp> ground_truth/<exp> --tol 5.0
  pixi run python3 plot_algebraic_connectivity.py <exp>
  pixi run python3 plot_loop_rotation_dist.py <exp> ground_truth/<exp>
  pixi run python3 plot_scalability.py <exp>

Usage:
    python3 evaluate_swarm_slam.py <swarm_slam_dir> [--gt_folder ground_truth]
    python3 evaluate_swarm_slam.py baselines/campus/Swarm-SLAM
    python3 evaluate_swarm_slam.py baselines/campus/Swarm-SLAM --gt_folder ground_truth --exp_name campus
"""

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.io import (load_gt_trajectory, load_alignment_from_evo_zip,
                      load_keyframes_csv, load_loop_closures_csv)
from utils.plot import (IEEE_RC, ROBOT_COLORS, apply_alignment, save_fig,
                        find_tum_position)

ROBOT_NAMES_FILE = "robot_names.yaml"


# ---------------------------------------------------------------------------
# Helpers: robot name loading
# ---------------------------------------------------------------------------

def load_robot_names(exp_dir: Path) -> dict[int, str]:
    """Load robot_id -> name mapping from <exp_dir>/robot_names.yaml."""
    yaml_path = exp_dir / ROBOT_NAMES_FILE
    if not yaml_path.exists():
        return {}
    mapping = {}
    with open(yaml_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("robot") and ":" in line:
                key, val = line.split(":", 1)
                try:
                    robot_id = int(key.strip().replace("robot", "").replace("_name", ""))
                    mapping[robot_id] = val.strip()
                except ValueError:
                    pass
    return mapping


# ---------------------------------------------------------------------------
# Helpers: Swarm-SLAM file discovery
# ---------------------------------------------------------------------------

def find_best_g2o(swarm_dir: Path) -> tuple[Path | None, str]:
    """Return (g2o_path, snapshot_name) for the most complete global pose graph."""
    best_g2o, best_lines, best_ts = None, 0, ""
    for robot_dir in sorted(swarm_dir.glob("*_experiment_robot_*")):
        for snap_dir in sorted(robot_dir.iterdir()):
            g2o = snap_dir / "optimized_global_pose_graph.g2o"
            if g2o.exists():
                lines = sum(1 for _ in open(g2o))
                if lines > best_lines:
                    best_lines = lines
                    best_g2o = g2o
                    best_ts = snap_dir.name
    return best_g2o, best_ts


def find_latest_pose_timestamps(swarm_dir: Path, robot_id: int) -> Path | None:
    """Return path to latest pose_timestampsN.csv for robot_id."""
    robot_dirs = list(swarm_dir.glob(f"*_experiment_robot_{robot_id}"))
    if not robot_dirs:
        return None
    ts_file = None
    for snap_dir in sorted(robot_dirs[0].iterdir()):
        candidate = snap_dir / f"pose_timestamps{robot_id}.csv"
        if candidate.exists():
            ts_file = candidate
    return ts_file


def load_robot_log(swarm_dir: Path, robot_id: int) -> dict[str, float]:
    """Parse the latest log.csv for robot_id -> {key: value}."""
    robot_dirs = list(swarm_dir.glob(f"*_experiment_robot_{robot_id}"))
    if not robot_dirs:
        return {}
    log_file = None
    for snap_dir in sorted(robot_dirs[0].iterdir()):
        candidate = snap_dir / "log.csv"
        if candidate.exists():
            log_file = candidate
    if log_file is None:
        return {}
    stats = {}
    with open(log_file) as f:
        for line in f:
            if "," in line:
                k, _, v = line.strip().partition(",")
                try:
                    stats[k.strip()] = float(v.strip())
                except ValueError:
                    pass
    return stats


# ---------------------------------------------------------------------------
# Helpers: g2o parsing
# ---------------------------------------------------------------------------

def parse_g2o_vertices(g2o_path: Path) -> dict[int, list]:
    """Parse VERTEX_SE3:QUAT lines.

    Returns {robot_id: [(pose_index, x, y, z, qx, qy, qz, qw), ...]}.
    """
    robots: dict[int, list] = {}
    with open(g2o_path) as f:
        for line in f:
            if not line.startswith("VERTEX_SE3:QUAT"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            vid = int(parts[1])
            robot_id = ((vid >> 48) & 0xFF) - ord('A')
            pose_index = vid & ((1 << 48) - 1)
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            qx, qy, qz, qw = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])
            robots.setdefault(robot_id, []).append(
                (pose_index, x, y, z, qx, qy, qz, qw))
    return robots


def parse_g2o_inter_robot_edges(g2o_path: Path) -> list[dict]:
    """Parse EDGE_SE3:QUAT lines that cross robot boundaries.

    Edge format: EDGE_SE3:QUAT v1 v2 tx ty tz qx qy qz qw [info]
    Returns list of dicts: robot1, pose1, robot2, pose2, tx, ty, tz, qx, qy, qz, qw.
    """
    edges = []
    with open(g2o_path) as f:
        for line in f:
            if not line.startswith("EDGE_SE3:QUAT"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            v1, v2 = int(parts[1]), int(parts[2])
            r1 = ((v1 >> 48) & 0xFF) - ord('A')
            r2 = ((v2 >> 48) & 0xFF) - ord('A')
            if r1 == r2:
                continue
            edges.append({
                'robot1': r1, 'pose1': v1 & ((1 << 48) - 1),
                'robot2': r2, 'pose2': v2 & ((1 << 48) - 1),
                'tx': float(parts[3]), 'ty': float(parts[4]), 'tz': float(parts[5]),
                'qx': float(parts[6]), 'qy': float(parts[7]),
                'qz': float(parts[8]), 'qw': float(parts[9]),
            })
    return edges


# ---------------------------------------------------------------------------
# Helpers: timestamp loading and TUM building
# ---------------------------------------------------------------------------

def load_pose_timestamps(ts_path: Path) -> dict[int, float]:
    """Load pose_timestampsN.csv -> {pose_index: timestamp_s}."""
    mapping = {}
    with open(ts_path) as f:
        for row in csv.DictReader(f):
            vid = int(row["vertice_id"])
            pose_index = vid & ((1 << 48) - 1)
            mapping[pose_index] = int(row["sec"]) + int(row["nanosec"]) * 1e-9
    return mapping


def build_tum_lines(poses: list, ts_map: dict[int, float]) -> list[str]:
    """Build sorted TUM lines: 'ts x y z qx qy qz qw'."""
    lines = []
    for (pose_index, x, y, z, qx, qy, qz, qw) in poses:
        ts = ts_map.get(pose_index)
        if ts is None:
            continue
        lines.append(f"{ts:.9f} {x} {y} {z} {qx} {qy} {qz} {qw}")
    lines.sort(key=lambda l: float(l.split()[0]))
    return lines


def load_gt_tum_lines(gt_path: Path) -> list[str]:
    """Load GT file as sorted TUM lines."""
    timestamps_ns, positions, rotations_xyzw = load_gt_trajectory(gt_path)
    lines = []
    for ts_ns, pos, rot in zip(timestamps_ns, positions, rotations_xyzw):
        x, y, z = pos
        qx, qy, qz, qw = rot
        lines.append(f"{ts_ns * 1e-9:.9f} {x} {y} {z} {qx} {qy} {qz} {qw}")
    lines.sort(key=lambda l: float(l.split()[0]))
    return lines


# ---------------------------------------------------------------------------
# Trajectory plotting
# ---------------------------------------------------------------------------

def plot_trajectories(swarm_dir: Path, gt_dir: Path,
                      robot_names: dict[int, str],
                      ts_maps: dict[int, dict[int, float]]) -> None:
    """Plot aligned trajectories + GT + loop closure lines, matching evaluate.py style."""
    ape_zip = swarm_dir / "evo_ape.zip"
    if not ape_zip.exists():
        print("  Skipping trajectory plot: evo_ape.zip not found")
        return

    rotation, translation, scale = load_alignment_from_evo_zip(ape_zip)

    plt.rcParams.update({
        **IEEE_RC,
        'figure.figsize': (3.5, 2.2),
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })
    fig, ax = plt.subplots()

    # Load per-robot TUM trajectories and GT, build structures for loop lookup
    robot_trajs: dict[int, tuple] = {}   # {robot_id: (ts_arr, pos_arr)}
    robot_kf_maps: dict[int, dict] = {}  # {robot_id: {pose_index: ts_s}}

    gt_plotted = False
    for i, (robot_id, robot_name) in enumerate(sorted(robot_names.items())):
        tum_path = swarm_dir / f"{robot_name}.tum"
        if not tum_path.exists():
            continue
        from utils.io import read_tum_trajectory
        ts, pos, _ = read_tum_trajectory(str(tum_path))
        if len(pos) == 0:
            continue
        ts_arr, pos_arr = np.array(ts), np.array(pos)
        robot_trajs[robot_id] = (ts_arr, pos_arr)

        aligned = apply_alignment(pos_arr, rotation, translation, scale)
        ax.plot(aligned[:, 0], aligned[:, 1],
                color=ROBOT_COLORS[i % len(ROBOT_COLORS)],
                linewidth=1.0, label=robot_name)

        # Load GT
        gt_candidates = list(gt_dir.glob(f"{robot_name}.*"))
        if gt_candidates:
            _, gt_pos, _ = load_gt_trajectory(gt_candidates[0])
            if len(gt_pos) > 0:
                label = 'GT' if not gt_plotted else None
                ax.plot(gt_pos[:, 0], gt_pos[:, 1],
                        color='gray', linewidth=0.5, alpha=0.5,
                        linestyle='--', label=label)
                gt_plotted = True

        # Load keyframes
        kf_path = swarm_dir / robot_name / "distributed" / "kimera_distributed_keyframes.csv"
        if kf_path.exists():
            robot_kf_maps[robot_id] = load_keyframes_csv(str(kf_path))

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linewidth=0.3)

    def _place_legend():
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center',
                   bbox_to_anchor=(0.5, -0.04), ncol=len(handles),
                   framealpha=0.9, edgecolor='none',
                   handlelength=1.0, handletextpad=0.3, columnspacing=0.8)

    _place_legend()
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(bottom=0.18)

    # Save no-loops version
    for suffix in ('.pdf', '.png'):
        fig.savefig(str(swarm_dir / f"trajectories_aligned_no_loops{suffix}"),
                    bbox_inches='tight', pad_inches=0.02)
        print(f"  Saved: trajectories_aligned_no_loops{suffix}")
    fig.set_size_inches(1.67, 1.5)
    fig.tight_layout(pad=0.3)
    for suffix in ('.pdf', '.png'):
        fig.savefig(str(swarm_dir / f"trajectories_aligned_no_loops_half{suffix}"),
                    bbox_inches='tight', pad_inches=0.01)
    fig.set_size_inches(3.5, 2.2)
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(bottom=0.18)

    # Collect and draw loop closure lines
    origin_robot_id = min(robot_kf_maps.keys()) if robot_kf_maps else None
    if origin_robot_id is not None:
        lc_path = (swarm_dir / robot_names[origin_robot_id] /
                   "distributed" / "loop_closures.csv")
        loop_lines = []
        if lc_path.exists():
            seen = set()
            for lc in load_loop_closures_csv(str(lc_path)):
                key = frozenset([(lc['robot1'], lc['pose1']), (lc['robot2'], lc['pose2'])])
                if key in seen:
                    continue
                seen.add(key)
                r1, p1, r2, p2 = lc['robot1'], lc['pose1'], lc['robot2'], lc['pose2']
                if r1 not in robot_trajs or r2 not in robot_trajs:
                    continue
                if r1 not in robot_kf_maps or r2 not in robot_kf_maps:
                    continue
                ts1 = robot_kf_maps[r1].get(p1)
                ts2 = robot_kf_maps[r2].get(p2)
                if ts1 is None or ts2 is None:
                    continue
                raw1 = find_tum_position(ts1, robot_trajs[r1][0], robot_trajs[r1][1])
                raw2 = find_tum_position(ts2, robot_trajs[r2][0], robot_trajs[r2][1])
                if raw1 is None or raw2 is None:
                    continue
                a1 = apply_alignment(raw1.reshape(1, 3), rotation, translation, scale)[0]
                a2 = apply_alignment(raw2.reshape(1, 3), rotation, translation, scale)[0]
                loop_lines.append((a1, a2))

        lc_label_added = False
        for a1, a2 in loop_lines:
            ax.plot([a1[0], a2[0]], [a1[1], a2[1]],
                    color='#CC2222', linewidth=1.5, alpha=0.8, zorder=10,
                    label='Loop closure' if not lc_label_added else None)
            lc_label_added = True
        if loop_lines:
            _place_legend()
            plt.tight_layout(pad=0.3)
            plt.subplots_adjust(bottom=0.18)
        print(f"  Loop closures: {len(loop_lines)} drawn")

    for suffix in ('.pdf', '.png'):
        fig.savefig(str(swarm_dir / f"trajectories_aligned{suffix}"),
                    bbox_inches='tight', pad_inches=0.02)
        print(f"  Saved: trajectories_aligned{suffix}")
    fig.set_size_inches(1.67, 1.5)
    fig.tight_layout(pad=0.3)
    for suffix in ('.pdf', '.png'):
        fig.savefig(str(swarm_dir / f"trajectories_aligned_half{suffix}"),
                    bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Standard file writers
# ---------------------------------------------------------------------------

def write_keyframes_csv(out_path: Path, ts_map: dict[int, float]) -> None:
    """Write kimera_distributed_keyframes.csv (pose_index -> timestamp_ns).

    Format matches what load_keyframes_csv() in utils/io.py expects:
      keyframe_stamp_ns, keyframe_id, submap_id, qx, qy, qz, qw, tx, ty, tz
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['keyframe_stamp_ns', 'keyframe_id', 'submap_id',
                         'qx', 'qy', 'qz', 'qw', 'tx', 'ty', 'tz'])
        for pose_index, ts_s in sorted(ts_map.items(), key=lambda kv: kv[0]):
            ts_ns = int(ts_s * 1e9)
            writer.writerow([ts_ns, pose_index, 0,
                              0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


def write_loop_closures_csv(out_path: Path, edges: list[dict],
                             ts_maps: dict[int, dict[int, float]]) -> None:
    """Write loop_closures.csv for robot0 containing all inter-robot edges.

    Format: robot1,pose1,robot2,pose2,qx,qy,qz,qw,tx,ty,tz,stamp_ns
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['robot1', 'pose1', 'robot2', 'pose2',
                         'qx', 'qy', 'qz', 'qw', 'tx', 'ty', 'tz', 'stamp_ns'])
        for e in edges:
            ts_s = ts_maps.get(e['robot1'], {}).get(e['pose1'])
            stamp_ns = int(ts_s * 1e9) if ts_s is not None else 0
            writer.writerow([e['robot1'], e['pose1'], e['robot2'], e['pose2'],
                             e['qx'], e['qy'], e['qz'], e['qw'],
                             e['tx'], e['ty'], e['tz'], stamp_ns])


def write_bandwidth_npy(out_path: Path, total_bytes: float) -> None:
    """Write a minimal bandwidth .npy dict with a single final data point.

    Keys match what plot_scalability.py / plot_ablation.py expect:
      t_sec, bow_MB, vlc_MB, cbs_MB
    All bandwidth is placed in bow_MB (Swarm-SLAM uses BoW-style front-end).
    """
    total_mb = total_bytes / 1e6
    data = {
        't_sec':   np.array([0.0, 1.0]),        # dummy time axis (single cumulative value)
        'bow_MB':  np.array([0.0, total_mb]),
        'vlc_MB':  np.array([0.0, 0.0]),
        'cbs_MB':  np.array([0.0, 0.0]),
    }
    np.save(out_path, data, allow_pickle=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Swarm-SLAM ATE and write standard comparison files")
    parser.add_argument("swarm_dir", type=Path,
                        help="Swarm-SLAM run dir (e.g. baselines/campus/Swarm-SLAM)")
    parser.add_argument("--gt_folder", type=Path, default=Path("ground_truth"))
    parser.add_argument("--exp_name", default=None,
                        help="Experiment name (default: parent dir name)")
    parser.add_argument("--robot_names", type=Path, default=None,
                        help="Path to robot_names.yaml")
    parser.add_argument("--skip_ate", action="store_true",
                        help="Skip evo_ape (useful when evo_ape.zip already exists)")
    args = parser.parse_args()

    swarm_dir = args.swarm_dir.resolve()
    if not swarm_dir.exists():
        print(f"Error: {swarm_dir} does not exist")
        sys.exit(1)

    exp_name = args.exp_name or swarm_dir.parent.name
    gt_dir = args.gt_folder / exp_name
    if not gt_dir.exists():
        print(f"Error: GT directory {gt_dir} does not exist")
        sys.exit(1)

    robot_names_path = args.robot_names or (Path(exp_name) / ROBOT_NAMES_FILE)
    robot_names = load_robot_names(
        robot_names_path.parent if robot_names_path else Path(exp_name))

    print(f"Swarm-SLAM dir : {swarm_dir}")
    print(f"GT dir         : {gt_dir}")
    print(f"Robot names    : {robot_names}")

    # ------------------------------------------------------------------
    # Find best g2o and parse it
    # ------------------------------------------------------------------
    g2o_path, snap_ts = find_best_g2o(swarm_dir)
    if g2o_path is None:
        print("Error: no optimized_global_pose_graph.g2o found")
        sys.exit(1)
    print(f"\nUsing g2o: {g2o_path} (snapshot: {snap_ts})")

    print("Parsing g2o vertices...")
    robots = parse_g2o_vertices(g2o_path)
    print(f"Found robot IDs: {sorted(robots.keys())}")

    print("Parsing inter-robot edges...")
    inter_edges = parse_g2o_inter_robot_edges(g2o_path)
    print(f"Found {len(inter_edges)} inter-robot edges")

    # ------------------------------------------------------------------
    # Load per-robot timestamps
    # ------------------------------------------------------------------
    ts_maps: dict[int, dict[int, float]] = {}
    for robot_id in sorted(robots.keys()):
        ts_path = find_latest_pose_timestamps(swarm_dir, robot_id)
        if ts_path is None:
            print(f"  Robot {robot_id}: no pose_timestamps file, skipping")
            continue
        ts_maps[robot_id] = load_pose_timestamps(ts_path)

    # ------------------------------------------------------------------
    # Build per-robot TUM trajectories
    # ------------------------------------------------------------------
    per_robot_est: dict[int, list[str]] = {}
    per_robot_gt: dict[int, list[str]] = {}

    for robot_id in sorted(robots.keys()):
        if robot_id not in ts_maps:
            continue
        robot_name = robot_names.get(robot_id, f"robot{robot_id}")
        poses = robots[robot_id]
        ts_map = ts_maps[robot_id]

        tum_lines = build_tum_lines(poses, ts_map)
        if not tum_lines:
            print(f"  Robot {robot_id} ({robot_name}): no matched timestamps, skipping")
            continue

        gt_candidates = list(gt_dir.glob(f"{robot_name}.*"))
        if not gt_candidates:
            print(f"  Robot {robot_id} ({robot_name}): no GT file in {gt_dir}, skipping")
            continue
        gt_lines = load_gt_tum_lines(gt_candidates[0])

        per_robot_est[robot_id] = tum_lines
        per_robot_gt[robot_id] = gt_lines
        print(f"  Robot {robot_id} ({robot_name}): "
              f"{len(tum_lines)} est poses, {len(gt_lines)} GT poses")

    if not per_robot_est:
        print("Error: no robots with matched timestamps and GT")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Save per-robot TUM files
    # ------------------------------------------------------------------
    for robot_id, lines in per_robot_est.items():
        robot_name = robot_names.get(robot_id, f"robot{robot_id}")
        tum_out = swarm_dir / f"{robot_name}.tum"
        with open(tum_out, 'w') as f:
            f.write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # Write standard files for comparison scripts
    # ------------------------------------------------------------------
    print("\nWriting standard comparison files...")

    # 1. Per-robot kimera_distributed_keyframes.csv
    origin_robot_id = min(ts_maps.keys())
    for robot_id, ts_map in ts_maps.items():
        robot_name = robot_names.get(robot_id, f"robot{robot_id}")
        kf_path = swarm_dir / robot_name / "distributed" / "kimera_distributed_keyframes.csv"
        write_keyframes_csv(kf_path, ts_map)
        print(f"  Keyframes: {kf_path} ({len(ts_map)} entries)")

    # 1b. robot_names.yaml so discover_robots() can identify robots by ID
    yaml_out = swarm_dir / "robot_names.yaml"
    with open(yaml_out, 'w') as f:
        for robot_id, name in sorted(robot_names.items()):
            if robot_id in ts_maps:
                f.write(f"robot{robot_id}_name: {name}\n")
    print(f"  robot_names.yaml: {yaml_out}")

    # 2. loop_closures.csv in origin robot's distributed dir
    #    (all inter-robot edges from the global g2o, attributed to robot0)
    origin_name = robot_names.get(origin_robot_id, f"robot{origin_robot_id}")
    lc_path = swarm_dir / origin_name / "distributed" / "loop_closures.csv"
    write_loop_closures_csv(lc_path, inter_edges, ts_maps)
    print(f"  Loop closures: {lc_path} ({len(inter_edges)} edges)")

    # 3. Bandwidth .npy in baseline root (baselines/<exp>/<exp>-Swarm-SLAM_bandwidth.npy)
    #    Sum total_front_end_cumulative_communication_bytes across all robots
    total_bytes = 0.0
    for robot_id in sorted(ts_maps.keys()):
        stats = load_robot_log(swarm_dir, robot_id)
        b = stats.get("total_front_end_cumulative_communication_bytes", 0.0)
        total_bytes += b
        if b:
            robot_name = robot_names.get(robot_id, f"robot{robot_id}")
            print(f"  Robot {robot_id} ({robot_name}) comm: {b/1e6:.1f} MB")

    bw_npy_path = swarm_dir.parent / f"{exp_name}-Swarm-SLAM_bandwidth.npy"
    write_bandwidth_npy(bw_npy_path, total_bytes)
    print(f"  Bandwidth npy: {bw_npy_path} (total {total_bytes/1e6:.1f} MB)")

    # ------------------------------------------------------------------
    # ATE evaluation
    # ------------------------------------------------------------------
    ape_zip = swarm_dir / "evo_ape.zip"
    ape_txt = swarm_dir / "evo_ape.txt"
    ape_plot = swarm_dir / "evo.pdf"

    if args.skip_ate and ape_zip.exists():
        print(f"\nSkipping evo_ape (--skip_ate, {ape_zip} exists)")
    else:
        for f in [ape_zip, ape_plot]:
            if f.exists():
                f.unlink()

        with tempfile.NamedTemporaryFile(mode='w', suffix='_est.tum', delete=False) as f_est, \
             tempfile.NamedTemporaryFile(mode='w', suffix='_gt.tum', delete=False) as f_gt:
            est_tmp = f_est.name
            gt_tmp = f_gt.name
            for robot_id in sorted(per_robot_est.keys()):
                for line in per_robot_est[robot_id]:
                    f_est.write(line + "\n")
                for line in per_robot_gt[robot_id]:
                    f_gt.write(line + "\n")

        cmd = ["evo_ape", "tum", gt_tmp, est_tmp, "-va",
               "--align", "--t_max_diff", "1.5",
               "--plot_mode", "xy",
               "--save_plot", str(ape_plot),
               "--save_results", str(ape_zip)]
        print(f"\nRunning: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True)
            print("\n" + "=" * 40 + "\nEVO APE RESULTS\n" + "=" * 40)
            print(result.stdout)
            with open(ape_txt, 'w') as f:
                f.write(result.stdout)
        except subprocess.CalledProcessError as e:
            print("\nError running evo_ape:")
            print(e.stderr)
            sys.exit(1)
        except FileNotFoundError:
            print("\nError: 'evo_ape' not found. Run inside pixi shell.")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Trajectory plots
    # ------------------------------------------------------------------
    print("\nPlotting trajectories...")
    plot_trajectories(swarm_dir, gt_dir, robot_names, ts_maps)

    print(f"\nDone. Next steps to include Swarm-SLAM in comparison plots:")
    print(f"  pixi run python3 evaluate_loops_recall.py {exp_name} "
          f"ground_truth/{exp_name} --tol 5.0")
    print(f"  pixi run python3 plot_algebraic_connectivity.py {exp_name}")
    print(f"  pixi run python3 plot_loop_rotation_dist.py {exp_name} "
          f"ground_truth/{exp_name}")
    print(f"  pixi run python3 plot_scalability.py {exp_name}")


if __name__ == "__main__":
    main()
