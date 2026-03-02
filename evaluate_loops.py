#!/usr/bin/env python3
"""
Visualize inter-robot loop closures on top of all robot trajectories.

For each robot subdirectory in the experiment folder the script expects:
  <robot_dir>/dpgo/Robot <N>.tum                           – TUM trajectory
  <robot_dir>/distributed/kimera_distributed_keyframes.csv – pose-index → timestamp
  <robot_dir>/distributed/loop_closures.csv                – loop closure pairs

Lookup chain per loop closure (robot_i, pose_i) <-> (robot_j, pose_j):
  1. keyframe CSV of robot_i  : pose_i  → wall-clock timestamp (ns → s)
  2. TUM file of robot_i      : timestamp → XYZ position (nearest match)
  3. Repeat for robot_j
  4. Draw a line between the two world-frame positions

Usage:
    python evaluate_loops.py <experiment_folder>
    python evaluate_loops.py a5678
    python evaluate_loops.py a5678 --save loops_viz.pdf
    python evaluate_loops.py a5678 --subsample 10
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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

# Max allowed gap (seconds) between a keyframe timestamp and the nearest TUM
# pose.  Larger gaps indicate a missing or mismatched trajectory segment.
MAX_TIMESTAMP_GAP_S = 2.5


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_tum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a TUM trajectory file.

    Returns:
        timestamps : (N,)  float64, seconds
        positions  : (N,3) float64, XYZ metres
    """
    timestamps, positions = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    timestamps.append(float(parts[0]))
                    positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    continue
    return np.array(timestamps, dtype=np.float64), np.array(positions, dtype=np.float64)


def load_keyframes(path: Path) -> dict[int, float]:
    """Load kimera_distributed_keyframes.csv.

    Returns:
        {keyframe_id: timestamp_seconds}
    """
    kf_map: dict[int, float] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                kid = int(row['keyframe_id'])
                ts_s = float(row['keyframe_stamp_ns']) / 1e9
                kf_map[kid] = ts_s
            except (ValueError, KeyError):
                continue
    return kf_map


def load_loop_closures(path: Path) -> list[dict]:
    """Load loop_closures.csv.

    Returns list of dicts with keys: robot1, pose1, robot2, pose2.
    """
    loops: list[dict] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                loops.append({
                    'robot1': int(row['robot1']),
                    'pose1':  int(row['pose1']),
                    'robot2': int(row['robot2']),
                    'pose2':  int(row['pose2']),
                })
            except (ValueError, KeyError):
                continue
    return loops


# ---------------------------------------------------------------------------
# Position lookup
# ---------------------------------------------------------------------------

def find_position(ts_s: float,
                  timestamps: np.ndarray,
                  positions: np.ndarray) -> np.ndarray | None:
    """Return the XYZ position from a TUM trajectory nearest to ts_s.

    Returns None if the nearest match is further than MAX_TIMESTAMP_GAP_S away.
    """
    if len(timestamps) == 0:
        return None
    idx = int(np.argmin(np.abs(timestamps - ts_s)))
    if abs(timestamps[idx] - ts_s) > MAX_TIMESTAMP_GAP_S:
        return None
    return positions[idx]


# ---------------------------------------------------------------------------
# Robot discovery
# ---------------------------------------------------------------------------

def discover_robots(exp_dir: Path) -> dict[int, dict]:
    """Scan experiment_dir for robot subdirs that contain the required files.

    A valid robot subdir has:
      dpgo/Robot <N>.tum
      distributed/kimera_distributed_keyframes.csv
      distributed/loop_closures.csv

    Returns:
        {robot_id: {'tum': Path, 'keyframes': Path, 'loops': Path, 'name': str}}

    The robot_id is taken from the TUM filename ("Robot <N>.tum" → N).
    """
    robots: dict[int, dict] = {}
    for robot_dir in sorted(exp_dir.iterdir()):
        if not robot_dir.is_dir():
            continue
        dpgo = robot_dir / 'dpgo'
        dist = robot_dir / 'distributed'
        if not dpgo.is_dir() or not dist.is_dir():
            continue
        kf_path = dist / 'kimera_distributed_keyframes.csv'
        lc_path = dist / 'loop_closures.csv'
        if not kf_path.exists() or not lc_path.exists():
            continue
        for tum_path in sorted(dpgo.glob('Robot *.tum')):
            try:
                robot_id = int(tum_path.stem.split()[-1])
            except ValueError:
                continue
            if robot_id in robots:
                print(f'  Warning: duplicate robot_id {robot_id}, skipping {tum_path}')
                continue
            robots[robot_id] = {
                'tum':       tum_path,
                'keyframes': kf_path,
                'loops':     lc_path,
                'name':      robot_dir.name,
            }
    return robots


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Visualize inter-robot loop closures on top of robot trajectories.'
    )
    parser.add_argument('experiment_folder', type=Path,
                        help='Experiment folder (e.g. a5678)')
    parser.add_argument('--save', type=Path, default=None,
                        help='Output PDF path (a .png sibling is also saved). '
                             'Default: <experiment_folder>/loops_viz.pdf')
    parser.add_argument('--subsample', type=int, default=1,
                        help='Show 1-in-N loop closure lines to avoid overplotting '
                             '(default: 1 = show all)')
    args = parser.parse_args()

    exp_dir = args.experiment_folder.resolve()
    if not exp_dir.is_dir():
        print(f'Error: {exp_dir} is not a directory.')
        raise SystemExit(1)

    # --- discover robots ---------------------------------------------------
    robots = discover_robots(exp_dir)
    if not robots:
        print('No robots found. Each robot subdir must contain:')
        print('  dpgo/Robot <N>.tum')
        print('  distributed/kimera_distributed_keyframes.csv')
        print('  distributed/loop_closures.csv')
        raise SystemExit(1)

    print(f'Found {len(robots)} robot(s) in {exp_dir.name}:')
    for rid, info in sorted(robots.items()):
        print(f'  robot {rid} ({info["name"]}): {info["tum"].name}')

    # --- load trajectories and keyframe maps ------------------------------
    trajectories: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    keyframe_maps: dict[int, dict[int, float]] = {}

    for rid, info in sorted(robots.items()):
        ts, pos = load_tum(info['tum'])
        trajectories[rid] = (ts, pos)
        keyframe_maps[rid] = load_keyframes(info['keyframes'])
        print(f'  Robot {rid}: {len(ts)} TUM poses, '
              f'{len(keyframe_maps[rid])} keyframes')

    # --- collect and deduplicate loop closures ----------------------------
    seen: set[frozenset] = set()
    all_loops: list[dict] = []
    for rid, info in sorted(robots.items()):
        for lc in load_loop_closures(info['loops']):
            key: frozenset = frozenset([
                (lc['robot1'], lc['pose1']),
                (lc['robot2'], lc['pose2']),
            ])
            if key not in seen:
                seen.add(key)
                all_loops.append(lc)

    print(f'Total unique loop closures: {len(all_loops)}')

    # --- resolve endpoint positions ---------------------------------------
    loop_lines: list[tuple[np.ndarray, np.ndarray]] = []
    n_missing = 0
    for lc in all_loops:
        r1, p1 = lc['robot1'], lc['pose1']
        r2, p2 = lc['robot2'], lc['pose2']

        # Skip if we don't have trajectory data for either robot
        if r1 not in trajectories or r2 not in trajectories:
            n_missing += 1
            continue

        ts1 = keyframe_maps[r1].get(p1)
        ts2 = keyframe_maps[r2].get(p2)
        if ts1 is None or ts2 is None:
            n_missing += 1
            continue

        pos1 = find_position(ts1, *trajectories[r1])
        pos2 = find_position(ts2, *trajectories[r2])
        if pos1 is None or pos2 is None:
            n_missing += 1
            continue

        loop_lines.append((pos1, pos2))

    print(f'Resolved {len(loop_lines)} loop closure positions '
          f'({n_missing} unresolved)')

    # --- plot -------------------------------------------------------------
    plt.rcParams.update(IEEE_RC)
    fig, ax = plt.subplots()

    sorted_ids = sorted(trajectories.keys())
    colors = {rid: ROBOT_COLORS[i % len(ROBOT_COLORS)]
              for i, rid in enumerate(sorted_ids)}

    for rid in sorted_ids:
        ts, pos = trajectories[rid]
        if len(pos) == 0:
            continue
        name = robots[rid]['name']
        color = colors[rid]
        ax.plot(pos[:, 0], pos[:, 1],
                color=color, linewidth=0.8, label=name, zorder=3)
        ax.plot(pos[0, 0], pos[0, 1],
                'o', color=color, markersize=3, zorder=4)

    lc_label_added = False
    n_drawn = 0
    for k, (p1, p2) in enumerate(loop_lines):
        if k % args.subsample != 0:
            continue
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color='#CC2222', linewidth=0.5, alpha=0.5, zorder=5,
                label='loop closure' if not lc_label_added else None)
        lc_label_added = True
        n_drawn += 1

    print(f'Drew {n_drawn}/{len(loop_lines)} loop closure lines')

    title = f'{exp_dir.name} — {n_drawn}/{len(loop_lines)} loop closures shown'
    if args.subsample > 1:
        title += f' (1/{args.subsample})'
    ax.set_title(title, fontsize=6)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.legend(loc='best', fontsize=6, markerscale=3)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.3)
    plt.tight_layout()

    save_path = (args.save if args.save is not None
                 else exp_dir / 'loops_viz.pdf').resolve()
    for suffix in ('.pdf', '.png'):
        out = save_path.with_suffix(suffix)
        fig.savefig(out, bbox_inches='tight', dpi=300)
        print(f'Saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
