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
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils.io import read_tum_trajectory, load_keyframes_csv, load_loop_closures_csv
from utils.plot import IEEE_RC, ROBOT_COLORS, find_tum_position, save_fig


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
        ts, pos, _ = read_tum_trajectory(str(info['tum']))
        trajectories[rid] = (ts, pos)
        keyframe_maps[rid] = load_keyframes_csv(str(info['keyframes']))
        print(f'  Robot {rid}: {len(ts)} TUM poses, '
              f'{len(keyframe_maps[rid])} keyframes')

    # --- collect and deduplicate loop closures ----------------------------
    seen: set[frozenset] = set()
    all_loops: list[dict] = []
    for rid, info in sorted(robots.items()):
        for lc in load_loop_closures_csv(str(info['loops'])):
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

        if r1 not in trajectories or r2 not in trajectories:
            n_missing += 1
            continue

        ts1 = keyframe_maps[r1].get(p1)
        ts2 = keyframe_maps[r2].get(p2)
        if ts1 is None or ts2 is None:
            n_missing += 1
            continue

        pos1 = find_tum_position(ts1, *trajectories[r1])
        pos2 = find_tum_position(ts2, *trajectories[r2])
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
        color = colors[rid]
        ax.plot(pos[:, 0], pos[:, 1],
                color=color, linewidth=0.8, label=robots[rid]['name'], zorder=3)
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

    base = (args.save if args.save is not None
            else exp_dir / 'loops_viz.pdf').resolve()
    save_fig(fig, base)
    plt.close(fig)


if __name__ == '__main__':
    main()
