#!/usr/bin/env python3
"""
Plot the relative-rotation distribution of detected inter-robot loop closures,
using ground-truth trajectory orientations.

For each detected loop the GT trajectory orientations of both robots are looked
up at the loop timestamps (nearest-neighbour), and the angle of the relative
rotation is computed.  A histogram is plotted for each supplied method,
overlaid on a single axes.

Usage:
    python plot_loop_rotation_dist.py <exp_dir> <gt_dir> \
        [--baseline <baseline_dir>] [--label <label>] \
        [--baseline-label <label>] [--max-gap 2.5]

Example:
    python plot_loop_rotation_dist.py campus ground_truth/campus \
        --baseline baselines/campus/Kimera-Multi --label CoDE-SLAM
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml

from utils.io import load_gt_trajectory, load_keyframes_csv, load_loop_closures_csv
from utils.plot import IEEE_RC, ROBOT_COLORS, save_fig


# ---------------------------------------------------------------------------
# Robot discovery + loop loading
# ---------------------------------------------------------------------------

def discover_robots(exp_dir: Path) -> dict[int, str]:
    """Return {robot_id: robot_dir_name} (yaml first, dpgo fallback)."""
    yaml_path = exp_dir / 'robot_names.yaml'
    if yaml_path.exists():
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        id_to_name: dict[int, str] = {}
        for key, name in data.items():
            m = re.match(r'robot(\d+)_name', key)
            if m:
                id_to_name[int(m.group(1))] = name
        return id_to_name

    id_to_name = {}
    for robot_dir in sorted(exp_dir.iterdir()):
        if not robot_dir.is_dir():
            continue
        dpgo = robot_dir / 'dpgo'
        if not dpgo.is_dir():
            continue
        for tum in sorted(dpgo.glob('Robot *.tum')):
            try:
                rid = int(tum.stem.split()[-1])
            except ValueError:
                continue
            id_to_name[rid] = robot_dir.name
    return id_to_name


def load_detected_loops(exp_dir: Path, id_to_name: dict[int, str]) -> list[dict]:
    """Load unique inter-robot loop closures resolved to timestamps (s)."""
    kf_maps: dict[int, dict[int, float]] = {}
    for rid, rname in id_to_name.items():
        kf_path = exp_dir / rname / 'distributed' / 'kimera_distributed_keyframes.csv'
        if kf_path.exists():
            kf_maps[rid] = load_keyframes_csv(str(kf_path))

    seen: set[frozenset] = set()
    loops: list[dict] = []
    for rid, rname in id_to_name.items():
        lc_path = exp_dir / rname / 'distributed' / 'loop_closures.csv'
        if not lc_path.exists():
            continue
        for lc in load_loop_closures_csv(str(lc_path)):
            r1, p1 = lc['robot1'], lc['pose1']
            r2, p2 = lc['robot2'], lc['pose2']
            if r1 == r2:
                continue
            key: frozenset = frozenset([(r1, p1), (r2, p2)])
            if key in seen:
                continue
            seen.add(key)
            if r1 not in kf_maps or r2 not in kf_maps:
                continue
            t1 = kf_maps[r1].get(p1)
            t2 = kf_maps[r2].get(p2)
            if t1 is None or t2 is None:
                continue
            loops.append({'name1': id_to_name[r1], 't1_s': t1,
                          'name2': id_to_name[r2], 't2_s': t2})
    return loops


# ---------------------------------------------------------------------------
# GT rotation lookup
# ---------------------------------------------------------------------------

def load_gt_rotations(gt_dir: Path, robot_names: list[str]) -> dict[str, tuple]:
    """Return {robot_name: (timestamps_s (N,), rotations_xyzw (N,4))}."""
    result: dict[str, tuple] = {}
    for name in robot_names:
        for ext in ('.csv', '.txt'):
            p = gt_dir / (name + ext)
            if p.exists():
                ts_ns, _, rots = load_gt_trajectory(str(p))
                result[name] = (ts_ns / 1e9, rots)
                break
    return result


def nearest_rotation(ts_s: float, timestamps: np.ndarray,
                     rotations: np.ndarray, max_gap_s: float) -> np.ndarray | None:
    idx = int(np.argmin(np.abs(timestamps - ts_s)))
    if abs(timestamps[idx] - ts_s) > max_gap_s:
        return None
    return rotations[idx]


def relative_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angle (degrees) of the relative rotation between two unit quaternions (xyzw)."""
    dot = min(1.0, abs(float(np.dot(q1, q2))))
    return float(np.degrees(2.0 * np.arccos(dot)))


def compute_angles(loops: list[dict], gt_rots: dict, max_gap_s: float) -> np.ndarray:
    """Return array of GT relative rotation angles (degrees) for detected loops."""
    angles = []
    for lc in loops:
        n1, t1 = lc['name1'], lc['t1_s']
        n2, t2 = lc['name2'], lc['t2_s']
        if n1 not in gt_rots or n2 not in gt_rots:
            continue
        r1 = nearest_rotation(t1, *gt_rots[n1], max_gap_s)
        r2 = nearest_rotation(t2, *gt_rots[n2], max_gap_s)
        if r1 is None or r2 is None:
            continue
        angles.append(relative_angle_deg(r1, r2))
    return np.array(angles)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Plot GT relative-rotation distribution of detected loop closures.'
    )
    parser.add_argument('exp_dir', type=Path, help='Main experiment folder')
    parser.add_argument('gt_dir',  type=Path, help='GT folder with <robot>.csv/.txt files')
    parser.add_argument('--baseline', type=Path, default=None,
                        help='Baseline experiment folder to overlay')
    parser.add_argument('--label', default=None,
                        help='Legend label for main experiment (default: folder name)')
    parser.add_argument('--baseline-label', default=None, dest='baseline_label',
                        help='Legend label for baseline (default: folder name)')
    parser.add_argument('--max-gap', type=float, default=2.5, dest='max_gap',
                        help='Max timestamp gap for GT lookup in seconds (default: 2.5)')
    parser.add_argument('--max-angle', type=float, default=120.0, dest='max_angle',
                        help='Max angle for histogram x-axis in degrees (default: 120)')
    parser.add_argument('--bins', type=int, default=12,
                        help='Number of histogram bins (default: 12, i.e. 10° steps up to 120°)')
    args = parser.parse_args()

    exp_dir = args.exp_dir.resolve()
    gt_dir  = args.gt_dir.resolve()

    datasets: list[tuple[str, np.ndarray]] = []

    # ---- main experiment ----
    id_to_name = discover_robots(exp_dir)
    if not id_to_name:
        print(f'No robots found in {exp_dir}')
        raise SystemExit(1)
    loops_main = load_detected_loops(exp_dir, id_to_name)
    gt_rots    = load_gt_rotations(gt_dir, list(set(id_to_name.values())))
    angles_main = compute_angles(loops_main, gt_rots, args.max_gap)
    label_main  = args.label or exp_dir.name
    print(f'[{label_main}] {len(loops_main)} detected loops → {len(angles_main)} with GT rotation')
    datasets.append((label_main, angles_main))

    # ---- optional baseline ----
    if args.baseline:
        base_dir = args.baseline.resolve()
        id_to_name_b = discover_robots(base_dir)
        if not id_to_name_b:
            print(f'No robots found in {base_dir}')
        else:
            loops_base  = load_detected_loops(base_dir, id_to_name_b)
            gt_rots_b   = load_gt_rotations(gt_dir, list(set(id_to_name_b.values())))
            angles_base = compute_angles(loops_base, gt_rots_b, args.max_gap)
            label_base  = args.baseline_label or base_dir.name
            print(f'[{label_base}] {len(loops_base)} detected loops → {len(angles_base)} with GT rotation')
            datasets.append((label_base, angles_base))

    # ---- print breakdown ----
    bin_edges = np.linspace(0, args.max_angle, args.bins + 1)
    for label, angles in datasets:
        counts, _ = np.histogram(angles, bins=bin_edges)
        total = len(angles)
        print(f'\n{label} GT rotation distribution:')
        for i in range(len(bin_edges) - 1):
            pct = 100 * counts[i] / total if total > 0 else 0.0
            print(f'  {bin_edges[i]:5.1f}-{bin_edges[i+1]:5.1f}°: {counts[i]:4d}  ({pct:.1f}%)')

    # ---- plot ----
    plt.rcParams.update({**IEEE_RC, 'figure.figsize': (3.5, 2.8)})
    fig, ax = plt.subplots()

    for i, (label, angles) in enumerate(datasets):
        ax.hist(angles, bins=bin_edges, color=ROBOT_COLORS[i % len(ROBOT_COLORS)],
                alpha=0.7, label=label)

    ax.set_xlabel('GT Relative Rotation Angle (°)')
    ax.set_ylabel('Number of Detected Loops')
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.3)
    plt.tight_layout()

    out_path = exp_dir / 'loop_rotation_dist'
    save_fig(fig, out_path)
    plt.close(fig)


if __name__ == '__main__':
    main()
