#!/usr/bin/env python3
"""
Plot the relative-rotation distribution of detected inter-robot loop closures,
using ground-truth trajectory orientations.

For each detected loop the GT trajectory orientations of both robots are looked
up at the loop timestamps (nearest-neighbour), and the angle of the relative
rotation is computed.

If <exp_dir> contains variant sub-folders (each holding robot subdirs with
distributed/ or dpgo/ data), all variants are overlaid on the same axes.
Baselines are auto-discovered from baselines/<exp_dir.name>/*/ and shown with
dashed lines.  If no variants are found, <exp_dir> itself is evaluated directly
(backward-compatible).

Usage:
    python plot_loop_rotation_dist.py <exp_dir> <gt_dir>
    python plot_loop_rotation_dist.py campus ground_truth/campus
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


def _is_robot_dir(d: Path) -> bool:
    return (d / 'distributed').is_dir() or (d / 'dpgo').is_dir()


def discover_variants(exp_dir: Path) -> list[Path]:
    """Return subdirs of exp_dir that contain robot subdirs."""
    variants = []
    for d in sorted(exp_dir.iterdir()):
        if not d.is_dir():
            continue
        if any(_is_robot_dir(sub) for sub in d.iterdir() if sub.is_dir()):
            variants.append(d)
    return variants


def discover_baselines(exp_dir: Path) -> list[Path]:
    """Return baseline method dirs from baselines/<exp_dir.name>/*/."""
    baseline_root = exp_dir.parent / 'baselines' / exp_dir.name
    if not baseline_root.exists():
        return []
    return [d for d in sorted(baseline_root.iterdir())
            if d.is_dir() and discover_robots(d)]


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


def load_angles_for_dir(d: Path, gt_dir: Path, max_gap_s: float) -> tuple[str, np.ndarray] | None:
    """Discover robots, load loops, compute rotation angles. Returns (label, angles) or None."""
    id_to_name = discover_robots(d)
    if not id_to_name:
        print(f'  [SKIP] No robots found in {d.name}')
        return None
    loops = load_detected_loops(d, id_to_name)
    gt_rots = load_gt_rotations(gt_dir, list(set(id_to_name.values())))
    angles = compute_angles(loops, gt_rots, max_gap_s)
    print(f'[{d.name}] {len(loops)} detected loops → {len(angles)} with GT rotation')
    return (d.name, angles)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Plot GT relative-rotation distribution of detected loop closures.'
    )
    parser.add_argument('exp_dir', type=Path, help='Experiment folder')
    parser.add_argument('gt_dir',  type=Path, help='GT folder with <robot>.csv/.txt files')
    parser.add_argument('--max-gap', type=float, default=2.5, dest='max_gap',
                        help='Max timestamp gap for GT lookup in seconds (default: 2.5)')
    parser.add_argument('--max-angle', type=float, default=120.0, dest='max_angle',
                        help='Max angle for histogram x-axis in degrees (default: 120)')
    parser.add_argument('--bins', type=int, default=24,
                        help='Number of histogram bins (default: 24, i.e. 5° steps up to 120°)')
    args = parser.parse_args()

    exp_dir = args.exp_dir.resolve()
    gt_dir  = args.gt_dir.resolve()

    # Discover variants and baselines
    variants  = discover_variants(exp_dir)
    baselines = discover_baselines(exp_dir)

    variant_data:  list[tuple[str, np.ndarray]] = []
    baseline_data: list[tuple[str, np.ndarray]] = []

    if variants:
        print(f'Found {len(variants)} variant(s): {[v.name for v in variants]}')
        for v in variants:
            r = load_angles_for_dir(v, gt_dir, args.max_gap)
            if r:
                variant_data.append(r)
    else:
        # Single-experiment fallback
        r = load_angles_for_dir(exp_dir, gt_dir, args.max_gap)
        if r:
            variant_data.append(r)

    if baselines:
        print(f'Found {len(baselines)} baseline(s): {[b.name for b in baselines]}')
        for b in baselines:
            r = load_angles_for_dir(b, gt_dir, args.max_gap)
            if r:
                baseline_data.append(r)

    all_data = variant_data + baseline_data
    if not all_data:
        print('No data to plot.')
        raise SystemExit(1)

    # Print breakdown per dataset
    bin_edges = np.linspace(0, args.max_angle, args.bins + 1)
    for label, angles in all_data:
        counts, _ = np.histogram(angles, bins=bin_edges)
        total = len(angles)
        print(f'\n{label} GT rotation distribution:')
        for i in range(len(bin_edges) - 1):
            pct = 100 * counts[i] / total if total > 0 else 0.0
            print(f'  {bin_edges[i]:5.1f}-{bin_edges[i+1]:5.1f}°: {counts[i]:4d}  ({pct:.1f}%)')

    # Plot — one subplot per method, stacked vertically, shared x-axis
    n_rows = len(all_data)
    row_h  = 1.3
    plt.rcParams.update({**IEEE_RC, 'figure.figsize': (3.5, max(2.0, n_rows * row_h))})
    fig, axes = plt.subplots(n_rows, 1, sharex=True,
                             gridspec_kw={'hspace': 0.15})
    if n_rows == 1:
        axes = [axes]

    y_max = max(
        (np.histogram(angles, bins=bin_edges)[0].max() if len(angles) > 0 else 1)
        for _, angles in all_data
    )

    for i, (label, angles) in enumerate(all_data):
        ax = axes[i]
        is_baseline = i >= len(variant_data)
        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]

        if len(angles) > 0:
            counts, _ = np.histogram(angles, bins=bin_edges)
            ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges),
                   align='edge', color=color,
                   edgecolor='black' if is_baseline else color,
                   linewidth=0.4,
                   alpha=0.85)
        else:
            counts = np.zeros(len(bin_edges) - 1, dtype=int)

        # Method label inside panel
        tag = f'{label}  (n={len(angles)})'
        ax.text(0.98, 0.88, tag, transform=ax.transAxes,
                ha='right', va='top', fontsize=5,
                style='italic' if is_baseline else 'normal')

        ax.set_ylim(0, y_max * 1.25)
        ax.set_yticks([0, int(y_max)])
        ax.yaxis.set_tick_params(labelsize=4.5)
        ax.grid(True, axis='y', alpha=0.25, linestyle='--', linewidth=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Shared x-axis ticks and label
    axes[-1].set_xlabel('GT Relative Rotation Angle (°)', fontsize=6)
    tick_step = max(1, int((args.max_angle / 8) / 5) * 5)  # ~8 ticks, rounded to 5°
    axes[-1].set_xticks(np.arange(0, args.max_angle + 1, tick_step))
    axes[-1].xaxis.set_tick_params(labelsize=5)

    # Shared y label centred on figure
    fig.text(0.01, 0.5, 'Detected Loops', va='center', rotation='vertical', fontsize=6)

    fig.suptitle(f'{exp_dir.name} — GT Rotation of Detected Loops', fontsize=7, y=1.01)
    plt.tight_layout()

    save_fig(fig, exp_dir / 'loop_rotation_dist')
    plt.close(fig)


if __name__ == '__main__':
    main()
