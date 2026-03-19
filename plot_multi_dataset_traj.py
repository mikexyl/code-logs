#!/usr/bin/env python3
"""
Combined trajectory figure for multiple datasets (ns-as variant).
One subplot per dataset, shared legend at bottom.

Usage:
    python3 plot_multi_dataset_traj.py
    python3 plot_multi_dataset_traj.py --output multi_dataset_traj
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.io import (read_tum_trajectory, load_alignment_from_evo_zip,
                      umeyama, load_gt_trajectories_by_name,
                      load_keyframes_csv, load_loop_closures_csv)
from utils.plot import IEEE_RC, ROBOT_COLORS, save_fig, apply_alignment, find_tum_position

BASE = Path(__file__).parent

_DATASET_IDS = ['g123', 'g156', 'g2345', 'a12', 'gate']

def _load_dataset_aliases():
    p = BASE / 'dataset_aliases.yaml'
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}

def _build_datasets():
    aliases = _load_dataset_aliases()
    result = []
    for ds in _DATASET_IDS:
        label = aliases.get(ds, ds)
        result.append((ds, label, BASE / ds / 'ns-as', BASE / 'ground_truth' / ds))
    return result

DATASETS = _build_datasets()



def _load_tum_positions(ns_as_dir):
    """Load positions from Robot *.tum files in robot subdirs."""
    result = {}
    for rdir in sorted(ns_as_dir.iterdir()):
        if not rdir.is_dir():
            continue
        dpgo = rdir / 'dpgo'
        if not dpgo.is_dir():
            continue
        # pick last non-empty tum (handles CBS-style Robot N_ts.tum)
        tums = sorted([p for p in dpgo.glob('Robot *.tum') if p.stat().st_size > 0])
        if not tums:
            continue
        tum = tums[-1]
        try:
            rid = int(tum.stem.split()[-1].split('_')[0])
        except ValueError:
            continue
        ts, pos, _ = read_tum_trajectory(str(tum))
        if len(pos) == 0:
            continue
        result[rdir.name] = (rid, np.array(ts), np.array(pos))
    return result  # {robot_name: (rid, timestamps, positions)}


def _load_loop_lines(ns_as_dir, raw, R, t, s):
    """Return list of aligned (pt1, pt2) pairs for inter-robot loop closures."""
    # Build maps: rid -> (timestamps, positions) and rid -> kf_map
    rid_to_ts_pos = {}
    rid_to_kf = {}
    rid_to_rname = {}
    for rname, (rid, ts_arr, pos_arr) in raw.items():
        rid_to_ts_pos[rid] = (ts_arr, pos_arr)
        rid_to_rname[rid] = rname
        kf_path = ns_as_dir / rname / 'distributed' / 'kimera_distributed_keyframes.csv'
        if kf_path.exists():
            rid_to_kf[rid] = load_keyframes_csv(str(kf_path))

    if not rid_to_kf:
        return []

    # Collect and deduplicate inter-robot loops from all robot dirs
    all_loops = []
    seen = set()
    for rname in sorted(rid_to_rname.values()):
        lc_path = ns_as_dir / rname / 'distributed' / 'loop_closures.csv'
        if not lc_path.exists():
            continue
        for lc in load_loop_closures_csv(str(lc_path)):
            r1, p1, r2, p2 = lc['robot1'], lc['pose1'], lc['robot2'], lc['pose2']
            if r1 == r2:
                continue
            key = (min(r1, r2), min(p1, p2), max(r1, r2), max(p1, p2))
            if key not in seen:
                seen.add(key)
                all_loops.append(lc)

    lines = []
    for lc in all_loops:
        r1, p1, r2, p2 = lc['robot1'], lc['pose1'], lc['robot2'], lc['pose2']
        kf1 = rid_to_kf.get(r1, {})
        kf2 = rid_to_kf.get(r2, {})
        ts1 = kf1.get(p1)
        ts2 = kf2.get(p2)
        if ts1 is None or ts2 is None:
            continue
        tp1 = rid_to_ts_pos.get(r1)
        tp2 = rid_to_ts_pos.get(r2)
        if tp1 is None or tp2 is None:
            continue
        raw_pt1 = find_tum_position(ts1, tp1[0], tp1[1])
        raw_pt2 = find_tum_position(ts2, tp2[0], tp2[1])
        if raw_pt1 is None or raw_pt2 is None:
            continue
        pt1 = apply_alignment(np.array([raw_pt1]), R, t, s)[0]
        pt2 = apply_alignment(np.array([raw_pt2]), R, t, s)[0]
        lines.append((pt1, pt2))
    return lines


def load_dataset(ns_as_dir, gt_dir):
    """Returns (aligned_positions, gt_positions, loop_lines).
    aligned_positions and gt_positions are {robot_name: (N,3)}.
    loop_lines is a list of (pt1, pt2) aligned position pairs."""
    raw = _load_tum_positions(ns_as_dir)
    robot_names = list(raw.keys())
    gt_full = load_gt_trajectories_by_name(gt_dir, robot_names)
    gt = {name: pos for name, (_, pos, _) in gt_full.items()}

    # Find alignment: prefer evo_ape.zip at ns_as_dir, then lm_optimized/
    R, t, s = np.eye(3), np.zeros(3), 1.0
    for zip_path in [ns_as_dir / 'evo_ape.zip',
                     ns_as_dir / 'lm_optimized' / 'evo_ape.zip']:
        if zip_path.exists():
            R, t, s = load_alignment_from_evo_zip(str(zip_path))
            break
    else:
        # Umeyama fallback
        src_pts, dst_pts = [], []
        for rname, (_, _, pos) in raw.items():
            if rname not in gt:
                continue
            g = gt[rname]
            n = min(len(pos), len(g))
            if n < 3:
                continue
            idx = np.linspace(0, n - 1, min(n, 500), dtype=int)
            src_pts.append(pos[idx])
            dst_pts.append(g[idx])
        if src_pts:
            R, t, s = umeyama(np.vstack(src_pts), np.vstack(dst_pts))

    aligned = {rname: apply_alignment(pos, R, t, s)
               for rname, (_, _, pos) in raw.items()}
    loop_lines = _load_loop_lines(ns_as_dir, raw, R, t, s)
    return aligned, gt, loop_lines


def _draw_panel(ax, aligned, gt, label, legend_handles, robot_order, loop_lines=None):
    # GT
    gt_plotted = False
    for rname in robot_order:
        pos = gt.get(rname)
        if pos is None:
            continue
        ax.plot(pos[:, 0], pos[:, 1], color='gray', lw=0.7, alpha=0.5,
                ls='--', label='GT' if not gt_plotted else None)
        gt_plotted = True

    # Trajectories
    for i, rname in enumerate(robot_order):
        pos = aligned.get(rname)
        if pos is None:
            continue
        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        robot_label = f'R{i}'
        ax.plot(pos[:, 0], pos[:, 1], lw=1.2, color=color,
                label=robot_label if robot_label not in legend_handles else None)
        legend_handles[robot_label] = color

    # Loop closures
    if loop_lines:
        for pt1, pt2 in loop_lines:
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                    color='#CC2222', lw=1.2, alpha=0.85, zorder=10)
            ax.plot(pt1[0], pt1[1], 'o', color='#CC2222', ms=6.0,
                    mew=1.0, mfc='none', zorder=11)
            ax.plot(pt2[0], pt2[1], 'o', color='#CC2222', ms=6.0,
                    mew=1.0, mfc='none', zorder=11)
        legend_handles['__loops__'] = '#CC2222'

    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3, linewidth=0.3)
    ax.text(0.03, 0.03, label, transform=ax.transAxes,
            fontsize=7, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75, ec='none'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    print('Loading datasets...')
    loaded = []
    for ds_name, label, ns_as_dir, gt_dir in DATASETS:
        print(f'  {ds_name}...')
        aligned, gt, loop_lines = load_dataset(ns_as_dir, gt_dir)
        robot_order = sorted(aligned.keys())
        print(f'    robots: {robot_order}, loops: {len(loop_lines)}')
        loaded.append((label, aligned, gt, robot_order, loop_lines))

    plt.rcParams.update({
        **IEEE_RC,
        'figure.figsize': (7.16, 2.0),
        'axes.labelsize': 7,
        'legend.fontsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
    })

    n = len(loaded)
    fig, axes = plt.subplots(1, n, squeeze=False)
    axes = axes[0]

    legend_handles = {}  # robot_label -> color

    for col, (label, aligned, gt, robot_order, loop_lines) in enumerate(loaded):
        _draw_panel(axes[col], aligned, gt, label, legend_handles, robot_order, loop_lines)

    # Axis labels only on leftmost / bottom
    for col, ax in enumerate(axes):
        ax.set_xlabel('x (m)', fontsize=7)
        if col == 0:
            ax.set_ylabel('y (m)', fontsize=7)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', length=0)

    # Build legend: GT + robots + loops
    all_handles, all_labels = [], []
    all_handles.append(plt.Line2D([], [], color='gray', ls='--', lw=0.7, label='GT'))
    all_labels.append('GT')
    for rl, color in sorted(legend_handles.items()):
        if rl == '__loops__':
            continue
        all_handles.append(plt.Line2D([], [], color=color, lw=1.2))
        all_labels.append(rl)
    if '__loops__' in legend_handles:
        all_handles.append(plt.Line2D([], [], color='#CC2222', lw=0.8, alpha=0.7))
        all_labels.append('Loops')

    fig.legend(all_handles, all_labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.12), ncol=len(all_handles),
               framealpha=0.9, edgecolor='none',
               handlelength=1.2, handletextpad=0.4, columnspacing=1.0,
               fontsize=7)

    plt.tight_layout(pad=0.3, w_pad=0.3)

    out = BASE / args.output if args.output else BASE / 'multi_dataset_traj'
    save_fig(fig, out)
    plt.close(fig)
    print(f'Saved → {out}.pdf / .png')


if __name__ == '__main__':
    main()
