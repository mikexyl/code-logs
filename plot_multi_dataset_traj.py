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

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.io import read_tum_trajectory, load_gt_trajectory, load_alignment_from_evo_zip
from utils.plot import IEEE_RC, ROBOT_COLORS, save_fig, apply_alignment

BASE = Path(__file__).parent

DATASETS = [
    ('g123',  'GrAco\nLoop 1',  BASE / 'g123'  / 'ns-as', BASE / 'ground_truth' / 'g123'),
    ('g156',  'GrAco\nLoop 2',  BASE / 'g156'  / 'ns-as', BASE / 'ground_truth' / 'g156'),
    ('g2345', 'GrAco\nLoop 3',  BASE / 'g2345' / 'ns-as', BASE / 'ground_truth' / 'g2345'),
    ('a12',   'GrAco\nAerial',  BASE / 'a12'   / 'ns-as', BASE / 'ground_truth' / 'a12'),
    ('gate',  'Gate',           BASE / 'gate'  / 'ns-as', BASE / 'ground_truth' / 'gate'),
]


def _umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    n = src.shape[0]
    sigma2 = (sc ** 2).sum() / n
    H = sc.T @ dc / n
    U, D, Vt = np.linalg.svd(H)
    det_sign = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1., 1., det_sign])
    R = Vt.T @ S @ U.T
    s = (D * np.array([1., 1., det_sign])).sum() / sigma2 if sigma2 > 0 else 1.
    return R, mu_d - s * R @ mu_s, float(s)


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
        result[rdir.name] = (rid, np.array(pos))
    return result  # {robot_name: (rid, positions)}


def _load_gt(ns_as_dir, gt_dir):
    robot_names = [d.name for d in sorted(ns_as_dir.iterdir())
                   if d.is_dir() and (d / 'dpgo').is_dir()]
    gt = {}
    for rname in robot_names:
        for ext in ('.txt', '.csv'):
            p = gt_dir / (rname + ext)
            if p.exists():
                _, pos, _ = load_gt_trajectory(str(p))
                if len(pos):
                    gt[rname] = np.array(pos)
                break
    return gt


def load_dataset(ns_as_dir, gt_dir):
    """Returns (aligned_positions, gt_positions) both {robot_name: (N,3)}."""
    raw = _load_tum_positions(ns_as_dir)
    gt  = _load_gt(ns_as_dir, gt_dir)

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
        for rname, (_, pos) in raw.items():
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
            R, t, s = _umeyama(np.vstack(src_pts), np.vstack(dst_pts))

    aligned = {rname: apply_alignment(pos, R, t, s)
               for rname, (_, pos) in raw.items()}
    return aligned, gt


def _draw_panel(ax, aligned, gt, label, legend_handles, robot_order):
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
        aligned, gt = load_dataset(ns_as_dir, gt_dir)
        robot_order = sorted(aligned.keys())
        print(f'    robots: {robot_order}')
        loaded.append((label, aligned, gt, robot_order))

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

    for col, (label, aligned, gt, robot_order) in enumerate(loaded):
        _draw_panel(axes[col], aligned, gt, label, legend_handles, robot_order)

    # Axis labels only on leftmost / bottom
    for col, ax in enumerate(axes):
        ax.set_xlabel('x (m)', fontsize=7)
        if col == 0:
            ax.set_ylabel('y (m)', fontsize=7)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', length=0)

    # Build legend: GT + robots
    all_handles, all_labels = [], []
    all_handles.append(plt.Line2D([], [], color='gray', ls='--', lw=0.7, label='GT'))
    all_labels.append('GT')
    for rl, color in sorted(legend_handles.items()):
        all_handles.append(plt.Line2D([], [], color=color, lw=1.2))
        all_labels.append(rl)

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
