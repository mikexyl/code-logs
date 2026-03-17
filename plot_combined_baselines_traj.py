#!/usr/bin/env python3
"""
2×2 combined trajectory figure:
  col 0: Ours (CBS+ / Centralized GNC-GM)
  col 1: Baselines (DGS / ASAPP / Geodesic-MESA)
  row 0: campus/ns-as
  row 1: a5678/ns-as

Axes shared within each row. Single legend at bottom.

Usage:
    python3 plot_combined_baselines_traj.py
    python3 plot_combined_baselines_traj.py --output combined_baselines_traj
"""

import argparse
import sys
from pathlib import Path

import cbor2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.io import read_tum_trajectory, load_gt_trajectory, load_alignment_from_evo_zip
from utils.plot import IEEE_RC, save_fig, apply_alignment

BASE = Path(__file__).parent

# ── Style ────────────────────────────────────────────────────────────────────
METHOD_COLORS = {
    'GT':                '#444444',
    'CBS+':              '#377EB8',
    'Centralized GNC-GM':'#4DAF4A',
    'DGS':               '#E41A1C',
    'ASAPP':             '#FF7F00',
    'Geodesic-MESA':     '#984EA3',
}
METHOD_STYLES = {
    'GT':                ('--', 0.7, 0.5),
    'CBS+':              ('-',  1.2, 0.95),
    'Centralized GNC-GM':('-',  0.8, 0.7),
    'DGS':               ('-',  0.9, 0.85),
    'ASAPP':             ('--', 0.9, 0.85),
    'Geodesic-MESA':     (':',  0.9, 0.85),
}

OURS_METHODS      = ['CBS+', 'Centralized GNC-GM']
BASELINE_METHODS  = ['DGS', 'ASAPP', 'Geodesic-MESA']
LEGEND_ORDER      = ['GT', 'CBS+', 'Centralized GNC-GM', 'DGS', 'ASAPP', 'Geodesic-MESA']


# ── Data loading (adapted from plot_all_baselines_traj.py) ───────────────────

def _load_jrr(jrr_path):
    with open(jrr_path, 'rb') as f:
        d = cbor2.load(f)
    result = {}
    for rc, poses in d.get('solutions', {}).items():
        own = []
        for p in poses:
            key = p['key']
            if chr(key >> 56) != rc:
                continue
            pose_idx = key & 0x00FFFFFFFFFFFFFF
            t = np.array(p['translation'], dtype=float)
            rot = p['rotation']
            q = np.array([rot[1], rot[2], rot[3], rot[0]], dtype=float)
            own.append((pose_idx, t, q))
        own.sort(key=lambda x: x[0])
        result[rc] = own
    return result


def _find_latest_jrr(method_dir):
    cands = sorted(method_dir.glob('**/final_results.jrr.cbor'))
    if cands:
        return cands[-1]
    cands = sorted(method_dir.glob('**/iterations/*.jrr.cbor'))
    return cands[-1] if cands else None


def _jrr_positions(robot_map, jrr_poses):
    out = {}
    for rc in sorted(robot_map):
        if rc not in jrr_poses:
            continue
        ts_arr, _, _ = read_tum_trajectory(str(robot_map[rc]['tum']))
        n = len(ts_arr)
        pts = [t for idx, t, _ in jrr_poses[rc] if idx < n]
        if pts:
            out[rc] = np.array(pts)
    return out


def _tum_positions(tum_list):
    out = {}
    for rc, tum_path in tum_list:
        ts, pos, _ = read_tum_trajectory(str(tum_path))
        if not len(pos):
            continue
        ts, pos = np.array(ts), np.array(pos)
        order = np.argsort(ts)
        ts, pos = ts[order], pos[order]
        mask = np.abs(ts - np.median(ts)) < 1e7
        out[rc] = pos[mask]
    return out


def _umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    n = src.shape[0]
    sigma2 = (sc**2).sum() / n
    H = sc.T @ dc / n
    U, D, Vt = np.linalg.svd(H)
    det_sign = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1., 1., det_sign])
    R = Vt.T @ S @ U.T
    s = (D * np.array([1., 1., det_sign])).sum() / sigma2 if sigma2 > 0 else 1.
    return R, mu_d - s * R @ mu_s, float(s)


def _align_to_gt(positions, gt):
    src_pts, dst_pts = [], []
    for rc in sorted(positions):
        if rc not in gt:
            continue
        est, g = positions[rc], gt[rc]
        n = min(len(est), len(g))
        if n < 3:
            continue
        idx = np.linspace(0, n-1, min(n, 500), dtype=int)
        src_pts.append(est[idx])
        dst_pts.append(g[idx])
    if not src_pts:
        return positions
    R, t, s = _umeyama(np.vstack(src_pts), np.vstack(dst_pts))
    return {rc: apply_alignment(p, R, t, s) for rc, p in positions.items()}


def _build_robot_map(variant_dir):
    robot_map = {}
    for g2o in sorted(variant_dir.glob('*/dpgo/bpsam_robot_*.g2o')):
        rid = int(g2o.stem.split('_')[-1])
        rc = chr(ord('a') + rid)
        tum = g2o.parent / f'Robot {rid}.tum'
        if not tum.exists():
            cands = sorted(g2o.parent.glob('Robot *.tum'))
            tum = cands[0] if cands else None
        if tum:
            robot_map[rc] = {'tum': tum, 'robot_dir': g2o.parent.parent}
    return robot_map


def load_experiment(variant_dir: Path, gt_dir: Path):
    """
    Returns {method_name: {robot_char: (N,3) aligned positions}}
    and gt_by_robot {robot_char: (N,3)}.
    """
    robot_map = _build_robot_map(variant_dir)
    mesa_dir  = variant_dir / 'mesa_baselines'

    # GT
    gt_by_robot = {}
    for rc, info in robot_map.items():
        dir_name = info['robot_dir'].name
        for ext in ['.csv', '.txt']:
            p = gt_dir / (dir_name + ext)
            if p.exists():
                _, pos, _ = load_gt_trajectory(str(p))
                if len(pos):
                    gt_by_robot[rc] = np.array(pos)
                break

    methods = {}

    def _load_jrr_method(name, dir_name, zip_name):
        jrr = _find_latest_jrr(mesa_dir / dir_name)
        if not jrr:
            return
        pos = _jrr_positions(robot_map, _load_jrr(jrr))
        zp = mesa_dir / zip_name
        if zp.exists():
            R, t, s = load_alignment_from_evo_zip(str(zp))
            methods[name] = {rc: apply_alignment(p, R, t, s) for rc, p in pos.items()}
        else:
            methods[name] = _align_to_gt(pos, gt_by_robot)

    _load_jrr_method('DGS',           'dgs',           'dgs_evo_ape.zip')
    _load_jrr_method('ASAPP',         'asapp',         'asapp_evo_ape.zip')
    _load_jrr_method('Geodesic-MESA', 'geodesic-mesa', 'geodesic_mesa_evo_ape.zip')

    # CBS+
    cbs_plus_dir = variant_dir / 'cbs_plus'
    if cbs_plus_dir.exists():
        tum_list = []
        for robot_sub in sorted(cbs_plus_dir.iterdir()):
            dpgo = robot_sub / 'dpgo'
            if not dpgo.is_dir():
                continue
            tums = sorted([p for p in dpgo.glob('Robot*.tum') if p.stat().st_size > 0],
                          key=lambda p: int(p.stem.split('_')[-1]) if '_' in p.stem else 0)
            if not tums:
                continue
            last = tums[-1]
            rid = int(last.stem.split('_')[0].split(' ')[-1])
            tum_list.append((chr(ord('a') + rid), last))
        pos = _tum_positions(tum_list)
        zp = cbs_plus_dir / 'cbs_plus_evo_ape.zip'
        if zp.exists():
            R, t, s = load_alignment_from_evo_zip(str(zp))
            methods['CBS+'] = {rc: apply_alignment(p, R, t, s) for rc, p in pos.items()}
        else:
            methods['CBS+'] = _align_to_gt(pos, gt_by_robot)

    # Centralized GNC-GM
    lm_dir = variant_dir / 'lm_optimized'
    if lm_dir.exists():
        tum_list = []
        for robot_sub in sorted(lm_dir.iterdir()):
            dpgo = robot_sub / 'dpgo'
            if not dpgo.is_dir():
                continue
            tums = sorted([p for p in dpgo.glob('Robot *.tum') if p.stat().st_size > 0])
            if not tums:
                continue
            last = tums[-1]
            rid = int(last.stem.split(' ')[-1])
            tum_list.append((chr(ord('a') + rid), last))
        pos = _tum_positions(tum_list)
        zp = lm_dir / 'evo_ape.zip'
        if zp.exists():
            R, t, s = load_alignment_from_evo_zip(str(zp))
            methods['Centralized GNC-GM'] = {rc: apply_alignment(p, R, t, s) for rc, p in pos.items()}
        else:
            methods['Centralized GNC-GM'] = _align_to_gt(pos, gt_by_robot)

    return methods, gt_by_robot


# ── Plotting ─────────────────────────────────────────────────────────────────

def _draw_panel(ax, methods, gt_by_robot, show_methods, panel_label, legend_handles):
    # GT
    gt_plotted = False
    for rc in sorted(gt_by_robot):
        pos = gt_by_robot[rc]
        lbl = 'GT' if not gt_plotted else None
        ls, lw, alpha = METHOD_STYLES['GT']
        ax.plot(pos[:, 0], pos[:, 1], color=METHOD_COLORS['GT'],
                ls=ls, lw=lw, alpha=alpha, label=lbl)
        gt_plotted = True

    # Methods
    for name in show_methods:
        if name not in methods:
            continue
        ls, lw, alpha = METHOD_STYLES[name]
        color = METHOD_COLORS[name]
        first = True
        for rc in sorted(methods[name]):
            pos = methods[name][rc]
            lbl = name if first else None
            ax.plot(pos[:, 0], pos[:, 1], color=color, ls=ls, lw=lw, alpha=alpha, label=lbl)
            first = False

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linewidth=0.3)

    # Panel label inside bottom-left
    ax.text(0.03, 0.03, panel_label, transform=ax.transAxes,
            fontsize=8, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75, ec='none'))

    # Collect legend handles (once, from first panel)
    if not legend_handles:
        for name in ['GT'] + show_methods:
            ls, lw, alpha = METHOD_STYLES[name]
            h = mlines.Line2D([], [], color=METHOD_COLORS[name], ls=ls, lw=lw,
                              label=name)
            legend_handles[name] = h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    datasets = [
        ('campus', BASE / 'campus' / 'ns-as',  BASE / 'ground_truth' / 'campus'),
        ('a5678',  BASE / 'a5678'  / 'ns-as',  BASE / 'ground_truth' / 'a5678'),
    ]

    print('Loading data...')
    loaded = []
    for exp_name, vdir, gt_dir in datasets:
        print(f'  {exp_name}...')
        methods, gt = load_experiment(vdir, gt_dir)
        print(f'    methods: {sorted(methods)}')
        loaded.append((exp_name, methods, gt))

    # ── Figure ───────────────────────────────────────────────────────────────
    plt.rcParams.update({
        **IEEE_RC,
        'figure.figsize': (7.0, 5.5),
        'legend.fontsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
    })

    fig, axes = plt.subplots(2, 2, squeeze=False)

    # Share axes within each row
    for row in range(2):
        axes[row, 1].sharex(axes[row, 0])
        axes[row, 1].sharey(axes[row, 0])

    legend_handles_ours = {}
    legend_handles_base = {}

    for row, (exp_name, methods, gt) in enumerate(loaded):
        _draw_panel(axes[row, 0], methods, gt, OURS_METHODS,
                    'CBS+', legend_handles_ours)
        _draw_panel(axes[row, 1], methods, gt, BASELINE_METHODS,
                    'Baselines', legend_handles_base)

        # Hide shared x-tick labels on top row
        if row == 0:
            plt.setp(axes[row, 0].get_xticklabels(), visible=False)
            plt.setp(axes[row, 1].get_xticklabels(), visible=False)

        # Row label (experiment name) on left y-axis
        axes[row, 0].set_ylabel(f'{exp_name}\ny (m)', fontsize=7)
        axes[row, 1].set_ylabel('')

    axes[1, 0].set_xlabel('x (m)')
    axes[1, 1].set_xlabel('x (m)')

    # Single legend combining ours + baselines in fixed order
    all_handles = []
    all_labels  = []
    combined = {**legend_handles_ours, **legend_handles_base}
    for name in LEGEND_ORDER:
        if name in combined:
            all_handles.append(combined[name])
            all_labels.append(name)

    fig.legend(all_handles, all_labels, loc='lower center',
               bbox_to_anchor=(0.5, 0.0), ncol=len(all_handles),
               framealpha=0.9, edgecolor='none',
               handlelength=1.5, handletextpad=0.4, columnspacing=1.2,
               fontsize=7)

    plt.tight_layout(pad=0.3, h_pad=0.0, w_pad=0.3)
    plt.subplots_adjust(bottom=0.08)

    out = BASE / args.output if args.output else BASE / 'combined_baselines_traj'
    save_fig(fig, out)
    plt.close(fig)
    print(f'Saved → {out}.pdf / .png')


if __name__ == '__main__':
    main()
