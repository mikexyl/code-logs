#!/usr/bin/env python3
"""
Combined single-column trajectory figure: Kimera-Multi (top) + Ours/ns-as (bottom).
Both panels share axes, robot colors, and a single legend at the bottom.

Usage:
    python3 plot_combined_traj.py
    python3 plot_combined_traj.py --kimera_dir baselines/campus/Kimera-Multi \
                                  --ours_dir campus/ns-as \
                                  --gt_dir ground_truth/campus \
                                  --output campus/combined_traj
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.io import (read_tum_trajectory, load_gt_trajectory,
                      load_alignment_from_evo_zip, load_keyframes_csv,
                      load_loop_closures_csv)
from utils.plot import IEEE_RC, ROBOT_COLORS, save_fig, apply_alignment, find_tum_position


# Canonical physical-robot ordering for campus (matches Kimera-Multi robot_names.yaml)
DEFAULT_ROBOT_ORDER = ['acl_jackal', 'acl_jackal2', 'sparkal1', 'sparkal2', 'hathor', 'thoth']


# ---------------------------------------------------------------------------
# Kimera-Multi loading
# ---------------------------------------------------------------------------

def _load_poses_csv(csv_path: Path) -> dict[int, tuple]:
    poses = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                poses[int(row['pose_index'])] = (
                    float(row['tx']), float(row['ty']), float(row['tz']))
            except (KeyError, ValueError):
                continue
    return poses


def _find_latest_poses_csv(dist_dir: Path) -> Path | None:
    candidates = sorted(dist_dir.glob('kimera_distributed_poses_*.csv'),
                        key=lambda p: int(p.stem.split('_')[-1]))
    if candidates:
        return candidates[-1]
    fb = dist_dir / 'trajectory_optimized.csv'
    return fb if fb.exists() else None


def load_kimera_data(kimera_dir: Path, gt_dir: Path):
    """Returns (aligned_trajs, gt_trajs, loop_lines) keyed by robot_name."""
    # Robot name map from yaml
    robot_id_to_name: dict[int, str] = {}
    yaml_path = kimera_dir / 'robot_names.yaml'
    if yaml_path.exists():
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        for key, name in data.items():
            m = re.match(r'robot(\d+)_name', key)
            if m:
                robot_id_to_name[int(m.group(1))] = name

    R, t, s = load_alignment_from_evo_zip(str(kimera_dir / 'evo_ape.zip'))
    print(f"  Kimera-Multi alignment scale={s:.4f}")

    # Load poses
    raw_poses: dict[int, dict] = {}
    for rid, rname in robot_id_to_name.items():
        csv_path = _find_latest_poses_csv(kimera_dir / rname / 'distributed')
        if csv_path is None:
            continue
        raw_poses[rid] = _load_poses_csv(csv_path)

    # Aligned trajectories
    aligned_trajs: dict[str, np.ndarray] = {}
    for rid, rname in robot_id_to_name.items():
        poses = raw_poses.get(rid)
        if not poses:
            continue
        idxs = sorted(poses)
        pts = apply_alignment(np.array([poses[i] for i in idxs]), R, t, s)
        aligned_trajs[rname] = pts

    # GT
    gt_trajs: dict[str, np.ndarray] = {}
    for rname in robot_id_to_name.values():
        for ext in ('.csv', '.txt'):
            p = gt_dir / (rname + ext)
            if p.exists():
                _, pos, _ = load_gt_trajectory(str(p))
                if len(pos):
                    gt_trajs[rname] = np.array(pos)
                break

    # Loop closures
    seen: set = set()
    all_loops: list = []
    for rname in robot_id_to_name.values():
        lc_path = kimera_dir / rname / 'distributed' / 'loop_closures.csv'
        if not lc_path.exists():
            continue
        with open(lc_path) as f:
            for row in csv.DictReader(f):
                try:
                    r1, p1 = int(row['robot1']), int(row['pose1'])
                    r2, p2 = int(row['robot2']), int(row['pose2'])
                except (KeyError, ValueError):
                    continue
                key = (min(r1, r2), min(p1, p2), max(r1, r2), max(p1, p2))
                if key not in seen and r1 != r2:
                    seen.add(key)
                    all_loops.append((r1, p1, r2, p2))

    loop_lines: list = []
    for r1, p1, r2, p2 in all_loops:
        q1 = raw_poses.get(r1, {}).get(p1)
        q2 = raw_poses.get(r2, {}).get(p2)
        if q1 is None or q2 is None:
            continue
        loop_lines.append((
            apply_alignment(np.array([q1]), R, t, s)[0],
            apply_alignment(np.array([q2]), R, t, s)[0],
        ))

    print(f"  Kimera-Multi: {len(aligned_trajs)} robots, {len(loop_lines)} loop lines")
    return aligned_trajs, gt_trajs, loop_lines


# ---------------------------------------------------------------------------
# Ours (ns-as) loading
# ---------------------------------------------------------------------------

def load_ours_data(ours_dir: Path, gt_dir: Path):
    """Returns (aligned_trajs, gt_trajs, loop_lines) keyed by robot_name."""
    R, t, s = load_alignment_from_evo_zip(str(ours_dir / 'evo_ape.zip'))
    print(f"  Ours alignment scale={s:.4f}")

    robot_tum: dict[str, tuple] = {}     # robot_name -> (ts_arr, pos_arr)
    robot_tum_id: dict[str, int] = {}    # robot_name -> robot_id
    robot_kf: dict[int, dict] = {}       # robot_id -> {keyframe_id: ts_s}

    for rdir in sorted(ours_dir.iterdir()):
        if not rdir.is_dir():
            continue
        dpgo = rdir / 'dpgo'
        if not dpgo.is_dir():
            continue
        tums = [f for f in dpgo.glob('Robot *.tum') if f.stat().st_size > 0]
        if not tums:
            continue
        tum = tums[0]
        rid = int(tum.stem.split(' ')[-1])
        ts_arr, pos_arr, _ = read_tum_trajectory(str(tum))
        if len(pos_arr) == 0:
            continue
        robot_tum[rdir.name] = (np.array(ts_arr), np.array(pos_arr))
        robot_tum_id[rdir.name] = rid
        kf_path = rdir / 'distributed' / 'kimera_distributed_keyframes.csv'
        if kf_path.exists():
            robot_kf[rid] = load_keyframes_csv(str(kf_path))

    # Aligned trajectories
    aligned_trajs: dict[str, np.ndarray] = {}
    for rname, (_, pos_arr) in robot_tum.items():
        aligned_trajs[rname] = apply_alignment(pos_arr, R, t, s)

    # GT
    gt_trajs: dict[str, np.ndarray] = {}
    for rname in robot_tum:
        for ext in ('.csv', '.txt'):
            p = gt_dir / (rname + ext)
            if p.exists():
                _, pos, _ = load_gt_trajectory(str(p))
                if len(pos):
                    gt_trajs[rname] = np.array(pos)
                break

    # Loop closures
    rid_to_tum = {robot_tum_id[rn]: robot_tum[rn] for rn in robot_tum}
    seen: set = set()
    all_loops: list = []
    for rdir in sorted(ours_dir.iterdir()):
        if not rdir.is_dir():
            continue
        lc_path = rdir / 'distributed' / 'loop_closures.csv'
        if not lc_path.exists():
            continue
        for lc in load_loop_closures_csv(str(lc_path)):
            r1, p1, r2, p2 = lc['robot1'], lc['pose1'], lc['robot2'], lc['pose2']
            key = frozenset([(r1, p1), (r2, p2)])
            if key not in seen and r1 != r2:
                seen.add(key)
                all_loops.append((r1, p1, r2, p2))

    loop_lines: list = []
    n_unresolved = 0
    for r1, p1, r2, p2 in all_loops:
        if r1 not in robot_kf or r2 not in robot_kf:
            n_unresolved += 1
            continue
        ts1 = robot_kf[r1].get(p1)
        ts2 = robot_kf[r2].get(p2)
        if ts1 is None or ts2 is None:
            n_unresolved += 1
            continue
        tum1 = rid_to_tum.get(r1)
        tum2 = rid_to_tum.get(r2)
        if tum1 is None or tum2 is None:
            n_unresolved += 1
            continue
        raw1 = find_tum_position(ts1, tum1[0], tum1[1])
        raw2 = find_tum_position(ts2, tum2[0], tum2[1])
        if raw1 is None or raw2 is None:
            n_unresolved += 1
            continue
        loop_lines.append((
            apply_alignment(raw1.reshape(1, 3), R, t, s)[0],
            apply_alignment(raw2.reshape(1, 3), R, t, s)[0],
        ))

    print(f"  Ours: {len(aligned_trajs)} robots, {len(loop_lines)} drawn, "
          f"{n_unresolved} unresolved loop lines")
    return aligned_trajs, gt_trajs, loop_lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kimera_dir', default='baselines/campus/Kimera-Multi')
    ap.add_argument('--ours_dir',   default='campus/ns-as')
    ap.add_argument('--gt_dir',     default='ground_truth/campus')
    ap.add_argument('--output',     default=None)
    args = ap.parse_args()

    base = Path(__file__).parent

    def _abs(p): return Path(p) if Path(p).is_absolute() else base / p

    kimera_dir = _abs(args.kimera_dir)
    ours_dir   = _abs(args.ours_dir)
    gt_dir     = _abs(args.gt_dir)

    print("Loading Kimera-Multi...")
    km_trajs, km_gt, km_loops = load_kimera_data(kimera_dir, gt_dir)

    print("Loading Ours (ns-as)...")
    ours_trajs, ours_gt, ours_loops = load_ours_data(ours_dir, gt_dir)

    robot_order = DEFAULT_ROBOT_ORDER
    name_to_cidx = {name: i for i, name in enumerate(robot_order)}

    # ── Figure ────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        **IEEE_RC,
        'figure.figsize': (3.5, 5.5),
        'legend.fontsize': 6,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
    })
    fig, (ax_km, ax_ours) = plt.subplots(2, 1, sharex=True, sharey=True)

    def _draw_panel(ax, trajs, gt_trajs, loop_lines):
        # GT (gray dashed, one legend entry)
        gt_plotted = False
        for rname in robot_order:
            pos = gt_trajs.get(rname)
            if pos is not None:
                ax.plot(pos[:, 0], pos[:, 1], color='gray', lw=0.5,
                        alpha=0.5, ls='--', label='GT' if not gt_plotted else None)
                gt_plotted = True
        # Estimated trajectories
        for rname in robot_order:
            pts = trajs.get(rname)
            if pts is None:
                continue
            cidx = name_to_cidx.get(rname, 0)
            color = ROBOT_COLORS[cidx % len(ROBOT_COLORS)]
            label = f'R{robot_order.index(rname)}'
            ax.plot(pts[:, 0], pts[:, 1], lw=0.8, color=color, label=label)
        # Loop closures
        lc_added = False
        for pt1, pt2 in loop_lines:
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                    color='#CC2222', lw=0.8, alpha=0.7, zorder=10,
                    label='Loop closure' if not lc_added else None)
            lc_added = True
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linewidth=0.3)

    _draw_panel(ax_km,   km_trajs,   km_gt,   km_loops)
    _draw_panel(ax_ours, ours_trajs, ours_gt, ours_loops)

    ax_km.set_title('(a) Kimera-Multi', fontsize=7, loc='left', pad=2)
    ax_ours.set_title('(b) Ours', fontsize=7, loc='left', pad=2)
    ax_ours.set_xlabel('X (m)')
    for ax in (ax_km, ax_ours):
        ax.set_ylabel('Y (m)')

    # Single shared legend at bottom (collect from top panel which has all entries)
    handles, labels = ax_km.get_legend_handles_labels()
    # Ensure loop closure handle present (top panel always has loops)
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, 0.0), ncol=len(handles),
               framealpha=0.9, edgecolor='none',
               handlelength=1.0, handletextpad=0.3, columnspacing=0.8,
               fontsize=6)

    plt.tight_layout(pad=0.3, h_pad=0.4)
    plt.subplots_adjust(bottom=0.08)

    out = _abs(args.output) if args.output else ours_dir.parent / 'combined_traj'
    save_fig(fig, out)
    plt.close(fig)
    print(f"Saved → {out}.pdf / .png")


if __name__ == '__main__':
    main()
