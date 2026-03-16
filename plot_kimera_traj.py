#!/usr/bin/env python3
"""
Plot aligned trajectories + GT + loop closures for Kimera-Multi.

Usage:
    python3 plot_kimera_traj.py <variant_dir> [--gt_dir ground_truth/campus]
    python3 plot_kimera_traj.py baselines/campus/Kimera-Multi
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
from utils.io import load_alignment_from_evo_zip, load_gt_trajectory
from utils.plot import IEEE_RC, ROBOT_COLORS, save_fig, apply_alignment


def load_poses_csv(csv_path: Path) -> tuple[dict[int, tuple], dict[int, float]]:
    """Load poses indexed by pose_index.
    Returns (poses {idx: (x,y,z)}, timestamps {idx: t_sec}).
    timestamps dict is empty if no 'ns' column."""
    poses: dict[int, tuple] = {}
    timestamps: dict[int, float] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                idx = int(row['pose_index'])
                poses[idx] = (float(row['tx']), float(row['ty']), float(row['tz']))
                if 'ns' in row:
                    timestamps[idx] = float(row['ns']) * 1e-9
            except (KeyError, ValueError):
                continue
    return poses, timestamps


def find_latest_poses_csv(robot_dist_dir: Path) -> Path | None:
    """Return the highest-numbered kimera_distributed_poses_*.csv, or fall back
    to trajectory_optimized.csv."""
    candidates = sorted(robot_dist_dir.glob('kimera_distributed_poses_*.csv'),
                        key=lambda p: int(p.stem.split('_')[-1]))
    if candidates:
        return candidates[-1]
    fallback = robot_dist_dir / 'trajectory_optimized.csv'
    return fallback if fallback.exists() else None


def load_loop_closures(csv_path: Path) -> list[dict]:
    loops = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                loops.append({
                    'r1': int(row['robot1']), 'p1': int(row['pose1']),
                    'r2': int(row['robot2']), 'p2': int(row['pose2']),
                })
            except (KeyError, ValueError):
                continue
    return loops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('variant_dir')
    ap.add_argument('--gt_dir', default=None)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    vdir = Path(args.variant_dir)
    if not vdir.is_absolute():
        vdir = Path(__file__).parent / vdir

    # Load robot names yaml
    robot_id_to_name: dict[int, str] = {}
    yaml_path = vdir / 'robot_names.yaml'
    if yaml_path.exists():
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        for key, name in data.items():
            m = re.match(r'robot(\d+)_name', key)
            if m:
                robot_id_to_name[int(m.group(1))] = name
    # Fallback: discover robot dirs
    if not robot_id_to_name:
        for d in sorted(vdir.iterdir()):
            if (d / 'distributed' / 'trajectory_optimized.csv').exists():
                robot_id_to_name[len(robot_id_to_name)] = d.name

    robot_ids = sorted(robot_id_to_name)

    # Load alignment from evo_ape.zip (preferred), fall back to Umeyama
    evo_zip = vdir / 'evo_ape.zip'
    R_align, t_align, s_align = np.eye(3), np.zeros(3), 1.0
    if evo_zip.exists():
        R_align, t_align, s_align = load_alignment_from_evo_zip(str(evo_zip))
        print(f"Loaded alignment from {evo_zip.name}  scale={s_align:.4f}")
    else:
        print("No evo_ape.zip found — will use Umeyama alignment")

    # Load GT dir
    if args.gt_dir:
        gt_dir = Path(args.gt_dir)
    else:
        exp_name = vdir.parent.name
        gt_dir = Path(__file__).parent / 'ground_truth' / exp_name
    print(f"GT dir: {gt_dir}")

    # Load GT trajectories
    gt_by_rid: dict[int, np.ndarray] = {}
    if gt_dir.exists():
        for rid in robot_ids:
            rname = robot_id_to_name[rid]
            for ext in ('.csv', '.txt'):
                p = gt_dir / (rname + ext)
                if p.exists():
                    try:
                        _, pos, _ = load_gt_trajectory(str(p))
                        if len(pos):
                            gt_by_rid[rid] = np.array(pos)
                    except Exception:
                        pass
                    break
        print(f"GT loaded for robots: {sorted(gt_by_rid)}")

    # Load trajectories from latest kimera_distributed_poses_*.csv
    robot_poses: dict[int, dict] = {}      # robot_id -> {pose_idx: (x,y,z)}
    robot_ts:    dict[int, dict] = {}      # robot_id -> {pose_idx: t_sec}
    for rid in robot_ids:
        rname = robot_id_to_name[rid]
        dist_dir = vdir / rname / 'distributed'
        poses_csv = find_latest_poses_csv(dist_dir)
        if poses_csv is None:
            print(f"  Warning: no poses CSV for {rname}")
            continue
        poses, ts = load_poses_csv(poses_csv)
        robot_poses[rid] = poses
        robot_ts[rid]    = ts
        print(f"  {rname}: {len(poses)} poses from {poses_csv.name}")

    # Collect all loop closures across robots (deduplicated by pair)
    all_loops: list[dict] = []
    seen_pairs: set[tuple] = set()
    for rid in robot_ids:
        rname = robot_id_to_name[rid]
        lc_path = vdir / rname / 'distributed' / 'loop_closures.csv'
        if not lc_path.exists():
            continue
        for lc in load_loop_closures(lc_path):
            key = (min(lc['r1'], lc['r2']), min(lc['p1'], lc['p2']),
                   max(lc['r1'], lc['r2']), max(lc['p1'], lc['p2']))
            if key not in seen_pairs:
                seen_pairs.add(key)
                if lc['r1'] != lc['r2']:  # inter-robot only
                    all_loops.append(lc)
    print(f"Inter-robot loop closures: {len(all_loops)}")

    # If no evo_ape.zip, compute Umeyama alignment from timestamp-matched pairs
    if not evo_zip.exists() and gt_by_rid:
        def _umeyama(src: np.ndarray, dst: np.ndarray):
            mu_s, mu_d = src.mean(0), dst.mean(0)
            sc, dc = src - mu_s, dst - mu_d
            n = src.shape[0]
            sigma2 = (sc ** 2).sum() / n
            H = (sc.T @ dc) / n
            U, D, Vt = np.linalg.svd(H)
            det_sign = np.linalg.det(Vt.T @ U.T)
            S = np.diag([1.0, 1.0, det_sign])
            R_u = Vt.T @ S @ U.T
            s_u = (D * np.array([1.0, 1.0, det_sign])).sum() / sigma2 if sigma2 > 0 else 1.0
            return R_u, mu_d - s_u * R_u @ mu_s, float(s_u)

        src_pts, dst_pts = [], []
        for rid in robot_ids:
            if rid not in robot_poses or rid not in gt_by_rid:
                continue
            ts_map = robot_ts.get(rid, {})
            rname = robot_id_to_name[rid]
            gt_raw = None
            for ext in ('.csv', '.txt'):
                p = gt_dir / (rname + ext)
                if p.exists():
                    gt_ts_arr, gt_pos_arr, _ = load_gt_trajectory(str(p))
                    gt_raw = (gt_ts_arr, gt_pos_arr)
                    break
            if gt_raw is None:
                continue
            gt_ts_arr, gt_pos_arr = gt_raw
            if ts_map:
                idxs = sorted(ts_map)
                est_t = np.array([ts_map[i] for i in idxs])
                est_p = np.array([robot_poses[rid][i] for i in idxs])
                matched_src, matched_dst = [], []
                for t, ep in zip(est_t, est_p):
                    j = int(np.argmin(np.abs(gt_ts_arr - t)))
                    if abs(gt_ts_arr[j] - t) < 2.0:
                        matched_src.append(ep)
                        matched_dst.append(gt_pos_arr[j])
                if len(matched_src) >= 3:
                    idx = np.linspace(0, len(matched_src)-1, min(len(matched_src), 500), dtype=int)
                    src_pts.append(np.array(matched_src)[idx])
                    dst_pts.append(np.array(matched_dst)[idx])
            else:
                idxs = sorted(robot_poses[rid])
                est = np.array([robot_poses[rid][i] for i in idxs])
                n = min(len(est), len(gt_pos_arr))
                if n < 3:
                    continue
                idx = np.linspace(0, n-1, min(n, 500), dtype=int)
                src_pts.append(est[idx])
                dst_pts.append(gt_pos_arr[idx])
        if src_pts:
            R_align, t_align, s_align = _umeyama(np.vstack(src_pts), np.vstack(dst_pts))
            print(f"Umeyama alignment  scale={s_align:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        **IEEE_RC,
        'figure.figsize': (3.5, 2.2),
        'legend.fontsize': 6,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
    })
    fig, ax = plt.subplots()

    def _align(pts: np.ndarray) -> np.ndarray:
        return apply_alignment(pts, R_align, t_align, s_align)

    def _pose_pt(rid: int, pidx: int) -> np.ndarray | None:
        p = robot_poses.get(rid, {}).get(pidx)
        return _align(np.array([p]))[0] if p is not None else None

    # GT
    gt_plotted = False
    for rid in robot_ids:
        gt_pos = gt_by_rid.get(rid)
        if gt_pos is not None:
            ax.plot(gt_pos[:, 0], gt_pos[:, 1],
                    color='gray', lw=0.5, alpha=0.5, ls='--',
                    label='GT' if not gt_plotted else None)
            gt_plotted = True

    # Estimated trajectories
    for rid in robot_ids:
        poses = robot_poses.get(rid)
        if not poses:
            continue
        idxs = sorted(poses)
        pts = _align(np.array([poses[i] for i in idxs]))
        color = ROBOT_COLORS[rid % len(ROBOT_COLORS)]
        ax.plot(pts[:, 0], pts[:, 1], lw=1.0, color=color, label=f'R{rid}')

    # Loop closure lines
    lc_added = False
    for lc in all_loops:
        r1, p1, r2, p2 = lc['r1'], lc['p1'], lc['r2'], lc['p2']
        pt1 = _pose_pt(r1, p1)
        pt2 = _pose_pt(r2, p2)
        if pt1 is None or pt2 is None:
            continue
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                color='#CC2222', lw=1.5, alpha=0.8, zorder=10,
                label='Loop closure' if not lc_added else None)
        lc_added = True

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linewidth=0.3)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, 0.0), ncol=len(handles),
               framealpha=0.9, edgecolor='none',
               handlelength=1.0, handletextpad=0.3, columnspacing=0.8)
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(bottom=0.18)

    out = Path(args.output) if args.output else vdir / 'trajectories_aligned'
    save_fig(fig, out)
    plt.close(fig)
    print(f"Saved → {out}.pdf / .png")


if __name__ == '__main__':
    main()
