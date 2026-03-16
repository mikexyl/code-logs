#!/usr/bin/env python3
"""
Plot aligned trajectories with loop closure lines colored by translation error
vs ground truth relative pose.

Usage:
    python3 plot_loop_errors_map.py <variant_dir> <gt_dir>
    python3 plot_loop_errors_map.py campus/ns-as ground_truth/campus

Output:
    <variant_dir>/loop_errors_map.pdf/png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from utils.io import (read_tum_trajectory, load_alignment_from_evo_zip,
                      load_keyframes_csv, load_loop_closures_csv,
                      load_gt_trajectory)
from utils.plot import IEEE_RC, ROBOT_COLORS, save_fig, apply_alignment, find_tum_position


# ---------------------------------------------------------------------------
# Geometry helpers (mirrored from evaluate_loops_recall.py)
# ---------------------------------------------------------------------------

def _nearest_pose(ts_s, timestamps, positions, rotations, max_gap_s=2.5):
    idx = int(np.argmin(np.abs(timestamps - ts_s)))
    if abs(float(timestamps[idx]) - ts_s) > max_gap_s:
        return None
    return positions[idx], rotations[idx]


def _quat_xyzw_to_rot(q):
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def _rot_angle_deg(R):
    cos_val = float(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_val)))


# ---------------------------------------------------------------------------
# Robot discovery
# ---------------------------------------------------------------------------

def discover_robots(variant_dir: Path) -> dict[int, str]:
    """Return {robot_id: robot_dir_name} by scanning dpgo/Robot N.tum files."""
    id_to_name: dict[int, str] = {}
    for robot_dir in sorted(variant_dir.iterdir()):
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


# ---------------------------------------------------------------------------
# Load GT poses
# ---------------------------------------------------------------------------

def load_gt_poses(gt_dir: Path, robot_names: list[str]) -> dict[str, tuple]:
    """Return {robot_name: (timestamps_s, positions, rotations_xyzw)}."""
    result: dict[str, tuple] = {}
    for name in robot_names:
        for ext in ('.csv', '.txt'):
            p = gt_dir / (name + ext)
            if p.exists():
                ts_ns, pos, rots = load_gt_trajectory(str(p))
                result[name] = (ts_ns / 1e9, pos, rots)
                break
    return result


# ---------------------------------------------------------------------------
# Compute per-loop error and resolved world-frame endpoints
# ---------------------------------------------------------------------------

def collect_loop_data(
    variant_dir: Path,
    id_to_name: dict[int, str],
    gt_poses: dict[str, tuple],
    rotation, translation, scale,
    max_gap_s: float = 2.5,
    trans_abs: float = 2.0,
    trans_rel: float = 0.10,
    rot_thr: float = 40.0,
):
    """For each evaluable inter-robot loop, return a dict with:
        p1, p2  : aligned world-frame XY endpoints (np.ndarray shape (3,))
        trans_err: translation error vs GT (m)
        rot_err  : rotation error vs GT (deg)
        is_outlier: bool
    """
    # Build robot_id → keyframe_id → timestamp_s
    kf_maps: dict[int, dict[int, float]] = {}
    for rid, rname in id_to_name.items():
        kf_path = variant_dir / rname / 'distributed' / 'kimera_distributed_keyframes.csv'
        if kf_path.exists():
            kf_maps[rid] = load_keyframes_csv(str(kf_path))

    # Build robot_id → (timestamps_s, raw_positions) from dpgo TUM
    robot_trajs: dict[int, tuple] = {}
    for rid, rname in id_to_name.items():
        dpgo = variant_dir / rname / 'dpgo'
        tums = sorted(dpgo.glob('Robot *.tum')) if dpgo.exists() else []
        if tums:
            ts, pos, _ = read_tum_trajectory(str(tums[0]))
            robot_trajs[rid] = (np.array(ts), np.array(pos))

    seen: set[frozenset] = set()
    records = []

    for rid, rname in id_to_name.items():
        lc_path = variant_dir / rname / 'distributed' / 'loop_closures.csv'
        if not lc_path.exists():
            continue
        for lc in load_loop_closures_csv(str(lc_path)):
            r1, p1i = lc['robot1'], lc['pose1']
            r2, p2i = lc['robot2'], lc['pose2']
            is_intra = (r1 == r2)
            key = frozenset([(r1, p1i), (r2, p2i)])
            if key in seen:
                continue
            seen.add(key)

            # Need pose estimate for error computation
            has_pose = all(lc.get(k) is not None for k in ('tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'))

            # Resolve timestamps via keyframe maps
            if r1 not in kf_maps or r2 not in kf_maps:
                continue
            t1 = kf_maps[r1].get(p1i)
            t2 = kf_maps[r2].get(p2i)
            if t1 is None or t2 is None:
                continue

            n1 = id_to_name.get(r1)
            n2 = id_to_name.get(r2)
            if n1 is None or n2 is None:
                continue

            # Resolve world-frame endpoints from aligned TUM
            if r1 not in robot_trajs or r2 not in robot_trajs:
                continue
            raw1 = find_tum_position(t1, robot_trajs[r1][0], robot_trajs[r1][1])
            raw2 = find_tum_position(t2, robot_trajs[r2][0], robot_trajs[r2][1])
            if raw1 is None or raw2 is None:
                continue
            aligned1 = apply_alignment(raw1.reshape(1, 3), rotation, translation, scale)[0]
            aligned2 = apply_alignment(raw2.reshape(1, 3), rotation, translation, scale)[0]

            # Compute error vs GT
            trans_err = None
            rot_err = None
            is_outlier = None

            if has_pose and n1 in gt_poses and n2 in gt_poses:
                ts1_arr, pos1_arr, rot1_arr = gt_poses[n1]
                ts2_arr, pos2_arr, rot2_arr = gt_poses[n2]
                pose1 = _nearest_pose(t1, ts1_arr, pos1_arr, rot1_arr, max_gap_s)
                pose2 = _nearest_pose(t2, ts2_arr, pos2_arr, rot2_arr, max_gap_s)
                if pose1 is not None and pose2 is not None:
                    p_gt1, r_gt1 = pose1
                    p_gt2, r_gt2 = pose2
                    R1 = _quat_xyzw_to_rot(r_gt1)
                    R2 = _quat_xyzw_to_rot(r_gt2)
                    p_rel_gt = R1.T @ (p_gt2 - p_gt1)
                    R_rel_gt = R1.T @ R2
                    gt_dist = float(np.linalg.norm(p_rel_gt))
                    p_det = np.array([lc['tx'], lc['ty'], lc['tz']])
                    R_det = _quat_xyzw_to_rot(np.array([lc['qx'], lc['qy'], lc['qz'], lc['qw']]))
                    trans_err = float(np.linalg.norm(p_det - p_rel_gt))
                    rot_err = _rot_angle_deg(R_det.T @ R_rel_gt)
                    is_outlier = (trans_err > max(trans_abs, trans_rel * gt_dist)
                                  or rot_err > rot_thr)

            records.append({
                'p1': aligned1, 'p2': aligned2,
                'trans_err': trans_err, 'rot_err': rot_err,
                'is_outlier': is_outlier,
                'is_intra': is_intra,
            })

    return records


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(variant_dir: Path, gt_dir: Path, max_trans_err: float):
    evo_zip = variant_dir / 'evo_ape.zip'
    if not evo_zip.exists():
        print(f"Error: {evo_zip} not found")
        sys.exit(1)

    rotation, translation, scale = load_alignment_from_evo_zip(str(evo_zip))

    id_to_name = discover_robots(variant_dir)
    if not id_to_name:
        print("No robots found")
        sys.exit(1)
    print(f"Robots: {id_to_name}")

    gt_poses = load_gt_poses(gt_dir, list(id_to_name.values()))

    # Collect loop data
    records = collect_loop_data(variant_dir, id_to_name, gt_poses,
                                rotation, translation, scale)
    evaluable = [r for r in records if r['trans_err'] is not None]
    not_evaluable = [r for r in records if r['trans_err'] is None]
    print(f"Loops: {len(records)} total, {len(evaluable)} with error, "
          f"{len(not_evaluable)} without pose/GT")

    # Figure
    plt.rcParams.update({**IEEE_RC, 'figure.figsize': (4.5, 4.0)})
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    # Colormap: green (0) → orange (mid) → red (max) — all visible on white
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'GnOrRd', ['#1a9850', '#f46d43', '#a50026'])
    norm = mcolors.Normalize(vmin=0, vmax=max_trans_err)

    # Plot trajectories (ultra-light so loop lines dominate)
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(id_to_name))))
    for i, (rid, rname) in enumerate(sorted(id_to_name.items())):
        dpgo = variant_dir / rname / 'dpgo'
        tums = sorted(dpgo.glob('Robot *.tum')) if dpgo.exists() else []
        if not tums:
            continue
        ts, pos, _ = read_tum_trajectory(str(tums[0]))
        pos = np.array(pos)
        aligned = apply_alignment(pos, rotation, translation, scale)
        ax.plot(aligned[:, 0], aligned[:, 1], color=colors[i], linewidth=0.4,
                alpha=0.15, label=rname, zorder=2)

    # Plot GT trajectories (very faint)
    gt_plotted = False
    for rname, (ts_s, pos, _) in gt_poses.items():
        ax.plot(pos[:, 0], pos[:, 1], color='gray', linewidth=0.3,
                alpha=0.1, linestyle='--',
                label='GT' if not gt_plotted else None, zorder=1)
        gt_plotted = True

    # Plot loops without error estimate (gray, thin)
    for r in not_evaluable:
        p1, p2 = r['p1'], r['p2']
        ls = ':' if r['is_intra'] else '-'
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color='lightgray', linewidth=0.5, alpha=0.5, linestyle=ls, zorder=4)

    # Plot loops with error, colored by trans_err
    # inter-robot: solid, thicker; intra-robot: dashed, thinner
    for r in evaluable:
        p1, p2 = r['p1'], r['p2']
        color = cmap(norm(r['trans_err']))
        if r['is_intra']:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=color, linewidth=0.7, alpha=0.7, linestyle='--', zorder=5)
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=color, linewidth=1.2, alpha=0.85, linestyle='-', zorder=5)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Translation error (m)', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([0], [0], color='gray', linewidth=1.0, linestyle='-'),
                Line2D([0], [0], color='gray', linewidth=0.7, linestyle='--')]
    labels += ['inter-robot loop', 'intra-robot loop']
    ax.legend(handles, labels, loc='best', framealpha=0.8, fontsize=6, ncol=2)
    ax.grid(True, alpha=0.2, linewidth=0.3)
    plt.tight_layout(pad=0.5)

    save_fig(fig, variant_dir / 'loop_errors_map')
    plt.close(fig)
    print(f"Saved → {variant_dir}/loop_errors_map.pdf/png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('variant_dir', type=Path)
    parser.add_argument('gt_dir', type=Path)
    parser.add_argument('--max_err', type=float, default=20.0,
                        help='Colorbar max translation error in metres (default 20)')
    args = parser.parse_args()

    vdir = args.variant_dir.resolve()
    gdir = args.gt_dir.resolve()
    if not vdir.exists():
        print(f"Error: {vdir} does not exist"); sys.exit(1)
    if not gdir.exists():
        print(f"Error: {gdir} does not exist"); sys.exit(1)

    plot(vdir, gdir, args.max_err)


if __name__ == '__main__':
    main()
