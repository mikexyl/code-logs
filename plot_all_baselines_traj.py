#!/usr/bin/env python3
"""
Plot all baseline trajectories (DGS, ASAPP, geodesic-MESA, CBS+, centralized GNC-GM)
alongside ground truth in a single figure for campus/ns-as.

Usage:
    python3 plot_all_baselines_traj.py [variant_dir] [--gt_dir ground_truth/campus]
    python3 plot_all_baselines_traj.py campus/ns-as
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
from utils.plot import IEEE_RC, ROBOT_COLORS, save_fig, apply_alignment


# ---------------------------------------------------------------------------
# JRR loading (from evaluate_mesa_baselines.py)
# ---------------------------------------------------------------------------

def load_jrr(jrr_path: Path) -> dict:
    """Decode .jrr.cbor → {robot_char: [(pose_idx, t_xyz, q_xyzw), ...]}"""
    with open(jrr_path, 'rb') as f:
        d = cbor2.load(f)
    result = {}
    for robot_char, poses in d.get('solutions', {}).items():
        own = []
        for p in poses:
            key = p['key']
            if chr(key >> 56) != robot_char:
                continue
            pose_idx = key & 0x00FFFFFFFFFFFFFF
            t = np.array(p['translation'], dtype=float)
            rot = p['rotation']  # [w, x, y, z]
            q_xyzw = np.array([rot[1], rot[2], rot[3], rot[0]], dtype=float)
            own.append((pose_idx, t, q_xyzw))
        own.sort(key=lambda x: x[0])
        result[robot_char] = own
    return result


def find_latest_jrr(method_dir: Path) -> Path | None:
    candidates = sorted(method_dir.glob("**/final_results.jrr.cbor"))
    if candidates:
        return candidates[-1]
    # fallback: latest iteration snapshot
    candidates = sorted(method_dir.glob("**/iterations/*.jrr.cbor"))
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# Trajectory extractors
# ---------------------------------------------------------------------------

def jrr_to_positions(robot_map: dict, jrr_poses: dict) -> dict:
    """
    robot_map: {robot_char: {'tum': Path, 'robot_dir': Path}}
    Returns {robot_char: (N,3) estimated positions (unaligned)}
    """
    out = {}
    for rc in sorted(robot_map):
        if rc not in jrr_poses:
            continue
        ts_arr, _, _ = read_tum_trajectory(str(robot_map[rc]['tum']))
        n = len(ts_arr)
        pts = [t_xyz for idx, t_xyz, _ in jrr_poses[rc] if idx < n]
        if pts:
            out[rc] = np.array(pts)
    return out


def tum_dir_to_positions(robot_dirs: list[tuple[str, Path]]) -> dict:
    """
    robot_dirs: [(robot_char, tum_path), ...]
    Returns {robot_char: (N,3) positions}
    """
    out = {}
    for rc, tum_path in robot_dirs:
        _, pos, _ = read_tum_trajectory(str(tum_path))
        if len(pos):
            out[rc] = np.array(pos)
    return out


# ---------------------------------------------------------------------------
# Sim3 alignment helper (Umeyama) — used when no evo zip is available
# ---------------------------------------------------------------------------

def umeyama_alignment(src: np.ndarray, dst: np.ndarray):
    """
    Compute Sim3 alignment: dst ≈ s * R @ src.T + t
    src, dst: (N,3) matched point sets.
    Returns rotation (3,3), translation (3,), scale (float).
    """
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    n = src.shape[0]
    sigma2_s = (src_c ** 2).sum() / n
    H = (src_c.T @ dst_c) / n
    U, D, Vt = np.linalg.svd(H)
    det_sign = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, 1.0, det_sign])
    R = Vt.T @ S @ U.T
    s = (D * np.array([1.0, 1.0, det_sign])).sum() / sigma2_s if sigma2_s > 0 else 1.0
    t = mu_d - s * R @ mu_s
    return R, t, float(s)


def align_to_gt(positions_by_robot: dict, gt_by_robot: dict):
    """
    Compute a single Sim3 alignment (all robots concatenated) of positions to GT,
    then apply it. Returns aligned {robot_char: (N,3)}.
    """
    # Match by nearest timestamp isn't available here (no timestamps in positions_by_robot)
    # — use equal-length prefix matching per robot, then pool all for alignment
    src_pts, dst_pts = [], []
    for rc in sorted(positions_by_robot):
        if rc not in gt_by_robot:
            continue
        est = positions_by_robot[rc]
        gt = gt_by_robot[rc]
        n = min(len(est), len(gt))
        if n < 3:
            continue
        # downsample to at most 500 pts per robot
        idx = np.linspace(0, n - 1, min(n, 500), dtype=int)
        src_pts.append(est[idx])
        dst_pts.append(gt[idx])
    if not src_pts:
        return positions_by_robot
    src_all = np.vstack(src_pts)
    dst_all = np.vstack(dst_pts)
    R, t, s = umeyama_alignment(src_all, dst_all)
    return {rc: apply_alignment(pos, R, t, s) for rc, pos in positions_by_robot.items()}


# ---------------------------------------------------------------------------
# Build robot_map from a variant dir
# ---------------------------------------------------------------------------

ROBOT_DIR_TO_ID = {
    'hathor': 0, 'sparkal1': 1, 'sparkal2': 2,
    'thoth': 3, 'acl_jackal': 4, 'acl_jackal2': 5,
}


def build_robot_map(variant_dir: Path) -> dict:
    """Return {robot_char: {'tum': Path, 'robot_dir': Path}}"""
    robot_map = {}
    for g2o in sorted(variant_dir.glob("*/dpgo/bpsam_robot_*.g2o")):
        robot_id = int(g2o.stem.split("_")[-1])
        rc = chr(ord('a') + robot_id)
        tum = g2o.parent / f"Robot {robot_id}.tum"
        if not tum.exists():
            cands = sorted(g2o.parent.glob("Robot *.tum"))
            tum = cands[0] if cands else None
        if tum:
            robot_map[rc] = {'tum': tum, 'robot_dir': g2o.parent.parent}
    return robot_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('variant_dir', nargs='?', default='campus/ns-as')
    ap.add_argument('--gt_dir', default=None,
                    help='Ground truth directory (default: auto-detected)')
    ap.add_argument('--max_err', type=float, default=None)
    args = ap.parse_args()

    variant_dir = Path(args.variant_dir)
    if not variant_dir.is_absolute():
        variant_dir = Path(__file__).parent / variant_dir

    # Auto-detect gt_dir
    if args.gt_dir:
        gt_dir = Path(args.gt_dir)
    else:
        # e.g. campus/ns-as → ground_truth/campus
        exp_name = variant_dir.parent.name
        gt_dir = Path(__file__).parent / 'ground_truth' / exp_name
    print(f"Variant dir : {variant_dir}")
    print(f"GT dir      : {gt_dir}")

    mesa_dir = variant_dir / 'mesa_baselines'

    # -----------------------------------------------------------------------
    # Load GT trajectories
    # -----------------------------------------------------------------------
    gt_by_robot: dict[str, np.ndarray] = {}
    robot_chars = {}  # dir_name → char
    for dir_name, rid in ROBOT_DIR_TO_ID.items():
        rc = chr(ord('a') + rid)
        for ext in ['.csv', '.txt']:
            p = gt_dir / (dir_name + ext)
            if p.exists():
                _, pos, _ = load_gt_trajectory(str(p))
                if len(pos):
                    gt_by_robot[rc] = np.array(pos)
                    robot_chars[dir_name] = rc
                break
    print(f"GT robots loaded: {sorted(gt_by_robot)}")

    # -----------------------------------------------------------------------
    # Load each method
    # -----------------------------------------------------------------------

    # robot_map for MESA methods (source TUM for timestamp matching)
    robot_map = build_robot_map(variant_dir)
    print(f"Robot map chars: {sorted(robot_map)}")

    methods: list[tuple[str, dict, np.ndarray | None, np.ndarray | None, float]] = []
    # Each entry: (label, {rc: (N,3)}, R, t, s)
    # If R is None → use umeyama to GT

    # --- DGS ---
    jrr = find_latest_jrr(mesa_dir / 'dgs')
    if jrr:
        jrr_poses = load_jrr(jrr)
        pos = jrr_to_positions(robot_map, jrr_poses)
        zip_path = mesa_dir / 'dgs_evo_ape.zip'
        if zip_path.exists():
            R, t, s = load_alignment_from_evo_zip(zip_path)
            aligned = {rc: apply_alignment(p, R, t, s) for rc, p in pos.items()}
        else:
            aligned = align_to_gt(pos, gt_by_robot)
        methods.append(('DGS', aligned))
        print(f"DGS: {sorted(aligned)}")

    # --- ASAPP ---
    jrr = find_latest_jrr(mesa_dir / 'asapp')
    if jrr:
        jrr_poses = load_jrr(jrr)
        pos = jrr_to_positions(robot_map, jrr_poses)
        zip_path = mesa_dir / 'asapp_evo_ape.zip'
        if zip_path.exists():
            R, t, s = load_alignment_from_evo_zip(zip_path)
            aligned = {rc: apply_alignment(p, R, t, s) for rc, p in pos.items()}
        else:
            aligned = align_to_gt(pos, gt_by_robot)
        methods.append(('ASAPP', aligned))
        print(f"ASAPP: {sorted(aligned)}")

    # --- geodesic-MESA ---
    jrr = find_latest_jrr(mesa_dir / 'geodesic-mesa')
    if jrr:
        jrr_poses = load_jrr(jrr)
        pos = jrr_to_positions(robot_map, jrr_poses)
        zip_path = mesa_dir / 'geodesic_mesa_evo_ape.zip'
        if zip_path.exists():
            R, t, s = load_alignment_from_evo_zip(zip_path)
            aligned = {rc: apply_alignment(p, R, t, s) for rc, p in pos.items()}
        else:
            aligned = align_to_gt(pos, gt_by_robot)
        methods.append(('Geodesic-MESA', aligned))
        print(f"Geodesic-MESA: {sorted(aligned)}")

    # --- CBS+ ---
    cbs_plus_dir = variant_dir / 'cbs_plus'
    if cbs_plus_dir.exists():
        tum_list = []
        for robot_sub in sorted(cbs_plus_dir.iterdir()):
            dpgo = robot_sub / 'dpgo'
            if not dpgo.is_dir():
                continue
            tums = sorted(dpgo.glob("Robot*.tum"),
                          key=lambda p: int(p.stem.split('_')[-1]) if '_' in p.stem else 0)
            # skip empty files
            tums = [p for p in tums if p.stat().st_size > 0]
            if not tums:
                continue
            last = tums[-1]
            rid = int(last.stem.split('_')[0].split(' ')[-1])
            rc = chr(ord('a') + rid)
            tum_list.append((rc, last))
        pos = tum_dir_to_positions(tum_list)
        zip_path = cbs_plus_dir / 'cbs_plus_evo_ape.zip'
        if zip_path.exists():
            R, t, s = load_alignment_from_evo_zip(zip_path)
            aligned = {rc: apply_alignment(p, R, t, s) for rc, p in pos.items()}
        else:
            aligned = align_to_gt(pos, gt_by_robot)
        methods.append(('CBS+', aligned))
        print(f"CBS+: {sorted(aligned)}")

    # --- Centralized GNC-GM (lm_optimized) ---
    lm_dir = variant_dir / 'lm_optimized'
    if lm_dir.exists():
        tum_list = []
        for robot_sub in sorted(lm_dir.iterdir()):
            dpgo = robot_sub / 'dpgo'
            if not dpgo.is_dir():
                continue
            tums = sorted(dpgo.glob("Robot *.tum"))
            tums = [p for p in tums if p.stat().st_size > 0]
            if not tums:
                continue
            last = tums[-1]
            rid = int(last.stem.split(' ')[-1])
            rc = chr(ord('a') + rid)
            tum_list.append((rc, last))
        pos = tum_dir_to_positions(tum_list)
        zip_path = lm_dir / 'evo_ape.zip'
        if zip_path.exists():
            R, t, s = load_alignment_from_evo_zip(zip_path)
            aligned = {rc: apply_alignment(p, R, t, s) for rc, p in pos.items()}
        else:
            aligned = align_to_gt(pos, gt_by_robot)
        methods.append(('Centralized GNC-GM', aligned))
        print(f"Centralized GNC-GM: {sorted(aligned)}")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    METHOD_COLORS = [
        '#E41A1C',  # DGS         red
        '#FF7F00',  # ASAPP       orange
        '#984EA3',  # Geodesic-MESA purple
        '#377EB8',  # CBS+        blue
        '#4DAF4A',  # Centralized green
    ]
    METHOD_STYLES = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]

    plt.rcParams.update({**IEEE_RC, 'figure.figsize': (3.5, 3.5)})
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    # GT — plot each robot in light gray
    for rc, pos in sorted(gt_by_robot.items()):
        ax.plot(pos[:, 0], pos[:, 1], color='black', linewidth=0.7,
                linestyle='--', alpha=0.6,
                label='GT' if rc == sorted(gt_by_robot)[0] else None)

    # Methods
    for i, (label, aligned) in enumerate(methods):
        color = METHOD_COLORS[i % len(METHOD_COLORS)]
        ls = METHOD_STYLES[i % len(METHOD_STYLES)]
        first = True
        for rc in sorted(aligned):
            pos = aligned[rc]
            ax.plot(pos[:, 0], pos[:, 1], color=color, linewidth=0.8,
                    linestyle=ls, alpha=0.85,
                    label=label if first else None)
            first = False

    # Legend
    ax.legend(loc='best', framealpha=0.85, fontsize=6,
              handlelength=2.0, handletextpad=0.4, labelspacing=0.3)
    plt.tight_layout(pad=0.4)

    out = variant_dir / 'all_baselines_trajectories'
    save_fig(fig, out)
    plt.close(fig)


if __name__ == '__main__':
    main()
