#!/usr/bin/env python3
"""
Evaluate inter-robot loop closure recall against ground-truth loops.

For each detected loop closure (resolved to wall-clock timestamps via the
keyframe CSV), any GT loop for the same robot pair whose both timestamps fall
within --tol seconds of the detected timestamps is counted as "detected".

Recall is computed per robot-pair and overall for each GT angle threshold
found in <gt_dir>/gt_loops_angle*.csv.

If <experiment_dir> contains variant sub-folders (each holding robot subdirs
with distributed/ or dpgo/ data), all variants are evaluated and a
recall_comparison plot is saved to <experiment_dir>.  Otherwise the script
evaluates <experiment_dir> directly (backward-compatible behaviour).

Usage:
    python evaluate_loops_recall.py <experiment_dir> <gt_dir>
    python evaluate_loops_recall.py campus ground_truth/campus --tol 5.0
    python evaluate_loops_recall.py a5678 ground_truth/a5678 --tol 2.0

Output (per variant):
    <variant_dir>/loops_recall.txt   – human-readable stats
    <variant_dir>/loops_recall.csv   – machine-readable per-pair table
    <variant_dir>/loops_recall.pdf/png – single-variant recall curve

Output (multi-variant only):
    <experiment_dir>/recall_comparison.pdf/png – comparison across all variants
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import yaml

from utils.io import (load_keyframes_csv, load_loop_closures_csv, load_gt_trajectory,
                      load_variant_aliases, apply_variant_alias)
from utils.plot import IEEE_RC, ROBOT_COLORS, save_fig


# ---------------------------------------------------------------------------
# Robot discovery
# ---------------------------------------------------------------------------

def discover_robots(exp_dir: Path) -> dict[int, str]:
    """Return {robot_id: robot_dir_name}.

    First checks for a robot_names.yaml in exp_dir with keys robotN_name.
    Falls back to scanning for dpgo/Robot N.tum files.
    """
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


# ---------------------------------------------------------------------------
# Load detected loops → (robot_name, timestamp_s) pairs
# ---------------------------------------------------------------------------

def load_detected_loops(exp_dir: Path, id_to_name: dict[int, str]) -> list[dict]:
    """
    Load all unique inter-robot loop closures, resolved to timestamps (s).

    Returns a list of dicts: {idx, name1, t1_s, name2, t2_s, tx, ty, tz, qx, qy, qz, qw}.
    Pose fields (tx…qw) are None if not present in the CSV.
    """
    # Build robot_id → keyframe_id → timestamp_s
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
                continue  # skip intra-robot loops
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
            n1 = id_to_name.get(r1)
            n2 = id_to_name.get(r2)
            if n1 is None or n2 is None:
                continue
            entry: dict = {'idx': len(loops), 'name1': n1, 't1_s': t1, 'name2': n2, 't2_s': t2}
            for k in ('tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'):
                entry[k] = lc.get(k)
            loops.append(entry)

    return loops


# ---------------------------------------------------------------------------
# Load GT loops from gt_loops_angle*.csv
# ---------------------------------------------------------------------------

def load_gt_loops(gt_dir: Path) -> dict[int, list[dict]]:
    """
    Load all gt_loops_angle*.csv files from gt_dir.

    Returns {angle_deg (int): [{'robot_i', 't_i_s', 'robot_j', 't_j_s', '_key'}, ...]}.
    _key is a canonical tuple for deduplication across cumulative angle files.
    """
    result: dict[int, list[dict]] = {}
    for p in sorted(gt_dir.glob('gt_loops_angle*.csv'), key=lambda p: int(re.search(r'angle(\d+)', p.stem).group(1))):
        m = re.search(r'angle(\d+)', p.stem)
        if not m:
            continue
        angle = int(m.group(1))
        loops: list[dict] = []
        with open(p) as f:
            for row in csv.DictReader(f):
                ri, ti_ns = row['robot_i'], int(float(row['timestamp_i_ns']))
                rj, tj_ns = row['robot_j'], int(float(row['timestamp_j_ns']))
                if ri > rj:
                    ri, rj, ti_ns, tj_ns = rj, ri, tj_ns, ti_ns
                loops.append({
                    'robot_i': row['robot_i'],
                    't_i_s':   float(row['timestamp_i_ns']) / 1e9,
                    'robot_j': row['robot_j'],
                    't_j_s':   float(row['timestamp_j_ns']) / 1e9,
                    '_key':    (ri, ti_ns, rj, tj_ns),
                })
        result[angle] = loops
        print(f'  GT angle {angle:2d}°: {len(loops):6d} loops')
    return result


def compute_bucket_loops(
    gt_by_angle: dict[int, list[dict]],
    angles: list[int],
) -> dict[tuple[int, int], list[dict]]:
    """Split cumulative angle files into exclusive rotation-angle buckets.

    Bucket (prev, cur) contains only GT loops that appear in gt_loops_angle<cur>
    but NOT in gt_loops_angle<prev> (i.e. rotation in (prev°, cur°]).
    The first bucket is (0, angles[0]).
    """
    buckets: dict[tuple[int, int], list[dict]] = {}
    prev_keys: set = set()
    for i, angle in enumerate(angles):
        curr_loops = gt_by_angle[angle]
        curr_keys  = {l['_key'] for l in curr_loops}
        bucket_min = angles[i - 1] if i > 0 else 0
        buckets[(bucket_min, angle)] = [l for l in curr_loops if l['_key'] not in prev_keys]
        prev_keys = curr_keys
    return buckets


# ---------------------------------------------------------------------------
# Recall computation
# ---------------------------------------------------------------------------

def compute_recall(
    detected: list[dict],
    gt_loops: list[dict],
    tol_s: float,
    inlier_indices: set[int] | None = None,
) -> dict[tuple[str, str], tuple[int, int, int]]:
    """
    Compute per robot-pair recall against a GT loop list.

    A GT loop is "detected" if any detected loop covers the same robot pair
    and has both pose timestamps within tol_s of the GT pose timestamps
    (either orientation).

    Returns {(name_a, name_b): (n_detected, n_total, n_inlier_detected)} with sorted name pairs.
    n_inlier_detected counts GT loops whose matching detected loop is in inlier_indices.
    If inlier_indices is None, n_inlier_detected is always 0.
    """
    # Index detected loops by sorted robot-pair for fast lookup, tracking idx
    det_by_pair: dict[tuple[str, str], list[tuple[float, float, int]]] = {}
    for lc in detected:
        key = (min(lc['name1'], lc['name2']), max(lc['name1'], lc['name2']))
        if lc['name1'] <= lc['name2']:
            det_by_pair.setdefault(key, []).append((lc['t1_s'], lc['t2_s'], lc['idx']))
        else:
            det_by_pair.setdefault(key, []).append((lc['t2_s'], lc['t1_s'], lc['idx']))

    pair_counts: dict[tuple[str, str], list[int]] = {}
    for gt in gt_loops:
        ri, ti = gt['robot_i'], gt['t_i_s']
        rj, tj = gt['robot_j'], gt['t_j_s']
        pair_key = (min(ri, rj), max(ri, rj))
        if pair_key not in pair_counts:
            pair_counts[pair_key] = [0, 0, 0]
        pair_counts[pair_key][1] += 1

        # Normalise GT timestamps to sorted order
        if ri <= rj:
            gt_ta, gt_tb = ti, tj
        else:
            gt_ta, gt_tb = tj, ti

        candidates = det_by_pair.get(pair_key, [])
        for ta, tb, det_idx in candidates:
            if abs(ta - gt_ta) <= tol_s and abs(tb - gt_tb) <= tol_s:
                pair_counts[pair_key][0] += 1
                if inlier_indices is not None and det_idx in inlier_indices:
                    pair_counts[pair_key][2] += 1
                break

    return {k: (v[0], v[1], v[2]) for k, v in pair_counts.items()}


# ---------------------------------------------------------------------------
# Variant discovery
# ---------------------------------------------------------------------------

def _is_robot_dir(d: Path) -> bool:
    return (d / 'distributed').is_dir() or (d / 'dpgo').is_dir()


def discover_variants(exp_dir: Path) -> list[Path]:
    """Return subdirs of exp_dir that look like variant experiment dirs.

    A variant dir is a subdir that contains at least one robot subdir
    (identified by having a distributed/ or dpgo/ folder inside).
    """
    variants = []
    for d in sorted(exp_dir.iterdir()):
        if not d.is_dir():
            continue
        if any(_is_robot_dir(sub) for sub in d.iterdir() if sub.is_dir()):
            variants.append(d)
    return variants


def discover_baselines(exp_dir: Path) -> list[Path]:
    """Return baseline method dirs from baselines/<exp_dir.name>/*.

    Only includes dirs where discover_robots() returns results.
    """
    baseline_root = exp_dir.parent / 'baselines' / exp_dir.name
    if not baseline_root.exists():
        return []
    baselines = []
    for d in sorted(baseline_root.iterdir()):
        if not d.is_dir():
            continue
        if discover_robots(d):
            baselines.append(d)
    return baselines


# ---------------------------------------------------------------------------
# Per-variant evaluation
# ---------------------------------------------------------------------------

def evaluate_one(
    variant_dir: Path,
    buckets: dict[tuple[int, int], list[dict]],
    bucket_keys: list[tuple[int, int]],
    tol_s: float,
    gt_poses: dict[str, tuple] | None = None,
    max_gap_s: float = 2.5,
    trans_abs: float = 2.0,
    trans_rel: float = 0.10,
    rot_thr: float = 40.0,
) -> dict | None:
    """Evaluate recall for one variant dir. Saves CSV/txt.

    Returns {label, xs, recalls, inlier_recalls, inlier_pr} for plotting, or None on failure.
    inlier_recalls and inlier_pr are only present when gt_poses is provided and loops have poses.
    """
    id_to_name = discover_robots(variant_dir)
    if not id_to_name:
        print(f'  [SKIP] No robots found in {variant_dir.name}')
        return None
    print(f'Robots: { {v: k for k, v in id_to_name.items()} }')

    detected = load_detected_loops(variant_dir, id_to_name)
    n_detected = len(detected)
    print(f'Detected loops (unique, resolved): {n_detected}')

    # Tag inliers if GT poses available
    inlier_set: set[int] | None = None
    if gt_poses:
        inlier_set = tag_inliers(detected, gt_poses, max_gap_s, trans_abs, trans_rel, rot_thr)
        print(f'Inlier loops (pose quality): {len(inlier_set)}/{n_detected}')

    bucket_recall: dict[tuple[int, int], dict] = {}
    for bk in bucket_keys:
        bucket_recall[bk] = compute_recall(detected, buckets[bk], tol_s, inlier_set)

    # Stats + CSV
    stats_path = variant_dir / 'loops_recall.txt'
    csv_rows: list[dict] = []
    xs = [bmax for _, bmax in bucket_keys]
    recalls: list[float] = []
    inlier_recalls: list[float] = []

    with open(stats_path, 'w') as f:
        f.write(f'Loop Closure Recall — {variant_dir.name}\n')
        f.write(f'Time tolerance : {tol_s} s\n')
        f.write('=' * 60 + '\n\n')
        for bk in bucket_keys:
            bmin, bmax = bk
            pair_counts = bucket_recall[bk]
            total_det    = sum(v[0] for v in pair_counts.values())
            total_gt     = sum(v[1] for v in pair_counts.values())
            total_inlier = sum(v[2] for v in pair_counts.values())
            overall      = total_det / total_gt if total_gt > 0 else float('nan')
            inlier_rec   = total_inlier / total_gt if total_gt > 0 else float('nan')
            label = f'{bmin}-{bmax}°'
            f.write(f'bucket {label:8s}  overall recall: '
                    f'{overall:.3f}  ({total_det}/{total_gt})')
            if inlier_set is not None:
                f.write(f'  inlier recall: {inlier_rec:.3f}  ({total_inlier}/{total_gt})')
            f.write('\n')
            print(f'bucket {label:8s}  overall recall: '
                  f'{overall:.3f}  ({total_det}/{total_gt})', end='')
            if inlier_set is not None:
                print(f'  inlier recall: {inlier_rec:.3f}  ({total_inlier}/{total_gt})', end='')
            print()
            recalls.append(total_det / total_gt if total_gt > 0 else 0.0)
            inlier_recalls.append(total_inlier / total_gt if total_gt > 0 else 0.0)
            for (ra, rb), (nd, nt, ni) in sorted(pair_counts.items()):
                r = nd / nt if nt > 0 else float('nan')
                f.write(f'  {ra} <-> {rb}: {r:.3f}  ({nd}/{nt})\n')
                csv_rows.append({
                    'bucket_min': bmin, 'bucket_max': bmax,
                    'pair': f'{ra}<->{rb}',
                    'n_detected': nd, 'n_total': nt,
                    'n_inlier': ni,
                    'recall': f'{r:.4f}',
                })
            f.write('\n')

    print(f'Stats → {stats_path}')

    csv_path = variant_dir / 'loops_recall.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['bucket_min', 'bucket_max', 'pair', 'n_detected', 'n_total',
                           'n_inlier', 'recall'])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f'CSV   → {csv_path}')

    result: dict = {'label': variant_dir.name, 'xs': xs, 'recalls': recalls,
                    'n_detected': n_detected}
    if inlier_set is not None:
        result['inlier_recalls'] = inlier_recalls
        result['inlier_pr'] = len(inlier_set) / max(n_detected, 1)
        # Update inlier_counts.npy in the experiment folder so plot_ablation
        # can draw an inlier/PR curve without re-running GT comparison.
        # Keyed by variant dir name (e.g. "all", "ns-cs").
        counts_path = variant_dir.parent / 'inlier_counts.npy'
        try:
            counts: dict = (np.load(str(counts_path), allow_pickle=True).item()
                            if counts_path.exists() else {})
            counts[variant_dir.name] = len(inlier_set)
            np.save(str(counts_path), counts, allow_pickle=True)
            print(f'Updated inlier_counts.npy: {variant_dir.name}={len(inlier_set)}')
        except Exception as e:
            print(f'  Warning: could not update inlier_counts.npy: {e}')

        # Save inlier loop pairs for downstream use (e.g. algebraic connectivity)
        inlier_csv = variant_dir / 'inlier_loops.csv'
        with open(inlier_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['name1', 't1_s', 'name2', 't2_s'])
            writer.writeheader()
            for idx in sorted(inlier_set):
                lc = detected[idx]
                writer.writerow({'name1': lc['name1'], 't1_s': lc['t1_s'],
                                 'name2': lc['name2'], 't2_s': lc['t2_s']})
        print(f'Inlier loops → {inlier_csv}')
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_inlier_lines(ax, data: dict, color: str, is_baseline: bool) -> bool:
    """Draw inlier_recalls line for one series.

    Returns True if the line was drawn.
    """
    if 'inlier_recalls' not in data:
        return False
    ls = '-.' if is_baseline else '--'
    ax.plot(data['xs'], data['inlier_recalls'], color=color, marker='o', markersize=2,
            linestyle=ls, linewidth=0.8, alpha=0.7, label='_nolegend_')
    return True


def plot_single(data: dict, out_dir: Path) -> None:
    """Single-variant recall curve."""
    plt.rcParams.update({**IEEE_RC, 'figure.figsize': (3.5, 2.8)})
    fig, ax = plt.subplots()
    color = '#4C72B0'
    ax.plot(data['xs'], data['recalls'], color=color, marker='o', markersize=3, label='recall')
    drew_ir = _plot_inlier_lines(ax, data, color, is_baseline=False)
    handles, labels = ax.get_legend_handles_labels()
    if drew_ir:
        handles.append(mlines.Line2D([], [], color='gray', linestyle='--', linewidth=0.8))
        labels.append('inlier recall')
    if len(handles) > 1:
        ax.legend(handles, labels, loc='upper left', fontsize=6)
    ax.set_xticks(data['xs'])
    ax.set_xticklabels([f'{x}°' for x in data['xs']])
    ax.set_xlabel('GT Rotation Bucket Upper Bound')
    ax.set_ylabel('Recall')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.3)
    plt.tight_layout()
    save_fig(fig, out_dir / 'loops_recall')
    plt.close(fig)
    print(f'Plot  → {out_dir}/loops_recall.pdf')


def plot_comparison(variants: list[dict], baselines: list[dict], out_dir: Path) -> None:
    """Multi-variant recall comparison, with baselines shown as dashed lines.

    For each series, also draws inlier recall (dashed/dash-dot) and inlier/PR (dotted)
    if inlier data is available.
    """
    plt.rcParams.update({**IEEE_RC, 'figure.figsize': (3.5, 2.8)})
    fig, ax = plt.subplots()
    any_inlier_recall = False
    for i, d in enumerate(variants):
        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        ax.plot(d['xs'], d['recalls'], color=color, marker='o', markersize=3, label=d['label'])
        any_inlier_recall = _plot_inlier_lines(ax, d, color, is_baseline=False) or any_inlier_recall
    for i, d in enumerate(baselines):
        color = ROBOT_COLORS[(len(variants) + i) % len(ROBOT_COLORS)]
        ax.plot(d['xs'], d['recalls'], color=color, marker='o', markersize=3,
                linestyle='--', label=d['label'])
        any_inlier_recall = _plot_inlier_lines(ax, d, color, is_baseline=True) or any_inlier_recall
    xs = variants[0]['xs'] if variants else baselines[0]['xs']
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{x}°' for x in xs])
    ax.set_xlabel('GT Rotation Bucket Upper Bound')
    ax.set_ylabel('Recall')
    handles, labels = ax.get_legend_handles_labels()
    if any_inlier_recall:
        handles.append(mlines.Line2D([], [], color='gray', linestyle='--', linewidth=0.8))
        labels.append('inlier recall')
    ax.legend(handles, labels, loc='upper left', fontsize=6)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.3)
    plt.tight_layout()
    save_fig(fig, out_dir / 'recall_comparison')
    plt.close(fig)
    print(f'Comparison → {out_dir}/recall_comparison.pdf')


# ---------------------------------------------------------------------------
# Outlier analysis — compare detected vs GT relative pose
# ---------------------------------------------------------------------------

def _load_gt_poses(gt_dir: Path, robot_names: list[str]) -> dict[str, tuple]:
    """Return {robot_name: (timestamps_s (N,), positions (N,3), rotations_xyzw (N,4))}."""
    result: dict[str, tuple] = {}
    for name in robot_names:
        for ext in ('.csv', '.txt'):
            p = gt_dir / (name + ext)
            if p.exists():
                ts_ns, pos, rots = load_gt_trajectory(str(p))
                result[name] = (ts_ns / 1e9, pos, rots)
                break
    return result


def _nearest_pose(ts_s: float, timestamps: np.ndarray, positions: np.ndarray,
                  rotations: np.ndarray, max_gap_s: float):
    idx = int(np.argmin(np.abs(timestamps - ts_s)))
    if abs(float(timestamps[idx]) - ts_s) > max_gap_s:
        return None
    return positions[idx], rotations[idx]


def _quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
    """xyzw quaternion → 3×3 rotation matrix."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def _rot_angle_deg(R: np.ndarray) -> float:
    cos_val = float(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_val)))


def tag_inliers(
    detected: list[dict],
    gt_poses: dict[str, tuple],
    max_gap_s: float,
    trans_abs: float,
    trans_rel: float,
    rot_thr: float,
) -> set[int]:
    """Return indices (into detected) of loops whose relative pose estimate is accurate (inliers).

    A loop is an inlier if pose fields are present AND:
      - translation error <= max(trans_abs, trans_rel * GT distance)
      - AND rotation error <= rot_thr degrees
    Loops without pose data are not included.
    """
    inliers: set[int] = set()
    for lc in detected:
        if not all(lc.get(k) is not None for k in ('tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw')):
            continue
        n1, t1 = lc['name1'], lc['t1_s']
        n2, t2 = lc['name2'], lc['t2_s']
        if n1 not in gt_poses or n2 not in gt_poses:
            continue
        ts1, pos1, rot1 = gt_poses[n1]
        ts2, pos2, rot2 = gt_poses[n2]
        pose1 = _nearest_pose(t1, ts1, pos1, rot1, max_gap_s)
        pose2 = _nearest_pose(t2, ts2, pos2, rot2, max_gap_s)
        if pose1 is None or pose2 is None:
            continue
        p_gt1, r_gt1 = pose1
        p_gt2, r_gt2 = pose2
        R1 = _quat_xyzw_to_rot(r_gt1)
        R2 = _quat_xyzw_to_rot(r_gt2)
        p_rel_gt  = R1.T @ (p_gt2 - p_gt1)
        R_rel_gt  = R1.T @ R2
        p_det     = np.array([lc['tx'], lc['ty'], lc['tz']])
        R_det     = _quat_xyzw_to_rot(np.array([lc['qx'], lc['qy'], lc['qz'], lc['qw']]))
        gt_dist   = float(np.linalg.norm(p_rel_gt))
        trans_err = float(np.linalg.norm(p_det - p_rel_gt))
        rot_err   = _rot_angle_deg(R_det.T @ R_rel_gt)
        if trans_err <= max(trans_abs, trans_rel * gt_dist) and rot_err <= rot_thr:
            inliers.add(lc['idx'])
    return inliers


def compute_outlier_ratio(
    variant_dir: Path,
    id_to_name: dict[int, str],
    gt_poses: dict[str, tuple],
    max_gap_s: float,
    trans_abs: float,
    trans_rel: float,
    rot_thr: float,
) -> tuple[float, int] | None:
    """Compute outlier ratio for detected loops that carry a relative-pose estimate.

    A detected loop is an outlier if:
      - translation error > max(trans_abs, trans_rel * GT distance)
      - OR rotation error > rot_thr degrees

    Returns (outlier_ratio, n_evaluated) or None if no evaluable loops found.
    """
    kf_maps: dict[int, dict[int, float]] = {}
    for rid, rname in id_to_name.items():
        kf_path = variant_dir / rname / 'distributed' / 'kimera_distributed_keyframes.csv'
        if kf_path.exists():
            kf_maps[rid] = load_keyframes_csv(str(kf_path))

    seen: set[frozenset] = set()
    n_total = 0
    n_outliers = 0

    for rid, rname in id_to_name.items():
        lc_path = variant_dir / rname / 'distributed' / 'loop_closures.csv'
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

            # Skip if no pose estimate in the CSV
            if not all(k in lc for k in ('tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw')):
                continue

            if r1 not in kf_maps or r2 not in kf_maps:
                continue
            t1 = kf_maps[r1].get(p1)
            t2 = kf_maps[r2].get(p2)
            if t1 is None or t2 is None:
                continue

            n1 = id_to_name.get(r1)
            n2 = id_to_name.get(r2)
            if n1 not in gt_poses or n2 not in gt_poses:
                continue

            ts1, pos1, rot1 = gt_poses[n1]
            ts2, pos2, rot2 = gt_poses[n2]
            pose1 = _nearest_pose(t1, ts1, pos1, rot1, max_gap_s)
            pose2 = _nearest_pose(t2, ts2, pos2, rot2, max_gap_s)
            if pose1 is None or pose2 is None:
                continue

            p_gt1, r_gt1 = pose1  # (3,), xyzw (4,)
            p_gt2, r_gt2 = pose2

            # GT relative pose in robot-1's local frame
            R1 = _quat_xyzw_to_rot(r_gt1)
            R2 = _quat_xyzw_to_rot(r_gt2)
            p_rel_gt = R1.T @ (p_gt2 - p_gt1)
            R_rel_gt = R1.T @ R2

            # Detected relative pose
            p_det = np.array([lc['tx'], lc['ty'], lc['tz']])
            R_det = _quat_xyzw_to_rot(np.array([lc['qx'], lc['qy'], lc['qz'], lc['qw']]))

            gt_dist   = float(np.linalg.norm(p_rel_gt))
            trans_err = float(np.linalg.norm(p_det - p_rel_gt))
            rot_err   = _rot_angle_deg(R_det.T @ R_rel_gt)

            n_total += 1
            if trans_err > max(trans_abs, trans_rel * gt_dist) or rot_err > rot_thr:
                n_outliers += 1

    if n_total == 0:
        return None
    return n_outliers / n_total, n_total


def collect_outlier_stats(
    variant_dir: Path,
    id_to_name: dict[int, str],
    gt_poses: dict[str, tuple],
    max_gap_s: float,
    trans_abs: float,
    trans_rel: float,
    rot_thr: float,
) -> list[dict]:
    """Return per-loop dicts with gt_dist, gt_rot_deg, is_outlier for all evaluable loops."""
    kf_maps: dict[int, dict[int, float]] = {}
    for rid, rname in id_to_name.items():
        kf_path = variant_dir / rname / 'distributed' / 'kimera_distributed_keyframes.csv'
        if kf_path.exists():
            kf_maps[rid] = load_keyframes_csv(str(kf_path))

    seen: set[frozenset] = set()
    records: list[dict] = []

    for rid, rname in id_to_name.items():
        lc_path = variant_dir / rname / 'distributed' / 'loop_closures.csv'
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

            if not all(k in lc for k in ('tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw')):
                continue
            if r1 not in kf_maps or r2 not in kf_maps:
                continue
            t1 = kf_maps[r1].get(p1)
            t2 = kf_maps[r2].get(p2)
            if t1 is None or t2 is None:
                continue
            n1 = id_to_name.get(r1)
            n2 = id_to_name.get(r2)
            if n1 not in gt_poses or n2 not in gt_poses:
                continue

            ts1, pos1, rot1 = gt_poses[n1]
            ts2, pos2, rot2 = gt_poses[n2]
            pose1 = _nearest_pose(t1, ts1, pos1, rot1, max_gap_s)
            pose2 = _nearest_pose(t2, ts2, pos2, rot2, max_gap_s)
            if pose1 is None or pose2 is None:
                continue

            p_gt1, r_gt1 = pose1
            p_gt2, r_gt2 = pose2
            R1 = _quat_xyzw_to_rot(r_gt1)
            R2 = _quat_xyzw_to_rot(r_gt2)
            p_rel_gt = R1.T @ (p_gt2 - p_gt1)
            R_rel_gt = R1.T @ R2

            p_det = np.array([lc['tx'], lc['ty'], lc['tz']])
            R_det = _quat_xyzw_to_rot(np.array([lc['qx'], lc['qy'], lc['qz'], lc['qw']]))

            gt_dist    = float(np.linalg.norm(p_rel_gt))
            gt_rot_deg = _rot_angle_deg(R_rel_gt)
            trans_err  = float(np.linalg.norm(p_det - p_rel_gt))
            rot_err    = _rot_angle_deg(R_det.T @ R_rel_gt)
            is_outlier = trans_err > max(trans_abs, trans_rel * gt_dist) or rot_err > rot_thr

            det_dist    = float(np.linalg.norm(p_det))
            det_rot_deg = _rot_angle_deg(R_det)

            records.append({'gt_dist': gt_dist, 'gt_rot_deg': gt_rot_deg,
                            'trans_err': trans_err, 'rot_err': rot_err,
                            'det_dist': det_dist, 'det_rot_deg': det_rot_deg,
                            'is_outlier': is_outlier})

    return records


def _outlier_ratio_by_bin(
    records: list[dict], key: str, edges: 'list[float] | list[int]'
) -> tuple[list[float], list[float], list[int]]:
    """Bin records by key, return (bin_centres, outlier_ratios, counts)."""
    centres, ratios, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        subset = [r for r in records if lo <= r[key] < hi]
        n = len(subset)
        counts.append(n)
        centres.append((lo + hi) / 2)
        ratios.append(sum(r['is_outlier'] for r in subset) / n if n > 0 else float('nan'))
    return centres, ratios, counts


def plot_outlier_by_gt(
    variant_stats: list[tuple[str, list[dict]]],
    baseline_stats: list[tuple[str, list[dict]]],
    out_dir: Path,
) -> None:
    """Two-panel plot: outlier ratio vs GT translation distance and vs GT rotation angle."""
    dist_edges = [0, 5, 10, 15, 20, 25, 30, 40, 50, 70]
    rot_edges  = [0, 10, 20, 30, 40, 50, 60, 75, 90, 120]

    n_variants  = len(variant_stats)
    n_baselines = len(baseline_stats)
    all_series  = variant_stats + baseline_stats

    plt.rcParams.update({**IEEE_RC, 'figure.figsize': (7.0, 2.8)})
    fig, axes = plt.subplots(1, 2)
    ax_dist, ax_rot = axes[0], axes[1]

    for i, (label, records) in enumerate(all_series):
        is_bl = i >= n_variants
        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        ls = '--' if is_bl else '-'
        lw = 0.8

        centres, ratios, counts = _outlier_ratio_by_bin(records, 'gt_dist', dist_edges)
        mask = [not np.isnan(r) for r in ratios]
        ax_dist.plot(
            [c for c, m in zip(centres, mask) if m],
            [r for r, m in zip(ratios, mask) if m],
            color=color, linestyle=ls, linewidth=lw, marker='o', markersize=2.5, label=label,
        )

        centres, ratios, _ = _outlier_ratio_by_bin(records, 'gt_rot_deg', rot_edges)
        mask = [not np.isnan(r) for r in ratios]
        ax_rot.plot(
            [c for c, m in zip(centres, mask) if m],
            [r for r, m in zip(ratios, mask) if m],
            color=color, linestyle=ls, linewidth=lw, marker='o', markersize=2.5, label=label,
        )

    for ax, xlabel in [(ax_dist, 'GT translation distance (m)'),
                       (ax_rot, 'GT relative rotation (°)')]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Outlier ratio')
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.3)

    handles, labels_leg = ax_dist.get_legend_handles_labels()
    ax_rot.legend(handles, labels_leg, fontsize=5, loc='upper left')

    plt.tight_layout()
    save_fig(fig, out_dir / 'outlier_by_gt')
    plt.close(fig)
    print(f'Outlier-by-GT plot → {out_dir}/outlier_by_gt.pdf')


def plot_loop_error_histograms(
    variant_stats: list[tuple[str, list[dict]]],
    baseline_stats: list[tuple[str, list[dict]]],
    out_dir: Path,
    max_trans: float = 5.0,
    max_rot: float = 180.0,
    bins: int = 25,
) -> None:
    """Two-panel histogram of GT translation + rotation error for ALL detected loops.

    Left panel:  translation error distribution (m) — reveals fat tails in sloppy methods.
    Right panel: rotation error distribution (°).
    Threshold lines mark the inlier/outlier boundary used elsewhere.
    """
    n_variants = len(variant_stats)
    all_series = variant_stats + baseline_stats

    plt.rcParams.update({**IEEE_RC, 'figure.figsize': (7.0, 2.5)})
    fig, (ax_t, ax_r) = plt.subplots(1, 2)
    t_edges = np.linspace(0, max_trans, bins + 1)
    r_edges = np.linspace(0, max_rot,   bins + 1)

    for i, (label, records) in enumerate(all_series):
        is_bl = i >= n_variants
        t_errs = [r['trans_err'] for r in records]
        r_errs = [r['rot_err']   for r in records]
        if not t_errs:
            continue
        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        ls = '--' if is_bl else '-'
        kw = dict(histtype='step', color=color, linestyle=ls,
                  linewidth=1.0, density=True,
                  label=f'{label} (n={len(t_errs)})')
        ax_t.hist(t_errs, bins=t_edges, **kw)
        ax_r.hist(r_errs, bins=r_edges, **kw)

    ax_t.set_xlabel('GT Translation Error (m)')
    ax_t.set_ylabel('Density')
    ax_t.legend(fontsize=5, loc='upper right')
    ax_t.grid(True, alpha=0.3, linestyle='--', linewidth=0.3)

    ax_r.set_xlabel('GT Rotation Error (°)')
    ax_r.set_ylabel('Density')
    ax_r.legend(fontsize=5, loc='upper right')
    ax_r.grid(True, alpha=0.3, linestyle='--', linewidth=0.3)

    plt.tight_layout()
    save_fig(fig, out_dir / 'loop_error_hist')
    plt.close(fig)
    print(f'Loop error hist → {out_dir}/loop_error_hist.pdf')


def plot_loop_measurement_hist(
    variant_stats: list[tuple[str, list[dict]]],
    baseline_stats: list[tuple[str, list[dict]]],
    out_dir: Path,
    max_trans: float = 20.0,
    max_rot: float = 180.0,
    bins: int = 30,
) -> None:
    """Two-panel histogram of the detected loop relative-pose measurements.

    Left panel:  translation magnitude ||t_detected|| (m)
    Right panel: rotation angle of R_detected (°)

    Shows the distribution of what the loop closure detector is actually proposing,
    independent of GT accuracy.
    """
    n_variants = len(variant_stats)
    all_series = variant_stats + baseline_stats

    plt.rcParams.update({**IEEE_RC, 'figure.figsize': (7.0, 2.5)})
    fig, (ax_t, ax_r) = plt.subplots(1, 2)
    t_edges = np.linspace(0, max_trans, bins + 1)
    r_edges = np.linspace(0, max_rot,   bins + 1)

    for i, (label, records) in enumerate(all_series):
        is_bl = i >= n_variants
        t_meas = [r['det_dist']    for r in records]
        r_meas = [r['det_rot_deg'] for r in records]
        if not t_meas:
            continue
        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        ls = '--' if is_bl else '-'
        kw = dict(histtype='step', color=color, linestyle=ls,
                  linewidth=1.0, density=True,
                  label=f'{label} (n={len(t_meas)})')
        ax_t.hist(t_meas, bins=t_edges, **kw)
        ax_r.hist(r_meas, bins=r_edges, **kw)

    ax_t.set_xlabel('Detected Translation (m)')
    ax_t.set_ylabel('Density')
    ax_t.legend(fontsize=5, loc='upper right')
    ax_t.grid(True, alpha=0.3, linestyle='--', linewidth=0.3)

    ax_r.set_xlabel('Detected Rotation (°)')
    ax_r.set_ylabel('Density')
    ax_r.legend(fontsize=5, loc='upper right')
    ax_r.grid(True, alpha=0.3, linestyle='--', linewidth=0.3)

    plt.tight_layout()
    save_fig(fig, out_dir / 'loop_measurement_hist')
    plt.close(fig)
    print(f'Loop measurement hist → {out_dir}/loop_measurement_hist.pdf')


def plot_outlier_comparison(
    variant_data: list[tuple[str, float, int]],
    baseline_data: list[tuple[str, float, int]],
    out_dir: Path,
    trans_abs: float,
    trans_rel: float,
    rot_thr: float,
) -> None:
    """Bar chart of outlier ratios. Variants solid, baselines hatched."""
    plt.rcParams.update({**IEEE_RC, 'figure.figsize': (3.5, 2.8)})
    fig, ax = plt.subplots()

    all_data = variant_data + baseline_data
    labels  = [d[0] for d in all_data]
    ratios  = [d[1] for d in all_data]
    counts  = [d[2] for d in all_data]
    xs = list(range(len(all_data)))

    colors = (
        [ROBOT_COLORS[i % len(ROBOT_COLORS)] for i in range(len(variant_data))] +
        [ROBOT_COLORS[(len(variant_data) + i) % len(ROBOT_COLORS)]
         for i in range(len(baseline_data))]
    )
    hatches = [''] * len(variant_data) + ['//'] * len(baseline_data)

    bars = ax.bar(xs, ratios, color=colors, edgecolor='black', linewidth=0.5)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'n={count}', ha='center', va='bottom', fontsize=5)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=6)
    ax.set_ylabel('Outlier Ratio')
    ax.set_ylim(0, 1.1)
    ax.set_title(f'Loop Outlier Ratio (trans > max({trans_abs:.1f}m, {trans_rel*100:.0f}% GT dist) or rot > {rot_thr:.0f}°)',
                 fontsize=7)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.3)
    plt.tight_layout()
    save_fig(fig, out_dir / 'outlier_comparison')
    plt.close(fig)
    print(f'Outlier plot → {out_dir}/outlier_comparison.pdf')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate loop closure recall against GT loops.'
    )
    parser.add_argument('experiment_dir', type=Path,
                        help='Experiment folder (e.g. campus) or a variant subfolder')
    parser.add_argument('gt_dir', type=Path,
                        help='Folder containing gt_loops_angle*.csv files')
    parser.add_argument('--tol', type=float, default=2.0,
                        help='Timestamp tolerance in seconds (default: 2.0)')
    parser.add_argument('--max-angle', type=int, default=None, dest='max_angle',
                        help='Only evaluate angle thresholds up to this value in degrees')
    parser.add_argument('--trans-abs', type=float, default=2.0, dest='trans_abs',
                        help='Absolute translation error floor for inlier/outlier detection in metres (default: 2.0)')
    parser.add_argument('--trans-rel', type=float, default=0.10, dest='trans_rel',
                        help='Relative translation error threshold as fraction of GT distance (default: 0.10 = 10%%)')
    parser.add_argument('--rot-thr', type=float, default=40.0, dest='rot_thr',
                        help='Rotation error threshold for outlier detection in degrees (default: 40.0)')
    parser.add_argument('--max-gap', type=float, default=2.5, dest='max_gap',
                        help='Max GT timestamp gap for outlier pose lookup in seconds (default: 2.5)')
    args = parser.parse_args()

    exp_dir = args.experiment_dir.resolve()
    gt_dir  = args.gt_dir.resolve()

    print(f'Experiment : {exp_dir.name}')
    print(f'GT dir     : {gt_dir}')
    print(f'Time tol   : {args.tol} s\n')

    # Load GT loops once (shared across all variants)
    print('Loading GT loops...')
    gt_by_angle = load_gt_loops(gt_dir)
    if not gt_by_angle:
        print('No gt_loops_angle*.csv files found.')
        raise SystemExit(1)
    angles = sorted(gt_by_angle.keys())
    if args.max_angle is not None:
        angles = [a for a in angles if a <= args.max_angle]

    buckets = compute_bucket_loops(gt_by_angle, angles)
    bucket_keys = sorted(buckets.keys())

    # Discover variants or fall back to single-experiment mode
    variants  = discover_variants(exp_dir)
    baselines = discover_baselines(exp_dir)

    eval_dirs_variant  = variants  if variants  else []
    eval_dirs_baseline = baselines if baselines else []
    single_mode = not variants and not baselines

    if variants:
        print(f'\nFound {len(variants)} variant(s): {[v.name for v in variants]}')
    if baselines:
        print(f'Found {len(baselines)} baseline(s): {[b.name for b in baselines]}')

    variant_results:  list[dict] = []
    baseline_results: list[dict] = []

    # Load GT poses early so evaluate_one can tag inliers per-bucket
    print('\nLoading GT poses for inlier tagging...')
    all_eval_dirs_tmp = (eval_dirs_variant + eval_dirs_baseline) if not single_mode else [exp_dir]
    all_robot_names: set[str] = set()
    for d in all_eval_dirs_tmp:
        all_robot_names.update(discover_robots(d).values())
    gt_poses = _load_gt_poses(gt_dir, sorted(all_robot_names))
    if not gt_poses:
        print('  No GT pose files found — inlier lines will be omitted.')

    eval_kwargs = dict(gt_poses=gt_poses or None,
                       max_gap_s=args.max_gap,
                       trans_abs=args.trans_abs,
                       trans_rel=args.trans_rel,
                       rot_thr=args.rot_thr)

    if single_mode:
        r = evaluate_one(exp_dir, buckets, bucket_keys, args.tol, **eval_kwargs)
        if r:
            plot_single(r, exp_dir)
            variant_results.append(r)
    else:
        for v in eval_dirs_variant:
            print(f'\n--- {v.name} ---')
            r = evaluate_one(v, buckets, bucket_keys, args.tol, **eval_kwargs)
            if r:
                variant_results.append(r)
        for b in eval_dirs_baseline:
            print(f'\n--- {b.name} (baseline) ---')
            r = evaluate_one(b, buckets, bucket_keys, args.tol, **eval_kwargs)
            if r:
                baseline_results.append(r)
        # Apply variant aliases / filter for comparison plots
        aliases = load_variant_aliases()
        def _alias_results(results):
            out = []
            for r in results:
                disp = apply_variant_alias(aliases, r['label'])
                if disp is not None:
                    out.append(dict(r, label=disp))
            return out
        plot_var = _alias_results(variant_results)
        plot_bl  = _alias_results(baseline_results)

        all_results = variant_results + baseline_results
        all_plot    = plot_var + plot_bl
        if len(all_plot) >= 2:
            plot_comparison(plot_var, plot_bl, exp_dir)
        elif all_plot:
            plot_single(all_plot[0], exp_dir)

    # ------------------------------------------------------------------
    # Outlier analysis: compare detected relative pose vs GT
    # ------------------------------------------------------------------
    print(f'\n--- Outlier analysis (trans > max({args.trans_abs:.1f}m, {args.trans_rel*100:.0f}% GT dist), rot_thr={args.rot_thr}°) ---')

    if not gt_poses:
        print('  No GT pose files found — skipping outlier analysis.')
        return

    outlier_variant:  list[tuple[str, float, int]] = []
    outlier_baseline: list[tuple[str, float, int]] = []

    for d in (eval_dirs_variant if not single_mode else [exp_dir]):
        disp = apply_variant_alias(aliases, d.name)
        if disp is None:
            continue
        id_to_name = discover_robots(d)
        result = compute_outlier_ratio(
            d, id_to_name, gt_poses, args.max_gap, args.trans_abs, args.trans_rel, args.rot_thr)
        if result is None:
            print(f'  [{d.name}] No loops with pose data — skipping.')
        else:
            ratio, n = result
            print(f'  [{d.name}] outliers {ratio:.3f}  ({int(ratio*n)}/{n})')
            outlier_variant.append((disp, ratio, n))

    for d in eval_dirs_baseline:
        disp = apply_variant_alias(aliases, d.name)
        if disp is None:
            continue
        id_to_name = discover_robots(d)
        result = compute_outlier_ratio(
            d, id_to_name, gt_poses, args.max_gap, args.trans_abs, args.trans_rel, args.rot_thr)
        if result is None:
            print(f'  [{d.name}] No loops with pose data — skipping.')
        else:
            ratio, n = result
            print(f'  [{d.name}] outliers {ratio:.3f}  ({int(ratio*n)}/{n})')
            outlier_baseline.append((disp, ratio, n))

    out_plot_dir = exp_dir
    all_outlier = outlier_variant + outlier_baseline
    if len(all_outlier) >= 1:
        plot_outlier_comparison(outlier_variant, outlier_baseline, out_plot_dir,
                                args.trans_abs, args.trans_rel, args.rot_thr)
    else:
        print('  No outlier data to plot.')

    # Per-loop outlier stats broken down by GT translation and rotation
    gt_stats_variant:  list[tuple[str, list[dict]]] = []
    gt_stats_baseline: list[tuple[str, list[dict]]] = []

    for d in (eval_dirs_variant if not single_mode else [exp_dir]):
        disp = apply_variant_alias(aliases, d.name)
        if disp is None:
            continue
        id_to_name = discover_robots(d)
        records = collect_outlier_stats(
            d, id_to_name, gt_poses, args.max_gap, args.trans_abs, args.trans_rel, args.rot_thr)
        if records:
            gt_stats_variant.append((disp, records))

    for d in eval_dirs_baseline:
        disp = apply_variant_alias(aliases, d.name)
        if disp is None:
            continue
        id_to_name = discover_robots(d)
        records = collect_outlier_stats(
            d, id_to_name, gt_poses, args.max_gap, args.trans_abs, args.trans_rel, args.rot_thr)
        if records:
            gt_stats_baseline.append((disp, records))

    if gt_stats_variant or gt_stats_baseline:
        plot_outlier_by_gt(gt_stats_variant, gt_stats_baseline, out_plot_dir)
        plot_loop_error_histograms(gt_stats_variant, gt_stats_baseline, out_plot_dir)
        plot_loop_measurement_hist(gt_stats_variant, gt_stats_baseline, out_plot_dir)
    else:
        print('  No per-loop data for GT-breakdown plot.')


if __name__ == '__main__':
    main()
