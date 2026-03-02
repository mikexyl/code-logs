#!/usr/bin/env python3
"""
Evaluate inter-robot loop closure recall against ground-truth loops.

For each detected loop closure (resolved to wall-clock timestamps via the
keyframe CSV), any GT loop for the same robot pair whose both timestamps fall
within --tol seconds of the detected timestamps is counted as "detected".

Recall is computed per robot-pair and overall for each GT angle threshold
found in <gt_dir>/gt_loops_angle*.csv.  A box plot shows the per-pair recall
distribution across thresholds.

Usage:
    python evaluate_loops_recall.py <experiment_dir> <gt_dir>
    python evaluate_loops_recall.py a5678 ground_truth/a5678
    python evaluate_loops_recall.py a5678 ground_truth/a5678 --tol 2.0

Output:
    <experiment_dir>/loops_recall.txt   – human-readable stats
    <experiment_dir>/loops_recall.csv   – machine-readable per-pair table
    <experiment_dir>/loops_recall.pdf/png – box plot
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils.io import load_keyframes_csv, load_loop_closures_csv
from utils.plot import IEEE_RC, ROBOT_COLORS, save_fig


# ---------------------------------------------------------------------------
# Robot discovery
# ---------------------------------------------------------------------------

def discover_robots(exp_dir: Path) -> dict[int, str]:
    """Return {robot_id: robot_dir_name} by scanning for dpgo/Robot N.tum."""
    id_to_name: dict[int, str] = {}
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

    Returns a list of dicts: {name1, t1_s, name2, t2_s}.
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
            loops.append({'name1': n1, 't1_s': t1, 'name2': n2, 't2_s': t2})

    return loops


# ---------------------------------------------------------------------------
# Load GT loops from gt_loops_angle*.csv
# ---------------------------------------------------------------------------

def load_gt_loops(gt_dir: Path) -> dict[int, list[dict]]:
    """
    Load all gt_loops_angle*.csv files from gt_dir.

    Returns {angle_deg (int): [{'robot_i', 't_i_s', 'robot_j', 't_j_s'}, ...]}.
    """
    result: dict[int, list[dict]] = {}
    for p in sorted(gt_dir.glob('gt_loops_angle*.csv')):
        m = re.search(r'angle(\d+)', p.stem)
        if not m:
            continue
        angle = int(m.group(1))
        loops: list[dict] = []
        with open(p) as f:
            for row in csv.DictReader(f):
                loops.append({
                    'robot_i': row['robot_i'],
                    't_i_s':   float(row['timestamp_i_ns']) / 1e9,
                    'robot_j': row['robot_j'],
                    't_j_s':   float(row['timestamp_j_ns']) / 1e9,
                })
        result[angle] = loops
        print(f'  GT angle {angle:2d}°: {len(loops):6d} loops')
    return result


# ---------------------------------------------------------------------------
# Recall computation
# ---------------------------------------------------------------------------

def compute_recall(
    detected: list[dict],
    gt_loops: list[dict],
    tol_s: float,
) -> dict[tuple[str, str], tuple[int, int]]:
    """
    Compute per robot-pair recall against a GT loop list.

    A GT loop is "detected" if any detected loop covers the same robot pair
    and has both pose timestamps within tol_s of the GT pose timestamps
    (either orientation).

    Returns {(name_a, name_b): (n_detected, n_total)} with sorted name pairs.
    """
    # Index detected loops by sorted robot-pair for fast lookup
    det_by_pair: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for lc in detected:
        key = (min(lc['name1'], lc['name2']), max(lc['name1'], lc['name2']))
        # store as (t_for_min_robot, t_for_max_robot)
        if lc['name1'] <= lc['name2']:
            det_by_pair.setdefault(key, []).append((lc['t1_s'], lc['t2_s']))
        else:
            det_by_pair.setdefault(key, []).append((lc['t2_s'], lc['t1_s']))

    pair_counts: dict[tuple[str, str], list[int]] = {}
    for gt in gt_loops:
        ri, ti = gt['robot_i'], gt['t_i_s']
        rj, tj = gt['robot_j'], gt['t_j_s']
        pair_key = (min(ri, rj), max(ri, rj))
        if pair_key not in pair_counts:
            pair_counts[pair_key] = [0, 0]
        pair_counts[pair_key][1] += 1

        # Normalise GT timestamps to sorted order
        if ri <= rj:
            gt_ta, gt_tb = ti, tj
        else:
            gt_ta, gt_tb = tj, ti

        candidates = det_by_pair.get(pair_key, [])
        matched = any(
            abs(ta - gt_ta) <= tol_s and abs(tb - gt_tb) <= tol_s
            for ta, tb in candidates
        )
        if matched:
            pair_counts[pair_key][0] += 1

    return {k: (v[0], v[1]) for k, v in pair_counts.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate loop closure recall against GT loops.'
    )
    parser.add_argument('experiment_dir', type=Path,
                        help='Experiment folder (e.g. a5678)')
    parser.add_argument('gt_dir', type=Path,
                        help='Folder containing gt_loops_angle*.csv files '
                             '(e.g. ground_truth/a5678)')
    parser.add_argument('--tol', type=float, default=2.0,
                        help='Timestamp tolerance in seconds (default: 2.0)')
    args = parser.parse_args()

    exp_dir = args.experiment_dir.resolve()
    gt_dir  = args.gt_dir.resolve()

    print(f'Experiment : {exp_dir.name}')
    print(f'GT dir     : {gt_dir}')
    print(f'Time tol   : {args.tol} s\n')

    # Robots
    id_to_name = discover_robots(exp_dir)
    if not id_to_name:
        print('No robots found (need dpgo/Robot N.tum).')
        raise SystemExit(1)
    print(f'Robots: { {v: k for k, v in id_to_name.items()} }')

    # Detected loops
    detected = load_detected_loops(exp_dir, id_to_name)
    print(f'Detected loops (unique, resolved): {len(detected)}\n')

    # GT loops
    print('Loading GT loops...')
    gt_by_angle = load_gt_loops(gt_dir)
    if not gt_by_angle:
        print('No gt_loops_angle*.csv files found.')
        raise SystemExit(1)
    angles = sorted(gt_by_angle.keys())

    # Recall per angle per pair
    recall_data: dict[int, dict[tuple[str, str], tuple[int, int]]] = {}
    for angle in angles:
        recall_data[angle] = compute_recall(detected, gt_by_angle[angle], args.tol)

    # -----------------------------------------------------------------------
    # Stats output
    # -----------------------------------------------------------------------
    stats_path = exp_dir / 'loops_recall.txt'
    csv_rows: list[dict] = []

    with open(stats_path, 'w') as f:
        f.write(f'Loop Closure Recall — {exp_dir.name}\n')
        f.write(f'Time tolerance : {args.tol} s\n')
        f.write('=' * 60 + '\n\n')
        for angle in angles:
            pair_counts = recall_data[angle]
            total_det = sum(v[0] for v in pair_counts.values())
            total_gt  = sum(v[1] for v in pair_counts.values())
            overall   = total_det / total_gt if total_gt > 0 else float('nan')
            f.write(f'angle <= {angle:2d}°  overall recall: '
                    f'{overall:.3f}  ({total_det}/{total_gt})\n')
            print(f'angle <= {angle:2d}°  overall recall: '
                  f'{overall:.3f}  ({total_det}/{total_gt})')
            for (ra, rb), (nd, nt) in sorted(pair_counts.items()):
                r = nd / nt if nt > 0 else float('nan')
                f.write(f'  {ra} <-> {rb}: {r:.3f}  ({nd}/{nt})\n')
                csv_rows.append({
                    'angle': angle, 'pair': f'{ra}<->{rb}',
                    'n_detected': nd, 'n_total': nt,
                    'recall': f'{r:.4f}',
                })
            f.write('\n')

    print(f'\nStats → {stats_path}')

    csv_path = exp_dir / 'loops_recall.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['angle', 'pair', 'n_detected', 'n_total', 'recall'])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f'CSV   → {csv_path}')

    # -----------------------------------------------------------------------
    # Box plot: per-pair recall distribution across angle thresholds
    # -----------------------------------------------------------------------
    plt.rcParams.update({**IEEE_RC, 'figure.figsize': (3.5, 2.8)})
    fig, ax = plt.subplots()

    box_data = []
    for angle in angles:
        vals = [
            nd / nt if nt > 0 else float('nan')
            for nd, nt in recall_data[angle].values()
        ]
        box_data.append([v for v in vals if not np.isnan(v)])

    positions = list(range(len(angles)))
    bp = ax.boxplot(
        box_data, positions=positions, patch_artist=True,
        widths=0.5,
        medianprops={'color': 'black', 'linewidth': 1.2},
        whiskerprops={'linewidth': 0.8},
        capprops={'linewidth': 0.8},
        flierprops={'marker': 'x', 'markersize': 4, 'markeredgewidth': 0.8},
    )
    for patch, color in zip(bp['boxes'], ROBOT_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    # Overlay individual pair recall dots with jitter
    rng = np.random.default_rng(0)
    for xi, angle in zip(positions, angles):
        vals = [
            nd / nt if nt > 0 else float('nan')
            for nd, nt in recall_data[angle].values()
        ]
        vals = [v for v in vals if not np.isnan(v)]
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), xi) + jitter, vals,
                   color='#333333', s=10, zorder=5, alpha=0.8, linewidths=0)

    # Annotate overall recall above each box
    for xi, angle in zip(positions, angles):
        pair_counts = recall_data[angle]
        total_det = sum(v[0] for v in pair_counts.values())
        total_gt  = sum(v[1] for v in pair_counts.values())
        overall   = total_det / total_gt if total_gt > 0 else float('nan')
        ax.text(xi, 1.03, f'{overall:.2f}',
                ha='center', va='bottom', fontsize=6, color='#333333')

    ax.set_xticks(positions)
    ax.set_xticklabels([f'{a}°' for a in angles])
    ax.set_xlabel('GT Rotation Threshold')
    ax.set_ylabel('Recall (per robot pair)')
    ax.set_ylim(-0.05, 1.12)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.3)
    plt.tight_layout()

    save_fig(fig, exp_dir / 'loops_recall')
    plt.close(fig)
    print(f'Plot  → {exp_dir}/loops_recall.pdf')


if __name__ == '__main__':
    main()
