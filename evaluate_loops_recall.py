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

import matplotlib.pyplot as plt
import yaml

from utils.io import load_keyframes_csv, load_loop_closures_csv
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
            loops.append({'name1': n1, 't1_s': t1, 'name2': n2, 't2_s': t2})

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


# ---------------------------------------------------------------------------
# Per-variant evaluation
# ---------------------------------------------------------------------------

def evaluate_one(
    variant_dir: Path,
    buckets: dict[tuple[int, int], list[dict]],
    bucket_keys: list[tuple[int, int]],
    tol_s: float,
) -> dict | None:
    """Evaluate recall for one variant dir. Saves CSV/txt.

    Returns {label, xs, recalls} for plotting, or None on failure.
    """
    id_to_name = discover_robots(variant_dir)
    if not id_to_name:
        print(f'  [SKIP] No robots found in {variant_dir.name}')
        return None
    print(f'Robots: { {v: k for k, v in id_to_name.items()} }')

    detected = load_detected_loops(variant_dir, id_to_name)
    print(f'Detected loops (unique, resolved): {len(detected)}')

    bucket_recall: dict[tuple[int, int], dict] = {}
    for bk in bucket_keys:
        bucket_recall[bk] = compute_recall(detected, buckets[bk], tol_s)

    # Stats + CSV
    stats_path = variant_dir / 'loops_recall.txt'
    csv_rows: list[dict] = []
    xs = [bmax for _, bmax in bucket_keys]
    recalls = []

    with open(stats_path, 'w') as f:
        f.write(f'Loop Closure Recall — {variant_dir.name}\n')
        f.write(f'Time tolerance : {tol_s} s\n')
        f.write('=' * 60 + '\n\n')
        for bk in bucket_keys:
            bmin, bmax = bk
            pair_counts = bucket_recall[bk]
            total_det = sum(v[0] for v in pair_counts.values())
            total_gt  = sum(v[1] for v in pair_counts.values())
            overall   = total_det / total_gt if total_gt > 0 else float('nan')
            label = f'{bmin}-{bmax}°'
            f.write(f'bucket {label:8s}  overall recall: '
                    f'{overall:.3f}  ({total_det}/{total_gt})\n')
            print(f'bucket {label:8s}  overall recall: '
                  f'{overall:.3f}  ({total_det}/{total_gt})')
            recalls.append(total_det / total_gt if total_gt > 0 else 0.0)
            for (ra, rb), (nd, nt) in sorted(pair_counts.items()):
                r = nd / nt if nt > 0 else float('nan')
                f.write(f'  {ra} <-> {rb}: {r:.3f}  ({nd}/{nt})\n')
                csv_rows.append({
                    'bucket_min': bmin, 'bucket_max': bmax,
                    'pair': f'{ra}<->{rb}',
                    'n_detected': nd, 'n_total': nt,
                    'recall': f'{r:.4f}',
                })
            f.write('\n')

    print(f'Stats → {stats_path}')

    csv_path = variant_dir / 'loops_recall.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['bucket_min', 'bucket_max', 'pair', 'n_detected', 'n_total', 'recall'])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f'CSV   → {csv_path}')

    return {'label': variant_dir.name, 'xs': xs, 'recalls': recalls}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_single(data: dict, out_dir: Path) -> None:
    """Single-variant recall curve."""
    plt.rcParams.update({**IEEE_RC, 'figure.figsize': (3.5, 2.8)})
    fig, ax = plt.subplots()
    ax.plot(data['xs'], data['recalls'], color='#4C72B0', marker='o', markersize=3)
    ax.set_xticks(data['xs'])
    ax.set_xticklabels([f'{x}°' for x in data['xs']])
    ax.set_xlabel('GT Rotation Bucket Upper Bound')
    ax.set_ylabel('Recall')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.3)
    plt.tight_layout()
    save_fig(fig, out_dir / 'loops_recall')
    plt.close(fig)
    print(f'Plot  → {out_dir}/loops_recall.pdf')


def plot_comparison(variants: list[dict], out_dir: Path) -> None:
    """Multi-variant recall comparison."""
    plt.rcParams.update({**IEEE_RC, 'figure.figsize': (3.5, 2.8)})
    fig, ax = plt.subplots()
    for i, d in enumerate(variants):
        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        ax.plot(d['xs'], d['recalls'], color=color, marker='o', markersize=3, label=d['label'])
    xs = variants[0]['xs']
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{x}°' for x in xs])
    ax.set_xlabel('GT Rotation Bucket Upper Bound')
    ax.set_ylabel('Recall')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.3)
    plt.tight_layout()
    save_fig(fig, out_dir / 'recall_comparison')
    plt.close(fig)
    print(f'Comparison → {out_dir}/recall_comparison.pdf')


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
    variants = discover_variants(exp_dir)
    if variants:
        print(f'\nFound {len(variants)} variant(s): {[v.name for v in variants]}\n')
        results = []
        for v in variants:
            print(f'--- {v.name} ---')
            r = evaluate_one(v, buckets, bucket_keys, args.tol)
            if r:
                results.append(r)
            print()
        if len(results) >= 2:
            plot_comparison(results, exp_dir)
        elif results:
            plot_single(results[0], exp_dir)
    else:
        # Single-experiment mode (backward compatible)
        r = evaluate_one(exp_dir, buckets, bucket_keys, args.tol)
        if r:
            plot_single(r, exp_dir)


if __name__ == '__main__':
    main()
