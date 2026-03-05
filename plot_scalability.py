#!/usr/bin/env python3
"""
Scalability / Pareto-front scatter plot.

X-axis: Total BoW + VLC communication bandwidth (MB)   [lower is better]
Y-axis: Number of verified inlier loop closures         [higher is better]
        OR algebraic connectivity λ₂                   (--metric ac)

Usage:
    python plot_scalability.py campus
    python plot_scalability.py gate
    python plot_scalability.py campus --bucket 20
    python plot_scalability.py campus --inlier-recall
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils.plot import IEEE_RC, ROBOT_COLORS, save_fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_bandwidth_mb(npy_path: Path) -> float:
    """Return final cumulative BoW + VLC bandwidth in MB."""
    d = np.load(npy_path, allow_pickle=True).item()
    return float(d['bow_MB'][-1]) + float(d['vlc_MB'][-1])


def _load_recall(recall_csv: Path, bucket_max: int, inlier: bool) -> float | None:
    """Return aggregate recall for the 0–bucket_max° rotation bucket.

    If inlier=True, uses n_inlier / n_total (inlier recall).
    Otherwise uses n_detected / n_total (overall recall).
    Returns None if the CSV does not exist or bucket not found.
    """
    if not recall_csv.exists():
        return None
    n_hit = 0
    n_total = 0
    with open(recall_csv) as f:
        for row in csv.DictReader(f):
            if int(row['bucket_min']) == 0 and int(row['bucket_max']) == bucket_max:
                n_hit   += int(row['n_inlier']) if inlier else int(row['n_detected'])
                n_total += int(row['n_total'])
    if n_total == 0:
        return None
    return n_hit / n_total


def _pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return Pareto-optimal points (min x, max y). Sorted by x ascending."""
    pts = sorted(points, key=lambda p: p[0])
    front: list[tuple[float, float]] = []
    best_y = -float('inf')
    for x, y in pts:
        if y > best_y:
            front.append((x, y))
            best_y = y
    return front


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Scalability / Pareto-front scatter: bandwidth vs inlier loops.'
    )
    parser.add_argument('folder', type=Path)
    parser.add_argument('--bucket', type=int, default=10,
                        help='GT rotation bucket max in degrees for recall (default: 10)')
    parser.add_argument('--inlier-recall', action='store_true', dest='inlier_recall',
                        help='Use inlier recall instead of overall recall on Y-axis')
    args = parser.parse_args()

    folder = args.folder.resolve()
    exp = folder.name

    # ------------------------------------------------------------------
    # Collect data for each variant / baseline
    # ------------------------------------------------------------------
    # bandwidth npy pattern: <folder>/<exp>-<variant>_bandwidth.npy
    # inlier_loops.csv:      <folder>/<variant>/inlier_loops.csv
    # baselines:             baselines/<exp>/<method>/inlier_loops.csv
    #                        baselines/<exp>/<exp>-<method>_bandwidth.npy

    entries: list[dict] = []   # {label, bw_mb, metric_val, is_baseline}

    def _add_entry(label: str, bw_npy: Path, recall_csv: Path, is_baseline: bool) -> None:
        if not bw_npy.exists():
            return
        bw  = _load_bandwidth_mb(bw_npy)
        val = _load_recall(recall_csv, args.bucket, args.inlier_recall)
        if val is None:
            print(f'  [{label}] No recall data — skipping.')
            return
        entries.append({'label': label, 'bw': bw, 'val': val, 'is_baseline': is_baseline})

    # Variants
    for bw_npy in sorted(folder.glob(f'{exp}-*_bandwidth.npy')):
        variant = bw_npy.stem.replace(f'{exp}-', '').replace('_bandwidth', '')
        recall_csv = folder / variant / 'loops_recall.csv'
        _add_entry(variant, bw_npy, recall_csv, is_baseline=False)

    # Baselines
    baseline_root = folder.parent / 'baselines' / exp
    if baseline_root.exists():
        for bw_npy in sorted(baseline_root.glob(f'{exp}-*_bandwidth.npy')):
            method = bw_npy.stem.replace(f'{exp}-', '').replace('_bandwidth', '')
            recall_csv = baseline_root / method / 'loops_recall.csv'
            _add_entry(method, bw_npy, recall_csv, is_baseline=True)

    if not entries:
        print('No data found.')
        return

    recall_type = 'inlier recall' if args.inlier_recall else 'recall'
    for e in sorted(entries, key=lambda x: x['bw']):
        print(f"  {e['label']:20s}  bw={e['bw']:.1f} MB  "
              f"{recall_type}@{args.bucket}°={e['val']:.3f}"
              f"  {'[baseline]' if e['is_baseline'] else ''}")

    # ------------------------------------------------------------------
    # Pareto front
    # ------------------------------------------------------------------
    all_pts = [(e['bw'], e['val']) for e in entries]
    front   = _pareto_front(all_pts)

    # Extend front to plot edges
    front_xs = [p[0] for p in front]
    front_ys = [p[1] for p in front]

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    n_variants  = sum(1 for e in entries if not e['is_baseline'])
    n_baselines = sum(1 for e in entries if e['is_baseline'])
    n_total     = len(entries)

    plt.rcParams.update({**IEEE_RC, 'figure.figsize': (3.5, 2.8)})
    fig, ax = plt.subplots()

    # Shade "good" quadrant (top-left) relative to worst baseline or worst variant
    worst_bw  = max(e['bw']  for e in entries)
    best_val  = max(e['val'] for e in entries)

    # Draw Pareto staircase
    # Build staircase: for each consecutive pair of front points, draw horizontal then vertical
    step_xs: list[float] = []
    step_ys: list[float] = []
    for i, (x, y) in enumerate(front):
        if i == 0:
            step_xs.append(x)
            step_ys.append(y)
        else:
            # horizontal from previous x to this x at previous y
            step_xs.append(x)
            step_ys.append(step_ys[-1])
            # vertical to this y
            step_xs.append(x)
            step_ys.append(y)

    ax.plot(step_xs, step_ys, color='#2ecc71', linewidth=0.8, linestyle='--',
            alpha=0.7, zorder=1, label='Pareto front')

    # Per-label nudge offsets (dx, dy) in data units — avoids crowding
    # Populated with manual offsets for known overlapping labels
    bw_span  = worst_bw * 1.15
    val_span = best_val * 1.15
    label_offsets: dict[str, tuple[float, float]] = {
        'ns-cs':      ( bw_span * 0.04,  val_span *  0.04),
        'no-scoring': ( bw_span * 0.04,  val_span * -0.06),
        'Kimera-Multi': ( bw_span * 0.00, val_span * -0.07),
    }
    default_offset = (bw_span * 0.02, val_span * 0.04)

    # Scatter points
    for i, e in enumerate(entries):
        color  = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        marker = 's' if e['is_baseline'] else 'o'
        size   = 36 if e['is_baseline'] else 40
        ax.scatter(e['bw'], e['val'], color=color, marker=marker,
                   s=size, zorder=4,
                   edgecolors='black' if e['is_baseline'] else 'none',
                   linewidths=0.6)

        dx, dy = label_offsets.get(e['label'], default_offset)
        ax.annotate(
            e['label'],
            xy=(e['bw'], e['val']),
            xytext=(e['bw'] + dx, e['val'] + dy),
            fontsize=4.5,
            arrowprops=dict(arrowstyle='-', color='#888888', lw=0.3),
            zorder=5,
        )

    # "Better" corner arrow
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.annotate('better →', xy=(bw_span * 0.05, val_span * 0.05),
                fontsize=4, color='#555555', style='italic',
                xytext=(bw_span * 0.05, val_span * 0.05))

    # Axes labels and formatting
    recall_label = ('Inlier Recall' if args.inlier_recall else 'Recall') + f' @{args.bucket}°'
    ax.set_xlabel('Total Comm. Bandwidth (MB)  [BoW + VLC, lower is better →]')
    ax.set_ylabel(recall_label + '  [higher is better ↑]')
    ax.set_title(f'{exp} — Bandwidth vs. Loop Closure Recall', fontsize=7)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.3)
    ax.set_xlim(left=0, right=bw_span)
    ax.set_ylim(bottom=0, top=val_span)

    # Legend: variant (circle) vs baseline (square)
    import matplotlib.lines as mlines
    handles = [mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                              markersize=4, label='variant'),
               mlines.Line2D([], [], color='gray', marker='s', linestyle='None',
                              markersize=4, markeredgecolor='black', markeredgewidth=0.5,
                              label='baseline')]
    if n_baselines > 0:
        ax.legend(handles=handles, fontsize=4.5, loc='lower right', framealpha=0.7)

    plt.tight_layout()
    out = folder / 'scalability'
    save_fig(fig, out)
    plt.close(fig)
    print(f'\nSaved to {out}.pdf / .png')


if __name__ == '__main__':
    main()
