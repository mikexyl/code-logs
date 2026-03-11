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

from utils.io import load_variant_aliases, apply_variant_alias
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

    Accumulates all rows with bucket_max <= the target (i.e. all ranges
    [0,10], [10,20], ... up to bucket_max).
    If inlier=True, uses n_inlier / n_total (inlier recall).
    Otherwise uses n_detected / n_total (overall recall).
    Returns None if the CSV does not exist or no matching rows found.
    """
    if not recall_csv.exists():
        return None
    n_hit = 0
    n_total = 0
    with open(recall_csv) as f:
        for row in csv.DictReader(f):
            if int(row['bucket_min']) >= 0 and int(row['bucket_max']) <= bucket_max:
                n_hit   += int(row['n_inlier']) if inlier else int(row['n_detected'])
                n_total += int(row['n_total'])
    if n_total == 0:
        return None
    return n_hit / n_total


def _count_inliers(inlier_csv: Path) -> int:
    """Count inlier loops from inlier_loops.csv (lines minus header)."""
    if not inlier_csv.exists():
        return 0
    return sum(1 for _ in open(inlier_csv)) - 1


def plot_yield(
    entries: list[dict],   # {label, bw, inliers, is_baseline}
    folder: Path,
    exp: str,
) -> None:
    """Bar chart of verified inliers per MB of bandwidth (True Positive Yield)."""
    # Sort by yield descending
    data = sorted(entries, key=lambda e: e['yield'], reverse=True)

    labels   = [e['label']  for e in data]
    yields   = [e['yield']  for e in data]
    is_bl    = [e['is_baseline'] for e in data]
    n        = len(data)

    _fw = max(3.5, n * 0.65)
    plt.rcParams.update({**IEEE_RC, 'figure.figsize': (_fw, _fw * 9 / 16)})
    fig, ax = plt.subplots()

    colors = [ROBOT_COLORS[i % len(ROBOT_COLORS)] for i in range(n)]
    xs = list(range(n))

    for i, (y, color, bl) in enumerate(zip(yields, colors, is_bl)):
        ax.bar(xs[i], y, color=color, width=0.6,
               edgecolor='black' if bl else 'none',
               linewidth=0.8,
               hatch='///' if bl else None)
        ax.text(xs[i], y + max(yields) * 0.02, f'{y:.2f}',
                ha='center', va='bottom', fontsize=5.5)

    # Annotate inliers and MB under each bar
    for i, e in enumerate(data):
        ax.text(xs[i], -max(yields) * 0.08,
                f"{e['inliers']}÷{e['bw']:.0f}",
                ha='center', va='top', fontsize=4, color='#555555')

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=6)
    ax.set_ylabel('True Positive Yield\n(Verified Inliers / MB)')
    ax.set_title(f'{exp} — Inlier Loop Closures per MB of Bandwidth', fontsize=7)
    ax.set_ylim(bottom=-max(yields) * 0.18)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.3)
    ax.axhline(0, color='black', linewidth=0.4)
    plt.tight_layout()

    out = folder / 'yield'
    save_fig(fig, out)
    plt.close(fig)
    print(f'Yield plot → {out}.pdf / .png')


def _load_ate_rmse(evo_zip: Path) -> float | None:
    """Return APE RMSE (m) from an evo_ape.zip."""
    if not evo_zip.exists():
        return None
    import zipfile, json
    with zipfile.ZipFile(evo_zip) as z:
        with z.open('stats.json') as f:
            return float(json.load(f)['rmse'])


def _pareto_front(points: list[tuple[float, float]],
                  minimize_y: bool = False) -> list[tuple[float, float]]:
    """Return Pareto-optimal points. Sorted by x ascending.

    minimize_y=False (default): min x, max y  (recall mode)
    minimize_y=True:            min x, min y  (ATE mode)
    """
    pts = sorted(points, key=lambda p: p[0])
    front: list[tuple[float, float]] = []
    if minimize_y:
        best_y = float('inf')
        for x, y in pts:
            if y < best_y:
                front.append((x, y))
                best_y = y
    else:
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
    parser.add_argument('--buckets', type=int, nargs='+', default=[10, 20, 30],
                        metavar='DEG',
                        help='GT rotation bucket max(es) in degrees for recall (default: 10 20 30)')
    parser.add_argument('--inlier-recall', action='store_true', dest='inlier_recall',
                        help='Use inlier recall instead of overall recall on Y-axis')
    parser.add_argument('--yield', action='store_true', dest='yield_plot',
                        help='Also produce a True Positive Yield bar chart (inliers/MB)')
    parser.add_argument('--ate', action='store_true',
                        help='Also produce a bandwidth vs ATE scatter plot')
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

    aliases = load_variant_aliases()
    entries: list[dict] = []   # {label, bw, vals: {bucket: float}, is_baseline}

    def _add_entry(raw: str, bw_npy: Path, recall_csv: Path, is_baseline: bool) -> None:
        disp = apply_variant_alias(aliases, raw)
        if disp is None:
            return
        if not bw_npy.exists():
            return
        bw = _load_bandwidth_mb(bw_npy)
        vals: dict[int, float] = {}
        for b in args.buckets:
            v = _load_recall(recall_csv, b, args.inlier_recall)
            if v is not None:
                vals[b] = v
        if not vals:
            print(f'  [{raw}] No recall data — skipping.')
            return
        entries.append({'label': disp, 'bw': bw, 'vals': vals, 'is_baseline': is_baseline})

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
        vals_str = '  '.join(f"{recall_type}@{b}°={e['vals'][b]:.3f}"
                             for b in args.buckets if b in e['vals'])
        print(f"  {e['label']:20s}  bw={e['bw']:.1f} MB  {vals_str}"
              f"  {'[baseline]' if e['is_baseline'] else ''}")

    # ------------------------------------------------------------------
    # Plot — one subplot per bucket
    # ------------------------------------------------------------------
    import matplotlib.lines as mlines

    n_baselines = sum(1 for e in entries if e['is_baseline'])
    buckets     = args.buckets
    n_panels    = len(buckets)
    recall_type = 'Inlier Recall' if args.inlier_recall else 'Recall'

    worst_bw = max(e['bw'] for e in entries)
    bw_span  = worst_bw * 1.15

    fig_w = max(3.5, n_panels * 3.0)
    plt.rcParams.update({**IEEE_RC, 'figure.figsize': (fig_w, fig_w * 9 / 16)})
    fig, axes = plt.subplots(1, n_panels, sharey=False)
    if n_panels == 1:
        axes = [axes]

    label_offsets: dict[str, tuple[float, float]] = {
        'ns-cs':        ( 0.04,  0.04),
        'no-scoring':   ( 0.04, -0.06),
        'Kimera-Multi': ( 0.00, -0.07),
    }

    for ax, bucket in zip(axes, buckets):
        bucket_entries = [e for e in entries if bucket in e['vals']]
        if not bucket_entries:
            ax.set_visible(False)
            continue

        best_val = max(e['vals'][bucket] for e in bucket_entries)
        val_span = best_val * 1.15

        # Pareto front
        all_pts = [(e['bw'], e['vals'][bucket]) for e in bucket_entries]
        front   = _pareto_front(all_pts)
        step_xs: list[float] = []
        step_ys: list[float] = []
        for i, (x, y) in enumerate(front):
            if i == 0:
                step_xs.append(x); step_ys.append(y)
            else:
                step_xs.append(x); step_ys.append(step_ys[-1])
                step_xs.append(x); step_ys.append(y)
        ax.plot(step_xs, step_ys, color='#2ecc71', linewidth=0.8,
                linestyle='--', alpha=0.7, zorder=1)

        for i, e in enumerate(entries):
            if bucket not in e['vals']:
                continue
            color  = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            marker = 's' if e['is_baseline'] else 'o'
            size   = 36 if e['is_baseline'] else 40
            val    = e['vals'][bucket]
            ax.scatter(e['bw'], val, color=color, marker=marker, s=size, zorder=4,
                       edgecolors='black' if e['is_baseline'] else 'none', linewidths=0.6)
            rel = label_offsets.get(e['label'], (0.02, 0.04))
            dx, dy = rel[0] * bw_span, rel[1] * val_span
            ax.annotate(e['label'], xy=(e['bw'], val),
                        xytext=(e['bw'] + dx, val + dy),
                        fontsize=4.5,
                        arrowprops=dict(arrowstyle='-', color='#888888', lw=0.3),
                        zorder=5)

        ax.annotate('better →', xy=(bw_span * 0.05, val_span * 0.05),
                    fontsize=4, color='#555555', style='italic',
                    xytext=(bw_span * 0.05, val_span * 0.05))

        ax.set_xlabel('Bandwidth (MB)', fontsize=5.5)
        ax.set_ylabel(f'{recall_type} @{bucket}°  [↑]', fontsize=5.5)
        ax.set_title(f'@{bucket}°', fontsize=6)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.3)
        ax.set_xlim(left=0, right=bw_span)
        ax.set_ylim(bottom=0, top=val_span)

    # Shared legend on last visible axis
    handles = [mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                              markersize=4, label='variant'),
               mlines.Line2D([], [], color='gray', marker='s', linestyle='None',
                              markersize=4, markeredgecolor='black',
                              markeredgewidth=0.5, label='baseline')]
    if n_baselines > 0:
        axes[-1].legend(handles=handles, fontsize=4.5, loc='lower right', framealpha=0.7)

    fig.suptitle(f'{exp} — Bandwidth vs. Loop Closure Recall', fontsize=7)
    plt.tight_layout()
    out = folder / 'scalability'
    save_fig(fig, out)
    plt.close(fig)
    print(f'\nSaved to {out}.pdf / .png')

    # ------------------------------------------------------------------
    # Yield plot (always produced, not gated on --yield flag)
    # ------------------------------------------------------------------
    yield_entries: list[dict] = []

    def _add_yield_entry(raw: str, bw_npy: Path, inlier_csv: Path,
                         is_baseline: bool) -> None:
        disp = apply_variant_alias(aliases, raw)
        if disp is None:
            return
        if not bw_npy.exists():
            return
        bw = _load_bandwidth_mb(bw_npy)
        n  = _count_inliers(inlier_csv)
        if bw <= 0:
            return
        yield_entries.append({'label': disp, 'bw': bw, 'inliers': n,
                               'yield': n / bw, 'is_baseline': is_baseline})

    for bw_npy in sorted(folder.glob(f'{exp}-*_bandwidth.npy')):
        variant = bw_npy.stem.replace(f'{exp}-', '').replace('_bandwidth', '')
        _add_yield_entry(variant, bw_npy, folder / variant / 'inlier_loops.csv',
                         is_baseline=False)

    if baseline_root.exists():
        for bw_npy in sorted(baseline_root.glob(f'{exp}-*_bandwidth.npy')):
            method = bw_npy.stem.replace(f'{exp}-', '').replace('_bandwidth', '')
            _add_yield_entry(method, bw_npy,
                             baseline_root / method / 'inlier_loops.csv',
                             is_baseline=True)

    if yield_entries:
        print()
        for e in sorted(yield_entries, key=lambda x: x['yield'], reverse=True):
            print(f"  {e['label']:20s}  {e['inliers']:3d} inliers / {e['bw']:.1f} MB"
                  f"  = {e['yield']:.2f} inliers/MB"
                  f"  {'[baseline]' if e['is_baseline'] else ''}")
        plot_yield(yield_entries, folder, exp)

    # ------------------------------------------------------------------
    # ATE scatter plot (--ate)
    # ------------------------------------------------------------------
    if args.ate:
        ate_entries: list[dict] = []

        def _add_ate_entry(raw: str, bw_npy: Path, evo_zip: Path,
                           is_baseline: bool) -> None:
            disp = apply_variant_alias(aliases, raw)
            if disp is None:
                return
            if not bw_npy.exists():
                return
            bw  = _load_bandwidth_mb(bw_npy)
            ate = _load_ate_rmse(evo_zip)
            if ate is None:
                print(f'  [{raw}] No ATE data — skipping.')
                return
            ate_entries.append({'label': disp, 'bw': bw, 'ate': ate,
                                 'is_baseline': is_baseline})

        for bw_npy in sorted(folder.glob(f'{exp}-*_bandwidth.npy')):
            variant = bw_npy.stem.replace(f'{exp}-', '').replace('_bandwidth', '')
            _add_ate_entry(variant, bw_npy, folder / variant / 'evo_ape.zip',
                           is_baseline=False)
        if baseline_root.exists():
            for bw_npy in sorted(baseline_root.glob(f'{exp}-*_bandwidth.npy')):
                method = bw_npy.stem.replace(f'{exp}-', '').replace('_bandwidth', '')
                _add_ate_entry(method, bw_npy,
                               baseline_root / method / 'evo_ape.zip',
                               is_baseline=True)

        if ate_entries:
            print('\n--- ATE ---')
            for e in sorted(ate_entries, key=lambda x: x['ate']):
                print(f"  {e['label']:20s}  bw={e['bw']:.1f} MB  ATE RMSE={e['ate']:.3f} m"
                      f"  {'[baseline]' if e['is_baseline'] else ''}")

            import matplotlib.lines as _mlines

            n_bl_ate = sum(1 for e in ate_entries if e['is_baseline'])
            worst_bw_ate = max(e['bw']  for e in ate_entries)
            best_ate     = min(e['ate'] for e in ate_entries)
            worst_ate    = max(e['ate'] for e in ate_entries)
            bw_span_ate  = worst_bw_ate * 1.15
            ate_margin   = (worst_ate - best_ate) * 0.3
            ate_ymin     = best_ate  - ate_margin
            ate_ymax     = worst_ate + ate_margin
            ate_span     = ate_ymax - ate_ymin

            all_pts_ate = [(e['bw'], e['ate']) for e in ate_entries]
            front_ate   = _pareto_front(all_pts_ate, minimize_y=True)
            step_xs_ate: list[float] = []
            step_ys_ate: list[float] = []
            for i, (x, y) in enumerate(front_ate):
                if i == 0:
                    step_xs_ate.append(x); step_ys_ate.append(y)
                else:
                    step_xs_ate.append(x); step_ys_ate.append(step_ys_ate[-1])
                    step_xs_ate.append(x); step_ys_ate.append(y)

            plt.rcParams.update({**IEEE_RC, 'figure.figsize': (3.5, 2.0)})
            fig_ate, ax_ate = plt.subplots()

            ax_ate.plot(step_xs_ate, step_ys_ate, color='#2ecc71', linewidth=0.8,
                        linestyle='--', alpha=0.7, zorder=1, label='Pareto front')

            label_offsets_ate: dict[str, tuple[float, float]] = {
                'ns-cs':        ( 0.04,  0.04),
                'no-scoring':   ( 0.04, -0.06),
                'Kimera-Multi': ( 0.00, -0.07),
            }
            for i, e in enumerate(ate_entries):
                color  = ROBOT_COLORS[i % len(ROBOT_COLORS)]
                marker = 's' if e['is_baseline'] else 'o'
                size   = 36 if e['is_baseline'] else 40
                ax_ate.scatter(e['bw'], e['ate'], color=color, marker=marker,
                               s=size, zorder=4,
                               edgecolors='black' if e['is_baseline'] else 'none',
                               linewidths=0.6)
                rel = label_offsets_ate.get(e['label'], (0.02, 0.04))
                dx = rel[0] * bw_span_ate
                dy = rel[1] * ate_span
                ax_ate.annotate(e['label'], xy=(e['bw'], e['ate']),
                                xytext=(e['bw'] + dx, e['ate'] + dy),
                                fontsize=4.5,
                                arrowprops=dict(arrowstyle='-', color='#888888', lw=0.3),
                                zorder=5)

            ax_ate.annotate('← better', xy=(bw_span_ate * 0.75, ate_ymin + ate_span * 0.07),
                            fontsize=4, color='#555555', style='italic',
                            xytext=(bw_span_ate * 0.75, ate_ymin + ate_span * 0.07))

            ax_ate.set_xlabel('Total Comm. Bandwidth (MB)  [BoW + VLC, lower →]')
            ax_ate.set_ylabel('ATE RMSE (m)  [lower is better ↓]')
            ax_ate.set_title(f'{exp} — Bandwidth vs. ATE', fontsize=7)
            ax_ate.grid(True, alpha=0.25, linestyle='--', linewidth=0.3)
            ax_ate.set_xlim(left=0, right=bw_span_ate)
            ax_ate.set_ylim(bottom=ate_ymin, top=ate_ymax)

            handles_ate = [
                _mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                               markersize=4, label='variant'),
                _mlines.Line2D([], [], color='gray', marker='s', linestyle='None',
                               markersize=4, markeredgecolor='black',
                               markeredgewidth=0.5, label='baseline'),
            ]
            if n_bl_ate > 0:
                ax_ate.legend(handles=handles_ate, fontsize=4.5,
                              loc='upper right', framealpha=0.7)

            plt.tight_layout()
            out_ate = folder / 'scalability_ate'
            save_fig(fig_ate, out_ate)
            plt.close(fig_ate)
            print(f'ATE plot → {out_ate}.pdf / .png')


if __name__ == '__main__':
    main()
