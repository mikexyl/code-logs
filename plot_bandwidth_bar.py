#!/usr/bin/env python3
"""
Stacked bar chart of bandwidth usage (BoW / VLC / CBS / DPGO) across experiment variants.

For our method variants: stacks are BoW + VLC + CBS.
For Kimera-Multi: stacks are BoW + VLC + DPGO (read from dpgo_log_*.csv files).

Usage:
    python3 plot_bandwidth_bar.py <exp_folder> [--variants v1 v2 ...]
    python3 plot_bandwidth_bar.py campus --variants ns ns-as mixvpr ns-ncs-nproj Kimera-Multi
"""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))
from utils.io import load_variant_aliases
from utils.plot import IEEE_RC, save_fig

# Colors for each bandwidth module
MODULE_COLORS = {
    'bow_MB':  '#4C72B0',   # blue   — BoW
    'vlc_MB':  '#DD8452',   # orange — VLC
    'cbs_MB':  '#55A868',   # green  — DPGO (our CBS backend)
    'dpgo_MB': '#55A868',   # green  — DPGO (Kimera-Multi backend, same color)
}
MODULE_LABELS = {
    'bow_MB':  'Place Recognition',
    'vlc_MB':  'Geometric Verification',
    'cbs_MB':  'DPGO',
    'dpgo_MB': 'DPGO',
}
MODULE_ORDER = ['bow_MB', 'vlc_MB', 'cbs_MB', 'dpgo_MB']

# Variants whose backend bandwidth comes from dpgo_log files instead of npy cbs_MB
DPGO_VARIANTS = {'Kimera-Multi'}


def find_bandwidth_npy(exp_dir: Path, exp_name: str, variant: str) -> Path | None:
    name_tries = [variant]
    if variant == 'ns':
        name_tries.append('no-scoring')
    for name in name_tries:
        stem = f"{exp_name}-{name}_bandwidth.npy"
        for p in [
            exp_dir / stem,
            exp_dir / variant / stem,
            Path(__file__).parent / 'baselines' / exp_name / stem,
            Path(__file__).parent / 'baselines' / exp_name / variant / stem,
        ]:
            if p.exists():
                return p
    return None


def load_kimera_dpgo_mb(variant_dir: Path) -> float:
    """Sum bytes_received (last row per robot) across all dpgo_log_*.csv files."""
    total = 0
    for robot_dir in sorted(variant_dir.iterdir()):
        dist_dir = robot_dir / 'distributed'
        if not dist_dir.is_dir():
            continue
        logs = sorted(dist_dir.glob('dpgo_log_*.csv'),
                      key=lambda p: int(p.stem.split('_')[-1]))
        if not logs:
            continue
        last_bytes = 0
        with open(logs[-1]) as f:
            for row in csv.DictReader(f):
                row = {k.strip(): v.strip() for k, v in row.items() if k and v}
                try:
                    last_bytes = int(row['bytes_received'])
                except (KeyError, ValueError):
                    continue
        total += last_bytes
    return total / 1e6


def load_totals(npy_path: Path, variant: str, baselines_dir: Path) -> dict[str, float]:
    """Return {module: total_MB} for a variant."""
    d = np.load(npy_path, allow_pickle=True).item()
    totals = {m: float(d[m][-1]) if len(d.get(m, [])) else 0.0
              for m in ['bow_MB', 'vlc_MB', 'cbs_MB']}
    totals['dpgo_MB'] = 0.0

    if variant in DPGO_VARIANTS:
        # Move cbs_MB → dpgo_MB (it's 0 for Kimera-Multi anyway), then load real DPGO bytes
        totals['cbs_MB'] = 0.0
        variant_dir = baselines_dir / variant
        if variant_dir.exists():
            totals['dpgo_MB'] = load_kimera_dpgo_mb(variant_dir)

    return totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exp_folder', help='Experiment folder (e.g. campus)')
    ap.add_argument('--variants', nargs='+', default=None)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    exp_dir = Path(args.exp_folder)
    if not exp_dir.is_absolute():
        exp_dir = Path(__file__).parent / exp_dir
    exp_name = exp_dir.name
    baselines_dir = Path(__file__).parent / 'baselines' / exp_name

    aliases = load_variant_aliases()

    if args.variants:
        requested = args.variants
    elif aliases:
        requested = list(aliases.keys())
    else:
        requested = ['ns', 'ns-as', 'mixvpr', 'ns-ncs-nproj', 'Kimera-Multi']

    if aliases:
        alias_order = {k: i for i, k in enumerate(aliases.keys())}
        variant_order = sorted(requested, key=lambda v: alias_order.get(v, len(alias_order)))
    else:
        variant_order = requested

    # Load data
    data = {}
    for v in variant_order:
        p = find_bandwidth_npy(exp_dir, exp_name, v)
        if p is None:
            print(f"Warning: no bandwidth npy found for '{v}', skipping")
            continue
        data[v] = load_totals(p, v, baselines_dir)
        print(f"  {v}: bow={data[v]['bow_MB']:.2f}  vlc={data[v]['vlc_MB']:.2f}  "
              f"cbs={data[v]['cbs_MB']:.2f}  dpgo={data[v]['dpgo_MB']:.2f}  "
              f"total={sum(data[v].values()):.2f} MB  [{p.name}]")

    if not data:
        print("No data found.")
        return

    variants = list(data.keys())
    labels = [aliases.get(v, v) for v in variants]

    # Only include modules that have non-zero values somewhere
    active_modules = [m for m in MODULE_ORDER
                      if any(data[v][m] > 0 for v in variants)]

    plt.rcParams.update({**IEEE_RC, 'figure.figsize': (3.5, 2.2)})
    fig, ax = plt.subplots()

    x = np.arange(len(variants))
    bar_width = 0.55
    bottoms = np.zeros(len(variants))

    handles = []
    seen_labels = set()
    for module in active_modules:
        vals = np.array([data[v][module] for v in variants])
        ax.bar(x, vals, bar_width, bottom=bottoms,
               color=MODULE_COLORS[module])
        bottoms += vals
        lbl = MODULE_LABELS[module]
        if lbl not in seen_labels:
            handles.append(mpatches.Patch(color=MODULE_COLORS[module], label=lbl))
            seen_labels.add(lbl)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=7)
    ax.set_ylabel('Bandwidth (MB)')
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    fig.legend(handles=handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.0),
               ncol=len(handles), framealpha=0.85,
               fontsize=6, handlelength=1.0,
               handletextpad=0.4, columnspacing=1.0)
    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(top=0.86)
    out = Path(args.output) if args.output else exp_dir / 'bandwidth_bar'
    save_fig(fig, out)
    plt.close(fig)


if __name__ == '__main__':
    main()
