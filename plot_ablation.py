#!/usr/bin/env python3
"""
Plot ablation comparisons from saved .npy data files.

Scans a folder for files matching *_<type>.npy (types: bandwidth, loops).
When multiple variants exist for the same type they are overlaid on one plot.

Usage:
    python plot_ablation.py <folder>
    python plot_ablation.py campus
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


IEEE_RC = {
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.figsize': (3.5, 3.5 * 3 / 4),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'patch.linewidth': 0.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

KNOWN_TYPES = ("bandwidth", "loops")


def _short_label(stem: str, type_name: str) -> str:
    """Strip the trailing _<type> suffix to get a human-readable variant label."""
    suffix = f"_{type_name}"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def group_npy_files(folder: Path) -> dict[str, list[Path]]:
    """Return {type_name: [path, ...]} for all known-type .npy files in folder."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(folder.glob("*.npy")):
        for t in KNOWN_TYPES:
            if p.stem.endswith(f"_{t}"):
                groups[t].append(p)
                break
    return groups


# ---------------------------------------------------------------------------
# Per-type comparison plots
# ---------------------------------------------------------------------------

def plot_bandwidth_comparison(paths: list[Path], folder: Path) -> None:
    """
    Overlay total cumulative bandwidth (bow + vlc + cbs) for each variant.
    All variants are truncated to the shortest experiment duration.
    """
    datasets = [(np.load(p, allow_pickle=True).item(), p) for p in paths]
    t_end = min(d["t_sec"][-1] for d, _ in datasets)

    plt.rcParams.update(IEEE_RC)
    fig, ax = plt.subplots()

    for i, (d, p) in enumerate(datasets):
        t_sec = d["t_sec"]
        mask  = t_sec <= t_end
        total = d["bow_MB"] + d["vlc_MB"] + d["cbs_MB"]
        label = _short_label(p.stem, "bandwidth")
        color = COLORS[i % len(COLORS)]
        ax.plot(t_sec[mask], total[mask], color=color, label=label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative Bandwidth (MB)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.3)
    plt.tight_layout()

    for suffix in (".pdf", ".png"):
        out = folder / f"bandwidth_comparison{suffix}"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"Saved to {out}")
    plt.close(fig)


def plot_loops_comparison(paths: list[Path], folder: Path) -> None:
    """
    Two-panel plot:
      top:    PR (solid) and GV (dashed) counts per variant
      bottom: GV/PR ratio per variant (0 where PR == 0)
    All variants are truncated to the shortest experiment duration.
    """
    datasets = [(np.load(p, allow_pickle=True).item(), p) for p in paths]
    t_end = min(d["t_sec"][-1] for d, _ in datasets)

    plt.rcParams.update(IEEE_RC)
    fig, (ax_counts, ax_ratio) = plt.subplots(
        2, 1, sharex=True, figsize=(3.5, 3.5),
    )
    fig.subplots_adjust(hspace=0.08)

    for i, (d, p) in enumerate(datasets):
        t_sec = d["t_sec"]
        mask  = t_sec <= t_end
        label = _short_label(p.stem, "loops")
        color = COLORS[i % len(COLORS)]

        pr = d["pr_total"][mask]
        gv = d["gv_total"][mask]
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(pr > 0, gv / pr, 0.0)

        ax_counts.plot(t_sec[mask], pr, color=color, linestyle="-",  label=f"{label} PR")
        ax_counts.plot(t_sec[mask], gv, color=color, linestyle="--", label=f"{label} GV")
        ax_ratio.plot(t_sec[mask], ratio, color=color, label=label)

    ax_counts.set_ylabel("Count")
    ax_counts.legend(loc="upper left")
    ax_counts.grid(True, alpha=0.3, linestyle="--", linewidth=0.3)

    ax_ratio.set_xlabel("Time (s)")
    ax_ratio.set_ylabel("GV / PR Ratio")
    ax_ratio.legend(loc="upper left")
    ax_ratio.grid(True, alpha=0.3, linestyle="--", linewidth=0.3)

    plt.tight_layout()

    for suffix in (".pdf", ".png"):
        out = folder / f"loops_comparison{suffix}"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"Saved to {out}")
    plt.close(fig)


PLOTTERS = {
    "bandwidth": plot_bandwidth_comparison,
    "loops":     plot_loops_comparison,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ablation comparisons from saved *_bandwidth.npy / *_loops.npy files."
    )
    parser.add_argument("folder", type=Path,
                        help="Folder to scan for .npy data files")
    args = parser.parse_args()

    folder: Path = args.folder.resolve()
    if not folder.exists():
        print(f"Error: {folder} does not exist.")
        raise SystemExit(1)

    groups = group_npy_files(folder)
    if not groups:
        print(f"No *_bandwidth.npy or *_loops.npy files found in {folder}")
        return

    for type_name, paths in sorted(groups.items()):
        print(f"\n[{type_name}] {len(paths)} variant(s):")
        for p in paths:
            print(f"  {p.name}")
        if len(paths) < 2:
            print("  (only one variant — skipping comparison plot)")
            continue
        PLOTTERS[type_name](paths, folder)


if __name__ == "__main__":
    main()
