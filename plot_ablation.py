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
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils.plot import IEEE_RC, ROBOT_COLORS, save_fig, mark_endpoint

COLORS = ROBOT_COLORS

KNOWN_TYPES = ("bandwidth", "loops")


def _short_label(stem: str, type_name: str) -> str:
    """Strip the trailing _<type> suffix to get a human-readable variant label."""
    suffix = f"_{type_name}"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def group_npy_files(folder: Path) -> dict[str, list[Path]]:
    """
    Return {type_name: [path, ...]} for all known-type .npy files.
    Scans both the main folder and baselines/<folder.name>/ if it exists.
    """
    groups: dict[str, list[Path]] = defaultdict(list)

    search_dirs = [folder]
    baseline_dir = folder.parent / "baselines" / folder.name
    if baseline_dir.exists():
        search_dirs.append(baseline_dir)

    for search_dir in search_dirs:
        for p in sorted(search_dir.glob("*.npy")):
            for t in KNOWN_TYPES:
                if p.stem.endswith(f"_{t}"):
                    groups[t].append(p)
                    break
    return groups


def _load_loops(path: Path) -> dict | None:
    """
    Load a loops .npy file and normalise to {t_sec, pr, gv}.

    Supports two schemas:
      - main system:  pr_total / gv_total
      - Kimera-Multi: bow_matches / num_loop_closures
    """
    d = np.load(path, allow_pickle=True).item()
    if "pr_total" in d and "gv_total" in d:
        return {"t_sec": d["t_sec"], "pr": d["pr_total"], "gv": d["gv_total"]}
    if "bow_matches" in d and "num_loop_closures" in d:
        return {"t_sec": d["t_sec"], "pr": d["bow_matches"], "gv": d["num_loop_closures"]}
    return None


def _load_bandwidth(path: Path) -> dict | None:
    """Load a bandwidth .npy file and normalise to {t_sec, total}.
    CBS (backend) bandwidth is excluded so all methods are comparable."""
    d = np.load(path, allow_pickle=True).item()
    if "bow_MB" in d and "vlc_MB" in d:
        return {"t_sec": d["t_sec"], "total": d["bow_MB"] + d["vlc_MB"]}
    return None


# ---------------------------------------------------------------------------
# Per-type comparison plots
# ---------------------------------------------------------------------------

def plot_bandwidth_comparison(paths: list[Path], folder: Path) -> None:
    """
    Overlay cumulative BoW + VLC bandwidth for each variant (CBS excluded).
    All variants are truncated to the shortest experiment duration.
    """
    datasets = [(p, _load_bandwidth(p)) for p in paths]
    datasets = [(p, d) for p, d in datasets if d is not None]
    if not datasets:
        print("  No parseable bandwidth data — skipping.")
        return
    t_end = min(d["t_sec"][-1] for _, d in datasets)

    plt.rcParams.update(IEEE_RC)
    fig, ax = plt.subplots()

    for i, (p, d) in enumerate(datasets):
        t_sec = d["t_sec"]
        mask  = t_sec <= t_end
        t_m, v_m = t_sec[mask], d["total"][mask]
        label = _short_label(p.stem, "bandwidth")
        color = COLORS[i % len(COLORS)]
        ax.plot(t_m, v_m, color=color, label=label)
        mark_endpoint(ax, t_m, v_m, color, fmt="{:.1f}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative Bandwidth (MB)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.3)
    ax.annotate("* CBS backend bandwidth not included",
                xy=(0.99, 0.02), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=6, color="gray",
                style="italic")
    plt.tight_layout()

    save_fig(fig, folder / "bandwidth_comparison")

    # Log-scale version
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.3)
    save_fig(fig, folder / "bandwidth_comparison_log")

    plt.close(fig)


def plot_loops_comparison(paths: list[Path], folder: Path) -> None:
    """
    Two-panel plot:
      top:    PR (solid) and GV (dashed) counts per variant
      bottom: GV/PR ratio per variant (0 where PR == 0)
    All variants are truncated to the shortest experiment duration.
    Supports both main-system (pr_total/gv_total) and baseline
    (bow_matches/num_loop_closures) schemas.
    """
    datasets = [(p, _load_loops(p)) for p in paths]
    datasets = [(p, d) for p, d in datasets if d is not None]
    if not datasets:
        print("  No parseable loops data — skipping.")
        return
    t_end = min(d["t_sec"][-1] for _, d in datasets)

    plt.rcParams.update(IEEE_RC)
    fig, (ax_counts, ax_ratio) = plt.subplots(
        2, 1, sharex=True, figsize=(3.5, 3.5),
    )
    fig.subplots_adjust(hspace=0.08)

    for i, (p, d) in enumerate(datasets):
        t_sec = d["t_sec"]
        mask  = t_sec <= t_end
        label = _short_label(p.stem, "loops")
        color = COLORS[i % len(COLORS)]

        pr = d["pr"][mask]
        gv = d["gv"][mask]
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(pr > 0, gv / pr, 0.0)

        t_m = t_sec[mask]
        ax_counts.plot(t_m, pr,    color=color, linestyle="-",  label=f"{label} PR")
        ax_counts.plot(t_m, gv,    color=color, linestyle="--", label=f"{label} GV")
        ax_ratio.plot( t_m, ratio, color=color, label=label)
        mark_endpoint(ax_counts, t_m, pr,    color, fmt="{:.0f}")
        mark_endpoint(ax_counts, t_m, gv,    color, fmt="{:.0f}")
        mark_endpoint(ax_ratio,  t_m, ratio, color, fmt="{:.2f}")

    ax_counts.set_ylabel("Count")
    ax_counts.legend(loc="upper left")
    ax_counts.grid(True, alpha=0.3, linestyle="--", linewidth=0.3)

    ax_ratio.set_xlabel("Time (s)")
    ax_ratio.set_ylabel("GV / PR Ratio")
    ax_ratio.legend(loc="upper left")
    ax_ratio.grid(True, alpha=0.3, linestyle="--", linewidth=0.3)

    plt.tight_layout()

    save_fig(fig, folder / "loops_comparison")

    # Log-scale version: both panels
    ax_counts.set_yscale("log")
    ax_ratio.set_yscale("log")
    ax_counts.grid(True, alpha=0.3, linestyle="--", linewidth=0.3)
    ax_ratio.grid(True, alpha=0.3, linestyle="--", linewidth=0.3)
    save_fig(fig, folder / "loops_comparison_log")

    plt.close(fig)


def group_recall_files(folder: Path) -> list[Path]:
    """Return all loops_recall.csv files for this experiment.

    Looks in:
      <folder>/loops_recall.csv            (main experiment)
      baselines/<folder.name>/*/loops_recall.csv  (one per baseline method)
    """
    paths: list[Path] = []
    main = folder / "loops_recall.csv"
    if main.exists():
        paths.append(main)
    baseline_dir = folder.parent / "baselines" / folder.name
    if baseline_dir.exists():
        for method_dir in sorted(baseline_dir.iterdir()):
            p = method_dir / "loops_recall.csv"
            if p.exists():
                paths.append(p)
    return paths


def _load_recall(path: Path) -> dict | None:
    """Load a loops_recall.csv and return {buckets, recalls, label}.

    Supports bucket format (bucket_min, bucket_max columns).
    Overall recall per bucket is computed by summing n_detected and n_total
    across all robot pairs.
    """
    agg: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                bmin = int(row["bucket_min"])
                bmax = int(row["bucket_max"])
                agg[(bmin, bmax)][0] += int(row["n_detected"])
                agg[(bmin, bmax)][1] += int(row["n_total"])
            except (KeyError, ValueError):
                continue
    if not agg:
        return None
    bucket_keys = sorted(agg.keys())
    recalls = [agg[b][0] / agg[b][1] if agg[b][1] > 0 else 0.0 for b in bucket_keys]
    labels  = [f"{bmin}-{bmax}°" for bmin, bmax in bucket_keys]
    return {"bucket_keys": bucket_keys, "labels": labels,
            "recalls": recalls, "label": path.parent.name}


def plot_recall_comparison(paths: list[Path], folder: Path) -> None:
    """Grouped bar chart of per-bucket recall for each variant."""
    datasets = [(p, _load_recall(p)) for p in paths]
    datasets = [(p, d) for p, d in datasets if d is not None]
    if not datasets:
        print("  No parseable recall data — skipping.")
        return

    # All datasets must share the same bucket structure
    bucket_labels = datasets[0][1]["labels"]
    n_buckets  = len(bucket_labels)
    n_variants = len(datasets)
    width = 0.8 / n_variants

    plt.rcParams.update({**IEEE_RC, "figure.figsize": (3.5, 2.8)})
    fig, ax = plt.subplots()

    for i, (p, d) in enumerate(datasets):
        color  = COLORS[i % len(COLORS)]
        label  = "CoDE-SLAM" if d["label"] == folder.name else d["label"]
        offset = (i - (n_variants - 1) / 2) * width
        xs = [j + offset for j in range(n_buckets)]
        ax.bar(xs, d["recalls"], width=width * 0.9, color=color,
               alpha=0.85, label=label)

    ax.set_xticks(list(range(n_buckets)))
    ax.set_xticklabels(bucket_labels, rotation=45, ha="right")
    ax.set_xlabel("GT Rotation Bucket")
    ax.set_ylabel("Recall")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.3)
    plt.tight_layout()

    save_fig(fig, folder / "recall_comparison")
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

    # Recall comparison (loops_recall.csv files)
    recall_paths = group_recall_files(folder)
    if len(recall_paths) >= 2:
        print(f"\n[recall] {len(recall_paths)} variant(s):")
        for p in recall_paths:
            print(f"  {p}")
        plot_recall_comparison(recall_paths, folder)
    elif recall_paths:
        print(f"\n[recall] only one variant found — skipping comparison plot")


if __name__ == "__main__":
    main()
