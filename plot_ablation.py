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

from utils.io import load_variant_aliases, apply_variant_alias, is_robot_dir, discover_variants
from utils.plot import IEEE_RC, ROBOT_COLORS, save_fig, mark_endpoint

COLORS = ROBOT_COLORS

KNOWN_TYPES = ("bandwidth", "loops")


def _short_label(stem: str, type_name: str) -> str:
    """Strip the trailing _<type> suffix to get a human-readable variant label."""
    suffix = f"_{type_name}"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _variant_key(raw_label: str, folder_name: str) -> str:
    """Strip '<folder_name>-' prefix from a npy stem label to get the variant key.

    e.g. raw_label='campus-ns-as', folder_name='campus' → 'ns-as'
    Falls back to raw_label if prefix not present.
    """
    prefix = folder_name + "-"
    if raw_label.startswith(prefix):
        return raw_label[len(prefix):]
    return raw_label


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


def _load_inlier_counts(folder: Path) -> dict[str, int]:
    """Load inlier_counts.npy from folder (and baselines subfolder), keyed by variant name."""
    counts: dict[str, int] = {}
    for search_dir in [folder, folder.parent / "baselines" / folder.name]:
        p = search_dir / "inlier_counts.npy"
        if p.exists():
            try:
                counts.update(np.load(str(p), allow_pickle=True).item())
            except Exception:
                pass
    return counts


def _match_inlier_count(label: str, inlier_counts: dict[str, int]) -> int | None:
    """Find inlier count for a npy label by substring match against variant dir names.

    e.g. label="campus-all" matches key="all"; label="campus-ns-cs" matches key="ns-cs".
    Selects the longest matching key to avoid ambiguity.
    """
    best_key = max(
        (k for k in inlier_counts if k in label),
        key=len,
        default=None,
    )
    return inlier_counts[best_key] if best_key is not None else None


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

def plot_bandwidth_comparison(paths: list[Path], folder: Path,
                              aliases: dict | None = None) -> None:
    """
    Overlay cumulative BoW + VLC bandwidth for each variant (CBS excluded).
    All variants are truncated to the shortest experiment duration.
    """
    if aliases is None:
        aliases = {}
    datasets = [(p, _load_bandwidth(p)) for p in paths]
    datasets = [(p, d) for p, d in datasets if d is not None]
    # Apply alias filtering
    filtered = []
    for p, d in datasets:
        raw = _short_label(p.stem, "bandwidth")
        disp = apply_variant_alias(aliases, _variant_key(raw, folder.name))
        if disp is None:
            continue
        filtered.append((disp, d))
    datasets_labelled = filtered
    if not datasets_labelled:
        print("  No parseable bandwidth data — skipping.")
        return
    # Normalise t_sec to start from 0 for each dataset (they may be absolute timestamps)
    datasets_labelled = [(lbl, {**d, "t_sec": d["t_sec"] - d["t_sec"][0]})
                         for lbl, d in datasets_labelled]
    t_end = min(d["t_sec"][-1] for _, d in datasets_labelled)

    plt.rcParams.update(IEEE_RC)
    fig, ax = plt.subplots()

    for i, (label, d) in enumerate(datasets_labelled):
        t_sec = d["t_sec"]
        mask  = t_sec <= t_end
        t_m, v_m = t_sec[mask], d["total"][mask]
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


def plot_loops_comparison(paths: list[Path], folder: Path,
                          aliases: dict | None = None) -> None:
    """
    Two-panel plot:
      top:    PR (solid) and GV (dashed) counts per variant
      bottom: GV/PR ratio per variant (0 where PR == 0)
    All variants are truncated to the shortest experiment duration.
    Supports both main-system (pr_total/gv_total) and baseline
    (bow_matches/num_loop_closures) schemas.
    """
    if aliases is None:
        aliases = {}
    datasets = [(p, _load_loops(p)) for p in paths]
    datasets = [(p, d) for p, d in datasets if d is not None]
    # Apply alias filtering; replace label with display name
    filtered = []
    for p, d in datasets:
        raw = _short_label(p.stem, "loops")
        disp = apply_variant_alias(aliases, _variant_key(raw, folder.name))
        if disp is None:
            continue
        filtered.append((disp, d))
    datasets_labelled = filtered
    if not datasets_labelled:
        print("  No parseable loops data — skipping.")
        return
    # Normalise t_sec to start from 0 (timestamps may be absolute)
    datasets_labelled = [(lbl, {**d, "t_sec": d["t_sec"] - d["t_sec"][0]})
                         for lbl, d in datasets_labelled]
    t_end = min(d["t_sec"][-1] for _, d in datasets_labelled)

    # Load inlier counts (variant_name → n_inliers) from companion file
    inlier_counts = _load_inlier_counts(folder)

    plt.rcParams.update(IEEE_RC)
    fig, (ax_counts, ax_ratio) = plt.subplots(
        2, 1, sharex=True, figsize=(3.5, 2.0),
    )
    fig.subplots_adjust(hspace=0.08)

    any_inlier = False
    for i, (label, d) in enumerate(datasets_labelled):
        t_sec = d["t_sec"]
        mask  = t_sec <= t_end
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

        # Inlier/PR curve: (inlier_count / gv_final) * gv/pr = inlier/pr
        n_inlier = _match_inlier_count(label, inlier_counts)
        if n_inlier is not None:
            gv_final = float(gv[-1]) if gv[-1] > 0 else 1.0
            inlier_rate = n_inlier / gv_final
            with np.errstate(invalid="ignore", divide="ignore"):
                inlier_pr = np.where(pr > 0, inlier_rate * gv / pr, 0.0)
            ax_ratio.plot(t_m, inlier_pr, color=color, linestyle="--",
                          linewidth=0.8, alpha=0.7, label="_nolegend_")
            mark_endpoint(ax_ratio, t_m, inlier_pr, color, fmt="{:.2f}")
            any_inlier = True

    ax_counts.set_ylabel("Count")
    ax_counts.legend(loc="upper left")
    ax_counts.grid(True, alpha=0.3, linestyle="--", linewidth=0.3)

    ax_ratio.set_xlabel("Time (s)")
    ax_ratio.set_ylabel("GV / PR Ratio")
    if any_inlier:
        import matplotlib.lines as mlines
        handles, labels = ax_ratio.get_legend_handles_labels()
        handles.append(mlines.Line2D([], [], color='gray', linestyle='--', linewidth=0.8))
        labels.append('inlier/PR')
        ax_ratio.legend(handles, labels, loc="upper left")
    else:
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


def group_recall_files(folder: Path) -> tuple[list[Path], list[Path]]:
    """Return (variant_paths, baseline_paths) of loops_recall.csv files.

    Variant paths: one per variant subfolder (subdir containing robot dirs),
    or the top-level <folder>/loops_recall.csv if no variant subfolders exist.
    Baseline paths: baselines/<folder.name>/*/loops_recall.csv.
    """
    variant_paths: list[Path] = []

    # Check for variant subfolders
    variant_dirs = discover_variants(folder)
    if variant_dirs:
        for d in variant_dirs:
            p = d / "loops_recall.csv"
            if p.exists():
                variant_paths.append(p)
    else:
        # Fallback: single top-level CSV
        main = folder / "loops_recall.csv"
        if main.exists():
            variant_paths.append(main)

    baseline_paths: list[Path] = []
    baseline_dir = folder.parent / "baselines" / folder.name
    if baseline_dir.exists():
        for method_dir in sorted(baseline_dir.iterdir()):
            p = method_dir / "loops_recall.csv"
            if p.exists():
                baseline_paths.append(p)

    return variant_paths, baseline_paths


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


def plot_recall_comparison(variant_paths: list[Path], baseline_paths: list[Path],
                           folder: Path, aliases: dict | None = None) -> None:
    """Line-curve recall per rotation bucket. Baselines shown as dashed lines."""
    if aliases is None:
        aliases = {}
    variant_data  = [(p, _load_recall(p)) for p in variant_paths]
    baseline_data = [(p, _load_recall(p)) for p in baseline_paths]
    variant_data  = [(p, d) for p, d in variant_data  if d is not None]
    baseline_data = [(p, d) for p, d in baseline_data if d is not None]
    # Apply alias filtering and replace label
    variant_data  = [(p, dict(d, label=apply_variant_alias(aliases, d["label"])))
                     for p, d in variant_data
                     if apply_variant_alias(aliases, d["label"]) is not None]
    baseline_data = [(p, dict(d, label=apply_variant_alias(aliases, d["label"])))
                     for p, d in baseline_data
                     if apply_variant_alias(aliases, d["label"]) is not None]

    all_data = variant_data + baseline_data
    if not all_data:
        print("  No parseable recall data — skipping.")
        return

    bucket_keys = all_data[0][1]["bucket_keys"]
    xs = [bmax for _, bmax in bucket_keys]

    plt.rcParams.update({**IEEE_RC, "figure.figsize": (3.5, 2.0)})
    fig, ax = plt.subplots()

    for i, (p, d) in enumerate(variant_data):
        color = COLORS[i % len(COLORS)]
        ax.plot(xs, d["recalls"], color=color, marker="o", markersize=3, label=d["label"])

    for i, (p, d) in enumerate(baseline_data):
        color = COLORS[(len(variant_data) + i) % len(COLORS)]
        ax.plot(xs, d["recalls"], color=color, marker="o", markersize=3,
                linestyle="--", label=d["label"])

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x}°" for x in xs])
    ax.set_xlabel("GT Rotation Bucket Upper Bound")
    ax.set_ylabel("Recall")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.3)
    plt.tight_layout()

    save_fig(fig, folder / "recall_comparison")
    plt.close(fig)


PLOTTERS: dict[str, object] = {
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

    aliases = load_variant_aliases()

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
        PLOTTERS[type_name](paths, folder, aliases)

    # Recall comparison (loops_recall.csv files)
    variant_recall, baseline_recall = group_recall_files(folder)
    total_recall = len(variant_recall) + len(baseline_recall)
    if total_recall >= 2:
        print(f"\n[recall] {len(variant_recall)} variant(s), {len(baseline_recall)} baseline(s):")
        for p in variant_recall + baseline_recall:
            print(f"  {p}")
        plot_recall_comparison(variant_recall, baseline_recall, folder, aliases)
    elif total_recall == 1:
        print(f"\n[recall] only one variant found — skipping comparison plot")


if __name__ == "__main__":
    main()
