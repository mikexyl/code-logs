#!/usr/bin/env python3
"""
Compute the algebraic connectivity (Fiedler value, λ₂ of the Laplacian) of
the combined multi-robot pose graph for each variant in an experiment folder.

The graph is built from:
  - Nodes / odometry edges: kimera_distributed_keyframes.csv (one chain per robot)
  - Inter-robot edges:      inlier_loops.csv  (written by evaluate_loops_recall.py)

inlier_loops.csv contains name1, t1_s, name2, t2_s for each inlier loop.
Each timestamp is matched to the nearest keyframe to get the pose-graph node ID.

Usage:
    python plot_algebraic_connectivity.py campus
    python plot_algebraic_connectivity.py gate
"""

import argparse
import csv
from pathlib import Path

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from utils.io import (load_keyframes_csv, load_variant_aliases, apply_variant_alias,
                      is_robot_dir, discover_robots)
from utils.plot import IEEE_RC, ROBOT_COLORS, save_fig

_POSE_ID_MULT = 10_000_000   # robot_id * MULT + pose_index → unique integer node ID


def _robot_id_map(variant_dir: Path) -> dict[str, int]:
    """Map robot dir name → integer robot ID (inverted from discover_robots)."""
    return {name: rid for rid, name in discover_robots(variant_dir).items()}


def _build_kf_index(variant_dir: Path, rid_map: dict[str, int]
                    ) -> dict[str, tuple[np.ndarray, list[int]]]:
    """For each robot name, return (sorted_timestamps_s, pose_indices) from keyframes CSV."""
    result: dict[str, tuple[np.ndarray, list[int]]] = {}
    for name, rid in rid_map.items():
        kf_csv = variant_dir / name / "distributed" / "kimera_distributed_keyframes.csv"
        if not kf_csv.exists():
            continue
        kf = load_keyframes_csv(kf_csv)   # pose_index -> timestamp_s
        sorted_items = sorted(kf.items(), key=lambda x: x[1])
        ts = np.array([t for _, t in sorted_items])
        poses = [p for p, _ in sorted_items]
        result[name] = (ts, poses)
    return result


def _nearest_pose_idx(ts: np.ndarray, poses: list[int], t: float, max_gap: float = 2.5
                      ) -> int | None:
    """Return pose_index of the keyframe nearest to t, or None if gap > max_gap."""
    idx = int(np.searchsorted(ts, t))
    idx = max(0, min(idx, len(ts) - 1))
    return poses[idx] if abs(ts[idx] - t) <= max_gap else None


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(variant_dir: Path) -> nx.Graph:
    G = nx.Graph()

    rid_map = _robot_id_map(variant_dir)
    kf_index = _build_kf_index(variant_dir, rid_map)

    # Odometry chains
    for name, rid in rid_map.items():
        if name not in kf_index:
            continue
        ts, poses = kf_index[name]
        nodes = [rid * _POSE_ID_MULT + p for p in poses]
        G.add_nodes_from(nodes)
        for i in range(len(nodes) - 1):
            G.add_edge(nodes[i], nodes[i + 1])

    # Inlier inter-robot loop closure edges
    inlier_csv = variant_dir / "inlier_loops.csv"
    if not inlier_csv.exists():
        return G

    name_to_rid = {name: rid for name, rid in rid_map.items()}

    with open(inlier_csv) as f:
        for row in csv.DictReader(f):
            name1, t1 = row["name1"], float(row["t1_s"])
            name2, t2 = row["name2"], float(row["t2_s"])
            if name1 not in kf_index or name2 not in kf_index:
                continue
            p1 = _nearest_pose_idx(*kf_index[name1], t1)
            p2 = _nearest_pose_idx(*kf_index[name2], t2)
            if p1 is None or p2 is None:
                continue
            r1 = name_to_rid[name1]
            r2 = name_to_rid[name2]
            G.add_edge(r1 * _POSE_ID_MULT + p1, r2 * _POSE_ID_MULT + p2)

    return G


# ---------------------------------------------------------------------------
# Variant / baseline discovery
# ---------------------------------------------------------------------------

VariantEntry = tuple[str, Path, bool]  # (label, dir, is_baseline)


def _discover_all(folder: Path) -> list[VariantEntry]:
    """Return variants + baselines as (raw_name, dir, is_baseline) triples."""
    from utils.io import discover_variants as _dv, discover_baselines as _db
    variant_dirs = _dv(folder)
    results: list[VariantEntry] = []
    if variant_dirs:
        for d in variant_dirs:
            results.append((d.name, d, False))
    else:
        results.append((folder.name, folder, False))
    for d in _db(folder):
        results.append((d.name, d, True))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute algebraic connectivity of multi-robot pose graphs "
                    "using inlier_loops.csv from evaluate_loops_recall.py."
    )
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.exists():
        print(f"Error: {folder} does not exist.")
        raise SystemExit(1)

    aliases  = load_variant_aliases()
    variants = _discover_all(folder)
    print(f"Experiment: {folder.name}  (unweighted, inlier loops only)")
    print(f"{'Variant':<22} {'Nodes':>7} {'Edges':>7} {'LC edges':>9} "
          f"{'Connected':>10} {'λ₂':>12}")
    print("-" * 72)

    labels, values, is_baseline = [], [], []

    for raw_label, vdir, is_bl in variants:
        label = apply_variant_alias(aliases, raw_label)
        if label is None:
            continue
        G = build_graph(vdir)
        if G.number_of_nodes() == 0:
            print(f"{label:<22}  (no keyframe data found — skipped)")
            continue

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        lc_edges = sum(
            1 for u, v in G.edges()
            if u // _POSE_ID_MULT != v // _POSE_ID_MULT
        )
        connected = nx.is_connected(G)

        if connected:
            lam2 = nx.algebraic_connectivity(G, weight=None, method="tracemin_lu")
        else:
            lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            lam2 = nx.algebraic_connectivity(lcc, weight=None, method="tracemin_lu")

        conn_str = "yes" if connected else \
            f"no(LCC {max(len(c) for c in nx.connected_components(G))})"
        print(f"{label:<22} {n_nodes:>7} {n_edges:>7} {lc_edges:>9} "
              f"{conn_str:>10} {lam2:>12.4e}")

        labels.append(label)
        values.append(lam2)
        is_baseline.append(is_bl)

    if not values:
        print("No data to plot.")
        return

    # Bar chart
    _fw = max(3.5, len(labels) * 0.75)
    plt.rcParams.update({**IEEE_RC, "figure.figsize": (_fw, _fw * 9 / 16)})
    fig, ax = plt.subplots()
    x = np.arange(len(labels))
    colors = [ROBOT_COLORS[i % len(ROBOT_COLORS)] for i in range(len(labels))]

    for i, (v, color, bl) in enumerate(zip(values, colors, is_baseline)):
        ax.bar(x[i], v, color=color, width=0.6,
               edgecolor="black" if bl else "none",
               linewidth=0.8,
               hatch="///" if bl else None)
        ax.text(x[i], v + max(values) * 0.01, f"{v:.2e}",
                ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(r"Algebraic Connectivity $\lambda_2$ (unweighted, inlier loops)")
    ax.set_title(f"{folder.name}", fontsize=7)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.3)
    plt.tight_layout()

    out = folder / "algebraic_connectivity"
    save_fig(fig, out)
    plt.close(fig)
    print(f"\nSaved to {out}.pdf / .png")


if __name__ == "__main__":
    main()
