#!/usr/bin/env python3
"""
Compute the algebraic connectivity (Fiedler value, λ₂ of the Laplacian) of
the combined multi-robot pose graph for each variant in an experiment folder.

All DPGO g2o files (bpsam_robot_*.g2o) from every robot's dpgo/ directory are
merged into a single undirected graph.  Inter-robot edges that appear in
multiple robot files are deduplicated.  Edge weights are taken from the trace
of the 6×6 information matrix stored in each EDGE_SE3:QUAT line.

A bar chart comparing algebraic connectivity across variants (and baselines)
is saved as algebraic_connectivity.pdf/png inside the experiment folder.

Usage:
    python plot_algebraic_connectivity.py <experiment_folder>
    python plot_algebraic_connectivity.py campus
    python plot_algebraic_connectivity.py gate --unweighted
"""

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import IEEE_RC, ROBOT_COLORS, save_fig


# ---------------------------------------------------------------------------
# G2O parsing
# ---------------------------------------------------------------------------

def _info_trace(parts: list[str], offset: int) -> float:
    """Return trace of the 6×6 upper-triangular info matrix in an EDGE line.

    The upper triangle is stored row-by-row: 21 values starting at `offset`.
    Diagonal indices (0-based within the 21 values): 0,6,11,15,18,20.
    """
    diag_offsets = [0, 6, 11, 15, 18, 20]
    try:
        return sum(float(parts[offset + d]) for d in diag_offsets)
    except (IndexError, ValueError):
        return 1.0


def load_g2o(path: Path) -> tuple[set[int], dict[frozenset, float]]:
    """Parse a single g2o file.

    Returns:
        vertices: set of integer vertex IDs
        edges:    {frozenset({id1, id2}): weight}  (weight = info trace)
    """
    vertices: set[int] = set()
    edges: dict[frozenset, float] = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "VERTEX_SE3:QUAT":
                vertices.add(int(parts[1]))
            elif parts[0] == "EDGE_SE3:QUAT":
                id1, id2 = int(parts[1]), int(parts[2])
                key = frozenset({id1, id2})
                weight = _info_trace(parts, offset=10)  # pose fields: 3+4=7, plus ids+tag = offset 10
                # Keep the maximum weight if the same edge appears in multiple files
                if key not in edges or edges[key] < weight:
                    edges[key] = weight
    return vertices, edges


def build_graph(g2o_files: list[Path], weighted: bool) -> nx.Graph:
    """Combine multiple g2o files into one networkx graph."""
    all_vertices: set[int] = set()
    all_edges: dict[frozenset, float] = {}

    for p in g2o_files:
        verts, edges = load_g2o(p)
        all_vertices |= verts
        for key, w in edges.items():
            if key not in all_edges or all_edges[key] < w:
                all_edges[key] = w

    G = nx.Graph()
    G.add_nodes_from(all_vertices)
    for key, w in all_edges.items():
        id1, id2 = tuple(key)
        G.add_edge(id1, id2, weight=w if weighted else 1.0)
    return G


# ---------------------------------------------------------------------------
# Variant discovery
# ---------------------------------------------------------------------------

def _is_robot_dir(d: Path) -> bool:
    return (d / "distributed").is_dir() or (d / "dpgo").is_dir()


def find_g2o_files(variant_dir: Path) -> list[Path]:
    """Return all bpsam_robot_*.g2o files under a variant directory."""
    return sorted(variant_dir.rglob("bpsam_robot_*.g2o"))


def discover_variants(folder: Path) -> list[tuple[str, list[Path]]]:
    """Return [(label, [g2o_paths]), ...] for variants and baselines."""
    results = []

    # Variant subdirs
    variant_dirs = [
        d for d in sorted(folder.iterdir())
        if d.is_dir() and any(_is_robot_dir(s) for s in d.iterdir() if s.is_dir())
    ]
    if variant_dirs:
        for d in variant_dirs:
            files = find_g2o_files(d)
            if files:
                results.append((d.name, files))
    else:
        files = find_g2o_files(folder)
        if files:
            results.append((folder.name, files))

    # Baselines
    baseline_dir = folder.parent / "baselines" / folder.name
    if baseline_dir.exists():
        for method_dir in sorted(baseline_dir.iterdir()):
            if method_dir.is_dir():
                files = find_g2o_files(method_dir)
                if files:
                    results.append((method_dir.name, files))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute algebraic connectivity of combined multi-robot pose graphs."
    )
    parser.add_argument("folder", type=Path, help="Experiment folder to scan")
    parser.add_argument("--unweighted", action="store_true",
                        help="Use unweighted graph (default: weight = info matrix trace)")
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.exists():
        print(f"Error: {folder} does not exist.")
        raise SystemExit(1)

    weighted = not args.unweighted
    variants = discover_variants(folder)
    if not variants:
        print("No bpsam_robot_*.g2o files found.")
        raise SystemExit(1)

    print(f"Experiment: {folder.name}  ({'weighted' if weighted else 'unweighted'})")
    print(f"{'Variant':<20} {'Nodes':>6} {'Edges':>7} {'Connected':>10} {'λ₂ (alg. conn.)':>16}")
    print("-" * 65)

    labels, values, is_baseline = [], [], []

    for label, g2o_files in variants:
        G = build_graph(g2o_files, weighted=weighted)
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        connected = nx.is_connected(G)

        if connected:
            lam2 = nx.algebraic_connectivity(G, weight="weight", method="tracemin_lu")
        else:
            # Compute on the largest connected component
            lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            lam2 = nx.algebraic_connectivity(lcc, weight="weight", method="tracemin_lu")

        conn_str = "yes" if connected else f"no (LCC {max(len(c) for c in nx.connected_components(G))})"
        print(f"{label:<20} {n_nodes:>6} {n_edges:>7} {conn_str:>10} {lam2:>16.4f}")

        # Determine if baseline (not a variant subdir of folder)
        baseline_dir = folder.parent / "baselines" / folder.name
        is_bl = (g2o_files[0].is_relative_to(baseline_dir)) if g2o_files else False

        labels.append(label)
        values.append(lam2)
        is_baseline.append(is_bl)

    # Bar chart
    plt.rcParams.update({**IEEE_RC, "figure.figsize": (max(3.5, len(labels) * 0.7), 2.8)})
    fig, ax = plt.subplots()

    x = np.arange(len(labels))
    colors = [ROBOT_COLORS[i % len(ROBOT_COLORS)] for i in range(len(labels))]

    for i, (v, color, bl) in enumerate(zip(values, colors, is_baseline)):
        bar = ax.bar(x[i], v, color=color, width=0.6,
                     edgecolor="black" if bl else "none",
                     linewidth=0.8,
                     hatch="///" if bl else None)
        ax.text(x[i], v + max(values) * 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(r"Algebraic Connectivity $\lambda_2$")
    weight_note = "weighted (info trace)" if weighted else "unweighted"
    ax.set_title(f"{folder.name} — pose graph {weight_note}", fontsize=7)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.3)
    plt.tight_layout()

    out = folder / "algebraic_connectivity"
    save_fig(fig, out)
    plt.close(fig)
    print(f"\nSaved to {out}.pdf / .png")


if __name__ == "__main__":
    main()
