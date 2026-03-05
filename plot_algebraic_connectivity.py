#!/usr/bin/env python3
"""
Compute the algebraic connectivity (Fiedler value, λ₂ of the Laplacian) of
the combined multi-robot pose graph for each variant in an experiment folder.

Two pose-graph formats are supported:
  - DPGO g2o  (bpsam_robot_*.g2o): used by our system variants.
    Edge weights = trace of the 6×6 info matrix.
  - Kimera-Multi measurements.csv  (robot_src,pose_src,robot_dst,pose_dst,...,weight):
    used by the Kimera-Multi baseline.  Edge weights = weight column.

Inter-robot edges that appear in multiple robot files are deduplicated.
A bar chart comparing algebraic connectivity across variants and baselines
is saved as algebraic_connectivity.pdf/png inside the experiment folder.

Usage:
    python plot_algebraic_connectivity.py <experiment_folder>
    python plot_algebraic_connectivity.py campus
    python plot_algebraic_connectivity.py gate --unweighted
"""

import argparse
import csv
from pathlib import Path

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import IEEE_RC, ROBOT_COLORS, save_fig

# Large multiplier to encode (robot_id, pose_index) as a single integer
_POSE_ID_MULT = 1_000_000


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


def load_kimera_measurements(paths: list[Path]) -> tuple[set[int], dict[frozenset, float]]:
    """Parse Kimera-Multi measurements.csv files.

    Node IDs are encoded as robot_id * _POSE_ID_MULT + pose_index.
    Edge weight is taken from the 'weight' column.
    """
    vertices: set[int] = set()
    edges: dict[frozenset, float] = {}
    for path in paths:
        with open(path) as f:
            for row in csv.DictReader(f):
                try:
                    id1 = int(row["robot_src"]) * _POSE_ID_MULT + int(row["pose_src"])
                    id2 = int(row["robot_dst"]) * _POSE_ID_MULT + int(row["pose_dst"])
                    w = float(row["weight"])
                except (KeyError, ValueError):
                    continue
                vertices.add(id1)
                vertices.add(id2)
                key = frozenset({id1, id2})
                if key not in edges or edges[key] < w:
                    edges[key] = w
    return vertices, edges


def build_graph_from_g2o(g2o_files: list[Path], weighted: bool) -> nx.Graph:
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


def build_graph_from_measurements(meas_files: list[Path], weighted: bool) -> nx.Graph:
    """Build networkx graph from Kimera-Multi measurements.csv files."""
    vertices, edges = load_kimera_measurements(meas_files)
    G = nx.Graph()
    G.add_nodes_from(vertices)
    for key, w in edges.items():
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


def find_measurements_files(variant_dir: Path) -> list[Path]:
    """Return all measurements.csv files under a variant directory."""
    return sorted(variant_dir.rglob("measurements.csv"))


# Each entry: (label, files, format) where format is "g2o" or "measurements"
VariantEntry = tuple[str, list[Path], str]


def discover_variants(folder: Path) -> list[VariantEntry]:
    """Return [(label, files, format), ...] for variants and baselines."""
    results: list[VariantEntry] = []

    def _add(label: str, d: Path) -> None:
        g2o = find_g2o_files(d)
        if g2o:
            results.append((label, g2o, "g2o"))
            return
        meas = find_measurements_files(d)
        if meas:
            results.append((label, meas, "measurements"))

    # Variant subdirs
    variant_dirs = [
        d for d in sorted(folder.iterdir())
        if d.is_dir() and any(_is_robot_dir(s) for s in d.iterdir() if s.is_dir())
    ]
    if variant_dirs:
        for d in variant_dirs:
            _add(d.name, d)
    else:
        _add(folder.name, folder)

    # Baselines
    baseline_dir = folder.parent / "baselines" / folder.name
    if baseline_dir.exists():
        for method_dir in sorted(baseline_dir.iterdir()):
            if method_dir.is_dir():
                _add(method_dir.name, method_dir)

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
        print("No pose graph files found.")
        raise SystemExit(1)

    print(f"Experiment: {folder.name}  ({'weighted' if weighted else 'unweighted'})")
    print(f"{'Variant':<20} {'Nodes':>6} {'Edges':>7} {'Connected':>10} {'λ₂ (alg. conn.)':>16}")
    print("-" * 65)

    labels, values, is_baseline = [], [], []
    baseline_dir = folder.parent / "baselines" / folder.name

    for label, files, fmt in variants:
        if fmt == "g2o":
            G = build_graph_from_g2o(files, weighted=weighted)
        else:
            G = build_graph_from_measurements(files, weighted=weighted)

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        connected = nx.is_connected(G)

        if connected:
            lam2 = nx.algebraic_connectivity(G, weight="weight", method="tracemin_lu")
        else:
            lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            lam2 = nx.algebraic_connectivity(lcc, weight="weight", method="tracemin_lu")

        conn_str = "yes" if connected else f"no (LCC {max(len(c) for c in nx.connected_components(G))})"
        print(f"{label:<20} {n_nodes:>6} {n_edges:>7} {conn_str:>10} {lam2:>16.4e}")

        is_bl = files[0].is_relative_to(baseline_dir) if files else False
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
