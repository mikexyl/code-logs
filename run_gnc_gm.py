#!/usr/bin/env python3
"""
Run centralized GNC+GM on the merged pose graph of a variant, then plot
loop closure lines colored by GNC weight (1=inlier, 0=outlier).

Usage:
    python3 run_gnc_gm.py <variant_dir> <gt_dir>
    python3 run_gnc_gm.py campus/no-scoring ground_truth/campus

Output:
    <variant_dir>/gnc_weights_map.pdf/png
    <variant_dir>/gnc_weights.csv   — (k1, k2, weight) for all loop edges
"""

import argparse
import csv
import sys
from pathlib import Path

import gtsam
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from utils.io import read_tum_trajectory, load_alignment_from_evo_zip
from utils.plot import IEEE_RC, save_fig, apply_alignment, find_tum_position


# ---------------------------------------------------------------------------
# Load and combine per-robot g2o files
# ---------------------------------------------------------------------------

def load_combined_graph(variant_dir: Path):
    """Load all bpsam_robot_*.g2o files and combine into one graph.

    Returns (graph, odom_initial, chr_to_name) where:
        odom_initial : Values obtained by optimizing odometry-only (no loops),
                       so GNC starts from a loop-free estimate and can detect
                       inconsistent loop closures.
        chr_to_name  : {robot_chr (int): robot_dir_name (str)}

    NOTE: g2o vertex index i == TUM row i (positions match exactly).
    Do NOT use the keyframe CSV for position lookup — DPGO pose indices and
    Kimera keyframe IDs are different numbering systems (~10x apart).
    """
    graph = gtsam.NonlinearFactorGraph()
    dpgo_initial = gtsam.Values()   # DPGO-optimized (used to seed odom-only run)
    chr_to_name: dict[int, str] = {}

    seen_keys: set[int] = set()

    for g2o_path in sorted(variant_dir.glob('*/dpgo/bpsam_robot_*.g2o')):
        robot_id = int(g2o_path.stem.split('_')[-1])
        robot_chr = ord('a') + robot_id
        robot_name = g2o_path.parent.parent.name

        sub_graph, sub_init = gtsam.readG2o(str(g2o_path), True)

        for i in range(sub_graph.size()):
            graph.add(sub_graph.at(i))

        for k in sub_init.keys():
            if k not in seen_keys:
                dpgo_initial.insert(k, sub_init.atPose3(k))
                seen_keys.add(k)

        chr_to_name[robot_chr] = robot_name

        print(f"  loaded {robot_name} (chr='{chr(robot_chr)}'): "
              f"{sub_graph.size()} factors, {sub_init.size()} poses")

    print(f"Combined: {graph.size()} factors, {dpgo_initial.size()} poses")

    # Build odometry-only graph and optimize to get a loop-free initial estimate
    print("  Computing odometry-only initial estimate for GNC...")
    odom_graph = gtsam.NonlinearFactorGraph()
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-6)
    for robot_chr in sorted(set(gtsam.Symbol(k).chr() for k in dpgo_initial.keys())):
        own_keys = [k for k in dpgo_initial.keys() if gtsam.Symbol(k).chr() == robot_chr]
        first_key = min(own_keys, key=lambda k: gtsam.Symbol(k).index())
        odom_graph.addPriorPose3(first_key, dpgo_initial.atPose3(first_key), prior_noise)

    for i in range(graph.size()):
        f = graph.at(i)
        keys = f.keys()
        if len(keys) == 2:
            c1 = gtsam.Symbol(keys[0]).chr()
            c2 = gtsam.Symbol(keys[1]).chr()
            if c1 == c2:
                odom_graph.add(f)

    odom_result = gtsam.GaussNewtonOptimizer(odom_graph, dpgo_initial).optimize()
    print(f"  Odometry-only initial error: {graph.error(odom_result):.2f}")
    return graph, odom_result, chr_to_name


# ---------------------------------------------------------------------------
# Run GNC+GM
# ---------------------------------------------------------------------------

def run_gnc_gm(graph: gtsam.NonlinearFactorGraph,
               initial: gtsam.Values) -> tuple[gtsam.Values, np.ndarray, list[int]]:
    """Run GNC+GM. Returns (result, weights, loop_factor_indices).

    Odometry edges (same robot chr) are fixed as known inliers.
    Loop closure edges (different chr) are subject to GNC weighting.
    """
    known_inliers = []
    loop_indices = []

    # First pass: classify original factors
    for i in range(graph.size()):
        f = graph.at(i)
        keys = f.keys()
        if len(keys) != 2:
            known_inliers.append(i)
            continue
        c1 = gtsam.Symbol(keys[0]).chr()
        c2 = gtsam.Symbol(keys[1]).chr()
        if c1 == c2:
            known_inliers.append(i)  # odometry
        else:
            loop_indices.append(i)   # inter-robot loop closure

    # Add a prior on every robot's first pose to fix gauge freedom
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-6)
    for robot_chr in sorted(set(gtsam.Symbol(k).chr() for k in initial.keys())):
        own_keys = [k for k in initial.keys() if gtsam.Symbol(k).chr() == robot_chr]
        first_key = min(own_keys, key=lambda k: gtsam.Symbol(k).index())
        prior_idx = graph.size()  # index of the factor about to be added
        graph.addPriorPose3(first_key, initial.atPose3(first_key), prior_noise)
        known_inliers.append(prior_idx)

    print(f"Known inliers (odometry): {len(known_inliers)}")
    print(f"Loop closure factors:     {len(loop_indices)}")

    params = gtsam.GncGaussNewtonParams()
    params.setLossType(gtsam.GncLossType.GM)
    params.setKnownInliers(known_inliers)
    params.setVerbosityGNC(gtsam.GncGaussNewtonParams.Verbosity.SUMMARY)

    optimizer = gtsam.GncGaussNewtonOptimizer(graph, initial, params)
    result = optimizer.optimize()
    weights = np.array(optimizer.getWeights())

    print(f"GNC weights: min={weights.min():.4f} max={weights.max():.4f} "
          f"mean={weights.mean():.4f}")
    loop_weights = weights[loop_indices]
    n_inlier = (loop_weights > 0.5).sum()
    print(f"Loop inliers (w>0.5): {n_inlier}/{len(loop_indices)}")

    return result, weights, loop_indices


# ---------------------------------------------------------------------------
# Resolve loop edges to world-frame positions
# ---------------------------------------------------------------------------

def resolve_loop_positions(
    graph: gtsam.NonlinearFactorGraph,
    loop_indices: list[int],
    weights: np.ndarray,
    chr_to_name: dict[int, str],
    variant_dir: Path,
    rotation, translation, scale,
) -> list[dict]:
    """For each loop edge, resolve both endpoints to aligned world-frame XY.

    g2o vertex index i == TUM row i (verified: positions match exactly).
    We use direct array indexing rather than keyframe-CSV timestamp lookup,
    since the DPGO pose indices and Kimera keyframe IDs are different systems.

    Returns list of {p1, p2, weight, name1, name2}.
    """
    # Build robot TUM position arrays (indexed by g2o vertex index)
    robot_pos: dict[int, np.ndarray] = {}
    for robot_chr, robot_name in chr_to_name.items():
        dpgo = variant_dir / robot_name / 'dpgo'
        tums = sorted(dpgo.glob('Robot *.tum')) if dpgo.exists() else []
        if tums:
            _, pos, _ = read_tum_trajectory(str(tums[0]))
            robot_pos[robot_chr] = np.array(pos)

    records = []
    for idx in loop_indices:
        f = graph.at(idx)
        keys = f.keys()
        k1, k2 = keys[0], keys[1]
        sym1, sym2 = gtsam.Symbol(k1), gtsam.Symbol(k2)
        c1, c2 = sym1.chr(), sym2.chr()
        i1, i2 = sym1.index(), sym2.index()

        w = float(weights[idx])

        n1 = chr_to_name.get(c1)
        n2 = chr_to_name.get(c2)

        if c1 not in robot_pos or c2 not in robot_pos:
            continue
        if i1 >= len(robot_pos[c1]) or i2 >= len(robot_pos[c2]):
            continue

        raw1 = robot_pos[c1][i1]
        raw2 = robot_pos[c2][i2]

        p1 = apply_alignment(raw1.reshape(1, 3), rotation, translation, scale)[0]
        p2 = apply_alignment(raw2.reshape(1, 3), rotation, translation, scale)[0]

        records.append({'p1': p1, 'p2': p2, 'weight': w, 'name1': n1, 'name2': n2})

    print(f"Resolved {len(records)}/{len(loop_indices)} loop edges to world positions")
    return records


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_gnc_weights(variant_dir: Path, records: list[dict],
                     chr_to_name: dict[int, str],
                     rotation, translation, scale):
    plt.rcParams.update({**IEEE_RC, 'figure.figsize': (4.5, 4.0)})
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    # Colormap: red (outlier) → green (inlier).
    # Weights are log-normalised: clip at eps so near-zero outliers spread
    # across the red-orange band while the high-weight inliers pop green.
    all_w = np.array([r['weight'] for r in records])
    eps = 1e-4
    vmax = float(np.clip(all_w.max(), eps, None))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'OutlierInlier', ['#a50026', '#f46d43', '#1a9850'])
    norm = mcolors.LogNorm(vmin=eps, vmax=vmax)

    # Ultra-light trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(chr_to_name))))
    for i, (robot_chr, robot_name) in enumerate(sorted(chr_to_name.items())):
        dpgo = variant_dir / robot_name / 'dpgo'
        tums = sorted(dpgo.glob('Robot *.tum')) if dpgo.exists() else []
        if not tums:
            continue
        ts, pos, _ = read_tum_trajectory(str(tums[0]))
        pos = np.array(pos)
        aligned = apply_alignment(pos, rotation, translation, scale)
        ax.plot(aligned[:, 0], aligned[:, 1], color=colors[i], linewidth=0.4,
                alpha=0.15, label=robot_name, zorder=2)

    # Loop lines colored by GNC weight (clip to eps before log mapping)
    for r in records:
        p1, p2 = r['p1'], r['p2']
        color = cmap(norm(max(r['weight'], eps)))
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=color, linewidth=1.2, alpha=0.85, zorder=5)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(f'GNC weight (log scale, max={vmax:.2f})', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='best', framealpha=0.8, fontsize=6, ncol=2)
    ax.grid(True, alpha=0.2, linewidth=0.3)
    plt.tight_layout(pad=0.5)

    save_fig(fig, variant_dir / 'gnc_weights_map')
    plt.close(fig)
    print(f"Saved → {variant_dir}/gnc_weights_map.pdf/png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('variant_dir', type=Path)
    parser.add_argument('gt_dir', type=Path, nargs='?', default=None,
                        help='Not used for GNC, kept for consistency')
    args = parser.parse_args()

    vdir = args.variant_dir.resolve()
    if not vdir.exists():
        print(f"Error: {vdir} does not exist"); sys.exit(1)

    evo_zip = vdir / 'evo_ape.zip'
    if not evo_zip.exists():
        print(f"Error: {evo_zip} not found (needed for alignment)"); sys.exit(1)

    rotation, translation, scale = load_alignment_from_evo_zip(str(evo_zip))

    print("=== Loading combined graph ===")
    graph, initial, chr_to_name = load_combined_graph(vdir)

    print("\n=== Running GNC+GM ===")
    result, weights, loop_indices = run_gnc_gm(graph, initial)

    # Save weights CSV
    weights_csv = vdir / 'gnc_weights.csv'
    with open(weights_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['factor_idx', 'key1', 'key2', 'weight'])
        for idx in loop_indices:
            keys = graph.at(idx).keys()
            writer.writerow([idx, keys[0], keys[1], f'{weights[idx]:.6f}'])
    print(f"Saved weights → {weights_csv}")

    print("\n=== Resolving loop positions ===")
    records = resolve_loop_positions(
        graph, loop_indices, weights, chr_to_name,
        vdir, rotation, translation, scale)

    print("\n=== Plotting ===")
    plot_gnc_weights(vdir, records, chr_to_name, rotation, translation, scale)


if __name__ == '__main__':
    main()
