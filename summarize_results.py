#!/usr/bin/env python3
"""
Summarize GV/PR ratio, ATE RMSE, total bandwidth, and Recall@30deg
for all variants across all experiments (excluding baselines).
Saves results to summary_results.csv.
"""

import csv
import json
import zipfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent

_POSE_ID_MULT = 10_000_000


def load_variant_aliases(root: Path) -> set:
    """Load allowed variant names from variant_aliases.yaml. Returns a set of keys."""
    path = root / "variant_aliases.yaml"
    if not path.exists():
        return None
    with open(path) as f:
        data = yaml.safe_load(f)
    return set(data.keys()) if data else None


def get_ate_rmse(variant_dir: Path):
    """Extract ATE RMSE from evo_ape.zip."""
    zip_path = variant_dir / "evo_ape.zip"
    if not zip_path.exists():
        return None
    try:
        with zipfile.ZipFile(zip_path) as z:
            with z.open("stats.json") as f:
                stats = json.load(f)
        return stats["rmse"]
    except Exception:
        return None


def get_recall_at_30(variant_dir: Path):
    """
    Compute cumulative recall for loops with rotation <= 30 degrees (includes outliers).
    Returns (detected_recall, inlier_recall) or (None, None).
    """
    csv_path = variant_dir / "loops_recall.csv"
    if not csv_path.exists():
        return None, None
    try:
        df = pd.read_csv(csv_path)
        sub = df[df["bucket_max"] <= 30]
        total_gt = sub["n_total"].sum()
        if total_gt == 0:
            return None, None
        detected = sub["n_detected"].sum() / total_gt
        inlier = sub["n_inlier"].sum() / total_gt
        return detected, inlier
    except Exception:
        return None, None


def get_bandwidth(exp_dir: Path, variant_name: str):
    """
    Extract total bandwidth (BoW + VLC + CBS) in MB.

    Priority:
    1. *_bandwidth.npy (pre-extracted from .rrd)
    2. distributed/lcd_log.csv per robot (bow_bytes + vlc_bytes, final row)
    3. dpgo/stats_robot_*.csv per robot (bytes_sent, CBS-only variants)
    """
    exp_name = exp_dir.name
    variant_dir = exp_dir / variant_name

    # 1. Pre-extracted .npy
    candidates = [
        exp_dir / f"{exp_name}-{variant_name}_bandwidth.npy",
        exp_dir / f"{variant_name}_bandwidth.npy",
        variant_dir / f"{variant_name}_bandwidth.npy",
    ]
    for p in candidates:
        if p.exists():
            try:
                bw = np.load(p, allow_pickle=True).item()
                total = float(bw["bow_MB"][-1]) + float(bw["vlc_MB"][-1]) + float(bw["cbs_MB"][-1])
                return total
            except Exception:
                continue

    # 2. lcd_log.csv: sum final bow_bytes + vlc_bytes across all robots
    total_bytes = 0.0
    found_lcd = False
    for robot_dir in sorted(variant_dir.iterdir()) if variant_dir.is_dir() else []:
        lcd = robot_dir / "distributed" / "lcd_log.csv"
        if lcd.exists():
            try:
                import csv as _csv
                with open(lcd) as f:
                    rows = [{k.strip(): v for k, v in row.items()} for row in _csv.DictReader(f)]
                if rows:
                    last = rows[-1]
                    total_bytes += float(last.get("bow_bytes", 0) or 0)
                    total_bytes += float(last.get("vlc_bytes", 0) or 0)
                    found_lcd = True
            except Exception:
                pass
    if found_lcd:
        return total_bytes / 1e6

    # 3. stats_robot_*.csv: sum final bytes_sent across all robots (CBS-only)
    total_bytes = 0.0
    found_stats = False
    for robot_dir in sorted(variant_dir.iterdir()) if variant_dir.is_dir() else []:
        for stats_file in sorted((robot_dir / "dpgo").glob("stats_robot_*.csv")) if (robot_dir / "dpgo").is_dir() else []:
            try:
                import csv as _csv
                with open(stats_file) as f:
                    rows = list(_csv.DictReader(f))
                if rows:
                    total_bytes += float(rows[-1].get("bytes_sent", 0) or 0)
                    found_stats = True
            except Exception:
                pass
    if found_stats:
        return total_bytes / 1e6

    return None


def get_gv_pr_ratio(exp_dir: Path, variant_name: str):
    """
    Compute GV/PR ratio from *_loops.npy.
    """
    exp_name = exp_dir.name

    candidates = [
        exp_dir / f"{exp_name}-{variant_name}_loops.npy",
        exp_dir / f"{variant_name}_loops.npy",
        exp_dir / variant_name / f"{variant_name}_loops.npy",
    ]
    for p in candidates:
        if p.exists():
            try:
                lp = np.load(p, allow_pickle=True).item()
                pr = float(lp["pr_total"][-1])
                gv = float(lp["gv_total"][-1])
                if pr == 0:
                    return None
                return gv / pr
            except Exception:
                continue
    return None


def get_algebraic_connectivity(variant_dir: Path):
    """
    Compute λ₂ of the unweighted pose graph (odometry chains + inlier inter-robot loops).
    Uses LCC when the graph is disconnected. Returns None if no keyframe data found.
    Mirrors the logic in plot_algebraic_connectivity.py.
    """
    from utils.io import load_keyframes_csv, discover_robots

    try:
        rid_map = {name: rid for rid, name in discover_robots(variant_dir).items()}
        if not rid_map:
            return None

        # Build keyframe index per robot
        kf_index = {}
        for name, rid in rid_map.items():
            kf_csv = variant_dir / name / "distributed" / "kimera_distributed_keyframes.csv"
            if not kf_csv.exists():
                continue
            kf = load_keyframes_csv(kf_csv)
            sorted_items = sorted(kf.items(), key=lambda x: x[1])
            kf_index[name] = (
                np.array([t for _, t in sorted_items]),
                [p for p, _ in sorted_items],
            )

        if not kf_index:
            return None

        G = nx.Graph()

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
        if inlier_csv.exists():
            with open(inlier_csv) as f:
                for row in csv.DictReader(f):
                    name1, t1 = row["name1"], float(row["t1_s"])
                    name2, t2 = row["name2"], float(row["t2_s"])
                    if name1 not in kf_index or name2 not in kf_index:
                        continue
                    ts1, poses1 = kf_index[name1]
                    ts2, poses2 = kf_index[name2]
                    idx1 = int(np.searchsorted(ts1, t1))
                    idx1 = max(0, min(idx1, len(ts1) - 1))
                    idx2 = int(np.searchsorted(ts2, t2))
                    idx2 = max(0, min(idx2, len(ts2) - 1))
                    if abs(ts1[idx1] - t1) > 2.5 or abs(ts2[idx2] - t2) > 2.5:
                        continue
                    r1, r2 = rid_map[name1], rid_map[name2]
                    G.add_edge(r1 * _POSE_ID_MULT + poses1[idx1],
                               r2 * _POSE_ID_MULT + poses2[idx2])

        if G.number_of_nodes() == 0:
            return None

        if nx.is_connected(G):
            graph = G
        else:
            graph = G.subgraph(max(nx.connected_components(G), key=len)).copy()

        return nx.algebraic_connectivity(graph, weight=None, method="tracemin_lu")
    except Exception:
        return None


def has_no_loops_detected(variant_dir: Path) -> bool:
    """Return True if the variant has distributed/ dirs but zero inter-robot loop closures detected.

    Checks raw loop_closures.csv files (not loops_recall.csv, which only counts loops that
    matched a GT pair and would miss false-positive detections).
    Returns False if no distributed/ dirs exist (CBS-only variants — unknown, not zero).
    """
    lc_files = list(variant_dir.glob("*/distributed/loop_closures.csv"))
    if not lc_files:
        return False  # No distributed dir — can't determine, don't flag
    total_loops = 0
    for lc in lc_files:
        try:
            import csv as _csv
            with open(lc) as f:
                rows = list(_csv.DictReader(f))
            total_loops += len(rows)
        except Exception:
            pass
    return total_loops == 0


def is_robot_dir(d: Path) -> bool:
    return d.is_dir() and ((d / "distributed").is_dir() or (d / "dpgo").is_dir())


def discover_variants(exp_dir: Path):
    """Return variant subdirs (dirs containing at least one robot dir)."""
    variants = []
    for sub in sorted(exp_dir.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name in ("baselines", "ground_truth", "lm_optimized"):
            continue
        if any(is_robot_dir(c) for c in sub.iterdir() if c.is_dir()):
            variants.append(sub)
    return variants


def main():
    rows = []

    allowed_variants = load_variant_aliases(ROOT)

    # Experiments to process (top-level dirs with variant structure)
    skip_dirs = {"baselines", "ground_truth", "utils", "src", "build", "third_party", "__pycache__"}

    for exp_dir in sorted(ROOT.iterdir()):
        if not exp_dir.is_dir():
            continue
        if exp_dir.name.startswith("."):
            continue
        if exp_dir.name in skip_dirs:
            continue

        variants = discover_variants(exp_dir)
        if not variants:
            continue

        for vdir in variants:
            if allowed_variants is not None and vdir.name not in allowed_variants:
                continue
            ate = get_ate_rmse(vdir)
            recall30, inlier_recall30 = get_recall_at_30(vdir)
            bw = get_bandwidth(exp_dir, vdir.name)
            gv_pr = get_gv_pr_ratio(exp_dir, vdir.name)
            ac = get_algebraic_connectivity(vdir)
            no_loops = has_no_loops_detected(vdir)

            # Skip variants with no results at all
            if ate is None and recall30 is None and bw is None and gv_pr is None and ac is None:
                continue

            rows.append({
                "experiment": exp_dir.name,
                "variant": vdir.name,
                "ate_rmse_m": round(ate, 4) if ate is not None else "",
                "recall_at_30deg": round(recall30, 4) if recall30 is not None else "",
                "inlier_recall_at_30deg": round(inlier_recall30, 4) if inlier_recall30 is not None else "",
                "total_bw_MB": round(bw, 3) if bw is not None else "",
                "gv_pr_ratio": round(gv_pr, 4) if gv_pr is not None else "",
                "algebraic_connectivity": f"{ac:.4e}" if ac is not None else "",
                "no_loops_detected": "TRUE" if no_loops else "",
            })

    df = pd.DataFrame(rows, columns=["experiment", "variant", "ate_rmse_m", "recall_at_30deg", "inlier_recall_at_30deg", "total_bw_MB", "gv_pr_ratio", "algebraic_connectivity", "no_loops_detected"])
    out_path = ROOT / "summary_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
