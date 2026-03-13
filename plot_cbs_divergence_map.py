#!/usr/bin/env python3
"""
Plot the latest CBS/CBS+ global trajectory and color inter-robot loops by the
sum of the two endpoint TV divergences.

Usage:
    python3 plot_cbs_divergence_map.py <variant_dir> <cbs_dir>
    python3 plot_cbs_divergence_map.py campus/no-scoring campus/no-scoring/cbs_plus

Output:
    <cbs_dir>/loop_divergence_tv_map.pdf/png
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from utils.io import load_keyframes_csv, load_loop_closures_csv, read_tum_trajectory
from utils.plot import IEEE_RC, ROBOT_COLORS, save_fig


def _find_latest_robot_files(cbs_dir: Path, pattern: str) -> dict[int, Path]:
    """Return the highest-timestamp file per robot id for a pattern with two groups."""
    result: dict[int, tuple[int, Path]] = {}
    regex = re.compile(pattern)
    for path in cbs_dir.glob("*/dpgo/*"):
        match = regex.fullmatch(path.name)
        if not match:
            continue
        robot_id = int(match.group(1))
        stamp = int(match.group(2))
        prev = result.get(robot_id)
        if prev is None or stamp > prev[0]:
            result[robot_id] = (stamp, path)
    return {rid: path for rid, (_, path) in result.items()}


def _discover_robots_from_latest_tums(cbs_dir: Path) -> dict[int, str]:
    latest_tums = _find_latest_robot_files(cbs_dir, r"Robot (\d+)_(\d+)\.tum")
    id_to_name: dict[int, str] = {}
    for rid, path in latest_tums.items():
        id_to_name[rid] = path.parent.parent.name
    return id_to_name


def _trim_terminal_timestamp_jump(timestamps: np.ndarray, positions: np.ndarray, quats: np.ndarray):
    """Drop the trailing snapshot row if it uses a wall-clock timestamp jump."""
    keep = len(timestamps)
    for i in range(1, len(timestamps)):
        if timestamps[i] - timestamps[i - 1] > 1000.0:
            keep = i
            break
    return timestamps[:keep], positions[:keep], quats[:keep]


def _load_latest_trajectories(cbs_dir: Path) -> dict[int, dict]:
    latest_tums = _find_latest_robot_files(cbs_dir, r"Robot (\d+)_(\d+)\.tum")
    trajectories: dict[int, dict] = {}
    for rid, path in latest_tums.items():
        ts, pos, quat = read_tum_trajectory(str(path))
        ts_arr = np.asarray(ts, dtype=float)
        pos_arr = np.asarray(pos, dtype=float)
        quat_arr = np.asarray(quat, dtype=float)
        ts_arr, pos_arr, quat_arr = _trim_terminal_timestamp_jump(ts_arr, pos_arr, quat_arr)
        trajectories[rid] = {
            "path": path,
            "timestamps": ts_arr,
            "positions": pos_arr,
        }
    return trajectories


def _load_latest_divergence(cbs_dir: Path) -> dict[int, dict[int, float]]:
    """
    Load the highest-timestamp TV divergence file per robot.

    The same numeric suffix can appear multiple times under different prefixes.
    Aggregate them by suffix so each trajectory-state index gets one total TV
    divergence value.
    """
    latest_divs = _find_latest_robot_files(cbs_dir, r"divergence_tv_robot_(\d+)_(\d+)\.txt")
    result: dict[int, dict[int, float]] = {}
    for rid, path in latest_divs.items():
        idx_to_div: dict[int, float] = {}
        with path.open() as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                match = re.search(r"(\d+)$", parts[0])
                if not match:
                    continue
                idx = int(match.group(1))
                idx_to_div[idx] = idx_to_div.get(idx, 0.0) + float(parts[1])
        result[rid] = idx_to_div
    return result


def _nearest_index(ts_s: float, timestamps_s: np.ndarray, max_gap_s: float = 2.5) -> int | None:
    if timestamps_s.size == 0:
        return None
    idx = int(np.argmin(np.abs(timestamps_s - ts_s)))
    if abs(float(timestamps_s[idx]) - ts_s) > max_gap_s:
        return None
    return idx


def _collect_loop_records(
    variant_dir: Path,
    id_to_name: dict[int, str],
    trajectories: dict[int, dict],
    divergences: dict[int, dict[int, float]],
    max_gap_s: float,
) -> list[dict]:
    kf_maps: dict[int, dict[int, float]] = {}
    for rid, name in id_to_name.items():
        kf_path = variant_dir / name / "distributed" / "kimera_distributed_keyframes.csv"
        if kf_path.exists():
            kf_maps[rid] = load_keyframes_csv(str(kf_path))

    seen: set[frozenset] = set()
    records: list[dict] = []

    for rid, name in id_to_name.items():
        lc_path = variant_dir / name / "distributed" / "loop_closures.csv"
        if not lc_path.exists():
            continue
        for loop in load_loop_closures_csv(str(lc_path)):
            r1, p1 = loop["robot1"], loop["pose1"]
            r2, p2 = loop["robot2"], loop["pose2"]
            if r1 == r2:
                continue

            key = frozenset(((r1, p1), (r2, p2)))
            if key in seen:
                continue
            seen.add(key)

            if r1 not in kf_maps or r2 not in kf_maps:
                continue
            if r1 not in trajectories or r2 not in trajectories:
                continue

            t1 = kf_maps[r1].get(p1)
            t2 = kf_maps[r2].get(p2)
            if t1 is None or t2 is None:
                continue

            traj1 = trajectories[r1]
            traj2 = trajectories[r2]
            idx1 = _nearest_index(t1, traj1["timestamps"], max_gap_s=max_gap_s)
            idx2 = _nearest_index(t2, traj2["timestamps"], max_gap_s=max_gap_s)
            if idx1 is None or idx2 is None:
                continue

            div1 = divergences.get(r1, {}).get(idx1)
            div2 = divergences.get(r2, {}).get(idx2)
            if div1 is None or div2 is None:
                continue

            records.append({
                "p1": traj1["positions"][idx1],
                "p2": traj2["positions"][idx2],
                "tv_total": float(div1 + div2),
                "idx1": idx1,
                "idx2": idx2,
                "r1": r1,
                "r2": r2,
            })
    return records


def plot(variant_dir: Path, cbs_dir: Path, max_gap_s: float, cmap_name: str):
    if not variant_dir.is_dir():
        print(f"Error: variant dir not found: {variant_dir}")
        sys.exit(1)
    if not cbs_dir.is_dir():
        print(f"Error: CBS dir not found: {cbs_dir}")
        sys.exit(1)

    id_to_name = _discover_robots_from_latest_tums(cbs_dir)
    if not id_to_name:
        print(f"Error: no timestamped Robot *.tum files found under {cbs_dir}")
        sys.exit(1)

    trajectories = _load_latest_trajectories(cbs_dir)
    divergences = _load_latest_divergence(cbs_dir)
    loop_records = _collect_loop_records(
        variant_dir=variant_dir,
        id_to_name=id_to_name,
        trajectories=trajectories,
        divergences=divergences,
        max_gap_s=max_gap_s,
    )
    if not loop_records:
        print("Error: no inter-robot loops could be matched to latest trajectories with divergence values")
        sys.exit(1)

    plt.rcParams.update(IEEE_RC)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    for rid, name in sorted(id_to_name.items()):
        traj = trajectories[rid]
        xy = traj["positions"][:, :2]
        color = ROBOT_COLORS[rid % len(ROBOT_COLORS)]
        ax.plot(xy[:, 0], xy[:, 1], "-", color=color, linewidth=1.2, alpha=0.95, label=name)
        ax.plot(xy[-1, 0], xy[-1, 1], "o", color=color, markersize=3, zorder=5)

    tv_vals = np.array([rec["tv_total"] for rec in loop_records], dtype=float)
    norm = mcolors.Normalize(vmin=float(np.min(tv_vals)), vmax=float(np.max(tv_vals)))
    cmap = plt.get_cmap(cmap_name)

    for rec in sorted(loop_records, key=lambda item: item["tv_total"]):
        xs = [rec["p1"][0], rec["p2"][0]]
        ys = [rec["p1"][1], rec["p2"][1]]
        ax.plot(xs, ys, "-", color=cmap(norm(rec["tv_total"])), linewidth=1.0, alpha=0.85)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Loop endpoint TV divergence sum")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{variant_dir.name} latest CBS+ loops colored by TV divergence")
    ax.legend(frameon=True, loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_base = cbs_dir / "loop_divergence_tv_map"
    save_fig(fig, str(out_base))
    print(f"Saved {out_base}.pdf/.png")
    print(f"Matched {len(loop_records)} inter-robot loops")
    print(f"TV total range: {float(np.min(tv_vals)):.6f} .. {float(np.max(tv_vals)):.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant_dir", type=Path, help="Original variant dir with distributed/ metadata")
    parser.add_argument("cbs_dir", type=Path, help="CBS or CBS+ dir containing timestamped Robot *.tum and divergence files")
    parser.add_argument("--max-gap", type=float, default=2.5, help="Max timestamp mismatch in seconds for matching loops to latest trajectory")
    parser.add_argument("--cmap", default="plasma", help="Matplotlib colormap for loop TV divergence")
    args = parser.parse_args()
    plot(args.variant_dir, args.cbs_dir, max_gap_s=args.max_gap, cmap_name=args.cmap)


if __name__ == "__main__":
    main()
