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
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from utils.io import (
    load_alignment_from_evo_zip,
    load_keyframes_csv,
    load_loop_closures_csv,
    read_tum_trajectory,
)
from utils.plot import IEEE_RC, ROBOT_COLORS, apply_alignment, save_fig


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
            "quaternions": quat_arr,
        }
    return trajectories


def _find_gt_path(gt_dir: Path, robot_name: str) -> Path | None:
    for ext in (".txt", ".csv"):
        path = gt_dir / f"{robot_name}{ext}"
        if path.exists():
            return path
    return None


def _write_combined_tum(path: Path, records: list[tuple[np.ndarray, np.ndarray]]) -> None:
    lines: list[str] = []
    for timestamps, data in records:
        for idx in range(len(timestamps)):
            tx, ty, tz = data[idx, 0:3]
            qx, qy, qz, qw = data[idx, 3:7]
            lines.append(
                f"{timestamps[idx]:.9f} {tx:.9f} {ty:.9f} {tz:.9f} "
                f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
            )
    lines.sort(key=lambda line: float(line.split()[0]))
    path.write_text("".join(lines))


def _run_evo_alignment(
    gt_dir: Path,
    variant_dir: Path,
    cbs_dir: Path,
    id_to_name: dict[int, str],
    trajectories: dict[int, dict],
) -> tuple[np.ndarray, np.ndarray, float]:
    zip_path = cbs_dir / "latest_evo_ape.zip"
    txt_path = cbs_dir / "latest_evo_ape.txt"
    for output_path in (zip_path, txt_path):
        if output_path.exists():
            output_path.unlink()
    est_records: list[tuple[np.ndarray, np.ndarray]] = []
    gt_records: list[tuple[np.ndarray, np.ndarray]] = []

    for rid, name in sorted(id_to_name.items()):
        gt_path = _find_gt_path(gt_dir, name)
        if gt_path is None:
            continue

        est = trajectories.get(rid)
        if est is None or est["timestamps"].size == 0:
            continue

        gt_ts, gt_pos, gt_quat = read_tum_trajectory(str(gt_path))
        if len(gt_ts) == 0:
            continue

        est_data = np.hstack([est["positions"], est["quaternions"]])
        gt_data = np.hstack([np.asarray(gt_pos, dtype=float), np.asarray(gt_quat, dtype=float)])
        est_records.append((est["timestamps"], est_data))
        gt_records.append((np.asarray(gt_ts, dtype=float), gt_data))

    if not est_records or not gt_records:
        print(f"Error: insufficient GT/estimate data for evo alignment in {variant_dir}")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        est_tum = tmp / "combined_est.tum"
        gt_tum = tmp / "combined_gt.tum"
        _write_combined_tum(est_tum, est_records)
        _write_combined_tum(gt_tum, gt_records)

        cmd = [
            "evo_ape", "tum", str(gt_tum), str(est_tum), "-va",
            "--align", "--t_max_diff", "1.5",
            "--plot_mode", "xy",
            "--save_results", str(zip_path),
        ]
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, "MPLBACKEND": "Agg"},
            )
        except subprocess.CalledProcessError as exc:
            print("Error running evo_ape:")
            print(exc.stderr)
            print(exc.stdout)
            sys.exit(1)
        txt_path.write_text(result.stdout)

    return load_alignment_from_evo_zip(str(zip_path))


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


def _build_char_to_rid(cbs_dir: Path) -> dict[str, int]:
    """
    Derive the GTSAM symbol character → robot_id mapping by reading the first
    vertex from each robot's latest g2o file.
    """
    char_to_rid: dict[str, int] = {}
    latest_g2os = _find_latest_robot_files(cbs_dir, r"cbs_robot_(\d+)_(\d+)\.g2o")
    for rid, g2o_path in latest_g2os.items():
        with g2o_path.open() as fh:
            for line in fh:
                if line.startswith("VERTEX_SE3:QUAT"):
                    vid = int(line.split()[1])
                    char_to_rid[chr(vid >> 56)] = rid
                    break
    return char_to_rid


def _collect_loop_records(
    cbs_dir: Path,
    id_to_name: dict[int, str],
    trajectories: dict[int, dict],
    divergences: dict[int, dict[int, float]],
) -> list[dict]:
    """
    Read inter-robot edges directly from the CBS g2o files and look up TV
    divergence by GTSAM Symbol index (= TUM array index = divergence file key).
    """
    char_to_rid = _build_char_to_rid(cbs_dir)
    print(f"GTSAM char → robot_id: {char_to_rid}")

    latest_g2os = _find_latest_robot_files(cbs_dir, r"cbs_robot_(\d+)_(\d+)\.g2o")

    seen: set[frozenset] = set()
    records: list[dict] = []
    n_no_rid = n_no_traj = n_out_of_range = n_no_div = 0

    for rid, g2o_path in latest_g2os.items():
        with g2o_path.open() as fh:
            for line in fh:
                if not line.startswith("EDGE"):
                    continue
                parts = line.split()
                vid1, vid2 = int(parts[1]), int(parts[2])
                c1, gi1 = chr(vid1 >> 56), int(vid1 & 0x00FFFFFFFFFFFFFF)
                c2, gi2 = chr(vid2 >> 56), int(vid2 & 0x00FFFFFFFFFFFFFF)
                if c1 == c2:
                    continue  # intra-robot edge

                key = frozenset(((c1, gi1), (c2, gi2)))
                if key in seen:
                    continue
                seen.add(key)

                r1 = char_to_rid.get(c1)
                r2 = char_to_rid.get(c2)
                if r1 is None or r2 is None:
                    n_no_rid += 1
                    continue
                if r1 not in trajectories or r2 not in trajectories:
                    n_no_traj += 1
                    continue

                traj1 = trajectories[r1]
                traj2 = trajectories[r2]
                if gi1 >= len(traj1["positions"]) or gi2 >= len(traj2["positions"]):
                    n_out_of_range += 1
                    continue

                div1 = divergences.get(r1, {}).get(gi1, 0.0)
                div2 = divergences.get(r2, {}).get(gi2, 0.0)
                if div1 == 0.0 and div2 == 0.0:
                    n_no_div += 1

                records.append({
                    "p1": traj1["positions"][gi1],
                    "p2": traj2["positions"][gi2],
                    "tv_total": float(div1 + div2),
                    "gi1": gi1,
                    "gi2": gi2,
                    "r1": r1,
                    "r2": r2,
                })

    print(
        f"CBS inter-robot edges: {len(seen)} unique | "
        f"no_rid={n_no_rid} no_traj={n_no_traj} out_of_range={n_out_of_range} "
        f"both_div_zero={n_no_div} matched={len(records)}"
    )
    return records


def plot(variant_dir: Path, cbs_dir: Path, max_gap_s: float, cmap_name: str):
    if not variant_dir.is_dir():
        print(f"Error: variant dir not found: {variant_dir}")
        sys.exit(1)
    if not cbs_dir.is_dir():
        print(f"Error: CBS dir not found: {cbs_dir}")
        sys.exit(1)
    gt_dir = Path("ground_truth") / variant_dir.parts[0]
    if not gt_dir.is_dir():
        print(f"Error: GT dir not found: {gt_dir}")
        sys.exit(1)

    id_to_name = _discover_robots_from_latest_tums(cbs_dir)
    if not id_to_name:
        print(f"Error: no timestamped Robot *.tum files found under {cbs_dir}")
        sys.exit(1)

    trajectories = _load_latest_trajectories(cbs_dir)
    divergences = _load_latest_divergence(cbs_dir)
    rotation, translation, scale = _run_evo_alignment(
        gt_dir=gt_dir,
        variant_dir=variant_dir,
        cbs_dir=cbs_dir,
        id_to_name=id_to_name,
        trajectories=trajectories,
    )
    loop_records = _collect_loop_records(
        cbs_dir=cbs_dir,
        id_to_name=id_to_name,
        trajectories=trajectories,
        divergences=divergences,
    )
    if not loop_records:
        print("Error: no inter-robot edges found in CBS g2o files")
        sys.exit(1)

    plt.rcParams.update(IEEE_RC)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    for rid, name in sorted(id_to_name.items()):
        gt_path = _find_gt_path(gt_dir, name)
        if gt_path is not None:
            _, gt_pos, _ = read_tum_trajectory(str(gt_path))
            gt_xy = np.asarray(gt_pos, dtype=float)[:, :2]
            if len(gt_xy) > 0:
                ax.plot(
                    gt_xy[:, 0], gt_xy[:, 1], "--",
                    color="0.75", linewidth=0.35, alpha=0.35,
                    label="Ground Truth" if rid == min(id_to_name) else None,
                )

        traj = trajectories[rid]
        aligned = apply_alignment(traj["positions"], rotation, translation, scale)
        xy = aligned[:, :2]
        color = ROBOT_COLORS[rid % len(ROBOT_COLORS)]
        ax.plot(xy[:, 0], xy[:, 1], "-", color=color, linewidth=0.45, alpha=0.22, label=name)
        ax.plot(xy[-1, 0], xy[-1, 1], "o", color=color, markersize=2, alpha=0.45, zorder=5)

    # --- Pose-level TV divergence scatter (for debugging / verification) ---
    fig_div, ax_div = plt.subplots(figsize=(7.0, 5.2))
    ax_div.set_title(f"{variant_dir.name} pose TV divergence (seq index)")
    all_pose_divs: list[float] = []
    for rid, name in sorted(id_to_name.items()):
        div_dict = divergences.get(rid, {})
        traj = trajectories[rid]
        aligned_all = apply_alignment(traj["positions"], rotation, translation, scale)
        # Try matching by sequential array index (0-based position in trajectory)
        scatter_xy: list[np.ndarray] = []
        scatter_dv: list[float] = []
        for i in range(len(traj["timestamps"])):
            dv = div_dict.get(i)
            if dv is not None:
                scatter_xy.append(aligned_all[i, :2])
                scatter_dv.append(dv)
                all_pose_divs.append(dv)
        print(
            f"  Robot {rid} ({name}): {len(div_dict)} divergence entries, "
            f"{len(scatter_dv)}/{len(traj['timestamps'])} matched by array index"
        )
        if scatter_xy:
            xy_arr = np.array(scatter_xy)
            sc = ax_div.scatter(
                xy_arr[:, 0], xy_arr[:, 1],
                c=scatter_dv, cmap="hot_r", s=4, alpha=0.8, zorder=4,
                label=name,
            )
        else:
            color = ROBOT_COLORS[rid % len(ROBOT_COLORS)]
            ax_div.plot(
                aligned_all[:, 0], aligned_all[:, 1], "-",
                color=color, linewidth=0.5, alpha=0.3, label=f"{name} (no div)",
            )
    if all_pose_divs:
        sm_div = plt.cm.ScalarMappable(
            norm=mcolors.Normalize(vmin=min(all_pose_divs), vmax=max(all_pose_divs)),
            cmap="hot_r",
        )
        sm_div.set_array([])
        cbar_div = fig_div.colorbar(sm_div, ax=ax_div, fraction=0.046, pad=0.03)
        cbar_div.set_label("TV divergence (per pose)")
    ax_div.set_aspect("equal", adjustable="box")
    ax_div.set_xlabel("x [m]")
    ax_div.set_ylabel("y [m]")
    ax_div.legend(frameon=True, loc="best", fontsize=8)
    ax_div.grid(True, alpha=0.25)
    fig_div.tight_layout()
    out_div = cbs_dir / "pose_divergence_tv_map"
    save_fig(fig_div, str(out_div))
    print(f"Saved {out_div}.pdf/.png")
    plt.close(fig_div)

    tv_vals = np.array([rec["tv_total"] for rec in loop_records], dtype=float)
    norm = mcolors.Normalize(vmin=float(np.min(tv_vals)), vmax=float(np.max(tv_vals)))
    cmap = plt.get_cmap(cmap_name)

    for rec in sorted(loop_records, key=lambda item: item["tv_total"]):
        p1 = apply_alignment(rec["p1"].reshape(1, 3), rotation, translation, scale)[0]
        p2 = apply_alignment(rec["p2"].reshape(1, 3), rotation, translation, scale)[0]
        xs = [p1[0], p2[0]]
        ys = [p1[1], p2[1]]
        ax.plot(xs, ys, "-", color=cmap(norm(rec["tv_total"])), linewidth=1.0, alpha=0.85)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Loop endpoint TV divergence sum")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{variant_dir.name} latest CBS+ in GT frame")
    ax.legend(frameon=True, loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_base = cbs_dir / "loop_divergence_tv_map"
    save_fig(fig, str(out_base))
    print(f"Saved {out_base}.pdf/.png")
    print(f"Saved {cbs_dir / 'latest_evo_ape.zip'}")
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
