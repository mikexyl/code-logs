#!/usr/bin/env python3
"""
Evaluate ATE for MESA baseline results (.jrr.cbor) against ground truth.

Pipeline:
  1. For each method, find the latest final_results.jrr.cbor in mesa_baselines/<method>/.
  2. Decode poses per robot, match timestamps from original Robot N.tum files by pose index.
  3. Write concatenated estimated + GT TUM files.
  For cbs/cbs_plus variants, uses the Robot <N>_<ts>.tum files saved by the method
  directly (already contain real timestamps), picking the last (highest suffix) per robot.
  4. Run evo_ape tum --align against ground truth.
  5. Print ATE RMSE table and update ate_results.csv.

Usage:
    python3 evaluate_mesa_baselines.py <variant_folder> [--gt_folder ground_truth]
    python3 evaluate_mesa_baselines.py g2345/ns-as
    python3 evaluate_mesa_baselines.py g2345/ns-as --gt_folder ground_truth --gt_exp_name g2345
"""

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import cbor2
import matplotlib.pyplot as plt
import numpy as np

from utils.io import read_tum_trajectory
from utils.plot import IEEE_RC, ROBOT_COLORS, save_fig

METHODS = ["dgs", "asapp", "geodesic-mesa", "centralized"]

ATE_CSV = Path(__file__).parent / "ate_results.csv"
ATE_FIELDNAMES = ["experiment", "variant", "method",
                  "ate_rmse", "ate_mean", "ate_median", "ate_std", "ate_min", "ate_max",
                  "total_comm_rounds", "total_bytes_communicated", "iterations_1pct"]


# ---------------------------------------------------------------------------
# Robot char → TUM path discovery
# ---------------------------------------------------------------------------

def build_robot_map(variant_dir: Path) -> dict[str, dict]:
    """Return {robot_char: {'tum': Path, 'robot_id': int}} by loading g2o files.

    Supports two layouts:
      Old: */dpgo/bpsam_robot_<id>.g2o  →  Robot <id>.tum
      New: */dpgo/cbs_robot_<id>_<ts>.g2o (latest)  →  Robot <id>_<ts>.tum (latest)
    """
    robot_map: dict[str, dict] = {}

    # Old layout
    for g2o_path in sorted(variant_dir.glob("*/dpgo/bpsam_robot_*.g2o")):
        robot_id = int(g2o_path.stem.split("_")[-1])
        expected_char = chr(ord('a') + robot_id)
        tum_path = g2o_path.parent / f"Robot {robot_id}.tum"
        if not tum_path.exists():
            candidates = sorted(g2o_path.parent.glob("Robot *.tum"))
            tum_path = candidates[0] if candidates else None
        if tum_path is None:
            print(f"  Warning: no TUM file found for {g2o_path}, skipping")
            continue
        robot_map[expected_char] = {
            'tum': tum_path,
            'robot_id': robot_id,
            'robot_dir': g2o_path.parent.parent,
        }

    if robot_map:
        return robot_map

    # New layout: cbs_robot_<id>_<ts>.g2o — pick latest per robot
    latest_g2o: dict[int, tuple[int, Path]] = {}
    for g2o_path in variant_dir.glob("*/dpgo/cbs_robot_*.g2o"):
        parts = g2o_path.stem.split("_")
        try:
            robot_id, ts = int(parts[2]), int(parts[3])
        except (IndexError, ValueError):
            continue
        if robot_id not in latest_g2o or ts > latest_g2o[robot_id][0]:
            latest_g2o[robot_id] = (ts, g2o_path)

    for robot_id, (ts, g2o_path) in sorted(latest_g2o.items()):
        expected_char = chr(ord('a') + robot_id)
        dpgo = g2o_path.parent
        # Pick latest non-empty Robot <id>_<ts>.tum
        candidates = sorted(
            [p for p in dpgo.glob(f"Robot {robot_id}_*.tum") if p.stat().st_size > 0],
            key=lambda p: int(p.stem.split("_")[-1])
        )
        tum_path = candidates[-1] if candidates else None
        if tum_path is None:
            print(f"  Warning: no TUM file found for robot {robot_id}, skipping")
            continue
        robot_map[expected_char] = {
            'tum': tum_path,
            'robot_id': robot_id,
            'robot_dir': g2o_path.parent.parent,
        }

    return robot_map


def find_gt_path(robot_dir: Path, gt_dir: Path) -> Path | None:
    """Find ground truth file for a robot dir. Tries .txt then .csv."""
    stem = robot_dir.name  # e.g. 'g2'
    for ext in ['.txt', '.csv']:
        p = gt_dir / (stem + ext)
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# JRR decoder
# ---------------------------------------------------------------------------

def load_jrr(jrr_path: Path) -> dict[str, list[tuple[int, np.ndarray, np.ndarray]]]:
    """Decode a .jrr.cbor file.

    Returns {robot_char: [(pose_index, translation_xyz, quat_xyzw), ...]} sorted by index.
    Only includes poses whose key char matches the robot char (own poses).
    """
    with open(jrr_path, 'rb') as f:
        d = cbor2.load(f)

    result: dict[str, list] = {}
    for robot_char, poses in d.get('solutions', {}).items():
        own_poses = []
        for p in poses:
            key = p['key']
            key_char = chr(key >> 56)
            if key_char != robot_char:
                continue  # cross-robot variable, skip
            pose_idx = key & 0x00FFFFFFFFFFFFFF
            t = np.array(p['translation'], dtype=float)  # xyz
            rot = p['rotation']  # [w, x, y, z]
            q_xyzw = np.array([rot[1], rot[2], rot[3], rot[0]], dtype=float)
            own_poses.append((pose_idx, t, q_xyzw))
        own_poses.sort(key=lambda x: x[0])
        result[robot_char] = own_poses
    return result


# ---------------------------------------------------------------------------
# Write combined TUM
# ---------------------------------------------------------------------------

def write_combined_tum(
    robot_map: dict[str, dict],
    jrr_poses: dict[str, list],
    out_path: Path,
) -> bool:
    """Write estimated poses (matched by pose_index → TUM row) as combined TUM."""
    lines = []
    for robot_char in sorted(robot_map.keys()):
        if robot_char not in jrr_poses:
            print(f"  Warning: no poses for robot {robot_char} in results")
            continue
        tum_path = robot_map[robot_char]['tum']
        # Load timestamps from source TUM (row index → timestamp)
        ts_list, _, _ = read_tum_trajectory(str(tum_path))
        ts_arr = np.array(ts_list)

        poses = jrr_poses[robot_char]
        n_tum = len(ts_arr)
        n_ok = 0
        for pose_idx, t_xyz, q_xyzw in poses:
            if pose_idx >= n_tum:
                print(f"  Warning: robot {robot_char} pose_idx={pose_idx} >= tum len={n_tum}")
                continue
            ts = ts_arr[pose_idx]
            tx, ty, tz = t_xyz
            qx, qy, qz, qw = q_xyzw
            lines.append(f"{ts:.9f} {tx:.9f} {ty:.9f} {tz:.9f} "
                          f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
            n_ok += 1
        print(f"  robot {robot_char}: {n_ok}/{len(poses)} poses written")

    if not lines:
        return False
    # Sort by timestamp for evo
    lines.sort(key=lambda l: float(l.split()[0]))
    out_path.write_text("".join(lines))
    return True


def write_combined_gt(robot_map: dict[str, dict], gt_dir: Path, out_path: Path) -> bool:
    """Write combined GT TUM (concatenate all robot GT files)."""
    lines = []
    for robot_char in sorted(robot_map.keys()):
        robot_dir = robot_map[robot_char]['robot_dir']
        gt_path = find_gt_path(robot_dir, gt_dir)
        if gt_path is None:
            print(f"  Warning: no GT file for {robot_dir.name} in {gt_dir}")
            continue
        ext = gt_path.suffix.lower()
        with open(gt_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if ext == '.csv':
                    parts = line.split(',')
                    if len(parts) < 8:
                        continue
                    try:
                        ts_s = float(parts[0]) / 1e9
                    except ValueError:
                        continue
                    x, y, z = parts[1], parts[2], parts[3]
                    qw, qx, qy, qz = parts[4], parts[5], parts[6], parts[7]
                    lines.append(f"{ts_s:.9f} {x} {y} {z} {qx} {qy} {qz} {qw}\n")
                else:
                    lines.append(line + "\n")
    if not lines:
        return False
    lines.sort(key=lambda l: float(l.split()[0]))
    out_path.write_text("".join(lines))
    return True


# ---------------------------------------------------------------------------
# Trajectory sanity-check plot
# ---------------------------------------------------------------------------

def plot_trajectories_sanity(
    robot_map: dict[str, dict],
    jrr_poses: dict[str, list],
    gt_dir: Path,
    out_path: Path,
    method: str,
) -> None:
    """Plot per-robot estimated vs GT trajectories in XY and save to PDF/PNG."""
    plt.rcParams.update({**IEEE_RC, 'figure.figsize': (4.0, 4.0)})
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(method)

    gt_plotted = False
    for i, robot_char in enumerate(sorted(robot_map.keys())):
        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        robot_dir = robot_map[robot_char]['robot_dir']

        # --- estimated ---
        if robot_char in jrr_poses:
            tum_path = robot_map[robot_char]['tum']
            ts_list, _, _ = read_tum_trajectory(str(tum_path))
            ts_arr = np.array(ts_list)
            poses = jrr_poses[robot_char]
            pts = []
            for pose_idx, t_xyz, _ in poses:
                if pose_idx < len(ts_arr):
                    pts.append(t_xyz[:2])
            if pts:
                pts = np.array(pts)
                ax.plot(pts[:, 0], pts[:, 1], color=color,
                        linewidth=0.6, label=f'robot {robot_char}')

        # --- GT (optional, never blocks the estimated plot) ---
        try:
            gt_path = find_gt_path(robot_dir, gt_dir)
            if gt_path is not None:
                gt_ts, gt_pos, _ = read_tum_trajectory(str(gt_path))
                if len(gt_pos) > 0:
                    gt_pos = np.array(gt_pos)
                    ax.plot(gt_pos[:, 0], gt_pos[:, 1], color=color,
                            linewidth=0.6, linestyle='--',
                            label='GT' if not gt_plotted else None)
                    gt_plotted = True
        except Exception as e:
            print(f"  Warning: could not load GT for robot {robot_char}: {e}")

    # Legend: proxy for dashed=GT only if GT was actually plotted
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    if gt_plotted:
        handles += [Line2D([0], [0], color='gray', linewidth=0.6, linestyle='--')]
        labels += ['GT']
    ax.legend(handles, labels, loc='best', framealpha=0.8, fontsize=6)
    plt.tight_layout(pad=0.5)
    save_fig(fig, out_path, suffixes=('.pdf', '.png'))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Run evo_ape
# ---------------------------------------------------------------------------

def run_evo_ape(gt_tum: Path, est_tum: Path, out_dir: Path, label: str) -> dict | None:
    """Run evo_ape tum and return stats dict, saving results in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{label}_evo_ape.zip"
    txt_path = out_dir / f"{label}_evo_ape.txt"
    plot_path = out_dir / f"{label}_evo_ape.pdf"
    for p in (zip_path, plot_path):
        p.unlink(missing_ok=True)
    cmd = [
        "evo_ape", "tum", str(gt_tum), str(est_tum),
        "-va", "--align", "--t_max_diff", "1.5",
        "--plot_mode", "xy",
        "--save_plot", str(plot_path),
        "--save_results", str(zip_path),
    ]
    print(f"  $ {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        with open(txt_path, 'w') as f:
            f.write(result.stdout)
        # Parse stats from stdout
        stats: dict[str, float] = {}
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) == 2 and parts[0] in ("max", "mean", "median", "min", "rmse", "std"):
                try:
                    stats[parts[0]] = float(parts[1])
                except ValueError:
                    pass
        if "rmse" in stats:
            return stats
    except subprocess.CalledProcessError as e:
        print(f"  evo_ape failed (rc={e.returncode}):\n{e.stderr}\n{e.stdout[-500:]}")
    except FileNotFoundError:
        print("  evo_ape not found")
    return None


def read_comm_stats(method_dir: Path) -> tuple[int, int, int] | tuple[None, None, None]:
    """Read communication stats at the first iteration within 1% of the final residual.

    If residual column (col 5) is present, finds the earliest iteration where
    residual <= 1.01 * final_residual and returns (rounds, bytes, iteration) at that point.
    Falls back to the last data line when residual column is absent.

    Returns (total_rounds, total_bytes, iteration) or (None, None, None) if unavailable.
    """
    candidates = sorted(method_dir.glob("*/iter_time_comm.txt"))
    if not candidates:
        return None, None, None
    txt = candidates[-1]  # newest run
    rows = []
    for line in txt.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        rows.append(line.split())
    if not rows:
        return None, None, None

    # Check if residual column is present
    if len(rows[0]) >= 5:
        try:
            residuals = [float(r[4]) for r in rows]
            final_residual = residuals[-1]
            threshold = 1.01 * final_residual
            for r, residual in zip(rows, residuals):
                if residual <= threshold:
                    return int(r[2]), int(r[3]) if len(r) >= 4 else None, int(r[0])
        except (IndexError, ValueError):
            pass

    # Fallback: use last row (no iteration granularity)
    last = rows[-1]
    try:
        return int(last[2]), int(last[3]) if len(last) >= 4 else None, int(last[0])
    except (IndexError, ValueError):
        return None, None, None


def update_ate_csv(experiment: str, variant: str, method: str, stats: dict,
                   total_comm_rounds: int | None = None,
                   total_bytes_communicated: int | None = None,
                   iterations_1pct: int | None = None) -> None:
    """Insert or update a row in ate_results.csv."""
    rows = []
    key = (experiment, variant, method)
    if ATE_CSV.exists():
        with open(ATE_CSV, newline="") as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames or []
            rows = list(reader)
        rows = [r for r in rows if (r["experiment"], r["variant"], r["method"]) != key]
    else:
        existing_fields = []
    row = {
        "experiment": experiment,
        "variant": variant,
        "method": method,
        "ate_rmse":   round(stats.get("rmse",   float("nan")), 4),
        "ate_mean":   round(stats.get("mean",   float("nan")), 4),
        "ate_median": round(stats.get("median", float("nan")), 4),
        "ate_std":    round(stats.get("std",    float("nan")), 4),
        "ate_min":    round(stats.get("min",    float("nan")), 4),
        "ate_max":    round(stats.get("max",    float("nan")), 4),
        "total_comm_rounds": total_comm_rounds if total_comm_rounds is not None else "",
        "total_bytes_communicated": total_bytes_communicated if total_bytes_communicated is not None else "",
        "iterations_1pct": iterations_1pct if iterations_1pct is not None else "",
    }
    rows.append(row)
    rows.sort(key=lambda r: (r["experiment"], r["variant"], r["method"]))
    with open(ATE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ATE_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_latest_jrr(method_dir: Path) -> Path | None:
    """Return the most recent final_results.jrr.cbor under method_dir."""
    candidates = sorted(method_dir.glob("**/final_results.jrr.cbor"))
    return candidates[-1] if candidates else None


def evaluate_variant(variant_dir: Path, gt_dir: Path, methods: list[str],
                     exp_name: str | None = None) -> None:
    print(f"\n=== Evaluating MESA baselines: {variant_dir} ===")

    csv_exp = exp_name or variant_dir.parent.name
    csv_variant = variant_dir.name

    mesa_dir = variant_dir / "mesa_baselines"
    if not mesa_dir.exists():
        print(f"  No mesa_baselines/ dir found in {variant_dir}")
        evaluate_dpgo_variants(variant_dir, csv_exp, csv_variant, gt_dir=gt_dir)
        return

    robot_map = build_robot_map(variant_dir)
    if not robot_map:
        print("  No robot g2o files found, cannot build robot map")
        return
    print(f"  Robots: {sorted(robot_map.keys())}")

    results: dict[str, dict | None] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Build GT TUM once
        gt_tum = tmp / "gt_combined.tum"
        if not write_combined_gt(robot_map, gt_dir, gt_tum):
            print("  Failed to write combined GT TUM")
            return

        for method in methods:
            method_dir = mesa_dir / method
            if not method_dir.exists():
                print(f"\n  [{method}] not found, skipping")
                continue

            jrr_path = find_latest_jrr(method_dir)
            if jrr_path is None:
                print(f"\n  [{method}] no final_results.jrr.cbor found, skipping")
                continue

            print(f"\n  [{method}] {jrr_path.relative_to(variant_dir)}")
            out_dir = mesa_dir  # write evo results to mesa_baselines/, which we own
            est_tum = tmp / f"{method}_est.tum"

            jrr_poses = load_jrr(jrr_path)
            if not write_combined_tum(robot_map, jrr_poses, est_tum):
                print(f"  [{method}] failed to write estimated TUM")
                results[method] = None
                continue

            plot_trajectories_sanity(
                robot_map, jrr_poses, gt_dir,
                out_dir / f"{method}_trajectories", method,
            )

            stats = run_evo_ape(gt_tum, est_tum, out_dir, method)
            results[method] = stats

    # Summary table + CSV update
    print("\n" + "="*50)
    print(f"ATE RMSE Summary — {csv_exp}/{csv_variant}")
    print("="*50)
    print(f"{'Method':<20} {'ATE RMSE (m)':>14} {'Rounds':>10} {'Bytes':>14} {'Iters':>10}")
    print("-"*72)
    for method in methods:
        s = results.get(method)
        method_dir = mesa_dir / method
        rounds, nbytes, iters = read_comm_stats(method_dir) if method_dir.exists() else (None, None, None)
        rounds_str = str(rounds) if rounds is not None else "N/A"
        bytes_str = str(nbytes) if nbytes is not None else "N/A"
        iters_str = str(iters) if iters is not None else "N/A"
        if s is not None:
            print(f"{method:<20} {s.get('rmse', float('nan')):>14.4f} {rounds_str:>10} {bytes_str:>14} {iters_str:>10}")
            update_ate_csv(csv_exp, csv_variant, method, s, rounds, nbytes, iters)
        else:
            print(f"{method:<20} {'N/A':>14} {rounds_str:>10} {bytes_str:>14} {iters_str:>10}")
    print("="*50)
    print(f"  Updated {ATE_CSV}")

    # Evaluate DPGO variants (cbs, cbs_plus) if present
    evaluate_dpgo_variants(variant_dir, csv_exp, csv_variant, gt_dir=gt_dir)


DPGO_METHODS = ["cbs", "cbs_plus"]


def read_dpgo_stats_at_1pct(method_dir: Path) -> tuple[int, int, int] | tuple[None, None, None]:
    """Read convergence stats from per-robot stats_robot_*.csv files.

    Sums bytes_sent across robots per iteration (not recv, to avoid double-counting).
    CBS/CBS+ residuals are non-monotonic (start near zero, rise, then oscillate), so
    1%-of-final convergence detection is unreliable — we report the last iteration
    (total rounds run). Each row corresponds to one synchronous algorithm round.
    Returns (total_bytes_sent, total_rounds, None).
    """
    stat_files = sorted(method_dir.glob("*/dpgo/stats_robot_*.csv"))
    if not stat_files:
        return None, None, None

    robot_data: list[list[dict]] = []
    for f in stat_files:
        rows = []
        with open(f) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    rows.append({
                        'iteration': int(row['iteration']),
                        'bytes_sent': int(row['bytes_sent']),
                    })
                except (KeyError, ValueError):
                    continue
        if rows:
            robot_data.append(rows)

    if not robot_data:
        return None, None, None

    n_iters = min(len(r) for r in robot_data)
    total_bytes_sent = sum(r[n_iters - 1]['bytes_sent'] for r in robot_data)
    total_rounds = robot_data[0][n_iters - 1]['iteration']
    return total_bytes_sent, total_rounds, None


def load_g2o_poses(g2o_path: Path) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Parse VERTEX_SE3:QUAT lines from a g2o file.

    Returns list of (pose_idx, translation_xyz, quat_xyzw), where pose_idx is
    decoded from the GTSAM symbol key as `key & 0x00FFFFFFFFFFFFFF`.
    """
    poses = []
    with open(g2o_path) as fh:
        for line in fh:
            if not line.startswith('VERTEX_SE3:QUAT'):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            key = int(parts[1])
            pose_idx = key & 0x00FFFFFFFFFFFFFF
            t = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
            q = np.array([float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])])
            poses.append((pose_idx, t, q))
    return poses


def evaluate_dpgo_ate(
    variant_dir: Path,
    method: str,
    gt_dir: Path,
    csv_exp: str,
    csv_variant: str,
) -> dict | None:
    """Evaluate ATE for a cbs/cbs_plus DPGO variant.

    Reads the last (highest timestamp suffix) non-empty Robot <N>_<ts>.tum file
    per robot — these already contain real timestamps — and concatenates them
    for evo_ape, without needing the original Robot N.tum for index lookup.
    """
    method_dir = variant_dir / method
    robot_subdirs = sorted(d for d in method_dir.iterdir() if d.is_dir())
    if not robot_subdirs:
        return None

    est_lines: list[str] = []
    gt_lines: list[str] = []

    for robot_subdir in robot_subdirs:
        dpgo_dir = robot_subdir / "dpgo"
        if not dpgo_dir.exists():
            continue

        # Find the last non-empty TUM file (highest timestamp suffix)
        tum_files = sorted(
            f for f in dpgo_dir.glob("Robot *.tum") if f.stat().st_size > 0
        )
        if not tum_files:
            print(f"  [{method}] no non-empty TUM files in {dpgo_dir}, skipping")
            continue
        last_tum = tum_files[-1]

        ts_list, pos_list, quat_list = read_tum_trajectory(str(last_tum))
        n_ok = len(ts_list)
        for ts, (tx, ty, tz), (qx, qy, qz, qw) in zip(ts_list, pos_list, quat_list):
            est_lines.append(
                f"{ts:.9f} {tx:.9f} {ty:.9f} {tz:.9f} "
                f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
            )
        print(f"  [{method}] {robot_subdir.name}: {n_ok} poses from {last_tum.name}")

        # GT for this robot
        orig_robot_dir = variant_dir / robot_subdir.name
        gt_path = find_gt_path(orig_robot_dir, gt_dir)
        if gt_path is None:
            print(f"  [{method}] no GT for {robot_subdir.name}")
            continue
        ext = gt_path.suffix.lower()
        with open(gt_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if ext == '.csv':
                    parts = line.split(',')
                    if len(parts) < 8:
                        continue
                    try:
                        ts_s = float(parts[0]) / 1e9
                    except ValueError:
                        continue
                    x, y, z = parts[1], parts[2], parts[3]
                    qw, qx, qy, qz = parts[4], parts[5], parts[6], parts[7]
                    gt_lines.append(f"{ts_s:.9f} {x} {y} {z} {qx} {qy} {qz} {qw}\n")
                else:
                    gt_lines.append(line + "\n")

    if not est_lines or not gt_lines:
        print(f"  [{method}] insufficient data for evo_ape")
        return None

    est_lines.sort(key=lambda l: float(l.split()[0]))
    gt_lines.sort(key=lambda l: float(l.split()[0]))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        est_tum = tmp / f"{method}_est.tum"
        gt_tum = tmp / f"{method}_gt.tum"
        est_tum.write_text("".join(est_lines))
        gt_tum.write_text("".join(gt_lines))

        out_dir = method_dir
        stats = run_evo_ape(gt_tum, est_tum, out_dir, method)

    return stats


def evaluate_dpgo_variants(variant_dir: Path, csv_exp: str, csv_variant: str,
                            gt_dir: Path | None = None) -> None:
    """Evaluate cbs/cbs_plus DPGO variants: comm stats + ATE."""
    found_any = False
    for method in DPGO_METHODS:
        method_dir = variant_dir / method
        if not method_dir.exists():
            continue
        found_any = True
        nbytes, iters, _ = read_dpgo_stats_at_1pct(method_dir)
        if nbytes is None:
            print(f"  [{method}] no stats_robot_*.csv found, skipping")
            continue
        mb = nbytes / 1e6
        size_str = f"{mb:.2f} MB" if mb >= 0.01 else f"{nbytes/1e3:.2f} KB"
        print(f"  [{method}] iters@1%={iters}  bytes_sent@1%={nbytes} ({size_str})")

        # ATE evaluation from g2o vertices
        ate_stats: dict = {}
        if gt_dir is not None:
            ate_result = evaluate_dpgo_ate(variant_dir, method, gt_dir, csv_exp, csv_variant)
            if ate_result:
                ate_stats = ate_result
                print(f"  [{method}] ATE RMSE = {ate_stats.get('rmse', float('nan')):.4f} m")
            else:
                # Preserve existing ATE stats from CSV
                if ATE_CSV.exists():
                    with open(ATE_CSV) as f:
                        for row in csv.DictReader(f):
                            if (row['experiment'] == csv_exp and row['variant'] == csv_variant
                                    and row['method'] == method):
                                ate_stats = {
                                    'rmse':   float(row.get('ate_rmse',   'nan') or 'nan'),
                                    'mean':   float(row.get('ate_mean',   'nan') or 'nan'),
                                    'median': float(row.get('ate_median', 'nan') or 'nan'),
                                    'std':    float(row.get('ate_std',    'nan') or 'nan'),
                                    'min':    float(row.get('ate_min',    'nan') or 'nan'),
                                    'max':    float(row.get('ate_max',    'nan') or 'nan'),
                                }
                                break

        update_ate_csv(csv_exp, csv_variant, method, ate_stats,
                       total_comm_rounds=None, total_bytes_communicated=nbytes,
                       iterations_1pct=iters)
    if found_any:
        print(f"  Updated {ATE_CSV} with DPGO variant stats")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MESA baselines ATE.")
    parser.add_argument("folder", type=Path, help="Variant folder (e.g. g2345/ns-as)")
    parser.add_argument("--gt_folder", type=Path, default=Path("ground_truth"),
                        help="Root ground truth folder (default: ground_truth)")
    parser.add_argument("--gt_exp_name", default=None,
                        help="GT experiment name if different from parent folder")
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        metavar="METHOD")
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.exists():
        print(f"Error: {folder} does not exist", file=sys.stderr)
        sys.exit(1)

    gt_folder = args.gt_folder
    if not gt_folder.is_absolute():
        gt_folder = Path(__file__).parent / gt_folder
    exp_name = args.gt_exp_name or folder.parent.name
    gt_dir = gt_folder / exp_name

    if not gt_dir.exists():
        print(f"Error: GT directory {gt_dir} not found", file=sys.stderr)
        sys.exit(1)

    evaluate_variant(folder, gt_dir, args.methods, exp_name=exp_name)


if __name__ == "__main__":
    main()
