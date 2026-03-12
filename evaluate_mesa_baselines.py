#!/usr/bin/env python3
"""
Evaluate ATE for MESA baseline results (.jrr.cbor) against ground truth.

Pipeline:
  1. For each method, find the latest final_results.jrr.cbor in mesa_baselines/<method>/.
  2. Decode poses per robot, match timestamps from original Robot N.tum files by pose index.
  3. Write concatenated estimated + GT TUM files.
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
import numpy as np

from utils.io import read_tum_trajectory

METHODS = ["dgs", "asapp", "geodesic-mesa", "centralized"]

ATE_CSV = Path(__file__).parent / "ate_results.csv"
ATE_FIELDNAMES = ["experiment", "variant", "method",
                  "ate_rmse", "ate_mean", "ate_median", "ate_std", "ate_min", "ate_max",
                  "total_comm_rounds", "total_bytes_communicated"]


# ---------------------------------------------------------------------------
# Robot char → TUM path discovery
# ---------------------------------------------------------------------------

def build_robot_map(variant_dir: Path) -> dict[str, dict]:
    """Return {robot_char: {'tum': Path, 'robot_id': int}} by loading g2o files."""
    robot_map: dict[str, dict] = {}
    for g2o_path in sorted(variant_dir.glob("*/dpgo/bpsam_robot_*.g2o")):
        robot_id = int(g2o_path.stem.split("_")[-1])
        expected_char = chr(ord('a') + robot_id)
        # Find Robot N.tum in the same dpgo dir
        tum_path = g2o_path.parent / f"Robot {robot_id}.tum"
        if not tum_path.exists():
            # Try any Robot *.tum
            candidates = sorted(g2o_path.parent.glob("Robot *.tum"))
            tum_path = candidates[0] if candidates else None
        if tum_path is None:
            print(f"  Warning: no TUM file found for {g2o_path}, skipping")
            continue
        robot_map[expected_char] = {
            'tum': tum_path,
            'robot_id': robot_id,
            'robot_dir': g2o_path.parent.parent,  # e.g. g2345/ns-as/g2
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


def read_comm_stats(method_dir: Path) -> tuple[int, int] | tuple[None, None]:
    """Read total communication rounds and bytes from the latest iter_time_comm.txt.

    Returns (total_rounds, total_bytes) from the last data line, or (None, None) if unavailable.
    The file has 3 columns (pre-bytes tracking) or 4 columns (post-bytes tracking).
    """
    candidates = sorted(method_dir.glob("*/iter_time_comm.txt"))
    if not candidates:
        return None, None
    txt = candidates[-1]  # newest run
    last_data = None
    for line in txt.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        last_data = line.split()
    if last_data is None:
        return None, None
    try:
        total_rounds = int(last_data[2])
        total_bytes = int(last_data[3]) if len(last_data) >= 4 else None
        return total_rounds, total_bytes
    except (IndexError, ValueError):
        return None, None


def update_ate_csv(experiment: str, variant: str, method: str, stats: dict,
                   total_comm_rounds: int | None = None,
                   total_bytes_communicated: int | None = None) -> None:
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

    mesa_dir = variant_dir / "mesa_baselines"
    if not mesa_dir.exists():
        print(f"  No mesa_baselines/ dir found in {variant_dir}")
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

            stats = run_evo_ape(gt_tum, est_tum, out_dir, method)
            results[method] = stats

    # Summary table + CSV update
    csv_exp = exp_name or variant_dir.parent.name
    csv_variant = variant_dir.name
    print("\n" + "="*50)
    print(f"ATE RMSE Summary — {csv_exp}/{csv_variant}")
    print("="*50)
    print(f"{'Method':<20} {'ATE RMSE (m)':>14} {'Rounds':>10} {'Bytes':>14}")
    print("-"*60)
    for method in methods:
        s = results.get(method)
        method_dir = mesa_dir / method
        rounds, nbytes = read_comm_stats(method_dir) if method_dir.exists() else (None, None)
        rounds_str = str(rounds) if rounds is not None else "N/A"
        bytes_str = str(nbytes) if nbytes is not None else "N/A"
        if s is not None:
            print(f"{method:<20} {s.get('rmse', float('nan')):>14.4f} {rounds_str:>10} {bytes_str:>14}")
            update_ate_csv(csv_exp, csv_variant, method, s, rounds, nbytes)
        else:
            print(f"{method:<20} {'N/A':>14} {rounds_str:>10} {bytes_str:>14}")
    print("="*50)
    print(f"  Updated {ATE_CSV}")


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
