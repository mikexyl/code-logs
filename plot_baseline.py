#!/usr/bin/env python3
"""
Plot loop-closure statistics and evaluate ATE for Kimera-Multi baselines.

Usage:
    # Plot BoW matches + loop closures over time
    python plot_baseline.py baselines/campus

    # Also compute ATE (looks for GT in ground_truth/<exp>/<robot>.csv)
    python plot_baseline.py baselines/campus --ate

    # Override GT root folder
    python plot_baseline.py baselines/campus --ate --gt_folder /path/to/gt

    # Save plot to a specific path
    python plot_baseline.py baselines/campus --save output.pdf
"""

import argparse
import io
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


IEEE_RC = {
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.figsize': (3.5, 3.5 * 3 / 4),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'patch.linewidth': 0.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}


def load_lcd_log(csv_path: Path) -> pd.DataFrame:
    """
    Read an lcd_log.csv file.

    The header has 10 named columns but data rows have 15+ values plus a trailing
    comma, so we skip the header row and address columns by integer index:
      0: stamp_ns, 1: bow_matches, 4: num_loop_closures

    Returns a DataFrame with columns: stamp_ns, bow_matches, num_loop_closures.
    """
    df = pd.read_csv(
        csv_path,
        header=None,
        skiprows=1,
        usecols=[0, 1, 4],
        names=["stamp_ns", "bow_matches", "num_loop_closures"],
        skipinitialspace=True,
        on_bad_lines='skip',
    )
    df["stamp_ns"] = pd.to_numeric(df["stamp_ns"], errors="coerce")
    df["bow_matches"] = pd.to_numeric(df["bow_matches"], errors="coerce")
    df["num_loop_closures"] = pd.to_numeric(df["num_loop_closures"], errors="coerce")
    df = df.dropna(subset=["stamp_ns", "bow_matches", "num_loop_closures"])
    df = df.sort_values("stamp_ns").reset_index(drop=True)
    return df


def sum_series(dfs: list[pd.DataFrame], col: str, t_common_ns: np.ndarray) -> np.ndarray:
    """Interpolate each robot's series onto a common nanosecond axis and sum."""
    total = np.zeros(len(t_common_ns), dtype=np.float64)
    for df in dfs:
        t = df["stamp_ns"].values.astype(np.float64)
        v = df[col].values.astype(np.float64)
        interp = np.interp(t_common_ns, t, v, left=0.0, right=v[-1])
        total += interp
    return total


def plot_kimera_multi(experiment_dir: Path, output: Path | None = None) -> None:
    method_dir = experiment_dir / "Kimera-Multi"
    if not method_dir.exists():
        print(f"No Kimera-Multi folder found in {experiment_dir}")
        return

    lcd_files = sorted(method_dir.glob("*/distributed/lcd_log.csv"))
    if not lcd_files:
        print(f"No lcd_log.csv files found under {method_dir}")
        return

    print(f"Found {len(lcd_files)} robot(s):")
    dfs = []
    for p in lcd_files:
        robot = p.parent.parent.name
        df = load_lcd_log(p)
        print(f"  {robot}: {len(df)} rows, "
              f"bow_matches max={df['bow_matches'].max():.0f}, "
              f"loop_closures max={df['num_loop_closures'].max():.0f}")
        dfs.append(df)

    # Common time axis in nanoseconds
    t_min_ns = min(df["stamp_ns"].iloc[0] for df in dfs)
    t_max_ns = max(df["stamp_ns"].iloc[-1] for df in dfs)
    t_common_ns = np.linspace(t_min_ns, t_max_ns, 800)
    t_sec = (t_common_ns - t_min_ns) / 1e9

    bow_total = sum_series(dfs, "bow_matches", t_common_ns)
    loops_total = sum_series(dfs, "num_loop_closures", t_common_ns)

    # --- Plot ---
    plt.rcParams.update(IEEE_RC)
    fig, ax = plt.subplots()
    ax.plot(t_sec, loops_total, label="Loop Closures", color="#4C72B0")
    ax.plot(t_sec, bow_total,   label="BoW Matches",   color="#DD8452")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.3)
    ax.set_xlim(t_sec[0], t_sec[-1])

    exp_name = experiment_dir.name
    base_name = f"{exp_name}_kimera_multi_loops"
    base = experiment_dir / base_name
    if output is not None:
        base = output.with_suffix("")

    pdf_path = base.with_suffix(".pdf")
    png_path = base.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"Saved to {pdf_path}")
    print(f"Saved to {png_path}")
    plt.close(fig)

    # --- Save raw data ---
    npy_path = base.with_suffix(".npy")
    np.save(str(npy_path), {
        "t_sec": t_sec,
        "bow_matches": bow_total,
        "num_loop_closures": loops_total,
    }, allow_pickle=True)
    print(f"Saved to {npy_path}")


# ---------------------------------------------------------------------------
# Trajectory conversion helpers
# ---------------------------------------------------------------------------

def _kimera_pose_csv_to_tum(csv_path: Path) -> list[tuple[float, float, float, float, float, float, float, float]]:
    """
    Read a kimera_distributed_poses_*.csv and return rows as
    (ts_s, tx, ty, tz, qx, qy, qz, qw) sorted by timestamp.

    CSV columns: ns, pose_index, qx, qy, qz, qw, tx, ty, tz
    """
    rows = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('ns'):
                continue
            parts = line.split(',')
            if len(parts) < 9:
                continue
            try:
                ts_s = float(parts[0]) / 1e9
                qx, qy, qz, qw = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                tx, ty, tz     = float(parts[6]), float(parts[7]), float(parts[8])
                rows.append((ts_s, tx, ty, tz, qx, qy, qz, qw))
            except ValueError:
                continue
    rows.sort(key=lambda r: r[0])
    return rows


def _gt_csv_to_tum(csv_path: Path) -> list[tuple[float, float, float, float, float, float, float, float]]:
    """
    Read a ground_truth CSV and return rows as
    (ts_s, x, y, z, qx, qy, qz, qw) sorted by timestamp.

    CSV columns: #timestamp_kf(ns), x, y, z, qw, qx, qy, qz
    """
    rows = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 8:
                continue
            try:
                ts_s = float(parts[0]) / 1e9
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                qw, qx, qy, qz = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                rows.append((ts_s, x, y, z, qx, qy, qz, qw))
            except ValueError:
                continue
    rows.sort(key=lambda r: r[0])
    return rows


def _write_tum(rows: list[tuple], path: Path) -> None:
    with open(path, 'w') as f:
        for r in rows:
            f.write(f"{r[0]:.9f} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} "
                    f"{r[4]:.6f} {r[5]:.6f} {r[6]:.6f} {r[7]:.6f}\n")


# ---------------------------------------------------------------------------
# Alignment helpers (mirrors evaluate.py)
# ---------------------------------------------------------------------------

def _load_alignment(zip_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Load Sim(3)/SE(3) alignment from an evo_ape result zip."""
    with zipfile.ZipFile(zip_path) as z:
        if "alignment_transformation_sim3.npy" in z.namelist():
            with z.open("alignment_transformation_sim3.npy") as f:
                data = np.load(io.BytesIO(f.read()))
            scale = float(np.linalg.norm(data[:3, 0]))
            R = data[:3, :3] / scale
            t = data[:3, 3]
            return R, t, scale
        elif "alignment_transformation_se3.npy" in z.namelist():
            with z.open("alignment_transformation_se3.npy") as f:
                data = np.load(io.BytesIO(f.read()))
            return data[:3, :3], data[:3, 3], 1.0
    print("Warning: no alignment found in zip, using identity.")
    return np.eye(3), np.zeros(3), 1.0


def _apply_alignment(positions: np.ndarray, R: np.ndarray, t: np.ndarray, scale: float) -> np.ndarray:
    return scale * (R @ positions.T).T + t


# ---------------------------------------------------------------------------
# Trajectory plot (per-robot, aligned)
# ---------------------------------------------------------------------------

TRAJ_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]


def plot_trajectories_kimera_multi(
    method_dir: Path,
    robots_data: list[tuple[str, list[tuple], list[tuple]]],
    evo_zip: Path,
) -> None:
    """
    Plot per-robot estimated trajectories (Sim3-aligned) and GT trajectories.

    robots_data: list of (robot_name, est_rows, gt_rows)
      where each row is (ts_s, x, y, z, qx, qy, qz, qw)
    """
    R, t, scale = _load_alignment(evo_zip)
    print(f"Alignment — scale: {scale:.6f}, t: {t.round(3)}")

    plt.rcParams.update(IEEE_RC)
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    gt_labeled = False
    for idx, (robot, est_rows, gt_rows) in enumerate(robots_data):
        color = TRAJ_COLORS[idx % len(TRAJ_COLORS)]

        if est_rows:
            pos_est = np.array([[r[1], r[2], r[3]] for r in est_rows])
            pos_est_aligned = _apply_alignment(pos_est, R, t, scale)
            ax.plot(pos_est_aligned[:, 0], pos_est_aligned[:, 1],
                    color=color, linewidth=1.0, label=robot)

        if gt_rows:
            pos_gt = np.array([[r[1], r[2], r[3]] for r in gt_rows])
            gt_label = "Ground Truth" if not gt_labeled else None
            ax.plot(pos_gt[:, 0], pos_gt[:, 1],
                    color="gray", linewidth=0.5, linestyle="--",
                    alpha=0.6, label=gt_label)
            gt_labeled = True

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="best", framealpha=0.9, edgecolor="none")
    ax.grid(True, alpha=0.3, linewidth=0.3)
    plt.tight_layout(pad=0.5)

    for suffix in (".pdf", ".png"):
        out = method_dir / f"trajectories_aligned{suffix}"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.02, dpi=300)
        print(f"Saved to {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# ATE evaluation
# ---------------------------------------------------------------------------

def evaluate_ate_kimera_multi(experiment_dir: Path, gt_folder: Path) -> None:
    """
    Combine the last kimera_distributed_poses_*.csv from every robot under
    baselines/<exp>/Kimera-Multi/, match against GT files in gt_folder/<exp>/,
    and run evo_ape tum on the combined trajectories.
    """
    method_dir = experiment_dir / "Kimera-Multi"
    exp_name = experiment_dir.name
    gt_exp_dir = gt_folder / exp_name

    if not gt_exp_dir.exists():
        print(f"GT folder not found: {gt_exp_dir}")
        return

    robot_dirs = sorted(d for d in method_dir.iterdir() if d.is_dir())
    if not robot_dirs:
        print(f"No robot folders found under {method_dir}")
        return

    robots_data: list[tuple[str, list[tuple], list[tuple]]] = []
    all_est_rows: list[tuple] = []
    all_gt_rows:  list[tuple] = []

    for robot_dir in robot_dirs:
        robot = robot_dir.name
        dist_dir = robot_dir / "distributed"

        # Last kimera_distributed_poses_*.csv (alphabetical sort → highest index last)
        pose_files = sorted(dist_dir.glob("kimera_distributed_poses_*.csv"))
        if not pose_files:
            print(f"  {robot}: no kimera_distributed_poses_*.csv found, skipping")
            continue
        pose_csv = pose_files[-1]

        # Ground truth
        gt_csv = gt_exp_dir / f"{robot}.csv"
        if not gt_csv.exists():
            print(f"  {robot}: GT not found at {gt_csv}, skipping")
            continue

        est_rows = _kimera_pose_csv_to_tum(pose_csv)
        gt_rows  = _gt_csv_to_tum(gt_csv)

        print(f"  {robot}: {len(est_rows)} est poses  |  {len(gt_rows)} GT poses  "
              f"(using {pose_csv.name})")

        robots_data.append((robot, est_rows, gt_rows))
        all_est_rows.extend(est_rows)
        all_gt_rows.extend(gt_rows)

    if not all_est_rows:
        print("No estimated trajectories loaded — aborting ATE evaluation.")
        return

    # Sort combined trajectories by timestamp
    all_est_rows.sort(key=lambda r: r[0])
    all_gt_rows.sort(key=lambda r: r[0])

    evo_zip = method_dir / "evo_ape.zip"
    evo_pdf = method_dir / "evo_ape.pdf"
    for p in (evo_zip, evo_pdf):
        if p.exists():
            p.unlink()

    with tempfile.TemporaryDirectory() as tmp:
        est_tum = Path(tmp) / "combined_est.tum"
        gt_tum  = Path(tmp) / "combined_gt.tum"
        _write_tum(all_est_rows, est_tum)
        _write_tum(all_gt_rows,  gt_tum)

        cmd = [
            "evo_ape", "tum", str(gt_tum), str(est_tum),
            "-va", "--align",
            "--save_results", str(evo_zip),
            "--save_plot",    str(evo_pdf),
            "--plot_mode", "xy",
        ]
        print(f"\nRunning: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("\n" + "=" * 40)
            print("EVO APE RESULTS")
            print("=" * 40)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("evo_ape failed:")
            print(e.stderr)
            print(e.stdout)
            return

    print("\nPlotting aligned trajectories...")
    plot_trajectories_kimera_multi(method_dir, robots_data, evo_zip)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Kimera-Multi lcd_log statistics and optionally evaluate ATE."
    )
    parser.add_argument(
        "experiment_dir", type=Path,
        help="Path to the experiment folder (e.g. baselines/campus)"
    )
    parser.add_argument(
        "--save", type=Path, default=None, metavar="FILE",
        help="Save plot to FILE instead of the experiment folder"
    )
    parser.add_argument(
        "--ate", action="store_true",
        help="Compute ATE with evo_ape against ground truth"
    )
    parser.add_argument(
        "--gt_folder", type=Path, default=Path("ground_truth"),
        help="Root folder for ground truth CSVs (default: ground_truth/)"
    )
    args = parser.parse_args()

    exp_dir: Path = args.experiment_dir.resolve()
    if not exp_dir.exists():
        print(f"Error: {exp_dir} does not exist.")
        raise SystemExit(1)

    plot_kimera_multi(exp_dir, output=args.save)

    if args.ate:
        gt_folder = args.gt_folder.resolve()
        print(f"\n--- ATE Evaluation (GT: {gt_folder}) ---")
        evaluate_ate_kimera_multi(exp_dir, gt_folder)


if __name__ == "__main__":
    main()
