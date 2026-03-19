#!/usr/bin/env python3

import os
import re
import sys
import subprocess
import argparse
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils.io import (read_tum_trajectory, load_alignment_from_evo_zip,
                      load_frame_transform, load_keyframes_csv,
                      load_loop_closures_csv, is_robot_dir, discover_variants)
from utils.plot import apply_alignment, apply_frame_transform, find_tum_position


def collect_loop_closure_lines(pairs, rotation, translation, scale):
    """Resolve inter-robot loop closure endpoints in the aligned world frame.

    For each robot in pairs, looks for:
      <robot_dir>/distributed/kimera_distributed_keyframes.csv
      <robot_dir>/distributed/loop_closures.csv

    Uses the keyframe CSV to map pose indices → timestamps, then looks up
    the nearest TUM position and applies the same Sim3 alignment used for
    the estimated trajectories.

    Returns a list of (aligned_pos1, aligned_pos2) numpy arrays (shape (3,)).
    Silently skips if the distributed CSV files are not present.
    """
    # Build robot_id → (timestamps, raw_positions) from the TUM files in pairs.
    robot_trajs = {}   # {robot_id: (ts_array, pos_array)}
    robot_kf_maps = {} # {robot_id: {keyframe_id: ts_s}}

    for p in pairs:
        robot_path = Path(p['robot_path'])
        try:
            robot_id = int(robot_path.stem.split()[-1])
        except ValueError:
            continue
        ts, pos, _ = read_tum_trajectory(str(robot_path))
        robot_trajs[robot_id] = (np.array(ts), np.array(pos))

        kf_path = robot_path.parent.parent / 'distributed' / 'kimera_distributed_keyframes.csv'
        if kf_path.exists():
            robot_kf_maps[robot_id] = load_keyframes_csv(str(kf_path))

    if not robot_kf_maps:
        return []

    # Collect and deduplicate loop closures from every robot's distributed folder.
    seen = set()
    all_loops = []
    for p in pairs:
        robot_path = Path(p['robot_path'])
        lc_path = robot_path.parent.parent / 'distributed' / 'loop_closures.csv'
        if not lc_path.exists():
            continue
        for lc in load_loop_closures_csv(str(lc_path)):
            key = frozenset([(lc['robot1'], lc['pose1']), (lc['robot2'], lc['pose2'])])
            if key not in seen:
                seen.add(key)
                all_loops.append(lc)

    if not all_loops:
        return []

    # Resolve each loop closure to a pair of aligned world-frame positions.
    lines = []
    n_unresolved = 0
    for lc in all_loops:
        r1, p1 = lc['robot1'], lc['pose1']
        r2, p2 = lc['robot2'], lc['pose2']
        if r1 not in robot_trajs or r2 not in robot_trajs:
            n_unresolved += 1
            continue
        if r1 not in robot_kf_maps or r2 not in robot_kf_maps:
            n_unresolved += 1
            continue
        ts1 = robot_kf_maps[r1].get(p1)
        ts2 = robot_kf_maps[r2].get(p2)
        if ts1 is None or ts2 is None:
            n_unresolved += 1
            continue
        raw1 = find_tum_position(ts1, robot_trajs[r1][0], robot_trajs[r1][1])
        raw2 = find_tum_position(ts2, robot_trajs[r2][0], robot_trajs[r2][1])
        if raw1 is None or raw2 is None:
            n_unresolved += 1
            continue
        aligned1 = apply_alignment(raw1.reshape(1, 3), rotation, translation, scale)[0]
        aligned2 = apply_alignment(raw2.reshape(1, 3), rotation, translation, scale)[0]
        lines.append((aligned1, aligned2))

    print(f"Loop closures: {len(lines)} drawn, {n_unresolved} unresolved")
    return lines


def _save_trajectory_plot(fig, folder, stem):
    """Save full-size (pdf+png) and half-column (pdf+png) versions of a trajectory figure."""
    for suffix in ('.pdf', '.png'):
        out = os.path.join(folder, f"{stem}{suffix}")
        fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
        print(f"Saved to {out}")

    orig_size = fig.get_size_inches()
    fig.set_size_inches(1.67, 1.5)
    fig.tight_layout(pad=0.3)
    for suffix in ('.pdf', '.png'):
        out = os.path.join(folder, f"{stem}_half{suffix}")
        fig.savefig(out, bbox_inches='tight', pad_inches=0.01)
        print(f"Saved to {out}")

    fig.set_size_inches(*orig_size)
    fig.tight_layout(pad=0.5)


def plot_aligned_trajectories(experiment_folder, pairs, tf_gt_robot=None):
    """
    Plot aligned robot trajectories with different colors and labels.
    Reads the alignment transformation from evo_ape's saved results.
    Formatted for IEEE single-column journal standard.

    Saves two variants:
      trajectories_aligned_no_loops.pdf/png  – trajectories + GT only
      trajectories_aligned.pdf/png           – same + loop closure lines

    Args:
        tf_gt_robot: optional 4x4 numpy array transforming points from the
                     ground truth frame into the robot/world frame.  When
                     provided it is applied to the GT trajectory before
                     plotting so both trajectories share a common frame.
    """
    evo_zip_path = os.path.join(experiment_folder, "evo_ape.zip")

    if not os.path.exists(evo_zip_path):
        print(f"Error: {evo_zip_path} not found. Run evo_ape first.")
        return

    # Load alignment transformation
    rotation, translation, scale = load_alignment_from_evo_zip(evo_zip_path)
    print(f"Alignment - Scale: {scale:.6f}")
    print(f"Translation: {translation}")

    # IEEE single-column formatting with Times New Roman
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{times}',
        'font.family': 'serif',
        'font.serif': ['Times'],
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'legend.fontsize': 6,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'figure.figsize': (3.5, 2.2),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 0.5,
        'lines.linewidth': 1.0,
        'patch.linewidth': 0.5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    fig, ax = plt.subplots()

    # Color map for different robots
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(pairs))))

    def _robot_short_label(robot_path: str) -> str:
        """Convert 'Robot 3_1234567.tum' → 'R3'."""
        stem = os.path.basename(robot_path).replace('.tum', '')
        # stem is like "Robot 3" or "Robot 3_1234567"
        m = re.search(r'Robot\s+(\d+)', stem)
        return f'R{m.group(1)}' if m else stem

    # Plot each robot's estimated (aligned) trajectory
    for idx, p in enumerate(pairs):
        robot_path = p['robot_path']
        robot_name = os.path.basename(robot_path).replace('.tum', '')
        _, est_positions, _ = read_tum_trajectory(robot_path)
        if len(est_positions) == 0:
            print(f"Warning: Empty trajectory for {robot_name}")
            continue
        aligned_positions = apply_alignment(est_positions, rotation, translation, scale)
        ax.plot(aligned_positions[:, 0], aligned_positions[:, 1],
                color=colors[idx % len(colors)], linewidth=1.0,
                label=_robot_short_label(robot_path))

    # Plot ground truth (each robot separately to avoid jump lines)
    gt_plotted = False
    T_gt = tf_gt_robot if tf_gt_robot is not None else np.eye(4)
    for p in pairs:
        _, gt_positions, _ = read_tum_trajectory(p['gt_path'])
        if len(gt_positions) > 0:
            gt_positions = apply_frame_transform(gt_positions, T_gt)
            label = 'GT' if not gt_plotted else None
            ax.plot(gt_positions[:, 0], gt_positions[:, 1],
                    color='gray', linewidth=0.5, alpha=0.5,
                    linestyle='--', label=label)
            gt_plotted = True

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linewidth=0.3)

    def _place_legend(ax, fig):
        handles, labels = ax.get_legend_handles_labels()
        n = len(handles)
        ncol = n  # single row
        fig.legend(handles, labels, loc='lower center',
                   bbox_to_anchor=(0.5, -0.04), ncol=ncol,
                   framealpha=0.9, edgecolor='none',
                   handlelength=1.0, handletextpad=0.3, columnspacing=0.8)

    _place_legend(ax, fig)
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(bottom=0.18)

    # --- Version 1: no loop closures ---
    _save_trajectory_plot(fig, experiment_folder, "trajectories_aligned_no_loops")

    # --- Version 2: with loop closure lines ---
    loop_lines = collect_loop_closure_lines(pairs, rotation, translation, scale)
    lc_label_added = False
    for p1, p2 in loop_lines:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color='#CC2222', linewidth=1.5, alpha=0.8, zorder=10,
                label='Loop closure' if not lc_label_added else None)
        lc_label_added = True
    if loop_lines:
        _place_legend(ax, fig)
        plt.tight_layout(pad=0.3)
        plt.subplots_adjust(bottom=0.18)

    _save_trajectory_plot(fig, experiment_folder, "trajectories_aligned")

    plt.close()


def find_trajectory_pairs(experiment_folder, gt_folder=None, gt_exp_name=None):
    """
    Recursively finds 'Robot *.tum' files and their corresponding GT files.

    gt_exp_name overrides the subfolder name used when looking up GT files
    under gt_folder.  Defaults to the basename of experiment_folder, which
    is correct for top-level experiments.  Pass the parent experiment name
    when evaluating a variant subfolder (e.g. 'campus' for 'campus/all/').

    Sub-directories that are themselves variants (i.e. contain robot dirs) are
    skipped so that nested lm_optimized / other sub-variant TUM files are not
    mixed into the parent evaluation.

    Returns a list of dicts with keys: robot_path, gt_path, timestamp.
    """
    pairs = []
    if gt_exp_name is None:
        gt_exp_name = os.path.basename(experiment_folder)

    for root, dirs, files in os.walk(experiment_folder):
        # Prune variant sub-directories: skip any child dir that is itself a
        # variant (i.e. contains robot dirs inside it).  This prevents TUM
        # files from nested lm_optimized/ or other sub-variants being mixed
        # into the parent evaluation.
        dirs[:] = [
            d for d in dirs
            if not any(
                is_robot_dir(sub)
                for sub in (Path(root) / d).iterdir()
                if sub.is_dir()
            )
        ]
        # Deduplicate timestamped snapshots: for Robot <id>_<ts>.tum keep only
        # the latest non-empty file per robot ID in this directory.
        tum_files = [f for f in files if f.startswith("Robot ") and f.endswith(".tum")]
        latest_tum: dict[str, str] = {}  # robot_id_str → filename
        for f in tum_files:
            stem = f[:-4]  # strip .tum
            parts = stem.split("_")
            # "Robot N" → no underscore in id; "Robot N_ts" → parts[-1] is ts
            # Extract robot id as the token after "Robot "
            robot_id_str = stem[len("Robot "):]  # e.g. "0" or "0_1773654443"
            base_id = robot_id_str.split("_")[0]  # e.g. "0"
            existing = latest_tum.get(base_id)
            if existing is None:
                latest_tum[base_id] = f
            else:
                # Compare by timestamp suffix if present, else keep existing
                def _ts(fname):
                    s = fname[:-4][len("Robot "):].split("_")
                    return int(s[1]) if len(s) > 1 else -1
                if _ts(f) > _ts(existing):
                    latest_tum[base_id] = f
        # Only keep non-empty latest files
        deduped_files = [
            f for f in latest_tum.values()
            if os.path.getsize(os.path.join(root, f)) > 0
        ]

        for file in deduped_files:
            if file.startswith("Robot ") and file.endswith(".tum"):
                robot_path = os.path.join(root, file)

                if gt_folder is not None:
                    rel_subpath = os.path.relpath(root, experiment_folder)
                    first_subdir = rel_subpath.split(os.sep)[0] if rel_subpath != '.' else ''
                    stem = first_subdir if first_subdir else "gt"
                    gt_path = os.path.join(gt_folder, gt_exp_name, stem + ".txt")
                    if not os.path.exists(gt_path):
                        csv_candidate = os.path.join(gt_folder, gt_exp_name, stem + ".csv")
                        if os.path.exists(csv_candidate):
                            gt_path = csv_candidate
                else:
                    gt_path = os.path.join(root, "gt.txt")
                    if not os.path.exists(gt_path):
                        csv_candidate = os.path.join(root, "gt.csv")
                        if os.path.exists(csv_candidate):
                            gt_path = csv_candidate

                if os.path.exists(gt_path):
                    try:
                        with open(robot_path, 'r') as f:
                            for line in f:
                                if not line.startswith("#"):
                                    parts = line.strip().split()
                                    if parts:
                                        timestamp = float(parts[0])
                                        pairs.append({
                                            'robot_path': robot_path,
                                            'gt_path': gt_path,
                                            'timestamp': timestamp
                                        })
                                        break
                    except Exception as e:
                        print(f"Error reading {robot_path}: {e}")
                else:
                    print(f"Warning: No gt.txt/.csv found for {robot_path} (looked in {gt_path})")

    pairs.sort(key=lambda x: x['timestamp'])
    return pairs




def run_evaluation(variant_dir: str, gt_folder, gt_exp_name: str, tf_file):
    """Run ATE/RPE + trajectory plot for a single experiment or variant dir."""
    print(f"\n{'='*50}")
    print(f"Evaluating: {variant_dir}")
    print(f"{'='*50}")

    pairs = find_trajectory_pairs(variant_dir, gt_folder=gt_folder, gt_exp_name=gt_exp_name)
    if not pairs:
        print("No valid Robot *.tum and gt.txt pairs found.")
        return

    print(f"Found {len(pairs)} trajectory segments.")
    for p in pairs:
        print(f"  - {p['robot_path']} (t={p['timestamp']})")

    with tempfile.TemporaryDirectory() as temp_dir:
        combined_est_path = os.path.join(temp_dir, "combined_est.tum")
        combined_gt_path  = os.path.join(temp_dir, "combined_gt.tum")

        print("\nCombining trajectories...")

        with open(combined_est_path, 'w') as outfile:
            for p in pairs:
                with open(p['robot_path'], 'r') as infile:
                    outfile.write(infile.read())

        with open(combined_gt_path, 'w') as outfile:
            for p in pairs:
                if os.path.splitext(p['gt_path'])[1].lower() == '.csv':
                    with open(p['gt_path'], 'r') as infile:
                        for line in infile:
                            line = line.strip()
                            if not line or line.startswith('#'):
                                continue
                            parts = line.split(',')
                            if len(parts) >= 8:
                                try:
                                    ts_s = float(parts[0]) / 1e9
                                except ValueError:
                                    continue
                                x, y, z = parts[1], parts[2], parts[3]
                                qw, qx, qy, qz = parts[4], parts[5], parts[6], parts[7]
                                outfile.write(f'{ts_s:.9f} {x} {y} {z} {qx} {qy} {qz} {qw}\n')
                else:
                    with open(p['gt_path'], 'r') as infile:
                        outfile.write(infile.read())

        # evo_ape
        print("\nRunning evo ape...")
        ape_zip  = os.path.join(variant_dir, "evo_ape.zip")
        ape_plot = os.path.join(variant_dir, "evo.pdf")
        for f in [ape_zip, ape_plot]:
            if os.path.exists(f):
                os.remove(f)

        ape_txt  = os.path.join(variant_dir, "evo_ape.txt")
        cmd = ["evo_ape", "tum", combined_gt_path, combined_est_path, "-va",
               "--align", "--t_max_diff", "1.5",
               "--plot_mode", "xy",
               "--save_plot", ape_plot, "--save_results", ape_zip]
        print(f"Command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True)
            print("\n" + "="*40 + "\nEVO APE RESULTS\n" + "="*40)
            print(result.stdout)
            print("="*40)
            with open(ape_txt, 'w') as fout:
                fout.write(result.stdout)
        except subprocess.CalledProcessError as e:
            print("\nError running evo_ape:")
            print(e.stderr)
            print(e.stdout)
            return
        except FileNotFoundError:
            print("\nError: 'evo_ape' not found.")
            return

        # evo_rpe
        print("\nRunning evo rpe...")
        rpe_zip  = os.path.join(variant_dir, "evo_rpe.zip")
        rpe_txt  = os.path.join(variant_dir, "evo_rpe.txt")
        rpe_plot = os.path.join(variant_dir, "evo_rpe.pdf")
        for f in [rpe_zip, rpe_plot]:
            if os.path.exists(f):
                os.remove(f)

        rpe_cmd = ["evo_rpe", "tum", combined_gt_path, combined_est_path, "-va",
                   "--align", "--t_max_diff", "1.5",
                   "--delta", "5", "--delta_unit", "m",
                   "--plot_mode", "xy",
                   "--save_plot", rpe_plot, "--save_results", rpe_zip]
        print(f"Command: {' '.join(rpe_cmd)}")
        try:
            result = subprocess.run(rpe_cmd, check=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True)
            print("\n" + "="*40 + "\nEVO RPE RESULTS\n" + "="*40)
            print(result.stdout)
            print("="*40)
            with open(rpe_txt, 'w') as fout:
                fout.write(result.stdout)
        except subprocess.CalledProcessError as e:
            print("\nError running evo_rpe:")
            print(e.stderr)
            print(e.stdout)
            return
        except FileNotFoundError:
            print("\nError: 'evo_rpe' not found.")
            return

        # Trajectory plot
        print("\nPlotting aligned trajectories...")
        tf_gt_robot = load_frame_transform(tf_file)
        if tf_file:
            print(f"Applying GT→robot transform from: {tf_file}")
        plot_aligned_trajectories(variant_dir, pairs, tf_gt_robot=tf_gt_robot)


def main():
    parser = argparse.ArgumentParser(
        description="Combine TUM trajectories and run evo ATE evaluation."
    )
    parser.add_argument("experiment_folder", help="Path to the experiment folder")
    parser.add_argument(
        "--gt_folder", default="ground_truth",
        help=(
            "Root folder containing ground truth data. GT files are expected under "
            "<gt_folder>/<experiment_name>/<robot>.txt. Defaults to 'ground_truth'."
        ),
    )
    parser.add_argument(
        "--gt_exp_name", default=None,
        help=(
            "Override the experiment name used to look up GT files under gt_folder. "
            "Useful when evaluating a single variant dir directly, e.g. "
            "'evaluate.py campus/ns-cs --gt_exp_name campus'."
        ),
    )
    parser.add_argument(
        "--tf_file", default=None,
        help=(
            "Path to a file specifying the SE3 transform from the GT frame to the "
            "robot frame (JSON/YAML with 'matrix' key, or plain-text 4x4 matrix)."
        ),
    )

    args = parser.parse_args()

    experiment_folder = os.path.abspath(args.experiment_folder)
    if not os.path.exists(experiment_folder):
        print(f"Error: Folder {experiment_folder} does not exist.")
        sys.exit(1)

    gt_folder = os.path.abspath(args.gt_folder) if args.gt_folder else None

    exp_dir = Path(experiment_folder)
    variants = discover_variants(exp_dir)

    gt_exp_name = args.gt_exp_name if args.gt_exp_name else exp_dir.name
    if gt_folder:
        print(f"Ground truth root: {gt_folder}")
        print(f"  -> using subfolder: {os.path.join(gt_folder, gt_exp_name)}")

    if variants:
        print(f"Found {len(variants)} variant(s): {[v.name for v in variants]}")
        # If the experiment folder itself also has direct trajectory pairs
        # (i.e. it is both a variant and a container of sub-variants, e.g. when
        # lm_optimized sits alongside the original robot dirs), evaluate it too.
        direct_pairs = find_trajectory_pairs(
            experiment_folder, gt_folder=gt_folder, gt_exp_name=gt_exp_name
        )
        if direct_pairs:
            run_evaluation(experiment_folder, gt_folder, gt_exp_name, args.tf_file)
        for v in variants:
            run_evaluation(str(v), gt_folder, gt_exp_name, args.tf_file)
    else:
        run_evaluation(experiment_folder, gt_folder, gt_exp_name, args.tf_file)


if __name__ == "__main__":
    main()
