#!/usr/bin/env python3
"""
Plot CBS vs CBS+ convergence snapshots as a 2×2 grid:
  rows: CBS, CBS+
  cols: ~20% iteration, final
Plus an ATE-vs-iterations curve panel at the bottom.

Usage:
    python3 plot_cbs_convergence.py <variant_dir>
    python3 plot_cbs_convergence.py campus/ns-as-incremental
"""

import argparse
import re
from pathlib import Path
from matplotlib.gridspec import GridSpec

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.io import load_alignment_from_evo_zip, load_gt_trajectory, umeyama
from utils.plot import IEEE_RC, ROBOT_COLORS, save_fig, apply_alignment


def get_sorted_timestamps(method_dir: Path, tol: int = 8) -> list[int]:
    """Return sorted representative timestamps, one per snapshot group.

    Robots save at slightly different wall-clock times.  We cluster all
    per-robot timestamps that fall within `tol` seconds of each other and
    return the minimum (earliest) timestamp per cluster as the group key.
    """
    all_ts: list[int] = []
    for dpgo in method_dir.glob("*/dpgo"):
        for f in dpgo.glob("Robot *_*.tum"):
            try:
                all_ts.append(int(f.stem.split("_")[-1]))
            except ValueError:
                pass
    all_ts = sorted(set(all_ts))
    if not all_ts:
        return []
    # Greedy clustering
    groups: list[int] = [all_ts[0]]
    for ts in all_ts[1:]:
        if ts - groups[-1] > tol:
            groups.append(ts)
    return groups


def count_snapshot_poses(method_dir: Path, ts: int, robot_dirs: list[str]) -> int:
    total = 0
    for rname in robot_dirs:
        for f in (method_dir / rname / "dpgo").glob(f"Robot *_{ts}.tum"):
            total += sum(1 for ln in open(f)
                         if ln.strip() and not ln.startswith("#"))
    return total


def pick_snapshot_indices(n: int, method_dir: Path,
                          tss: list[int], robot_dirs: list[str],
                          min_poses: int = 100) -> tuple[int, int, int]:
    """Return (early, mid, final) indices, skipping nearly-empty snapshots."""
    final = n - 1
    mid   = round(0.50 * (n - 1))
    # early: first snapshot with enough poses
    early = 0
    for i in range(n - 1):
        if count_snapshot_poses(method_dir, tss[i], robot_dirs) >= min_poses:
            early = i
            break
    # pick ~10% but at least the first meaningful one
    target_10 = max(early, round(0.10 * (n - 1)))
    early = target_10
    return early, mid, final


def get_dpgo_robot_id(method_dir: Path, rname: str) -> int | None:
    """Return the DPGO robot ID from the TUM filename (e.g. 'Robot 3_*.tum' → 3)."""
    for f in (method_dir / rname / "dpgo").glob("Robot *_*.tum"):
        parts = f.stem.split("_")
        try:
            return int(parts[0].split()[-1])
        except (ValueError, IndexError):
            pass
    return None


def load_snapshot(method_dir: Path, ts: int,
                  robot_dirs: list[str], tol: int = 8) -> dict[str, np.ndarray]:
    """Load {robot_dir_name: Nx3 positions} for a given snapshot group timestamp."""
    out = {}
    for rname in robot_dirs:
        dpgo = method_dir / rname / "dpgo"
        # Find the TUM file whose timestamp is closest to ts within tolerance
        candidates = []
        for f in dpgo.glob("Robot *_*.tum"):
            try:
                fts = int(f.stem.split("_")[-1])
            except ValueError:
                continue
            if abs(fts - ts) <= tol:
                candidates.append((abs(fts - ts), f))
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0])
        tum = candidates[0][1]
        rows = []
        with open(tum) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = line.split()
                if len(p) < 4:
                    continue
                rows.append([float(p[1]), float(p[2]), float(p[3])])
        if not rows:
            continue
        pos = np.array(rows)
        # filter stray timestamps (last line sometimes has unix write-time)
        median_t = np.median([float(line.split()[0])
                              for line in open(tum)
                              if line.strip() and not line.startswith("#")
                              and len(line.split()) >= 4])
        ts_arr = np.array([float(line.split()[0])
                           for line in open(tum)
                           if line.strip() and not line.startswith("#")
                           and len(line.split()) >= 4])
        mask = np.abs(ts_arr - median_t) < 1e7
        out[rname] = pos[mask]
    return out


def get_iteration_for_ts(method_dir: Path, ts: int) -> int | None:
    """Return the iteration number closest to timestamp ts from stats CSVs."""
    # timestamps are every 10 s; find stats row whose cumulative position matches
    # We use the sorted timestamp list to determine iteration index.
    # stats_robot_0.csv has 1 row per iteration; snapshots are evenly spaced.
    stats_files = list(method_dir.glob("*/dpgo/stats_robot_*.csv"))
    if not stats_files:
        return None
    tss = get_sorted_timestamps(method_dir)
    if ts not in tss:
        return None
    idx = tss.index(ts)
    n_snaps = len(tss)
    # read total iterations from any stats file
    try:
        import csv
        with open(stats_files[0]) as f:
            rows = list(csv.DictReader(f))
        total_iters = int(rows[-1]['iteration']) if rows else None
    except Exception:
        total_iters = None
    if total_iters and n_snaps > 1:
        iters_per_snap = total_iters / (n_snaps - 1)
        return round(idx * iters_per_snap) if idx > 0 else 1
    return None


def load_snapshot_with_ts(method_dir: Path, ts: int,
                          robot_order: list[str], tol: int = 8) -> dict[str, tuple]:
    """Load {robot_name: (timestamps_s, Nx3 positions)} for a snapshot group."""
    out = {}
    for rname in robot_order:
        dpgo = method_dir / rname / "dpgo"
        candidates = []
        for f in dpgo.glob("Robot *_*.tum"):
            try:
                fts = int(f.stem.split("_")[-1])
            except ValueError:
                continue
            if abs(fts - ts) <= tol:
                candidates.append((abs(fts - ts), f))
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0])
        rows = []
        with open(candidates[0][1]) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = line.split()
                if len(p) < 4:
                    continue
                rows.append([float(p[0]), float(p[1]), float(p[2]), float(p[3])])
        if not rows:
            continue
        arr = np.array(rows)
        median_t = np.median(arr[:, 0])
        mask = np.abs(arr[:, 0] - median_t) < 1e7
        out[rname] = (arr[mask, 0], arr[mask, 1:4])
    return out


def compute_ate_rmse(robot_ts_pos: dict, robot_order: list[str],
                     gt_ts_pos: dict, alignment, tol_s: float = 0.5) -> float | None:
    """Compute ATE RMSE (m) for a snapshot against GT, using nearest-neighbour matching."""
    errors = []
    for rname in robot_order:
        data = robot_ts_pos.get(rname)
        gt_data = gt_ts_pos.get(rname)
        if data is None or gt_data is None:
            continue
        ts_arr_s, pos = data
        if alignment is not None:
            pos = apply_alignment(pos, *alignment)
        gt_ts_ns, gt_pos = gt_data
        gt_ts_s = gt_ts_ns / 1e9
        for ts_s, p in zip(ts_arr_s, pos):
            idx = np.argmin(np.abs(gt_ts_s - ts_s))
            if np.abs(gt_ts_s[idx] - ts_s) <= tol_s:
                errors.append(np.linalg.norm(p - gt_pos[idx]))
    if not errors:
        return None
    return float(np.sqrt(np.mean(np.array(errors) ** 2)))


def compute_all_ates(method_dir: Path, tss: list[int], robot_order: list[str],
                     gt_ts_pos: dict) -> tuple[list, list]:
    """Compute ATE RMSE at every snapshot. Returns (iteration_numbers, ate_values).

    Alignment is derived via Umeyama from the final snapshot so it is correct
    for this method's optimization frame (which differs from the main DPGO frame).
    """
    import csv as _csv
    stats_files = list(method_dir.glob("*/dpgo/stats_robot_*.csv"))
    total_iters = None
    if stats_files:
        try:
            with open(stats_files[0]) as f:
                rows = list(_csv.DictReader(f))
            if rows:
                total_iters = int(rows[-1]['iteration'])
        except Exception:
            pass

    n = len(tss)
    if n == 0:
        return [], []

    # Compute Umeyama alignment from the final snapshot
    final_ts_pos = load_snapshot_with_ts(method_dir, tss[-1], robot_order)
    src_pts, dst_pts = [], []
    for rname in robot_order:
        data = final_ts_pos.get(rname)
        gt_data = gt_ts_pos.get(rname)
        if data is None or gt_data is None:
            continue
        ts_arr_s, pos = data
        gt_ts_ns, gt_pos = gt_data
        gt_ts_s = gt_ts_ns / 1e9
        for ts_s, p in zip(ts_arr_s, pos):
            idx = np.argmin(np.abs(gt_ts_s - ts_s))
            if np.abs(gt_ts_s[idx] - ts_s) <= 0.5:
                src_pts.append(p)
                dst_pts.append(gt_pos[idx])
    if len(src_pts) < 3:
        print(f"  Warning: not enough matched poses for Umeyama in {method_dir.name}")
        return [], []
    R, t, s = umeyama(np.array(src_pts), np.array(dst_pts))
    alignment = (R, t, s)

    iterations, ates = [], []
    for snap_idx, ts in enumerate(tss):
        robot_ts_pos = load_snapshot_with_ts(method_dir, ts, robot_order)
        ate = compute_ate_rmse(robot_ts_pos, robot_order, gt_ts_pos, alignment)
        if ate is None:
            continue
        if total_iters is not None and n > 1:
            iter_num = round(snap_idx * total_iters / (n - 1)) if snap_idx > 0 else 1
        else:
            iter_num = snap_idx
        iterations.append(iter_num)
        ates.append(ate)
    return iterations, ates


def plot_snapshot(ax, positions: dict[str, np.ndarray],
                  robot_order: list[str],
                  dpgo_ids: dict[str, int],
                  alignment, title: str,
                  gt_positions: dict[str, np.ndarray] | None = None):
    # Draw GT first (behind estimates)
    if gt_positions:
        for rname, gt_pos in gt_positions.items():
            rid = robot_order.index(rname) if rname in robot_order else 0
            color = ROBOT_COLORS[rid % len(ROBOT_COLORS)]
            ax.plot(gt_pos[:, 0], gt_pos[:, 1],
                    lw=0.5, color=color, alpha=0.3, linestyle='--')
    for rname in robot_order:
        pos = positions.get(rname)
        if pos is None or len(pos) == 0:
            continue
        rid = robot_order.index(rname)
        color = ROBOT_COLORS[rid % len(ROBOT_COLORS)]
        if alignment is not None:
            pos = apply_alignment(pos, *alignment)
        ax.plot(pos[:, 0], pos[:, 1], lw=0.6, color=color)
        ax.plot(pos[-1, 0], pos[-1, 1], 'o', ms=1.8, color=color)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3, linewidth=0.3)
    ax.tick_params(labelsize=14, pad=1)
    ax.text(0.03, 0.03, title, transform=ax.transAxes,
            fontsize=18, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75, ec='none'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('variant_dir')
    ap.add_argument('--gt_dir', default=None,
                    help='Ground truth directory (default: auto-detect)')
    ap.add_argument('--output', default=None)
    ap.add_argument('--full', action='store_true',
                    help='Generate full multi-snapshot plot (all snapshots)')
    ap.add_argument('--final-only', action='store_true',
                    help='Generate a single-column plot showing only the final trajectory')
    args = ap.parse_args()

    vdir = Path(args.variant_dir)
    if not vdir.is_absolute():
        vdir = Path(__file__).parent / vdir

    # Load alignment from evo_ape.zip
    evo_zip = vdir / 'evo_ape.zip'
    alignment = None  # (rotation, translation, scale)
    if evo_zip.exists():
        try:
            alignment = load_alignment_from_evo_zip(str(evo_zip))
            print(f"Loaded alignment from {evo_zip.name}")
        except Exception as e:
            print(f"Warning: could not load alignment: {e}")

    # Load GT trajectories
    gt_positions: dict[str, np.ndarray] = {}   # robot_dir_name -> Nx3
    gt_ts_pos: dict[str, tuple] = {}           # robot_dir_name -> (timestamps_ns, Nx3)
    if args.gt_dir:
        gt_dir = Path(args.gt_dir)
    else:
        # Auto-detect: ground_truth/<experiment_name>
        exp_name = vdir.parent.name
        gt_dir = Path(__file__).parent / 'ground_truth' / exp_name
    if gt_dir.exists():
        for csv_path in gt_dir.glob('*.csv'):
            rname = csv_path.stem
            if rname.startswith('gt_'):
                continue
            try:
                ts_ns, pos, _ = load_gt_trajectory(str(csv_path))
                gt_positions[rname] = pos
                gt_ts_pos[rname] = (ts_ns, pos)
            except Exception:
                pass
        print(f"Loaded GT for {len(gt_positions)} robots from {gt_dir}")
    else:
        print(f"No GT dir found at {gt_dir}")

    methods = {}
    for name in ['cbs', 'cbs_plus']:
        mdir = vdir / name
        if mdir.exists():
            methods[name] = mdir

    if not methods:
        print("No cbs/ or cbs_plus/ dirs found.")
        return

    # Discover robot dirs (consistent across methods)
    all_robots: set[str] = set()
    for mdir in methods.values():
        for d in mdir.iterdir():
            if d.is_dir() and (d / 'dpgo').is_dir():
                all_robots.add(d.name)
    robot_order = sorted(all_robots)

    # Map robot dir name → DPGO robot ID (from TUM filename)
    dpgo_ids: dict[str, int] = {}
    for mdir in methods.values():
        for rname in robot_order:
            if rname not in dpgo_ids:
                rid = get_dpgo_robot_id(mdir, rname)
                if rid is not None:
                    dpgo_ids[rname] = rid

    method_labels = {'cbs': 'CBS', 'cbs_plus': 'CBS+'}

    # Collect snapshot index lists per method
    method_snap_indices: dict[str, list[int]] = {}
    all_tss: dict[str, list[int]] = {}
    for name, mdir in methods.items():
        tss = get_sorted_timestamps(mdir)
        all_tss[name] = tss
        n = len(tss)
        if n == 0:
            continue
        if args.full:
            method_snap_indices[name] = list(range(n))
        elif args.final_only:
            method_snap_indices[name] = [n - 1]
        else:
            i_20 = round(0.20 * (n - 1))
            i_f  = n - 1
            method_snap_indices[name] = [i_20, i_f]

    n_methods = len(methods)
    # For --final-only: one row, one column per method
    if args.final_only:
        n_cols = n_methods
        n_traj_rows = 1
    else:
        n_cols = max((len(v) for v in method_snap_indices.values()), default=3)
        n_traj_rows = n_methods

    col_w, row_h, ate_h, spacer_h = 3.5, 3.0, 2.0, 0.6
    plt.rcParams.update({
        **IEEE_RC,
        'figure.figsize': (col_w * n_cols, row_h * n_traj_rows + spacer_h + ate_h),
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
    })
    fig = plt.figure()
    gs = GridSpec(n_traj_rows + 2, n_cols,
                  height_ratios=[row_h] * n_traj_rows + [spacer_h, ate_h],
                  hspace=0.05, wspace=0.02,
                  figure=fig)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)]
            for r in range(n_traj_rows)]
    # Share x/y among trajectory panels
    for r in range(n_traj_rows):
        for c in range(n_cols):
            if r > 0 or c > 0:
                axes[r][c].sharex(axes[0][0])
                axes[r][c].sharey(axes[0][0])
    ate_ax = fig.add_subplot(gs[n_traj_rows + 1, :])

    # First pass: load all snapshots and compute global bounds from finals
    all_snapshots = {}  # (row_idx, col_idx) -> (name, positions)
    global_x, global_y = [], []

    for row_idx, (name, mdir) in enumerate(methods.items()):
        tss = all_tss.get(name, [])
        snap_idxs = method_snap_indices.get(name, [])
        for col_idx, snap_idx in enumerate(snap_idxs):
            ts = tss[snap_idx]
            positions = load_snapshot(mdir, ts, robot_order)
            all_snapshots[(row_idx, col_idx)] = (name, positions)
            if snap_idx == snap_idxs[-1]:  # final snapshot
                for rname in robot_order:
                    pos = positions.get(rname)
                    if pos is not None and len(pos):
                        if alignment is not None:
                            pos = apply_alignment(pos, *alignment)
                        global_x.extend(pos[:, 0])
                        global_y.extend(pos[:, 1])

    pad = 20
    xmin = min(global_x) - pad if global_x else -100
    xmax = max(global_x) + pad if global_x else  100
    ymin = min(global_y) - pad if global_y else -100
    ymax = max(global_y) + pad if global_y else  100
    # Force equal x/y range so adjustable='box' doesn't leave gaps
    xrange = xmax - xmin
    yrange = ymax - ymin
    if xrange > yrange:
        cy = (ymin + ymax) / 2
        ymin, ymax = cy - xrange / 2, cy + xrange / 2
    else:
        cx = (xmin + xmax) / 2
        xmin, xmax = cx - yrange / 2, cx + yrange / 2

    # Column labels
    if args.full:
        col_labels_map: dict[tuple, str] = {}
        for name, snap_idxs in method_snap_indices.items():
            n = len(all_tss.get(name, []))
            for col_idx, snap_idx in enumerate(snap_idxs):
                pct = int(round(100 * snap_idx / max(n - 1, 1)))
                col_labels_map[(name, col_idx)] = f"{pct}%"
    elif args.final_only:
        col_labels_map = {(name, 0): 'Final' for name in methods}
    else:
        fixed = ['100 iterations', '500 iterations']
        col_labels_map = {(name, i): fixed[i]
                          for name in methods for i in range(2)}

    # Second pass: draw
    for (row_idx, col_idx), (name, positions) in all_snapshots.items():
        # --final-only: each method is a column in the single traj row
        if args.final_only:
            ax = axes[0][row_idx]
        else:
            ax = axes[row_idx][col_idx]
        lbl = col_labels_map.get((name, col_idx), '')
        title = f"{method_labels.get(name, name)} {lbl}"
        plot_snapshot(ax, positions, robot_order, dpgo_ids, alignment, title,
                      gt_positions=gt_positions)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    # Hide unused axes (if methods have different snapshot counts)
    if not args.final_only:
        for row_idx in range(n_methods):
            name = list(methods.keys())[row_idx]
            used = len(method_snap_indices.get(name, []))
            for col_idx in range(used, n_cols):
                axes[row_idx][col_idx].set_visible(False)

    # Axis labels only on outer edges
    for r in range(n_traj_rows):
        axes[r][0].set_ylabel('y (m)', fontsize=16, labelpad=1)
        for c in range(1, n_cols):
            plt.setp(axes[r][c].get_yticklabels(), visible=False)
            axes[r][c].tick_params(axis='y', length=0)
    for c in range(n_cols):
        axes[-1][c].set_xlabel('x (m)', fontsize=16, labelpad=1)
        for r in range(n_traj_rows - 1):
            plt.setp(axes[r][c].get_xticklabels(), visible=False)
            axes[r][c].tick_params(axis='x', length=0)

    # ATE vs iterations curve
    method_colors = {'cbs': '#4C72B0', 'cbs_plus': '#DD8452'}
    if gt_ts_pos:
        print("Computing ATE vs iterations...")
        for name, mdir in methods.items():
            tss = all_tss.get(name, [])
            iters, ates = compute_all_ates(mdir, tss, robot_order, gt_ts_pos)
            if iters:
                label = method_labels.get(name, name)
                color = method_colors.get(name, None)
                ate_ax.plot(iters, ates, marker='o', ms=3, lw=1.2,
                            label=label, color=color)
                print(f"  {name}: {len(iters)} snapshots, "
                      f"final ATE={ates[-1]:.3f} m")
    ate_ax.set_yscale('log')
    ate_ax.set_xlabel('Iteration', fontsize=16)
    ate_ax.set_ylabel('ATE (m)', fontsize=16)
    ate_ax.legend(fontsize=14, framealpha=0.9)
    ate_ax.grid(True, alpha=0.3, linewidth=0.3)
    ate_ax.tick_params(labelsize=14)

    plt.tight_layout(pad=0.5, h_pad=0.3, w_pad=0.0)
    # Align ATE axes left/right edges to the trajectory grid
    fig.canvas.draw()
    traj_x0 = min(axes[r][0].get_position().x0 for r in range(n_traj_rows))
    traj_x1 = max(axes[r][n_cols - 1].get_position().x1
                  for r in range(n_traj_rows)
                  if axes[r][n_cols - 1].get_visible())
    pos = ate_ax.get_position()
    ate_ax.set_position([traj_x0, pos.y0, traj_x1 - traj_x0, pos.height])

    suffix = '_full' if args.full else ('_final' if args.final_only else '')
    out = Path(args.output) if args.output else vdir / f'cbs_convergence{suffix}'
    save_fig(fig, out)
    plt.close(fig)
    print(f"Saved → {out}.pdf / .png")


if __name__ == '__main__':
    main()
