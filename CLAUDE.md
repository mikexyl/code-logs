# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Multi-robot SLAM evaluation toolkit. Processes Kimera-Multi output (g2o pose graphs, TUM trajectories, Rerun recordings), runs GTSAM offline optimization, evaluates ATE/RPE with `evo`, extracts ground-truth loop closures, and generates IEEE-formatted publication plots.

## Environment

All Python scripts must be run with pixi:

```bash
pixi shell                          # activate once
pixi run python3 <script.py> ...   # or prefix each command
```

Python 3.11. Key packages: numpy, matplotlib, scipy, evo, rerun-sdk, open3d, pyvista.

Build the C++ optimizer:
```bash
pixi run build    # recommended — also generates compile_commands.json
```
The binary is `build/optimize_offline`. **Note:** it hardcodes the base path `/workspaces/src/code-logs/` — edit `src/optimize_offline.cpp` if running elsewhere.

## Full Analysis Pipeline

A `/full-analysis` custom command is available (`.claude/commands/full-analysis.md`) that encodes the full pipeline steps. Run via `/full-analysis <exp> [gt_folder]` in Claude Code.

Run in this order for a complete evaluation of experiment `<exp>` (e.g. `campus`, `a5678`):

```bash
# 1. ATE/RPE + trajectory plots
python3 evaluate.py <exp> --gt_folder ground_truth

# 2. Loop recall + outlier analysis — writes loops_recall.csv and inlier_loops.csv per variant
python3 evaluate_loops_recall.py <exp> ground_truth/<exp> --tol 5.0
# UAV datasets: add --max-angle 60

# 3. Algebraic connectivity (requires inlier_loops.csv from step 2)
python3 plot_algebraic_connectivity.py <exp>

# 4. GT rotation distribution of detected loops
python3 plot_loop_rotation_dist.py <exp> ground_truth/<exp>

# 5. Scalability scatter (bandwidth vs recall / ATE)
python3 plot_scalability.py <exp>

# 6. Bandwidth & loops comparison + recall comparison
#    Reads *_bandwidth.npy / *_loops.npy extracted from .rrd via plot_rrd.py --bandwidth/--loops
#    Re-run extraction manually whenever new .rrd data arrives
python3 plot_ablation.py <exp>
```

**Dependency**: step 6 (`plot_ablation.py`) reads `*_bandwidth.npy` / `*_loops.npy` files that are
extracted separately from Rerun recordings via `plot_rrd.py --bandwidth` / `--loops`. These are **not**
regenerated automatically — run extraction whenever new `.rrd` data is available.

## Scripts

### evaluate.py — ATE/RPE evaluation
```bash
python3 evaluate.py <experiment_folder>
python3 evaluate.py <experiment_folder> --gt_folder ground_truth
python3 evaluate.py <experiment_folder> --tf_file tf_xsens_to_handsfree.json
```
Auto-discovers variant sub-folders (subdirs containing robot dirs with `distributed/` or `dpgo/`). Evaluates each variant independently, saving all outputs inside the variant dir. Falls back to evaluating `<experiment_folder>` directly if no variants are found.

Finds all `Robot *.tum` + matching GT files, concatenates per robot, runs `evo_ape tum` and `evo_rpe tum` (delta 5 m), saves `evo_ape.zip`, `evo_rpe.zip`, aligned trajectory plots (full + half-column). Generates two trajectory plot versions: `trajectories_aligned_no_loops.*` (clean) and `trajectories_aligned.*` (with loop closure lines overlaid).

**Loop overlay requires `distributed/` dirs**: loop closure lines are drawn only when each robot dir contains `distributed/kimera_distributed_keyframes.csv` and `distributed/loop_closures.csv`. Variants with only `dpgo/` (no `distributed/`) will produce identical `trajectories_aligned.*` and `trajectories_aligned_no_loops.*` plots.

When evaluating a variant sub-folder directly (e.g. `g2345/ns-as`), pass `--gt_exp_name <experiment>` so GT files are resolved from `ground_truth/<experiment>/` instead of `ground_truth/ns-as/`:
```bash
python3 evaluate.py g2345/ns-as --gt_folder ground_truth --gt_exp_name g2345
```

### extract_gt_loops.py — GT loop closure extraction
```bash
python3 extract_gt_loops.py ground_truth/campus --dist-xy 10.0 --dist-z 25.0 \
    --min-z 15.0 --angles 10 20 30 40 50 60 --plot
```
Downsamples GT trajectories to 1 Hz, uses a 2D KD-tree (XY) + Z filter + rotation magnitude check to find all inter-robot pose pairs. Sweeps multiple angle thresholds in one pass.

Key options:
- `--dist-xy` / `--dist-z`: separate XY and Z proximity thresholds (metres)
- `--min-z`: trim leading/trailing ground-level poses (use for UAV datasets to skip takeoff/landing; not needed for ground robots)
- `--angles`: one or more rotation thresholds in degrees
- `--plot --subsample N`: visualize 1-in-N loop lines on XY map

Output per threshold: `gt_loops_angle<N>.csv`, `gt_loops_viz_angle<N>.pdf/png`. Combined stats: `gt_loops_stats.txt`. The `tx,ty,tz` in the CSV are the relative pose in robot_i's **local** frame (not world-frame XY distance).

### evaluate_loops_recall.py — loop closure recall vs GT + outlier analysis
```bash
python3 evaluate_loops_recall.py <experiment_dir> ground_truth/<exp> \
    --tol 5.0 --max-angle 60
```
Auto-discovers variant sub-folders and `baselines/<exp>/*/` dirs. Evaluates each, saves per-variant `loops_recall.csv/txt`, and plots a `recall_comparison.pdf/png` across all variants and baselines (variants solid, baselines dashed).

Also performs **outlier analysis**: for each detected loop that carries a relative pose estimate (`tx,ty,tz,qx,qy,qz,qw` in `loop_closures.csv`), the detected pose is compared against the GT relative pose at the loop timestamps. A loop is an outlier if `translation_error > max(--trans-abs, --trans-rel × GT distance)` OR `rotation_error > --rot-thr`. Saves `outlier_comparison.pdf/png`.

`recall_comparison.pdf/png` shows two side-by-side subplots: recall vs GT loop relative rotation (left) and recall vs GT loop translation distance (right), with a shared legend below.

Key options:
- `--tol`: timestamp matching tolerance in seconds (default 2.0; 5.0 gives substantially higher recall)
- `--max-angle`: cap x-axis of the curve without re-running extraction
- `--trans-abs`: absolute translation error floor in metres (default 2.0)
- `--trans-rel`: relative translation error threshold as fraction of GT distance (default 0.10 = 10%)
- `--rot-thr`: rotation error threshold for outlier detection in degrees (default 40.0)
- `--max-gap`: max GT timestamp gap for outlier pose lookup in seconds (default 2.5)

### evaluate_loops.py — visualize detected loop closures
```bash
python3 evaluate_loops.py <experiment_folder>
python3 evaluate_loops.py a5678 --subsample 10
```
Draws loop closure lines on top of all robot TUM trajectories. Lookup chain: `loop_closures.csv` → keyframe CSV → timestamp → TUM position.

### plot_rrd.py — Rerun recording visualization
```bash
python3 plot_rrd.py <file.rrd>               # list entities
python3 plot_rrd.py <file.rrd> --landmarks   # 3D map + trajectory viewer (PyVista)
python3 plot_rrd.py <file.rrd> --bandwidth   # cumulative BOW/VLC/CBS bandwidth plot
python3 plot_rrd.py <file.rrd> --loops       # PR+GV loop counts over time
python3 plot_rrd.py <file.rrd> --bandwidth --save output.pdf
```

### plot_loop_rotation_dist.py — GT relative-rotation distribution of detected loops
```bash
python3 plot_loop_rotation_dist.py <exp_dir> <gt_dir>
python3 plot_loop_rotation_dist.py campus ground_truth/campus
```
For each detected inter-robot loop, looks up GT orientations at the loop timestamps (nearest-neighbour) and computes the relative rotation angle. Auto-discovers variant sub-folders and `baselines/<exp_dir.name>/*/`. Plots overlaid step histograms (variants solid, baselines dashed). Falls back to single-experiment mode if no variants found.

Key options: `--max-gap` (default 2.5 s), `--max-angle` (default 120°), `--bins` (default 12).

### plot_algebraic_connectivity.py — Fiedler value of combined pose graph
```bash
python3 plot_algebraic_connectivity.py <folder>
```
Reads `inlier_loops.csv` (written by `evaluate_loops_recall.py`) and `kimera_distributed_keyframes.csv` per variant, builds an unweighted NetworkX graph (odometry chains + inlier inter-robot edges), and computes λ₂ of the Laplacian. Auto-discovers variants and `baselines/<exp>/*/`. Saves `algebraic_connectivity.pdf/png`.

### plot_scalability.py — bandwidth vs recall/ATE Pareto scatter
```bash
python3 plot_scalability.py <folder>
python3 plot_scalability.py <folder> --ate
python3 plot_scalability.py <folder> --buckets 10 20 30
```
X-axis: total BoW+VLC bandwidth (MB). Default Y-axis: recall at multiple rotation buckets (multi-panel).
`--ate`: also produces `scalability_ate.pdf/png` with ATE RMSE on Y-axis.
Always produces `yield.pdf/png` (verified inliers per MB bar chart).
Auto-discovers variants and baselines. Reads `*_bandwidth.npy`, `loops_recall.csv`, and `evo_ape.zip`.

### plot_ablation.py — compare experiment variants
```bash
python3 plot_ablation.py <folder>
```
Scans `<folder>` and `baselines/<folder.name>/` for `*_bandwidth.npy` and `*_loops.npy` files. Each file stem becomes a labelled series on a shared axis. Also auto-discovers variant sub-folders with `loops_recall.csv` and plots a `recall_comparison`.

### run_mesa_baselines.py — run MESA baseline methods
```bash
python3 run_mesa_baselines.py <experiment_folder> [--methods dgs mbadmm geodesic-mesa]
python3 run_mesa_baselines.py g2345/ns-as
python3 run_mesa_baselines.py g2345   # discovers all variants
```
Auto-discovers per-robot g2o files, merges them, converts to JRL format via `g2o_to_jrl.py`, then runs each method (dgs, asapp, geodesic-mesa) via the `mesa-baselines` Docker image. Results land in `<exp>/mesa_baselines/<variant>/<method>/`.

### g2o_to_jrl.py — convert g2o to JRL format for MESA
```bash
python3 g2o_to_jrl.py robot0.g2o [robot1.g2o ...] -o output.jrl [-n name]
python3 g2o_to_jrl.py folder/*/dpgo/bpsam_robot_*.g2o -o dataset.jrl
```
Preserves per-robot variable assignment from GTSAM Symbol keys. Adds gauge-freedom priors on each robot's first pose.

### evaluate_mesa_baselines.py — ATE for MESA + CBS/CBS+ results
```bash
python3 evaluate_mesa_baselines.py <variant_folder> [--gt_folder ground_truth]
python3 evaluate_mesa_baselines.py g2345/ns-as --gt_exp_name g2345
```
Two evaluation paths in one script:

**MESA methods** (dgs, asapp, geodesic-mesa, centralized): decodes `final_results.jrr.cbor`, matches timestamps from original `Robot N.tum` files by pose index, runs `evo_ape`, updates `ate_results.csv`. Reads convergence stats from `iter_time_comm.txt` — the `Iteration` column is total synchronous algorithm rounds (each round involves all robots simultaneously). `TotalCommunications` counts individual robot messages (= iterations × n_robots).

**CBS/CBS+ DPGO variants**: auto-discovered from `<variant>/cbs/` and `<variant>/cbs_plus/` dirs. For ATE, reads the **last non-empty** `Robot <N>_<ts>.tum` file per robot (highest timestamp suffix) directly — these already contain real timestamps, no pose-index remapping needed. For comm stats, reads `stats_robot_*.csv` per robot and sums `bytes_sent` at the final row. Reports total synchronous rounds = last `iteration` value in the stats CSV (not multiplied by n_robots). CBS residuals are non-monotonic (start near 0, rise as edges are added, then oscillate) so 1%-of-final convergence detection is not used.

### run_all_evaluations.py — batch ATE evaluation across all datasets
```bash
python3 run_all_evaluations.py [--skip_optimize] [--skip_evaluate] [--output ate_results.csv]
```
Runs global LM+Huber optimization and ATE evaluation across all configured datasets/variants, collecting results into a single CSV.

### evaluate_swarm_slam.py — Swarm-SLAM baseline evaluation
```bash
python3 evaluate_swarm_slam.py baselines/campus/Swarm-SLAM --gt_folder ground_truth --exp_name campus
python3 evaluate_swarm_slam.py baselines/campus/Swarm-SLAM --gt_folder ground_truth --exp_name campus --skip_ate
```
Adapts the non-standard Swarm-SLAM output to the standard file layout expected by all comparison scripts.

Parses the best (most-edges) `optimized_global_pose_graph.g2o` snapshot per robot, loads `pose_timestampsN.csv` to map vertex IDs to timestamps, and writes:
- `robot_names.yaml` — so `discover_robots` can find the robots
- `<robot>/distributed/kimera_distributed_keyframes.csv` — keyframe_id → timestamp_ns
- `hathor/distributed/loop_closures.csv` — all inter-robot edges with relative poses
- `baselines/<exp>/campus-Swarm-SLAM_bandwidth.npy` — dummy bandwidth dict (zeros) for scalability plots
- Per-robot `dpgo/Robot <N>.tum` files for ATE evaluation
- `evo_ape.zip` — ATE results
- `trajectories_aligned.pdf/png` and `trajectories_aligned_no_loops.pdf/png`

**Swarm-SLAM vertex ID encoding**: `(ord(letter) << 48) | pose_index` where `letter = chr(ord('A') + robot_id)` in the g2o file. `pose_timestampsN.csv` uses `(0x67 << 56) | vertex_id` as the GTSAM symbol prefix.

Must be run before `evaluate_loops_recall.py`, `plot_algebraic_connectivity.py`, etc. to populate the standard files.

### plot_baseline.py — Kimera-Multi baseline stats
```bash
python3 plot_baseline.py baselines/campus [--ate] [--gt_folder ground_truth]
```
Reads `lcd_log.csv` from each robot's `distributed/` dir, plots BoW matches and loop-closure counts over time. With `--ate`, runs `evo_ape` against GT.

### plot_loop_errors_map.py — trajectory map with loop closures colored by error
```bash
python3 plot_loop_errors_map.py <variant_dir> <gt_dir>
python3 plot_loop_errors_map.py campus/ns-as ground_truth/campus
python3 plot_loop_errors_map.py campus/ns-as ground_truth/campus --max_err 15
```
Plots aligned robot trajectories with inter-robot loop closure lines colored by translation error vs ground truth (RdYlGn_r colormap: green = low error, red = high error). Requires `evo_ape.zip` for alignment, `distributed/kimera_distributed_keyframes.csv` and `loop_closures.csv` (with `tx,ty,tz,qx,qy,qz,qw` fields) per robot, and GT trajectory files in `<gt_dir>`.

Key option: `--max_err` sets the colorbar maximum in metres (default 20). Saves `loop_errors_map.pdf/png`.

### plot_multi_dataset_traj.py — combined trajectory figure across datasets
```bash
python3 plot_multi_dataset_traj.py
python3 plot_multi_dataset_traj.py --output multi_dataset_traj
```
Produces a 1×5 subplot figure with ns-as variant trajectories for g123, g156, g2345, a12, and gate. Each panel shows GT (dashed gray) + per-robot colored trajectories. Alignment is loaded from `evo_ape.zip` (or `lm_optimized/evo_ape.zip` as fallback); falls back to Umeyama if neither exists. Shared legend at bottom.

### plot_kimera_traj.py — single-variant Kimera-Multi trajectory
```bash
python3 plot_kimera_traj.py <variant_dir> [--gt_dir ground_truth/campus]
```
Plots aligned trajectories + GT + inter-robot loop closure lines for a Kimera-Multi variant. Loads poses from the highest-numbered `kimera_distributed_poses_*.csv` per robot. Alignment from `evo_ape.zip` or Umeyama fallback.

### plot_combined_baselines_traj.py — 2×2 ours-vs-baselines figure
```bash
python3 plot_combined_baselines_traj.py
python3 plot_combined_baselines_traj.py --output combined_baselines_traj
```
2×2 grid: rows = campus/ns-as and a5678/ns-as; columns = Ours (CBS+, Centralized GNC-GM) and Baselines (DGS, ASAPP, Geodesic-MESA). Alignment from per-method `evo_ape.zip` or Umeyama. Single legend at bottom.

### plot_g2o.py — pose graph visualization
```bash
python3 plot_g2o.py <file.g2o> [--only-2d] [--only-3d] [--three-planes]
```

## Shared Library: `utils/`

All scripts import from this package — **do not duplicate its functionality**.

### `utils/io.py`

| Function | Description |
|---|---|
| `read_tum_trajectory(path)` | TUM / CSV → `(timestamps_s, positions, quaternions_xyzw)` |
| `load_gt_trajectory(path)` | GT CSV/TUM → `(timestamps_ns, positions, rotations_xyzw)` |
| `load_keyframes_csv(path)` | `{keyframe_id: timestamp_s}` |
| `load_loop_closures_csv(path)` | list of loop dicts (robot1/pose1/robot2/pose2 + optional pose fields) |
| `load_alignment_from_evo_zip(path)` | `(R, t, scale)` from evo results zip |
| `load_frame_transform(path)` | 4×4 SE3 matrix from JSON/YAML/text |
| `load_variant_aliases(path)` | `{raw_name: display_label}` from `variant_aliases.yaml` |
| `apply_variant_alias(aliases, name)` | Return display label or `None` (skip) if not listed |
| `umeyama(src, dst)` | Sim(3) alignment via SVD → `(R, t, scale)` |
| `is_robot_dir(d)` | True if dir has `distributed/` or `dpgo/` subdir |
| `discover_variants(exp_dir)` | Subdirs of `exp_dir` that contain at least one robot dir |
| `discover_baselines(exp_dir)` | Dirs from `baselines/<exp_dir.name>/*/` with robots OR with `evo_ape.zip` directly (e.g. pre-evaluated Swarm-SLAM) |
| `discover_robots(exp_dir)` | `{robot_id: robot_dir_name}` — handles CBS-style `Robot N_ts.tum` |
| `load_gt_trajectories_by_name(gt_dir, names)` | `{name: (timestamps_s, positions, rotations_xyzw)}` |

### `utils/plot.py`

| Function / Constant | Description |
|---|---|
| `IEEE_RC` | rcParams dict for IEEE single-column style |
| `ROBOT_COLORS` | List of 6 hex colors for per-robot lines |
| `save_fig(fig, base_path)` | Save PDF + PNG at `base_path` |
| `apply_alignment(positions, R, t, scale)` | Apply Sim3: `scale * R @ pts + t` |
| `apply_frame_transform(positions, T)` | Apply 4×4 SE3 to (N,3) array |
| `find_tum_position(ts_s, timestamps, positions, max_gap_s)` | Nearest-neighbour lookup; `None` if gap > threshold |
| `find_nearest_pose(ts_s, timestamps, positions, rotations, max_gap_s)` | Returns `(position, rotation)` or `None` |
| `quat_xyzw_to_rotation_matrix(q)` | xyzw quaternion → 3×3 rotation matrix |
| `rotation_angle_deg(R)` | Rotation angle in degrees from 3×3 matrix |
| `mark_endpoint(ax, t_arr, v_arr, color)` | Dot + value annotation at last curve point |

To override figure size while keeping IEEE style:
```python
plt.rcParams.update({**IEEE_RC, 'figure.figsize': (3.5, 2.5)})
```
`save_fig(fig, base_path)` always saves both `.pdf` and `.png`.

## Data Layout

```
<experiment>/                        # e.g. campus/, gate/
├── <variant>/                       # e.g. all/, ns-cs/, mixvpr/  (optional)
│   ├── <robot_dir>/                 # e.g. acl_jackal/, g1/
│   │   ├── dpgo/
│   │   │   ├── bpsam_robot_<N>.g2o
│   │   │   └── Robot <N>.tum
│   │   └── distributed/
│   │       ├── kimera_distributed_keyframes.csv   # pose_index → timestamp_ns
│   │       ├── loop_closures.csv                  # inter-robot loop pairs + relative pose
│   │       └── lcd_log.csv                        # BoW/VLC byte counts over time
│   ├── cbs/                         # CBS DPGO variant output
│   │   └── <robot_dir>/
│   │       └── dpgo/
│   │           ├── cbs_robot_<N>_<ts>.g2o         # intermediate g2o snapshots
│   │           ├── Robot <N>_<ts>.tum             # trajectory snapshots (last = final)
│   │           └── stats_robot_<N>.csv            # per-iteration residual + bytes_sent
│   ├── cbs_plus/                    # CBS+ DPGO variant (same structure as cbs/)
│   ├── evo_ape.zip / evo_rpe.zip
│   ├── trajectories_aligned*.pdf/png
│   ├── loop_errors_map.pdf/png      # loop lines colored by GT translation error
│   ├── loops_recall.csv/txt
│   └── loops_recall.pdf/png
├── recall_comparison.pdf/png        # multi-variant comparison
├── outlier_comparison.pdf/png       # loop pose outlier ratio per variant
├── outlier_by_gt.pdf/png            # outlier ratio vs GT translation/rotation
├── algebraic_connectivity.pdf/png   # Fiedler value per variant
├── bandwidth_comparison.pdf/png     # cumulative bandwidth across variants
├── loops_comparison.pdf/png         # PR/GV loop counts across variants
└── loop_rotation_dist.pdf/png

ground_truth/
└── <experiment>/
    └── <robot_dir>.csv    # comma-sep, timestamp_ns, x y z qw qx qy qz
    └── <robot_dir>.txt    # TUM format, timestamp_s, x y z qx qy qz qw
```

Kimera-Multi baselines live under `baselines/<experiment>/Kimera-Multi/<robot>/`.

## Key Architecture Notes

**Timestamp conventions** (critical — mixing these causes silent bugs):
- GT CSV files: nanoseconds, quaternion order `qw qx qy qz`
- TUM files (.tum / .txt): seconds, quaternion order `qx qy qz qw`
- Keyframe CSVs: nanoseconds in `keyframe_stamp_ns` column
- All `utils/io.py` loaders normalize to seconds and `xyzw` order internally

**GTSAM key remapping** in optimizer: `new_key = robot_id * 10000 + original_pose_id`. In DPGO g2o files, `plot_g2o.py` groups by `vid // 10000000000000000` (GTSAM Symbol high-bit encoding).

**g2o info matrix**: GTSAM stores 6D info as (rotation, translation); `EDGE_SE3:QUAT` format is (translation, rotation). The `writeG2oNoIndex()` function in `src/optimize_offline.cpp` swaps these.

**Bandwidth `.npy` files**: saved as `allow_pickle=True` dicts with keys `t_sec`, `bow_MB`, `vlc_MB`, `cbs_MB`. Loop `.npy` files have `t_sec`, `pr_total`, `gv_total`.

**`inlier_loops.csv`**: written by `evaluate_loops_recall.py` per variant, contains `name1, t1_s, name2, t2_s` for each inlier loop closure. Consumed by `plot_algebraic_connectivity.py`. Must re-run `evaluate_loops_recall.py` before `plot_algebraic_connectivity.py` whenever the inlier criterion changes.

**Inlier/outlier threshold**: `trans_err ≤ max(trans_abs, trans_rel × gt_dist)` where defaults are `trans_abs=2.0 m`, `trans_rel=0.10` (10%). Rotation threshold: `rot_thr=40°`.

**`variant_aliases.yaml`**: maps internal variant/baseline directory names to paper-ready display labels. Only variants listed here appear in multi-variant comparison plots (recall, outlier, scalability, connectivity, rotation dist). If the file is absent, all variants are shown with raw names. Keys must exactly match directory names.

**Variant auto-discovery pattern**: all scripts use `discover_variants` / `discover_baselines` / `discover_robots` / `is_robot_dir` from `utils/io.py`. A sub-directory is a variant if it contains at least one robot dir (a dir with `distributed/` or `dpgo/` inside). Baselines are auto-discovered from `baselines/<exp_name>/*/`. `run_all_evaluations.py` keeps a local `discover_variants` that additionally excludes `lm_optimized/` dirs.

**`loop_closures.csv` relative pose fields**: columns `qx,qy,qz,qw,tx,ty,tz` store the detected relative pose T_{robot1}^{robot2} in robot1's local frame. `load_loop_closures_csv` loads these into the dict if present; scripts that do not need them ignore them.

**GT loop closure parameters used on real datasets**:
- UAV (a5678): `--dist-xy 10 --dist-z 25 --min-z 15 --angles 10..60 --tol 5`
- Ground (campus, gate): `--dist-xy 10 --dist-z 25` (no `--min-z`), same angles/tol

**CBS/CBS+ TUM file naming**: files are named `Robot <N>_<unix_ts>.tum` where the suffix is a Unix timestamp of when the snapshot was written. Multiple snapshots accumulate per run. The **last** file (highest suffix, non-empty) is the final trajectory. `evaluate_mesa_baselines.py` always picks this file for ATE evaluation. Empty TUM files (size 0) are skipped — they indicate a snapshot was started but not yet written.

**CBS/CBS+ iteration counting**: `stats_robot_<N>.csv` rows are per-robot iterations of the synchronous ADMM-like algorithm. All robots run in lockstep, so the `iteration` column is the same as the number of synchronous rounds. `bytes_sent` is cumulative. The reported `iterations_1pct` in `ate_results.csv` is the final iteration count (total rounds run), not a convergence threshold — CBS residuals start near zero, rise as loop edges are added, then oscillate rather than monotonically decreasing, making 1%-of-final detection unreliable.

**Iteration count comparability (MESA vs CBS)**: both express iterations as synchronous algorithm rounds (all robots participate each round). MESA's `TotalCommunications` = iterations × n_robots (total individual messages); its `Iteration` column = synchronous rounds. CBS's `iteration` column = synchronous rounds directly. Do not multiply CBS iterations by n_robots — they are already in the same unit as MESA's `Iteration`.
