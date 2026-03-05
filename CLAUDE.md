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

## Scripts

### evaluate.py — ATE/RPE evaluation
```bash
python3 evaluate.py <experiment_folder>
python3 evaluate.py <experiment_folder> --gt_folder ground_truth
python3 evaluate.py <experiment_folder> --tf_file tf_xsens_to_handsfree.json
```
Auto-discovers variant sub-folders (subdirs containing robot dirs with `distributed/` or `dpgo/`). Evaluates each variant independently, saving all outputs inside the variant dir. Falls back to evaluating `<experiment_folder>` directly if no variants are found.

Finds all `Robot *.tum` + matching GT files, concatenates per robot, runs `evo_ape tum` and `evo_rpe tum` (delta 5 m), saves `evo_ape.zip`, `evo_rpe.zip`, aligned trajectory plots (full + half-column). Generates two trajectory plot versions: `trajectories_aligned_no_loops.*` (clean) and `trajectories_aligned.*` (with loop closure lines overlaid).

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

### plot_ablation.py — compare experiment variants
```bash
python3 plot_ablation.py <folder>
```
Scans `<folder>` and `baselines/<folder.name>/` for `*_bandwidth.npy` and `*_loops.npy` files. Each file stem becomes a labelled series on a shared axis. Also auto-discovers variant sub-folders with `loops_recall.csv` and plots a `recall_comparison`.

### plot_baseline.py — Kimera-Multi baseline stats
```bash
python3 plot_baseline.py baselines/campus [--ate] [--gt_folder ground_truth]
```
Reads `lcd_log.csv` from each robot's `distributed/` dir, plots BoW matches and loop-closure counts over time. With `--ate`, runs `evo_ape` against GT.

### plot_g2o.py — pose graph visualization
```bash
python3 plot_g2o.py <file.g2o> [--only-2d] [--only-3d] [--three-planes]
```

## Shared Library: `utils/`

All scripts import from this package — do not duplicate its functionality.

- **`utils/io.py`**: `read_tum_trajectory`, `load_gt_trajectory`, `load_keyframes_csv`, `load_loop_closures_csv`, `load_alignment_from_evo_zip`, `load_frame_transform`
- **`utils/plot.py`**: `IEEE_RC` (rcParams dict), `ROBOT_COLORS`, `save_fig`, `apply_alignment`, `apply_frame_transform`, `find_tum_position`, `mark_endpoint`

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
│   ├── evo_ape.zip / evo_rpe.zip
│   ├── trajectories_aligned*.pdf/png
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

**Variant auto-discovery pattern** (used by `evaluate.py`, `evaluate_loops_recall.py`, `plot_ablation.py`, `plot_loop_rotation_dist.py`): a sub-directory is a variant if it contains at least one robot dir (a dir with `distributed/` or `dpgo/` inside). Baselines are auto-discovered from `baselines/<exp_name>/*/` using the same robot-dir check.

**`loop_closures.csv` relative pose fields**: columns `qx,qy,qz,qw,tx,ty,tz` store the detected relative pose T_{robot1}^{robot2} in robot1's local frame. `load_loop_closures_csv` loads these into the dict if present; scripts that do not need them ignore them.

**GT loop closure parameters used on real datasets**:
- UAV (a5678): `--dist-xy 10 --dist-z 25 --min-z 15 --angles 10..60 --tol 5`
- Ground (campus, gate): `--dist-xy 10 --dist-z 25` (no `--min-z`), same angles/tol
