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

### evaluate_loops_recall.py — loop closure recall vs GT
```bash
python3 evaluate_loops_recall.py <experiment_dir> ground_truth/<exp> \
    --tol 5.0 --max-angle 60
```
Resolves detected loop closures (from `loop_closures.csv` + `kimera_distributed_keyframes.csv`) to wall-clock timestamps, then computes recall against each `gt_loops_angle*.csv`. A GT loop is "detected" if any detected loop covers the same robot pair with both timestamps within `--tol` seconds. Plots a recall-vs-angle-threshold curve.

Key options:
- `--tol`: timestamp matching tolerance in seconds (default 2.0; 5.0 gives substantially higher recall)
- `--max-angle`: cap x-axis of the curve without re-running extraction

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

### plot_ablation.py — compare experiment variants
```bash
python3 plot_ablation.py <folder>
```
Scans `<folder>` and `baselines/<folder.name>/` for `*_bandwidth.npy` and `*_loops.npy` files. Each file stem becomes a labelled series on a shared axis.

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
<experiment>/                        # e.g. campus/, a5678/
├── <robot_dir>/                     # e.g. acl_jackal/, a5/, a6/
│   ├── dpgo/
│   │   ├── bpsam_robot_<N>.g2o
│   │   └── Robot <N>.tum
│   └── distributed/
│       ├── kimera_distributed_keyframes.csv   # pose_index → timestamp_ns
│       ├── loop_closures.csv                  # inter-robot loop pairs
│       └── lcd_log.csv                        # BoW/VLC byte counts over time
├── evo_ape.zip / evo_rpe.zip
└── trajectories_aligned*.pdf/png

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

**Bandwidth `.npy` files**: saved as `allow_pickle=True` dicts with keys `t_sec`, `bow_MB`, `vlc_MB`, `cbs_MB`. Loop `.npy` files have `t_sec`, `bow_matches`, `num_loop_closures`.

**GT loop closure parameters used on real datasets**:
- UAV (a5678): `--dist-xy 10 --dist-z 25 --min-z 15 --angles 10..60 --tol 5`
- Ground (campus): `--dist-xy 10 --dist-z 25` (no `--min-z`), same angles/tol
