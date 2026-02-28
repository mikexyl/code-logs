# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repo is a **multi-robot pose graph optimization and evaluation toolkit**. It processes SLAM output from multiple robots (in g2o format), optimizes the combined pose graph with GTSAM, and evaluates trajectory accuracy with the `evo` tool against ground truth.

## Environment Setup

Python dependencies are managed via [pixi](https://pixi.sh). Activate the environment before running any Python scripts:

```bash
pixi shell        # activate the pixi environment
# or prefix individual commands:
pixi run python3 evaluate.py <experiment_folder>
```

The `pixi.toml` pins Python 3.11, numpy, matplotlib, scipy, rerun-sdk, evo, open3d, datafusion, and pyvista.

## Building the C++ Optimizer

```bash
# Using pixi (recommended, generates compile_commands.json)
pixi run build

# Or manually from repo root
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

Dependencies: GTSAM, Boost (filesystem, system), C++17. The executable is `build/optimize_offline`.

## Running the Optimizer

```bash
# Default dataset ("gate" folder)
./build/optimize_offline

# Specify dataset folder name
./build/optimize_offline <dataset_name>

# Specify dataset and output directory
./build/optimize_offline <dataset_name> <output_dir>
```

**Important:** The optimizer hardcodes its base search path to `/workspaces/src/code-logs/<dataset_name>`. Adjust the `base_path` variable in `src/optimize_offline.cpp` if running from a different environment.

The optimizer reads `bpsam_robot_*.g2o` from `<dataset>/<robot_dir>/dpgo/` subdirectories, combines all robot graphs, adds a single prior on the first pose of the first robot to fix gauge freedom, and writes results to `<dataset>/optimized_results/`.

The optimizer uses **Levenberg-Marquardt** (GNC with gradient-norm criterion is implemented but commented out). Intra-robot odometry edges are classified as known inliers via `Symbol::chr()` comparison.

## Running Trajectory Evaluation

`evaluate.py` wraps `evo_ape` and `evo_rpe` for ATE/RPE evaluation of TUM-format trajectories:

```bash
python3 evaluate.py <experiment_folder>

# With a separate ground truth folder (default: ground_truth/)
python3 evaluate.py <experiment_folder> --gt_folder ground_truth

# With a frame transform from GT to robot frame (e.g. different IMU origins)
python3 evaluate.py <experiment_folder> --tf_file tf_xsens_to_handsfree.json
```

Ground truth lookup with `--gt_folder`: for a robot file at `<experiment>/<subdir>/dpgo/Robot N.tum`, the GT is read from `<gt_folder>/<experiment_name>/<subdir>.txt`.

The script:
1. Finds all `Robot *.tum` files and matching `gt.txt` (or external GT) files recursively
2. Concatenates trajectories sorted by first timestamp and runs `evo_ape tum` and `evo_rpe tum`
3. Saves `evo_ape.zip`, `evo_rpe.zip`, PDF/PNG plots to the experiment folder
4. Generates IEEE single-column publication-ready trajectory plots (`trajectories_aligned.pdf`, `trajectories_aligned_half.pdf`)

RPE is run with `--delta 5 --delta_unit m`.

## Visualizing g2o Pose Graphs

```bash
python3 plot_g2o.py <path/to/file.g2o>

# Options
python3 plot_g2o.py <file.g2o> --only-2d
python3 plot_g2o.py <file.g2o> --only-3d
python3 plot_g2o.py <file.g2o> --three-planes
python3 plot_g2o.py <file.g2o> --save output.png
```

## Visualizing Rerun .rrd Recordings

`plot_rrd.py` reads Rerun `.rrd` recording files produced by the multi-robot system:

```bash
# Print all entities/components in the recording
python3 plot_rrd.py <file.rrd>

# Visualize landmarks (Points3D) and trajectories (LineStrips3D) with PyVista
python3 plot_rrd.py <file.rrd> --landmarks

# Plot cumulative received bandwidth over time (stacked BOW / VLC / CBS)
python3 plot_rrd.py <file.rrd> --bandwidth

# Plot PR and GV loop counts over time (summed across robots)
python3 plot_rrd.py <file.rrd> --loops

# Save bandwidth/loop plot to a specific path (saves both PDF and PNG)
python3 plot_rrd.py <file.rrd> --bandwidth --save output.pdf
```

The `--landmarks` viewer expects:
- Landmark point clouds at entity paths ending in `/landmarks` (with optional `Points3D:colors`)
- Trajectories at entity paths containing `/traj/` (as `LineStrips3D:strips`)
- Transforms at parent entities to compose world-frame poses
- Interactive PyVista window with point size / trajectory width sliders and a screenshot button (saves `<rrd_stem>-map.pdf`)

The `--bandwidth` mode discovers robots from `/<robot>/received_bow_byte` entities and also reads `/<robot>/received_vlc_byte` and `/<robot>/bandwidth_recv_bytes`. Outputs IEEE single-column formatted PDF and PNG next to the `.rrd` file (or to `--save` path).

The `--loops` mode discovers robots from `/<robot>/num_pr_loops` and `/<robot>/num_gv_loops`, sums across all robots, and produces a line plot.

## Data Layout

Experiment folders (e.g. `gate/`, `g123/`, `a12/`, `a34/`, `a567/`) follow this structure:

```
<experiment>/
├── <robot_dir>/          # e.g. g1/, g2/, a1/, a2/ — one per robot
│   └── dpgo/
│       ├── bpsam_robot_<N>.g2o   # GTSAM pose graph output
│       ├── Robot <N>.tum         # Estimated trajectory (TUM format)
│       ├── gt.txt                # Ground truth trajectory (TUM format)
│       └── X.txt                 # DPGO intermediate results
├── optimized_results/    # Written by optimize_offline
│   ├── combined_optimized.g2o
│   ├── robot_<N>_initial.txt
│   ├── robot_<N>_optimized.txt
│   └── summary.txt
├── evo_ape.zip           # Written by evaluate.py
├── evo_rpe.zip
├── trajectories_aligned.png/pdf
└── trajectories_aligned_half.png/pdf
```

A `ground_truth/` folder at the repo root holds GT files organized by `<experiment_name>/<robot_dir>.txt` for use with `--gt_folder`.

## Key Architecture Notes

- **Key remapping** in the optimizer: `new_key = robot_id * 10000 + original_pose_id`. Each robot occupies a 10000-key block. In `plot_g2o.py`, robot grouping uses `vid // 10000000000000000` — this is for the DPGO g2o format where GTSAM `Symbol` keys encode robot ID in the high bits.
- **g2o information matrix convention**: GTSAM stores 6D info as (rotation, translation), but the g2o `EDGE_SE3:QUAT` format stores as (translation, rotation). `writeG2oNoIndex()` in `src/optimize_offline.cpp` handles this swap.
- **Alignment in evaluate.py**: The `evo_ape.zip` from `evo_ape --align` stores either `alignment_transformation_sim3.npy` or `alignment_transformation_se3.npy`. The `load_alignment_from_evo_zip()` function reads this to realign the estimated trajectories for plotting.
- **Frame transform file** (`tf_xsens_to_handsfree.json`): JSON with `rotation` (3x3) and `translation` (3,) keys specifying SE3 from GT IMU frame to robot IMU frame. Applied to GT positions before plotting so both share a common frame.
- **Publication formatting**: All plots use IEEE single-column style (3.5 in wide, 300 dpi, serif fonts, pdf/ps fonttype 42). `evaluate.py` also saves a half-column variant (1.67 in wide).
- The `CATKIN_IGNORE` file prevents catkin from treating this as a ROS package despite being inside a ROS workspace.
