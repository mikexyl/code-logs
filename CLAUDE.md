# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repo is a **multi-robot pose graph optimization and evaluation toolkit**. It processes SLAM output from multiple robots (in g2o format), optimizes the combined pose graph with GTSAM, and evaluates trajectory accuracy with the `evo` tool against ground truth.

## Building the C++ Optimizer

```bash
# From repo root
./build.sh
# or manually:
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

Dependencies: GTSAM, Boost (filesystem, system), C++17. The executable is `build/optimize_offline`.

## Running the Optimizer

```bash
# Default dataset ("gate" folder)
./build/optimize_offline

# Specify dataset folder name (looks under repo root)
./build/optimize_offline <dataset_name>

# Specify dataset and output directory
./build/optimize_offline <dataset_name> <output_dir>
```

The optimizer reads `bpsam_robot_*.g2o` from `<dataset>/<robot_dir>/dpgo/` subdirectories and writes results to `<dataset>/optimized_results/`.

## Running Trajectory Evaluation

`evaluate.py` wraps `evo_ape` and `evo_rpe` for ATE/RPE evaluation of TUM-format trajectories:

```bash
python3 evaluate.py <experiment_folder>

# With a frame transform from GT to robot frame (e.g. different IMU origins)
python3 evaluate.py <experiment_folder> --tf_file tf_xsens_to_handsfree.json
```

The script:
1. Finds all `Robot *.tum` files and matching `gt.txt` files recursively in the folder
2. Concatenates trajectories and runs `evo_ape tum` and `evo_rpe tum`
3. Saves `evo_ape.zip`, `evo_rpe.zip`, PDF/PNG plots to the experiment folder
4. Generates IEEE single-column publication-ready trajectory plots (`trajectories_aligned.pdf`, `trajectories_aligned_half.pdf`)

## Visualizing g2o Pose Graphs

```bash
python3 plot_g2o.py <path/to/file.g2o>

# Options
python3 plot_g2o.py <file.g2o> --only-2d
python3 plot_g2o.py <file.g2o> --only-3d
python3 plot_g2o.py <file.g2o> --three-planes
python3 plot_g2o.py <file.g2o> --save output.png
```

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

## Key Architecture Notes

- **Key remapping** in the optimizer: `new_key = robot_id * 10000 + original_pose_id`. Each robot occupies a 10000-key block. In `plot_g2o.py`, robot grouping uses `vid // 10000000000000000` — this is for the DPGO g2o format where GTSAM `Symbol` keys encode robot ID in the high bits.
- **g2o information matrix convention**: GTSAM stores 6D info as (rotation, translation), but the g2o `EDGE_SE3:QUAT` format stores as (translation, rotation). `writeG2oNoIndex()` in `optimize_offline.cpp` handles this swap.
- **Alignment in evaluate.py**: The `evo_ape.zip` from `evo_ape --align` stores either `alignment_transformation_sim3.npy` or `alignment_transformation_se3.npy`. The `load_alignment_from_evo_zip()` function reads this to realign the estimated trajectories for plotting.
- **Frame transform file** (`tf_xsens_to_handsfree.json`): JSON with `rotation` (3x3) and `translation` (3,) keys specifying SE3 from GT IMU frame to robot IMU frame. Applied to GT positions before plotting so both share a common frame.
- The `CATKIN_IGNORE` file prevents catkin from treating this as a ROS package despite being inside a ROS workspace.
