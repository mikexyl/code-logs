# Multi-Robot Pose Graph Optimization

This project reads multiple g2o pose graph files from different robots, combines them into a single unified pose graph, optimizes using GTSAM, and visualizes the results.

## Overview

The program performs the following steps:
1. **Reads** all `bpsam_robot_*.g2o` files from `/workspaces/src/code-logs/123/g*/dpgo/`
2. **Combines** all measurements into a single pose graph with remapped keys
3. **Optimizes** using GTSAM Levenberg-Marquardt optimizer
4. **Outputs** optimized poses and statistics
5. **Visualizes** initial vs optimized trajectories

## Files

- `src/optimize_offline.cpp` - Main C++ optimization program
- `CMakeLists.txt` - Build configuration
- `visualize_results.py` - Python visualization script
- `README.md` - This file

## Building

### Prerequisites
- GTSAM library
- Boost (filesystem, system)
- C++17 compiler
- CMake 3.10+

### Build Instructions

```bash
# From the code-logs directory
mkdir -p build
cd build
cmake ..
make
```

Or if using pixi (recommended):
```bash
pixi run build
```

## Running

### Run the optimizer
```bash
# Use default paths
./build/optimize_offline

# Or specify custom paths
./build/optimize_offline <base_path> <output_directory>

# Example:
./build/optimize_offline /workspaces/src/code-logs/123 /workspaces/src/code-logs/123/results
```

### Visualize results
```bash
# Install matplotlib if needed
pip install matplotlib numpy

# Run visualization
python3 visualize_results.py

# Or specify custom output directory
python3 visualize_results.py /workspaces/src/code-logs/123/optimized_results
```

## Input Format

The program expects g2o files in the following structure:
```
/workspaces/src/code-logs/123/
├── g1/
│   └── dpgo/
│       └── bpsam_robot_0.g2o
├── g2/
│   └── dpgo/
│       └── bpsam_robot_1.g2o
└── g3/
    └── dpgo/
        └── bpsam_robot_2.g2o
```

## Output

The optimizer creates the following output files in the specified output directory:

### Combined Results
- `combined_optimized.g2o` - Complete optimized pose graph in g2o format

### Per-Robot Trajectories
- `robot_<id>_initial.txt` - Initial trajectory for robot <id>
- `robot_<id>_optimized.txt` - Optimized trajectory for robot <id>

Format (2D): `robot_id pose_id x y theta`
Format (3D): `robot_id pose_id x y z qx qy qz qw`

### Statistics
- `summary.txt` - Summary statistics including:
  - Number of robots
  - Total poses and factors
  - Initial and final error
  - Error reduction percentage

### Visualizations
- `trajectories_2d.png` - 2D trajectory comparison (initial vs optimized)
- `trajectories_3d.png` - 3D trajectory comparison (if applicable)

## Key Remapping

To avoid conflicts between robot pose indices, the program uses the following key remapping scheme:
- Robot 0: keys 0-9999
- Robot 1: keys 10000-19999
- Robot 2: keys 20000-29999
- etc.

Formula: `new_key = robot_id * 10000 + original_pose_id`

## Optimization Parameters

The program uses GTSAM's Levenberg-Marquardt optimizer with:
- Max iterations: 100
- Relative error tolerance: 1e-5
- Absolute error tolerance: 1e-5

Prior factors are added to the first pose of each robot with:
- 2D: σ = [0.01, 0.01, 0.01] (x, y, theta)
- 3D: σ = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01] (x, y, z, roll, pitch, yaw)

## Supported Formats

The program automatically detects and handles:
- **2D Pose Graphs** (SE(2)): VERTEX_SE2 and EDGE_SE2
- **3D Pose Graphs** (SE(3)): VERTEX_SE3 and EDGE_SE3

## Example Usage

```bash
# 1. Build the project
cd /workspaces/src/code-logs
mkdir -p build && cd build
cmake ..
make

# 2. Run optimization
./optimize_offline

# 3. View results
cd ../123/optimized_results
cat summary.txt

# 4. Visualize
cd ../..
python3 visualize_results.py
```

## Troubleshooting

### Build errors
- Ensure GTSAM is properly installed and findable by CMake
- Check that Boost libraries are available
- Verify C++17 compiler support

### Runtime errors
- Verify g2o files exist in expected locations
- Check file permissions for output directory
- Ensure g2o files are properly formatted

### Visualization issues
- Install required Python packages: `pip install matplotlib numpy`
- For headless environments, modify the script to save plots without display

## References

- GTSAM: https://gtsam.org/
- g2o format specification: https://github.com/RainerKuemmerle/g2o
