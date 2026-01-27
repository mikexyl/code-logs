#!/bin/bash

# Build script for multi-robot pose graph optimizer

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "=== Building Multi-Robot Pose Graph Optimizer ==="

# Create build directory
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

cd build

# Configure
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
echo "Building..."
make -j$(nproc)

echo ""
echo "=== Build Complete ==="
echo "Executable: $SCRIPT_DIR/build/optimize_offline"
echo ""
echo "To run:"
echo "  ./build/optimize_offline"
echo ""
echo "To visualize results:"
echo "  python3 visualize_results.py"
