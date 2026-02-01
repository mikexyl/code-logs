#!/usr/bin/env python3

import os
import sys
import glob
import subprocess
import argparse
import tempfile
import shutil
import zipfile
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def read_tum_trajectory(filepath):
    """
    Read a TUM format trajectory file.
    Returns timestamps, positions (Nx3), and quaternions (Nx4, xyzw format).
    """
    timestamps = []
    positions = []
    quaternions = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 8:
                timestamps.append(float(parts[0]))
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                quaternions.append([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
    
    return np.array(timestamps), np.array(positions), np.array(quaternions)


def load_alignment_from_evo_zip(zip_path):
    """
    Load the alignment transformation from evo's saved results zip file.
    Returns rotation matrix (3x3), translation (3,), and scale (float).
    """
    import io
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Read the alignment parameters from the zip - evo stores as .npy (4x4 matrix)
        if 'alignment_transformation_sim3.npy' in z.namelist():
            with z.open('alignment_transformation_sim3.npy') as f:
                # Load numpy array from zip
                data = np.load(io.BytesIO(f.read()))
                # data is a 4x4 Sim(3) matrix: [s*R | t; 0 0 0 1]
                rotation_scaled = data[:3, :3]
                translation = data[:3, 3]
                # Extract scale from the rotation matrix (scale is uniform)
                scale = np.linalg.norm(rotation_scaled[:, 0])
                rotation = rotation_scaled / scale
                return rotation, translation, scale
        elif 'alignment_transformation_se3.npy' in z.namelist():
            with z.open('alignment_transformation_se3.npy') as f:
                data = np.load(io.BytesIO(f.read()))
                rotation = data[:3, :3]
                translation = data[:3, 3]
                return rotation, translation, 1.0
        else:
            # No alignment found, return identity
            print("Warning: No alignment transformation found in zip, using identity.")
            return np.eye(3), np.zeros(3), 1.0


def apply_alignment(positions, rotation, translation, scale):
    """
    Apply Sim(3) alignment transformation to positions.
    aligned = scale * R @ positions.T + translation
    """
    aligned = scale * (rotation @ positions.T).T + translation
    return aligned


def plot_aligned_trajectories(experiment_folder, pairs):
    """
    Plot aligned robot trajectories with different colors and labels.
    Reads the alignment transformation from evo_ape's saved results.
    Formatted for IEEE single-column journal standard.
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
    # Single column width: 3.5 inches (88.9mm)
    # Use Type 1 fonts for IEEE compatibility
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{times}',
        'font.family': 'serif',
        'font.serif': ['Times'],
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'figure.figsize': (3.5, 3.0),  # Single column width, reasonable height
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 0.5,
        'lines.linewidth': 1.0,
        'patch.linewidth': 0.5,
        'pdf.fonttype': 42,  # TrueType fonts for IEEE
        'ps.fonttype': 42,
    })
    
    # Set up the plot
    fig, ax = plt.subplots()
    
    # Color map for different robots
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(pairs))))
    
    # Plot each robot's trajectory
    for idx, p in enumerate(pairs):
        robot_path = p['robot_path']
        gt_path = p['gt_path']
        
        # Extract robot name from filename
        robot_name = os.path.basename(robot_path).replace('.tum', '')
        
        # Read trajectories
        _, est_positions, _ = read_tum_trajectory(robot_path)
        _, gt_positions, _ = read_tum_trajectory(gt_path)
        
        if len(est_positions) == 0:
            print(f"Warning: Empty trajectory for {robot_name}")
            continue
        
        # Apply alignment to estimated trajectory
        aligned_positions = apply_alignment(est_positions, rotation, translation, scale)
        
        # Plot estimated (aligned) trajectory
        ax.plot(aligned_positions[:, 0], aligned_positions[:, 1], 
                color=colors[idx % len(colors)], linewidth=1.0, 
                label=f'{robot_name}')
    
    # Plot ground truth (each robot separately to avoid jump lines)
    gt_plotted = False
    for p in pairs:
        _, gt_positions, _ = read_tum_trajectory(p['gt_path'])
        if len(gt_positions) > 0:
            # Only add label for the first GT trajectory
            label = 'Ground Truth' if not gt_plotted else None
            ax.plot(gt_positions[:, 0], gt_positions[:, 1], color='gray', linewidth=0.5, alpha=0.5, 
                    linestyle='--', label=label)
            gt_plotted = True
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='best', framealpha=0.9, edgecolor='none')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linewidth=0.3)
    
    # Tight layout for publication
    plt.tight_layout(pad=0.5)
    
    # Save the plot (single column: 3.5 inches)
    output_path = os.path.join(experiment_folder, "trajectories_aligned.pdf")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.02)
    print(f"\nTrajectory plot saved to: {output_path}")
    
    # Also save as PNG for quick preview
    output_png = os.path.join(experiment_folder, "trajectories_aligned.png")
    plt.savefig(output_png, bbox_inches='tight', pad_inches=0.02)
    print(f"Trajectory plot saved to: {output_png}")
    
    # Save half-column sized copy (1.67 inches for IEEE half-column)
    fig.set_size_inches(1.67, 1.5)
    plt.tight_layout(pad=0.3)
    
    output_path_half = os.path.join(experiment_folder, "trajectories_aligned_half.pdf")
    plt.savefig(output_path_half, bbox_inches='tight', pad_inches=0.01)
    print(f"Half-column plot saved to: {output_path_half}")
    
    output_png_half = os.path.join(experiment_folder, "trajectories_aligned_half.png")
    plt.savefig(output_png_half, bbox_inches='tight', pad_inches=0.01)
    print(f"Half-column plot saved to: {output_png_half}")
    
    plt.close()


def find_trajectory_pairs(experiment_folder):
    """
    Recursively finds 'Robot *.tum' files and their corresponding 'gt.txt'.
    Returns a list of tuples: (robot_file_path, gt_file_path, first_timestamp)
    """
    pairs = []
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(experiment_folder):
        for file in files:
            if file.startswith("Robot ") and file.endswith(".tum"):
                robot_path = os.path.join(root, file)
                gt_path = os.path.join(root, "gt.txt")
                
                if os.path.exists(gt_path):
                    # Read the first timestamp from the robot file for sorting
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
                    print(f"Warning: No gt.txt found for {robot_path}")
                    
    # Sort pairs by timestamp
    pairs.sort(key=lambda x: x['timestamp'])
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Combine TUM trajectories and run evo ATE evaluation.")
    parser.add_argument("experiment_folder", help="Path to the experiment folder")
    
    args = parser.parse_args()
    
    experiment_folder = os.path.abspath(args.experiment_folder)
    
    if not os.path.exists(experiment_folder):
        print(f"Error: Folder {experiment_folder} does not exist.")
        sys.exit(1)
        
    print(f"Searching for trajectories in {experiment_folder}...")
    pairs = find_trajectory_pairs(experiment_folder)
    
    if not pairs:
        print("No valid Robot *.tum and gt.txt pairs found.")
        sys.exit(1)
        
    print(f"Found {len(pairs)} trajectory segments.")
    for p in pairs:
        print(f"  - {p['robot_path']} (t={p['timestamp']})")
        
    # Create temporary file for combined trajectories
    with tempfile.TemporaryDirectory() as temp_dir:
        combined_est_path = os.path.join(temp_dir, "combined_est.tum")
        combined_gt_path = os.path.join(temp_dir, "combined_gt.tum")
        
        print("\nCombining trajectories...")
        
        # Combine Robot trajectories
        with open(combined_est_path, 'w') as outfile:
            for p in pairs:
                with open(p['robot_path'], 'r') as infile:
                    outfile.write(infile.read())
                    # Ensure newline between files if missing
                    if outfile.tell() > 0: # Check if we wrote anything
                         infile.seek(0, os.SEEK_END)
                         if infile.tell() > 0: # Check if input file was not empty
                             pass # We might want to ensure a newline, usually files strictly following TUM format have a newline, but let's be safe.
                             # Actually plain read/write is safer to avoid adding extra lines if not needed, 
                             # but let's just assume files are well formed or we concatenate directly. 
                             # Safest is just concatenation.
        
        # Combine GT trajectories
        # Note: We need to combine GTs in the same order as Robots to match the time segments roughly, 
        # BUT evo matches by timestamp, so order in file doesn't strictly matter as long as timestamps are unique.
        # However, duplicates might be an issue if GTs overlap. Input GTs likely don't overlap if robots don't.
        # Wait, the user said "find the 'Robot <id>.tum' files... ground truth files are named 'gt.txt' under the same folder"
        # Since timestamps are global (unix time likely), simple concatenation works for evo.
        
        with open(combined_gt_path, 'w') as outfile:
            for p in pairs:
                with open(p['gt_path'], 'r') as infile:
                    outfile.write(infile.read())

        print("Running evo ape...")
        
        cmd = ["evo_ape", "tum", combined_gt_path, combined_est_path, "-va"]
        cmd.append("--align")
        cmd.extend(["--plot_mode", "xy", "--save_plot", experiment_folder + "/evo.pdf"])
        cmd.extend(["--save_results", experiment_folder + "/evo_ape.zip"])

        # if evo.pdf exists, remove it
        if os.path.exists(experiment_folder + "/evo.pdf"):
            os.remove(experiment_folder + "/evo.pdf")

        # if evo.zip exists, remove it
        if os.path.exists(experiment_folder + "/evo_ape.zip"):
            os.remove(experiment_folder + "/evo_ape.zip")

            
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("\n" + "="*40)
            print("EVO APE RESULTS")
            print("="*40)
            print(result.stdout)
            print("="*40)
        except subprocess.CalledProcessError as e:
            print("\nError running evo_ape:")
            print(e.stderr)
            print(e.stdout)
            sys.exit(e.returncode)
        except FileNotFoundError:
             print("\nError: 'evo_ape' command not found. Please ensure evo is installed and in your PATH.")
             sys.exit(1)

        # Run evo_rpe with delta 100 meters
        print("\nRunning evo rpe...")
        
        rpe_cmd = ["evo_rpe", "tum", combined_gt_path, combined_est_path, "-va"]
        rpe_cmd.append("--align")
        rpe_cmd.extend(["--delta", "5", "--delta_unit", "m"])
        rpe_cmd.extend(["--plot_mode", "xy", "--save_plot", experiment_folder + "/evo_rpe.pdf"])
        rpe_cmd.extend(["--save_results", experiment_folder + "/evo_rpe.zip"])

        # if evo_rpe.pdf exists, remove it
        if os.path.exists(experiment_folder + "/evo_rpe.pdf"):
            os.remove(experiment_folder + "/evo_rpe.pdf")

        # if evo_rpe.zip exists, remove it
        if os.path.exists(experiment_folder + "/evo_rpe.zip"):
            os.remove(experiment_folder + "/evo_rpe.zip")

        print(f"Command: {' '.join(rpe_cmd)}")
        
        try:
            result = subprocess.run(rpe_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("\n" + "="*40)
            print("EVO RPE RESULTS")
            print("="*40)
            print(result.stdout)
            print("="*40)
        except subprocess.CalledProcessError as e:
            print("\nError running evo_rpe:")
            print(e.stderr)
            print(e.stdout)
            sys.exit(e.returncode)
        except FileNotFoundError:
             print("\nError: 'evo_rpe' command not found. Please ensure evo is installed and in your PATH.")
             sys.exit(1)

        # Plot aligned trajectories
        print("\nPlotting aligned trajectories...")
        plot_aligned_trajectories(experiment_folder, pairs)

if __name__ == "__main__":
    main()
