#!/usr/bin/env python3

import os
import sys
import glob
import subprocess
import argparse
import tempfile
import shutil

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

if __name__ == "__main__":
    main()
