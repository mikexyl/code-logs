"""Shared I/O helpers for trajectory, loop-closure, and alignment data."""

import csv as _csv
import io as _io
import json
import os
import zipfile

import numpy as np


def read_tum_trajectory(filepath):
    """Read a TUM (.tum / .txt) or CSV trajectory file.

    TUM format:  space-separated, timestamp in seconds,      x y z qx qy qz qw
    CSV format:  comma-separated, timestamp in nanoseconds,  x y z qw qx qy qz

    Returns:
        timestamps  : (N,) float64, seconds
        positions   : (N,3) float64
        quaternions : (N,4) float64, xyzw order
    """
    timestamps, positions, quaternions = [], [], []
    is_csv = os.path.splitext(filepath)[1].lower() == '.csv'
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',') if is_csv else line.split()
            if len(parts) < 8:
                continue
            try:
                if is_csv:
                    # CSV: timestamp_ns  x y z  qw qx qy qz
                    timestamps.append(float(parts[0]) / 1e9)
                    positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    # reorder wxyz → xyzw
                    quaternions.append([float(parts[5]), float(parts[6]),
                                        float(parts[7]), float(parts[4])])
                else:
                    # TUM: timestamp_s  x y z  qx qy qz qw
                    timestamps.append(float(parts[0]))
                    positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    quaternions.append([float(parts[4]), float(parts[5]),
                                        float(parts[6]), float(parts[7])])
            except ValueError:
                continue
    return np.array(timestamps), np.array(positions), np.array(quaternions)


def load_keyframes_csv(path):
    """Load kimera_distributed_keyframes.csv.

    Returns:
        {keyframe_id (int): timestamp_s (float)}
    """
    kf_map = {}
    with open(path) as f:
        for row in _csv.DictReader(f):
            try:
                kf_map[int(row['keyframe_id'])] = float(row['keyframe_stamp_ns']) / 1e9
            except (ValueError, KeyError):
                continue
    return kf_map


def load_loop_closures_csv(path):
    """Load loop_closures.csv.

    Returns:
        list of dicts with keys: robot1, pose1, robot2, pose2
    """
    loops = []
    with open(path) as f:
        for row in _csv.DictReader(f):
            try:
                loops.append({
                    'robot1': int(row['robot1']), 'pose1': int(row['pose1']),
                    'robot2': int(row['robot2']), 'pose2': int(row['pose2']),
                })
            except (ValueError, KeyError):
                continue
    return loops


def load_gt_trajectory(path):
    """Load a ground-truth trajectory file. Supports .csv and .txt (TUM) formats.

    CSV:  comma-separated, timestamp in nanoseconds, qw qx qy qz
    TUM:  space-separated, timestamp in seconds,     qx qy qz qw

    Returns:
        timestamps : (N,)  int64, nanoseconds
        positions  : (N,3) float64, XYZ metres
        rotations  : (N,4) float64, xyzw quaternion
    """
    is_tum = str(path).endswith('.txt')
    timestamps, positions, rotations = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split() if is_tum else line.split(',')
            if len(parts) < 8:
                continue
            try:
                if is_tum:
                    ts = int(float(parts[0]) * 1_000_000_000)
                    qx, qy, qz, qw = (float(parts[4]), float(parts[5]),
                                      float(parts[6]), float(parts[7]))
                else:
                    ts = int(float(parts[0]))
                    qw, qx, qy, qz = (float(parts[4]), float(parts[5]),
                                      float(parts[6]), float(parts[7]))
                timestamps.append(ts)
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                rotations.append([qx, qy, qz, qw])
            except ValueError:
                continue
    return (np.array(timestamps, dtype=np.int64),
            np.array(positions,  dtype=np.float64),
            np.array(rotations,  dtype=np.float64))


def load_alignment_from_evo_zip(zip_path):
    """Load the Sim3/SE3 alignment transformation from an evo results zip.

    Returns:
        rotation    : (3,3) float64
        translation : (3,)  float64
        scale       : float
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        if 'alignment_transformation_sim3.npy' in z.namelist():
            with z.open('alignment_transformation_sim3.npy') as f:
                data = np.load(_io.BytesIO(f.read()))
            rotation_scaled = data[:3, :3]
            translation = data[:3, 3]
            scale = float(np.linalg.norm(rotation_scaled[:, 0]))
            return rotation_scaled / scale, translation, scale
        elif 'alignment_transformation_se3.npy' in z.namelist():
            with z.open('alignment_transformation_se3.npy') as f:
                data = np.load(_io.BytesIO(f.read()))
            return data[:3, :3], data[:3, 3], 1.0
    print('Warning: no alignment transformation found in zip, using identity.')
    return np.eye(3), np.zeros(3), 1.0


def load_frame_transform(tf_file):
    """Load an SE3 transform from a JSON/YAML/plain-text file.

    JSON/YAML: 'matrix' (4x4) or 'rotation' (3x3) + 'translation' (3,) keys.
    Plain text: 4x4 whitespace-delimited matrix.

    Returns a 4x4 numpy array. Returns identity if tf_file is None.
    """
    if tf_file is None:
        return np.eye(4)
    ext = os.path.splitext(tf_file)[1].lower()
    if ext in ('.json', '.yaml', '.yml'):
        if ext == '.json':
            with open(tf_file) as f:
                data = json.load(f)
        else:
            import yaml
            with open(tf_file) as f:
                data = yaml.safe_load(f)
        if 'matrix' in data:
            T = np.array(data['matrix'], dtype=float)
            assert T.shape == (4, 4), 'matrix must be 4x4'
        elif 'rotation' in data and 'translation' in data:
            T = np.eye(4)
            T[:3, :3] = np.array(data['rotation'], dtype=float)
            T[:3, 3]  = np.array(data['translation'], dtype=float)
        else:
            raise ValueError(f'Unrecognized keys in {tf_file}. '
                             "Expected 'matrix' or 'rotation'+'translation'.")
    else:
        T = np.loadtxt(tf_file)
        assert T.shape == (4, 4), 'Plain-text transform file must be 4x4'
    return T
