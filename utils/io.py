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
            except (ValueError, KeyError, TypeError):
                continue
    return kf_map


def load_loop_closures_csv(path):
    """Load loop_closures.csv.

    Returns:
        list of dicts with keys: robot1, pose1, robot2, pose2,
        and optionally tx, ty, tz, qx, qy, qz, qw if present in the CSV.
    """
    loops = []
    with open(path) as f:
        for row in _csv.DictReader(f):
            try:
                lc = {
                    'robot1': int(row['robot1']), 'pose1': int(row['pose1']),
                    'robot2': int(row['robot2']), 'pose2': int(row['pose2']),
                }
                for field in ('tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'):
                    if row.get(field) not in (None, ''):
                        lc[field] = float(row[field])
                loops.append(lc)
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


def load_variant_aliases(path=None) -> dict:
    """Load variant display-name aliases from variant_aliases.yaml.

    Searches in order: explicit path, current working directory, repo root
    (the directory containing this utils/ package).

    Returns:
        dict[str, str]: {raw_folder_name: display_label}
        Empty dict if the file is not found — callers should show all variants
        with raw names in that case.
    """
    import yaml as _yaml
    candidates = []
    if path:
        candidates.append(path)
    candidates += [
        os.path.join(os.getcwd(), 'variant_aliases.yaml'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'variant_aliases.yaml'),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p) as f:
                data = _yaml.safe_load(f)
            return {str(k): str(v) for k, v in (data or {}).items()}
    return {}


def apply_variant_alias(aliases: dict, name: str):
    """Return the display label for a variant name, or None to skip it.

    - If aliases is empty (no config file found): returns name unchanged.
    - If aliases is non-empty and name is listed: returns aliases[name].
    - If aliases is non-empty and name is NOT listed: returns None (skip).
    """
    if not aliases:
        return name
    return aliases.get(name)


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Compute Sim(3) alignment from *src* to *dst* via SVD (Umeyama 1991).

    Args:
        src: (N, 3) source points
        dst: (N, 3) destination points

    Returns:
        rotation    : (3, 3) rotation matrix
        translation : (3,)   translation vector  (dst ≈ s * R @ src + t)
        scale       : float
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    n = src.shape[0]
    sigma2 = (sc ** 2).sum() / n
    H = sc.T @ dc / n
    U, D, Vt = np.linalg.svd(H)
    det_sign = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1., 1., det_sign])
    R = Vt.T @ S @ U.T
    s = (D * np.array([1., 1., det_sign])).sum() / sigma2 if sigma2 > 0 else 1.
    return R, mu_d - s * R @ mu_s, float(s)


def is_robot_dir(d) -> bool:
    """Return True if *d* looks like a robot experiment directory.

    A robot dir is identified by having a ``distributed/`` or ``dpgo/``
    subdirectory.
    """
    from pathlib import Path as _Path
    d = _Path(d)
    return (d / 'distributed').is_dir() or (d / 'dpgo').is_dir()


def discover_variants(exp_dir) -> list:
    """Return subdirs of *exp_dir* that contain at least one robot subdir.

    A subdir qualifies as a variant if any of its immediate children is a
    robot dir (contains ``distributed/`` or ``dpgo/``).
    """
    from pathlib import Path as _Path
    exp_dir = _Path(exp_dir)
    variants = []
    for d in sorted(exp_dir.iterdir()):
        if not d.is_dir():
            continue
        try:
            if any(is_robot_dir(sub) for sub in d.iterdir() if sub.is_dir()):
                variants.append(d)
        except PermissionError:
            continue
    return variants


def discover_baselines(exp_dir) -> list:
    """Return baseline method dirs from ``baselines/<exp_dir.name>/*/``.

    Only includes dirs for which :func:`discover_robots` returns results.
    """
    from pathlib import Path as _Path
    exp_dir = _Path(exp_dir)
    baseline_root = exp_dir.parent / 'baselines' / exp_dir.name
    if not baseline_root.exists():
        return []
    return [d for d in sorted(baseline_root.iterdir())
            if d.is_dir() and discover_robots(d)]


def discover_robots(exp_dir, yaml_fallback: bool = True) -> dict:
    """Return ``{robot_id: robot_dir_name}`` for all robots in *exp_dir*.

    Discovery order:

    1. If ``robot_names.yaml`` exists and *yaml_fallback* is True, parse it.
    2. Otherwise scan ``*/dpgo/Robot *.tum`` files.  Both plain
       ``Robot N.tum`` and CBS-style ``Robot N_<timestamp>.tum`` are handled.

    Args:
        exp_dir      : experiment or variant directory to scan
        yaml_fallback: if True, try ``robot_names.yaml`` first (Kimera-Multi)

    Returns:
        dict mapping integer robot ID → robot directory name
    """
    import re as _re
    import yaml as _yaml
    from pathlib import Path as _Path
    exp_dir = _Path(exp_dir)

    if yaml_fallback:
        yaml_path = exp_dir / 'robot_names.yaml'
        if yaml_path.exists():
            with open(yaml_path) as f:
                data = _yaml.safe_load(f)
            id_to_name: dict = {}
            for key, name in (data or {}).items():
                m = _re.match(r'robot(\d+)_name', str(key))
                if m:
                    id_to_name[int(m.group(1))] = str(name)
            if id_to_name:
                return id_to_name

    id_to_name = {}
    for robot_dir in sorted(exp_dir.iterdir()):
        if not robot_dir.is_dir():
            continue
        dpgo = robot_dir / 'dpgo'
        if not dpgo.is_dir():
            continue
        for tum in sorted(dpgo.glob('Robot *.tum')):
            try:
                # handles "Robot N.tum" and "Robot N_<timestamp>.tum"
                rid = int(tum.stem.split()[-1].split('_')[0])
            except ValueError:
                continue
            id_to_name[rid] = robot_dir.name
    return id_to_name


def load_gt_trajectories_by_name(gt_dir, robot_names: list) -> dict:
    """Load GT trajectories for the given robot names.

    Searches for ``<gt_dir>/<name>.csv`` then ``<gt_dir>/<name>.txt``.

    Returns:
        ``{robot_name: (timestamps_s, positions, rotations_xyzw)}``
        where *timestamps_s* is a float64 array in seconds (converted from
        nanoseconds), *positions* is (N, 3), and *rotations_xyzw* is (N, 4).
    """
    from pathlib import Path as _Path
    gt_dir = _Path(gt_dir)
    result: dict = {}
    for name in robot_names:
        for ext in ('.csv', '.txt'):
            p = gt_dir / (name + ext)
            if p.exists():
                ts_ns, pos, rots = load_gt_trajectory(str(p))
                if len(pos):
                    result[name] = (ts_ns.astype(np.float64) / 1e9, pos, rots)
                break
    return result


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
