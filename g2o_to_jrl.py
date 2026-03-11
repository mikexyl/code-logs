#!/usr/bin/env python3
"""
Convert per-robot GTSAM g2o files to JRL (JSON Robot Log) format for MESA.

Preserves the existing per-robot variable assignment from GTSAM Symbol keys.
Factor ownership: a factor is owned by the robot that owns its first variable.
A prior factor is added on each robot's first pose to fix gauge freedom.

Usage:
    python3 g2o_to_jrl.py <robot0.g2o> [robot1.g2o ...] -o output.jrl [-n name]
    python3 g2o_to_jrl.py folder/*/dpgo/bpsam_robot_*.g2o -o dataset.jrl
"""

import argparse
import json
import sys
from pathlib import Path

import gtsam
import numpy as np

# Noise for the gauge-freedom prior on each robot's first pose.
# Tight enough to fix gauge freedom while being numerically stable.
# Matches g2o-2-mr-jrl's compute_prior_sigmas<Pose3>() defaults:
#   rotation=2 rad (~115°), translation=100 m
PRIOR_SIGMAS_ROT = 2.0   # rad
PRIOR_SIGMAS_TRANS = 1e2  # m

# JRL type tags (must match jrl/IOMeasurements.h)
TAG_BETWEEN_POSE3 = "BetweenFactorPose3"
TAG_PRIOR_POSE3   = "PriorFactorPose3"
TAG_POSE3         = "Pose3"


# ---------------------------------------------------------------------------
# Pose / covariance serialisers matching JRL's IOMeasurements.cpp
# ---------------------------------------------------------------------------

def ser_pose3(pose: gtsam.Pose3) -> dict:
    """Serialise Pose3 to JRL format: translation=[x,y,z], rotation=[w,x,y,z]."""
    q = pose.rotation().toQuaternion()  # Eigen quaternion: coeffs = [x,y,z,w]
    t = pose.translation()
    return {
        "type": "Pose3",
        "translation": [float(t[0]), float(t[1]), float(t[2])],
        "rotation": [float(q.w()), float(q.x()), float(q.y()), float(q.z())],
    }


def ser_cov(noise_model) -> list:
    """Return full d×d covariance as flat column-major list (Eigen layout)."""
    try:
        R_chol = noise_model.R()
        cov = np.linalg.inv(R_chol.T @ R_chol)
    except Exception:
        cov = np.eye(6) * 1e-4
    # Eigen/nlohmann stores column-major: iterate column first
    d = cov.shape[0]
    return [float(cov[r, c]) for c in range(d) for r in range(d)]


def prior_noise() -> gtsam.noiseModel.Diagonal:
    sigmas = np.array([
        PRIOR_SIGMAS_ROT, PRIOR_SIGMAS_ROT, PRIOR_SIGMAS_ROT,
        PRIOR_SIGMAS_TRANS, PRIOR_SIGMAS_TRANS, PRIOR_SIGMAS_TRANS,
    ])
    return gtsam.noiseModel.Diagonal.Sigmas(sigmas)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(g2o_files: list[Path], output: Path, name: str) -> None:
    # 1. Load all g2o files
    all_graph = gtsam.NonlinearFactorGraph()
    all_values = gtsam.Values()
    for path in g2o_files:
        graph, values = gtsam.readG2o(str(path), True)
        for i in range(graph.size()):
            all_graph.push_back(graph.at(i))
        for k in values.keys():
            if not all_values.exists(k):
                all_values.insert(k, values.atPose3(k))

    print(f"  {all_values.size()} poses, {all_graph.size()} factors")

    # 2. Discover robots from key characters (GTSAM Symbol high-byte encoding)
    robot_chars: set[int] = set()
    for k in all_values.keys():
        robot_chars.add(gtsam.symbolChr(k))
    robots = sorted(robot_chars)
    robot_labels = [chr(c) for c in robots]
    print(f"  Robots: {robot_labels}")

    # Build per-robot sorted key lists
    chr_to_sorted_keys: dict[int, list] = {}
    for k in all_values.keys():
        c = gtsam.symbolChr(k)
        chr_to_sorted_keys.setdefault(c, []).append((gtsam.symbolIndex(k), k))
    for c in chr_to_sorted_keys:
        chr_to_sorted_keys[c].sort()

    # 3. Assign factors to robots
    #    A factor is owned by the robot that owns its FIRST variable.
    chr_to_factors: dict[int, list] = {c: [] for c in robots}
    for i in range(all_graph.size()):
        factor = all_graph.at(i)
        keys = list(factor.keys())
        if not keys:
            continue
        owner_chr = gtsam.symbolChr(keys[0])
        if owner_chr in chr_to_factors:
            chr_to_factors[owner_chr].append(factor)

    # 4. Build JRL JSON structure
    #    measurements[robot] = [ { stamp: 0, measurements: [...] } ]
    #    initialization[robot] = [ { key: uint64, type: "Pose3", ... } ]
    measurements_json: dict[str, list] = {}
    initialization_json: dict[str, list] = {}

    pnoise = prior_noise()

    for c in robots:
        label = chr(c)
        factors = chr_to_factors[c]
        sorted_keys = chr_to_sorted_keys.get(c, [])

        if not sorted_keys:
            continue

        meas_list = []

        # Add gauge-freedom prior on first pose of each robot
        first_key = sorted_keys[0][1]
        first_pose = all_values.atPose3(first_key)
        prior_cov = ser_cov(pnoise)
        meas_list.append({
            "type": TAG_PRIOR_POSE3,
            "key": int(first_key),
            "prior": ser_pose3(first_pose),
            "covariance": prior_cov,
        })

        # Add all BetweenFactorPose3 owned by this robot
        for factor in factors:
            if not isinstance(factor, gtsam.BetweenFactorPose3):
                continue
            k1, k2 = factor.keys()[0], factor.keys()[1]
            meas = factor.measured()
            cov = ser_cov(factor.noiseModel())
            meas_list.append({
                "type": TAG_BETWEEN_POSE3,
                "key1": int(k1),
                "key2": int(k2),
                "measurement": ser_pose3(meas),
                "covariance": cov,
            })

        measurements_json[label] = [{"stamp": 0, "measurements": meas_list}]

        # Initialization: all poses known to this robot (its own + cross-robot)
        init_list = []
        robot_known_keys: set[int] = set()
        for k in all_values.keys():
            # Include own keys
            if gtsam.symbolChr(k) == c:
                robot_known_keys.add(k)
        # Also include keys of other robots referenced in this robot's factors
        for factor in factors:
            for k in factor.keys():
                robot_known_keys.add(k)

        for k in sorted(robot_known_keys):
            pose = all_values.atPose3(k)
            entry = ser_pose3(pose)
            entry["key"] = int(k)
            init_list.append(entry)
        initialization_json[label] = init_list

    # JRL expects robots as a list of ASCII integer codes (nlohmann get<vector<char>>)
    jrl_doc = {
        "name": name,
        "robots": [ord(l) for l in robot_labels],
        "measurements": measurements_json,
        "initialization": initialization_json,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(jrl_doc))
    print(f"  Wrote {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert per-robot g2o files to JRL format.")
    parser.add_argument("g2o_files", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("-n", "--name", default=None)
    args = parser.parse_args()

    name = args.name or args.output.stem
    g2o_files = sorted(args.g2o_files)
    print(f"Converting {len(g2o_files)} g2o file(s) → {args.output}")
    convert(g2o_files, args.output, name)


if __name__ == "__main__":
    main()
