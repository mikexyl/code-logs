# AGENTS.md

This file provides guidance to Codex when working with code in this repository. It is derived from and should stay aligned with [`CLAUDE.md`](/home/mikexyl/workspaces/kimera_noetic_ws/src/code-logs/CLAUDE.md).

## Overview

Multi-robot SLAM evaluation toolkit. The repo processes Kimera-Multi outputs, runs offline optimization, evaluates ATE/RPE with `evo`, extracts ground-truth loop closures, and generates publication plots.

## Environment

- Run Python through `pixi`.
- Preferred commands:
  - `pixi shell`
  - `pixi run python3 <script.py> ...`
  - `pixi run build`
- Python version: 3.11.
- Common packages: `numpy`, `matplotlib`, `scipy`, `evo`, `rerun-sdk`, `open3d`, `pyvista`.

The C++ optimizer binary is `build/optimize_offline`. It hardcodes `/workspaces/src/code-logs/` in [`src/optimize_offline.cpp`](/home/mikexyl/workspaces/kimera_noetic_ws/src/code-logs/src/optimize_offline.cpp), so adjust that path if the workspace moves.

## Working Rules

- Reuse `utils/` helpers instead of duplicating parsing, alignment, or plotting logic.
- Keep new figures consistent with IEEE styling via `utils.plot.IEEE_RC`.
- Use `save_fig(fig, base_path)` to emit both `.pdf` and `.png`.
- Preserve existing variant auto-discovery behavior unless there is a clear bug.
- Be careful with timestamp units and quaternion ordering; this repo mixes multiple conventions.

## Full Analysis Pipeline

For a complete experiment analysis, run these in order:

```bash
pixi run python3 evaluate.py <exp> --gt_folder ground_truth
pixi run python3 evaluate_loops_recall.py <exp> ground_truth/<exp> --tol 5.0
pixi run python3 plot_algebraic_connectivity.py <exp>
pixi run python3 plot_loop_rotation_dist.py <exp> ground_truth/<exp>
pixi run python3 plot_scalability.py <exp>
pixi run python3 plot_ablation.py <exp>
```

Notes:
- For UAV datasets such as `a5678`, add `--max-angle 60` to `evaluate_loops_recall.py`.
- `plot_algebraic_connectivity.py` depends on `inlier_loops.csv` from `evaluate_loops_recall.py`.
- `plot_ablation.py` expects `*_bandwidth.npy` and `*_loops.npy` extracted separately from `.rrd` files via `plot_rrd.py --bandwidth` and `plot_rrd.py --loops`.

## Script Notes

- [`evaluate.py`](/home/mikexyl/workspaces/kimera_noetic_ws/src/code-logs/evaluate.py): evaluates ATE/RPE, auto-discovers variants, writes outputs inside each variant, and overlays loops only when `distributed/` metadata exists.
- [`evaluate_loops_recall.py`](/home/mikexyl/workspaces/kimera_noetic_ws/src/code-logs/evaluate_loops_recall.py): computes recall and outlier metrics, writes `loops_recall.csv` and `inlier_loops.csv`, and powers several downstream plots.
- [`extract_gt_loops.py`](/home/mikexyl/workspaces/kimera_noetic_ws/src/code-logs/extract_gt_loops.py): extracts GT inter-robot loop closures with separate XY/Z thresholds and optional UAV-specific trimming.
- [`plot_rrd.py`](/home/mikexyl/workspaces/kimera_noetic_ws/src/code-logs/plot_rrd.py): inspects Rerun recordings and extracts bandwidth or loop-count arrays.
- [`plot_loop_rotation_dist.py`](/home/mikexyl/workspaces/kimera_noetic_ws/src/code-logs/plot_loop_rotation_dist.py): plots GT relative rotation of detected loops.
- [`plot_algebraic_connectivity.py`](/home/mikexyl/workspaces/kimera_noetic_ws/src/code-logs/plot_algebraic_connectivity.py): computes Laplacian `lambda_2` using odometry plus inlier inter-robot edges.
- [`plot_scalability.py`](/home/mikexyl/workspaces/kimera_noetic_ws/src/code-logs/plot_scalability.py): plots bandwidth vs recall or ATE and emits yield plots.
- [`plot_ablation.py`](/home/mikexyl/workspaces/kimera_noetic_ws/src/code-logs/plot_ablation.py): compares extracted bandwidth/loop curves and recall across variants.
- [`evaluate_mesa_baselines.py`](/home/mikexyl/workspaces/kimera_noetic_ws/src/code-logs/evaluate_mesa_baselines.py): evaluates both MESA methods and CBS/CBS+ outputs; CBS uses the last non-empty timestamped `.tum` snapshot per robot.

## Shared Utilities

Prefer these helpers:

- [`utils/io.py`](/home/mikexyl/workspaces/kimera_noetic_ws/src/code-logs/utils/io.py)
  - `read_tum_trajectory`
  - `load_gt_trajectory`
  - `load_keyframes_csv`
  - `load_loop_closures_csv`
  - `load_alignment_from_evo_zip`
  - `load_frame_transform`
- [`utils/plot.py`](/home/mikexyl/workspaces/kimera_noetic_ws/src/code-logs/utils/plot.py)
  - `IEEE_RC`
  - `ROBOT_COLORS`
  - `save_fig`
  - `apply_alignment`
  - `apply_frame_transform`
  - `find_tum_position`
  - `mark_endpoint`

Use:

```python
plt.rcParams.update({**IEEE_RC, "figure.figsize": (3.5, 2.5)})
```

when a script needs a custom size without losing house style.

## Critical Conventions

- GT CSV files use nanosecond timestamps and quaternion order `qw qx qy qz`.
- TUM files use seconds and quaternion order `qx qy qz qw`.
- Keyframe CSVs store nanoseconds in `keyframe_stamp_ns`.
- `utils/io.py` normalizes trajectories internally to seconds and `xyzw`.

- `loop_closures.csv` relative poses represent `T_robot1^robot2` in robot1's local frame.
- `inlier_loops.csv` contains `name1, t1_s, name2, t2_s` and is regenerated by `evaluate_loops_recall.py`.
- Bandwidth `.npy` files are pickled dicts with keys `t_sec`, `bow_MB`, `vlc_MB`, `cbs_MB`.
- Loop-count `.npy` files use `t_sec`, `pr_total`, `gv_total`.

- `variant_aliases.yaml` controls which variants appear in comparison plots. If a variant is missing there, it may be skipped from multi-variant figures.
- Variant auto-discovery treats a directory as a variant if it contains at least one robot directory with `distributed/` or `dpgo/`.

## Architecture Notes

- Optimizer key remapping uses `new_key = robot_id * 10000 + original_pose_id`.
- In GTSAM/g2o interchange, SE(3) information ordering differs; [`src/optimize_offline.cpp`](/home/mikexyl/workspaces/kimera_noetic_ws/src/code-logs/src/optimize_offline.cpp) already handles the swap in `writeG2oNoIndex()`.
- CBS/CBS+ `stats_robot_<N>.csv` iteration counts are synchronous rounds and should not be multiplied by robot count.
- MESA `Iteration` is also synchronous rounds, while `TotalCommunications` is iterations times robot count.

## Data Layout

Typical structure:

```text
<experiment>/
  <variant>/
    <robot_dir>/
      dpgo/
      distributed/
    cbs/
    cbs_plus/
  recall_comparison.pdf/png
  outlier_comparison.pdf/png
  algebraic_connectivity.pdf/png

ground_truth/
  <experiment>/
    <robot_dir>.csv
    <robot_dir>.txt
```

Kimera-Multi baselines live under `baselines/<experiment>/Kimera-Multi/<robot>/`.
