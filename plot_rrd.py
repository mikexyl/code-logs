#!/usr/bin/env python3
"""
Query and visualize data from a Rerun .rrd recording file.

Usage:
    # Print all available topics
    python plot_rrd.py <file.rrd>

    # Visualize landmarks with Open3D
    python plot_rrd.py <file.rrd> --landmarks
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import rerun as rr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_dataset(rrd_file: Path):
    """Start a local Rerun server and return a catalog DatasetEntry."""
    server = rr.server.Server(datasets={"rrd": [rrd_file]})
    client = rr.catalog.CatalogClient(server.url())
    return client.get_dataset("rrd"), server  # keep server alive


def _last_valid(table, column_name: str):
    """Return the last non-null value in a PyArrow table column as a Python list."""
    try:
        col = table.column(column_name)
    except KeyError:
        return None
    for i in range(len(col) - 1, -1, -1):
        if col[i].is_valid:
            return col[i].as_py()
    return None


def _quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternion to 3x3 rotation matrix."""
    x, y, z, w = q.flatten()
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def _get_transform_matrix(table, entity: str) -> np.ndarray:
    """Extract a 4x4 SE3 transform from Rerun Transform3D columns; identity if absent."""
    T = np.eye(4)

    # Translation
    for suffix in ["Translation3D:translation", "Transform3D:translation"]:
        val = _last_valid(table, f"{entity}:{suffix}")
        if val is not None:
            T[:3, 3] = np.array(val, dtype=np.float64).flatten()[:3]
            break

    # Rotation: try quaternion (xyzw in Rerun)
    for suffix in ["RotationQuat:quaternion", "Transform3D:quaternion"]:
        val = _last_valid(table, f"{entity}:{suffix}")
        if val is not None:
            T[:3, :3] = _quat_xyzw_to_matrix(np.array(val, dtype=np.float64))
            return T

    # Rotation: try 3x3 matrix
    for suffix in ["TransformMat3x3:coeffs", "RotationMat3x3:coeffs"]:
        val = _last_valid(table, f"{entity}:{suffix}")
        if val is not None:
            T[:3, :3] = np.array(val, dtype=np.float64).reshape(3, 3)
            return T

    return T


def _unpack_rgba(packed: list[int]) -> np.ndarray:
    """Convert a list of RGBA uint32 values to an Nx3 float array in [0, 1]."""
    arr = np.array(packed, dtype=np.uint32)
    r = ((arr >> 24) & 0xFF).astype(np.float32) / 255.0
    g = ((arr >> 16) & 0xFF).astype(np.float32) / 255.0
    b = ((arr >> 8) & 0xFF).astype(np.float32) / 255.0
    return np.stack([r, g, b], axis=1)



# ---------------------------------------------------------------------------
# Schema printer
# ---------------------------------------------------------------------------

def print_schema(rrd_file: Path) -> None:
    archive = rr.recording.load_archive(str(rrd_file))
    recordings = list(archive.all_recordings())
    print(f"Found {len(recordings)} recording(s) in archive.")

    for i, recording in enumerate(recordings):
        schema = recording.schema()

        print(f"\n{'='*60}")
        print(f"Recording {i}: {recording.application_id()}  "
              f"(id: {recording.recording_id()})")
        print(f"{'='*60}")

        print("\n=== Timeline / Index Columns ===")
        for col in schema.index_columns():
            print(f"  {col.name}")

        print("\n=== Entity / Component Columns ===")
        by_entity: dict[str, list[str]] = defaultdict(list)
        for col in schema.component_columns():
            by_entity[col.entity_path].append(col.component)

        for entity_path in sorted(by_entity):
            print(f"  {entity_path}")
            for component in sorted(by_entity[entity_path]):
                print(f"    - {component}")

        print(f"\nTotal entities : {len(by_entity)}")
        print(f"Total components: {sum(len(v) for v in by_entity.values())}")


# ---------------------------------------------------------------------------
# Landmark visualizer
# ---------------------------------------------------------------------------

def visualize_landmarks(rrd_file: Path) -> None:
    import pyvista as pv

    print(f"Loading: {rrd_file}")
    dataset, server = _open_dataset(rrd_file)  # noqa: F841  keep server alive

    # Discover landmark entities
    archive = rr.recording.load_archive(str(rrd_file))
    recording = list(archive.all_recordings())[0]
    landmark_entities = sorted(
        {col.entity_path for col in recording.schema().component_columns()
         if col.entity_path.endswith("/landmarks")}
    )

    traj_entities = sorted(
        {col.entity_path for col in recording.schema().component_columns()
         if "/traj/" in col.entity_path}
    )

    if not landmark_entities and not traj_entities:
        print("No landmark or trajectory entities found.")
        return

    print(f"Found landmark entities:   {landmark_entities}")
    print(f"Found trajectory entities: {traj_entities}")

    # Fallback colors per robot when the recording has no per-point color
    palette = [
        [0.2, 0.6, 1.0],
        [1.0, 0.4, 0.2],
        [0.2, 0.9, 0.4],
        [0.9, 0.2, 0.8],
    ]

    clouds = []   # list of (pts Nx3, rgb Nx3)
    for idx, entity in enumerate(landmark_entities):
        print(f"\nQuerying last frame of '{entity}' ...")

        # entity  = "map/<robot_name>/landmarks"
        # parent  = "map/<robot_name>"
        parent_entity = entity.rsplit("/", 1)[0]

        # --- query landmark entity (points + its local transform) ---
        view = dataset.filter_contents(entity)
        table = view.reader(index="log_time").to_arrow_table()

        positions = _last_valid(table, f"{entity}:Points3D:positions")
        if positions is None:
            print("  No position data found, skipping.")
            continue

        pts = np.array(positions, dtype=np.float64)
        print(f"  {len(pts)} points")

        # T2: transform logged at /map/<robot_name>/landmarks
        T_landmarks = _get_transform_matrix(table, entity)

        # --- query parent entity for its transform ---
        parent_view = dataset.filter_contents(parent_entity)
        parent_table = parent_view.reader(index="log_time").to_arrow_table()
        # T1: transform logged at /map/<robot_name>
        T_robot = _get_transform_matrix(parent_table, parent_entity)

        # Compose: T_world = T_robot @ T_landmarks, then apply to points
        T_world = T_robot @ T_landmarks
        ones = np.ones((len(pts), 1), dtype=np.float64)
        pts_world = (T_world @ np.hstack([pts, ones]).T).T[:, :3]
        print(f"  T_robot translation: {T_robot[:3, 3].round(3)}")
        print(f"  T_landmarks translation: {T_landmarks[:3, 3].round(3)}")

        pts = pts_world.astype(np.float32)

        colors_raw = _last_valid(table, f"{entity}:Points3D:colors")
        if colors_raw is not None:
            rgb = _unpack_rgba(colors_raw).astype(np.float32)
        else:
            color = np.array(palette[idx % len(palette)], dtype=np.float32)
            rgb = np.tile(color, (len(pts), 1))

        clouds.append((pts, rgb))

    # --- Load trajectories ---
    trajs = []   # list of (list of Nx3 float32 strips, color [r,g,b])
    for idx, entity in enumerate(traj_entities):
        print(f"\nQuerying trajectory '{entity}' ...")

        view = dataset.filter_contents(entity)
        table = view.reader(index="log_time").to_arrow_table()

        # Try LineStrips3D (growing strip — last frame has full trajectory)
        strips_raw = _last_valid(table, f"{entity}:LineStrips3D:strips")
        if strips_raw is not None:
            strips = [np.array(s, dtype=np.float64) for s in strips_raw if len(s) >= 2]
        else:
            # Fallback: individual positions logged per frame
            pos_raw = _last_valid(table, f"{entity}:Points3D:positions")
            if pos_raw is None:
                print("  No trajectory data found, skipping.")
                continue
            strips = [np.array(pos_raw, dtype=np.float64)]

        print(f"  {sum(len(s) for s in strips)} points across {len(strips)} strip(s)")

        # Apply transform at /map/traj/<robot_name>
        T = _get_transform_matrix(table, entity)
        print(f"  T_traj translation: {T[:3, 3].round(3)}")

        transformed_strips = []
        for s in strips:
            ones = np.ones((len(s), 1), dtype=np.float64)
            s_world = (T @ np.hstack([s, ones]).T).T[:, :3].astype(np.float32)
            transformed_strips.append(s_world)

        # Read per-strip colors from the recording
        colors_raw = _last_valid(table, f"{entity}:LineStrips3D:colors")
        if colors_raw is not None:
            strip_colors = _unpack_rgba(colors_raw).tolist()  # one [r,g,b] per strip
            print(f"  using recorded colors ({len(strip_colors)} entries)")
        else:
            # Debug: show available columns to help identify the correct name
            traj_cols = [c for c in table.schema.names if entity in c and "color" in c.lower()]
            if traj_cols:
                print(f"  color columns found but not matched: {traj_cols}")
            strip_colors = [palette[idx % len(palette)]] * len(transformed_strips)
            print(f"  using fallback palette color")

        trajs.append((transformed_strips, strip_colors))

    if not clouds and not trajs:
        print("No data to visualize.")
        return

    if clouds:
        print("\n=== Point Cloud Summary ===")
        for entity, (pts, _) in zip(landmark_entities, clouds):
            print(f"  {entity}: {len(pts)} points  "
                  f"bbox=[{pts.min(axis=0).round(2)}, {pts.max(axis=0).round(2)}]")

    print("\nLaunching PyVista viewer  (Q to quit) ...")
    plotter = pv.Plotter(window_size=(1280, 960))
    plotter.set_background([1.0, 1.0, 1.0])

    actors = []
    for pts, rgb in clouds:
        cloud = pv.PolyData(pts)
        cloud["rgb"] = (rgb * 255).astype(np.uint8)
        actor = plotter.add_points(
            cloud,
            scalars="rgb",
            rgb=True,
            point_size=2.0,
            render_points_as_spheres=False,
        )
        actors.append(actor)

    for strips, strip_colors in trajs:
        for strip, color in zip(strips, strip_colors):
            if len(strip) >= 2:
                line = pv.lines_from_points(strip)
                plotter.add_mesh(line, color=color, line_width=2.0)

    def set_point_size(value: float) -> None:
        for actor in actors:
            actor.GetProperty().SetPointSize(value)
        plotter.render()

    plotter.add_slider_widget(
        callback=set_point_size,
        rng=[1, 20],
        value=2.0,
        title="Point Size",
        pointa=(0.025, 0.1),
        pointb=(0.225, 0.1),
        style="modern",
    )

    plotter.show()



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query and visualize a Rerun .rrd file."
    )
    parser.add_argument("rrd_file", type=Path, help="Path to the .rrd file")
    parser.add_argument(
        "--landmarks", action="store_true",
        help="Visualize /map/*/landmarks and /map/traj/* with PyVista"
    )
    args = parser.parse_args()

    rrd_file: Path = args.rrd_file.resolve()
    if not rrd_file.exists():
        print(f"Error: {rrd_file} does not exist.")
        raise SystemExit(1)

    if args.landmarks:
        visualize_landmarks(rrd_file)
    else:
        print_schema(rrd_file)


if __name__ == "__main__":
    main()
