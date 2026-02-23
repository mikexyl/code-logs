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
    col = table.column(column_name)
    for i in range(len(col) - 1, -1, -1):
        if col[i].is_valid:
            return col[i].as_py()
    return None


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

    if not landmark_entities:
        print("No landmark entities found.")
        return

    print(f"Found landmark entities: {landmark_entities}")

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

        view = dataset.filter_contents(entity)
        table = view.reader(index="log_time").to_arrow_table()

        positions = _last_valid(table, f"{entity}:Points3D:positions")
        if positions is None:
            print("  No position data found, skipping.")
            continue

        pts = np.array(positions, dtype=np.float32)
        print(f"  {len(pts)} points")

        colors_raw = _last_valid(table, f"{entity}:Points3D:colors")
        if colors_raw is not None:
            rgb = _unpack_rgba(colors_raw).astype(np.float32)
        else:
            color = np.array(palette[idx % len(palette)], dtype=np.float32)
            rgb = np.tile(color, (len(pts), 1))

        clouds.append((pts, rgb))

    if not clouds:
        print("No data to visualize.")
        return

    print("\n=== Point Cloud Summary ===")
    for entity, (pts, _) in zip(landmark_entities, clouds):
        print(f"  {entity}: {len(pts)} points  "
              f"bbox=[{pts.min(axis=0).round(2)}, {pts.max(axis=0).round(2)}]")

    print("\nLaunching PyVista viewer  (Q to quit) ...")
    plotter = pv.Plotter(window_size=(1280, 960))
    plotter.set_background([0.05, 0.05, 0.05])

    for pts, rgb in clouds:
        cloud = pv.PolyData(pts)
        cloud["rgb"] = (rgb * 255).astype(np.uint8)
        plotter.add_points(
            cloud,
            scalars="rgb",
            rgb=True,
            point_size=2.0,
            render_points_as_spheres=False,
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
        help="Visualize /map/*/landmarks with Open3D"
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
