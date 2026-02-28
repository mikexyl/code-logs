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

        # Read colors — stored as Points3D:colors (per-point RGBA)
        # All points in one robot's trajectory share the same color, so use [0].
        colors_raw = _last_valid(table, f"{entity}:Points3D:colors")
        if colors_raw is not None:
            uniform_color = _unpack_rgba(colors_raw)[0].tolist()
            strip_colors = [uniform_color] * len(transformed_strips)
        else:
            strip_colors = [palette[idx % len(palette)]] * len(transformed_strips)

        trajs.append((transformed_strips, strip_colors))

    if not clouds and not trajs:
        print("No data to visualize.")
        return

    if clouds:
        print("\n=== Point Cloud Summary ===")
        for entity, (pts, _) in zip(landmark_entities, clouds):
            print(f"  {entity}: {len(pts)} points  "
                  f"bbox=[{pts.min(axis=0).round(2)}, {pts.max(axis=0).round(2)}]")

    # --- Save landmark point clouds to .pcd files ---
    try:
        import open3d as o3d
        for entity, (pts, rgb) in zip(landmark_entities, clouds):
            entity_tag = entity.strip("/").replace("/", "_")
            pcd_path = rrd_file.with_name(f"{rrd_file.stem}_{entity_tag}.pcd")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
            o3d.io.write_point_cloud(str(pcd_path), pcd)
            print(f"Saved landmarks to {pcd_path}")
    except ImportError:
        print("open3d not available — skipping .pcd export.")

    # --- Save trajectories to .npy files ---
    for entity, (strips, _) in zip(traj_entities, trajs):
        entity_tag = entity.strip("/").replace("/", "_")
        npy_path = rrd_file.with_name(f"{rrd_file.stem}_{entity_tag}.npy")
        strips_arr = np.empty(len(strips), dtype=object)
        for i, s in enumerate(strips):
            strips_arr[i] = s
        np.save(str(npy_path), strips_arr, allow_pickle=True)
        print(f"Saved trajectory to {npy_path}")

    print("\nLaunching PyVista viewer  (Q to quit) ...")
    plotter = pv.Plotter(window_size=(1280, 960))
    plotter.set_background([1.0, 1.0, 1.0])
    plotter.render_window.PointSmoothingOn()  # enables anti-aliased fractional point sizes

    actors = []
    for pts, rgb in clouds:
        cloud = pv.PolyData(pts)
        cloud["rgb"] = (rgb * 255).astype(np.uint8)
        actor = plotter.add_points(
            cloud,
            scalars="rgb",
            rgb=True,
            point_size=1.2,
            render_points_as_spheres=False,
        )
        actors.append(actor)

    traj_actors = []
    for strips, strip_colors in trajs:
        for strip, color in zip(strips, strip_colors):
            if len(strip) >= 2:
                dark_color = [max(0.0, c * 0.55) for c in color]
                line = pv.lines_from_points(strip)
                traj_actor = plotter.add_mesh(line, color=dark_color, line_width=4.0)
                traj_actors.append(traj_actor)

    def set_point_size(value: float) -> None:
        for actor in actors:
            actor.GetProperty().SetPointSize(value)
        plotter.render()

    def set_line_width(value: float) -> None:
        for actor in traj_actors:
            actor.GetProperty().SetLineWidth(value)
        plotter.render()

    slider_widgets = []
    checkbox_widget = None
    screenshot_text_actor = None

    def take_screenshot() -> None:
        import matplotlib.pyplot as plt
        # Hide UI overlays
        for w in slider_widgets:
            w.Off()
        if checkbox_widget is not None:
            checkbox_widget.Off()
        if screenshot_text_actor is not None:
            screenshot_text_actor.SetVisibility(False)
        plotter.render()

        img = plotter.screenshot(None)  # returns RGB numpy array

        # Restore UI overlays
        for w in slider_widgets:
            w.On()
        if checkbox_widget is not None:
            checkbox_widget.On()
        if screenshot_text_actor is not None:
            screenshot_text_actor.SetVisibility(True)
        plotter.render()

        path = str(rrd_file.with_name(rrd_file.stem + "-map.pdf"))
        h, w_px = img.shape[:2]
        fig, ax = plt.subplots(figsize=(w_px / 100, h / 100), dpi=100)
        ax.imshow(img)
        ax.axis("off")
        fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"Screenshot saved to {path}")

    slider_widgets.append(plotter.add_slider_widget(
        callback=set_point_size,
        rng=[1.0, 8.0],
        value=1.2,
        title="Point Size",
        pointa=(0.025, 0.10),
        pointb=(0.225, 0.10),
        style="modern",
    ))
    slider_widgets.append(plotter.add_slider_widget(
        callback=set_line_width,
        rng=[1.0, 10.0],
        value=4.0,
        title="Traj Width",
        pointa=(0.025, 0.02),
        pointb=(0.225, 0.02),
        style="modern",
    ))
    checkbox_widget = plotter.add_checkbox_button_widget(
        callback=lambda _: take_screenshot(),
        value=False,
        position=(10, 100),
        size=30,
        color_on="steelblue",
        color_off="steelblue",
        background_color="white",
    )
    screenshot_text_actor = plotter.add_text(
        "Screenshot", position=(45, 105), font_size=9, color="black"
    )

    plotter.show()



# ---------------------------------------------------------------------------
# Bandwidth visualizer
# ---------------------------------------------------------------------------

def plot_bandwidth(rrd_file: Path, output: Path | None = None) -> None:
    """
    Plot cumulative received bandwidth over time, stacked by module.

    Discovers robots via entities of the form:
        /<robot>/received_bow_byte   (BOW vectors)
        /<robot>/received_vlc_byte   (VLC frames)
        /<robot>/bandwidth_recv_bytes  (CBS backend)

    Sums all robots together and draws a stacked-area chart.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    dataset, server = _open_dataset(rrd_file)  # noqa: F841  keep server alive

    archive = rr.recording.load_archive(str(rrd_file))
    recording = list(archive.all_recordings())[0]
    schema = recording.schema()

    # Discover robot names from /<robot>/received_bow_byte entities
    robots = sorted({
        col.entity_path.lstrip("/").split("/")[0]
        for col in schema.component_columns()
        if col.entity_path.endswith("/received_bow_byte")
    })
    print(f"Found robots: {robots}")

    def get_series(entity_path: str) -> pd.Series:
        """Return a pd.Series (datetime64[ns] index → float value)."""
        view = dataset.filter_contents(entity_path)
        table = view.reader(index="log_time").to_arrow_table()
        col_key = f"{entity_path}:Scalars:scalars"
        try:
            df = table.select(["log_time", col_key]).to_pandas()
        except KeyError:
            return pd.Series(dtype=float)
        df = df.dropna(subset=[col_key])
        df["value"] = df[col_key].apply(lambda v: float(np.asarray(v).flat[0]))
        return pd.Series(df["value"].values, index=pd.to_datetime(df["log_time"]))

    # Collect per-robot series
    bow_all, vlc_all, cbs_all = [], [], []
    for robot in robots:
        bow_all.append(get_series(f"/{robot}/received_bow_byte"))
        vlc_all.append(get_series(f"/{robot}/received_vlc_byte"))
        cbs_all.append(get_series(f"/{robot}/bandwidth_recv_bytes"))

    def sum_series(series_list: list[pd.Series], t_common: pd.DatetimeIndex) -> np.ndarray:
        """Interpolate each series onto t_common and sum."""
        total = np.zeros(len(t_common), dtype=np.float64)
        t_ns = t_common.view("int64").astype(np.float64)
        for s in series_list:
            if s.empty:
                continue
            s_ns = s.index.view("int64").astype(np.float64)
            interp = np.interp(t_ns, s_ns, s.values.astype(np.float64),
                               left=0.0, right=float(s.iloc[-1]))
            total += interp
        return total

    # Common time axis spanning the full experiment
    all_series = bow_all + vlc_all + cbs_all
    non_empty = [s for s in all_series if not s.empty]
    if not non_empty:
        print("No bandwidth data found.")
        return

    t_min = min(s.index.min() for s in non_empty)
    t_max = max(s.index.max() for s in non_empty)
    t_common = pd.date_range(t_min, t_max, periods=800)

    bow_total = sum_series(bow_all, t_common) / 1e6   # → MB
    vlc_total = sum_series(vlc_all, t_common) / 1e6
    cbs_total = sum_series(cbs_all, t_common) / 1e6

    # x-axis: seconds from experiment start
    t_sec = (t_common - t_min).total_seconds().values

    # IEEE Transactions single-column formatting (3.5 in wide, 4:3 aspect ratio)
    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'figure.figsize': (3.5, 3.5 * 2 / 4),   # single column, 4:3
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 0.5,
        'lines.linewidth': 1.0,
        'patch.linewidth': 0.5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    fig, ax = plt.subplots()
    ax.stackplot(
        t_sec,
        bow_total, vlc_total, cbs_total,
        labels=["Global Desc.", "Sequences", "Beliefs"],
        colors=["#4C72B0", "#DD8452", "#55A868"],
        alpha=0.80,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Communication (MB)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.3)
    ax.set_yscale("linear")
    ax.set_xlim(t_sec[0], t_sec[-1])
    plt.tight_layout()

    # Derive base path: experiment folder (parent of rrd file), stem = "<rrd_stem>_bandwidth"
    base = rrd_file.parent / (rrd_file.stem + "_bandwidth")
    if output is not None:
        # If the caller explicitly passed a path, honour it but still save both formats
        base = output.with_suffix("")

    pdf_path = base.with_suffix(".pdf")
    png_path = base.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"Saved to {pdf_path}")
    print(f"Saved to {png_path}")
    plt.close(fig)

    npy_path = base.with_suffix(".npy")
    np.save(str(npy_path), {
        "t_sec": t_sec,
        "bow_MB": bow_total,
        "vlc_MB": vlc_total,
        "cbs_MB": cbs_total,
    }, allow_pickle=True)
    print(f"Saved to {npy_path}")


# ---------------------------------------------------------------------------
# Loop counter visualizer
# ---------------------------------------------------------------------------

def plot_loops(rrd_file: Path, output: Path | None = None) -> None:
    """
    Plot the number of place-recognition loops and geometrically-verified loops
    over time, summed across all robots.

    Discovers robots via entities of the form:
        /<robot>/num_pr_loops   (place recognition detections)
        /<robot>/num_gv_loops   (geometric verification detections)
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    dataset, server = _open_dataset(rrd_file)  # noqa: F841  keep server alive

    archive = rr.recording.load_archive(str(rrd_file))
    recording = list(archive.all_recordings())[0]
    schema = recording.schema()

    # Discover robot names from /<robot>/num_pr_loops entities
    robots = sorted({
        col.entity_path.lstrip("/").split("/")[0]
        for col in schema.component_columns()
        if col.entity_path.endswith("/num_pr_loops")
    })
    if not robots:
        robots = sorted({
            col.entity_path.lstrip("/").split("/")[0]
            for col in schema.component_columns()
            if col.entity_path.endswith("/num_gv_loops")
        })
    print(f"Found robots: {robots}")

    def get_series(entity_path: str) -> pd.Series:
        """Return a pd.Series (datetime64[ns] index → float value)."""
        view = dataset.filter_contents(entity_path)
        table = view.reader(index="log_time").to_arrow_table()
        col_key = f"{entity_path}:Scalars:scalars"
        try:
            df = table.select(["log_time", col_key]).to_pandas()
        except KeyError:
            return pd.Series(dtype=float)
        df = df.dropna(subset=[col_key])
        df["value"] = df[col_key].apply(lambda v: float(np.asarray(v).flat[0]))
        return pd.Series(df["value"].values, index=pd.to_datetime(df["log_time"]))

    pr_all, gv_all = [], []
    for robot in robots:
        pr_all.append(get_series(f"/{robot}/num_pr_loops"))
        gv_all.append(get_series(f"/{robot}/num_gv_loops"))

    all_series = pr_all + gv_all
    non_empty = [s for s in all_series if not s.empty]
    if not non_empty:
        print("No loop data found.")
        return

    t_min = min(s.index.min() for s in non_empty)
    t_max = max(s.index.max() for s in non_empty)
    t_common = pd.date_range(t_min, t_max, periods=800)

    def sum_series(series_list: list[pd.Series], t_common: pd.DatetimeIndex) -> np.ndarray:
        """Interpolate each series onto t_common and sum."""
        total = np.zeros(len(t_common), dtype=np.float64)
        t_ns = t_common.view("int64").astype(np.float64)
        for s in series_list:
            if s.empty:
                continue
            s_ns = s.index.view("int64").astype(np.float64)
            interp = np.interp(t_ns, s_ns, s.values.astype(np.float64),
                               left=0.0, right=float(s.iloc[-1]))
            total += interp
        return total

    pr_total = sum_series(pr_all, t_common)
    gv_total = sum_series(gv_all, t_common)

    t_sec = (t_common - t_min).total_seconds().values

    # IEEE Transactions single-column formatting (3.5 in wide, 4:3 aspect ratio)
    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'figure.figsize': (3.5, 3.5 * 3 / 4),   # single column, 4:3
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 0.5,
        'lines.linewidth': 1.0,
        'patch.linewidth': 0.5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    fig, ax = plt.subplots()
    ax.plot(t_sec, pr_total, label="PR Loops",  color="#4C72B0")
    ax.plot(t_sec, gv_total, label="GV Loops",  color="#DD8452")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Number of Loops Detected")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.3)
    plt.tight_layout()

    base = rrd_file.parent / (rrd_file.stem + "_loops")
    if output is not None:
        base = output.with_suffix("")

    pdf_path = base.with_suffix(".pdf")
    png_path = base.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"Saved to {pdf_path}")
    print(f"Saved to {png_path}")
    plt.close(fig)

    npy_path = base.with_suffix(".npy")
    np.save(str(npy_path), {
        "t_sec": t_sec,
        "pr_total": pr_total,
        "gv_total": gv_total,
    }, allow_pickle=True)
    print(f"Saved to {npy_path}")


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
    parser.add_argument(
        "--bandwidth", action="store_true",
        help="Plot cumulative bandwidth usage (BOW / VLC / CBS) over time"
    )
    parser.add_argument(
        "--loops", action="store_true",
        help="Plot PR and GV loop counts over time (summed across all robots)"
    )
    parser.add_argument(
        "--save", type=Path, default=None, metavar="FILE",
        help="Save plot to FILE instead of opening an interactive window"
    )
    args = parser.parse_args()

    rrd_file: Path = args.rrd_file.resolve()
    if not rrd_file.exists():
        print(f"Error: {rrd_file} does not exist.")
        raise SystemExit(1)

    if args.landmarks:
        visualize_landmarks(rrd_file)
    elif args.bandwidth:
        plot_bandwidth(rrd_file, output=args.save)
    elif args.loops:
        plot_loops(rrd_file, output=args.save)
    else:
        print_schema(rrd_file)


if __name__ == "__main__":
    main()
