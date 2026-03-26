#!/usr/bin/env python3
"""
Plot ns-as variant trajectories on a satellite image background.

GPS data (lat/lon) from *_gps.csv files is used to georeference the local ENU
coordinate frame, then all trajectories (GT + estimated) are plotted on top of
satellite tiles fetched from Esri World Imagery.

Usage:
    python3 plot_satellite_traj.py <exp_dir> --gps_dir /data/graco
    python3 plot_satellite_traj.py a5678 --gps_dir /data/graco --gt_dir ground_truth/a5678
    python3 plot_satellite_traj.py a5678 --gps_dir /data/graco --zoom 17 --output sat_traj
"""

import argparse
import io
import math
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.io import load_gt_trajectory, load_alignment_from_evo_zip, discover_robots
from utils.plot import IEEE_RC, ROBOT_COLORS, save_fig, apply_alignment


# ---------------------------------------------------------------------------
# Tile helpers
# ---------------------------------------------------------------------------

TILE_SOURCES = {
    'esri':  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    'bing':  None,   # uses quadkey — handled separately
    'google': 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
}
TILE_SIZE = 256


def _xy_to_quadkey(x: int, y: int, zoom: int) -> str:
    quadkey = []
    for i in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if x & mask:
            digit += 1
        if y & mask:
            digit += 2
        quadkey.append(str(digit))
    return ''.join(quadkey)


def _deg2tile(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    lat_r = math.radians(lat_deg)
    n = 2 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n)
    return xtile, ytile


def _tile2deg(xtile: int, ytile: int, zoom: int) -> tuple[float, float]:
    """Return (lat, lon) of the NW corner of a tile."""
    n = 2 ** zoom
    lon = xtile / n * 360.0 - 180.0
    lat_r = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat = math.degrees(lat_r)
    return lat, lon


def fetch_tile(x: int, y: int, z: int, source: str = 'esri') -> np.ndarray | None:
    if source == 'bing':
        qk = _xy_to_quadkey(x, y, z)
        server = (x + y) % 4
        url = f'https://ecn.t{server}.tiles.virtualearth.net/tiles/a{qk}.jpeg?g=1'
        fmt = 'jpg'
    elif source == 'google':
        url = TILE_SOURCES['google'].format(x=x, y=y, z=z)
        fmt = 'jpg'
    else:
        url = TILE_SOURCES['esri'].format(x=x, y=y, z=z)
        fmt = 'jpg'
    try:
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        img = mpimg.imread(io.BytesIO(resp.content), format=fmt)
        return img
    except Exception as e:
        print(f"  Warning: could not fetch tile ({x},{y},{z}) from {source}: {e}")
        return None


def fetch_satellite_background(lat_min: float, lat_max: float,
                                lon_min: float, lon_max: float,
                                zoom: int, source: str = 'esri') -> tuple[np.ndarray, tuple]:
    """
    Fetch and stitch satellite tiles covering the bounding box.
    Returns (image_array, (lon_left, lon_right, lat_bottom, lat_top)).
    """
    x0, y0 = _deg2tile(lat_max, lon_min, zoom)  # NW corner
    x1, y1 = _deg2tile(lat_min, lon_max, zoom)  # SE corner
    # ensure correct ordering
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)

    nx = x1 - x0 + 1
    ny = y1 - y0 + 1
    print(f"  Fetching {nx * ny} tiles (zoom={zoom}, {nx}×{ny}, source={source})...")

    canvas = np.zeros((ny * TILE_SIZE, nx * TILE_SIZE, 3), dtype=np.uint8)
    for ix, tx in enumerate(range(x0, x1 + 1)):
        for iy, ty in enumerate(range(y0, y1 + 1)):
            tile = fetch_tile(tx, ty, zoom, source)
            if tile is not None:
                tile_uint = (tile * 255).astype(np.uint8) if tile.max() <= 1.0 else tile.astype(np.uint8)
                canvas[iy * TILE_SIZE:(iy + 1) * TILE_SIZE,
                       ix * TILE_SIZE:(ix + 1) * TILE_SIZE] = tile_uint[:, :, :3]

    # Geographic extent of the stitched image
    lat_top, lon_left   = _tile2deg(x0,     y0,     zoom)
    lat_bottom, lon_right = _tile2deg(x1 + 1, y1 + 1, zoom)
    return canvas, (lon_left, lon_right, lat_bottom, lat_top)


# ---------------------------------------------------------------------------
# ENU ↔ GPS conversion
# ---------------------------------------------------------------------------

EARTH_R = 6_378_137.0  # WGS-84 equatorial radius in metres


def enu_origin_from_gps_and_gt(gps_ts: np.ndarray, gps_lat: np.ndarray, gps_lon: np.ndarray,
                                gt_ts_s: np.ndarray, gt_pos: np.ndarray,
                                tol_s: float = 0.5) -> tuple[float, float] | None:
    """
    Estimate the GPS origin (lat0, lon0) of the local ENU frame by matching
    GPS and GT timestamps and fitting lat/lon = lat0 + y/R, lon0 + x/(R*cos(lat0)).
    """
    lat0_acc, lon0_acc, n = 0.0, 0.0, 0
    for i, ts in enumerate(gps_ts):
        idx = np.argmin(np.abs(gt_ts_s - ts))
        if np.abs(gt_ts_s[idx] - ts) > tol_s:
            continue
        x, y = gt_pos[idx, 0], gt_pos[idx, 1]
        lat_r = math.radians(gps_lat[i])
        lat0_acc += gps_lat[i] - math.degrees(y / EARTH_R)
        lon0_acc += gps_lon[i] - math.degrees(x / (EARTH_R * math.cos(lat_r)))
        n += 1
    if n == 0:
        return None
    return lat0_acc / n, lon0_acc / n


def enu_to_latlon(positions: np.ndarray, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    lat_r = math.radians(lat0)
    lats = lat0 + np.degrees(positions[:, 1] / EARTH_R)
    lons = lon0 + np.degrees(positions[:, 0] / (EARTH_R * math.cos(lat_r)))
    return lats, lons


# ---------------------------------------------------------------------------
# GPS file loading and robot matching
# ---------------------------------------------------------------------------

def load_gps_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (timestamps_s, latitudes, longitudes)."""
    data = np.loadtxt(path, delimiter=',', skiprows=1, usecols=(0, 1, 2))
    return data[:, 0], data[:, 1], data[:, 2]


def match_gps_to_robots(gps_dir: Path, robot_names: list[str],
                         gt_dir: Path) -> dict[str, Path]:
    """
    Match *_gps.csv files to robot names by comparing start timestamps with GT files.
    Returns {robot_name: gps_csv_path}.
    """
    gps_files = sorted(gps_dir.glob("*_gps.csv"))
    # Build {start_ts: path} for GPS files
    gps_starts = {}
    for f in gps_files:
        try:
            ts = float(open(f).readlines()[1].split(',')[0])
            gps_starts[ts] = f
        except Exception:
            pass

    result = {}
    for rname in robot_names:
        # Find GT file for this robot
        gt_candidates = list(gt_dir.glob(f"{rname}.txt")) + list(gt_dir.glob(f"{rname}.csv"))
        if not gt_candidates:
            continue
        gt_line = open(gt_candidates[0]).readline().strip()
        # skip comment lines
        for line in open(gt_candidates[0]):
            if not line.startswith('#'):
                gt_line = line
                break
        try:
            gt_ts = float(gt_line.split()[0] if '.txt' in str(gt_candidates[0])
                          else gt_line.split(',')[0])
        except Exception:
            continue
        # Find closest GPS start timestamp
        if not gps_starts:
            continue
        best_ts = min(gps_starts, key=lambda t: abs(t - gt_ts))
        if abs(best_ts - gt_ts) < 5.0:  # within 5 s
            result[rname] = gps_starts[best_ts]
            print(f"  Matched {rname} → {gps_starts[best_ts].name} (Δt={best_ts-gt_ts:.2f}s)")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exp_dir')
    ap.add_argument('--gps_dir', required=True,
                    help='Directory containing *_gps.csv files')
    ap.add_argument('--gt_dir', default=None,
                    help='GT directory (default: ground_truth/<exp>)')
    ap.add_argument('--variant', default='ns-as',
                    help='Variant to plot (default: ns-as)')
    ap.add_argument('--zoom', type=int, default=17,
                    help='Tile zoom level (default: 17)')
    ap.add_argument('--pad', type=float, default=0.0003,
                    help='Lat/lon padding around trajectory bounding box')
    ap.add_argument('--source', default='google', choices=['esri', 'bing', 'google'],
                    help='Tile source (default: google)')
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.is_absolute():
        exp_dir = Path(__file__).parent / exp_dir

    if args.gt_dir:
        gt_dir = Path(args.gt_dir)
    else:
        gt_dir = Path(__file__).parent / 'ground_truth' / exp_dir.name
    gps_dir = Path(args.gps_dir)
    variant_dir = exp_dir / args.variant

    # Discover robots
    robot_map = discover_robots(variant_dir)  # {robot_id: dir_name}
    robot_names = list(robot_map.values())
    print(f"Robots: {robot_names}")

    # Match GPS files to robots
    print("Matching GPS files to robots...")
    gps_match = match_gps_to_robots(gps_dir, robot_names, gt_dir)
    if not gps_match:
        print("Error: no GPS files matched. Check --gps_dir and GT timestamps.")
        return

    # Load GT trajectories and estimate ENU origin per robot
    print("Estimating ENU origins from GPS+GT...")
    gt_latlon: dict[str, tuple] = {}  # robot → (lats, lons)
    origins: list[tuple] = []
    for rname in robot_names:
        gt_candidates = list(gt_dir.glob(f"{rname}.txt")) + list(gt_dir.glob(f"{rname}.csv"))
        if not gt_candidates:
            continue
        try:
            ts_ns, pos, _ = load_gt_trajectory(str(gt_candidates[0]))
            gt_ts_s = ts_ns / 1e9
        except Exception as e:
            print(f"  Warning: could not load GT for {rname}: {e}")
            continue

        if rname in gps_match:
            gps_ts, gps_lat, gps_lon = load_gps_file(gps_match[rname])
            origin = enu_origin_from_gps_and_gt(gps_ts, gps_lat, gps_lon, gt_ts_s, pos)
            if origin:
                origins.append(origin)
                print(f"  {rname}: origin = ({origin[0]:.6f}, {origin[1]:.6f})")
                lats, lons = enu_to_latlon(pos, *origin)
                gt_latlon[rname] = (lats, lons)

    if not origins:
        print("Error: could not estimate ENU origin. Check GPS/GT timestamp overlap.")
        return
    # Use mean origin across all robots
    lat0 = float(np.mean([o[0] for o in origins]))
    lon0 = float(np.mean([o[1] for o in origins]))
    print(f"  Mean ENU origin: ({lat0:.6f}, {lon0:.6f})")
    # Recompute all GT with mean origin
    for rname in list(gt_latlon.keys()):
        gt_candidates = list(gt_dir.glob(f"{rname}.txt")) + list(gt_dir.glob(f"{rname}.csv"))
        ts_ns, pos, _ = load_gt_trajectory(str(gt_candidates[0]))
        gt_latlon[rname] = enu_to_latlon(pos, lat0, lon0)

    # Load alignment and estimated trajectories for variant
    alignment = None
    evo_zip = variant_dir / 'evo_ape.zip'
    if evo_zip.exists():
        try:
            alignment = load_alignment_from_evo_zip(str(evo_zip))
            print(f"Loaded alignment from {evo_zip.name}")
        except Exception as e:
            print(f"Warning: could not load alignment: {e}")

    est_latlon: dict[str, tuple] = {}
    for rid, rdir_name in robot_map.items():
        rdir = variant_dir / rdir_name
        # Find TUM file in dpgo/
        tum_files = sorted((rdir / 'dpgo').glob('Robot *.tum')) if (rdir / 'dpgo').is_dir() else []
        if not tum_files:
            continue
        tum = tum_files[-1]
        rows = []
        with open(tum) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                p = line.split()
                if len(p) >= 4:
                    rows.append([float(p[1]), float(p[2]), float(p[3])])
        if not rows:
            continue
        pos = np.array(rows)
        if alignment is not None:
            pos = apply_alignment(pos, *alignment)
        lats, lons = enu_to_latlon(pos, lat0, lon0)
        est_latlon[rdir_name] = (lats, lons)

    # Compute bounding box from all trajectories
    all_lats = np.concatenate([v[0] for v in list(gt_latlon.values()) + list(est_latlon.values())])
    all_lons = np.concatenate([v[1] for v in list(gt_latlon.values()) + list(est_latlon.values())])
    pad = args.pad
    lat_min, lat_max = all_lats.min() - pad, all_lats.max() + pad
    lon_min, lon_max = all_lons.min() - pad, all_lons.max() + pad

    # Fetch satellite tiles
    print("Fetching satellite tiles...")
    bg_img, (img_lon_l, img_lon_r, img_lat_b, img_lat_t) = fetch_satellite_background(
        lat_min, lat_max, lon_min, lon_max, args.zoom, args.source)

    # Darken and reduce contrast of satellite image
    bg_f = bg_img.astype(np.float32) / 255.0
    bg_f = 0.5 + (bg_f - 0.5) * 0.6   # reduce contrast (blend toward mid-gray)
    bg_f = bg_f * 0.8                   # darken
    bg_f = np.clip(bg_f, 0, 1)

    # Plot
    plt.rcParams.update({**IEEE_RC, 'figure.figsize': (3.5, 3.5)})
    fig, ax = plt.subplots()
    ax.imshow(bg_f, extent=[img_lon_l, img_lon_r, img_lat_b, img_lat_t],
              aspect='auto', origin='upper', zorder=0)

    robot_list = sorted(robot_map.values())
    for i, rname in enumerate(robot_list):
        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        # GT dashed — white halo + colored dash for visibility on satellite
        if rname in gt_latlon:
            lats, lons = gt_latlon[rname]
            ax.plot(lons, lats, lw=2.5, color='white', alpha=0.6,
                    linestyle='--', zorder=1)
            ax.plot(lons, lats, lw=1.2, color=color, alpha=0.9,
                    linestyle='--', zorder=2)
        # Estimated solid
        if rname in est_latlon:
            lats, lons = est_latlon[rname]
            ax.plot(lons, lats, lw=0.8, color=color, zorder=3,
                    label=f'Drone {i}')
            ax.plot(lons[-1], lats[-1], 'o', ms=2, color=color, zorder=4)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.axis('off')

    import matplotlib.lines as mlines
    handles, labels = ax.get_legend_handles_labels()
    gt_handle = mlines.Line2D([], [], color='gray', linestyle='--', lw=1.2, label='Ground Truth')
    ax.legend(handles=handles + [gt_handle], labels=labels + ['Ground Truth'],
              fontsize=6, loc='best')

    out = Path(args.output) if args.output else exp_dir / f'satellite_traj_{args.variant}'
    save_fig(fig, out)
    plt.close(fig)
    print(f"Saved → {out}.pdf / .png")


if __name__ == '__main__':
    main()
