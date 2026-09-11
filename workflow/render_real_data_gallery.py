#!/usr/bin/env python3
"""Render DREEMS real-data maps beside published FEEMS/EEMS figures."""

# ruff: noqa: F722, F821, UP037

from __future__ import annotations

import argparse
import html
import io
import json
import math
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import matplotlib
import numpy as np
import requests
from beartype import beartype
from jaxtyping import Bool, Float, Int, Real, Shaped, UInt8, jaxtyped
from matplotlib import colors as mcolors
from matplotlib import tri as mtri
from matplotlib.collections import LineCollection
from PIL import Image
from scipy.spatial import cKDTree

from dreems.utils.validation import _integer_array, _real_array, validate_graph_arrays
from dreems.utils.workflow_io import (
    _atomic_output,
    load_config,
    sha256,
    write_csv,
    write_json_finite,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SIGNED_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "dreems_signed_red_transparent_blue",
    [
        "#d95f02",
        (1.0, 1.0, 1.0, 0.0),
        "#2c7bb6",
    ],
    N=257,
)
SD_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "dreems_sd_magma",
    ["#0b1026", "#5b1a73", "#d64f1f", "#ffd166"],
)
WEB_MERCATOR_LIMIT = 20_037_508.342789244
WEB_MERCATOR_CRS = ccrs.epsg(3857)
GEOGRAPHIC_CRS = ccrs.PlateCarree()


@dataclass(frozen=True)
class TopoBasemap:
    """A tile mosaic in Web Mercator, or the explicit stock-image fallback."""

    image: UInt8[np.ndarray, "height width 3"] | None
    image_extent: tuple[float, float, float, float] | None
    source: str


@beartype
def parse_args() -> argparse.Namespace:
    """Separate rendering one dataset from assembling the completed gallery."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    case = commands.add_parser("case", help="Render one current DREEMS result.")
    case.add_argument("--prepared-npz", type=Path, required=True)
    case.add_argument("--prepared-json", type=Path, required=True)
    case.add_argument("--dreems-npz", type=Path, required=True)
    case.add_argument("--dreems-json", type=Path, required=True)
    case.add_argument("--topo-cache", type=Path, required=True)
    case.add_argument("--grid-side", type=int, default=420)
    case.add_argument("--topo-zoom", type=int, default=9)
    case.add_argument("--topo-max-tiles", type=int, default=600)
    case.add_argument("--topo-max-pixels", type=int, default=3200)
    case.add_argument("--dpi", type=int, default=360)
    case.add_argument("--output-png", type=Path, required=True)
    case.add_argument("--output-pdf", type=Path, required=True)
    case.add_argument("--output-json", type=Path, required=True)

    index = commands.add_parser("index", help="Assemble the published comparison gallery.")
    index.add_argument("--config", type=Path, required=True)
    index.add_argument("--row-json", type=Path, nargs="+", required=True)
    index.add_argument("--output-dir", type=Path, required=True)
    index.add_argument("--output-index", type=Path, required=True)
    index.add_argument("--output-manifest-csv", type=Path, required=True)
    index.add_argument("--output-manifest-json", type=Path, required=True)
    index.add_argument("--output-overview", type=Path, required=True)
    index.add_argument("--literature-report", type=Path, required=True)
    index.add_argument("--output-literature-report", type=Path, required=True)
    return parser.parse_args()


@jaxtyped(typechecker=beartype)
def centered(values: Real[np.ndarray, "*axes"]) -> Float[np.ndarray, "*axes"]:
    """Center finite values while preserving the missing-value mask."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    if np.any(finite):
        result[finite] = values[finite] - float(np.mean(values[finite]))
    return result


@jaxtyped(typechecker=beartype)
def edge_midpoints(
    positions: Real[np.ndarray, "nodes 2"], edges: Int[np.ndarray, "edges 2"]
) -> Float[np.ndarray, "edges 2"]:
    """Locate each edge in the same coordinate system as the graph."""
    return 0.5 * (positions[edges[:, 0]] + positions[edges[:, 1]])


@jaxtyped(typechecker=beartype)
def median_edge_length(
    positions: Real[np.ndarray, "nodes 2"], edges: Int[np.ndarray, "edges 2"]
) -> float:
    """Set the interpolation length scale from nonzero geographic edge lengths."""
    lengths = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    lengths = lengths[np.isfinite(lengths) & (lengths > 0.0)]
    return float(np.median(lengths)) if lengths.size else 1.0


@jaxtyped(typechecker=beartype)
def graph_triangles(
    node_count: int, edges: Shaped[np.ndarray, "*edge_axes"]
) -> Int[np.ndarray, "triangles 3"]:
    """List each three-edge cycle once, in the graph's existing node order."""
    edges = _integer_array(np.asarray(edges), name="edges", dtype=np.int32)
    if node_count < 1 or edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("triangle search requires nodes and a two-column edge array")
    if np.any(edges < 0) or np.any(edges >= node_count):
        raise ValueError("triangle edges must refer to graph nodes")
    adjacency = [set() for _ in range(node_count)]
    for source, target in np.asarray(edges, dtype=np.int64):
        source = int(source)
        target = int(target)
        if source != target:
            adjacency[source].add(target)
            adjacency[target].add(source)
    triangles = [
        (source, target, third)
        for source in range(node_count)
        for target in adjacency[source]
        if target > source
        for third in adjacency[source].intersection(adjacency[target])
        if third > target
    ]
    return np.asarray(triangles, dtype=np.int32).reshape((-1, 3))


@jaxtyped(typechecker=beartype)
def graph_face_triangles(
    positions: Real[np.ndarray, "nodes 2"], edges: Int[np.ndarray, "edges 2"]
) -> Int[np.ndarray, "triangles 3"]:
    """Exclude degenerate cycles and cycles enclosing another graph node."""
    triangles = graph_triangles(positions.shape[0], edges)
    faces: list[Int[np.ndarray, "3"]] = []
    for triangle in triangles:
        vertices = positions[triangle]
        first = vertices[1] - vertices[0]
        second = vertices[2] - vertices[0]
        determinant = first[0] * second[1] - first[1] * second[0]
        if abs(float(determinant)) <= 1.0e-12:
            continue
        lower = np.min(vertices, axis=0)
        upper = np.max(vertices, axis=0)
        candidates = np.flatnonzero(np.all((positions >= lower) & (positions <= upper), axis=1))
        candidates = candidates[~np.isin(candidates, triangle)]
        relative = positions[candidates] - vertices[0]
        first_coordinate = (relative[:, 0] * second[1] - relative[:, 1] * second[0]) / determinant
        second_coordinate = (first[0] * relative[:, 1] - first[1] * relative[:, 0]) / determinant
        contains_node = np.any(
            (first_coordinate > 1.0e-9)
            & (second_coordinate > 1.0e-9)
            & ((first_coordinate + second_coordinate) < 1.0 - 1.0e-9)
        )
        if not contains_node:
            faces.append(triangle)
    return np.asarray(faces, dtype=np.int32).reshape((-1, 3))


@jaxtyped(typechecker=beartype)
def triangle_union_support_mask(
    positions: Real[np.ndarray, "nodes 2"],
    triangles: Int[np.ndarray, "triangles 3"],
    query_points: Real[np.ndarray, "points 2"],
) -> Bool[np.ndarray, "points"]:
    """Find points inside any graph triangle when Matplotlib cannot index the mesh."""
    support = np.zeros(query_points.shape[0], dtype=bool)
    query_tree = cKDTree(query_points)
    for triangle in triangles:
        vertices = positions[triangle]
        center = np.mean(vertices, axis=0)
        radius = float(np.max(np.linalg.norm(vertices - center, axis=1)))
        candidates = np.asarray(
            query_tree.query_ball_point(center, radius * (1.0 + 1.0e-10)),
            dtype=np.int64,
        )
        if not candidates.size:
            continue
        first = vertices[1] - vertices[0]
        second = vertices[2] - vertices[0]
        determinant = first[0] * second[1] - first[1] * second[0]
        if abs(float(determinant)) <= 1.0e-12:
            continue
        relative = query_points[candidates] - vertices[0]
        first_coordinate = (relative[:, 0] * second[1] - relative[:, 1] * second[0]) / determinant
        second_coordinate = (first[0] * relative[:, 1] - first[1] * relative[:, 0]) / determinant
        inside = (
            (first_coordinate >= -1.0e-9)
            & (second_coordinate >= -1.0e-9)
            & ((first_coordinate + second_coordinate) <= 1.0 + 1.0e-9)
        )
        support[candidates[inside]] = True
    return support


@jaxtyped(typechecker=beartype)
def graph_support_mask(
    positions: Real[np.ndarray, "nodes 2"],
    edges: Int[np.ndarray, "edges 2"],
    query_points: Real[np.ndarray, "points 2"],
) -> Bool[np.ndarray, "points"]:
    """Restrict a smoothed map to graph faces, or near edges if there are no faces."""
    triangles = graph_face_triangles(positions, edges)
    if triangles.size:
        triangulation = mtri.Triangulation(
            positions[:, 0],
            positions[:, 1],
            triangles=triangles,
        )
        try:
            return np.asarray(
                triangulation.get_trifinder()(
                    query_points[:, 0],
                    query_points[:, 1],
                )
                >= 0,
                dtype=bool,
            )
        except RuntimeError:
            return triangle_union_support_mask(
                positions,
                triangles,
                query_points,
            )

    midpoint_tree = cKDTree(edge_midpoints(positions, edges))
    distances, _ = midpoint_tree.query(query_points, k=1)
    return distances <= 0.75 * median_edge_length(positions, edges)


@jaxtyped(typechecker=beartype)
def interpolate_edges(
    positions: Real[np.ndarray, "nodes 2"],
    edges: Int[np.ndarray, "edges 2"],
    values: Real[np.ndarray, "edges"],
    *,
    grid_side: int,
    kernel_bandwidth_edges: float = 1.8,
    neighbors: int = 32,
    mask_radius_edges: float = 4.0,
) -> tuple[Float[np.ndarray, "grid_side grid_side"], tuple[float, float, float, float]]:
    """Smooth edge values for display without inferring migration outside the graph."""
    if isinstance(grid_side, bool) or grid_side < 2:
        raise ValueError("grid_side must be at least 2")
    if isinstance(neighbors, bool) or neighbors < 1:
        raise ValueError("neighbors must be positive")
    scales = np.asarray([kernel_bandwidth_edges, mask_radius_edges])
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("interpolation length scales must be positive and finite")
    values = np.asarray(values, dtype=np.float64)
    midpoints = edge_midpoints(positions, edges)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.full((grid_side, grid_side), np.nan), (0.0, 1.0, 0.0, 1.0)
    xy = midpoints[finite]
    z = values[finite]
    xmin, ymin = np.nanmin(positions, axis=0)
    xmax, ymax = np.nanmax(positions, axis=0)
    pad = 0.05 * max(xmax - xmin, ymax - ymin, 1.0e-6)
    xmin -= pad
    xmax += pad
    ymin -= pad
    ymax += pad
    xs = np.linspace(xmin, xmax, grid_side)
    ys = np.linspace(ymin, ymax, grid_side)
    xx, yy = np.meshgrid(xs, ys)
    query_points = np.column_stack([xx.ravel(), yy.ravel()])
    support = graph_support_mask(positions, edges, query_points)
    tree = cKDTree(xy)
    k = min(max(int(neighbors), 1), xy.shape[0])
    distances, indices = tree.query(query_points, k=k)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    edge_length = median_edge_length(positions, edges)
    bandwidth = max(float(kernel_bandwidth_edges) * edge_length, 1.0e-9)
    max_distance = max(float(mask_radius_edges) * edge_length, bandwidth)
    weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
    weights[distances > max_distance] = 0.0
    denominator = np.sum(weights, axis=1)
    interpolated = np.full(denominator.shape, np.nan, dtype=np.float64)
    valid = (denominator > 1.0e-12) & support
    interpolated[valid] = np.sum(weights[valid] * z[indices[valid]], axis=1) / denominator[valid]
    return interpolated.reshape(xx.shape), (xmin, xmax, ymin, ymax)


@jaxtyped(typechecker=beartype)
def map_extent(
    sample_lonlat: Real[np.ndarray, "samples 2"],
    node_lonlat: Real[np.ndarray, "nodes 2"],
) -> tuple[float, float, float, float]:
    """Pad the sample and graph bounds within the supported latitude range."""
    coordinates = np.vstack((sample_lonlat, node_lonlat))
    lon_min, lat_min = np.nanmin(coordinates, axis=0)
    lon_max, lat_max = np.nanmax(coordinates, axis=0)
    width = max(float(lon_max - lon_min), 1.0e-4)
    height = max(float(lat_max - lat_min), 1.0e-4)
    pad_lon = max(0.15, 0.09 * width)
    pad_lat = max(0.15, 0.09 * height)
    return (
        float(max(lon_min - pad_lon, -179.9)),
        float(min(lon_max + pad_lon, 179.9)),
        float(max(lat_min - pad_lat, -84.0)),
        float(min(lat_max + pad_lat, 84.0)),
    )


@jaxtyped(typechecker=beartype)
def projected_extent(
    positions: Real[np.ndarray, "nodes 2"],
) -> tuple[float, float, float, float]:
    """Pad a graph already expressed in projected coordinates."""
    xmin, ymin = np.nanmin(positions, axis=0)
    xmax, ymax = np.nanmax(positions, axis=0)
    width = max(float(xmax - xmin), 1.0)
    height = max(float(ymax - ymin), 1.0)
    return (
        float(xmin - 0.025 * width),
        float(xmax + 0.025 * width),
        float(ymin - 0.025 * height),
        float(ymax + 0.025 * height),
    )


@beartype
def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    """Map longitude and latitude to a Web Mercator tile, clipping polar latitudes."""
    if isinstance(zoom, bool) or zoom < 0 or not np.all(np.isfinite([lon, lat])):
        raise ValueError("tile coordinates must be finite and zoom nonnegative")
    lat = float(np.clip(lat, -85.05112878, 85.05112878))
    tile_count = 2 ** int(zoom)
    x = int((lon + 180.0) / 360.0 * tile_count)
    lat_rad = math.radians(lat)
    y = int(
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * tile_count
    )
    return (
        int(np.clip(x, 0, tile_count - 1)),
        int(np.clip(y, 0, tile_count - 1)),
    )


@beartype
def tile_mercator_extent(
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    zoom: int,
) -> tuple[float, float, float, float]:
    """Return the outer bounds of an inclusive tile rectangle, in meters."""
    tile_size = 2.0 * WEB_MERCATOR_LIMIT / (2 ** int(zoom))
    xmin = -WEB_MERCATOR_LIMIT + x_min * tile_size
    xmax = -WEB_MERCATOR_LIMIT + (x_max + 1) * tile_size
    ymax = WEB_MERCATOR_LIMIT - y_min * tile_size
    ymin = WEB_MERCATOR_LIMIT - (y_max + 1) * tile_size
    return xmin, xmax, ymin, ymax


@beartype
def topo_tile_bounds(
    extent: tuple[float, float, float, float],
    zoom: int,
) -> tuple[int, int, int, int]:
    """Find the inclusive tile rectangle covering a geographic extent."""
    lon_min, lon_max, lat_min, lat_max = extent
    x0, y0 = lonlat_to_tile(lon_min, lat_max, zoom)
    x1, y1 = lonlat_to_tile(lon_max, lat_min, zoom)
    x_min, x_max = sorted((x0, x1))
    y_min, y_max = sorted((y0, y1))
    return x_min, x_max, y_min, y_max


@beartype
def topo_tile_count(
    extent: tuple[float, float, float, float],
    zoom: int,
) -> int:
    """Count tiles before allocating or downloading a mosaic."""
    x_min, x_max, y_min, y_max = topo_tile_bounds(extent, zoom)
    return (x_max - x_min + 1) * (y_max - y_min + 1)


@beartype
def select_topo_zoom(
    extent: tuple[float, float, float, float],
    *,
    maximum_zoom: int,
    maximum_tiles: int,
) -> int:
    """Choose the finest requested zoom that fits the tile budget."""
    if maximum_zoom < 0:
        raise ValueError("maximum_zoom must be nonnegative")
    if maximum_tiles < 1:
        raise ValueError("maximum_tiles must be positive")
    for zoom in range(maximum_zoom, -1, -1):
        if topo_tile_count(extent, zoom) <= maximum_tiles:
            return zoom
    raise ValueError("extent cannot be represented within the tile limit")


@beartype
def fetch_esri_topo_tile(
    *,
    cache_dir: Path,
    zoom: int,
    tile_x: int,
    tile_y: int,
    headers: dict[str, str],
) -> Path:
    """Reuse a cached tile or atomically save a verified image over HTTPS."""
    cached = cache_dir / f"z{zoom}" / str(tile_x) / f"{tile_y}.jpg"
    cached.parent.mkdir(parents=True, exist_ok=True)
    if cached.exists() and cached.stat().st_size > 0:
        return cached
    url = (
        "https://services.arcgisonline.com/ArcGIS/rest/services/"
        f"World_Topo_Map/MapServer/tile/{zoom}/{tile_y}/{tile_x}"
    )
    response = requests.get(url, timeout=20, headers=headers)
    response.raise_for_status()
    with Image.open(io.BytesIO(response.content)) as tile:
        tile.verify()
    with _atomic_output(cached) as stream:
        stream.write(response.content)
    return cached


@beartype
def limit_topo_mosaic_size(
    mosaic: Image.Image,
    *,
    maximum_pixels: int,
) -> Image.Image:
    """Bound the longest image side while retaining aspect ratio and fine detail."""
    if maximum_pixels < 256:
        raise ValueError("maximum_pixels must be at least one tile width")
    width, height = mosaic.size
    longest_side = max(width, height)
    if longest_side <= maximum_pixels:
        return mosaic
    scale = maximum_pixels / longest_side
    size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    return mosaic.resize(size, resample=Image.Resampling.LANCZOS)


@jaxtyped(typechecker=beartype)
def fetch_esri_topo_mosaic(
    extent: tuple[float, float, float, float],
    *,
    maximum_zoom: int,
    maximum_tiles: int,
    maximum_pixels: int,
    cache_dir: Path,
) -> tuple[UInt8[np.ndarray, "height width 3"], tuple[float, float, float, float], int]:
    """Fetch tiles concurrently, then assemble them in deterministic geographic order."""
    if maximum_pixels < 256:
        raise ValueError("maximum_pixels must be at least one tile width")
    zoom = select_topo_zoom(
        extent,
        maximum_zoom=maximum_zoom,
        maximum_tiles=maximum_tiles,
    )
    x_min, x_max, y_min, y_max = topo_tile_bounds(extent, zoom)
    tile_count = (x_max - x_min + 1) * (y_max - y_min + 1)
    if tile_count > maximum_tiles:
        raise ValueError(f"refusing to fetch {tile_count} topo tiles at zoom {zoom}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    mosaic = Image.new(
        "RGB",
        ((x_max - x_min + 1) * 256, (y_max - y_min + 1) * 256),
    )
    headers = {"User-Agent": "dreems-gallery/0.1"}
    tile_coordinates = [
        (tile_x, tile_y) for tile_x in range(x_min, x_max + 1) for tile_y in range(y_min, y_max + 1)
    ]

    def load_tile(coordinates: tuple[int, int]) -> Path:
        tile_x, tile_y = coordinates
        return fetch_esri_topo_tile(
            cache_dir=cache_dir,
            zoom=zoom,
            tile_x=tile_x,
            tile_y=tile_y,
            headers=headers,
        )

    with ThreadPoolExecutor(max_workers=min(8, tile_count)) as executor:
        cached_tiles = list(executor.map(load_tile, tile_coordinates))
    for (tile_x, tile_y), cached in zip(
        tile_coordinates,
        cached_tiles,
        strict=True,
    ):
        with Image.open(cached) as tile:
            mosaic.paste(
                tile.convert("RGB"),
                ((tile_x - x_min) * 256, (tile_y - y_min) * 256),
            )
    mosaic = limit_topo_mosaic_size(
        mosaic,
        maximum_pixels=maximum_pixels,
    )
    return (
        np.asarray(mosaic),
        tile_mercator_extent(
            x_min,
            x_max,
            y_min,
            y_max,
            zoom,
        ),
        zoom,
    )


@beartype
def load_topo_basemap(
    extent: tuple[float, float, float, float],
    *,
    cache_dir: Path,
    maximum_zoom: int,
    maximum_tiles: int,
    maximum_pixels: int,
    fetch_topo: bool,
) -> TopoBasemap:
    """Use stock imagery when tiles are unavailable, with a visible warning."""
    if not fetch_topo:
        return TopoBasemap(
            image=None,
            image_extent=None,
            source="Cartopy packaged stock image fallback",
        )
    try:
        image, image_extent, zoom = fetch_esri_topo_mosaic(
            extent,
            maximum_zoom=maximum_zoom,
            maximum_tiles=maximum_tiles,
            maximum_pixels=maximum_pixels,
            cache_dir=cache_dir,
        )
        return TopoBasemap(
            image=image,
            image_extent=image_extent,
            source=f"Esri World Topographic Map tiles, z={zoom}",
        )
    except (requests.RequestException, OSError) as error:
        warnings.warn(f"Topographic tiles unavailable; using stock imagery: {error}", stacklevel=2)
        return TopoBasemap(
            image=None,
            image_extent=None,
            source="Cartopy packaged stock image fallback",
        )


@beartype
def add_topo_basemap(
    ax: plt.Axes,
    extent: tuple[float, float, float, float],
    *,
    basemap: TopoBasemap,
    data_crs: ccrs.CRS,
    fetch_topo: bool,
) -> None:
    """Draw the chosen background and graticule in the graph's coordinate system."""
    if fetch_topo:
        ax.set_extent(extent, crs=data_crs)
        if basemap.image is not None and basemap.image_extent is not None:
            ax.imshow(
                basemap.image,
                origin="upper",
                extent=basemap.image_extent,
                transform=WEB_MERCATOR_CRS,
                interpolation="lanczos",
                zorder=0,
            )
        else:
            ax.stock_img()
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.stock_img()
    ax.gridlines(
        crs=GEOGRAPHIC_CRS,
        draw_labels=False,
        linewidth=0.25,
        color="#6b7280",
        alpha=0.35,
        linestyle="-",
    )


@jaxtyped(typechecker=beartype)
def sample_counts(
    sample_nodes: Shaped[np.ndarray, "*sample_axes"], node_count: int
) -> Float[np.ndarray, "node_count"]:
    """Count samples assigned to each node without changing graph order."""
    sample_nodes = _integer_array(np.asarray(sample_nodes), name="sample_nodes", dtype=np.int32)
    if (
        sample_nodes.ndim != 1
        or node_count < 1
        or np.any(sample_nodes < 0)
        or np.any(sample_nodes >= node_count)
    ):
        raise ValueError("sample_nodes must refer to graph nodes")
    return np.bincount(sample_nodes, minlength=node_count).astype(np.float64)


@jaxtyped(typechecker=beartype)
def add_samples(
    ax: plt.Axes,
    *,
    node_positions: Real[np.ndarray, "nodes 2"],
    edges: Int[np.ndarray, "edges 2"],
    sample_nodes: Int[np.ndarray, "samples"],
    graph_alpha: float,
    data_crs: ccrs.CRS,
) -> None:
    """Overlay the graph and sample counts; marker sizes use a compressed scale."""
    ax.add_collection(
        LineCollection(
            node_positions[edges],
            colors="#151515",
            linewidths=0.07,
            alpha=graph_alpha,
            transform=data_crs,
            zorder=3,
        )
    )
    counts = sample_counts(sample_nodes, node_positions.shape[0])
    sampled = np.flatnonzero(counts > 0.0)
    if not sampled.size:
        return
    maximum = max(float(np.max(counts[sampled])), 1.0)
    sizes = 10.0 + 34.0 * np.sqrt(counts[sampled] / maximum)
    ax.scatter(
        node_positions[sampled, 0],
        node_positions[sampled, 1],
        s=sizes,
        facecolor="#ffffff",
        edgecolor="#111111",
        linewidth=0.45,
        alpha=0.94,
        transform=data_crs,
        zorder=5,
    )


@jaxtyped(typechecker=beartype)
def robust_limit(values: Real[np.ndarray, "*axes"], percentile: float = 96.0) -> float:
    """Set a nonzero color limit from the finite absolute values."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return max(float(np.percentile(np.abs(finite), percentile)), 1.0e-6) if finite.size else 1.0


@jaxtyped(typechecker=beartype)
def plot_raw_edges(
    ax: plt.Axes,
    *,
    node_positions: Real[np.ndarray, "nodes 2"],
    edges: Int[np.ndarray, "edges 2"],
    sample_nodes: Int[np.ndarray, "samples"],
    values: Real[np.ndarray, "edges"],
    extent: tuple[float, float, float, float],
    vlim: float,
    basemap: TopoBasemap,
    data_crs: ccrs.CRS,
    fetch_topo: bool,
) -> str:
    """Draw MAP edge estimates with the same color scale as the contour panel."""
    add_topo_basemap(
        ax,
        extent,
        basemap=basemap,
        data_crs=data_crs,
        fetch_topo=fetch_topo,
    )
    ax.add_collection(
        LineCollection(
            node_positions[edges],
            array=np.asarray(values, dtype=np.float64),
            cmap=SIGNED_CMAP,
            norm=plt.Normalize(-vlim, vlim),
            linewidths=0.62,
            alpha=0.88,
            transform=data_crs,
            zorder=4,
        )
    )
    add_samples(
        ax,
        node_positions=node_positions,
        edges=edges,
        sample_nodes=sample_nodes,
        graph_alpha=0.04,
        data_crs=data_crs,
    )
    ax.set_title("DREEMS MAP-refined raw edges", fontsize=9.0)
    return basemap.source


@jaxtyped(typechecker=beartype)
def plot_interpolated(
    ax: plt.Axes,
    *,
    node_positions: Real[np.ndarray, "nodes 2"],
    edges: Int[np.ndarray, "edges 2"],
    sample_nodes: Int[np.ndarray, "samples"],
    values: Real[np.ndarray, "edges"],
    extent: tuple[float, float, float, float],
    grid_side: int,
    vmin: float,
    vmax: float,
    cmap: Any,
    basemap: TopoBasemap,
    title: str,
    contours: bool,
    data_crs: ccrs.CRS,
    fetch_topo: bool,
) -> str:
    """Draw a heatmap or contours; interpolation is only a display operation."""
    add_topo_basemap(
        ax,
        extent,
        basemap=basemap,
        data_crs=data_crs,
        fetch_topo=fetch_topo,
    )
    image, image_extent = interpolate_edges(
        node_positions,
        edges,
        values,
        grid_side=grid_side,
    )
    masked = np.ma.masked_invalid(image)
    if contours:
        xs = np.linspace(image_extent[0], image_extent[1], image.shape[1])
        ys = np.linspace(image_extent[2], image_extent[3], image.shape[0])
        ax.contourf(
            xs,
            ys,
            masked,
            levels=np.linspace(vmin, vmax, 13),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=0.22,
            extend="both",
            transform=data_crs,
            zorder=2,
        )
        line_levels = [level for level in np.linspace(vmin, vmax, 9) if abs(level) > 1.0e-8]
        ax.contour(
            xs,
            ys,
            masked,
            levels=line_levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=1.1,
            alpha=0.92,
            transform=data_crs,
            zorder=4,
        )
    else:
        ax.imshow(
            masked,
            origin="lower",
            extent=image_extent,
            transform=data_crs,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
            alpha=0.52,
            zorder=2,
        )
    add_samples(
        ax,
        node_positions=node_positions,
        edges=edges,
        sample_nodes=sample_nodes,
        graph_alpha=0.05,
        data_crs=data_crs,
    )
    ax.set_title(title, fontsize=9.0)
    return basemap.source


@beartype
def render_case(args: argparse.Namespace) -> None:
    """Render matched inputs and estimates without refitting either method."""
    metadata = json.loads(args.prepared_json.read_text())
    dreems_metadata = json.loads(args.dreems_json.read_text())
    if metadata["dataset_id"] != dreems_metadata["dataset_id"]:
        raise ValueError("DREEMS results and prepared input describe different datasets")
    for field, path in (
        ("prepared_npz_sha256", args.prepared_npz),
        ("prepared_json_sha256", args.prepared_json),
    ):
        if field in dreems_metadata and dreems_metadata[field] != sha256(path):
            raise ValueError(f"DREEMS results do not match {path}")
    with np.load(args.prepared_npz, allow_pickle=False) as data:
        sample_lonlat = _real_array(data["sample_lonlat"], name="sample_lonlat")
        node_lonlat = _real_array(data["node_lonlat"], name="node_lonlat")
        node_xy = _real_array(data["node_xy"], name="node_xy")
        edges = _integer_array(data["edges"], name="edges", dtype=np.int32)
        sample_nodes = _integer_array(data["sample_nodes"], name="sample_nodes", dtype=np.int32)
    validate_graph_arrays(positions=node_xy, edges=edges)
    if node_lonlat.shape != node_xy.shape or sample_lonlat.shape != (sample_nodes.size, 2):
        raise ValueError("geographic coordinates must align with nodes and samples")
    sample_counts(sample_nodes, node_xy.shape[0])
    with np.load(args.dreems_npz, allow_pickle=False) as data:
        prior_edge_log = centered(_real_array(data["prior_edge_log"], name="prior_edge_log"))
        refined_edge_log = centered(_real_array(data["edge_log"], name="edge_log"))
        edge_log_sd = _real_array(data["edge_log_sd"], name="edge_log_sd")
    if any(
        values.shape != (edges.shape[0],)
        for values in (prior_edge_log, refined_edge_log, edge_log_sd)
    ):
        raise ValueError("DREEMS edge output does not align with the prepared graph")
    if np.any(edge_log_sd < 0):
        raise ValueError("posterior SD must be nonnegative")

    is_global_mollweide = str(metadata.get("graph_builder", "")).startswith(
        "reconstructed_global_mollweide"
    )
    if is_global_mollweide:
        central_longitude = float(metadata["central_longitude"])
        map_projection = ccrs.Mollweide(central_longitude=central_longitude)
        data_crs = map_projection
        node_positions = 1000.0 * node_xy
        extent = projected_extent(node_positions)
        fetch_topo = False
    else:
        map_projection = GEOGRAPHIC_CRS
        data_crs = GEOGRAPHIC_CRS
        node_positions = node_lonlat
        extent = map_extent(sample_lonlat, node_lonlat)
        fetch_topo = True
    basemap = load_topo_basemap(
        extent,
        cache_dir=args.topo_cache,
        maximum_zoom=int(args.topo_zoom),
        maximum_tiles=int(args.topo_max_tiles),
        maximum_pixels=int(args.topo_max_pixels),
        fetch_topo=fetch_topo,
    )
    prior_vlim = robust_limit(prior_edge_log)
    map_vlim = robust_limit(refined_edge_log)
    sd_vmax = robust_limit(edge_log_sd)
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(22.0, 5.3),
        subplot_kw={"projection": map_projection},
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.020,
        right=0.990,
        top=0.84,
        bottom=0.18,
        wspace=0.035,
    )
    topo_source = plot_raw_edges(
        axes[0],
        node_positions=node_positions,
        edges=edges,
        sample_nodes=sample_nodes,
        values=refined_edge_log,
        extent=extent,
        vlim=map_vlim,
        basemap=basemap,
        data_crs=data_crs,
        fetch_topo=fetch_topo,
    )
    plot_interpolated(
        axes[1],
        node_positions=node_positions,
        edges=edges,
        sample_nodes=sample_nodes,
        values=prior_edge_log,
        extent=extent,
        grid_side=int(args.grid_side),
        vmin=-prior_vlim,
        vmax=prior_vlim,
        cmap=SIGNED_CMAP,
        basemap=basemap,
        title="DREEMS neural-prior heatmap",
        contours=False,
        data_crs=data_crs,
        fetch_topo=fetch_topo,
    )
    plot_interpolated(
        axes[2],
        node_positions=node_positions,
        edges=edges,
        sample_nodes=sample_nodes,
        values=refined_edge_log,
        extent=extent,
        grid_side=int(args.grid_side),
        vmin=-map_vlim,
        vmax=map_vlim,
        cmap=SIGNED_CMAP,
        basemap=basemap,
        title="DREEMS MAP-refined contours",
        contours=True,
        data_crs=data_crs,
        fetch_topo=fetch_topo,
    )
    plot_interpolated(
        axes[3],
        node_positions=node_positions,
        edges=edges,
        sample_nodes=sample_nodes,
        values=edge_log_sd,
        extent=extent,
        grid_side=int(args.grid_side),
        vmin=0.0,
        vmax=sd_vmax,
        cmap=SD_CMAP,
        basemap=basemap,
        title="DREEMS posterior SD heatmap",
        contours=False,
        data_crs=data_crs,
        fetch_topo=fetch_topo,
    )

    colorbar_specs = (
        (SIGNED_CMAP, -map_vlim, map_vlim, "MAP edge-log migration"),
        (SIGNED_CMAP, -prior_vlim, prior_vlim, "neural-prior edge-log migration"),
        (SIGNED_CMAP, -map_vlim, map_vlim, "MAP contour edge-log migration"),
        (SD_CMAP, 0.0, sd_vmax, "posterior SD"),
    )
    for axis, (cmap, vmin, vmax, label) in zip(axes, colorbar_specs, strict=True):
        bounds = axis.get_position()
        colorbar_axis = fig.add_axes(
            [bounds.x0 + 0.035, 0.075, max(bounds.width - 0.07, 0.08), 0.032]
        )
        mappable = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin, vmax),
        )
        mappable.set_array([])
        colorbar = fig.colorbar(
            mappable,
            cax=colorbar_axis,
            orientation="horizontal",
        )
        colorbar.ax.tick_params(labelsize=7.0)
        colorbar.set_label(label, fontsize=8.0)
    fig.suptitle(
        (
            f"{metadata['title']}\n"
            f"{metadata['samples']} samples | {metadata['maf_filtered_snps']} QC SNPs | "
            f"{metadata['observed_nodes']}/{metadata['nodes']} observed nodes | "
            f"{metadata['edges']} edges"
        ),
        fontsize=12.4,
        y=0.96,
    )
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    with _atomic_output(args.output_png) as stream:
        fig.savefig(stream, format="png", dpi=int(args.dpi))
    with _atomic_output(args.output_pdf) as stream:
        fig.savefig(stream, format="pdf")
    plt.close(fig)

    row = {
        **metadata,
        "figure_png": args.output_png.name,
        "figure_pdf": args.output_pdf.name,
        "topo_source": topo_source,
        "dreems_seconds": float(dreems_metadata["prediction_seconds"]),
        "dreems_result_json": str(args.dreems_json),
        "prepared_json": str(args.prepared_json),
        "prior_edge_log_sd": float(np.std(prior_edge_log)),
        "refined_edge_log_sd": float(np.std(refined_edge_log)),
        "prior_edge_log_display_limit": prior_vlim,
        "map_edge_log_display_limit": map_vlim,
        "posterior_sd_median": float(np.median(edge_log_sd)),
        "posterior_sd_p95": float(np.percentile(edge_log_sd, 95.0)),
    }
    write_json_finite(args.output_json, row)


@beartype
def stage_published_figures(
    output_dir: Path,
    datasets: dict[str, Any],
) -> dict[str, list[dict[str, str]]]:
    """Copy the selected original study figures and record their source hashes."""
    staged: dict[str, list[dict[str, str]]] = {}
    published_dir = output_dir / "published"
    published_dir.mkdir(parents=True, exist_ok=True)
    for dataset_id, config in datasets.items():
        figures = []
        for index, figure in enumerate(config.get("published_figures", []), start=1):
            source = Path(str(figure["path"]))
            if not source.is_file():
                raise FileNotFoundError(f"published figure does not exist: {source}")
            suffix = source.suffix.lower() or ".png"
            target = published_dir / f"{dataset_id}_{index:02d}{suffix}"
            shutil.copy2(source, target)
            figures.append(
                {
                    "file": str(target.relative_to(output_dir)),
                    "caption": str(figure["caption"]),
                    "source_url": str(figure.get("url", "")),
                    "source_path": str(source),
                    "sha256": sha256(target),
                }
            )
        staged[dataset_id] = figures
    return staged


@beartype
def write_contact_sheet(output_dir: Path, rows: list[dict[str, Any]], output: Path) -> None:
    """Build the overview from rendered panels, retaining their aspect ratios."""
    width = 1300
    gap = 24
    columns = 2
    images = []
    for row in rows:
        with Image.open(output_dir / str(row["figure_png"])) as source:
            image = source.convert("RGB")
            height = int(round(image.height * width / image.width))
            images.append(image.resize((width, height), Image.Resampling.LANCZOS))
    if not images:
        raise ValueError("cannot create an empty contact sheet")
    height = max(image.height for image in images)
    row_count = math.ceil(len(images) / columns)
    canvas = Image.new(
        "RGB",
        (
            columns * width + (columns + 1) * gap,
            row_count * height + (row_count + 1) * gap,
        ),
        "white",
    )
    for index, image in enumerate(images):
        column = index % columns
        row = index // columns
        canvas.paste(
            image,
            (gap + column * (width + gap), gap + row * (height + gap)),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=94)


@beartype
def assemble_index(args: argparse.Namespace) -> None:
    """Join finished panels, study figures, and provenance into the static gallery."""
    config = load_config(args.config)
    dataset_config = dict(config["datasets"])
    rows = [json.loads(path.read_text()) for path in args.row_json]
    args.output_literature_report.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.literature_report, args.output_literature_report)
    staged = stage_published_figures(args.output_dir, dataset_config)
    for row in rows:
        row["published_figures"] = staged.get(str(row["dataset_id"]), [])
        row["comparison_kind"] = (
            "qualitative_published_figure"
            if row["published_figures"]
            else "dreems_only_no_published_figure_available"
        )

    write_csv(
        args.output_manifest_csv,
        [
            {
                **{key: value for key, value in row.items() if key != "published_figures"},
                "published_figure_count": len(row["published_figures"]),
            }
            for row in rows
        ],
    )
    write_json_finite(
        args.output_manifest_json,
        {
            "comparison_scope": (
                "Qualitative comparison to published FEEMS/EEMS figures. The published "
                "panels may use different graph construction, QC, tuning, and rendering; "
                "no edgewise metric is reported."
            ),
            "literature_report": {
                "file": args.output_literature_report.name,
                "sha256": sha256(args.output_literature_report),
            },
            "datasets": rows,
        },
    )
    write_contact_sheet(args.output_dir, rows, args.output_overview)

    cards = []
    for row in rows:
        source_url = html.escape(str(row["source_url"]))
        source = html.escape(str(row["source"]))
        published = row["published_figures"]
        if published:
            original_panels = []
            for figure in published:
                image = (
                    f'<img loading="lazy" src="{html.escape(figure["file"])}" '
                    f'alt="{html.escape(figure["caption"])}">'
                )
                if figure["source_url"]:
                    image = f'<a href="{html.escape(figure["source_url"])}">{image}</a>'
                original_panels.append(
                    "<figure>"
                    f"{image}"
                    f"<figcaption>{html.escape(figure['caption'])}</figcaption>"
                    "</figure>"
                )
            original_html = "".join(original_panels)
        else:
            original_html = (
                '<div class="missing">No original rendered FEEMS/EEMS surface was '
                "available in the released study files. DREEMS is shown without a "
                "method comparison.</div>"
            )
        qc_provenance = str(row["qc_provenance"]).rstrip().rstrip(".")
        cards.append(
            "<section>"
            f"<h2>{html.escape(str(row['title']))}</h2>"
            f'<p class="meta"><a href="{source_url}">{source}</a>. '
            f"{html.escape(qc_provenance)}. "
            f"{int(row['samples'])} samples, {int(row['maf_filtered_snps'])} SNPs "
            f"after common MAF >= {float(row['min_maf']):.2f}, "
            f"{int(row['observed_nodes'])}/{int(row['nodes'])} observed graph nodes.</p>"
            '<div class="comparison">'
            "<figure>"
            f'<a href="{html.escape(str(row["figure_pdf"]))}">'
            f'<img loading="lazy" src="{html.escape(str(row["figure_png"]))}" '
            f'alt="Current DREEMS reanalysis of {html.escape(str(row["title"]))}"></a>'
            "<figcaption>Current DREEMS reanalysis using the selected checkpoint and "
            "source-QC, MAF-conditioned genotypes.</figcaption>"
            "</figure>"
            f'<div class="originals">{original_html}</div>'
            "</div>"
            "</section>"
        )

    html_text = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DREEMS and Published Real-Data Surfaces</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #151515; }}
main {{ max-width: 2200px; margin: 0 auto; }}
h1 {{ margin-bottom: 0.35rem; }}
h2 {{ margin-top: 2.6rem; margin-bottom: 0.35rem; }}
.note, .meta, figcaption {{ color: #4b5563; line-height: 1.45; }}
.note, .meta {{ max-width: 1200px; }}
.comparison {{ display: grid; grid-template-columns: minmax(600px, 1.45fr) minmax(360px, 0.75fr); gap: 24px; align-items: start; }}
.originals {{ display: grid; gap: 18px; }}
figure {{ margin: 0; }}
img {{ width: 100%; height: auto; border: 1px solid #d1d5db; background: white; }}
figcaption {{ font-size: 13px; margin-top: 6px; }}
.missing {{ padding: 36px; border: 1px dashed #9ca3af; color: #4b5563; background: #f9fafb; }}
section {{ border-top: 1px solid #e5e7eb; padding-top: 12px; margin-bottom: 48px; }}
a {{ color: #075985; }}
@media (max-width: 1050px) {{
  body {{ margin: 12px; }}
  .comparison {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<main>
<h1>DREEMS and Published FEEMS/EEMS Real-Data Surfaces</h1>
<p class="note">This page compares DREEMS estimates with FEEMS or EEMS maps published
for the same datasets. The original studies used their own graphs, filtering, and model
settings, so the figures are intended for visual comparison only.</p>
<p class="note">In the DREEMS panels, red indicates lower relative migration and blue
indicates higher relative migration. Values near the mean are transparent. White dots
mark sampled nodes, and larger dots indicate more samples. The heatmaps and contours
are smoothed views of the underlying edge estimates. The neural-prior and MAP panels
use separate centered color scales, shown beneath each panel.</p>
<p><a href="overview.png">DREEMS overview</a> |
<a href="{html.escape(args.output_literature_report.name)}">literature candidates</a> |
<a href="manifest.csv">manifest.csv</a> |
<a href="manifest.json">manifest.json</a></p>
{"".join(cards)}
</main>
</body>
</html>
"""
    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    with _atomic_output(args.output_index) as stream:
        stream.write(html_text.encode("utf-8"))


@beartype
def main() -> None:
    """Dispatch a single-panel render or gallery assembly."""
    args = parse_args()
    if args.command == "case":
        render_case(args)
    else:
        assemble_index(args)


if __name__ == "__main__":
    main()
