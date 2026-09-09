"""Geographic graph construction and migration-surface generators."""

# ruff: noqa: F722, F821, UP037

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from beartype import beartype
from jaxtyping import Bool, Float, Int, Real, jaxtyped
from scipy.ndimage import label
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from dreems.simulation_config import (
    _integer_setting,
    choose_weighted_name,
    draw_weighted_positive_float,
    draw_weighted_positive_int,
)
from dreems.validation import validate_graph_arrays


@beartype
def largest_component_mask(mask: np.ndarray) -> Bool[np.ndarray, "height width"]:  # noqa: F722
    """Keep the largest four-neighbor component, preserving the old tie-break."""
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    labels, count = label(mask)
    if count == 0:
        raise ValueError("mask has no active nodes")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    winners = np.flatnonzero(sizes == sizes.max())
    winner = winners[0]
    if winners.size > 1:
        # The old traversal visited components in this integer set's order.
        # Only ties need that convention; labeling handles the traversal itself.
        winner = next(
            labels.ravel()[flat]
            for flat in set(np.flatnonzero(mask).tolist())
            if sizes[labels.ravel()[flat]] == sizes.max()
        )
    return labels == winner


@beartype
def random_mask(
    rng: np.random.Generator,
    height: int,
    width: int,
    kind: str,
) -> np.ndarray:
    """Draw a connected habitat mask on a rectangular four-neighbor grid."""
    yy, xx = np.mgrid[0:height, 0:width]
    mask = np.ones((height, width), dtype=bool)
    if kind == "full":
        return mask
    if kind == "bay":
        center = rng.uniform(0.35, 0.65) * (height - 1)
        width_y = rng.uniform(0.14, 0.26) * height
        start_x = rng.uniform(0.48, 0.68) * width
        mask[(xx >= start_x) & (np.abs(yy - center) < width_y)] = False
    elif kind == "lake":
        cy = rng.uniform(0.35, 0.65) * (height - 1)
        cx = rng.uniform(0.35, 0.65) * (width - 1)
        ry = rng.uniform(0.10, 0.20) * height
        rx = rng.uniform(0.10, 0.20) * width
        mask[((yy - cy) / max(ry, 1e-6)) ** 2 + ((xx - cx) / max(rx, 1e-6)) ** 2 < 1.0] = False
    elif kind == "coast":
        phase = rng.uniform(0.0, 2.0 * math.pi)
        slope = rng.uniform(-0.15, 0.15)
        boundary = width * (
            rng.uniform(0.55, 0.80)
            + 0.10 * np.sin(2.0 * math.pi * yy / max(height, 1) + phase)
            + slope * (yy / max(height - 1, 1) - 0.5)
        )
        mask[xx > boundary] = False
    elif kind == "islands":
        mask = np.zeros((height, width), dtype=bool)
        num_islands = int(rng.integers(2, min(6, max(3, min(height, width) // 3)) + 1))
        centers = np.column_stack(
            [
                rng.uniform(0.15 * (height - 1), 0.85 * (height - 1), size=num_islands),
                rng.uniform(0.15 * (width - 1), 0.85 * (width - 1), size=num_islands),
            ]
        )
        for cy, cx in centers:
            ry = rng.uniform(0.10, 0.22) * height
            rx = rng.uniform(0.10, 0.22) * width
            mask |= ((yy - cy) / max(ry, 1.0)) ** 2 + ((xx - cx) / max(rx, 1.0)) ** 2 < 1.0
        for left, right in zip(centers[:-1], centers[1:], strict=True):
            steps = max(int(np.linalg.norm(right - left)) * 3, 2)
            for y, x in zip(
                np.linspace(left[0], right[0], steps),
                np.linspace(left[1], right[1], steps),
                strict=True,
            ):
                row = int(np.clip(round(y), 0, height - 1))
                col = int(np.clip(round(x), 0, width - 1))
                row_slice = slice(max(0, row - 1), min(height, row + 2))
                col_slice = slice(max(0, col - 1), min(width, col + 2))
                mask[row_slice, col_slice] = True
    elif kind == "corridor":
        mask = random_corridor_mask(rng, height, width)
    else:
        raise ValueError(f"unknown mask kind: {kind}")
    return largest_component_mask(mask)


@beartype
def random_corridor_mask(
    rng: np.random.Generator,
    height: int,
    width: int,
) -> np.ndarray:
    """Rasterize a meandering corridor with a few connected side branches."""
    rows = np.linspace(0.08 * height, 0.92 * height, int(rng.integers(5, 8)))
    cols = np.cumsum(rng.normal(scale=0.10 * width, size=rows.shape[0]))
    cols -= float(cols.mean())
    cols += rng.uniform(0.35, 0.65) * width
    yy, xx = np.mgrid[0:height, 0:width]
    distance = np.full((height, width), np.inf, dtype=np.float64)
    for (r0, c0), (r1, c1) in zip(
        np.column_stack([rows[:-1], cols[:-1]]),
        np.column_stack([rows[1:], cols[1:]]),
        strict=True,
    ):
        segment = np.asarray([r1 - r0, c1 - c0], dtype=np.float64)
        denom = max(float(np.dot(segment, segment)), 1e-12)
        t = np.clip(((yy - r0) * segment[0] + (xx - c0) * segment[1]) / denom, 0.0, 1.0)
        closest_r = r0 + t * segment[0]
        closest_c = c0 + t * segment[1]
        distance = np.minimum(distance, np.hypot(yy - closest_r, xx - closest_c))
    width_px = rng.uniform(0.07, 0.13) * min(height, width)
    mask = distance <= width_px
    for _ in range(int(rng.integers(1, 4))):
        anchor = int(rng.integers(1, rows.shape[0] - 1))
        branch_angle = rng.uniform(0.0, 2.0 * math.pi)
        branch_length = rng.uniform(0.16, 0.35) * max(height, width)
        r0 = rows[anchor]
        c0 = cols[anchor]
        r1 = r0 + branch_length * math.sin(branch_angle)
        c1 = c0 + branch_length * math.cos(branch_angle)
        segment = np.asarray([r1 - r0, c1 - c0], dtype=np.float64)
        denom = max(float(np.dot(segment, segment)), 1e-12)
        t = np.clip(((yy - r0) * segment[0] + (xx - c0) * segment[1]) / denom, 0.0, 1.0)
        branch_distance = np.hypot(yy - (r0 + t * segment[0]), xx - (c0 + t * segment[1]))
        mask |= branch_distance <= width_px * rng.uniform(0.45, 0.75)
    return largest_component_mask(mask)


@beartype
def unit_positions(height: int, width: int) -> Float[np.ndarray, "nodes 2"]:  # noqa: F722
    """Place row-major grid nodes in [0, 1] x [-1, 0]."""
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = cols.ravel().astype(np.float64) / max(width - 1, 1)
    y = -rows.ravel().astype(np.float64) / max(height - 1, 1)
    return np.column_stack([x, y])


@beartype
def masked_grid_graph(mask: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Number active nodes row-major; list horizontal edges before vertical edges."""
    height, width = mask.shape
    active_full = np.flatnonzero(mask.ravel()).astype(np.int32)
    full_to_active = -np.ones(mask.size, dtype=np.int32)
    full_to_active[active_full] = np.arange(active_full.shape[0], dtype=np.int32)
    horizontal_mask = mask[:, :-1] & mask[:, 1:]
    vertical_mask = mask[:-1, :] & mask[1:, :]
    horizontal_row, horizontal_col = np.nonzero(horizontal_mask)
    vertical_row, vertical_col = np.nonzero(vertical_mask)
    edge_row = np.concatenate((horizontal_row, vertical_row)).astype(np.int32)
    edge_col = np.concatenate((horizontal_col, vertical_col)).astype(np.int32)
    edge_kind = np.repeat([0, 1], [horizontal_row.size, vertical_row.size]).astype(np.int8)
    source = edge_row * width + edge_col
    target = source + np.where(edge_kind == 0, 1, width)
    edges = np.column_stack((full_to_active[source], full_to_active[target]))
    if not edges.size:
        raise ValueError("active mask has no active edges")
    graph = {
        "active_full": np.arange(active_full.shape[0], dtype=np.int32),
        "edges": edges,
        "source_active_full": active_full,
        "edge_kind": edge_kind,
        "edge_row": edge_row,
        "edge_col": edge_col,
    }
    return graph, unit_positions(height, width)[active_full]


@beartype
def feems_triangular_lattice_graph(
    n_rows: int = 8,
    n_columns: int = 12,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Build the triangular lattice geometry used by `feems.sim.setup_graph`."""
    rows = _integer_setting(n_rows, "n_rows", minimum=2)
    columns = _integer_setting(n_columns, "n_columns", minimum=2)
    row_height = math.sqrt(3.0) / 2.0
    row, col = np.indices((rows, columns), dtype=np.int32)
    positions = np.column_stack(((col + 0.5 * (row % 2)).ravel(), (row * row_height).ravel()))
    h_row, h_col = np.indices((rows, columns - 1), dtype=np.int32).reshape(2, -1)
    h_source = h_row * columns + h_col

    # Keep the historical order: all horizontal edges, then a straight and
    # diagonal edge from each node in turn. Diagonals alternate with row parity.
    v_row, v_col = np.repeat(
        np.indices((rows - 1, columns), dtype=np.int32).reshape(2, -1), 2, axis=1
    )
    v_kind = np.tile([1, 2], (rows - 1) * columns).astype(np.int8)
    target_col = v_col + np.where(v_kind == 1, 0, np.where(v_row % 2 == 0, -1, 1))
    keep = (target_col >= 0) & (target_col < columns)
    v_source = v_row * columns + v_col
    v_target = (v_row + 1) * columns + target_col
    edges = np.vstack(
        (
            np.column_stack((h_source, h_source + 1)),
            np.column_stack((v_source[keep], v_target[keep])),
        )
    ).astype(np.int32)
    graph = {
        "active_full": np.arange(rows * columns, dtype=np.int32),
        "edges": edges,
        "source_active_full": np.arange(rows * columns, dtype=np.int32),
        "edge_kind": np.concatenate((np.zeros(h_source.size, dtype=np.int8), v_kind[keep])),
        "edge_row": np.concatenate((h_row, v_row[keep])),
        "edge_col": np.concatenate((h_col, v_col[keep])),
    }
    return graph, positions


@beartype
def graph_from_positions_edges(
    positions: np.ndarray,
    edges: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Validate an external mesh and retain its node and edge order."""
    validate_graph_arrays(positions=positions, edges=edges)
    return {
        "active_full": np.arange(positions.shape[0], dtype=np.int32),
        "edges": np.asarray(edges, dtype=np.int32),
    }, np.asarray(positions, dtype=np.float64)


@lru_cache(maxsize=8)
@beartype
def load_world_land_geometry(resolution: str, cartopy_data_dir: str) -> Any:
    """Load and cache Natural Earth land polygons at the requested resolution."""
    import cartopy
    import cartopy.io.shapereader as shpreader
    from shapely.ops import unary_union

    data_dir = str(cartopy_data_dir or "").strip()
    if data_dir:
        cartopy.config["data_dir"] = Path(data_dir).expanduser().resolve()
    land_path = shpreader.natural_earth(
        resolution=str(resolution),
        category="physical",
        name="land",
    )
    return unary_union(list(shpreader.Reader(land_path).geometries()))


@beartype
def polygonal_geometry(geometry: Any) -> Any:
    """Keep polygonal parts of a clipped geometry, rejecting empty habitat."""
    from shapely.geometry import MultiPolygon, Polygon

    if isinstance(geometry, Polygon | MultiPolygon):
        return geometry
    polygons = [
        geom
        for geom in getattr(geometry, "geoms", [])
        if isinstance(geom, Polygon) and not geom.is_empty
    ]
    if not polygons:
        raise ValueError("geometry contains no polygonal land")
    return MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]


@beartype
def local_world_projection(lon: float, lat: float) -> tuple[Any, Any]:
    """Return forward/inverse local azimuthal projections, in kilometers."""
    from pyproj import CRS, Transformer

    crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat:.8f} +lon_0={lon:.8f} +datum=WGS84 +units=km +no_defs"
    )
    to_xy = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return to_xy, to_lonlat


@beartype
def project_geometry(geometry: Any, transformer: Any) -> Any:
    """Project a habitat polygon with a pyproj transformer."""
    from shapely.ops import transform

    return polygonal_geometry(transform(transformer.transform, geometry))


@beartype
def points_in_geometry(points: np.ndarray, geometry: Any) -> Bool[np.ndarray, "nodes"]:  # noqa: F821, UP037
    """Test habitat membership, including points exactly on its boundary."""
    import shapely

    shapely.prepare(geometry)
    return shapely.covers(geometry, shapely.points(points))


@beartype
def sample_global_landscape_region(
    rng: np.random.Generator,
    sim: Mapping[str, Any],
) -> dict[str, Any]:
    """Choose a land-centered world-map crop and project it to local kilometers."""
    from shapely.geometry import Point, box

    land = load_world_land_geometry(
        str(sim["global_landscape_resolution"]),
        str(sim["global_landscape_cartopy_data_dir"]),
    )
    min_box = float(sim["global_landscape_min_box_deg"])
    max_box = max(float(sim["global_landscape_max_box_deg"]), min_box)
    aspect_min = float(sim.get("global_landscape_box_aspect_min", 0.65))
    aspect_max = float(sim.get("global_landscape_box_aspect_max", 1.35))
    aspect_low = min(aspect_min, aspect_max)
    aspect_high = max(aspect_min, aspect_max)
    min_lat = float(sim["global_landscape_min_lat"])
    max_lat = float(sim["global_landscape_max_lat"])
    min_land_fraction = float(sim["global_landscape_min_land_fraction"])
    preferred_lon = sim.get("global_landscape_preferred_center_lon")
    preferred_lat = sim.get("global_landscape_preferred_center_lat")
    center_jitter = max(float(sim.get("global_landscape_center_jitter_deg", 0.0)), 0.0)
    for _attempt in range(int(sim["global_landscape_max_region_tries"])):
        if sim.get("global_landscape_box_width_choices") is not None:
            width = draw_weighted_positive_float(
                rng,
                choices=sim.get("global_landscape_box_width_choices"),
                weights=sim.get("global_landscape_box_width_weights"),
                fallback=float(rng.uniform(min_box, max_box)),
            )
            width = float(np.clip(width, min_box, max_box))
        else:
            width = float(rng.uniform(min_box, max_box))
        height = float(width * rng.uniform(aspect_low, aspect_high))
        if preferred_lon is not None and preferred_lat is not None:
            lon = float(preferred_lon)
            lat = float(preferred_lat)
            if center_jitter > 0.0:
                lon = float(rng.normal(lon, center_jitter))
                lat = float(rng.normal(lat, center_jitter))
        else:
            lon = float(rng.uniform(-175.0 + width / 2.0, 175.0 - width / 2.0))
            lat = float(rng.uniform(min_lat + height / 2.0, max_lat - height / 2.0))
        lon = float(np.clip(lon, -175.0 + width / 2.0, 175.0 - width / 2.0))
        lat = float(np.clip(lat, min_lat + height / 2.0, max_lat - height / 2.0))
        bbox = box(lon - width / 2.0, lat - height / 2.0, lon + width / 2.0, lat + height / 2.0)
        if not land.covers(Point(lon, lat)):
            continue
        clipped = land.intersection(bbox)
        if clipped.is_empty:
            continue
        try:
            clipped = polygonal_geometry(clipped)
        except ValueError:
            continue
        land_fraction = float(clipped.area / max(bbox.area, 1e-12))
        if land_fraction < min_land_fraction:
            continue
        to_xy, to_lonlat = local_world_projection(lon, lat)
        box_xy = project_geometry(bbox, to_xy)
        land_xy = project_geometry(clipped, to_xy)
        if land_xy.area <= 0.0 or box_xy.area <= 0.0:
            continue
        return {
            "bbox": bbox,
            "box_xy": box_xy,
            "land_xy": land_xy,
            "center_lon": float(lon),
            "center_lat": float(lat),
            "land_fraction": land_fraction,
            "to_lonlat": to_lonlat,
            "box_width_deg": float(width),
            "box_height_deg": float(height),
        }
    raise RuntimeError("could not find a suitable global landscape crop")


@jaxtyped(typechecker=beartype)
def hex_candidates(
    bounds: tuple[float, float, float, float], spacing: float
) -> Float[np.ndarray, "nodes 2"]:
    """Enumerate staggered rows, including a one-cell margin around the bounds."""
    xmin, ymin, xmax, ymax = bounds
    if not np.all(np.isfinite(bounds)) or xmin > xmax or ymin > ymax:
        raise ValueError("bounds must be finite and ordered xmin, ymin, xmax, ymax")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive")
    y_step = math.sqrt(3.0) * 0.5 * spacing
    margin = (xmin - spacing, xmax + spacing, ymin - y_step, ymax + y_step)
    if not np.all(np.isfinite(margin)) or any(
        outer == inner for outer, inner in zip(margin, (xmin, xmax, ymin, ymax), strict=True)
    ):
        raise ValueError("spacing must advance the coordinates without overflow")
    points: list[tuple[float, float]] = []
    row = 0
    y = ymin - y_step
    while y <= ymax + y_step:
        offset = 0.5 * spacing if row % 2 else 0.0
        x = xmin - spacing + offset
        while x <= xmax + spacing:
            points.append((float(x), float(y)))
            x += spacing
        y += y_step
        row += 1
    return np.asarray(points, dtype=np.float64)


@beartype
def hex_points_in_geometry(geometry: Any, spacing: float) -> np.ndarray:
    """Keep candidate hex-grid nodes covered by the habitat, including its boundary."""
    candidates = hex_candidates(geometry.bounds, spacing)
    return candidates[points_in_geometry(candidates, geometry)]


@beartype
def choose_hex_spacing(geometry: Any, target_nodes: int) -> tuple[float, np.ndarray]:
    """Search spacing geometrically for the closest attainable node count."""
    _integer_setting(target_nodes, "target_nodes")
    xmin, ymin, xmax, ymax = geometry.bounds
    extent = max(float(xmax - xmin), float(ymax - ymin), 1.0)
    low = max(extent / 700.0, 0.1)
    high = max(extent * 2.0, low * 2.0)
    best_spacing = high
    best_points = hex_points_in_geometry(geometry, high)
    best_error = abs(best_points.shape[0] - int(target_nodes))
    for _ in range(44):
        spacing = math.sqrt(low * high)
        points = hex_points_in_geometry(geometry, spacing)
        error = abs(points.shape[0] - int(target_nodes))
        if error < best_error:
            best_spacing = spacing
            best_points = points
            best_error = error
        if points.shape[0] > int(target_nodes):
            low = spacing
        else:
            high = spacing
    return float(best_spacing), best_points


@jaxtyped(typechecker=beartype)
def edges_from_hex_points(
    points: Real[np.ndarray, "nodes 2"], spacing: float
) -> Int[np.ndarray, "edges 2"]:
    """Connect points within 1.08 grid spacings, sorting edges by endpoint IDs."""
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain finite coordinates")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive")
    if points.shape[0] < 2:
        return np.empty((0, 2), dtype=np.int32)
    pairs = sorted(cKDTree(points).query_pairs(r=float(spacing) * 1.08))
    if not pairs:
        return np.empty((0, 2), dtype=np.int32)
    return np.asarray(pairs, dtype=np.int32)


@jaxtyped(typechecker=beartype)
def largest_graph_component(
    positions: Real[np.ndarray, "nodes 2"],
    edges: Int[np.ndarray, "edges 2"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the largest component and its original node IDs; ties use the first."""
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions must contain finite coordinates")
    if np.any(edges < 0) or np.any(edges >= positions.shape[0]):
        raise ValueError("edges must refer to valid node indices")
    if positions.shape[0] <= 1:
        return positions, edges, np.arange(positions.shape[0], dtype=np.int32)
    if not edges.size:
        return positions[:1], edges, np.asarray([0], dtype=np.int32)
    rows = np.r_[edges[:, 0], edges[:, 1]]
    cols = np.r_[edges[:, 1], edges[:, 0]]
    graph = sp.csr_matrix(
        (np.ones(rows.shape[0]), (rows, cols)),
        shape=(positions.shape[0], positions.shape[0]),
    )
    n_components, labels = connected_components(graph, directed=False)
    if n_components <= 1:
        return positions, edges, np.arange(positions.shape[0], dtype=np.int32)
    counts = np.bincount(labels)
    keep_label = int(np.argmax(counts))
    keep_nodes = np.flatnonzero(labels == keep_label).astype(np.int32)
    old_to_new = -np.ones(positions.shape[0], dtype=np.int32)
    old_to_new[keep_nodes] = np.arange(keep_nodes.shape[0], dtype=np.int32)
    keep_edges = (labels[edges[:, 0]] == keep_label) & (labels[edges[:, 1]] == keep_label)
    return positions[keep_nodes], old_to_new[edges[keep_edges]], keep_nodes


@beartype
def build_global_landscape_mesh(
    rng: np.random.Generator,
    sim: Mapping[str, Any],
) -> dict[str, Any]:
    """Mesh the whole crop, retaining water edges but requiring enough land sites."""
    target_nodes = draw_weighted_positive_int(
        rng,
        value=sim["global_landscape_target_nodes"],
        choices=sim.get("global_landscape_target_node_choices"),
        weights=sim.get("global_landscape_target_node_weights"),
        fallback=int(sim["global_landscape_target_nodes"]),
    )
    min_nodes = int(sim["global_landscape_min_nodes"])
    max_nodes = int(sim["global_landscape_max_nodes"])
    min_sampled_sites = int(sim["global_landscape_min_sampled_sites"])
    for _attempt in range(int(sim["global_landscape_max_region_tries"])):
        region = sample_global_landscape_region(rng, sim)
        spacing, positions = choose_hex_spacing(region["box_xy"], target_nodes)
        edges = edges_from_hex_points(positions, spacing)
        positions, edges, keep_nodes = largest_graph_component(positions, edges)
        if edges.shape[0] == 0 or positions.shape[0] < min_nodes or positions.shape[0] > max_nodes:
            continue
        land_mask = points_in_geometry(positions, region["land_xy"])
        if int(land_mask.sum()) < max(min_sampled_sites, 2):
            continue
        lon, lat = region["to_lonlat"].transform(positions[:, 0], positions[:, 1])
        region.update(
            {
                "positions": positions.astype(np.float64, copy=False),
                "lonlat": np.column_stack([lon, lat]).astype(np.float64),
                "edges": edges.astype(np.int32, copy=False),
                "spacing_km": float(spacing),
                "land_mask": land_mask.astype(bool, copy=False),
                "component_source_nodes": keep_nodes,
            }
        )
        return region
    raise RuntimeError("could not build a global landscape mesh")


@beartype
def standardize(values: np.ndarray) -> np.ndarray:
    """Center a field and give it unit SD; constant fields become zero."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    values = values - float(np.mean(values))
    sd = float(np.std(values))
    return values / sd if sd > 0.0 else values


@beartype
def scaled_positions(positions: np.ndarray) -> np.ndarray:
    """Center coordinates and scale the farthest node to unit radius."""
    coords = np.asarray(positions, dtype=np.float64) - np.mean(positions, axis=0, keepdims=True)
    scale = float(np.max(np.linalg.norm(coords, axis=1))) if coords.size else 0.0
    return coords / scale if scale > 0.0 else coords


@beartype
def draw_rff_params(
    rng: np.random.Generator,
    *,
    features_count: int,
    length_scale: float,
) -> dict[str, np.ndarray | float]:
    """Draw a stationary field using random Fourier features."""
    features_count = max(int(features_count), 1)
    return {
        "frequencies": rng.normal(size=(features_count, 2)) / max(float(length_scale), 1e-3),
        "phases": rng.uniform(0.0, 2.0 * math.pi, size=features_count),
        "weights": rng.normal(size=features_count),
    }


@beartype
def eval_rff_params(params: Mapping[str, Any], positions: np.ndarray) -> np.ndarray:
    """Evaluate saved Fourier features and standardize over the supplied nodes."""
    coords = scaled_positions(positions)
    frequencies = np.asarray(params["frequencies"], dtype=np.float64)
    phases = np.asarray(params["phases"], dtype=np.float64)
    weights = np.asarray(params["weights"], dtype=np.float64)
    field = math.sqrt(2.0 / frequencies.shape[0]) * (
        np.cos(coords @ frequencies.T + phases[None, :]) @ weights
    )
    return standardize(field)


@beartype
def box_scaled_positions(positions: np.ndarray) -> np.ndarray:
    """Scale each coordinate axis to [0, 1] for procedural surface features."""
    coords = np.asarray(positions, dtype=np.float64)
    lo = np.min(coords, axis=0, keepdims=True)
    span = np.ptp(coords, axis=0, keepdims=True)
    span = np.where(span > 0.0, span, 1.0)
    return (coords - lo) / span


@beartype
def landscape_feature(
    kind: str,
    *,
    strength: float,
    center: tuple[float, float] | None = None,
    radius: tuple[float, float] | None = None,
    width: float | None = None,
    slope: float = 0.0,
    intercept: float = 0.5,
    side: float = 1.0,
    phase: float = 0.0,
    amplitude: float = 0.0,
) -> dict[str, Any]:
    """Describe a procedural barrier or corridor in normalized map coordinates."""
    out: dict[str, Any] = {
        "kind": kind,
        "strength": float(strength),
        "slope": float(slope),
        "intercept": float(intercept),
        "side": float(side),
        "phase": float(phase),
        "amplitude": float(amplitude),
    }
    if center is not None:
        out["center"] = [float(center[0]), float(center[1])]
    if radius is not None:
        out["radius"] = [float(radius[0]), float(radius[1])]
    if width is not None:
        out["width"] = float(width)
    return out


@beartype
def jittered_pair(
    rng: np.random.Generator,
    center: tuple[float, float],
    *,
    scale: float,
) -> tuple[float, float]:
    """Perturb a feature center, keeping it within the central 90% of the map."""
    values = np.asarray(center, dtype=np.float64) + rng.normal(scale=float(scale), size=2)
    return float(np.clip(values[0], 0.05, 0.95)), float(np.clip(values[1], 0.05, 0.95))


@beartype
def draw_north_america_landscape_features(
    rng: np.random.Generator,
    local: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Draw a stylized North America motif; these are not measured terrain data."""
    jitter = float(local.get("landscape_feature_jitter", 0.06))
    water_strength = float(local.get("landscape_water_strength", 2.7))
    mountain_strength = float(local.get("landscape_mountain_strength", 1.8))
    coast_strength = float(local.get("landscape_coast_strength", 1.5))
    return [
        landscape_feature(
            "ellipse_low",
            center=jittered_pair(rng, (0.70, 0.66), scale=0.55 * jitter),
            radius=(float(rng.uniform(0.10, 0.16)), float(rng.uniform(0.12, 0.22))),
            strength=float(water_strength * rng.uniform(0.85, 1.35)),
        ),
        landscape_feature(
            "ellipse_low",
            center=jittered_pair(rng, (0.75, 0.43), scale=0.65 * jitter),
            radius=(float(rng.uniform(0.10, 0.18)), float(rng.uniform(0.035, 0.075))),
            strength=float(water_strength * rng.uniform(0.55, 0.95)),
        ),
        landscape_feature(
            "ridge_low",
            slope=float(rng.uniform(-0.75, -0.35)),
            intercept=float(rng.uniform(0.56, 0.74)),
            width=float(rng.uniform(0.035, 0.075)),
            strength=float(mountain_strength * rng.uniform(0.75, 1.25)),
        ),
        landscape_feature(
            "coast_low",
            slope=float(rng.uniform(-0.18, 0.18)),
            intercept=float(rng.uniform(0.12, 0.22)),
            width=float(rng.uniform(0.025, 0.06)),
            side=-1.0,
            phase=float(rng.uniform(0.0, 2.0 * math.pi)),
            amplitude=float(rng.uniform(0.015, 0.055)),
            strength=float(coast_strength * rng.uniform(0.55, 1.05)),
        ),
    ]


@beartype
def draw_random_landscape_features(
    rng: np.random.Generator,
    local: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Mix randomly placed low-migration regions, ridges, and coasts."""
    min_features = max(int(local.get("landscape_min_features", 3)), 1)
    max_features = max(int(local.get("landscape_max_features", 6)), min_features)
    count = int(rng.integers(min_features, max_features + 1))
    water_strength = float(local.get("landscape_water_strength", 2.7))
    mountain_strength = float(local.get("landscape_mountain_strength", 1.8))
    coast_strength = float(local.get("landscape_coast_strength", 1.5))
    features: list[dict[str, Any]] = []
    for _ in range(count):
        kind = choose_weighted_name(
            rng,
            ["ellipse_low", "ridge_low", "coast_low"],
            {"ellipse_low": 2.0, "ridge_low": 1.5, "coast_low": 1.0},
        )
        if kind == "ellipse_low":
            features.append(
                landscape_feature(
                    kind,
                    center=(float(rng.uniform(0.20, 0.82)), float(rng.uniform(0.18, 0.82))),
                    radius=(float(rng.uniform(0.06, 0.18)), float(rng.uniform(0.04, 0.18))),
                    strength=float(water_strength * rng.uniform(0.45, 1.25)),
                )
            )
        elif kind == "ridge_low":
            features.append(
                landscape_feature(
                    kind,
                    slope=float(rng.uniform(-1.25, 1.25)),
                    intercept=float(rng.uniform(0.18, 0.82)),
                    width=float(rng.uniform(0.025, 0.09)),
                    strength=float(mountain_strength * rng.uniform(0.45, 1.35)),
                )
            )
        else:
            features.append(
                landscape_feature(
                    kind,
                    slope=float(rng.uniform(-0.35, 0.35)),
                    intercept=float(rng.uniform(0.12, 0.88)),
                    width=float(rng.uniform(0.02, 0.075)),
                    side=float(rng.choice([-1.0, 1.0])),
                    phase=float(rng.uniform(0.0, 2.0 * math.pi)),
                    amplitude=float(rng.uniform(0.0, 0.08)),
                    strength=float(coast_strength * rng.uniform(0.45, 1.25)),
                )
            )
    return features


@beartype
def draw_multiscale_surface(
    rng: np.random.Generator,
    local: Mapping[str, Any],
) -> dict[str, Any]:
    """Draw a rotation-neutral mixture of smooth, regional, and local structure."""
    features_per_band = max(int(local["gp_features"]) // 3, 8)
    bands = []
    for low, high in ((0.65, 1.25), (0.22, 0.55), (0.07, 0.20)):
        length_scale = float(rng.uniform(low, high))
        bands.append(
            {
                "field": draw_rff_params(
                    rng,
                    features_count=features_per_band,
                    length_scale=length_scale,
                ),
                "length_scale": length_scale,
                "weight": float(rng.uniform(0.55, 1.25)),
            }
        )

    regions = []
    for _ in range(int(rng.integers(1, 4))):
        sign = float(rng.choice([-1.0, 1.0]))
        length_scale = float(rng.uniform(0.18, 0.70))
        regions.append(
            {
                "field": draw_rff_params(
                    rng,
                    features_count=features_per_band,
                    length_scale=length_scale,
                ),
                "length_scale": length_scale,
                "threshold": float(rng.uniform(-0.75, 0.75)),
                "interior_strength": sign * float(rng.uniform(0.45, 1.55)),
                "boundary_strength": (
                    -float(rng.uniform(0.25, 1.10)) if rng.random() < 0.65 else 0.0
                ),
            }
        )

    primitives = []
    for _ in range(int(rng.integers(1, 5))):
        center_angle = float(rng.uniform(0.0, 2.0 * math.pi))
        center_radius = float(math.sqrt(rng.uniform(0.0, 1.0)) * 0.72)
        elongated = bool(rng.random() < 0.6)
        if elongated:
            along_scale = float(rng.uniform(0.30, 0.90))
            across_scale = float(rng.uniform(0.035, 0.14))
        else:
            along_scale = float(rng.uniform(0.10, 0.34))
            across_scale = float(rng.uniform(0.08, 0.30))
        primitives.append(
            {
                "center": [
                    center_radius * math.cos(center_angle),
                    center_radius * math.sin(center_angle),
                ],
                "angle": float(rng.uniform(0.0, 2.0 * math.pi)),
                "along_scale": along_scale,
                "across_scale": across_scale,
                "power": float(rng.choice([2.0, 4.0])),
                "strength": float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.35, 1.45)),
            }
        )

    return {
        "kind": "multiscale",
        "std": float(local["log_rate_std"]),
        "bands": bands,
        "regions": regions,
        "primitives": primitives,
    }


@beartype
def multiscale_edge_values(
    surface: Mapping[str, Any],
    positions: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Combine smooth bands, thresholded regions, and oriented local features."""
    left = edges[:, 0]
    right = edges[:, 1]
    node_values = np.zeros(positions.shape[0], dtype=np.float64)
    for band in surface["bands"]:
        node_values += float(band["weight"]) * eval_rff_params(band["field"], positions)
    values = 0.5 * (node_values[left] + node_values[right])

    # Thresholded random fields create irregular regions and sharp boundaries.
    for region in surface["regions"]:
        field = eval_rff_params(region["field"], positions)
        inside = field >= float(region["threshold"])
        interior = inside[left] & inside[right]
        boundary = inside[left] != inside[right]
        values += float(region["interior_strength"]) * interior
        values += float(region["boundary_strength"]) * boundary

    coords = scaled_positions(positions)
    midpoint = 0.5 * (coords[left] + coords[right])
    for primitive in surface["primitives"]:
        angle = float(primitive["angle"])
        along_vector = np.asarray([math.cos(angle), math.sin(angle)])
        across_vector = np.asarray([-math.sin(angle), math.cos(angle)])
        delta = midpoint - np.asarray(primitive["center"], dtype=np.float64)
        along = np.abs(delta @ along_vector) / float(primitive["along_scale"])
        across = np.abs(delta @ across_vector) / float(primitive["across_scale"])
        power = float(primitive["power"])
        profile = np.exp(-0.5 * (along**power + across**power))
        values += float(primitive["strength"]) * profile
    return values


@beartype
def landscape_edge_penalty(
    features: Sequence[Mapping[str, Any]],
    positions: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Evaluate low-migration features at endpoints or edge midpoints."""
    coords = box_scaled_positions(positions)
    left = edges[:, 0]
    right = edges[:, 1]
    mid = 0.5 * (coords[left] + coords[right])
    penalty = np.zeros(edges.shape[0], dtype=np.float64)
    for feature in features:
        strength = float(feature.get("strength", 1.0))
        kind = str(feature.get("kind", "ellipse_low"))
        if kind == "ellipse_low":
            center = np.asarray(feature.get("center", [0.5, 0.5]), dtype=np.float64)
            radius = np.asarray(feature.get("radius", [0.12, 0.12]), dtype=np.float64)
            radius = np.maximum(radius, 1e-3)
            node_score = np.exp(
                -0.5
                * (
                    ((coords[:, 0] - center[0]) / radius[0]) ** 2
                    + ((coords[:, 1] - center[1]) / radius[1]) ** 2
                )
            )
            penalty += strength * 0.5 * (node_score[left] + node_score[right])
        elif kind == "ridge_low":
            slope = float(feature.get("slope", 0.0))
            intercept = float(feature.get("intercept", 0.5))
            width = max(float(feature.get("width", 0.05)), 1e-3)
            signed = mid[:, 1] - (intercept + slope * (mid[:, 0] - 0.5))
            penalty += strength * np.exp(-0.5 * (signed / width) ** 2)
        elif kind == "coast_low":
            slope = float(feature.get("slope", 0.0))
            intercept = float(feature.get("intercept", 0.5))
            width = max(float(feature.get("width", 0.05)), 1e-3)
            phase = float(feature.get("phase", 0.0))
            amplitude = float(feature.get("amplitude", 0.0))
            side = float(feature.get("side", 1.0))
            boundary = (
                intercept
                + slope * (mid[:, 1] - 0.5)
                + amplitude * np.sin(2.0 * math.pi * mid[:, 1] + phase)
            )
            signed = side * (mid[:, 0] - boundary)
            penalty += strength / (1.0 + np.exp(-np.clip(signed / width, -40.0, 40.0)))
        else:
            raise ValueError(f"unknown landscape feature kind: {kind!r}")
    return penalty


@beartype
def draw_surface(
    rng: np.random.Generator,
    *,
    surface_kind: str,
    local: Mapping[str, Any],
) -> dict[str, Any]:
    """Draw field parameters now and evaluate them on a graph later."""
    kind = str(surface_kind)
    if kind == "constant":
        return {"kind": kind, "std": 0.0}
    if kind in {
        "feems_homogeneous",
        "feems_heterogeneous_barrier",
        "feems_anisotropic",
    }:
        return {
            "kind": kind,
            "std": 0.0,
            "barrier_startpt": float(local["feems_lattice_barrier_startpt"]),
            "barrier_endpt": float(local["feems_lattice_barrier_endpt"]),
            "barrier_w": float(local["feems_lattice_barrier_w"]),
            "corridor_w": float(local["feems_lattice_corridor_w"]),
            "anisotropy_scaler": float(local["feems_lattice_anisotropy_scaler"]),
        }
    if kind == "multiscale":
        return draw_multiscale_surface(rng, local)
    features_count = 8 if kind == "smooth" else int(local["gp_features"])
    length_scale = float(local["migration_gp_length_scale"])
    surface: dict[str, Any] = {
        "kind": kind,
        "std": float(local["log_rate_std"]),
        "field": draw_rff_params(rng, features_count=features_count, length_scale=length_scale),
    }
    if kind in {"barrier", "corridor"}:
        surface.update(
            {
                "axis": int(rng.integers(0, 2)),
                "center": float(rng.uniform(-0.35, 0.35)),
                "width": float(rng.uniform(0.04, 0.14)),
                "strength": float(rng.uniform(1.5, 3.5)),
            }
        )
    if kind == "earth":
        surface.update(
            {
                "habitat": draw_rff_params(
                    rng,
                    features_count=max(int(local["gp_features"]), 32),
                    length_scale=float(rng.uniform(0.15, 0.45)),
                ),
                "habitat_weight": float(rng.normal(0.7, 0.35)),
                "mismatch_weight": float(rng.uniform(0.15, 0.65)),
                "length_weight": float(rng.uniform(0.05, 0.25)),
            }
        )
    if kind == "landscape":
        surface["landscape_features"] = draw_random_landscape_features(rng, local)
    if kind == "north_america_landscape":
        surface["landscape_features"] = draw_north_america_landscape_features(rng, local)
    return surface


@beartype
def edge_log_rates_from_surface(
    surface: Mapping[str, Any],
    positions: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Evaluate centered log conductances in the supplied edge order.

    Generic fields are rescaled to the requested SD. FEEMS lattice motifs
    instead preserve their explicit barrier/corridor or anisotropy ratios.
    """
    kind = str(surface["kind"])
    if kind == "constant":
        return np.zeros(edges.shape[0], dtype=np.float32)
    left = edges[:, 0]
    right = edges[:, 1]
    if kind in {
        "feems_homogeneous",
        "feems_heterogeneous_barrier",
        "feems_anisotropic",
    }:
        rates = np.ones(edges.shape[0], dtype=np.float64)
        if kind == "feems_heterogeneous_barrier":
            x = 0.5 * (positions[left, 0] + positions[right, 0])
            start = float(surface["barrier_startpt"])
            end = float(surface["barrier_endpt"])
            corridor_w = float(surface["corridor_w"])
            barrier_w = float(surface["barrier_w"])
            rates.fill(corridor_w)
            rates[(x > start) & (x < end)] = barrier_w
        elif kind == "feems_anisotropic":
            delta = positions[right] - positions[left]
            horizontal = np.abs(delta[:, 1]) <= 1e-12
            rates[horizontal] = float(surface["anisotropy_scaler"])
        logs = np.log(np.maximum(rates, 1e-12))
        return (logs - float(np.mean(logs))).astype(np.float32)

    if kind == "multiscale":
        values = multiscale_edge_values(surface, positions, edges)
        return (standardize(values) * float(surface["std"])).astype(np.float32)

    node_field = eval_rff_params(surface["field"], positions)
    values = 0.5 * (node_field[left] + node_field[right])
    coords = scaled_positions(positions)
    midpoint = 0.5 * (coords[left] + coords[right])
    if kind == "barrier":
        axis = int(surface["axis"])
        distance = (midpoint[:, axis] - float(surface["center"])) / float(surface["width"])
        values = values - float(surface["strength"]) * np.exp(-0.5 * distance**2)
    elif kind == "corridor":
        axis = int(surface["axis"])
        distance = (midpoint[:, axis] - float(surface["center"])) / float(surface["width"])
        values = values + float(surface["strength"]) * np.exp(-0.5 * distance**2)
    elif kind == "earth":
        habitat = eval_rff_params(surface["habitat"], positions)
        edge_habitat = 0.5 * (habitat[left] + habitat[right])
        habitat_mismatch = np.abs(habitat[left] - habitat[right])
        edge_lengths = np.linalg.norm(positions[left] - positions[right], axis=1)
        values = (
            values
            + float(surface["habitat_weight"]) * edge_habitat
            - float(surface["mismatch_weight"]) * habitat_mismatch
            - float(surface["length_weight"]) * standardize(edge_lengths)
        )
    elif kind in {"landscape", "north_america_landscape"}:
        values = values - landscape_edge_penalty(
            surface.get("landscape_features", []),
            positions,
            edges,
        )
    elif kind not in {"smooth", "gp"}:
        raise ValueError(f"unknown surface_kind={kind!r}")
    return (standardize(values) * float(surface["std"])).astype(np.float32)


@beartype
def node_log_q_from_surface(
    q_surface: Mapping[str, Any] | None,
    positions: np.ndarray,
) -> np.ndarray:
    """Evaluate centered log coalescence rates; an absent field means constant q."""
    if q_surface is None:
        return np.zeros(positions.shape[0], dtype=np.float32)
    return (eval_rff_params(q_surface["field"], positions) * float(q_surface["std"])).astype(
        np.float32
    )


@beartype
def draw_q_surface(
    rng: np.random.Generator,
    local: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Draw an independent local-coalescence field, or None for constant q."""
    if float(local["q_log_std"]) <= 0.0 or str(local.get("q_prior", "gp")) == "constant":
        return None
    features_count = 8 if str(local.get("q_prior", "gp")) == "smooth" else int(local["gp_features"])
    return {
        "std": float(local["q_log_std"]),
        "field": draw_rff_params(
            rng,
            features_count=features_count,
            length_scale=float(local["q_gp_length_scale"]),
        ),
    }
