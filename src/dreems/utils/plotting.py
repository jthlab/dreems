import contextily as cx
import matplotlib.pyplot as plt
import matplotlib as mpl

from collections import Counter
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
from pyproj import CRS

import numpy as np
import rasterio
from rasterio.plot import show
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

def draw_projected_grid_map(
    surface,
    sample_to_pop=None,
    linewidths=2.5,
    alpha=1.0,
    limit=1.0,
    map_margin_km=100.0,
    basemap_source=None,
    posterior=False,
    figure_path=None,
):
    # surface["node_position"] is assumed to contain x/y in kilometers
    # these are the projected positions
    positions_km = np.asarray(
        surface["node_position"].values,
        dtype=np.float64,
    )

    # Contextily and the PyProj CRS below use meters.
    positions_m = positions_km * 1000.0

    edge_nodes = np.asarray(surface["edge_node"].values, dtype=int)
    segments_m = positions_m[edge_nodes]

    edge_log_migration = surface["edge_log_migration_refined"].values
    edge_log_sd = surface["edge_log_migration_sd"].values
    sample_count = surface["node_sample_count"].values

    v_limit = max(
        float(np.max(np.abs(edge_log_migration))),
        0.1,
    ) * limit

    # ------------------------------------------------------------------
    # Reconstruct the CRS used to create node_position
    # ------------------------------------------------------------------

    local_pyproj_crs = CRS.from_proj4(
        f"+proj=aeqd "
        f"+lat_0={surface['projection_center_latitude'].values} "
        f"+lon_0={surface['projection_center_longitude'].values} "
        f"+R=6371008.8 "
        f"+units=m "
        f"+no_defs"
    )

    # choose the background map, there are many maps in cx.providers.
    # Look at: https://xyzservices.readthedocs.io/en/stable/introduction.html
    if basemap_source is None:
        basemap_source = cx.providers.Esri.WorldPhysical

    # plotting population colors based on majority vote.
    # many different populations can live on the same nearest node.
    node_colors = []
    pop_color_map = {}

    if sample_to_pop is not None:
        unique_pops = sorted(set(sample_to_pop.values()))
        pop_cmap = mpl.colormaps["tab10"].resampled(len(unique_pops))

        pop_color_map = {
            pop: pop_cmap(i)
            for i, pop in enumerate(unique_pops)
        }

        sample_ids = surface["sample_id"].values
        sample_nodes = surface["sample_nodes"].values

        node_to_samples = {
            i: [] for i in range(len(sample_count))
        }

        for sample_id, node_index in zip(sample_ids, sample_nodes):
            node_to_samples[int(node_index)].append(sample_id)

        for node_index, count in enumerate(sample_count):
            if count <= 0:
                continue

            populations = [
                sample_to_pop[sample_id]
                for sample_id in node_to_samples[node_index]
                if sample_id in sample_to_pop
            ]

            if populations:
                majority_population = Counter(
                    populations
                ).most_common(1)[0][0]

                node_colors.append(
                    pop_color_map[majority_population]
                )
            else:
                node_colors.append("white")
    else:
        node_colors = ["white"] * int(np.sum(sample_count > 0))

    # choose posterior or not
    if posterior:
        config = {
            "title": "Posterior uncertainty",
            "cbar_label": "posterior SD",
            "cmap": "magma",
            "norm": Normalize(
                vmin=0.0,
                vmax=max(float(edge_log_sd.max()), 0.1) * limit,
            ),
            "values": edge_log_sd,
        }
    else:
        config = {
            "title": "Relative migration",
            "cbar_label": "centered edge log migration",
            "cmap": "RdBu",
            "norm": TwoSlopeNorm(
                vmin=-v_limit,
                vcenter=0.0,
                vmax=v_limit,
            ),
            "values": edge_log_migration,
        }

    # Figure and basemap
    fig, ax = plt.subplots(
        figsize=(10, 18),
        constrained_layout=True,
    )

    margin_m = map_margin_km * 1000.0

    x_min, y_min = positions_m.min(axis=0)
    x_max, y_max = positions_m.max(axis=0)

    ax.set_xlim(x_min - margin_m, x_max + margin_m)
    ax.set_ylim(y_min - margin_m, y_max + margin_m)

    cx.add_basemap(
        ax,
        crs=local_pyproj_crs,
        source=basemap_source,
        zoom="auto",
        attribution_size=6,
        reset_extent=True,
        zorder=0,
    )

    # Networkx edges
    line_coll = LineCollection(
        segments_m,
        cmap=config["cmap"],
        norm=config["norm"],
        linewidths=linewidths,
        alpha=alpha,
        zorder=2,
    )

    line_coll.set_array(config["values"])
    ax.add_collection(line_coll)

    # Edge colorbar
    cbar = fig.colorbar(
        line_coll,
        ax=ax,
        fraction=0.015,
        pad=0.04,
        shrink=0.6,
        extend="both",
    )

    cbar.set_alpha(1.0)
    cbar.set_label(config["cbar_label"], fontsize=16)

    if hasattr(cbar, "solids") and cbar.solids is not None:
        cbar.solids.set_alpha(1.0)

    # Sample nodes
    occupied = sample_count > 0
    occupied_positions_m = positions_m[occupied]
    occupied_counts = sample_count[occupied]

    ax.scatter(
        occupied_positions_m[:, 0],
        occupied_positions_m[:, 1],
        # Preserve node-size scaling by sample count.
        s=20.0 + 15.0 * occupied_counts,
        c=node_colors,
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )

    # Population legend
    if sample_to_pop is not None and pop_color_map:
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                label=population,
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=8,
            )
            for population, color in pop_color_map.items()
        ]

        ax.legend(
            handles=legend_elements,
            title="Populations",
            loc="upper right",
            fontsize=12,        
            title_fontsize=12, 
            frameon=True,
            framealpha=0.9,
        )

    # Formatting
    ax.set_aspect("equal")

    # Coordinates internally remain meters, but display tick labels in km.
    km_formatter = FuncFormatter(
        lambda value, position: f"{value / 1000:g}"
    )

    ax.xaxis.set_major_formatter(km_formatter)
    ax.yaxis.set_major_formatter(km_formatter)

    ax.set_xlabel("Easting (km)", fontsize=16)
    ax.set_ylabel("Northing (km)", fontsize=16)
    ax.set_title(config["title"], fontsize=16, pad=10)

    if figure_path:
        fig.savefig(
            figure_path,
            bbox_inches="tight",
            dpi=300,
        )

    plt.show()

def edge_values_to_node_values(
    number_of_nodes,
    edge_nodes,
    edge_values,
):
    """
    Convert edge-level values to node-level values by averaging all
    edges incident to each node.
    """
    edge_nodes = np.asarray(edge_nodes, dtype=int)
    edge_values = np.asarray(edge_values, dtype=np.float64)

    node_sums = np.zeros(number_of_nodes, dtype=np.float64)
    node_counts = np.zeros(number_of_nodes, dtype=np.float64)

    # Add each edge value to both endpoint nodes.
    np.add.at(node_sums, edge_nodes[:, 0], edge_values)
    np.add.at(node_sums, edge_nodes[:, 1], edge_values)

    np.add.at(node_counts, edge_nodes[:, 0], 1.0)
    np.add.at(node_counts, edge_nodes[:, 1], 1.0)

    node_values = np.full(
        number_of_nodes,
        np.nan,
        dtype=np.float64,
    )

    valid = node_counts > 0
    node_values[valid] = node_sums[valid] / node_counts[valid]

    return node_values

def generalized_contour_surface(
    positions,
    node_values,
    *,
    grid_size=350,
    smoothing_km=100.0,
):
    """
    Interpolate node values to a grid and apply broad spatial smoothing.

    Parameters
    ----------
    positions
        Projected node coordinates in meters, shape (n_nodes, 2).
    node_values
        Scalar value associated with each node.
    grid_size
        Number of grid cells along each map dimension.
    smoothing_km
        Approximate Gaussian smoothing distance in kilometers.

        Larger values produce broader and more generalized contours.
    """
    positions = np.asarray(positions, dtype=np.float64)
    node_values = np.asarray(node_values, dtype=np.float64)

    valid_nodes = (
        np.isfinite(node_values)
        & np.isfinite(positions[:, 0])
        & np.isfinite(positions[:, 1])
    )

    interpolation_positions = positions[valid_nodes]
    interpolation_values = node_values[valid_nodes]

    if len(interpolation_values) < 3:
        raise ValueError(
            "At least three nodes with finite values are required "
            "to construct contours."
        )

    x_values = np.linspace(
        positions[:, 0].min(),
        positions[:, 0].max(),
        grid_size,
    )

    y_values = np.linspace(
        positions[:, 1].min(),
        positions[:, 1].max(),
        grid_size,
    )

    grid_x, grid_y = np.meshgrid(
        x_values,
        y_values,
    )

    # Linear interpolation leaves cells outside the node convex hull
    # as NaN, preventing contours from covering unrelated map areas.
    interpolated_values = griddata(
        points=interpolation_positions,
        values=interpolation_values,
        xi=(grid_x, grid_y),
        method="linear",
        fill_value=np.nan,
    )

    valid_grid = np.isfinite(interpolated_values)

    # Convert smoothing distance from kilometers to grid cells.
    dx_m = float(np.mean(np.diff(x_values)))
    dy_m = float(np.mean(np.diff(y_values)))
    mean_cell_size_m = 0.5 * (abs(dx_m) + abs(dy_m))

    smoothing_sigma = (
        smoothing_km * 1000.0 / mean_cell_size_m
    )

    # Perform NaN-aware Gaussian smoothing.
    numerator = gaussian_filter(
        np.where(valid_grid, interpolated_values, 0.0),
        sigma=smoothing_sigma,
        mode="nearest",
    )

    denominator = gaussian_filter(
        valid_grid.astype(np.float64),
        sigma=smoothing_sigma,
        mode="nearest",
    )

    smoothed_values = np.full_like(
        interpolated_values,
        np.nan,
    )

    usable = valid_grid & (denominator > 1.0e-12)

    smoothed_values[usable] = (
        numerator[usable] / denominator[usable]
    )

    return grid_x, grid_y, smoothed_values


def draw_projected_contour_map(
    surface,
    sample_to_pop=None,
    limit=1.0,
    contour_levels=9,
    contour_linewidth=2.0,
    contour_fill_alpha=0.20,
    contour_line_alpha=1.0,
    smoothing_km=100.0,
    grid_size=350,
    map_margin_km=100.0,
    basemap_source=None,
    basemap_alpha=1.0,
    show_contour_fill=True,
    show_contour_lines=True,
    show_axes=True,
    posterior=False,
    figure_path=None,
):
    """
    Plot broad generalized contours over a physical basemap.

    Assumes surface["node_position"] contains projected x/y coordinates
    in kilometers.
    """

    # 1. PROJECTED COORDINATES
    positions_km = np.asarray(
        surface["node_position"].values,
        dtype=np.float64,
    )

    if positions_km.ndim != 2 or positions_km.shape[1] != 2:
        raise ValueError(
            "surface['node_position'] must have shape "
            "(number_of_nodes, 2)"
        )

    # Contextily and PyProj use meters.
    positions_m = positions_km * 1000.0

    edge_nodes = np.asarray(
        surface["edge_node"].values,
        dtype=int,
    )

    edge_log_migration = np.asarray(
        surface["edge_log_migration_refined"].values,
        dtype=np.float64,
    )

    edge_log_sd = np.asarray(
        surface["edge_log_migration_sd"].values,
        dtype=np.float64,
    )

    sample_count = np.asarray(
        surface["node_sample_count"].values,
    )

    # 2. LOCAL AZIMUTHAL-EQUIDISTANT CRS
    local_pyproj_crs = CRS.from_proj4(
        f"+proj=aeqd "
        f"+lat_0={surface['projection_center_latitude'].values} "
        f"+lon_0={surface['projection_center_longitude'].values} "
        f"+R=6371008.8 "
        f"+units=m "
        f"+no_defs"
    )

    if basemap_source is None:
        basemap_source = cx.providers.Esri.WorldPhysical

    # 3. POPULATION COLORS
    node_colors = []
    pop_color_map = {}

    if sample_to_pop is not None:
        unique_populations = sorted(
            set(sample_to_pop.values())
        )

        population_cmap = mpl.colormaps["tab10"].resampled(
            len(unique_populations)
        )

        pop_color_map = {
            population: population_cmap(index)
            for index, population in enumerate(
                unique_populations
            )
        }

        sample_ids = surface["sample_id"].values
        sample_nodes = surface["sample_nodes"].values

        node_to_samples = {
            node_index: []
            for node_index in range(len(sample_count))
        }

        for sample_id, node_index in zip(
            sample_ids,
            sample_nodes,
        ):
            node_to_samples[int(node_index)].append(sample_id)

        for node_index, count in enumerate(sample_count):
            if count <= 0:
                continue

            populations = [
                sample_to_pop[sample_id]
                for sample_id in node_to_samples[node_index]
                if sample_id in sample_to_pop
            ]

            if populations:
                majority_population = Counter(
                    populations
                ).most_common(1)[0][0]

                node_colors.append(
                    pop_color_map[majority_population]
                )
            else:
                node_colors.append("white")

    else:
        node_colors = [
            "white"
        ] * int(np.sum(sample_count > 0))

    # 4. CONTOUR VALUES AND COLOR NORMALIZATION
    if posterior:
        edge_values = edge_log_sd

        value_limit = max(
            float(np.nanmax(edge_values)),
            0.1,
        ) * limit

        color_norm = Normalize(
            vmin=0.0,
            vmax=value_limit,
        )

        contour_values = np.linspace(
            0.0,
            value_limit,
            contour_levels,
        )

        color_map = "magma"
        colorbar_extend = "max"
        title = "Posterior uncertainty"
        colorbar_label = "posterior SD"

    else:
        edge_values = edge_log_migration

        value_limit = max(
            float(np.nanmax(np.abs(edge_values))),
            0.1,
        ) * limit

        color_norm = TwoSlopeNorm(
            vmin=-value_limit,
            vcenter=0.0,
            vmax=value_limit,
        )

        contour_values = np.linspace(
            -value_limit,
            value_limit,
            contour_levels,
        )

        # Blue for negative values and orange/red for positive values.
        color_map = "RdBu"
        colorbar_extend = "both"
        title = "Relative migration"
        colorbar_label = "centered edge log migration"

    # 5. GENERALIZED SPATIAL SURFACE
    node_values = edge_values_to_node_values(
        number_of_nodes=len(positions_m),
        edge_nodes=edge_nodes,
        edge_values=edge_values,
    )

    grid_x, grid_y, grid_values = generalized_contour_surface(
        positions=positions_m,
        node_values=node_values,
        grid_size=grid_size,
        smoothing_km=smoothing_km,
    )

    # 6. FIGURE AND MAP BOUNDS
    fig, ax = plt.subplots(
        figsize=(12, 10),
        constrained_layout=True,
    )

    x_min, y_min = positions_m.min(axis=0)
    x_max, y_max = positions_m.max(axis=0)

    margin_m = map_margin_km * 1000.0

    ax.set_xlim(
        x_min - margin_m,
        x_max + margin_m,
    )

    ax.set_ylim(
        y_min - margin_m,
        y_max + margin_m,
    )

    # 7. BASEMAP
    cx.add_basemap(
        ax,
        crs=local_pyproj_crs,
        source=basemap_source,
        zoom="auto",
        attribution_size=6,
        reset_extent=True,
        alpha=basemap_alpha,
        zorder=0,
    )

    # 8. FILLED CONTOUR REGIONS
    filled_contours = None

    if show_contour_fill:
        filled_contours = ax.contourf(
            grid_x,
            grid_y,
            grid_values,
            levels=contour_values,
            cmap=color_map,
            norm=color_norm,
            alpha=contour_fill_alpha,
            extend=colorbar_extend,
            antialiased=True,
            zorder=1,
        )

    # 9. CONTOUR LINES
    line_contours = None

    if show_contour_lines:
        line_contours = ax.contour(
            grid_x,
            grid_y,
            grid_values,
            levels=contour_values,
            cmap=color_map,
            norm=color_norm,
            linewidths=contour_linewidth,
            alpha=contour_line_alpha,
            zorder=2,
        )

    # Create an independent, fully opaque colorbar source.
    colorbar_mappable = mpl.cm.ScalarMappable(
        norm=color_norm,
        cmap=mpl.colormaps[color_map],
    )
    colorbar_mappable.set_array([])
    
    cbar = fig.colorbar(
        colorbar_mappable,
        ax=ax,
        fraction=0.025,
        pad=0.03,
        shrink=0.75,
        extend=colorbar_extend,
    )
    
    # Explicitly ensure full opacity.
    cbar.set_alpha(1.0)
    cbar.set_label(colorbar_label, fontsize=16)

    # 11. SAMPLE NODES

    occupied_nodes = sample_count > 0
    occupied_positions = positions_m[occupied_nodes]
    occupied_counts = sample_count[occupied_nodes]

    ax.scatter(
        occupied_positions[:, 0],
        occupied_positions[:, 1],
        s=20.0 + 20.0 * occupied_counts,
        c=node_colors,
        edgecolor="#333333",
        linewidth=1.0,
        zorder=4,
    )

    # 12. POPULATION LEGEND
    if sample_to_pop is not None and pop_color_map:
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                label=population,
                markerfacecolor=color,
                markeredgecolor="#333333",
                markersize=8,
            )
            for population, color in pop_color_map.items()
        ]

        ax.legend(
            handles=legend_elements,
            title="Populations",
            loc="upper right",
            fontsize=12,        
            title_fontsize=12, 
            frameon=True,
            framealpha=0.9,
        )

    # 13. FORMATTING
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=16, pad=10)

    if show_axes:
        kilometer_formatter = FuncFormatter(
            lambda value, position: f"{value / 1000:g}"
        )

        ax.xaxis.set_major_formatter(kilometer_formatter)
        ax.yaxis.set_major_formatter(kilometer_formatter)

        ax.set_xlabel("Easting (km)", fontsize=16)
        ax.set_ylabel("Northing (km)", fontsize=16)

        ax.grid(
            color="#555555",
            linewidth=0.4,
            alpha=0.25,
        )
    else:
        ax.set_axis_off()

    if figure_path:
        fig.savefig(
            figure_path,
            bbox_inches="tight",
            dpi=300,
        )

    plt.show()

    return fig, ax

def draw_projected_node_map(
    surface,
    sample_to_pop=None,
    posterior=False,
    contour_levels=9,
    contour_linewidth=1.5,
    contour_fill_alpha=0.35,
    contour_line_alpha=0.9,
    smoothing_km=100.0,
    grid_size=350,
    map_margin_km=100.0,
    basemap_source=None,
    basemap_alpha=1.0,
    show_contour_fill=True,
    show_contour_lines=True,
    show_nodes=True,
    show_sample_nodes=False,
    node_size=50,
    show_axes=True,
    vmin=None,
    vmax=None,
    figure_path=None,
):
    """
    Plot node log-q means or posterior SDs on a physical basemap.

    Assumptions
    -----------
    surface["node_position"] contains projected x/y coordinates in km.
    Each node-value array has shape (number_of_nodes,).

    posterior=False: plot surface["node_log_q_mean"].
    posterior=True:  plot surface["node_log_q_std"].

    show_nodes colors all nodes by the selected value.
    show_sample_nodes overlays sampled nodes, optionally colored by
    population using sample_to_pop.
    """
    if save_figure and figure_path is None:
        raise ValueError(
            "figure_path is required when save_figure=True."
        )

    # 1. NODE COORDINATES
    positions_km = np.asarray(
        surface["node_position"].values,
        dtype=float,
    )

    if positions_km.ndim != 2 or positions_km.shape[1] != 2:
        raise ValueError(
            "surface['node_position'] must have shape (n_nodes, 2)."
        )

    if not np.all(np.isfinite(positions_km)):
        raise ValueError("Node positions must all be finite.")

    positions_m = positions_km * 1000.0

    # 2. SELECT NODE VALUES
    if posterior:
        value_key = "node_log_q_sd"
        title = "Posterior uncertainty in node log q"
        colorbar_label = "Posterior SD of log q"
        color_map = "magma"
    else:
        value_key = "node_log_q_mean"
        title = "Posterior mean of node log q"
        colorbar_label = "Posterior mean of log q"
        color_map = "viridis"

    node_values = np.asarray(
        surface[value_key].values,
        dtype=float,
    )

    if node_values.shape != (len(positions_m),):
        raise ValueError(
            f"surface['{value_key}'] must have shape "
            f"({len(positions_m)},)."
        )

    finite_nodes = np.isfinite(node_values)
    finite_values = node_values[finite_nodes]

    if finite_values.size == 0:
        raise ValueError("No finite node values to plot.")

    # 3. COLOR NORMALIZATION
    if vmin is None:
        vmin = 0.0 if posterior else float(finite_values.min())

    if vmax is None:
        vmax = float(finite_values.max())

        # Handle constant values.
        if vmax == vmin:
            vmax = vmin + max(abs(vmin), 1.0) * 1e-6

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("Color limits must be finite, with vmax > vmin.")

    color_norm = Normalize(vmin=vmin, vmax=vmax)

    if contour_levels < 2:
        raise ValueError("contour_levels must be at least 2.")

    levels = np.linspace(vmin, vmax, contour_levels)

    below = np.any(finite_values < vmin)
    above = np.any(finite_values > vmax)

    if below and above:
        colorbar_extend = "both"
    elif below:
        colorbar_extend = "min"
    elif above:
        colorbar_extend = "max"
    else:
        colorbar_extend = "neither"

    # 4. LOCAL AZIMUTHAL-EQUIDISTANT CRS
    center_lat = float(
        np.asarray(surface["projection_center_latitude"].values).item()
    )
    center_lon = float(
        np.asarray(surface["projection_center_longitude"].values).item()
    )

    local_crs = CRS.from_proj4(
        f"+proj=aeqd "
        f"+lat_0={center_lat} "
        f"+lon_0={center_lon} "
        f"+R=6371008.8 "
        f"+units=m "
        f"+no_defs"
    )

    # 5. INTERPOLATED SURFACE
    if show_contour_fill or show_contour_lines:
        grid_x, grid_y, grid_values = generalized_contour_surface(
            positions=positions_m,
            node_values=node_values,
            grid_size=grid_size,
            smoothing_km=smoothing_km,
        )

    # 6. FIGURE AND MAP BOUNDS
    fig, ax = plt.subplots(
        figsize=(12, 10),
        constrained_layout=True,
    )

    x_min, y_min = positions_m.min(axis=0)
    x_max, y_max = positions_m.max(axis=0)
    margin_m = map_margin_km * 1000.0

    ax.set_xlim(x_min - margin_m, x_max + margin_m)
    ax.set_ylim(y_min - margin_m, y_max + margin_m)

    # 7. BASEMAP
    if basemap_source is None:
        basemap_source = cx.providers.Esri.WorldPhysical

    cx.add_basemap(
        ax,
        crs=local_crs,
        source=basemap_source,
        zoom="auto",
        attribution_size=6,
        reset_extent=True,
        alpha=basemap_alpha,
        zorder=0,
    )

    # 8. FILLED CONTOURS
    if show_contour_fill:
        ax.contourf(
            grid_x,
            grid_y,
            grid_values,
            levels=levels,
            cmap=color_map,
            norm=color_norm,
            alpha=contour_fill_alpha,
            extend=colorbar_extend,
            antialiased=True,
            zorder=1,
        )

    # 9. CONTOUR LINES
    if show_contour_lines:
        ax.contour(
            grid_x,
            grid_y,
            grid_values,
            levels=levels,
            cmap=color_map,
            norm=color_norm,
            linewidths=contour_linewidth,
            alpha=contour_line_alpha,
            zorder=2,
        )

    # 10. NODES COLORED BY THEIR VALUES
    if show_nodes:
        ax.scatter(
            positions_m[finite_nodes, 0],
            positions_m[finite_nodes, 1],
            c=node_values[finite_nodes],
            cmap=color_map,
            norm=color_norm,
            s=node_size,
            edgecolor="#333333",
            linewidth=0.7,
            zorder=3,
        )

    # 11. OPTIONAL SAMPLE-NODE OVERLAY
    if show_sample_nodes:
        sample_count = np.asarray(
            surface["node_sample_count"].values,
        )

        occupied = sample_count > 0
        occupied_indices = np.flatnonzero(occupied)
        sample_colors = ["white"] * len(occupied_indices)
        pop_color_map = {}

        if sample_to_pop:
            populations = sorted(set(sample_to_pop.values()))

            population_cmap = mpl.colormaps["tab20"].resampled(
                len(populations)
            )
            pop_color_map = {
                pop: population_cmap(i)
                for i, pop in enumerate(populations)
            }

            node_populations = {}

            for sample_id, node_index in zip(
                surface["sample_id"].values,
                surface["sample_nodes"].values,
            ):
                if sample_id in sample_to_pop:
                    node_populations.setdefault(
                        int(node_index), []
                    ).append(sample_to_pop[sample_id])

            for i, node_index in enumerate(occupied_indices):
                populations_at_node = node_populations.get(
                    int(node_index), []
                )

                if populations_at_node:
                    majority = Counter(
                        populations_at_node
                    ).most_common(1)[0][0]

                    sample_colors[i] = pop_color_map[majority]

        ax.scatter(
            positions_m[occupied, 0],
            positions_m[occupied, 1],
            s=20.0 + 20.0 * sample_count[occupied],
            c=sample_colors,
            edgecolor="#333333",
            linewidth=1.0,
            zorder=4,
        )

        if pop_color_map:
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    label=population,
                    markerfacecolor=color,
                    markeredgecolor="#333333",
                    markersize=8,
                )
                for population, color in pop_color_map.items()
            ]

            ax.legend(
                handles=handles,
                title="Populations",
                loc="upper right",
                fontsize=12,
                title_fontsize=12,
                framealpha=0.9,
            )

    # 12. OPAQUE COLORBAR
    mappable = mpl.cm.ScalarMappable(
        norm=color_norm,
        cmap=color_map,
    )
    mappable.set_array([])

    cbar = fig.colorbar(
        mappable,
        ax=ax,
        fraction=0.025,
        pad=0.03,
        shrink=0.75,
        extend=colorbar_extend,
    )
    cbar.set_label(colorbar_label, fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    # 13. FORMATTING
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=18, pad=10)

    if show_axes:
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value / 1000:g}")
        )
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value / 1000:g}")
        )

        ax.set_xlabel("Easting (km)", fontsize=16)
        ax.set_ylabel("Northing (km)", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)

        ax.grid(
            color="#555555",
            linewidth=0.4,
            alpha=0.25,
        )
    else:
        ax.set_axis_off()

    if figure_path:
        fig.savefig(
            figure_path,
            bbox_inches="tight",
            dpi=300,
            facecolor="white",
        )

    plt.show()

    return fig, ax