import sys
import os
from scipy.spatial import cKDTree
import numpy as np
import pickle
import networkx as nx
from shapely.geometry import Polygon, Point
import pandas as pd
import geopandas as gpd
from itertools import combinations
from feems import SpatialGraph, Viz
import matplotlib.pyplot as plt
from feems.cross_validation import run_cv, run_cv_joint
import time

def plot_samples(sample_coords, groups=None, pad=5):
    world = gpd.read_file("/home/jkliang/ne_110m_admin_0_countries_lakes/ne_110m_admin_0_countries_lakes.shp")

    fig, ax = plt.subplots(figsize=(30, 15))

    df = pd.DataFrame(sample_coords, columns=["longitude", "latitude"])

    minx, maxx = df["latitude"].min(), df["latitude"].max()
    miny, maxy = df["longitude"].min(), df["longitude"].max()

    # Plot the world map (countries with borders)
    world.plot(ax=ax, color='white', edgecolor='black')

    # Unpack coordinates into latitudes and longitudes
    lats, lons = zip(*sample_coords)

    ax.set_xlim(minx-pad, maxx+pad)
    ax.set_ylim(miny-pad, maxy+pad)

    # If no groups are provided, default to a single group (1)
    if groups is None:
        groups = [1] * len(sample_coords)

    # Get unique groups and assign a color to each
    unique_groups = sorted(np.unique(groups))
    num_groups = len(unique_groups)
    
    cmap = plt.colormaps['viridis'].resampled(num_groups)
    group_colors = {group: cmap(i) for i, group in enumerate(unique_groups)}

    # Map each point to its assigned color
    point_colors = [group_colors[g] for g in groups]

    # Plot the points (scatter plot)
    ax.scatter(
        lons,                      # Longitudes (x-axis)
        lats,                      # Latitudes (y-axis)
        c=point_colors,               # Point color
        s=100,                     # Point size
        label='Sample Points',     # Legend label
        zorder=2                   # Ensure points appear above the map
    )

    # Create a legend for the groups
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=f'Group {group}',
                   markerfacecolor=color, markersize=15)
        for group, color in group_colors.items()
    ]
    ax.legend(handles=legend_elements, fontsize=20, title="Groupings")
    
    # Customize the plot
    ax.set_title("World Map with Coordinates", fontsize=30)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.show()

# Function to generate equilateral triangles for the grid
def generate_triangles(minx, miny, maxx, maxy, grid_size):
    # You need this specific height with multiplier sqrt(3)/2 to make equilateral triangles
    height = grid_size * np.sqrt(3) / 2
    # recording all the triangles here to construct the grid
    triangles = []
    # Storing all the vertices because later on we will see use it to find
    # the closest deme/vertex for each sample coordinate much easier
    vertices = set()

    # Shift the starting point to align the vertices
    ### Look into why this works...
    x_start = minx - (minx % (grid_size / 2))
    y_start = miny - (miny % height)
    
    # The point of j is to count even and odd rows because we need to offset
    # triangles to make sure they line up correctly. If you are on an even row
    # what happens is that you shift all triangles left by grid_size / 2.
    # triangle1 is drawn in order of lower left, right corner, and then up corner
    # triangle2 shares the right lower corner of triangle1, then goes up right side, and then
    # draws left to meet at triangle1's up corner. Next iteration starts
    # at triangle 2 bottom corner
    #   .    .   .       .   .
    # .   . +  .    =  .   .
    j = 0
    offset = grid_size / 2
    # Hold y or horizontal levels and build triangles horizontonally level by level
    for y in np.arange(y_start, maxy, height):
        j = j + 1
        # even or odd horizontal level are offset/shifted slightly
        if j % 2 == 0:
            # shift all triangle to the left and increase right most bound
            for x in np.arange(x_start - offset, maxx + grid_size, grid_size):
                triangle1 = Polygon([(x, y), (x + grid_size, y), (x + grid_size / 2, y + height)])
                triangle2 = Polygon([(x + grid_size, y), (x + 1.5 * grid_size, y + height), (x + grid_size / 2, y + height)])
                triangles.extend([triangle1, triangle2])
                vertices.update([(x, y), (x + grid_size, y), (x + grid_size / 2, y + height), 
                                 (x + grid_size, y), (x + 1.5 * grid_size, y + height), (x + grid_size / 2, y + height)])
        else:
            for x in np.arange(x_start, maxx, grid_size):
                triangle1 = Polygon([(x, y), (x + grid_size, y), (x + grid_size / 2, y + height)])
                triangle2 = Polygon([(x + grid_size, y), (x + 1.5 * grid_size, y + height), (x + grid_size / 2, y + height)])
                triangles.extend([triangle1, triangle2])
                vertices.update([(x, y), (x + grid_size, y), (x + grid_size / 2, y + height), 
                                 (x + grid_size, y), (x + 1.5 * grid_size, y + height), (x + grid_size / 2, y + height)])

    return triangles, vertices

# Generates all the necessary graph data frames
def prepare_graph(polygon_coords, grid_size): 
    # Create a polygon from the given outer coordinates
    polygon = Polygon(polygon_coords)

    # Get the bounding box of the polygon
    minx, miny, maxx, maxy = polygon.bounds
    
    # Create a GeoDataFrame for the outer polygon
    polygon_gdf = gpd.GeoDataFrame([polygon], columns=['geometry'])
    
    # Generate grid cells within the bounding outer polygon
    triangles, vertices = generate_triangles(minx, miny, maxx, maxy, grid_size)
    
    # Filter triangles that intersect with the polygon
    grid_cells = [tri for tri in triangles if tri.intersects(polygon)]
    
    # Create a GeoDataFrame for the grid cells
    grid_gdf = gpd.GeoDataFrame(grid_cells, columns=['geometry'])
        
    ### Fix, I do not think we need minx, maxx, miny, maxy but for convenience we leave it
    return polygon_gdf, grid_gdf, minx, maxx, miny, maxy

def create_networkx_graph_from_grid(grid_gdf):
    # Step 1: Extract all unique vertices from the triangular grid cells
    all_vertices = []
    for geometry in grid_gdf['geometry']:
        if geometry.geom_type == 'Polygon':
            coords = list(geometry.exterior.coords[:-1])  # Exclude repeated closing point
            all_vertices.extend(coords)
    
    # Deduplicate and sort vertices by y first, then x
    unique_vertices = sorted(
        {(round(x, 4), round(y, 4)) for x, y in all_vertices},
        key=lambda coord: (coord[1], coord[0])  # Sort by y, then x
    )
    
    # Step 2: Create a networkx graph
    G = nx.Graph()
    
    # Add nodes with sorted vertices
    for i, (x, y) in enumerate(unique_vertices):
        G.add_node(i, pos=(x, y))
    
    # Add edges based on shared triangle sides
    for geometry in grid_gdf['geometry']:
        if geometry.geom_type == 'Polygon':
            coords = list(geometry.exterior.coords[:-1])
       
        # Round coordinates to match unique_vertices
        rounded_coords = [(round(x, 4), round(y, 4)) for x, y in coords]
        
        # Find node indices (using a coord-to-index map for efficiency)
        coord_to_index = {coord: i for i, coord in enumerate(unique_vertices)}
        node_indices = [coord_to_index[coord] for coord in rounded_coords]
        
        # Add edges between all pairs of nodes in this triangle
        for u, v in combinations(node_indices, 2):
            G.add_edge(u, v)
    
    return G, unique_vertices

# This function will draw the grid on top of the map
def draw_grid(polygon_coords, grid_size, pad=10):
    ### FIX: The only reason I need minx, maxx, miny, maxy is for plotting, I should find a way to not need them
    polygon_gdf, grid_gdf, minx, maxx, miny, maxy  = prepare_graph(polygon_coords, grid_size)
    # Load the world map
    ### FIX: Need to change the fact that I am loading a local download file
    ### FIX: We can try to find a better looking map
    world = gpd.read_file("/home/jkliang/ne_110m_admin_0_countries_lakes/ne_110m_admin_0_countries_lakes.shp")
    
    # Plot the polygon, grid, sample points, and nearest corners
    plt.figure(figsize=(50, 40)) 
    ax = world.plot(color='white', edgecolor='black')
    # padding is for zooming out the map a bit and not surrounding the outer polygon immediately
    # pad = 10
    # forcing the figure to shift and limit itself to view the outer polygon
    ax.set_xlim(minx-pad, maxx+pad)
    ax.set_ylim(miny-pad, maxy+pad)
    # figure decorations
    polygon_gdf.boundary.plot(ax=ax, edgecolor='blue')
    grid_gdf.boundary.plot(ax=ax, edgecolor='red', linestyle='-', alpha=0.5)
    
    # plt.legend()
    plt.tight_layout()
    plt.show()

def main(snakemake):
    # Get the output file and seed from Snakemake
    for f in snakemake.output:
        os.makedirs(os.path.dirname(f), exist_ok=True)
        
    data_path = snakemake.input[0]
    coord_path = snakemake.input[1]
    boundary_path = snakemake.input[2]
    # output_figure_path = snakemake.output[0]
    # posterior_output_figure_path = snakemake.output[1]
    # fishes_timing_path = snakemake.output[2]
    # fishes_output_path = snakemake.output[3]
    timing_path = snakemake.output[0]
    output_path = snakemake.output[1]

    with open(data_path, "rb") as f:
        genotypes = pickle.load(f)
    
    coord = np.loadtxt(coord_path)  # sample coordinates
    outer = np.loadtxt(boundary_path)

    grid_size = float(snakemake.params.grid_size)
    polygon_gdf, grid_gdf, minx, maxx, miny, maxy = prepare_graph(outer, grid_size)
    G, sorted_vertices = create_networkx_graph_from_grid(grid_gdf)

    grid = np.array([list(coord) for coord in nx.get_node_attributes(G, 'pos').values()])
    edges = np.array(list(G.edges)) + 1

    sp_graph = SpatialGraph(genotypes, coord, grid, edges, scale_snps=True)
    
    lamb_grid = np.geomspace(1e-3, 1e2, 10, endpoint=True)[::-1]
    lamb_q_grid = np.geomspace(1e-2, 1e2, 10, endpoint=True)[::-1]
    
    # run cross-validation over both smoothing parameters
    start = time.perf_counter()
    cv_err = run_cv_joint(sp_graph, lamb_grid, lamb_q_grid)
    end = time.perf_counter()
    elapsed_time = end - start
    
    # average over folds
    mean_cv_err = np.mean(cv_err, axis=0)

    min_idx = np.nanargmin(mean_cv_err)  # flat index
    row, col = np.unravel_index(min_idx, mean_cv_err.shape)
    
    # Retrieve the corresponding hyperparameters
    lamb_q_cv = lamb_q_grid[row]
    lamb_cv   = lamb_grid[col]

    with open(timing_path, "wb") as f:
        pickle.dump(elapsed_time, f)

    with open(output_path, "wb") as f:
        pickle.dump((lamb_q_cv, lamb_cv, mean_cv_err, genotypes, coord, grid, edges), f)
    
    return
        
if __name__ == "__main__":
    main(snakemake)
