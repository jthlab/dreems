from scipy.spatial import cKDTree
import numpy as np
import scipy.sparse as sp

import logging
import warnings
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
import rasterio
from rasterio.plot import show
import dreems

import time
import branca.colormap as cm
import folium

import pandas as pd
from shapely.geometry import Polygon, Point
import networkx as nx
from itertools import combinations
import xarray as xr

import msprime as msp
import random
import jax.experimental.sparse as jesp
import itertools as it
import tskit

from dreems.data.genotypes import _node_covariance, genotypes_create_dataset, standardized_genotypes_matrix

def create_lattice(n_rows=3, n_columns=4, q_anc=1e4, tau=5000000, Ne=1e4, corridor_w=0.1, barrier_w=0.01,
                       barrier_startpt=2.5, barrier_endpt=8.5, sample_size=1, anisotropy_scaler=1.0, q=1e4, num_blocks=1000, 
                   mutation_rate=1e-7, ploidy=2, seed=8, exact=False, subset=None):
        
    graph = nx.generators.lattice.triangular_lattice_graph(
        n_rows - 1, 2 * n_columns - 2, with_positions=True
    )
    graph = nx.convert_node_labels_to_integers(graph)
    positions = nx.get_node_attributes(graph, "pos")
    grid = np.array(list(positions.values()))
    edges = np.array(list(graph.edges))
    edges += 1  # 1 indexed nodes for feems
    
    demo = msp.Demography()
    demo.add_population(initial_size = q_anc, name = "anc")

    total_nodes = n_rows * n_columns
    edge_adj = jesp.BCOO.from_scipy_sparse(sp.triu(nx.adjacency_matrix(graph, np.arange(total_nodes)).astype(float)))


    for node in graph.nodes():
        pop_name = f"P{node}"
        demo.add_population(initial_size= q, name=pop_name)
        # sample a node or not
        graph.nodes[node]["sample_size"] = sample_size * 2

    tmp = [f"P{i}" for i in range(n_rows*n_columns)]
    demo.add_population_split(time = tau, derived=tmp, ancestral="anc")

    for edge in np.array(edge_adj.indices):
        i = edge[0]
        j = edge[1]
        x = np.mean([graph.nodes[i]["pos"][0], graph.nodes[j]["pos"][0]])
        y = np.mean([graph.nodes[i]["pos"][1], graph.nodes[j]["pos"][1]])
        left_neighbor = f"P{i}"
        right_neighbor = f"P{j}"

        if x <= barrier_startpt:
            graph[i][j]["w"] = corridor_w
            demo.set_symmetric_migration_rate(
                    populations=(left_neighbor, right_neighbor), rate=corridor_w)
        elif x >= barrier_endpt:
            graph[i][j]["w"] = corridor_w
            demo.set_symmetric_migration_rate(
                    populations=(left_neighbor, right_neighbor), rate=corridor_w)
        else:
            graph[i][j]["w"] = barrier_w
            demo.set_symmetric_migration_rate(
                    populations=(left_neighbor, right_neighbor), rate= barrier_w)
        
        if graph.nodes[i]["pos"][1] == graph.nodes[j]["pos"][1]:
            graph[i][j]["w"] = anisotropy_scaler * graph[i][j]["w"]
            demo.set_symmetric_migration_rate(
                populations=(left_neighbor, right_neighbor), rate=graph[i][j]["w"])

    # Set up sample sizes for each population using prefixed identifiers
    samples = {f"P{i}": sample_size for i in range(n_rows * n_columns)}

    if subset:
        keep_indices = set(subset)

        for key in list(samples.keys()):
            # Extract the integer after 'P'
            idx = int(key[1:])       
            if idx not in keep_indices:
                samples[key] = 0
                graph.nodes[idx]["sample_size"] = 0

    collection = []
    haplotype = []
    counter = 0
    seed_num = 1

    if exact:
        while counter < num_blocks:
            anc = msp.sim_ancestry(samples=samples, demography=demo, recombination_rate=0, sequence_length=1, random_seed = seed_num*seed*7, ploidy=ploidy)
            ts = msp.sim_mutations(anc, rate=mutation_rate, random_seed = seed_num*seed*2, model=msp.BinaryMutationModel())
            seed_num += 1
            if len(ts.genotype_matrix()) > 0 and 0.05 <= np.array(ts.genotype_matrix()[0]).mean()  <= 0.95:
                collection.append(ts)
                haplotype.append(ts.genotype_matrix()[0])
                counter += 1
    else:
        for i in range(1, num_blocks+1):
            anc = msp.sim_ancestry(samples=samples, demography=demo, recombination_rate=0, sequence_length=1, random_seed = seed*i*7, ploidy=ploidy)
            ts = msp.sim_mutations(anc, rate=mutation_rate, random_seed = seed*i*2, model=msp.BinaryMutationModel())
            if len(ts.genotype_matrix()) > 0 and 0.05 <= np.array(ts.genotype_matrix()[0]).mean()  <= 0.95: # and not np.all(np.array(ts.genotype_matrix()[0]) == 0):
                collection.append(ts)
                haplotype.append(ts.genotype_matrix()[0])

    sample_sizes_dict = nx.get_node_attributes(graph, "sample_size")
    pops = [[i] * int(sample_sizes_dict[i] / 2) for i in graph.nodes]
    pops = list(it.chain.from_iterable(pops))
    coord = grid[pops, :]

    return graph, grid, edges, coord, collection, np.array(haplotype)

def from_genotypes(genotypes, sample_coordinates, nodes, edges):
    coord = np.array(list(sample_coordinates.values()), dtype=float)
    sample_nodes = np.asarray(cKDTree(nodes).query(coord)[1], dtype=np.int32)
    node_sample_count = np.bincount(
        sample_nodes,
        minlength=nodes.shape[0],
    ).astype(np.int64)
    observed_nodes = np.flatnonzero(node_sample_count > 0).astype(np.int32)
    if observed_nodes.size < 2:
        raise ValueError("samples map to fewer than two graph nodes")

    standardized, effective_snps = standardized_genotypes_matrix(genotypes)
    covariance = _node_covariance(
        standardized,
        sample_nodes=sample_nodes,
        node_sample_count=node_sample_count,
        observed_nodes=observed_nodes,
    )
    
    direct_data = genotypes_create_dataset(
        sample_coordinates=sample_coordinates,
        node_positions=nodes,
        edges=edges,
        node_covariance=covariance,
        observed_nodes=observed_nodes,
        node_sample_count=node_sample_count,
        effective_snps=effective_snps,
        coordinate_names=("easting", "northing"),
        attrs={"name": "quickstart-direct-covariance"},
        sample_nodes = sample_nodes,
    )

    return direct_data

def dreems_infer_genotypes(genotypes, coord, nodes, edges):
    sample_ids = np.asarray([f"sample_{index}" for index in range(genotypes.shape[0])])
    sample_coordinates = {
        sample_id: (latitude, longitude)
        for sample_id, latitude, longitude in zip(
            sample_ids,
            coord[:,0],
            coord[:,1],
            strict=True,
        )
    }
    
    dreems_data = from_genotypes(genotypes, sample_coordinates, nodes, edges)

    surface = dreems.infer(dreems_data, refine=True)
    return surface

def dreems_infer_genotypes_timing(genotypes, coord, nodes, edges):
    sample_ids = np.asarray([f"sample_{index}" for index in range(genotypes.shape[0])])
    sample_coordinates = {
        sample_id: (latitude, longitude)
        for sample_id, latitude, longitude in zip(
            sample_ids,
            coord[:,0],
            coord[:,1],
            strict=True,
        )
    }
    
    dreems_data = from_genotypes(genotypes, sample_coordinates, nodes, edges)

    start_time = time.perf_counter()
    surface = dreems.infer(dreems_data, refine=True)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    return surface, elapsed_time

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

def create_grid(polygon_coords, grid_size, pad=10):
    ### FIX: The only reason I need minx, maxx, miny, maxy is for plotting, I should find a way to not need them
    polygon_gdf, grid_gdf, minx, maxx, miny, maxy  = prepare_graph(polygon_coords, grid_size)
    G, sorted_vertices = create_networkx_graph_from_grid(grid_gdf)
    
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {len(G.edges)}")
    
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

    ax.set_xlabel("Longitude", fontsize=18)
    ax.set_ylabel("Latitude", fontsize=18)
    
    # plt.legend()
    plt.tight_layout()
    plt.show()
    return G

def dreems_infer(data, sample_coordinates, nodes, edges):
    dreems_data = dreems.from_dataset(
        data,
        sample_coordinates = sample_coordinates,
        node_latitude=nodes[:,1],
        node_longitude=nodes[:,0],
        edges=edges,
        min_maf=0.05,
        min_variant_call_rate=0.8,
        attrs={"name": "quickstart"},
    )

    surface = dreems.infer(dreems_data, refine=True)
    return surface

def dreems_infer_timing(data, sample_coordinates, nodes, edges):
    dreems_data = dreems.from_dataset(
        data,
        sample_coordinates = sample_coordinates,
        node_latitude=nodes[:,1],
        node_longitude=nodes[:,0],
        edges=edges,
        min_maf=0.05,
        min_variant_call_rate=0.8,
        attrs={"name": "quickstart"},
    )

    start_time = time.perf_counter()
    surface = dreems.infer(dreems_data, refine=True)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    return surface, elapsed_time