import numpy as np
import networkx as nx
import random
import dreems
import xarray as xr

import sys
import os
import pickle
import dreems
from dreems.utils.utility import create_lattice, prepare_graph, create_networkx_graph_from_grid
from scipy.spatial import cKDTree

def main(snakemake):
    for f in snakemake.output:
        os.makedirs(os.path.dirname(f), exist_ok=True)

    file_path1 = snakemake.output[0]
    # output_folder = snakemake.params.output_folder
    # os.makedirs(os.path.dirname(output_folder), exist_ok=True)

    corridor_w=0.00005
    barrier_w=0.00001
    sample_size=10
    num_blocks=500 
    seed = 8
    ploidy=2

    rng = np.random.default_rng(12)
    all_nodes = np.arange(0, 96)
    subset = rng.choice(all_nodes, size=int(len(all_nodes) * 0.20), replace=False).tolist()
    
    graph, grid, edges, coord, collection, haplotype = create_lattice(n_rows=8, n_columns=12, q_anc=1e4, tau=5000000, corridor_w=corridor_w, barrier_w=barrier_w, barrier_startpt=2.5, barrier_endpt=8.5, sample_size=sample_size, anisotropy_scaler=1.0, q=1e4, num_blocks=num_blocks, mutation_rate=1e-7, ploidy=ploidy, subset=subset, exact=True, seed=seed)

    haplotypes = haplotype[np.max(haplotype, axis=1) <= 1].T
    print(f"Numer of snps: {haplotypes.shape[1]}")
    
    reshaped = haplotypes.reshape(int(haplotypes.shape[0] / ploidy), ploidy, haplotypes.shape[1])
    calls = reshaped.transpose(2, 0, 1)        

    sample_ids = np.asarray([f"sample_{index:03d}" for index in range(int(haplotypes.shape[0] / ploidy))])

    dreems_genotypes = xr.Dataset(
    data_vars={
            "call_genotype": (("variants", "samples", "ploidy"), calls),
            "sample_id": (("samples",), sample_ids),
        }
    )

    sample_longitude = coord[:,0]
    sample_latitude = coord[:,1]
    sample_coordinates = {
        sample_id: (longitude, latitude)
        for sample_id, latitude, longitude in zip(
            sample_ids,
            sample_latitude,
            sample_longitude,
            strict=True,
        )
    }

    dreem_genotypes = dreems.attach_sample_coordinates(dreems_genotypes, sample_coordinates)
    dreem_genotypes[["sample_latitude", "sample_longitude"]]

    data = dreems.from_dataset(
        dreem_genotypes,
        node_latitude=grid[:,1],
        node_longitude=grid[:,0],
        edges=edges-1,
        min_maf=0.05,
        min_variant_call_rate=0.8,
        attrs={"name": "quickstart"},
    )
    
    surface1 = dreems.infer(data)

    surface0 = surface1.copy(deep=True)
    weights_list = list(nx.get_edge_attributes(graph, 'w').values())
    new_data = [0.5 if x == 5e-05 else -0.5 if x == 1e-05 else x for x in weights_list]
    surface0["edge_log_migration_mean"].values = np.array(new_data)
    surface0["edge_log_migration_refined"].values = np.array(new_data)

    outer_polygon_coords = [
        (0.0,0.0),
        (0,6.06217783),
        (11.5, 6.06217783),
        (11.5, 0.0)
    ]
    
    grid_size = 2.0
    polygon_gdf, grid_gdf, minx, maxx, miny, maxy = prepare_graph(outer_polygon_coords, grid_size)
    G, sorted_vertices = create_networkx_graph_from_grid(grid_gdf)
    print(f"Total Number of Nodes: {G.number_of_nodes()}")
    grid = np.array([list(coordinate) for coordinate in nx.get_node_attributes(G, 'pos').values()])
    sample_nodes = np.asarray(cKDTree(grid).query(coord)[1], dtype=np.int32)  
    sample_ids = np.asarray([f"sample_{index:03d}" for index in range(int(len(coord)))])
    dreems_genotypes = xr.Dataset(
        data_vars={
                "call_genotype": (("variants", "samples", "ploidy"), calls),
                "sample_id": (("samples",), sample_ids),
            }
        )
    sample_longitude = coord[:,0]
    sample_latitude = coord[:,1]
    sample_coordinates = {
        sample_id: (longitude, latitude)
        for sample_id, latitude, longitude in zip(
            sample_ids,
            sample_latitude,
            sample_longitude,
            strict=True,
        )
    }
    dreem_genotypes = dreems.attach_sample_coordinates(dreems_genotypes, sample_coordinates)
    dreem_genotypes[["sample_latitude", "sample_longitude"]]
    data = dreems.from_dataset(
        dreem_genotypes,
        node_latitude=grid[:,1],
        node_longitude=grid[:,0],
        edges=np.array(list(G.edges)),
        min_maf=0.05,
        min_variant_call_rate=0.8,
        attrs={"name": "quickstart"},
    )
    surface2 = dreems.infer(data)

    outer_polygon_coords = [
        (0.0,0.0),
        (0,6.06217783),
        (11.5, 6.06217783),
        (11.5, 0.0)
    ]

    grid_size = 0.5
    polygon_gdf, grid_gdf, minx, maxx, miny, maxy = prepare_graph(outer_polygon_coords, grid_size)
    G, sorted_vertices = create_networkx_graph_from_grid(grid_gdf)
    print(f"Total Number of Nodes: {G.number_of_nodes()}")
    grid = np.array([list(coordinate) for coordinate in nx.get_node_attributes(G, 'pos').values()])
    sample_nodes = np.asarray(cKDTree(grid).query(coord)[1], dtype=np.int32)

    sample_ids = np.asarray([f"sample_{index:03d}" for index in range(int(len(coord)))])
    dreems_genotypes = xr.Dataset(
        data_vars={
                "call_genotype": (("variants", "samples", "ploidy"), calls),
                "sample_id": (("samples",), sample_ids),
            }
        )
    sample_longitude = coord[:,0]
    sample_latitude = coord[:,1]
    sample_coordinates = {
        sample_id: (longitude, latitude)
        for sample_id, latitude, longitude in zip(
            sample_ids,
            sample_latitude,
            sample_longitude,
            strict=True,
        )
    }
    
    dreem_genotypes = dreems.attach_sample_coordinates(dreems_genotypes, sample_coordinates)
    dreem_genotypes[["sample_latitude", "sample_longitude"]]
    
    data = dreems.from_dataset(
        dreem_genotypes,
        node_latitude=grid[:,1],
        node_longitude=grid[:,0],
        edges=np.array(list(G.edges)),
        min_maf=0.05,
        min_variant_call_rate=0.8,
        attrs={"name": "quickstart"},
    )
    surface3 = dreems.infer(data)

    outer_polygon_coords = [
        (0.0,0.0),
        (0,6.06217783),
        (11.5, 6.06217783),
        (11.5, 0.0)
    ]
    
    grid_size = 0.25
    polygon_gdf, grid_gdf, minx, maxx, miny, maxy = prepare_graph(outer_polygon_coords, grid_size)
    G, sorted_vertices = create_networkx_graph_from_grid(grid_gdf)
    print(f"Total Number of Nodes: {G.number_of_nodes()}")
    grid = np.array([list(coordinate) for coordinate in nx.get_node_attributes(G, 'pos').values()])
    sample_nodes = np.asarray(cKDTree(grid).query(coord)[1], dtype=np.int32)
    
    sample_ids = np.asarray([f"sample_{index:03d}" for index in range(int(len(coord)))])
    dreems_genotypes = xr.Dataset(
        data_vars={
                "call_genotype": (("variants", "samples", "ploidy"), calls),
                "sample_id": (("samples",), sample_ids),
            }
        )
    sample_longitude = coord[:,0]
    sample_latitude = coord[:,1]
    sample_coordinates = {
        sample_id: (longitude, latitude)
        for sample_id, latitude, longitude in zip(
            sample_ids,
            sample_latitude,
            sample_longitude,
            strict=True,
        )
    }
    dreem_genotypes = dreems.attach_sample_coordinates(dreems_genotypes, sample_coordinates)
    dreem_genotypes[["sample_latitude", "sample_longitude"]]
    data = dreems.from_dataset(
        dreem_genotypes,
        node_latitude=grid[:,1],
        node_longitude=grid[:,0],
        edges=np.array(list(G.edges)),
        min_maf=0.05,
        min_variant_call_rate=0.8,
        attrs={"name": "quickstart"},
    )
    surface4 = dreems.infer(data)
    with open(file_path1, "wb") as f:
        pickle.dump((surface0, surface1, surface2, surface3, surface4), f)
    
    return
        
if __name__ == "__main__":
    main(snakemake)
