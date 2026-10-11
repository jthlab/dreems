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

def main(snakemake):
    for f in snakemake.output:
        os.makedirs(os.path.dirname(f), exist_ok=True)

    file_path1 = snakemake.output[0]
    # output_folder = snakemake.params.output_folder
    # os.makedirs(os.path.dirname(output_folder), exist_ok=True)

    corridor_w=0.00005
    barrier_w=0.00001
    sample_size=5
    num_blocks=200 
    seed = 10
    subset=None
    ploidy=4
    
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

    rng = np.random.default_rng(2)
    n_samples_per_node = 5
    all_nodes = np.arange(0, 96)
    arr = rng.choice(all_nodes, size=int(len(all_nodes) * 0.20), replace=False).tolist()
    graph, grid, edges, coord, collection, haplotype = create_lattice(n_rows=8, n_columns=12, q_anc=1e4, tau=5000000, corridor_w=corridor_w, barrier_w=barrier_w, barrier_startpt=2.5, barrier_endpt=8.5, sample_size=sample_size, anisotropy_scaler=1.0, q=1e4, num_blocks=num_blocks, mutation_rate=1e-7, ploidy=ploidy, subset=arr, exact=True, seed=seed)

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
    
    surface2 = dreems.infer(data)

    corridor_w=0.00001
    barrier_w=0.00001
    sample_size=5
    num_blocks=200 
    seed = 20
    subset=None
    ploidy=4
    
    graph, grid, edges, coord, collection, haplotype = create_lattice(n_rows=8, n_columns=12, q_anc=1e4, tau=5000000, corridor_w=corridor_w, barrier_w=barrier_w, barrier_startpt=2.5, barrier_endpt=8.5, sample_size=sample_size, anisotropy_scaler=5.0, q=1e4, num_blocks=num_blocks, mutation_rate=1e-7, ploidy=ploidy, subset=subset, exact=True, seed=seed)

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
    
    surface3 = dreems.infer(data)

    rng = np.random.default_rng(2)
    n_samples_per_node = 5
    all_nodes = np.arange(0, 96)
    arr = rng.choice(all_nodes, size=int(len(all_nodes) * 0.20), replace=False).tolist()
    graph, grid, edges, coord, collection, haplotype = create_lattice(n_rows=8, n_columns=12, q_anc=1e4, tau=5000000, corridor_w=corridor_w, barrier_w=barrier_w, barrier_startpt=2.5, barrier_endpt=8.5, sample_size=sample_size, anisotropy_scaler=5.0, q=1e4, num_blocks=num_blocks, mutation_rate=1e-7, ploidy=ploidy, subset=arr, exact=True, seed=seed)

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
    
    surface4 = dreems.infer(data)

    with open(file_path1, "wb") as f:
        pickle.dump((surface1, surface2, surface3, surface4), f)
    
    return
        
if __name__ == "__main__":
    main(snakemake)
