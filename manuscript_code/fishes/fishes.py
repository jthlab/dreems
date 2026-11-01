import sys
import os
from scipy.spatial import cKDTree
import numpy as np
import pickle
import dreems
from dreems.utils.utility import dreems_infer_timing, create_grid
from dreems.utils.snakemake_plot import draw_figure_map
import networkx as nx
import xarray as xr
import time

def main(snakemake):
    # Get the output file and seed from Snakemake
    for f in snakemake.output:
        os.makedirs(os.path.dirname(f), exist_ok=True)
        
    fishes_data_path = snakemake.input[0]
    fishes_coord_path = snakemake.input[1]
    fishes_boundary_path = snakemake.input[2]
    output_figure_path = snakemake.output[0]
    posterior_output_figure_path = snakemake.output[1]
    fishes_timing_path = snakemake.output[2]
    
    ds = xr.open_zarr(fishes_data_path)
    coord = np.loadtxt(fishes_coord_path)  # sample coordinates
    outer = np.loadtxt(fishes_boundary_path)

    grid_size = float(snakemake.params.grid_size)
    G = create_grid(outer, grid_size=grid_size, pad = 0.01)
    nodes = np.array([list(coord) for coord in nx.get_node_attributes(G, 'pos').values()])
    edges = np.array(list(G.edges))

    def assign_population(sample):
        if "BUFF" in sample or "DUCK" in sample:
            return "Allohistium anas"
        if sample.startswith("Amay"):
            return "Allohistium maydeni"
        return "Allohistium cinereum"
    
    sample_ids = ds['sample_id'].values
    pop_map = {sample: assign_population(sample) for sample in sample_ids}

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

    surface, elapsed_time = dreems_infer_timing(ds, sample_coordinates, nodes, edges)
    
    fig, ax = draw_figure_map(
        surface,
        sample_to_pop=pop_map,
        smoothing_sigma=5.0,
        contour_levels=14,
        contour_linewidth=2.0,
        contour_fill_alpha=0.20,
        figure_path=output_figure_path,
        posterior=False
    )

    fig, ax = draw_figure_map(
        surface,
        sample_to_pop=pop_map,
        smoothing_sigma=5.0,
        contour_levels=14,
        contour_linewidth=2.0,
        contour_fill_alpha=0.20,
        figure_path=posterior_output_figure_path,
        posterior=True
    )

    with open(fishes_timing_path, "wb") as f:
        pickle.dump(elapsed_time, f)
    
    return
        
if __name__ == "__main__":
    main(snakemake)
