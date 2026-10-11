import sys
import os
from scipy.spatial import cKDTree
import numpy as np
import pickle
from dreems.utils.utility import dreems_infer_timing, create_grid
from dreems.utils.snakemake_plot import draw_figure_map
import pandas as pd
import networkx as nx
import xarray as xr

def main(snakemake):
    # Get the output file and seed from Snakemake
    for f in snakemake.output:
        os.makedirs(os.path.dirname(f), exist_ok=True)
        
    data_path = snakemake.input[0]
    coord_path = snakemake.input[1]
    boundary_path = snakemake.input[2]
    output_figure_path = snakemake.output[0]
    posterior_output_figure_path = snakemake.output[1]
    timing_path = snakemake.output[2]

    ds = xr.open_zarr(data_path)
    outer = np.loadtxt(boundary_path)
    df = pd.read_excel(coord_path)

    id_to_region = dict(zip(df["id"], df["region"]))
    
    prefixes_to_update = ("BARA", "JWWE", "FAYE")
    
    pop_map = {
        sample_id: (
            "UP-MI" if sample_id.startswith(prefixes_to_update)
            else "LP-MI"
        ) if region == "MI" else region
        for sample_id, region in id_to_region.items()
    }
    
    sample_coordinates = {
        row["id"]: (row["long"], row["lat"])
        for _, row in df.iterrows()
    }

    grid_size = float(snakemake.params.grid_size)
    G = create_grid(outer, grid_size=grid_size, pad = 0.01)
    nodes = np.array([list(coord) for coord in nx.get_node_attributes(G, 'pos').values()])
    edges = np.array(list(G.edges))

    surface, elapsed_time = dreems_infer_timing(ds, sample_coordinates, nodes, edges)
    
    fig, ax = draw_figure_map(
        surface,
        sample_to_pop=pop_map,
        smoothing_sigma=5.0,
        contour_levels=14,
        contour_linewidth=2.0,
        contour_fill_alpha=0.20,
        figure_path=output_figure_path,
        posterior=False,
        location="upper right",
    )

    fig, ax = draw_figure_map(
        surface,
        sample_to_pop=pop_map,
        smoothing_sigma=5.0,
        contour_levels=14,
        contour_linewidth=2.0,
        contour_fill_alpha=0.20,
        figure_path=posterior_output_figure_path,
        posterior=True,
        location="upper right",
    )

    with open(timing_path, "wb") as f:
        pickle.dump(elapsed_time, f)
    
    return
        
if __name__ == "__main__":
    main(snakemake)
