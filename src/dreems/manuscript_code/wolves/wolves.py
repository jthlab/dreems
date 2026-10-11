import sys
import os
import numpy as np
import pickle
import dreems
from dreems.utils.utility import dreems_infer_genotypes_timing
from dreems.utils.snakemake_plot import draw_figure_map

def main(snakemake):
    # Get the output file and seed from Snakemake
    for f in snakemake.output:
        os.makedirs(os.path.dirname(f), exist_ok=True)
        
    wolves_data_path = snakemake.input[0]
    output_figure_path = snakemake.output[0]
    posterior_output_figure_path = snakemake.output[1]
    wolves_timing_path = snakemake.output[2]

    with open(wolves_data_path, "rb") as f:
        loaded = pickle.load(f)
    
    genotypes = loaded["genotypes"]
    edges = loaded["edges"]
    grid = loaded["grid"]
    coord = loaded["coord"]

    surface, elapsed_time = dreems_infer_genotypes_timing(genotypes, coord, grid, edges-1)

    fig, ax = draw_figure_map(
        surface,
        sample_to_pop=None,
        smoothing_sigma=5.0,
        contour_levels=14,
        contour_linewidth=2.0,
        contour_fill_alpha=0.20,
        figure_path=output_figure_path,
        posterior=False
    )

    fig, ax = draw_figure_map(
        surface,
        sample_to_pop=None,
        smoothing_sigma=5.0,
        contour_levels=14,
        contour_linewidth=2.0,
        contour_fill_alpha=0.20,
        figure_path=posterior_output_figure_path,
        posterior=True
    )

    with open(wolves_timing_path, "wb") as f:
        pickle.dump(elapsed_time, f)
    
    return
        
if __name__ == "__main__":
    main(snakemake)
