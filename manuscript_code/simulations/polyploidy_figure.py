import string
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import sys
import os
import pickle

def plot_figure(surfaces, filepath):
    # Custom labels for each of the 4 subplots (adjust as needed)
    titles = [
        f"Dense Sampling (Heterogeneous)",
        f"Sparse Sampling (Heterogeneous)",
        f"Dense Sampling (Anisotropy)",
        f"Sparse Sampling (Anisotropy)"
    ]
    
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(
        2, 2, figsize=(8, 8), sharex=True, sharey=True, layout="compressed"
    )
    
    panel_letters = list(string.ascii_uppercase)
    migration_lines_list = []
    
    # Loop over the 2x2 grid (4 total panels)
    for idx, surface in enumerate(surfaces):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        letter = panel_letters[idx]  # A, B, C, D
    
        # Extract properties from the surface dictionary
        positions = surface["node_position"].values
        edge_nodes = surface["edge_node"].values
        segments = positions[edge_nodes]
        # edge_log_migration = surface["edge_log_migration_mean"].values
        edge_log_migration = surface["edge_log_migration_refined"].values
        sample_count = surface["node_sample_count"].values
    
        # Calculate dynamic dynamic symmetric colormap bounds [-limit, limit]
        limit = max(float(np.max(np.abs(edge_log_migration))), 0.1)
        vmin, vmax = -limit, limit
    
        # Draw edge network
        lines = LineCollection(
            segments,
            cmap="RdBu",
            norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
            linewidths=1.8,
        )
        lines.set_array(edge_log_migration)
        ax.add_collection(lines)
        migration_lines_list.append(lines)
    
        # Plot sampled nodes
        has_samples = sample_count > 0
        if np.any(has_samples):
            ax.scatter(
                positions[has_samples, 0],
                positions[has_samples, 1],
                s=8.0,
                facecolor="white",
                edgecolor="black",
                linewidth=0.6,
                zorder=3,
            )
    
        # Clean up individual axes
        ax.autoscale()
        ax.set_aspect("equal")
        ax.axis("off")
    
        # Set individual subplot title
        ax.set_title(titles[idx], fontsize=11, pad=6)
    
        # Add Panel Letter (Top-Left corner)
        ax.text(
            0.02,
            0.96,
            letter,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
        )
    
    # Shared Horizontal Colorbar at Bottom
    cbar = fig.colorbar(
        migration_lines_list[-1],
        ax=axes,
        orientation="horizontal",
        shrink=0.5,
        pad=0.03,
        aspect=20,
    )
    cbar.set_label("log10(w)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()

    return

def main(snakemake):
    # Get the output file and seed from Snakemake
    for f in snakemake.output:
        os.makedirs(os.path.dirname(f), exist_ok=True)

    filepath1 = snakemake.input[0]
    filepath2 = snakemake.input[1]
    output_filepath1 = snakemake.output[0]
    output_filepath2 = snakemake.output[1]

    with open(filepath1, "rb") as f:
        surface1, surface2, surface3, surface4 = pickle.load(f)
        
    surfaces = [surface1, surface2, surface3, surface4]

    plot_figure(surfaces, output_filepath1)

    with open(filepath2, "rb") as f:
        surface1, surface2, surface3, surface4 = pickle.load(f)
        
    surfaces = [surface1, surface2, surface3, surface4]

    plot_figure(surfaces, output_filepath2)
    
    return
        
if __name__ == "__main__":
    main(snakemake)