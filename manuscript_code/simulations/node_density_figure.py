import string
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import sys
import os
import pickle

def plot_figure(surfaces, filepath):
    titles = [
        f"{len(surfaces[i]['node_position'].values)} Nodes" + (" (True Surface)" if i == 0 else "")
        for i in range(5)
    ]

    fig = plt.figure(figsize=(8, 10))  # Slightly decreased height from 11 to 10 to reduce ambient white space

    # --- NESTED LAYOUT SETUP ---
    # 1. hspace controls vertical space between Panel A and the bottom 2x2 grid.
    #    Making hspace negative (e.g., -0.25 or -0.3) pulls the bottom block up.
    outer_gs = fig.add_gridspec(2, 1, height_ratios=[0.6, 3.0], hspace=-0.9)

    # Top Block: Spans full width for panel A
    top_gs = outer_gs[0].subgridspec(1, 1)
    ax_top = fig.add_subplot(top_gs[0, 0])

    # 2. hspace controls vertical space between Row 2 (B, C) and Row 3 (D, E).
    #    Make this more negative (e.g., -0.55 or -0.6) to reduce white space even further.
    bottom_gs = outer_gs[1].subgridspec(2, 2, hspace=-0.7, wspace=0.15)

    axes = [
        ax_top,
        fig.add_subplot(bottom_gs[0, 0]),
        fig.add_subplot(bottom_gs[0, 1]),
        fig.add_subplot(bottom_gs[1, 0]),
        fig.add_subplot(bottom_gs[1, 1]),
    ]

    panel_letters = list(string.ascii_uppercase)
    migration_lines_list = []

    for idx in range(5):
        surface = surfaces[idx]
        ax = axes[idx]
        letter = panel_letters[idx]

        positions = surface["node_position"].values
        edge_nodes = surface["edge_node"].values
        segments = positions[edge_nodes]

        edge_log_migration = surface["edge_log_migration_refined"].values
        sample_count = surface["node_sample_count"].values

        if idx == 0:
            vmin, vmax = -0.75, 0.75
        else:
            limit = max(float(np.max(np.abs(edge_log_migration))), 0.1)
            vmin, vmax = -limit, limit

        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        lines = LineCollection(
            segments,
            cmap="RdBu",
            norm=norm,
            linewidths=1.8,
        )
        lines.set_array(edge_log_migration)
        ax.add_collection(lines)
        migration_lines_list.append(lines)

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

        ax.autoscale()
        ax.set_aspect("equal")
        ax.axis("off")

        ax.text(
            0.0,
            0.98,
            letter,
            transform=ax.transAxes,
            fontsize=15,
            va="top",
            ha="left",
        )

        ax.set_title(titles[idx], fontsize=11, pad=-5) # Reduced pad to pull titles closer to line graphs

    # --- SHARED HORIZONTAL COLORBAR AT BOTTOM ---
    # Decreased pad to pull colorbar closer to bottom subplots
    cbar = fig.colorbar(
        migration_lines_list[-1],
        ax=axes,
        orientation="horizontal",
        shrink=0.5,
        pad=0.01,
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
    filepath3 = snakemake.input[2]
    output_filepath1 = snakemake.output[0]
    output_filepath2 = snakemake.output[1]
    output_filepath3 = snakemake.output[2]

    with open(filepath1, "rb") as f:
        surface0, surface1, surface2, surface3, surface4 = pickle.load(f)
        
    surfaces = [surface0, surface2, surface1, surface3, surface4]

    plot_figure(surfaces, output_filepath1)

    with open(filepath2, "rb") as f:
        surface0, surface1, surface2, surface3, surface4 = pickle.load(f)
        
    surfaces = [surface0, surface2, surface1, surface3, surface4]

    plot_figure(surfaces, output_filepath2)

    with open(filepath3, "rb") as f:
        surface0, surface1, surface2, surface3, surface4 = pickle.load(f)
        
    surfaces = [surface0, surface2, surface1, surface3, surface4]

    plot_figure(surfaces, output_filepath3)
    
    return
        
if __name__ == "__main__":
    main(snakemake)