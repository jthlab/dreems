from matplotlib.colors import Normalize, TwoSlopeNorm
import rasterio
from rasterio.plot import show
from matplotlib.collections import LineCollection
import os
import sys
import pickle
from scipy.spatial import cKDTree
from feems.viz import recover_nnz_entries
import matplotlib.pyplot as plt
import numpy as np
from feems import SpatialGraph
import networkx as nx

def draw_figure_map(positions, edge_nodes, edge_log_migration, sample_count, linewidths=2.5, alpha=0.4, limit=1.0, save_figure=False, filepath=None):
    # 1. DATA PREPARATION
    segments = positions[edge_nodes]

    v_limit = max(float(np.max(np.abs(edge_log_migration))), 0.1) * limit

    x_min, y_min = positions.min(axis=0)
    x_max, y_max = positions.max(axis=0)
    pad_x = (x_max - x_min) * 0.05 if x_max != x_min else 1.0
    pad_y = (y_max - y_min) * 0.05 if y_max != y_min else 1.0

    # 2. SELECT CONFIGURATION BASED ON 'posterior' FLAG
    config = {
            "title": "Relative migration",
            "cbar_label": "centered edge log migration",
            "cmap": "RdBu",  # Standard reversed colormap for migration
            "norm": TwoSlopeNorm(vmin=-v_limit, vcenter=0.0, vmax=v_limit),
            "values": edge_log_migration,
        }

    # 3. RASTER PATH & PLOTTING
    raster_path = "/home/jkliang/dreems/KL_code/final_paper_figures/HYP_HR_SR_OB_DR/HYP_HR_SR_OB_DR.tif"

    with rasterio.open(raster_path) as src:
        fig, ax = plt.subplots(figsize=(10, 18), constrained_layout=True)

        # --- 1. RASTER BASEMAP ---
        show(src, ax=ax, zorder=1)

        # --- 2. NETWORK EDGES ---
        line_coll = LineCollection(
            segments,
            cmap=config["cmap"],
            norm=config["norm"],
            linewidths=linewidths,
            alpha=alpha,    # Translucent lines allow terrain to show through
            zorder=2,
        )
        line_coll.set_array(config["values"])
        ax.add_collection(line_coll)

        # --- COLORBAR ---
        cbar = fig.colorbar(
            line_coll,
            ax=ax,
            fraction=0.015,
            pad=0.04,
            shrink=0.6,
            extend="both"
        )
        cbar.set_alpha(1.0)
        cbar.solids.set_alpha(1.0)
        cbar.set_label(config["cbar_label"], fontsize=16)

        # --- 3. SAMPLE NODES ---
        mask = sample_count > 0
        pos_valid = positions[mask]
        count_valid = sample_count[mask]

        ax.scatter(
            pos_valid[:, 0],
            pos_valid[:, 1],
            s=20.0 + (10.0 * count_valid),
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )

        # --- 4. MAP LIMITS & FORMATTING ---
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_aspect("equal")
        ax.set_xlabel("latitude", fontsize=16)
        ax.set_ylabel("longitude", fontsize=16)
        ax.set_title(config["title"], fontsize=16, pad=10)

        # --- 5. SAVE FIGURE ---
        if save_figure:
            plt.savefig(
                filepath, 
                format="pdf", 
                bbox_inches="tight",  # Prevents legend/label clipping
                dpi=300               # Ensures vector resolution remains sharp
            )

        plt.show()
        return

def plot_cv_err(lamb_q_grid, lamb_grid, lamb_cv, mean_cv_err, filepath):
    n_lines = len(lamb_q_grid)
    colors = [plt.cm.plasma(i / n_lines) for i in range(n_lines)]
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'd', 'P', 'H', '+', 'x', '|'][:n_lines]
    
    plt.figure(dpi=200)
    for i, lq in enumerate(lamb_q_grid):
        plt.plot(lamb_grid, mean_cv_err[i, :], 
                 '-o', color=colors[i], marker=markers[i], 
                 linewidth=2, alpha=0.8, label=rf'$\lambda_q$ = {lq:.3f}')
    
    plt.grid()
    plt.xlabel(r'$\lambda$')
    plt.semilogx()
    plt.ylabel('LOO-CV error')
    plt.legend(title=r'$\lambda_q$')
    err_min = np.nanmin(mean_cv_err)
    err_max = np.nanmax(mean_cv_err)
    margin = (err_max - err_min) * 0.05   # 5% padding
    plt.ylim(err_min - margin, err_max + margin)
    plt.axvline(lamb_cv, linewidth=2, color='orange', linestyle='--')
    plt.savefig(
                    filepath, 
                    format="pdf", 
                    bbox_inches="tight",  # Prevents legend/label clipping
                    dpi=300               # Ensures vector resolution remains sharp
                )
    plt.show()

def main(snakemake):
    # Get the output file and seed from Snakemake
    for f in snakemake.output:
        os.makedirs(os.path.dirname(f), exist_ok=True)
        
    ticks_path = snakemake.input[0]
    wolves_path = snakemake.input[1]
    fishes_path = snakemake.input[2]
    ticks_dense_path = snakemake.input[3]

    ticks_figure_path = snakemake.output[0]
    wolves_figure_path = snakemake.output[1]
    fishes_figure_path = snakemake.output[2]
    ticks_cv_figure_path = snakemake.output[3]
    wolves_cv_figure_path = snakemake.output[4]
    fishes_cv_figure_path = snakemake.output[5]
    ticks_dense_figure_path = snakemake.output[6]
    ticks_dense_cv_figure_path = snakemake.output[7]

    # CV grid used for every example        
    lamb_grid = np.geomspace(1e-3, 1e2, 10, endpoint=True)[::-1]
    lamb_q_grid = np.geomspace(1e-2, 1e2, 10, endpoint=True)[::-1]
    
    ######## Ticks ########
    with open(ticks_path, "rb") as f:
        lamb_q_cv, lamb_cv, mean_cv_err, genotypes, coord, grid, edges = pickle.load(f)

    sp_graph = SpatialGraph(genotypes, coord, grid, edges, scale_snps=True)
    sp_graph.fit(lamb=lamb_cv, lamb_q=lamb_q_cv)

    positions = sp_graph.node_pos
    sample_nodes = np.asarray(cKDTree(grid).query(coord)[1], dtype=np.int32)
    sample_count = np.bincount(sample_nodes, minlength=len(positions))
    edge_nodes = list(np.column_stack(nx.adjacency_matrix(sp_graph).nonzero()))
    weights = recover_nnz_entries(sp_graph)
    norm_log_weights = np.log10(weights) - np.mean(np.log10(weights))

    draw_figure_map(positions, edge_nodes, norm_log_weights, sample_count, linewidths=2.5, alpha=0.8, limit=1.0, save_figure=True, filepath=ticks_figure_path)
    plot_cv_err(lamb_q_grid, lamb_grid, lamb_cv, mean_cv_err, ticks_cv_figure_path)

    ######## Wolves ########
    with open(wolves_path, "rb") as f:
        lamb_q_cv, lamb_cv, mean_cv_err, genotypes, coord, grid, edges = pickle.load(f)

    sp_graph = SpatialGraph(genotypes, coord, grid, edges, scale_snps=True)
    sp_graph.fit(lamb=lamb_cv, lamb_q=lamb_q_cv)

    positions = sp_graph.node_pos
    sample_nodes = np.asarray(cKDTree(grid).query(coord)[1], dtype=np.int32)
    sample_count = np.bincount(sample_nodes, minlength=len(positions))
    edge_nodes = list(np.column_stack(nx.adjacency_matrix(sp_graph).nonzero()))
    weights = recover_nnz_entries(sp_graph)
    norm_log_weights = np.log10(weights) - np.mean(np.log10(weights))

    draw_figure_map(positions, edge_nodes, norm_log_weights, sample_count, linewidths=2.5, alpha=0.8, limit=1.0, save_figure=True, filepath=wolves_figure_path)
    plot_cv_err(lamb_q_grid, lamb_grid, lamb_cv, mean_cv_err, wolves_cv_figure_path)

    ######## Fishes ########
    with open(fishes_path, "rb") as f:
        lamb_q_cv, lamb_cv, mean_cv_err, genotypes, coord, grid, edges = pickle.load(f)

    sp_graph = SpatialGraph(genotypes, coord, grid, edges, scale_snps=True)
    sp_graph.fit(lamb=lamb_cv, lamb_q=lamb_q_cv)

    positions = sp_graph.node_pos
    sample_nodes = np.asarray(cKDTree(grid).query(coord)[1], dtype=np.int32)
    sample_count = np.bincount(sample_nodes, minlength=len(positions))
    edge_nodes = list(np.column_stack(nx.adjacency_matrix(sp_graph).nonzero()))
    weights = recover_nnz_entries(sp_graph)
    norm_log_weights = np.log10(weights) - np.mean(np.log10(weights))

    draw_figure_map(positions, edge_nodes, norm_log_weights, sample_count, linewidths=2.5, alpha=0.8, limit=1.0, save_figure=True, filepath=fishes_figure_path)
    plot_cv_err(lamb_q_grid, lamb_grid, lamb_cv, mean_cv_err, fishes_cv_figure_path)

    ######## Ticks ########
    with open(ticks_dense_path, "rb") as f:
        lamb_q_cv, lamb_cv, mean_cv_err, genotypes, coord, grid, edges = pickle.load(f)

    sp_graph = SpatialGraph(genotypes, coord, grid, edges, scale_snps=True)
    sp_graph.fit(lamb=lamb_cv, lamb_q=lamb_q_cv)

    positions = sp_graph.node_pos
    sample_nodes = np.asarray(cKDTree(grid).query(coord)[1], dtype=np.int32)
    sample_count = np.bincount(sample_nodes, minlength=len(positions))
    edge_nodes = list(np.column_stack(nx.adjacency_matrix(sp_graph).nonzero()))
    weights = recover_nnz_entries(sp_graph)
    norm_log_weights = np.log10(weights) - np.mean(np.log10(weights))

    draw_figure_map(positions, edge_nodes, norm_log_weights, sample_count, linewidths=2.5, alpha=0.8, limit=1.0, save_figure=True, filepath=ticks_dense_figure_path)
    plot_cv_err(lamb_q_grid, lamb_grid, lamb_cv, mean_cv_err, ticks_dense_cv_figure_path)
    
    return
        
if __name__ == "__main__":
    main(snakemake)