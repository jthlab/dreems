# Blacklegged Ticks

In this example we will examine spatial SNP data for the blacklegged tick (Ixodes scapularis), to analyze population gene flow and connectivity across the Midwestern United States. For more details regarding the study please visit:

Dong, D.-y., S. M. Paskewitz, J. I. Tsao, and S. D. Schoville. 2025. “Genetic and Landscape Connectivity of Blacklegged Ticks During Range Expansion in Select States of the Midwestern USA.” Ecology and Evolution 15, no. 10: e72360. [https://doi.org/10.1002/ece3.72360.](https://onlinelibrary.wiley.com/doi/10.1002/ece3.72360)

Please download three files listed under the name '03.pruned.vcf.gz', 'Metadata.xlsx', and 'polygon_outer' using [this link](https://datadryad.org/dataset/doi:10.5061/dryad.c866t1gh7#readme). The files are a VCF that has pruned, linkage disequilibrium (LD)-controlled variants but has not yet been imputed for missing data, a text file with sample coordinates, and a text file with coordinates of an outer polygon.

Users must ensure that the vcf is LD-pruned, but you can leave missing data as is. DREEMS will automatically impute missing data using the mean. 

## Convert a compressed VCF file into a Zarr file

```python
import pysam
from sgkit.io.vcf import vcf_to_zarr

vcf_gz = "03.pruned.vcf.gz"

# Index the bgzipped VCF file
pysam.tabix_index(vcf_gz, preset="vcf", force=True)

# Convert to zarr file
vcf_to_zarr(vcf_gz, "./ticks.zarr")
```

## Load in DREEMS input
There are three mandatory inputs to DREEMS, a Zarr file, the sample coordinates, and the grid. For sample coordinates, you need to create a dictionary mapping sample names to their coordinates in (latitude, longitude) form. We extract the coordinates from `Metadata.xlsx`.

```python
import xarray as xr
import numpy as np
import pandas as pd

# load in Zarr file
ds = xr.open_zarr("./ticks.zarr")
outer = np.loadtxt("polygon_outer.txt") # outer polygon for constructing grid
df = pd.read_excel("Metadata.xlsx")

# extract sample_coordinates from metadata
# the sample coordinates should be (latitude, longitude) pairs
sample_coordinates = {
    row["id"]: (row["long"], row["lat"])
    for _, row in df.iterrows()
}
```

## Create a grid using a given outer polygon
User's need to provide an outer polygon or a constructed grid over their geographic region of interest. If you do not have a grid, please see the [data preprocessing tutorial](./data_preprocessing.md) on how to construct a grid around your geographic region of interest. The author's of the paper provided an outer polygon so we will use that.

```python
from dreems.utility import create_grid
import networkx as nx

G = create_grid(outer, grid_size=0.38, pad=0.1)

nodes = np.array([list(coord) for coord in nx.get_node_attributes(G, 'pos').values()])
edges = np.array(list(G.edges))
```

## Running DREEMS
The function `dreems_infer` will internally filter for SNPs with a call rate of greater than 0.8, impute the missing data using the mean, and apply a MAF filtering of 0.05. Please see the API if you wish to override these parameters. These preprocessing steps are crucial for migration surface inference. 

```python
from dreems.utility import dreems_infer

surface = dreems_infer(data=ds, sample_coordinates=sample_coordinates, nodes=nodes, edges=edges)
```

## Plotting surface
`pop_map` is an optional dictionary mapping individual samples to population labels for population-based color coding. If omitted, all sample points are given a uniform color.

```python
from dreems.plotting import draw_projected_contour_map

id_to_region = dict(zip(df["id"], df["region"]))

prefixes_to_update = ("BARA", "JWWE", "FAYE")

pop_map = {
    sample_id: (
        "UP-MI" if sample_id.startswith(prefixes_to_update)
        else "LP-MI"
    ) if region == "MI" else region
    for sample_id, region in id_to_region.items()
}

draw_projected_contour_map(
    surface,
    sample_to_pop=pop_map,
    smoothing_sigma=5,
    contour_levels=15,
    contour_linewidth=2.0,
    contour_fill_alpha=0.20,
    save_figure=True,
    figure_path="./ticks.pdf"
)
```

<iframe
    src="_static/ticks.pdf"
    width="100%"
    height="700px"
    style="border: none;">
</iframe>

Here's what it would look like without population-based coloring. For other ways to visualize a migration surface, please see [Visualizing Surfaces](./plotting.md).

```python
draw_projected_contour_map(
    surface,
    sample_to_pop=None,
    smoothing_sigma=5,
    contour_levels=15,
    contour_linewidth=2.0,
    contour_fill_alpha=0.20,
)
```

<iframe
    src="_static/ticks_no_color.pdf"
    width="100%"
    height="700px"
    style="border: none;">
</iframe>

