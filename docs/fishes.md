# Allohistium Fishes

In this example we will examine spatial SNP data for the Allohistium fishes, to analyze population gene flow and connectivity across river systems. For more details regarding the study please visit:

Daniel J MacGuigan, Adam Taylor, Ava Ghezelayagh, Julia E Wood, Jeffrey W Simmons, Jon M Mollish, Thomas J Near, Genomic and Phenotypic Delimitation of Species in a Temperate Aquatic Biodiversity Hotspot, Systematic Biology, Volume 75, Issue 5, September 2026, Pages 911–929, [https://doi.org/10.1093/sysbio/syaf083.](https://academic.oup.com/sysbio/article/75/5/911/8346017)

Please download files listed under the name 'VCFs.tar' and 'FEEMS.tar' using [this link](https://datadryad.org/dataset/doi:10.5061/dryad.r4xgxd2q4#readme). We will be using the files 'Allo.coord.txt', 'Allo.boundary.txt', and 'Allohistium.m95p.unlinked.vcf'. The files are a VCF that has pruned, linkage disequilibrium (LD)-controlled variants, a text file with sample coordinates, and a text file with coordinates of an outer polygon.

Users must ensure that the vcf is LD-pruned, but you can leave missing data as is. DREEMS will automatically impute missing data using the mean. 

## Convert a compressed VCF file into a Zarr file

```python
import pysam
from sgkit.io.vcf import vcf_to_zarr

vcf = "Allohistium.m95p.unlinked.vcf"
vcf_gz = vcf_file + ".gz"

# Step 1: Compress the VCF using bgzip (if not already compressed)
pysam.tabix_compress(vcf_file, vcf_gz, force=True)

# Index the bgzipped VCF file
pysam.tabix_index(vcf_gz, preset="vcf", force=True)

# Convert to zarr file
vcf_to_zarr(vcf_gz, "./Allo.zarr")
```

## Load in DREEMS input
There are three mandatory inputs to DREEMS, a Zarr file, the sample coordinates, and the grid. 

```python
import xarray as xr
import numpy

# load in Zarr file
ds = xr.open_zarr("./Allo.zarr")
coord = np.loadtxt("Allo.coord.txt") # sample coordinates
outer = np.loadtxt("Allo.boundary.txt") # outer polygon for constructing grid
sample_ids = ds['sample_id'].values

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
```

## Create a grid using a given outer polygon
User's need to provide an outer polygon or a constructed grid over their geographic region of interest. If you do not have a grid, please see the [data preprocessing tutorial](./data_preprocessing.md) on how to construct a grid around your geographic region of interest. The author's of the paper provided an outer polygon so we will use that.

```python
from dreems.utility import create_grid
import networkx as nx

G = create_grid(outer, grid_size=0.18, pad=0.1)

nodes = np.array([list(coord) for coord in nx.get_node_attributes(G, 'pos').values()])
edges = np.array(list(G.edges))
```
## Running DREEMS
```python
from dreems.utility import dreems_infer

surface = dreems_infer(data=ds, sample_coordinates=sample_coordinates, nodes=nodes, edges=edges)
```

By default, DREEMS will filter for SNPs with a call rate of greater than 0.8, impute the missing data using the mean, and apply a MAF filtering of 0.05. These preprocessing steps are crucial for migration surface inference. 

## Plotting surface
```python
from dreems.plotting import draw_projected_contour_map

def assign_population(sample):
    if "BUFF" in sample or "DUCK" in sample:
        return "Allohistium anas"
    if sample.startswith("Amay"):
        return "Allohistium maydeni"
    return "Allohistium cinereum"

pop_map = {sample: assign_population(sample) for sample in sample_ids}
    
draw_projected_contour_map(
    surface,
    sample_to_pop=pop_map,
    smoothing_km=10.0,
    contour_levels=15,
    contour_linewidth=2.0,
    contour_fill_alpha=0.20,
    save_figure=True,
    figure_path="./fishes_contour.pdf"
)
```

<iframe
    src="_static/fishes_contour.pdf"
    width="100%"
    height="700px"
    style="border: none;">
</iframe>
