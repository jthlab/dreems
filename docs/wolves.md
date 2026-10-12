# Using FEEMS Inputs

For users that are interested in running `FEEMS` and comparing to `DREEMS`, we offer a function that can accept nearly identical inputs as `FEEMS`. `FEEMS` uses a diploid genotype matrix whose entries are the count of the minor alleles with dimensions (number_of_samples, number_of_variants), the sample coordinates (as an array), and the nodes and edges of a grid. To construct a grid, you may follow `FEEMS'` tutorial or see how DREEMS does it by viewing the [data processing tutorial](./data_preprocessing.md), either way is acceptable. 

In this example we will examine spatial SNP data for the North American grey wolves, to analyze population gene flow and connectivity across North America. For more details regarding the original migration surface analysis of this dataset please visit:

Joseph Marcus, Wooseok Ha, Rina Foygel Barber, John Novembre (2021) Fast and flexible estimation of effective migration surfaces eLife 10:e61927. [https://doi.org/10.7554/eLife.61927](https://elifesciences.org/articles/61927)

We obtained the datasets from [`FEEMS` github](https://github.com/NovembreLab/feems) and by following their tutorial to obtain the inputs `genotypes`, `coord`, `edges`, and `grid`. We saved evverything into .pkl file for convenience.  

Users must ensure that the genotypes matrix is LD-pruned, but you can leave missing data as is. DREEMS will automatically impute missing data using the mean. 

## Read in FEEMS inputs and run DREEMS

```python
import pickle
from dreems.utility import dreems_infer_genotypes

with open("wolves_data.pkl", "rb") as f:
    loaded = pickle.load(f)

genotypes = loaded["genotypes"]
edges = loaded["edges"]
nodes = loaded["grid"]
coord = loaded["coord"]

surface = dreems_infer_genotypes(genotypes, coord, nodes, edges-1)
```

Please note, FEEMS indexes the nodes starting from 1, DREEMS indexes the nodes starting from 0, hence why we require `edges-1`.

## Plotting surface
```python
from dreems.plotting import draw_projected_contour_map

fig, ax = draw_projected_contour_map(
    surface,
    sample_to_pop=None,
    smoothing_km=5,
    contour_levels=15,
    contour_linewidth=2.0,
    contour_fill_alpha=0.20,
    save_figure=True,
    figure_path="./wolves.pdf",
    posterior=False
)
```

<iframe
    src="_static/wolves.pdf"
    width="100%"
    height="700px"
    style="border: none;">
</iframe>

## Posterior SD surface

```python
fig, ax = draw_projected_contour_map(
    surface,
    sample_to_pop=None,
    smoothing_km=5,
    contour_levels=15,
    contour_linewidth=2.0,
    contour_fill_alpha=0.20,
    save_figure=True,
    figure_path="./wolves_posterior.pdf",
    posterior=True
)
```

<iframe
    src="_static/wolves_posterior.pdf"
    width="100%"
    height="700px"
    style="border: none;">
</iframe>
