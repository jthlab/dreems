---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: "0.13"
    jupytext_version: "1.19.5"
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# DREEMS quickstart

DREEMS (Deep Recovery of Effective Migration Surfaces) maps a geographic graph
and node-level genetic covariance to relative
migration on every graph edge. This notebook demonstrates both public input
paths:

1. construct the covariance from an sgkit-compatible genotype dataset; and
2. supply an existing node covariance matrix directly.

The bundled production checkpoint is used automatically.

Execute the MyST source from the repository root with:

```bash
uv sync --group notebook
uv run --no-sync jupytext \
    --to ipynb --execute examples/dreems_quickstart.md
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm

import dreems

rng = np.random.default_rng(20260730)
```

## Define a geographic graph

Graph nodes are locations and each row of `edges` contains the integer indices
of two connected nodes. Geographic coordinates are given as latitude and
longitude; DREEMS projects them to a local Cartesian coordinate system.

```{code-cell} ipython3
n_rows, n_columns = 5, 6
row, column = np.indices((n_rows, n_columns))
node_latitude = 43.5 + 0.35 * row.ravel()
node_longitude = -91.0 + 0.45 * column.ravel()

horizontal_edges = [
    (r * n_columns + c, r * n_columns + c + 1)
    for r in range(n_rows)
    for c in range(n_columns - 1)
]
vertical_edges = [
    (r * n_columns + c, (r + 1) * n_columns + c)
    for r in range(n_rows - 1)
    for c in range(n_columns)
]
edges = np.asarray(horizontal_edges + vertical_edges, dtype=np.int32)

node_latitude.shape, edges.shape
```

## Prepare genotype calls

This cell creates a small, reproducible diploid dataset for the example. Its
variables and dimensions match the sgkit convention used by VCF-Zarr:
`call_genotype(variants, samples, ploidy)` and `sample_id(samples)`. DREEMS
also accepts uniformly haploid or polyploid calls through the same interface.

```{code-cell} ipython3
sampled_nodes = np.asarray([0, 2, 5, 7, 9, 12, 14, 17, 19, 22, 24, 27, 29])
samples_per_node = np.asarray([2, 3, 2, 4, 3, 2, 4, 2, 3, 2, 4, 3, 2])
sample_nodes = np.repeat(sampled_nodes, samples_per_node)
n_samples = sample_nodes.size
n_variants = 800

sample_ids = np.asarray([f"sample_{index:03d}" for index in range(n_samples)])
sample_latitude = node_latitude[sample_nodes] + rng.normal(0.0, 0.015, n_samples)
sample_longitude = node_longitude[sample_nodes] + rng.normal(0.0, 0.015, n_samples)
sample_coordinates = {
    sample_id: (latitude, longitude)
    for sample_id, latitude, longitude in zip(
        sample_ids,
        sample_latitude,
        sample_longitude,
        strict=True,
    )
}

# Spatially structured allele probabilities make the inferred surface nontrivial.
base_frequency = np.clip(rng.beta(2.5, 2.5, n_variants), 0.05, 0.95)
base_log_odds = np.log(base_frequency / (1.0 - base_frequency))
sample_x = (node_longitude[sample_nodes] - node_longitude.mean()) / node_longitude.std()
sample_y = (node_latitude[sample_nodes] - node_latitude.mean()) / node_latitude.std()
spatial_loadings = rng.normal(0.0, 0.32, (2, n_variants))
barrier_loading = rng.normal(0.0, 0.28, n_variants)
barrier_side = np.where(node_longitude[sample_nodes] < -89.9, -1.0, 1.0)
log_odds = (
    base_log_odds[None, :]
    + sample_x[:, None] * spatial_loadings[0]
    + sample_y[:, None] * spatial_loadings[1]
    + barrier_side[:, None] * barrier_loading
)
allele_probability = 1.0 / (1.0 + np.exp(-log_odds))
alleles = rng.binomial(
    1,
    allele_probability[:, :, None],
    size=(n_samples, n_variants, 2),
)
calls = np.transpose(alleles, (1, 0, 2)).astype(np.int8)
missing = rng.random((n_variants, n_samples)) < 0.01
calls[missing] = -1

genotypes = xr.Dataset(
    data_vars={
        "call_genotype": (("variants", "samples", "ploidy"), calls),
        "sample_id": (("samples",), sample_ids),
    }
)
genotypes
```

Sample coordinates may already be stored as sample-indexed
`sample_latitude` and `sample_longitude` coordinates in the VCF-Zarr dataset.
When they are supplied separately, attach them by sample ID:

```{code-cell} ipython3
genotypes = dreems.attach_sample_coordinates(genotypes, sample_coordinates)
genotypes[["sample_latitude", "sample_longitude"]]
```

For a real VCF-Zarr dataset, replace the synthetic dataset above with an
sgkit-loaded dataset. Convert VCF to
[VCF-Zarr](https://sgkit-dev.github.io/bio2zarr/vcf2zarr/overview.html) first:

```bash
vcf2zarr convert cohort.vcf.gz cohort.vcz
```

```python
import sgkit

genotypes = sgkit.load_dataset("cohort.vcz")
genotypes = dreems.attach_sample_coordinates(genotypes, sample_coordinates)
```

## Construct the canonical DREEMS dataset

`from_vcf_zarr` filters common biallelic SNPs, mean-imputes missing dosages,
standardizes SNPs, assigns samples to their nearest graph node, and computes
the node-level covariance and sampling precision.

```{code-cell} ipython3
data = dreems.from_vcf_zarr(
    genotypes,
    node_latitude=node_latitude,
    node_longitude=node_longitude,
    edges=edges,
    min_maf=0.05,
    min_variant_call_rate=0.8,
    attrs={"name": "quickstart"},
)
data
```

The covariance is stored only for sampled graph nodes. `node_sample_count`
describes sample density over the complete graph, and `effective_snps`
controls the amount of information attributed to the covariance estimate.
For linked data, supply an LD-adjusted `effective_snps` to
`from_vcf_zarr`.

```{code-cell} ipython3
data[["node_covariance", "node_sample_count", "effective_snps"]]
```

## Infer the migration surface

`infer` loads the bundled checkpoint and applies the production trust-region
MAP refinement by default.

```{code-cell} ipython3
surface = dreems.infer(data)
surface[
    [
        "edge_log_migration_mean",
        "edge_log_migration_sd",
        "edge_relative_migration_mean",
        "edge_log_migration_refined",
        "edge_relative_migration_refined",
    ]
]
```

The global migration scale is not identifiable. DREEMS therefore reports
centered edge log migration and relative migration with geometric mean one.
The refined variables are MAP point estimates; the posterior SD describes the
neural estimate. Blue edges below have higher relative migration, while red
edges have lower relative migration.

```{code-cell} ipython3
positions = surface["node_position"].values
edge_nodes = surface["edge_node"].values
segments = positions[edge_nodes]
edge_log_migration = surface["edge_log_migration_refined"].values
edge_log_sd = surface["edge_log_migration_sd"].values
sample_count = surface["node_sample_count"].values

limit = max(float(np.max(np.abs(edge_log_migration))), 0.1)
figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

migration_lines = LineCollection(
    segments,
    cmap="RdBu",
    norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    linewidths=4.0,
)
migration_lines.set_array(edge_log_migration)
axes[0].add_collection(migration_lines)
figure.colorbar(migration_lines, ax=axes[0], label="centered edge log migration")

uncertainty_lines = LineCollection(
    segments,
    cmap="magma",
    norm=Normalize(vmin=0.0, vmax=max(float(edge_log_sd.max()), 0.1)),
    linewidths=4.0,
)
uncertainty_lines.set_array(edge_log_sd)
axes[1].add_collection(uncertainty_lines)
figure.colorbar(uncertainty_lines, ax=axes[1], label="posterior SD")

for axis, title in zip(
    axes,
    ("Relative migration", "Posterior uncertainty"),
    strict=True,
):
    axis.scatter(
        positions[:, 0],
        positions[:, 1],
        s=12.0 + 18.0 * sample_count,
        facecolor="white",
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
    )
    axis.autoscale()
    axis.set_aspect("equal")
    axis.set_xlabel("easting (km)")
    axis.set_ylabel("northing (km)")
    axis.set_title(title)

plt.show()
```

Pass `refine=False` when the neural estimate alone is required:

```python
neural_surface = dreems.infer(data, refine=False)
```

## Start from an existing node covariance

When node covariance has already been computed, construct the same canonical
dataset directly from NumPy arrays. Covariance rows must follow
`observed_nodes`, and positive `node_sample_count` entries must identify those
same nodes.

```{code-cell} ipython3
direct_data = dreems.create_dataset(
    node_positions=data["node_position"].values,
    edges=data["edge_node"].values,
    node_covariance=data["node_covariance"].values,
    observed_nodes=data.coords["observed_nodes"].values,
    node_sample_count=data["node_sample_count"].values,
    effective_snps=data["effective_snps"].item(),
    coordinate_names=("easting", "northing"),
    attrs={"name": "quickstart-direct-covariance"},
)
dreems.validate_dataset(direct_data)
direct_data
```

Inference is then the same one-line call:

```python
surface = dreems.infer(direct_data)
```
