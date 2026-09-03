# DREEMS

DREEMS (Deep Recovery of Effective Migration Surfaces) is a neural estimator
for effective migration surfaces. The production model predicts centered edge
log-conductances on a spatial graph and reports held-out-calibrated edge
posterior standard deviations. A learned graph-level gate shrinks the centered
surface toward homogeneous migration when the data do not support spatial
contrast. FEEMS baselines are exposed through a small wrapper around the
Novembre Lab FEEMS implementation.

This repository contains the production package, the workflows used to train
and evaluate the released model, direct tests, and the manuscript with frozen
result snapshots. Exploratory architecture searches, superseded launchers,
and generated workflow state are intentionally left out.

## Install

The default installation is CPU-only and works on machines without an NVIDIA
driver. Install [uv](https://docs.astral.sh/uv/getting-started/installation/),
clone the repository, and run:

```bash
uv sync --frozen
uv run --no-sync dreems --help
```

For FEEMS baselines, first install the SuiteSparse/CHOLMOD development library
required by `scikit-sparse`, then install the optional FEEMS dependencies. For
example, on Ubuntu or Debian:

```bash
sudo apt-get install libsuitesparse-dev
uv sync --frozen --extra feems
```

On macOS with Homebrew, install SuiteSparse with `brew install suite-sparse`
before running the same `uv sync` command. The neural DREEMS estimator does not
require FEEMS or SuiteSparse.

CUDA is opt-in. On a Linux host with a compatible NVIDIA driver, install the
CUDA-enabled JAX extra with:

```bash
uv sync --frozen --extra cuda
```

Do not use the CUDA extra on ordinary laptops, login nodes, or CPU-only
machines. It installs several gigabytes of NVIDIA runtime libraries.

## Develop And Test

The complete test suite includes workflow and rendering tests, so both the
`dev` and `workflow` dependency groups are required:

```bash
uv sync --frozen --group dev --group workflow
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
JAX_PLATFORMS=cpu uv run --no-sync pytest -q
```

These commands are exercised from a clean checkout in GitHub Actions. The
packaged checkpoint is also loaded from the built wheel outside the source
tree.

Run the complete release gate with:

```bash
uv sync --frozen --group dev --group workflow --group notebook
uv run --no-sync snakemake \
    -s workflow/quality.Snakefile \
    --profile workflow/profiles/local
```

On Great Lakes, replace the local profile with `workflow/profiles/slurm`.

## Workflow

The Snakemake workflow in `workflow/` generates each simulation in an
independent job, prepares diffusion-band and all-pair conditioner features per
example, computes mergeable normalization statistics, trains JAX models, and
evaluates held-out examples independently. One-example worker outputs are
temporary: Snakemake compacts them into reusable raw packs before feature
preparation, avoiding hundreds of thousands of small scratch files.

The production lineage is recorded in separate configs:

- `workflow/config.snp_range_experiment.yaml` records the 61,440-example
  broad-SNP corpus and its 4,096-example held-out corpus.
- `workflow/config.full_retrain.yaml` records the from-scratch core fit,
  sampling-mask counterfactuals, and frozen graph-level gate fit.
- `workflow/config.surface_gate_release.yaml` records fresh posterior-SD
  calibration, verification, and deterministic release packaging.
- `workflow/config.example.yaml` reduces simulation sizes and optimizer steps
  for an executable smoke run.

See [workflow/README.md](workflow/README.md) for checkpoint provenance and the
complete run procedure.

Install workflow dependencies for CPU-only preparation and evaluation with:

```bash
uv sync --frozen --group workflow --group dev
```

For GPU training, include the CUDA extra:

```bash
uv sync --frozen --extra cuda --group workflow --group dev
```

Dry-run the small example locally:

```bash
uv run --no-sync snakemake \
    -s workflow/Snakefile \
    --configfile workflow/config.example.yaml \
    --profile workflow/profiles/local \
    -n
```

To materialize only the first smoke raw pack:

```bash
uv run --no-sync snakemake \
    -s workflow/Snakefile \
    --configfile workflow/config.example.yaml \
    --profile workflow/profiles/local \
    results/dreems_example/raw/train/pack_0.pkl
```

The full corpus requires the Rust simulator and CUDA environment:

```bash
module load rust/1.92.0  # omit on systems where rustc is already available
uv sync --frozen --extra cuda --group workflow --group rust --group dev
uv run maturin develop --release --manifest-path rust/island_sim/Cargo.toml
```

On `maturin`, the local runner defaults to 60 cores, 240 GB RAM, and one GPU:

```bash
DREEMS_RUN_ROOT=/path/to/scratch/dreems/conditioned_diffusion_bands \
workflow/run_maturin.sh
```

Override resources with `DREEMS_LOCAL_CORES`, `DREEMS_LOCAL_MEM_MB`, and
`DREEMS_LOCAL_GPUS`. Set `DREEMS_CONFIG` to use the smoke or another explicit
config.

On Great Lakes, the Slurm profile uses account `jonth0`. The controller itself
may run as one Slurm job; Snakemake submits every worker:

```bash
DREEMS_RUN_ROOT=/scratch/jonth_root/jonth0/jonth/dreems/conditioned_diffusion_bands \
workflow/run_great_lakes.sh
```

From an existing controller allocation, run:

```bash
uv run --no-sync snakemake \
    -s workflow/Snakefile \
    --configfile workflow/config.production.yaml \
    --profile workflow/profiles/slurm \
    --config outdir=/path/to/scratch/dreems/conditioned_diffusion_bands
```

The production simulator treats `q` as local coalescence rate, uses symmetric
graph conductances, conditions markers on MAF 0.05-0.95, and emits covariance
summaries directly. Training and FEEMS benchmarking are deliberately separate;
FEEMS is not used to generate or optimize the released neural model.

## Inference API

DREEMS inference is exposed as a black-box map from covariance and geography to
a migration surface. The canonical input and output are `xarray.Dataset`
objects.

See the executable [Jupytext/MyST quickstart](examples/dreems_quickstart.md)
for complete genotype and direct-covariance examples.

```python
import dreems

data = dreems.create_dataset(
    node_positions=node_positions,          # (nodes, 2), Cartesian coordinates
    edges=edges,                            # (edges, 2), integer node indices
    node_covariance=node_covariance,        # (observed_nodes, observed_nodes)
    observed_nodes=observed_nodes,          # graph index for each covariance row
    node_sample_count=node_sample_count,    # (nodes,)
    effective_snps=n_snps,
)

surface = dreems.infer(data)
edge_log = surface["edge_log_migration_refined"]
relative_migration = surface["edge_relative_migration_refined"]
```

`node_sample_count` and `effective_snps` are required precision information,
not tuning parameters. The genotype convenience path computes both
automatically, using the retained SNP count by default; pass an
LD-adjusted `effective_snps` value when appropriate.

### From VCF-Zarr

DREEMS accepts an xarray genotype dataset using the standard sgkit dimensions
and variables: `call_genotype(variants, samples, ploidy)` and
`sample_id(samples)`. Convert VCF to Zarr with
[bio2zarr](https://sgkit-dev.github.io/bio2zarr/vcf2zarr/overview.html), then
open the result with sgkit or xarray.

Per-sample locations are xarray coordinates indexed by `samples`; they are not
dataset-wide attributes. They may already be present as `sample_latitude` and
`sample_longitude`, or can be attached from a sample-ID mapping:

```python
import sgkit
import dreems

genotypes = sgkit.load_dataset("cohort.vcz")

data = dreems.from_vcf_zarr(
    genotypes,
    sample_coordinates={
        "sample_001": (42.28, -83.74),  # (latitude, longitude)
        "sample_002": (42.35, -83.61),
    },
    node_latitude=node_latitude,
    node_longitude=node_longitude,
    edges=edges,
)
surface = dreems.infer(data)
```

The constructor accepts uniformly haploid, diploid, or polyploid allele calls.
It retains biallelic variants passing common-MAF and call-rate filters,
mean-imputes missing dosages, standardizes each SNP, assigns samples to their
nearest graph node, and computes node-level covariance. Ploidy affects allele
frequency and MAF calculation; the trained model consumes the resulting
standardized covariance and does not require ploidy-specific weights.

DREEMS surfaces are relative: the global log offset is not identifiable.
`edge_log_migration_mean` and `edge_relative_migration_mean` are the trained
neural posterior summaries; the latter has geometric mean one. By default,
`dreems.infer` also reports the covariance-refined MAP point estimate as
`edge_log_migration_refined` and `edge_relative_migration_refined`. The
reported posterior SD belongs to the neural posterior, not to the refined
point estimate. It is multiplicatively calibrated by sampled-node density on
held-out simulations; `edge_log_migration_sd_raw` retains the uncalibrated
value when the two differ. `surface.attrs["dreems_surface_weight"]` reports the
learned graph-level shrinkage weight: zero is a homogeneous centered surface,
and one retains the full predicted contrast. The lower-level
`DREEMSResult.surface_weight` field exposes the same value. Pass `refine=False`
to skip refinement.

The lower-level numpy API remains available through `DREEMSInput`,
`NodeCovarianceInput`, and `load_model`.

The lower-level array interface is also available from the command line:

```bash
dreems \
  --covariance sample_covariance.npy \
  --sample-coordinates sample_coordinates.npy \
  --node-coordinates node_coordinates.npy \
  --edges edges.npy \
  --effective-snps 50000 \
  --output surface.csv
```

Pass `--checkpoint path/to/checkpoint.dreems.zip` or
`dreems.load_model("path/to/checkpoint.dreems.zip")` to override the packaged
topology-invariant diffusion-band surface-gate checkpoint.

For testing against exact FEEMS, use `--feems` with a genotype matrix. Official
FEEMS fitting requires genotypes/SNPs, not just a covariance matrix:

```bash
dreems --feems \
  --genotypes genotypes.npy \
  --sample-coordinates sample_coordinates.npy \
  --node-coordinates node_coordinates.npy \
  --edges edges.npy \
  --output feems_surface.csv
```

## Package Layout

- `dreems.datasets`: canonical xarray schema, validation, and surface output.
- `dreems.genotypes`: sgkit/VCF-Zarr coordinate attachment, QC, and aggregation.
- `dreems.features`: graph, genotype, observation, normalization, and
  linearized-resistance conditioner features.
- `dreems.inference`: stable checkpoint loading and prediction API for users.
- `dreems.model`: production GNN forward pass.
- `dreems.losses`: supervised Gaussian NLL, slope, correlation, and orientation
  losses.
- `dreems.pipeline`: artifact-oriented CLI used by the Snakemake workflow.
- `dreems.train`: single-device JAX training loop and checkpoint writer.
- `dreems.checkpoints`: checkpoint loading and prediction API.
- `dreems.calibration`: post-hoc posterior SD calibration.
- `dreems.metrics`: centered edge-log surface metrics.
- `dreems.feems`: FEEMS baseline protocols with leave-one-out CV defaults.

## Manuscript And Results

The manuscript is in
[`reports/dreems_genetics_manuscript`](reports/dreems_genetics_manuscript).
Its committed data snapshots, figures, and tables are the evidence used in the
compiled paper. They correspond to the predecessor checkpoint SHA-256
`f0b677d2cc028252002d15fac843ca7347b1b352eb2286a18bd202b724aa92b4`
and the 512-example FEEMS leave-one-observed-deme-out comparison recorded in
`workflow/h2h.Snakefile`. The packaged production checkpoint is newer; the
frozen manuscript results must be regenerated before attributing them to it.

Install the manuscript dependencies and rebuild the PDF from the committed
assets with:

```bash
uv sync --frozen --group manuscript
make -C reports/dreems_genetics_manuscript check
```

See the manuscript README before refreshing assets from full training, H2H,
calibration, and empirical-analysis directories. Those source artifacts are
too large for Git and are not needed to compile or audit the frozen paper.
