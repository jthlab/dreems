# DREEMS

DREEMS (Deep Recovery of Effective Migration Surfaces) is a neural estimator
for effective migration surfaces. The production model predicts centered edge
log-conductances on a spatial graph and reports posterior standard deviations
calibrated on held-out simulations. A learned graph-level gate shrinks the
centered surface toward homogeneous migration when the data do not support spatial
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
uv run --no-sync ruff check src tests workflow
uv run --no-sync ruff format --check src tests workflow
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
evaluates simulations. One-example worker outputs are
temporary: Snakemake compacts them into reusable raw packs before feature
preparation, avoiding hundreds of thousands of small scratch files.

`workflow/release.Snakefile` is the current training entry point. It builds raw
packs, sampling views, the core model, and the frozen-core surface gate in one
DAG. It does not need an older checkpoint or pre-existing sampling-view files.
Both launchers use this workflow and `workflow/config.release.yaml` by default.
Set `outdir` to scratch on Great Lakes and `publish_dir` to a durable directory.

```bash
uv run --frozen --no-sync snakemake all \
    -s workflow/release.Snakefile \
    --configfile workflow/config.release.yaml \
    --profile workflow/profiles/local \
    --config outdir=/path/to/work publish_dir=/path/to/models
```

This recipe trains on the full production corpus.
For a small smoke run, use `workflow/Snakefile` with `workflow/config.example.yaml`.
The training workflow's evaluation includes examples used for early stopping;
independent performance estimates come from the separate H2H workflow.

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
`DREEMS_LOCAL_GPUS`. For another workflow, set both `DREEMS_CONFIG` and
`DREEMS_SNAKEFILE`; the smoke run uses `config.example.yaml` with `Snakefile`.

On Great Lakes, the Slurm profile uses account `jonth0`. The controller itself
may run as one Slurm job; Snakemake submits every worker:

```bash
DREEMS_RUN_ROOT=/scratch/jonth_root/jonth0/jonth/dreems/conditioned_diffusion_bands \
workflow/run_great_lakes.sh
```

From an existing controller allocation, run:

```bash
uv run --frozen --no-sync snakemake \
    -s workflow/release.Snakefile \
    --configfile workflow/config.release.yaml \
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

`node_sample_count` and `effective_snps` describe sampling precision and must
match the data. The genotype convenience path computes both
automatically, using the retained SNP count by default; pass an
LD-adjusted `effective_snps` value when appropriate.

The covariance rows and columns must follow `observed_nodes` in the same order.
If `observed_nodes` is omitted, rows must follow the increasing indices of nodes
with positive sample counts. Counts refer to individuals.
Use a connected graph with no self edges or duplicate undirected edges.
Indices and counts must be integer-valued; fractional values are rejected.
Integral floats, such as those read from CSV, are accepted.

### From VCF-Zarr

DREEMS accepts an xarray genotype dataset using the standard sgkit dimensions
and variables: `call_genotype(variants, samples, ploidy)` and
`sample_id(samples)`. Convert VCF to Zarr with
[bio2zarr](https://sgkit-dev.github.io/bio2zarr/vcf2zarr/overview.html), then
open the result with sgkit or xarray.

Per-sample locations are xarray coordinates indexed by `samples`.
They may already be present as `sample_latitude` and
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

Calls must contain signed integer allele codes. Dosages and genotype
probabilities need conversion before using this constructor.
A `-1` allele or a true `call_genotype_mask` entry makes the whole sample call
missing. Mixed ploidy (`-2` padding) is rejected. A sample with no called genotypes
among the retained variants must be removed before conversion; it cannot count
as an observation simply because its missing values can be imputed.

The default filters are MAF >= 0.05 and variant call rate >= 0.8. Apply study-specific
sample, read-depth, genotype-quality, and LD filters before conversion. This
constructor does not perform those checks or infer an LD-adjusted SNP count.

DREEMS surfaces are relative: the global log offset is not identifiable.
`edge_log_migration_mean` is the neural posterior mean log rate.
`edge_relative_migration_mean` is its exponential, normalized to geometric mean
one; it is not the arithmetic posterior mean of migration. By default,
`dreems.infer` also reports the covariance-refined MAP point estimate as
`edge_log_migration_refined` and `edge_relative_migration_refined`. The
reported posterior SD describes the neural posterior. Refinement does not
provide an updated SD. The neural SD is multiplied by a factor fitted for each
sampling-density bin on held-out simulations; `edge_log_migration_sd_raw` retains the uncalibrated
value when the two differ. `surface.attrs["dreems_surface_weight"]` reports the
learned graph-level shrinkage weight: zero is a homogeneous centered surface,
and one retains the full predicted contrast. The lower-level
`DREEMSResult.surface_weight` field exposes the same value. Pass `refine=False`
to skip refinement.

The lower-level numpy API remains available through `DREEMSInput`,
`NodeCovarianceInput`, and `load_model`. Unlike `dreems.infer`, the lower-level
`predict` methods default to neural inference only. Pass
`refinement_config=True` to add MAP refinement there, or `--refine` on the CLI.
Reuse a model loaded with `dreems.load_model()` when predicting several datasets.

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
`dreems.load_model("path/to/checkpoint.dreems.zip")` to override the
packaged checkpoint.

For comparison with the official FEEMS implementation, use `--feems` with
a genotype matrix:

```bash
dreems --feems \
  --genotypes genotypes.npy \
  --sample-coordinates sample_coordinates.npy \
  --node-coordinates node_coordinates.npy \
  --edges edges.npy \
  --output feems_surface.csv
```

## Package Layout

```text
src/dreems/
    data/          Genotype inputs, graph and surface generation, simulations
    neural/        Equinox model, features, linear initializer, checkpoints
    training/      Losses, optimizer loop, sampling views, training workers
    inference/     Prediction API, covariance refinement, SD calibration
    evaluation/    FEEMS comparisons, metrics, scenarios, galleries
    utils/         Array validation and file I/O
    feems.py       Adapter for the official FEEMS package
    cli.py         The dreems command
    __init__.py    Public Python API
```

Start with `neural/model.py` for the architecture, `training/train.py` for
training, or `inference/api.py` for predictions. `data/simulation.py` assembles
training examples; `data/island_sim.py` provides the Rust and Python coalescent
backends. The packaged weights remain in `data/dreems_diffusion_bands.dreems.zip`.

Use the top-level API in application code: `dreems.create_dataset`,
`dreems.from_vcf_zarr`, `dreems.infer`, and `dreems.load_model`. Internal imports
use the subpackages, such as `from dreems.training.train import DREEMSTrainConfig`.
The CLI and `dreems.feems` adapter retain their existing interfaces.

Snakemake workers use the same layout, for example
`python -m dreems.training.pipeline` and `python -m dreems.evaluation.h2h`.
Separate FEEMS and neural dependencies let model changes reuse saved FEEMS
fits. The [workflow guide](workflow/README.md) describes the training and
evaluation protocols.

Array annotations name their dimensions: positions are `(nodes, 2)`, endpoints
are `(edges, 2)`, and migration estimates are `(edges,)`. `Graph`, `PreparedExample`,
`DREEMSTrainArrays`, and `DREEMSTrainBatch` document the dictionaries passed
between simulation, preparation, and training. Jaxtyping checks function
arguments and results; explicit validators also check indices, finite values,
and dictionary contents. Generic archive readers retain arbitrary shapes until
the caller validates the fields.

### FEEMS Adapter

`dreems.feems` calls the official package with a complete diploid dosage matrix
(samples by SNPs), sample-node indices, `graph={"edges": edges}`, and node
coordinates. Results follow the input edge order. Handle missing calls and
apply MAF and LD filters before fitting. The adapter removes monomorphic SNPs
and applies FEEMS' heterozygosity scaling.

`fit_feems` takes a `FEEMSConfig` that selects the variance model, penalty grid,
and validation procedure. Variance can be fixed (`fixed_q`), one fitted scalar
(`scalar_q`), or a fitted value per node (`free_q`).
`fit_feems_recommended_loo` selects the publication protocol, including the
stricter tolerance in the final refit. Worker count controls fold parallelism.
See [the workflow procedure](workflow/README.md#feems-head-to-head) for the
grids, tolerances, and pinned FEEMS revision.

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
