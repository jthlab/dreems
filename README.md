# DREEMS (Deep Recovery of Effective Migration Surfaces)

**DREEMS** is a python package that implements a neural estimator for effective migration surfaces. The model predicts centered edge log-migration on a spatial graph and reports posterior standard deviations. To read the full paper, please see: 

## Documentation

Full documentation is available at **[demestats.readthedocs.org](https://demestats.readthedocs.org)**.

## Installation

`DREEMS` will eventually be available on PyPI. Until then, please install directly from GitHub:

```bash
pip install git+https://github.com/jthlab/dreems
```

We recommend using a virtual environment (e.g., `venv` or `conda`) to avoid conflicts with system packages. This default installation is CPU-only and works on machines without an NVIDIA driver. You need a GPU only if you wish to modify the graph neural network architecture and retrain the entire model from scratch. 

## License

MIT License. See `docs/index.md` or `LICENSE` for details.

## Install

To works on machines with an NVIDIA driver. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone this github repository, and run:

```bash
uv sync --frozen
uv run --no-sync dreems --help
```

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
