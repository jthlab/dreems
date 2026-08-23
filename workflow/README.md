# DREEMS Training And Calibration Workflow

This directory contains the Snakemake path used to generate the synthetic
corpus, construct DREEMS features, fit the model, and evaluate held-out
simulations. `calibration.Snakefile` canonicalizes the input representation and
calibrates posterior SDs on disjoint held-out splits. `config.production.yaml`
and `config.calibration.yaml` record the released recipe. `config.example.yaml`
is a small CPU smoke run with the same feature and model architecture.

## Released Model Provenance

The packaged checkpoint is
`src/dreems/data/dreems_diffusion_bands.dreems.zip`. Its archive SHA-256 is:

```text
f0b677d2cc028252002d15fac843ca7347b1b352eb2286a18bd202b724aa92b4
```

The serialized Equinox parameters have SHA-256
`d7899a6bb419e4d017e093af958ab9750c2a947ccc72087ccf7bfcd18a3c2984`.
They are the step-29,000 checkpoint from `graph_balanced_retrain_20260821`.
Canonical-input migration, SD calibration, and release packaging change only
metadata; the audited parameter bytes are unchanged.

The checkpoint was fit from scratch on 61,440 training examples, with 1,024 of
4,096 held-out examples used for validation. The selected validation loss was
`0.1572438609728124` at step 29,000. Training stopped at step 41,000 and
restored that checkpoint. The exact contract is recorded in
`config.production.yaml` and enforced by `tests/test_production_workflow.py`.
`promote_checkpoint.py` verifies the saved best step against the complete
training record and writes the deterministic release archive.

## Statistical Recipe

Each synthetic example is generated independently with a deterministic seed.
The base production corpus uses:

- the sparse Rust island coalescent simulator;
- direct sample-covariance output rather than retained genotype matrices;
- symmetric migration and diploid samples;
- markers conditioned on MAF 0.05-0.95;
- 2,000-10,000 retained SNPs per example;
- masked grids, FEEMS triangular lattices, and global landmass graphs;
- graph sizes extending to roughly 1,400 nodes, broad sampling density, and
  variable local coalescence rate `q`;
- constant, smooth, barrier, Gaussian-process, earth-like, and
  landscape-conditioned migration surfaces.

The workflow downloads the pinned Natural Earth 50m land archive once on the
controller, verifies its SHA-256, and exposes the extracted Cartopy cache as a
shared input to every landscape job. Compute workers do not need internet
access.

The released predictor uses the leading 20 components of the observed
covariance, transforms those signals through five fixed graph-diffusion bands,
and includes the all-pair diagonal-adjoint linearized-resistance conditioner.
Coordinate axes and signed edge directions are excluded from model features;
endpoint summaries are symmetric and path support is shared across all equally
short routes. Uniform rotation, reflection, scale, edge reversal, and node
relabeling therefore do not change predictions.
The neural network is a 10-layer, 128-channel edge-aware residual GNN with a
three-layer context branch. The prepared node and edge dimensions are 109 and
253.

Training minimizes one fixed objective. Every term is first reduced within a
graph and then averaged across the batch, so graph size does not set a
simulation's influence:

```text
edge Gaussian NLL
  + 0.50 * context edge MSE
  + 0.50 * node-q Gaussian NLL
  + 0.25 * edge-slope loss
  + 0.20 * centered edge-correlation loss
  + 0.35 * high-versus-low edge-orientation loss
```

Training uses single-device Adam with learning rate `5e-5`, global gradient
norm clipping at 1, batch size 8, and at most 120,000 steps. It checks
validation every 1,000 steps, uses patience 12 after a 20,000-step warmup, and
restores the best checkpoint.

The training corpus already stores node covariance. The release workflow
therefore relabels the checkpoint to use the canonical node-covariance path;
the old and new feature arrays are exactly equal for every training input, and
the serialized parameters do not change. Genotypes and sample covariance are
collapsed to this same representation during inference.

Posterior-SD calibration uses test IDs 1024-2047 for fitting, 2048-3071 for
selecting raw, global, or density-aware scaling by graph-balanced Gaussian NLL,
and 3072-4095 for untouched verification. Density-aware scaling was selected.
On verification, graph-balanced 95% coverage changed from 93.2% to 95.7%, and
NLL changed from -0.131 to -0.185. Lower-level intervals remain conservative,
so the factorized Gaussian posterior should not be read as a complete joint
uncertainty model.

## DAG And Artifacts

The important artifact boundaries are:

```text
staging/raw/{split}/chunk_N.pkl               (temporary; one job per example)
    -> raw/{split}/pack_P.pkl                 (persistent reusable corpus)
    -> prepared_batches/{role}/pack_P.pkl
    -> normalization/group_P.stats.json
    -> normalization.json
    -> model/checkpoint*.dreems.zip
    -> eval/chunks/chunk_N/summary.json        (temporary)
    -> eval/summary.{json,csv}

test IDs 1024-2047 -> calibration fit
test IDs 2048-3071 -> calibration selection
test IDs 3072-4095 -> calibration verification
    -> checkpoint.calibrated.dreems.zip
    -> verification/{report.json,by_sampling.csv}
```

Pack, normalization, training, and aggregate evaluation rules record Snakemake
benchmarks under `benchmarks/`. Per-example rules do not leave successful
benchmark or log files behind; failed Slurm logs are retained by the executor.
Both execution profiles store Snakemake provenance in one SQLite database at
`.snakemake/metadata.db` instead of one metadata file per output.

One `generate_raw_example` job still creates exactly one example. Completed
worker outputs are immediately collected into deterministic 64-example raw
packs, then deleted as Snakemake temporary files. The raw packs contain the same
ordered example dictionaries and one aggregate audit sidecar. Feature packs are
built from them directly. This preserves independent retries and reusable raw
data while reducing the production raw corpus from 131,072 persistent files to
2,048 files. Snakemake owns scheduling, caching, retries, and artifact
dependencies.

Simulation, feature, and training configuration are stored as separate
content-addressed snapshots. Raw examples depend only on the simulation
snapshot, the one-example corpus worker, geographic motif code, and simulator
code. Changing optimizer settings or training implementation therefore does
not regenerate the raw covariance corpus; changing feature construction starts
again at preparation, and changing the simulation recipe starts again at raw
generation.

## Smoke Run

Install the locked CPU environment:

```bash
uv sync --frozen --group dev --group workflow
```

Inspect the complete small DAG:

```bash
uv run --frozen --no-sync snakemake \
    -s workflow/Snakefile \
    --configfile workflow/config.example.yaml \
    --profile workflow/profiles/local \
    -n
```

Run only one reproducible smoke pack:

```bash
uv run --frozen --no-sync snakemake \
    -s workflow/Snakefile \
    --configfile workflow/config.example.yaml \
    --profile workflow/profiles/local \
    results/dreems_example/raw/train/pack_0.pkl
```

Run the complete CPU smoke workflow by omitting the final target. The smoke
config uses the Python simulator only when the optional Rust extension is not
installed. It reduces corpus size, graph size, SNP count, and optimizer steps;
it must not be used as a scientific training run.

## Production Run

Build the Rust simulator and install CUDA JAX before generating the full corpus:

```bash
module load rust/1.92.0
uv sync --frozen --extra cuda --group dev --group workflow --group rust
uv run maturin develop --release --manifest-path rust/island_sim/Cargo.toml
```

On Great Lakes, keep generated artifacts on scratch and let Snakemake submit all
worker jobs through its Slurm executor:

```bash
DREEMS_RUN_ROOT=/scratch/jonth_root/jonth1/jonth/dreems/topology_diffusion_bands \
workflow/run_great_lakes.sh
```

The controller uses Slurm account `jonth0`. The production workflow requires
61,440 independent training simulations and 4,096 independent test simulations;
do not launch it as a smoke test. Paths and scheduler resources may
be overridden without changing the statistical recipe. Changes under
`simulation` or `dreems` define a different experiment.

After training, run the calibration DAG against the same test corpus:

```bash
uv run --frozen --no-sync snakemake \
    -s workflow/calibration.Snakefile \
    --configfile workflow/config.calibration.yaml \
    --profile workflow/profiles/slurm
```

Override `checkpoint`, `raw_source_dir`, and `outdir` with `--config` when the
production run lives on scratch rather than under `results/`.

New corpora use `raw_source_layout: packs`. Existing corpora with
`raw/{split}/chunk_N.pkl` remain readable by setting `raw_source_dir` and
`raw_source_layout: examples`; no conversion is required for archived runs.

The main workflow records the statistical procedure and deterministic
simulation seeds. Floating-point training can vary slightly across JAX,
accelerator, and driver versions, so a new fit is not promised to be
byte-identical to the packaged checkpoint.

FEEMS head-to-head evaluation is intentionally separate from model training.
FEEMS is neither an input to the released model nor part of its training loss.

## FEEMS Head-To-Head

`h2h.Snakefile` compares the packaged checkpoint with FEEMS on a fixed
synthetic test set. Each example must contain genotypes, sample-to-node
assignments, graph geometry, and the generating edge log migration rates.
The test set is an input artifact; this workflow does not regenerate it.

The FEEMS baseline follows the v2 procedure in the official
[cross-validation notebook](https://github.com/NovembreLab/feems/blob/main/docsrc/notebooks/cross-validation.ipynb).
It jointly selects `lambda` and `lambda_q` on the notebook's 10 by 5 grids.
FEEMS' default `n_folds=None` holds out one observed deme at a time, scores
predicted allele frequencies at that deme, and refits all samples using the
selected penalties. CV uses the notebook's `factr=1e10`; the selected model is
then refit with `SpatialGraph.fit` defaults, including the stricter
`factr=1e7`. Each observed-deme fold is a separate Snakemake job. A final rule
collects the fold scores, selects the two penalties, and refits once. The grid
order, warm starts, score, and final fit match FEEMS. The locked optional
dependency pins FEEMS commit `aa890b0e8ef583dd657a6a696af591b732d1d8be`.

Install the FEEMS and workflow dependencies:

```bash
uv sync --frozen --extra feems --group workflow --group dev
```

On Great Lakes, `scikit-sparse` must be built against the SuiteSparse copy in
the Sage module:

```bash
module load sage/9.6
CFLAGS="-I${SAGE_INC}" \
CPPFLAGS="-I${SAGE_INC}" \
LDFLAGS="-L${SAGE_LIB} -Wl,-rpath,${SAGE_LIB}" \
uv sync --frozen --extra feems --group workflow --group dev
```

Set the input paths in `config.h2h.yaml`, inspect the DAG, then run it through
the appropriate Snakemake profile:

```bash
uv run --frozen --no-sync snakemake \
    -s workflow/h2h.Snakefile \
    --configfile workflow/config.h2h.yaml \
    --profile workflow/profiles/slurm \
    -n

uv run --frozen --no-sync snakemake \
    -s workflow/h2h.Snakefile \
    --configfile workflow/config.h2h.yaml \
    --profile workflow/profiles/slurm
```

FEEMS and DREEMS are separate per-example rules, and FEEMS CV is split into one
rule per held-out deme. A failed fold can be retried without repeating the
others. A checkpoint change therefore reuses FEEMS results, while a FEEMS
implementation change leaves neural results untouched. The aggregate refuses
results whose protocol is not the documented leave-one-observed-deme-out
procedure. It also depends on a numerical parity test between the official
serial FEEMS routine and the fold-parallel adapter.
