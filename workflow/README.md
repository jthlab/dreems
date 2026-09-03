# DREEMS Training And Calibration Workflow

This directory contains the Snakemake paths used to generate the synthetic
corpus, construct DREEMS features, fit the model, and evaluate held-out
simulations. The production release has three explicit configs:
`config.snp_range_experiment.yaml` records the broad-SNP corpus,
`config.full_retrain.yaml` records the from-scratch core and frozen surface-gate
fits, and `config.surface_gate_release.yaml` records fresh SD calibration and
release packaging. `config.example.yaml` is a small CPU smoke run.

## Released Model Provenance

The packaged checkpoint is
`src/dreems/data/dreems_diffusion_bands.dreems.zip`. Its archive SHA-256 is:

```text
454cb50f4c7ef1213e001d700d1bf9e9f6a353b5c8be5619aae06e9d7a22cc15
```

The serialized Equinox parameters have SHA-256
`19a12e428d54e6b437ba7928e935cd1dbb8434ec83c34d506949b5fed84952ec`.
The package version is 0.3.0.

The model lineage is:

1. The broad corpus contains 61,440 Rust coalescent simulations spanning
   50-10,000 retained SNPs. The core training stream adds 8,192 structured
   sampling-mask views, while 1,152 held-out graphs drive early stopping.
2. `Snakefile` fit the v2 initializer-gate model from random initialization.
   Its selected checkpoint SHA-256 is
   `86cd7b1aecf36ce5bf0cf3a709cc3b7be721baa4b356d9578595ad3d9d0cc671`.
   Validation loss was lowest at step 38,000; training stopped at step 50,000
   and restored that state.
3. The same workflow froze the core and trained only the graph-level surface
   gate. The selected checkpoint SHA-256 is
   `7ad003203fa3cfe9e3f196d3f724a1cdfc984bdfd77bf3bb21630e9d51c7cf5c`.
   Validation loss was lowest at step 9,000; the run completed 10,000 steps
   and restored that state.
4. `surface_gate_release.Snakefile` fit SD calibration on cached independent
   predictions and wrote the deterministic production archive above.

`promote_surface_gate.py` verifies exact equality of every pre-gate Equinox
parameter between stages 2 and 3, checks the complete 51-point core and
41-point gate histories, confirms that the broad and hard-case reports used
the selected checkpoint, and verifies that calibration changes metadata but
not parameter bytes. `tests/test_production_workflow.py` pins the final hashes
and contract.

## Statistical Recipe

Each synthetic example is generated independently with a deterministic seed.
The base production corpus uses:

- the sparse Rust island coalescent simulator;
- direct sample-covariance output rather than retained genotype matrices;
- symmetric migration and diploid samples;
- markers conditioned on MAF 0.05-0.95;
- 50-10,000 retained SNPs per example, balanced across four log-scale bins;
- masked grids, FEEMS triangular lattices, and global landmass graphs;
- graph sizes extending to roughly 1,400 nodes, broad sampling density, and
  variable local coalescence rate `q`;
- constant, smooth, barrier, Gaussian-process, earth-like, and
  landscape-conditioned migration surfaces.

The workflow downloads the pinned Natural Earth 50m land archive once on the
controller, verifies its SHA-256, and exposes the extracted Cartopy cache as a
shared input to every landscape job. Compute workers do not need internet
access.

The production predictor uses the leading 20 components of the observed
covariance, transforms those signals through five fixed graph-diffusion bands,
and includes the all-pair diagonal-adjoint linearized-resistance conditioner.
Coordinate axes and signed edge directions are excluded from model features;
endpoint summaries are symmetric and path support is shared across all equally
short routes. Uniform rotation, reflection, scale, edge reversal, and node
relabeling therefore do not change predictions.
The neural network is a 10-layer, 128-channel edge-aware residual GNN with a
three-layer context branch. The prepared node and edge dimensions are 109 and
253. An edgewise gate blends the neural proposal with the linearized
initializer. A second, graph-level gate multiplies the centered surface by a
number in `[0, 1]`, allowing the model to represent homogeneous migration
without retaining unsupported spatial contrast.

The base fit minimizes the objective below. Every term is first reduced within
a graph and then averaged across the batch, so graph size does not set a
simulation's influence:

```text
edge Gaussian NLL
  + 0.50 * context edge MSE
  + 0.50 * node-q Gaussian NLL
  + 0.25 * edge-slope loss
  + 0.20 * centered edge-correlation loss
  + 0.35 * high-versus-low edge-orientation loss
```

Core training uses single-device Adam with learning rate `5e-5`, global gradient
norm clipping at 1, batch size 8, and at most 120,000 steps. It checks
validation every 1,000 steps, uses patience 12 after a 20,000-step warmup, and
restores the best checkpoint.

The core loss also includes initializer-regret and paired-mask consistency
terms. Counterfactual views are 11.8% of the core training stream. The final
stage freezes every core parameter and trains only the surface gate for 10,000
steps at learning rate `5e-4`. Its loss includes binary cross-entropy for
whether the true centered edge surface has RMS greater than 0.05.

Fresh posterior-SD calibration uses 512 cached predictions from the immutable
broad H2H, stratified by simulation family and sampled-node density into 171
fit, 171 selection, and 170 untouched verification graphs. Density-aware
scales of 0.939, 0.935, and 0.918 were selected by graph-balanced Gaussian NLL.
On verification, graph-balanced NLL changed from 0.1033 to 0.0994, 80%
coverage changed from 84.52% to 82.02%, and 95% coverage changed from 95.95%
to 94.83%. The factorized Gaussian posterior should not be read as a complete
joint uncertainty model.

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

base simulations + sampling-mask views -> v2 initializer-gate core
    -> frozen-core v3 surface-gate checkpoint
    -> immutable broad-H2H predictions
    -> calibration fit (171 graphs)
    -> calibration selection (171 graphs)
    -> calibration verification (170 graphs)
    -> deterministic packaged checkpoint
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
DREEMS_RUN_ROOT=/scratch/jonth_root/jonth0/jonth/dreems/topology_diffusion_bands \
workflow/run_great_lakes.sh
```

## Broad SNP-Count Retraining

`config.snp_range_experiment.yaml` retrains DREEMS from scratch on the same
graphs, migration surfaces, sampling designs, architecture, and loss as the
graph-balanced base model. It changes only the SNP-count distribution. The four ranges
below receive equal numbers of examples, with counts drawn log-uniformly inside
each range:

```text
50-128
129-512
513-2,000
2,001-10,000
```

The base corpus used 2,000-10,000 SNPs. This experiment tests whether
DREEMS can learn the much stronger finite-SNP noise found in small datasets
without increasing the base corpus's upper limit. Existing training
examples cannot be downsampled exactly because they retain only the final
covariance, not individual genotype columns. The new covariance corpus must
therefore be simulated.

On Great Lakes, launch the experiment with the Rust simulator required:

```bash
DREEMS_BUILD_RUST=1 \
DREEMS_CONFIG=workflow/config.snp_range_experiment.yaml \
DREEMS_RUN_ROOT=/scratch/jonth_root/jonth0/jonth/dreems_snp_range \
sbatch workflow/run_great_lakes.sh
```

This experiment does not modify the packaged checkpoint.

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

Compare the base and broad SNP-count checkpoints on that same independent
test corpus with:

```bash
uv run --frozen --no-sync snakemake \
    -s workflow/snp_range_comparison.Snakefile \
    --configfile workflow/config.snp_range_comparison.yaml \
    --profile workflow/profiles/slurm
```

The comparison evaluates each checkpoint once per 64-example pack, keeps the
paired edge predictions on scratch, and writes the aggregate table and visual
gallery to `publish_dir`. Set the checkpoint, corpus, and output paths in the
config file or override them with `--config`; the test examples are never
regenerated by this workflow.

New corpora use `raw_source_layout: packs`. Existing corpora with
`raw/{split}/chunk_N.pkl` remain readable by setting `raw_source_dir` and
`raw_source_layout: examples`; no conversion is required for archived runs.

The main workflow records the statistical procedure and deterministic
simulation seeds. Floating-point training can vary slightly across JAX,
accelerator, and driver versions, so a new fit is not promised to be
byte-identical to the packaged checkpoint.

FEEMS head-to-head evaluation is intentionally separate from model training.
FEEMS is neither an input to the production model nor part of its training loss.

## FEEMS Head-To-Head

`h2h.Snakefile` compares the packaged checkpoint with FEEMS on a fixed
synthetic test set. Each example must contain genotypes, sample-to-node
assignments, graph geometry, and the generating edge log migration rates.
Ordinary H2H runs treat the test set as an input artifact. A config with
`scenario_config` instead generates each declared scenario as its own cached
Snakemake job.

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

### Sampling-Mask Audit

`config.feems_scenarios.yaml` crosses barrier, anisotropic, and homogeneous
truths with four fixed sampling masks over three simulation replicates. It
uses the production Rust coalescent and MAF ascertainment. The report separates
the linearized initializer, raw DREEMS estimate, MAP refinement, and FEEMS LOO,
then measures whether estimates remain suspiciously similar when the truth
changes but the sampling mask does not.

```bash
DREEMS_SNAKEFILE=workflow/h2h.Snakefile \
DREEMS_CONFIG=workflow/config.feems_scenarios.yaml \
sbatch workflow/run_great_lakes.sh
```

### Sampling-Robustness Training

`sampling_robustness.Snakefile` repairs sensitivity to where samples were
collected without generating another SNP corpus. It selects densely observed
simulations from the existing raw train and test splits, then derives four
observations from each simulation: the original dense sample, a random 10%
sample, two separated outer regions, and a sample with one end of the graph
missing. All four observations retain the same graph, covariance submatrix,
and migration truth. Batches keep each four-member group together so the loss
can compare sparse predictions with their dense anchor.

The training mix also replays ordinary examples from the source corpus. This
guards against improving the structured masks by degrading the broader data
distribution. The workflow first trains the small initializer gate with the
graph-balanced neural core frozen, then fine-tunes the complete model at a lower
learning rate. Both checkpoints are evaluated on held-out replay and paired
counterfactual examples.

```bash
DREEMS_SNAKEFILE=workflow/sampling_robustness.Snakefile \
DREEMS_CONFIG=workflow/config.sampling_robustness.yaml \
sbatch workflow/run_great_lakes.sh
```

### Full Retraining

`config.full_retrain.yaml` is layered after `config.snp_range_experiment.yaml`.
It reuses the saved Rust/MAF-conditioned raw covariance corpus, re-prepares all
features with the current node-covariance convention, and trains the GNN plus
initializer gate from random initialization. The broad 61,440-example corpus
remains dominant; 8,192 counterfactual sampling views make up 11.8% of the
combined stream. Snakemake then adds the surface gate and trains only that head
on balanced original and counterfactual views. Freezing the core during this
second stage prevents the gate objective from destabilizing the edge model.

```bash
DREEMS_CONFIG_FILES=workflow/config.snp_range_experiment.yaml:workflow/config.full_retrain.yaml \
DREEMS_SNAKEFILE=workflow/Snakefile \
sbatch workflow/run_great_lakes.sh \
  /home/jonth/turbo/dreems/full_retrain_20260902/checkpoint.best.dreems.zip
```

The source corpus path and all output paths are explicit in the config. Core
and gate checkpoints, prepared masks, validation histories, and evaluations are
separate Snakemake artifacts. The publish rule copies both selected training
stages and their complete histories to Turbo. Changing a training
hyperparameter therefore does not repeat covariance simulation or mask
construction.

### Surface Gate And Release

The staged section of `Snakefile` adds one graph-level gate to the selected
full-retrain core and freezes every preceding parameter. The gate learns when
a homogeneous centered surface is preferable to the full edge prediction.

`surface_gate_release.Snakefile` recalibrates posterior SD from cached broad-H2H
predictions. It uses deterministic family- and density-balanced fit, selection,
and verification splits, checks parameter lineage and H2H provenance, and
writes the packaged checkpoint atomically:

```bash
uv run --frozen --no-sync snakemake \
    -s workflow/surface_gate_release.Snakefile \
    --configfile workflow/config.surface_gate_release.yaml \
    --profile workflow/profiles/slurm
```

The checked-in release config records the exact archived inputs used for the
0.3.0 package. Override its scratch and Turbo paths only when reproducing the
same promotion from a relocated artifact archive.
