# DREEMS Training And Calibration Workflow

This directory contains the Snakemake paths used to generate the synthetic
corpus, construct DREEMS features, fit the model, and evaluate held-out
simulations. The production release has four explicit stages:
`config.production.yaml` records the graph-balanced base fit,
`config.sampling_robustness.yaml` records the counterfactual sampling-mask
fine-tune, `config.null_surface_gate.yaml` records the frozen-core surface-gate
fit, and `config.surface_gate_release.yaml` records fresh SD calibration and
release packaging. `config.example.yaml` is a small CPU smoke run.

## Released Model Provenance

The packaged checkpoint is
`src/dreems/data/dreems_diffusion_bands.dreems.zip`. Its archive SHA-256 is:

```text
10d21a6326e848f48a7eccdd1a508d86f54effc55f79774486a626037a9e3530
```

The serialized Equinox parameters have SHA-256
`d46045e3a5bbdc870427e2ca93b07daaac5a1d94ff7bbaa25e8f82972e3bec76`.
The package version is 0.2.0.

The model lineage is:

1. The graph-balanced base model was fit from scratch on 61,440 simulations,
   with 1,024 of 4,096 held-out simulations used for validation. Its archive
   SHA-256 was `f0b677d2cc028252002d15fac843ca7347b1b352eb2286a18bd202b724aa92b4`.
2. `sampling_robustness.Snakefile` added structured sampling masks and replayed
   ordinary examples, then fine-tuned the complete v2 initializer-gate model.
   The selected candidate SHA-256 is
   `0e3e56c2c3098c3f0b04d0dfdad067ddb6bb1a928c79ca9048dec505763c61d3`.
3. `null_surface_gate.Snakefile` froze that candidate and trained only a
   graph-level gate. The raw gate checkpoint SHA-256 is
   `df21d92755edeb9f37ac1dcc4251f4f1523f0096ce3ea5c6f914dec41075bdfc`.
   The selected validation loss was `0.14434671436902136` at step 9,250; the
   run completed 10,000 steps and restored the selected state.
4. `surface_gate_release.Snakefile` fit SD calibration on cached independent
   predictions and wrote the deterministic production archive above.

`promote_surface_gate.py` verifies exact equality of every pre-gate Equinox
parameter between stages 2 and 3, checks the complete 41-point gate training
history, and verifies that calibration changes metadata but not parameter
bytes. `tests/test_production_workflow.py` pins the final hashes and contract.

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

Training uses single-device Adam with learning rate `5e-5`, global gradient
norm clipping at 1, batch size 8, and at most 120,000 steps. It checks
validation every 1,000 steps, uses patience 12 after a 20,000-step warmup, and
restores the best checkpoint.

The sampling-robustness stage adds initializer-regret and paired-mask
consistency terms while fine-tuning the complete model at learning rate
`1e-5`. The final stage freezes those parameters and trains only the surface
gate. Its loss includes the same prediction terms plus binary cross-entropy
for whether the true centered edge surface has RMS greater than 0.05.

Fresh posterior-SD calibration uses 688 independent gated predictions,
stratified by simulation family and sampled-node density into 230 fit, 227
selection, and 231 untouched verification graphs. Density-aware scaling was
selected by graph-balanced Gaussian NLL. On verification, graph-balanced NLL
changed from 0.0823 to 0.0212, 80% coverage changed from 80.18% to 81.55%, and
95% coverage changed from 92.89% to 93.67%. The factorized Gaussian posterior
should not be read as a complete joint uncertainty model.

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

sampling-mask replay -> v2 initializer-gate candidate
    -> frozen-core v3 surface-gate checkpoint
    -> independent gated predictions
    -> calibration fit (230 graphs)
    -> calibration selection (227 graphs)
    -> calibration verification (231 graphs)
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

### Joint Full Retraining

`config.full_retrain.yaml` is layered after `config.snp_range_experiment.yaml`.
It reuses the saved Rust/MAF-conditioned raw covariance corpus, re-prepares all
features with the current node-covariance convention, and trains the complete
GNN plus initializer and surface gates from random initialization. The broad
61,440-example corpus remains dominant; 8,192 counterfactual sampling views
make up 11.8% of the combined stream. No checkpoint is passed to the trainer.

```bash
DREEMS_CONFIG_FILES=workflow/config.snp_range_experiment.yaml:workflow/config.full_retrain.yaml \
DREEMS_SNAKEFILE=workflow/Snakefile \
sbatch workflow/run_great_lakes.sh \
  /home/jonth/turbo/dreems/full_retrain_20260902/checkpoint.best.dreems.zip
```

The source corpus path and all output paths are explicit in the config. The
prepared masks, checkpoints, validation histories, and evaluations are managed
as separate Snakemake artifacts. Changing a training hyperparameter therefore
does not repeat covariance simulation or mask construction.

### Surface Gate And Release

`null_surface_gate.Snakefile` loads the selected sampling-robustness candidate,
adds one graph-level gate, and freezes every preceding parameter. The gate
learns when a homogeneous centered surface is preferable to the full edge
prediction. Its held-out comparisons are separate from the gate-training
validation set.

`surface_gate_release.Snakefile` then recalibrates posterior SD from cached
independent predictions. It uses deterministic family- and density-balanced
fit, selection, and verification splits, checks parameter lineage, and writes
the packaged checkpoint atomically:

```bash
uv run --frozen --no-sync snakemake \
    -s workflow/surface_gate_release.Snakefile \
    --configfile workflow/config.surface_gate_release.yaml \
    --profile workflow/profiles/slurm
```

The checked-in release config records the exact archived inputs used for the
0.2.0 package. Override its scratch and Turbo paths only when reproducing the
same promotion from a relocated artifact archive.
