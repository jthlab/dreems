# DREEMS Training And Calibration Workflow

This directory contains the Snakemake paths used to generate the synthetic
corpus, construct DREEMS features, fit the model, and evaluate held-out
simulations. The production release has four relevant configs:
`config.snp_range_experiment.yaml` records the broad-SNP corpus,
`config.sampling_robustness.yaml` records the derived sampling masks,
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
   sampling-mask views, while 1,152 validation examples drive early stopping.
   Views of the same held-out graph are not independent graphs.
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
the selected checkpoint, and verifies that calibration preserves the parameter
bytes. `tests/test_production_workflow.py` pins the final hashes and contract.

## Statistical Recipe

Each synthetic example is generated independently with a deterministic seed.
`simulation.seed` takes precedence over the top-level `seed`; if both are absent
or null, the base seed is 12001. The worker snapshot records the resolved seed.
The historical split and example-index offsets are unchanged, so existing raw
packs with explicit seeds remain reproducible.

The base production corpus uses:

- the sparse Rust island coalescent simulator;
- sample covariance accumulated directly during simulation;
- symmetric migration and diploid samples;
- markers filtered to derived-allele frequency 0.05-0.95, or MAF at least 0.05;
- 50-10,000 retained SNPs per example, balanced across four log-scale bins;
- masked grids, FEEMS triangular lattices, and global landmass graphs;
- graph sizes extending to roughly 1,400 nodes, broad sampling density, and
  variable local coalescence rate `q`;
- constant, smooth, barrier, Gaussian-process, earth-like, and
  landscape-conditioned migration surfaces.

The production setting is `island_marker_model: conditioned_random_marker`:
complete a genealogy, choose one marker uniformly along its total branch length,
then apply the frequency filter. Every completed tree gets a marker, so this
mode does not reject trees according to mutation probability. `first_mutation`
and the small-graph msprime adapter use different marker-selection procedures;
they are not interchangeable backends for this production recipe. The Python
island fallback implements the same three marker modes, but seeds are specific
to the backend. Production generation requires Rust explicitly.

The workflow downloads the pinned Natural Earth 50m land archive once on the
controller, verifies its SHA-256, and exposes the extracted Cartopy cache as a
shared input to every landscape job. Compute workers do not need internet
access.

The production predictor uses the leading 20 components of the observed
covariance, transforms those signals through five fixed graph-diffusion bands,
and includes the all-pair diagonal-adjoint linearized-resistance conditioner.
Coordinate axes and signed edge directions are excluded from model features;
endpoint summaries are symmetric and path support is shared across all equally
short routes. This removes explicit dependence on coordinate axes and edge
orientation. Floating-point rounding and nonunique eigenvectors at repeated
covariance eigenvalues can still perturb the computed features under equivalent
graph representations.
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

### Prepared Packs And Normalization

Preparation writes ordered lists of feature dictionaries plus a small dimension
sidecar. Training scans those sidecars to choose fixed padding sizes; it reads
the pack itself only when dimension metadata is absent. Feature widths must
agree across examples and packs. Counts in metadata are checked before integer
conversion, so a fractional count cannot silently change a training limit.

Normalization sums each feature and its square in float64 over training data
only. Node and edge statistics are weighted by their row counts. The stored
target moments use uncentered log rates; normalization
then centers each graph's targets and divides by the saved target scale.
Feature SDs use the configured floor; target SDs retain the existing `1e-6`
offset. Validation and inference reuse these training statistics.

Streaming batches shuffle packs and examples with a fixed seed. Paired sampling
views stay together with their anchor; group sizes must agree and divide the
batch size. Leftover examples carry into the next pack or epoch. An empty input
set raises an error. Progress is written as flushed JSON
records, while Snakemake owns scheduling, retries, and reuse of completed files.

The evaluation worker can read multiple packs. Its `replicate` IDs follow input
order across those packs even though prediction batches are sorted by graph
size. Prediction arrays have matching replicate IDs and offsets; CSV rows are
sorted by replicate ID. Separate evaluation jobs also retain their source pack,
and aggregation assigns `replicate_global` across jobs. CSV and NPZ outputs are
replaced only after serialization completes.

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

The Great Lakes profile queries live job status with `squeue`, rather than
waiting for accounting records to appear in `sacct`. Completed jobs currently
remain visible for four hours (`MinJobAge=14400`). On another cluster, check this
retention setting or override it with `--slurm-status-command sacct`.

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
DREEMS_SNAKEFILE=workflow/Snakefile \
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

FEEMS folds and final fits call `dreems.feems_benchmark` directly. They depend on
the FEEMS adapter, shared input/score utilities, and the locked environment, not
on the neural model or gallery renderer. DREEMS cases track the checkpoint plus
feature, inference, refinement, and calibration code. Each method has its own
Snakemake benchmark file. The combined `python -m dreems.h2h` commands remain
available for direct use.

Real-data rendering tracks grid resolution, topography limits, zoom, and DPI as
rule parameters. Changing those settings redraws maps without refitting the
model. An environment lock change can still invalidate both methods: numerical
library changes need verification too.

`real_data_gallery.Snakefile` consumes prepared covariance NPZs and their JSON
metadata. These are archived, already-filtered inputs; the inference worker does
not apply MAF filtering, imputation, or other genotype QC. For a new dataset, use
the public genotype/xarray API to prepare inputs first. New prediction reports
record hashes of both prepared files and the checkpoint. The renderer checks
recorded input hashes and array alignment before drawing a panel. Older reports
remain readable, but their input identity cannot be established retroactively.

Topographic tiles use normal HTTPS certificate verification. Sites with a local
certificate authority can configure Requests through `REQUESTS_CA_BUNDLE`.
If a download fails, rendering warns and uses the lower-resolution stock map;
configuration and programming errors are not treated as network failures.
Rendering, including interpolation and map clipping, does not alter edge
estimates. It also does not add environmental covariates to inference.

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

### Reusing Saved FEEMS Fits

Set `feems_archive` and `feems_archive_checksum` to reuse an existing
`feems_results.tar.gz` instead of fitting FEEMS again. The restore rule checks
the supplied archive's digest even if it has been renamed or moved. It accepts
only the expected regular JSON/NPZ files, validates all requested cases, and
then replaces those files individually. Other results in the work directory
are left alone. Packaging uses one deterministic archive writer for both newly
computed and restored fits.

New FEEMS and DREEMS case reports record the SHA-256 of their input pickle.
Archive reuse requires that hash to match the current input; aggregation also
rejects a pair with different recorded hashes. These are byte-level checks, so
benchmark input files should be kept unchanged. The restore rule additionally
checks graph coordinates, edge order, generating truth, sample counts, SNP
count, and the declared LOO protocol. Loading and checking the raw inputs runs
in a separate CPU job.

The historical release archives predate input hashes. Their configs explicitly
set `allow_legacy_feems_archive: true`. Those runs still perform the structural
checks, but `feems_archive_verification.json` labels their genotype identity
`legacy_unverified`: matching geometry and sample counts cannot prove that
the original fit used the same genotype matrix. The workflow does not add a
new hash to an old fit and present it as original evidence. This opt-in is for
reproducing retained results; new benchmarks should leave it disabled.

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

### Fresh Release Training

`release.Snakefile` composes the existing generation/training rules with
sampling-view preparation. It uses the scientific settings in
`config.snp_range_experiment.yaml` and `config.full_retrain.yaml`, replaces their
historical storage paths, and applies overrides from `config.release.yaml`.
The default run has 61,440 training examples and 4,096 additional examples for
validation and diagnostics. It is a full production run.

On Great Lakes:

```bash
export DREEMS_RUN_ROOT=/scratch/jonth_root/jonth0/$USER/dreems/release
mkdir -p "$DREEMS_RUN_ROOT"
sbatch --output="$DREEMS_RUN_ROOT/controller-%j.out" \
    --error="$DREEMS_RUN_ROOT/controller-%j.err" \
    workflow/run_great_lakes.sh \
    --config publish_dir=/home/jonth/turbo/dreems/release
```

The controller uses the native Slurm executor under `jonth0`. Generation remains
one job per example; packing, sampling views, training, and evaluation have
separate rules. Both launchers default to this release workflow. On Maturin:

```bash
DREEMS_RUN_ROOT=/path/to/work bash workflow/run_maturin.sh \
    --config publish_dir=/path/to/models
```

The release workflow keeps its `.snakemake` state and default worker logs under
`outdir`. Relative storage paths are resolved before
changing the working directory. Snakemake opens its controller log before reading
that setting, so a small `.snakemake/log` directory can remain in the checkout.
To put that log on scratch too, pass `--directory "$DREEMS_RUN_ROOT"` and use
absolute storage paths in any config overrides.

Set `raw_source_dir` to an existing corpus's `raw` directory to reuse it. Set
`train_examples`, `test_examples`, and `raw_pack_size` to that corpus's counts
and packing scheme. A new run gets its own `outdir`; the source packs remain
inputs rather than targets to regenerate. The workflow prepares the sampling
views itself, with no dependency on a pretrained checkpoint.

The core uses all selected training masks and the first
`core_validation_mask_packs` of masked validation data. The later gate uses all
selected masks and ordinary replay packs, with the core frozen. Graph features
are configured once for both ordinary and masked examples. Optimizer changes
do not invalidate those features. A changed mask seed does invalidate masked
views, but not raw simulations or ordinary replay features.

The gate's validation limit includes every selected validation pack. A partial
last raw pack can make the actual count smaller than that limit; the training
summary records the loaded count. This does not discard validation examples.
The ordinary `eval/summary.json` includes early-stopping validation examples and
is a training diagnostic. Use `h2h.Snakefile` for
independent model comparisons. SD calibration and verified package promotion
can run separately or through the optional `validated_release` target below.

### Sampling-View Preparation

`sampling_robustness.Snakefile` derives sampling views without generating another
SNP corpus or fitting a model. It selects densely observed
simulations from the existing raw train and test splits, then derives four
observations from each simulation: the original dense sample, a random 10%
sample, two separated outer regions, and a sample with one end of the graph
missing. All four observations retain the same graph, covariance submatrix,
and migration truth. Batches keep each four-member group together so the loss
can compare sparse predictions with their dense anchor.

It also prepares ordinary replay packs from the source corpus. The release
trainer combines these with the masked examples. The old initializer-gate
fine-tune and whole-model fine-tune have been removed from this preparation
workflow; they were not stages of the released from-scratch model.

Selection is a Snakemake checkpoint, so its input metadata can be generated
earlier in the same DAG. Standalone preparation discovers an existing corpus;
`source_pack_counts: {train: N, test: M}` declares packs that do not yet exist.
The release entry point supplies those counts automatically.

```bash
uv run --frozen --no-sync snakemake prepared \
    -s workflow/sampling_robustness.Snakefile \
    --configfile workflow/config.sampling_robustness.yaml \
    --profile workflow/profiles/slurm \
    --config source_corpus=/path/to/corpus outdir=/path/to/views
```

### Full Retraining

The commands below reuse existing raw packs. On another machine, set
`source_corpus` in the sampling config to the corpus root and `raw_source_dir`
in the full-retrain config to its `raw` subdirectory. Update the output and
supplemental-pack paths too.
The checked-in absolute paths record the release run; replace them when running elsewhere.

For the historical two-command procedure, first build sampling views and replay
packs. This target needs no checkpoint:

```bash
DREEMS_SNAKEFILE=workflow/sampling_robustness.Snakefile \
DREEMS_CONFIG=workflow/config.sampling_robustness.yaml \
sbatch workflow/run_great_lakes.sh prepared \
  --config source_corpus=/path/to/corpus outdir=/path/to/views
```

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

On Maturin, use the same environment variables with
`bash workflow/run_maturin.sh` instead of `sbatch workflow/run_great_lakes.sh`.
Both launchers accept `DREEMS_CONFIG_FILES` and `DREEMS_SNAKEFILE`. Training uses
a private temporary staging directory and does not require Slurm environment
variables when run locally.

`tests/test_training_workflow.py` exercises simulation, preparation, training,
evaluation, publishing, and frozen-core gate training on a tiny CPU corpus. It
also checks that an unchanged rerun does nothing and that changing the learning
rate leaves raw data, training feature packs, and normalization up to date.
Temporary per-example evaluation features are rebuilt when needed. The test
checks workflow behavior on a deliberately tiny corpus. It also starts the release
entry point from an empty directory and checks that it creates sampling views,
trains both stages, evaluates the selected gate, calibrates its SD, and writes a
candidate in one invocation. The test's cached FEEMS results are synthetic test
fixtures. They test reuse and provenance.

### Evaluate A Release Candidate

The `validated_release` target connects the existing training, H2H, calibration,
and promotion rules. Enable it with a `release_validation` mapping in your
release config. Each benchmark accepts an ordinary H2H config file with local
overrides. For example:

```yaml
release_validation:
  broad:
    configfile: workflow/config.full_retrain_h2h.yaml
    input_pattern: /path/to/broad/example_{example}.pkl
    metadata_pattern: /path/to/broad/example_{example}.metadata.json
    feems_archive: /path/to/broad/feems_results.tar.gz
    feems_archive_checksum: /path/to/broad/feems_results.tar.gz.sha256
  hard:
    configfile: workflow/config.full_retrain_hard_edges.yaml
    input_pattern: /path/to/hard/example_{example}.pkl
    metadata_pattern: /path/to/hard/example_{example}.metadata.json
    feems_archive: /path/to/hard/feems_results.tar.gz
    feems_archive_checksum: /path/to/hard/feems_results.tar.gz.sha256
```

The H2H configs declare the example IDs or range; override those too when using
a different benchmark. Use data independent of training and early stopping.
The workflow cannot establish independence from filenames alone. Config-file
paths are relative to the repository; data paths are resolved from the launch
directory before Snakemake changes its working directory. Absolute data paths
are least ambiguous on the cluster.

```bash
uv run --frozen --no-sync snakemake validated_release \
    -s workflow/release.Snakefile \
    --configfile /path/to/release.yaml \
    --profile workflow/profiles/slurm
```

This writes training artifacts under `outdir/model`, broad and hard reports and
galleries under `outdir/benchmarks`, and the calibrated candidate at
`outdir/release/dreems_candidate.dreems.zip`. It never replaces the packaged
checkpoint. Keep `outdir` on scratch; copy reviewed release artifacts to Turbo.
The default `all` target remains training-only.

Saved FEEMS archives are restored and checked by the H2H module. Changing the
neural training settings does not refit FEEMS or regenerate raw simulations.
Without an archive, the ordinary H2H rules fit FEEMS; inspect a dry run before
launching. New neural predictions record their checkpoint's SHA-256. This
target requires matching hashes during calibration, so predictions from an
older model cannot silently calibrate a new one.

The broad predictions are partitioned into disjoint SD fit, selection, and
verification subsets. Only the last subset estimates held-out coverage.
The H2H reports describe the selected gate before SD recalibration.
The current inference code uses raw model SDs for the MAP prior, so attaching
calibration metadata leaves both neural means and MAP estimates unchanged.
Evaluate the final candidate with `h2h.Snakefile` before reporting its performance.
Promotion verifies lineage and consistency; judging accuracy and improvement
over FEEMS still requires reviewing those results.

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

Legacy prediction archives without checkpoint hashes remain supported by this
standalone workflow for historical reproduction. Their checkpoint identity
cannot be verified. Set `require_checkpoint_hash: true` for new runs; that
requires new neural predictions while reusing saved FEEMS fits.
