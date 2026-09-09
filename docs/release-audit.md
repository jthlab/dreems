# Release code audit

This audit is complete as of September 9, 2026. It covers the current package and the workflow needed
to reproduce its simulations, training, validation, and published comparisons.
It is not another model experiment. Earlier sections retain the progress notes
from each pass; the final verification is recorded at the end. Start with the
[short summary](release-summary.md). At audit completion, changes had not yet
been committed or pushed.

## Compatibility requirements

- Keep public call signatures and the fields of saved objects.
- Preserve the order of `DREEMS` fields and random-key splits in its constructor.
  Equinox checkpoints depend on the former; seeded initialization depends on both.
- Use JAX and Equinox for model computation and Equinox leaf serialization for
  weights. Do not introduce another model format.
- Separate behavior fixes from refactors and test both. A refactor should not
  require new simulations, retraining, or rerunning FEEMS.
- Leave the existing 1,000-SNP gallery work in the checkout untouched.

## Progress

- [x] Inspect the neural model, losses, and checkpoint format.
- [x] Consolidate batched inference and remove repeated model-field lists from
  gate upgrades. Upgrades now replace only the new gate on the existing model.
- [x] Make the existing beartype/jaxtyping checks explicit on model and loss
  functions instead of installing them through a package import hook.
- [x] Document the output units, masks, gates, and loss definitions. Share the
  centering operation used by the three shape losses.
- [x] Compare original and refactored predictions, gradients, initialized weights,
  gate upgrades, and Equinox parameter bytes.
- [x] Consolidate checkpoint I/O and use one Equinox update function for full-model
  and gate-only training. Add enforced types to training configs and entry points.
- [x] Audit validation failure handling and training metadata; stop invalid runs
  without replacing a previously saved best checkpoint.
- [x] Consolidate NumPy and xarray input validation, preserve mapping aliases,
  and document row order, precision counts, output units, and refinement defaults.
- [x] Audit genotype QC, missing-call masks, ploidy assumptions, and conversion types.
- [x] Consolidate result validation and refinement input conversion. Check seeded
  sampling masks, preserve aligned sample representations, and retire the unused
  heuristic shrinkage module without changing the learned production gate.
- [x] Finish the public API type/documentation audit.
- [x] Consolidate covariance factors and geographic conductances; compare feature
  construction and the all-pair initializer against frozen source.
- [x] Audit numerical refinement, sparse-solve failures, and trust-region bounds.
- [x] Consolidate simulator dispatch and Python genealogy loops; check seeded
  Rust outputs, usable-SNP counts, ploidy, and sample ordering.
- [x] Repair corpus seed inheritance and atomic worker output; check file formats,
  seeded draws, and workflow reuse.
- [x] Reject invalid dosage arrays, mixture weights, counts, and sampling ranges.
  Preserve seeded draws and ordered graph arrays; repair empty-sample retries.
- [x] Finish the remaining graph/sampling documentation and parameter-semantics review.
- [x] Trace the core/gate training DAG and archive the retired null-surface and
  forward-evidence experiment workflows outside the checkout.
- [x] Exercise local simulation, sampling-view preparation, core training,
  frozen-core gate training, publishing, and cache reuse through Snakemake.
- [x] Verify the same stages in one fresh-run DAG, without a saved corpus or
  pretrained checkpoint.
- [x] Share calibration fitting and metrics, validate observations and splits,
  and reproduce the packaged SD scales from all 512 saved release predictions.
- [x] Finish reviewing benchmark, calibration, and real-data workflow dependencies.
- [x] Review the remaining rendering and scenario helpers for repeated I/O,
  unclear code, missing docstrings, and unenforced types.
- [x] Separate FEEMS workers from neural evaluation and share input validation
  and atomic report writers. Add dependency tests for inference and rendering.
- [x] Validate saved FEEMS archives without replacing unrelated results. Record
  input hashes for new fits and label the limitations of pre-hash archives.
- [x] Document and type the training workers, consolidate their normalization
  and report code, and test streaming batches and multi-pack evaluation IDs.
- [x] Share FEEMS graph checks and atomic fold writes, document its fitting
  modes, and reject silent index and complex-number conversions at public inputs.
- [x] Share atomic checkpoint writes, check promotion records and frozen tree
  structure, and reproduce the released archive byte for byte from retained inputs.
- [x] Record the checkpoint identity in new H2H predictions and calibration
  reports; reject mismatched predictions before creating a release candidate.
- [x] Compose training, independent H2Hs, SD calibration, and candidate creation
  in one optional target; test a fresh run and reuse of saved FEEMS results.
- [x] Share real-data report writers and raw-calibration archive readers, validate
  scenario settings and renderer inputs, and preserve retained numerical outputs.
- [x] Check prepared-input hashes before rendering, restore HTTPS verification
  for map tiles, and track scenario generator dependencies.
- [x] Share comparison and notebook writers, validate matched artifacts, and
  test imported notebook rules and sampling-view dependencies.
- [x] Run the complete test suite, workflow dry run, notebook, and wheel checks
  for the model, loss, training-step, and checkpoint changes.
- [x] Repeat release checks after the remaining audit work.

## Review Scope

The main release path is not the entire repository. The real-data workers,
scenario generator, and raw-case calibration worker have now been reviewed.
The gate comparison, exact notebook helpers, general benchmark renderer, and
SNP-range comparison workflow have also been reviewed. The final public API
and graph/sampling review passed its focused and seeded comparison checks.
The final full release checks passed as well. This audit does not establish new
predictive performance or justify changing the published numerical results.

## Findings

### The notebook worker no longer matched the genotype API

The earlier QC audit required signed integer calls, as used by sgkit. The
notebook worker still supplied unsigned bytes from its saved simulation. Unit
tests covered allele filtering and dataset construction separately, but not
their connection. The retained-data comparison exposed the resulting error
before any case could be rebuilt.

The worker now converts validated binary calls to signed bytes when constructing
the genotype dataset. No allele values, SNP choices, or MAF thresholds change.
A regression test follows unsigned calls through the public genotype path and
checks covariance against a direct calculation. The old-code comparison uses
the archived genotype reader from before the signed-input check. It does not
pretend the broken interim combination produced a valid result.

### Comparison reports could silently reinterpret saved inputs

The gate comparison converted scenario IDs, edge indices, and SNP counts to
integers before checking them. It also accepted negative SDs and did not verify
the fixed truth/mask order before labelling panels. The reader now validates
those arrays first and checks a gated checkpoint hash when the prediction
contains one. Legacy files without hashes still have that provenance limit.
The shared metric reader rejects complex inputs rather than discarding their
imaginary components; its existing handling of nonfinite pairs is unchanged.

CSV case IDs are now parsed as exact integers. This rejects fractional IDs
without rounding large identifiers through a float. Duplicate case IDs are
rejected before selecting gallery examples. Plots now handle all-zero MSEs
without taking a minimum of an empty array, and missing correlations no longer
discard the finite values of an entire method when choosing the reference line.
The label for post-MAP alpha now says it is a projection, not an exact gate
multiplier: refinement can change the pattern as well as the amplitude.

### Notebook preparation depended on shell substitution and module globals

The comparison workflow substituted paths into its source Markdown with `sed`.
Paths containing quotes or substitution characters could produce invalid code.
Preparation now parses MyST with Jupytext, replaces the single expected code
placeholder with a quoted Python path, and writes the notebook atomically.
The original Markdown is untouched; there is no intermediate Markdown copy.

Both notebook workflows now resolve their source directory when imported as
Snakemake modules. Shell commands use explicit rule parameters for the project
environment. Tests prepare notebooks through imported rules, including unusual
paths, and check that simulation calls, seeds, and SNP settings are untouched.
Feature and genotype-reader changes invalidate derived notebook inputs without
rerunning the shared base simulation. Model-only source edits do not invalidate
comparison-bundle construction; its requested checkpoint remains a dependency.
Common readers and the environment lock are tracked where they affect the result.

### Archived real-data inputs were converted before validation

The prediction worker cast edge indices and SNP counts to integers before
constructing the validated input object. Fractional values could be truncated;
complex coordinates and covariance values could lose their imaginary part.
The worker now passes the original arrays and count through the public
validator before loading weights. The raw calibration worker similarly rejects
complex errors and SDs before conversion.

Scenario settings now reject fractional counts, non-string names, invalid
probabilities, reversed barrier bounds, and nonpositive rates before drawing
any data. This preserves valid seeds, masks, truth fields, and simulation output.
The scenario generation rule now tracks the Python/Rust simulator sources and
locked environment, rather than just its command wrapper and config.

### Real-data panels could combine misaligned artifacts

Prediction reports now record hashes of the prepared covariance archive and
its metadata, in addition to the checkpoint hash. Inputs are checked again
before publishing predictions. Rendering checks recorded hashes, dataset IDs,
graph indices, vector lengths, and finite nonnegative SDs before plotting.
Legacy reports without hashes remain usable; their original input identity is
not retroactively verified. The files are replaced atomically one at a time;
Snakemake still owns recovery of a task with several outputs.

### Map downloads disabled TLS checks and hid unrelated errors

The tile downloader used `verify=False`, while its fallback caught every
exception and suppressed warnings. It now uses normal certificate checks,
validates downloaded image content before caching it, and warns on network or
I/O failure before falling back to stock imagery. Configuration and programming
errors propagate. Existing cached tiles remain usable without another download.

### Browser reports duplicated writers and could contain nonstandard JSON

Both real-data workers kept their own hash and report writers. They now use the
shared atomic I/O functions. Browser JSON converts nonfinite values to `null`,
including values nested inside NumPy arrays, and handles NumPy booleans. The
scientific report writer retains its previous representation. PNG/PDF and HTML
outputs are also replaced only after they have been written successfully.

### Cached neural predictions did not identify their checkpoint

H2H prediction NPZs and per-example reports did not include a checkpoint hash.
The aggregate report named the requested checkpoint, but could not detect
prediction files left over from another model. New files record the checkpoint's
SHA-256. Inference checks that the file has not changed while it was being used;
aggregation rejects mismatched hashes or a mixture of hashed and unhashed rows.
Entirely legacy row sets remain readable with an explicit unverified-identity
label.

Surface-gate calibration now checks any recorded prediction hash, and its strict
mode requires one on every input. New calibration reports carry that identity
through promotion. Historical unlabelled archives remain usable for exact
reproduction, but their model identity cannot be recovered from the filenames.

### Calibration dependencies included unrelated source and broke as modules

The standalone calibration workflow depended on every package module. Editing
FEEMS or a renderer could therefore repeat neural calibration. Its rules now
name the code and environment each stage uses. The surface-gate workflow also
separates calibration dependencies from model-loading and promotion code.

Both workflows used the parent workflow's directory and shell globals when
imported as Snakemake modules. This could point at the wrong source directory
or fail while formatting a command. They now resolve their own directory and
pass shell paths through named inputs and parameters. Regression tests import
them from a relocated wrapper and check both reuse and invalidation.

### Fresh training and release evaluation were separate workflows

The optional `validated_release` target composes the existing modules: fresh
simulation and sampling views, core and gate training, independent H2Hs, SD
calibration, and provenance-checked candidate creation. It writes to scratch,
not the packaged checkpoint. Training-only `all` is unchanged, and saved FEEMS
archives remain independent of the neural checkpoint.

Its H2H reports describe the selected gate before SD recalibration. Plain
prediction means do not change, but calibrated SDs can affect MAP refinement.
Final-candidate MAP evaluation remains a separate H2H run. The workflow does
not impose an accuracy threshold or establish data independence from paths.

### Promotion accepted inconsistent calibration and training records

The release script converted calibration example lists directly to sets. A list
with repeated IDs could therefore pass as a valid split. It also checked that
the chosen calibration name existed, but not that the attached numerical scales
were the scales fitted under that name or that the choice minimized selection
NLL. The checks now require nonempty, duplicate-free, disjoint lists of integer
IDs; agreement between fitted, selected, verified, and attached calibration;
and finite selection scores for every candidate.

Training verification previously compared just best step and loss. It now checks
the checkpoint's recorded best metrics against the matching history row, and
checks the configured early-stopping metric. The historical field named
`best_validation_loss` stores that chosen metric, not necessarily total loss.
A saved best-step snapshot need not contain the full run history. Final
checkpoints also omit two bookkeeping keys from their best-metrics snapshot;
those remain valid.

The frozen-core comparison checked only tree leaves. It now compares PyTree
structure and leaf dtypes as well, so identical arrays in a different container
or Equinox structure cannot pass as an unchanged field. The existing H2H
checkpoint-hash guard is retained. Promotion hashes all input artifacts before
reading them, checks the hashes again before replacing the destination, and
refuses to overwrite an input path. Failed validation or writing leaves the
previous release intact.

These are consistency checks, not proof of training-data independence. In
particular, legacy calibration prediction NPZs still lack a checkpoint hash.
The caller must supply predictions from the stated source checkpoint. New
prediction hashes and the strict candidate workflow address this for future
runs, as described above; they cannot verify historical unlabelled NPZs.

### Checkpoint reads could silently reinterpret malformed inputs

Duplicate ZIP members were collapsed into a dictionary during loading. They are
now rejected rather than selecting one copy implicitly. Valid additional members
remain supported and are retained during metadata edits. Training, calibration,
migration, and promotion share the existing atomic output helper; promotion keeps
its separate deterministic ZIP headers and canonical JSON formatting.

Architecture dimensions no longer truncate fractional metadata or accept string
booleans as enabled gates. Density-dependent calibration also rejects invalid
sample counts and node indices before counting occupied nodes. This does not
change the density definition: it is the fraction of sampled nodes, not the
number of individuals. The checkpoint functions now have explicit type checks
and short docstrings without changing saved object fields or call signatures.

### A FEEMS parity fixture only compared failed CV scores

The old four-node, eight-SNP fixture produced all-NaN errors from FEEMS' joint
CV routine. The parity test used `equal_nan=True`, so matching failures passed.
Its separate call-recording tests still checked the penalty and warm-start
order, but the numerical test did not demonstrate agreement on usable scores.

The fixture now uses nine demes, four diploid samples per deme, and 512
MAF-conditioned SNPs from the deterministic Python coalescent reference. The
test requires every serial and fold-parallel score to be finite before checking
their numerical agreement. This is a small test fixture, not a replacement
training corpus or a new FEEMS accuracy benchmark. The earlier verification
sections' claims about this numerical test should be read with that limitation.

### FEEMS input and configuration errors could be silently changed

The direct FEEMS adapter cast node and edge IDs to int32 before checking them.
Fractional IDs could therefore select a different graph or sample assignment.
It now shares the genotype and graph validators used by DREEMS. Duplicate
node coordinates are also rejected: FEEMS assigns samples by nearest coordinate,
so ties cannot reliably preserve the caller's explicit node assignments.

`FEEMSConfig.parallel_cv_jobs` silently truncated fractional worker counts and
clamped nonpositive counts to one. Config validation now rejects malformed
counts, bounds, penalty vectors, and CV modes. Valid penalty order, zero
penalties, nonpositive LOO sentinels, fit tolerances, and saved fields remain
unchanged. All adapter functions now have explicit runtime checks and docstrings.

The named fitting helpers could also report the wrong CV mode when handed a
config built for another helper. Each routine now records the mode it actually
runs without mutating the caller's config. The generic dispatcher still follows
the supplied config. This report correction does not change the fits.

### A fold could be published under the wrong input hash

The independent LOO worker computed its input hash after fitting. Replacing
the input file during a fit could therefore label old-data scores with the new
file's hash. It now hashes before loading and checks again before publishing.
Fold files use the shared atomic writer, retaining the existing uncompressed
NPZ schema and NaNs for failed penalty combinations. Invalid fold IDs and complex
error matrices are no longer accepted as reusable scores.

### Complex input arrays lost their imaginary components

NumPy's conversion to float64 could discard imaginary values before input
validation. This affected covariance, coordinates, genotype dosages, and result
arrays. A shared conversion now accepts real numeric data and rejects complex,
string, and object arrays. NumPy, xarray, VCF-Zarr coordinates, the CLI, and
benchmark inputs use the same check. Valid array-like inputs retain their
float64 values, shapes, and order.

### Evaluation repeated IDs across input packs

`pipeline.command_evaluate` assigned `replicate_offset + index_within_pack`
without advancing the offset between packs. A call with several inputs could
therefore write duplicate IDs in both the CSV and prediction NPZ. Sorting the
graphs into similarly sized batches made position-based matching unsafe too.
The worker now advances the offset by the size of each pack and preserves the
mapping from IDs to flattened prediction-array offsets. Tests cover a limit that
stops partway through the second pack and batches whose graph order differs from
input order.

The main workflow supplies one prepared input to each evaluation job. Its
aggregate uses `replicate_global` to distinguish rows from different jobs. This
bug is in the multi-input worker path; it is not evidence that the existing
per-example FEEMS H2H was mispaired.

### Streaming could hang or silently change batch size

An empty path list, or a set of entirely empty prepared packs, let the iterator
repeat forever without producing a batch. Nonpositive batch sizes were not
checked either. These inputs now fail with an explicit error; zero requested
steps still return an empty iterator.

The iterator also assumed all paired sampling groups had the same size without
checking it. Mixing groups of two and four views could produce a batch of 12
when eight was requested. Group sizes must now agree, divide the batch size,
and contain exactly one anchor. The released four-view convention and the RNG
call order are unchanged. A frozen-code comparison reproduced both old failures
and checked 480 valid seeded batches for exact equality.

### Normalization accepted malformed dimensions and moments

Feature widths were read from the first example only. A later one-column array
could broadcast into a wider statistics vector. Counts read from JSON were cast
with `int`, and nonfinite moments could reach the saved normalization. The
workers now check dimensions, finite real values, matching sum/squared-sum
shapes, positive counts, and valid variance floors. The calculation still uses
the same float64 accumulation order, row weighting, and target-scale convention.

Repeated calculations for node features, edge features, and both targets now
use the same field loop. Prepared-pack reads share a format check; progress
messages share one small logger. The workers have explicit beartype checks and
docstrings, and interval coverage checks matching array dimensions with
jaxtyping. Evaluation and H2H now share atomic CSV serialization; evaluation
also uses the existing atomic NPZ writer. The public parameter names/defaults,
stored object fields, and checkpoint format are unchanged.

### Invalid simulation settings could silently change the mixture

The choice helpers truncated fractional counts, clipped some nonpositive values,
ignored malformed weights, and replaced all-zero weights with uniform sampling.
Several range helpers also used a default when only one bound was supplied. A
mistyped recipe could therefore run while drawing a different distribution from
the one the user specified.

Configuration validation now checks counts, probabilities, paired bounds, named
families, and sampling overrides before generation. Invalid choices and weights
raise errors instead of selecting a fallback. Absent weights still use the old
convention: weight one for a surface candidate and zero for an unlisted graph
family. Shared weight maps may contain names used by other graph families.
Reversed endpoints remain supported in the draw helpers that historically sorted
them. Valid recipe dictionaries and RNG draw order are unchanged.

The generic sampling designs and the FEEMS-lattice design have different
historical `max_samples` conventions. This pass documents those conventions;
it does not redesign the sampling distribution or change existing corpus seeds.

### Empty sampling draws failed before the worker could retry

A small sparse lattice fixture exposed an empty sampling draw. The simulator
rejected it before `simulate_raw_example_from_seed` reached its check for a pair
of observed nodes. The intended retry therefore did not run.

Sampling is now checked before genetic simulation. The worker catches only the
specific insufficient-sampling exception, uses its existing deterministic retry
seeds and attempt limit, and records the successful retry number. Genuine
simulator errors still propagate. This also avoids simulating genetics for a
single sampled deme that cannot supply a resistance contrast.

### Direct dosage inputs needed explicit bounds

The NumPy input path checked finiteness and polymorphism but did not reject
negative missing-data sentinels or dosages above two. Raw diploid arrays now
require finite dosages in [0, 2] at inference, feature construction, and the
FEEMS adapter. Fractional imputed dosages remain valid. Missing calls must first
go through the genotype-dataset QC path. The CLI also rejects an explicit zero
SNP count instead of silently replacing it with the inferred count.

The shared validator is now an explicit source dependency of generation, feature
preparation, training, and evaluation. This keeps Snakemake from reusing outputs
after the validation behavior on which they depend changes.

### Benchmark dependencies could rerun FEEMS or leave DREEMS predictions stale

The H2H rules used the same runner file for both methods. Editing neural
evaluation could therefore rerun FEEMS, while edits to the model, features, or
inference modules were not all declared inputs to DREEMS evaluation.

FEEMS fold and final-fit commands now live in `dreems.feems_benchmark`. Common
case loading and scoring fields live in `dreems.benchmark_data`; array checks
live in `dreems.validation`. The combined H2H CLI imports the same command
handlers, so existing command lines still work. These are module moves, not
alternative fitting implementations. The FEEMS grid, CV procedure, tolerances,
and score definitions are unchanged.

Each method now declares its own source dependencies and benchmark output.
Both track `pyproject.toml` and `uv.lock`, since a numerical dependency change
can affect either method. This conservative environment dependency is deliberate.

### Real-data rendering settings were not fully tracked

Real-data inference depended on every package module, including unrelated
simulation and plotting code. Rendering tracked grid size and zoom but not DPI
or the topography limits. A rendering-only change could leave stale images, and
unrelated package edits could cause unnecessary inference.

The inference rule now lists the modules it uses. Rendering records grid size,
zoom, tile and pixel limits, DPI, and the tile-cache location as parameters.
Changing those settings does not require a new prediction.

Importing these workflows as Snakemake modules also exposed a path bug:
`workflow.basedir` refers to the outer Snakefile. Source paths now use the
directory of the currently parsed Snakefile. Shell commands receive the project
directory through rule parameters instead of relying on outer-workflow globals.

### Benchmark parsing could truncate invalid indices

The benchmark loader narrowed edge endpoints and sample-node indices to int32
before validating them. A fractional or overflowing value could become a
different valid index. Both workers now use the shared integer and graph
validators before any narrowing conversion. Nonfinite generating log rates are
also rejected before scoring. Valid input arrays retain their order and dtypes.

### JSON conversion and report writers were duplicated

The xarray, CLI, and benchmark paths had separate NumPy-to-JSON conversion code.
In particular, the xarray conversion did not handle NumPy boolean attributes.
They now share `workflow_io.json_ready`. CLI CSV/JSON and benchmark NPZ output
also use the existing atomic writer, preserving an older file if serialization
fails. CSV column order and JSON formatting are unchanged.

### Result objects accepted invalid uncertainties and edge indices

`DREEMSResult` checked array shapes but accepted negative SDs, nonfinite rates
and coordinates, and edge endpoints outside the node array. Negative indices
could silently export coordinates from the end of the array; negative SDs could
reverse interval endpoints.

Result construction now validates all rate and SD arrays in one loop, requires
finite coordinates, and checks endpoint bounds. Zero SDs and disconnected edge
subsets remain valid in a result container. Inference inputs still require a
connected graph. Array-like inputs are converted before type checks, and field
order, defaults, output dtypes, and centering conventions are unchanged. MAP
outputs remain separate point estimates, not a reinterpretation of neural SDs.

The FEEMS CLI had used a `DREEMSResult` with NaN SDs merely to build its edge
table. Shared table export now handles absent FEEMS uncertainty directly,
without constructing an invalid neural result. The existing table fields,
column order, and NaN marker for unavailable SDs are preserved. CSV and JSON
regression tests cover this path; the stricter result validator does not imply
that FEEMS supplies posterior uncertainty.

### Sampling masks could leave two representations out of alignment

When an example contained both covariance and genotypes, augmentation subset
the covariance but retained the full genotype matrix. Optional sample
coordinates were also left untouched. Every stored sample representation now
uses the same row mask, with both covariance axes subset in that order. The
graph, truth, SNP set, and effective SNP count remain unchanged.

Index arrays are checked before conversion, and the covariance keeps its stored
dtype without an unnecessary float64 round trip. This does not change valid
covariance-only or genotype-only training examples. A new mixed-input test
covers the formerly inconsistent case.

### Disconnected graphs could produce invalid topology masks

The farthest-node helper only checked whether *any* sampled node was reachable.
That condition was always true for its starting node, so a disconnected graph
could produce misleading distances and masks without an error. The replacement
uses SciPy's unweighted shortest paths and checks that all graph nodes are
reachable. It preserves the seeded starting node, tie order, and random choice
of direction. No migration rate or geographic axis enters the mask construction.

Base selection also rejects nonfinite or out-of-range sampling fractions. A
NaN fraction previously passed the dense-sampling filter because its comparison
with the threshold was false.

### Unused heuristic shrinkage code remained in the package

`null_shrinkage.py` was referenced only by its own six tests after the historical
experiment workflows were retired. The module and tests are now archived outside
the checkout, with a checksum in `retired-workflows.json`. This is not the learned
surface gate in the production checkpoint, which is retained unchanged.

### Calibration could silently omit invalid observations

The fitter discarded nonfinite residuals, invalid SDs, and invalid graph sampling
fractions. If nothing remained, it returned raw SDs as though no calibration were
needed. It also excluded positive SDs below `floor`, rather than bounding their
denominator. Empty or malformed inputs could therefore produce a misleading
calibration result or an optimistic coverage report.

Calibration now requires finite residuals, positive finite SDs for fitting, and
valid sampling fractions. Small positive SDs stay in the calculation with a
floored denominator. Numerical overflow raises an error instead of writing an
infinite NLL or scale. Applying a scale still permits zero SDs, but fitting a
Gaussian calibration to them does not. Ordinary valid inputs retain the same
calculations. Both worker scripts now use one implementation of candidate fits
and graph-balanced NLL/coverage in `dreems.calibration`.

The range-based calibration workflow now rejects empty, negative, fractional,
and overlapping fit/selection/verification ranges before scheduling jobs. The
caller must still supply a corpus independent of training and early stopping;
disjoint ranges alone cannot establish that provenance.

### Cached posterior SDs were not necessarily raw

The cached-prediction calibration script called the stored `edge_sd` values raw.
H2H files instead contain SDs after the source checkpoint's calibration. Fitting
those values and replacing the source scale would apply the wrong scale when
the resulting checkpoint was used on new data.

The script now removes the source checkpoint's scale before fitting replacement
metadata. It still requires prediction files from that checkpoint; legacy NPZ
files do not contain a checkpoint hash that can verify this automatically. A
test with a source scale of two catches the previous error. This did not affect
the released model: its calibration source checkpoint was uncalibrated, and the
complete 512-example check below reproduces the published results exactly.

Packed prediction offsets are now checked before slicing, including integer IDs,
strictly increasing offsets, and complete coverage of the edge arrays. Corrupt
offsets could previously create empty or truncated cases without an error.

### Current checkpoints failed the old calibration entry point

The range-based workflow always requested a v1-to-v2 feature migration, even for
the current v2 checkpoint. It now copies an already-current checkpoint unchanged
and only invokes migration for older input. The copy and per-case NPZ writes use
the existing atomic writer. A full calibration DAG test uses the packaged model
and verifies that attaching calibration preserves its Equinox parameter bytes.

Calibration seeds and promotion example-count requirements are now tracked as
Snakemake parameters. Both calibration workflows also track the shared numerical
module as a code input. Previously these settings or dependencies could change
without invalidating saved calibration or promotion outputs.

### Corpus seed inheritance was broken

The simulation defaults included `seed: None`, so `sim.get("seed", fallback)`
never used the fallback. Corpus generation and packing then failed at
`int(None)`. The workflow also excluded the top-level seed from worker snapshots.

The config reader now resolves the seed once: simulation seed, top-level seed,
then 12001. Null means use the next fallback; zero remains a valid explicit seed.
The snapshot stores that effective seed. Existing explicit-seed snapshots retain
their serialized contents and hash, and the historical example seed formula is
unchanged. Invalid seeds fail before generation or output writing.

This does not disable Snakemake's source tracking: changing a source input can
still schedule regeneration in the original corpus DAG. To reuse a saved corpus
as input to a new run, use `raw_source_dir` with a separate output directory, as
the release training recipe does. No production generation DAG was launched
during this audit.

Sampling-scale validation also allowed NaN or infinity in several settings, and
SNP log-bin bounds were truncated to integers. These inputs now fail validation;
valid finite settings and integer bin bounds are unchanged. The historical
fallback rules for mixture weights are documented but not changed in this pass.

### Failed worker writes could truncate existing packs

Raw/prepared pickle packs and JSON sidecars were written directly to their final
paths. The shared writers now close a complete temporary file on the same
filesystem before replacing the destination. An ordinary serialization, write,
or rename exception leaves the old file intact and removes temporary output.
A replacement retains the destination's existing permission bits. A process
killed without cleanup may leave a hidden temporary directory, but
not a partial replacement. This is not a transaction across a pack and its
sidecar; Snakemake still reruns an incomplete multi-output rule.

The config loader also treated `[]`, `0`, `false`, and empty strings as empty
mappings. Only null or an empty YAML document now means an empty config. Other
non-mappings and non-string mapping keys raise an explicit error.

### Inference without variance bounds failed

`model.predict_with_diagnostics` obtained a NumPy view of a JAX array, then added
the normalization correction in place. That view can be read-only. Clipping
normally allocated a writable array and hid the problem; checkpoints without
bounds took the failing path.

Fixed by taking writable copies before clipping and scaling. In-place scaling
preserves the output dtype. The regression test covers bounded and unbounded
predictions. This does not change bounded predictions from the released
checkpoint.

### A loss was mistaken for a correlation metric

The training log's `correlation` and `validation_correlation` entries contain
`masked_centered_correlation_loss`, not Pearson r. Smaller is better. The loss
uses a stabilized denominator and gives inactive graphs zero loss, so even
`1 - validation_correlation` is not an ordinary mean correlation.

The names remain unchanged for log compatibility. Their meaning is now stated in
the loss docstring. Earlier interpretations treating this field as Pearson r
were incorrect. Test-set Pearson correlations are a separate calculation.

### Runtime checks were hidden, not absent

`__init__.py` installed an import hook for `model` and `losses`. The initial audit
missed this and incorrectly described these annotations as unenforced. Explicit
decorators replace that hook; direct calls and JIT tracing still check shapes.
No model parameters or fields are changed by this move.

### Checkpoint saves could overwrite a valid archive on failure

`train.save_checkpoint` wrote directly to the destination archive, although
`checkpoints.py` already had an atomic archive writer. Training now uses that
writer too. Tests simulate a failed write both with and without an existing
checkpoint: the old archive remains readable, and no partial archive remains.

Loading, calibration updates, and metadata migrations also share the same
archive reader. Incomplete archives now raise a specific `ValueError` rather
than an incidental missing-ZIP-member exception.

### Nonfinite training could continue and appear successful

Nonfinite validation metrics skipped the early-stopping counter. Runs with
repeated NaNs could consume their full step budget. Training also returned the
result of an invalid final update, and an empty batch stream returned an untrained
model as if the call had succeeded.

The failure policy is now explicit: nonfinite training losses, gradients,
optimizer state, weights, or validation metrics raise `FloatingPointError` with
the step number. A failure does not call the checkpoint callback or return a
successful result. Previously saved best checkpoints remain intact. An empty
stream raises `ValueError`. Ordinary finite plateaus still use the configured
early-stopping patience; there is no arbitrary upper cutoff for a finite loss.

The gradient/state check uses JAX tree operations inside the compiled update.
It catches a finite loss whose derivative is nonfinite, not just NaNs in the
reported scalar. Numerical config settings now reject NaN and infinity before
expensive preparation or training begins. No configuration fields, defaults,
loss equations, or optimizer settings changed.

### Local training assumed a Slurm job ID

All three main training rules expanded `SLURM_JOB_ID` without a fallback.
Snakemake runs Bash with `set -u`, so training on a local executor without that
variable failed before entering Python. The three rules now share one command
and use a private `mktemp` directory with cleanup on exit. The gate stage
inherits that command using `use rule`. Rule names, output paths, configuration
snapshots, and training settings are preserved.

The local launcher also ignored the `DREEMS_CONFIG_FILES` and `DREEMS_SNAKEFILE`
options implemented by the Great Lakes launcher. Both launchers now accept
them; tests check that file order and arguments containing spaces are preserved.

### Publishing parsed filenames as colon-separated records

The publish shell split `source:destination` strings at the first colon. Valid
paths containing colons were therefore copied incorrectly. It now uses paired
Bash arrays. The core publish rule inherits the final-model publish command.
Copy-then-rename behavior is unchanged.

### Release training needs sampling views, not an older fine-tune

`config.full_retrain.yaml` depends on packs produced by
`sampling_robustness.Snakefile`. The latter's default target also ran two older
fine-tuning stages. A named `prepared` target now builds just its derived views
and replay packs. These are the needed inputs for training a new core; they do
not require a pretrained checkpoint or new SNP simulations.

The new small-corpus test also exposed a hard-coded 64-example assumption in
replay preparation and validation counts. Replay preparation now reads the
whole pack, and validation counts come from raw metadata. The release's complete
64-example packs are unchanged. Stage-config contents are now rule parameters
so Snakemake can detect a changed configuration instead of retaining stale JSON.

The release presets still contain the original run's absolute artifact paths.
The workflow README now distinguishes these from portable defaults and lists
which paths must be set on another machine. A fully self-contained release DAG
remains work to do; the local-run fixes alone do not establish that.

### Retired experiment workflows

Seventeen files, totaling 3,451 lines, were archived outside the repository and
removed after checking their references. They implement the old null-surface,
forward-evidence, and pre-gate promotion experiments. The current core/gate
training, calibration, release, benchmark, and real-data paths do not use them.

The archive was checked against the original files before removal. Its path,
SHA-256, and complete file list are in [retired-workflows.json](retired-workflows.json).
No public model objects, weights, raw corpus, or saved FEEMS results were removed.

### Input conversion could change graph indices and precision

Edge and observed-node indices were cast to int32 before validation. Fractional
values could become valid but different indices, and large integer values could
overflow. SNP counts were also truncated. Sample counts used an approximate
integer test that accepted small fractional parts.

Inputs now share an exact, range-checked integer conversion. Integral floats
remain supported for CSV input, as do NumPy integer scalars. Xarray validation
uses the same `NodeCovarianceInput` constructor rather than recasting and
rechecking the arrays in several places. NumPy input objects and the xarray API
now share numeric constraints: node covariance must be positive semidefinite within
roundoff tolerance, and graphs must be connected without duplicate edges.

### Repeated inference could retain an earlier fit

`result_to_dataset` copied all variables from an earlier result. If a later call
disabled refinement or did not supply calibrated SDs or node-q estimates, the
old optional outputs remained even though the attributes described the new fit.

The exporter now replaces its own output variables, removes obsolete refinement
metadata, and preserves unrelated input variables and attributes. It also checks
graph and node order before attaching results to a dataset.

### Edge-record export did quadratic work

`DREEMSResult.to_edge_records` centered the full surface inside the per-edge
loop and repeatedly compared the full raw and calibrated SD arrays. These
whole-array operations now run once per export. A test counts centering calls;
the records retain their original schema and values.

### Smaller API and workflow fixes

- CLI `.npz` loading now closes its archive, and a one-row CSV edge list stays a
  two-dimensional matrix. FEEMS CLI input uses the same graph validation before
  casting indices.
- `from_vcf_zarr` and `from_genotype_dataset` are two names for one conversion
  function. Their public signatures are unchanged. The CLI covariance converter
  also reuses the Python implementation without another forwarding function.
- Refinement updates a frozen result with `dataclasses.replace` rather than
  repeating every unchanged field in another constructor call.
- The quality workflow no longer treats generated `__pycache__` files as source
  dependencies.

### Failed sparse solves could look like valid initializer features

SciPy can return nonfinite values from a sparse solve without raising. The
initializer did not check the returned potentials or conductances. Downstream
centering could replace nonfinite entries with zeros, while the conditioner
still set its validity flag to one.

Both solves now reject nonfinite results. Disconnected graphs are rejected
before solving. Inference raises a descriptive error; callers that explicitly
request the existing `fail_on_error=False` fallback receive zero feature values
and a zero validity flag. Tests force failure in each solve and check both modes.
This fixes a failure path, not an observed explanation for benchmark errors.

### The covariance eigensolver fallback changed negative modes into variance

After repeated eigensolver failures, the old fallback used singular values as
eigenvalues. For an indefinite covariance, that turns negative modes positive
instead of discarding them. It now tries a different symmetric eigensolver that
preserves eigenvalue signs. A forced-failure test uses a matrix with a negative
mode; normal NumPy eigendecomposition is unchanged.

### Padding could silently broadcast malformed targets and features

NumPy assignment accepts a single row or scalar where a full per-edge or per-node
array was intended. Padding could therefore repeat malformed data across a graph.
The padding function now checks row counts, feature widths, index types and
bounds, and requested padding dimensions before copying each example.

### Diffusion and initializer feature meanings

The diffusion implementation now shares its reference-conductance calculation
with the initializer, and computes edge contrasts from the concatenated bands.
The band columns and their order are unchanged. With no graph edges, the old
special case incorrectly discarded the entire signal. The zero-Laplacian case
now retains the signal in the final low-pass band, so the bands still sum to the
standardized input. Production graphs are connected and have edges.

The initializer remains a smoothed diagonal approximation to the normal
equations. It uses every observed pair: its batch size bounds temporary memory,
not pair count. A dense reference test checks 300 pairs across four batch sizes
and weighted/unweighted cases. No new initializer or model feature is introduced.

The feature names containing `precision` need careful interpretation. When the
input is covariance, their fourth-moment calculation operates on eigenfactors,
not observed SNP columns. They are deterministic support heuristics, not measured
genotype fourth moments or calibrated uncertainties. The docstring now says so;
the formulas remain unchanged for checkpoint compatibility. Repeated covariance
eigenvalues also leave the score basis nonunique despite its sign convention.

### Genotype masks and unsupported observations

The genotype converter previously looked only at negative allele values. It
ignored an explicit missing-call mask, accepted fractional allele codes, and
treated mixed-ploidy fill values as ordinary missing calls. It also counted a
sample with no calls in the retained variants as observed after mean imputation.

The adapter now requires signed integer allele codes, honors the named call
mask, rejects mixed-ploidy padding, and rejects samples with no usable calls.
The mask convention and the distinction between -1 missing and -2 fill values
follow the [sgkit genotype specification](https://sgkit-dev.github.io/sgkit/latest/generated/sgkit.variables.call_genotype_spec.html)
and [mask specification](https://sgkit-dev.github.io/sgkit/latest/generated/sgkit.variables.call_genotype_mask_spec.html).
For a partially missing call, the whole sample dosage remains missing, as before.

Sample latitude and longitude are now validated when already attached to an
xarray dataset, not just when supplied as a sample-ID mapping. This validation
and node-coordinate validation happen before the genotype reductions.

The MAF and call-rate defaults, mean imputation, sample-variance scaling, node
assignment, and covariance calculation are unchanged for supported input.
Xarray reductions remain chunkable; only QC summaries and the final node
covariance are materialized. The README now states which study-specific QC and
LD filtering users must perform separately. Uniform polyploid inputs remain
supported; this is not a newly trained polyploid model.

### Refinement could hide a numerical failure

The initializer and refinement used separate grounded-Laplacian solves. The
refinement copy did not check whether SciPy returned finite potentials. Its
centering helper also mean-filled nonfinite gradients and initial predictions,
which could hide a failed calculation.

Both paths now use the same checked solve. Refinement validates initial fields,
uncertainties, observation counts, graph connectivity, and config values before
optimization. It rejects nonfinite objectives, gradients, or updates instead of
returning a surface. Shape and integer-index checks prevent accidental NumPy
broadcasting or truncation. The line-graph Laplacian and centering helper are
shared as well. Supported calls retain their parameters, defaults, and result
field order.

### Per-edge clipping could violate the trust region after recentering

The optional maximum per-edge update clipped changes in SD units, then centered
the resulting log field. With unequal SDs, that centering can move an edge
outside its bound. A symmetric test case had hidden this error.

The clipped path now recenters before contracting to satisfy both bounds. This
is a feasible radial contraction, not a weighted nearest-point projection. The
default refinement loop sets the per-edge maximum to zero and uses only the RMS
radius; its calculation is unchanged. There is no evidence from this finding
that the released default results violated their trust radius.

### What the refinement objective actually means

The API retains the name MAP refinement for compatibility, but its implemented
criterion is off-diagonal covariance least squares plus a smoothness penalty,
subject to a hard SD-scaled trust region around the neural prediction. It does
not use the FEEMS Gaussian likelihood or a soft Gaussian-prior penalty. Two
nonnegative nuisance scales are fitted at each step when a q field is supplied;
the q field itself stays fixed. Refinement does not update posterior SDs.

The docs now distinguish these points and explain the analytic Green-function
gradient. This optional CPU refinement remains separate from the JAX model;
the public NumPy call cannot be differentiated through. Losses in a history row
describe the point before its update, whereas trust diagnostics describe the
point after the update. The final fit is scored separately. Covariance `r2`
uses zero as its baseline and is not squared Pearson correlation.

### The Python covariance fallback did not honor its SNP count or ploidy

The Python path generated the requested number of segregating markers, then
dropped columns with constant individual dosage. It could therefore return fewer
usable columns than requested, unlike Rust, which keeps simulating until the
usable count is reached. Its polymorphism filter also hard-coded two copies per
individual, rejecting valid high-frequency polyploid markers.

Python covariance generation now streams standardized dosage columns and counts
only those with positive variance. It raises if the attempt budget is exhausted.
The variance test works at any supported fixed ploidy. Genotype output still
returns segregating markers, which need not have variable individual dosage.
Public return fields and dtypes are unchanged. The Python covariance RNG stream
changes as part of this correction; its old seeded covariance is not preserved.
The production corpus requires Rust, whose seeded outputs remain unchanged.

### Simulator validation happened after narrowing indices

The public simulation calls cast counts and indices before validating them.
Fractional values could be truncated; large values could overflow int32. A zero
attempt limit was treated as an absent limit. Strided NumPy arrays could pass
Python validation but fail the Rust extension's contiguous-slice requirement.

The shared dispatcher now validates integer types and bounds before conversion,
accepts NumPy integer scalars, and provides contiguous arrays to Rust. It rejects
zero/negative attempt limits and seeds outside uint64. Ploidy is limited to
1-255 because the existing genotype output is uint8; the dtype was not widened.
Covariance output requires at least two individuals. Invalid input fails before
starting either backend.

### An attempt limit could not stop a noncoalescing tree

The full-genealogy modes complete each tree before accepting a marker. A graph
can allow migration forever without a common ancestor, for example if all local
coalescence rates are zero. The outer attempt counter does not limit the events
within such a tree.

Before either backend starts a full-genealogy mode, a communicating-class check
now requires all sampled lineages to reach one closed migration class with
positive coalescence somewhere in that class. Directed migration toward a single
coalescing sink is still allowed; disconnected unsampled components do not by
themselves invalidate a graph. First-mutation mode need not complete the tree
and does not impose this condition. This validation does not change connected,
positive-q production simulations or their random draws.

### Marker modes were described as if they had the same sampling law

The docstring described stopping on a mutation, although the production setting
uses a completed tree with a uniformly placed branch marker. The three supported
modes remain unchanged:

- `first_mutation` stops at the first mutation backward in time and rejects an
  attempt if the MRCA is reached first.
- `random_marker` samples a branch-length-weighted marker on a completed tree
  and accepts that tree with probability `1-exp(-mutation_rate*tree_length)`.
- `conditioned_random_marker` samples a branch-length-weighted marker on every
  completed tree, without that no-mutation rejection. Mutation rate does not
  change this mode's output distribution.

These are distinct ascertainment laws. The small-graph msprime adapter selects
one site from each mutation-bearing replicate; it is not the always-marked-tree
production mode. Frequency bounds are on derived-allele counts; symmetric bounds
such as 0.05-0.95 implement a MAF filter. Saved mode names and aliases are retained.

The Python implementations now share the migration/coalescence event loop, and
both public simulation APIs share validation and backend dispatch. Unused lineage
location bookkeeping and repeated within-deme sample-index construction were
removed. The msprime haplotype adapter now uses stable sorting to restore caller
sample order in one array operation rather than maintaining per-deme cursors.

### Remaining workflow dependencies

The fresh training entry point now composes corpus generation, sampling views,
and both fitting stages, with portable output paths and execution tests. The
historical configs remain records of the released model's settings, not the
default launch recipe. Calibration has also been exercised on the current model.

The H2H fit, real-data renderer, scenario-generation, and calibration dependencies
have since been tested, including imported-rule execution. The release workflow
now creates a candidate's own independent benchmark and calibration records;
historical configs retain the original release paths only as a record.
The SNP-range comparison workflow still needs review. Archive restoration and
provenance are addressed below. Historical fits without recorded input hashes remain a
documented limitation; a later checksum cannot establish their original inputs.

### FEEMS archive restoration and provenance

The old restore command extracted arbitrary tar members and replaced an entire
results directory. It also assumed that the filename in the checksum sidecar
still named the supplied archive. That fails after relocation or renaming.
`benchmark_archive.py` now checks the supplied file's digest and accepts only
regular per-case JSON/NPZ members. It rejects links, traversal, duplicates, and
unexpected files. All selected cases are validated before any existing results
are replaced; unrelated files are left alone. One deterministic archive writer
replaces the two former shell packaging paths.

Matching graph dimensions did not establish that FEEMS and DREEMS used the same
input data. New case reports record the raw input file's SHA-256 and reject a
file that changes during fitting. Aggregation rejects unequal recorded input
hashes, malformed hashes, and case IDs that previously could be silently cast
to integers. Archive restoration checks the hash plus geometry, edge order,
truth, sample counts, SNP count, and declared LOO protocol. None of this changes
the fits, metrics, or model serialization.

The retained 512-case FEEMS archive has no original input hashes. It is reused
only through an explicit `allow_legacy_feems_archive` setting and marked
`legacy_unverified` in its verification report. Structural agreement supports
reuse but cannot establish genotype identity. No retrospective hash is written
into an old fit. The aggregate records this limitation instead of calling the
archive's inputs verified.

The integration tests also exposed literal `{example}` templates being expanded
by Snakemake before the archive and aggregate workers received them. Those
parameters now use functions, as the existing gallery template parameters do.

The direct NumPy dosage and explicit-zero SNP-count issues have been fixed and
tested separately from sgkit allele-code QC, as described above.

## Verification so far

On 2026-09-08, the model/loss comparison against a frozen source snapshot passed
on a Great Lakes CPU worker submitted through Snakemake with account `jonth0`.
All numerical comparisons had maximum absolute difference zero:

- All four combinations of initializer and surface gates on small padded graphs.
- Model gradients through JIT, gate upgrades, and initialized parameter leaves.
- Eight loss/gradient comparisons, including squared and Huber slope penalties.
- Predictions from the packaged production checkpoint.
- Scaled NumPy predictions from that checkpoint, including unchanged dtypes.
- Equinox serialized parameter bytes for all four small model variants.
- Three-step training runs in all three scopes: full model, initializer gate,
  and surface gate. Final parameters, logged losses, validation histories, and
  checkpoint-selection summaries matched the previous implementation.

The old unbounded-variance inference failure was reproduced separately, and the
new path returned finite predictions. This is evidence of compatibility, not
evidence that model accuracy has improved.

The final quality run passed 185 tests in 102 seconds, with 20 warnings from
statistical/FEEMS tests. Ruff check and formatting, the production workflow dry
run, the executed quickstart, and installing the wheel in a separate environment
all passed. No GPU, corpus regeneration, or benchmark rerun was needed.

Numerical comparison results are in [release-audit-parity.json](release-audit-parity.json).
Reference source, the comparison script, and logs are retained outside the repo at
`/home/jonth/turbo/dreems/release_audit_20260908/`. Temporary worker artifacts are at
`/scratch/jonth_root/jonth0/jonth/dreems/release_audit_20260908/`.

The production checkpoint remains byte-identical to commit `85df370`. Its SHA-256
is `454cb50f4c7ef1213e001d700d1bf9e9f6a353b5c8be5619aae06e9d7a22cc15`.

The quality workflow accepts `--config output_dir=/path/to/scratch` so its
notebook and test artifacts need not be written into the checkout. The default
`artifacts/quality` directory is unchanged for local use.

## API audit verification

The API pass on 2026-09-08 passed 210 tests in 104 seconds, including 25 new
validation/export regressions. Ruff, workflow dry run, the executed quickstart,
and the installed-wheel check also passed through Snakemake on `jonth0`.

The released checkpoint was run with both the saved source and the edited API
on a deterministic six-node example. Genotypes, sample covariance, and node
covariance gave exactly matching means, SDs, q predictions, dtypes, and edge
records. A three-step MAP refinement also matched exactly, including diagnostics
after excluding elapsed-time fields. That exclusion was needed because the
first comparison incorrectly required runtime measurements to be equal.

Public call parameters/defaults and dataclass field order were checked against
the saved source. Created xarray inputs were identical. The reference also
reproduced truncation of an edge endpoint from 1.25 to 1 and an SNP count from
50.5 to 50, and retention of an old refined surface after requesting a neural-only
fit. The edited code rejects the fractional inputs and removes the stale fit.

Results are in [release-api-audit-parity.json](release-api-audit-parity.json).
The source snapshot, script, and logs are retained outside the repo at
`/home/jonth/turbo/dreems/release_api_audit_20260908/`. Temporary worker files are
under `/scratch/jonth_root/jonth0/jonth/dreems/release_api_audit_20260908/`.
The production weights were not changed or retrained.

## Training and genotype audit verification

The final run passed 357 tests in 350 seconds, including 106 new parameterized
checks and regression cases. Ruff, the workflow dry run, the executed quickstart,
and the separate installed-wheel check also passed through Snakemake on Great
Lakes CPU workers under `jonth0`. The chunked-genotype test ran successfully; it
was not skipped for a missing Dask installation.

The source comparison covered three-step training in all three scopes: full
model, initializer gate, and surface gate. Final parameter leaves, losses,
validation history, checkpoint-selection summaries, and metadata were identical
to the saved source. Public parameter names/defaults and dataclass field order
were also unchanged.

Six genotype datasets covered ploidies 1, 2, and 4 with ordinary and
dateline-crossing coordinates. Supported inputs produced identical xarray
datasets. Training-array preparation matched with and without validation data.
All 1,530 comparisons were exact; the summary is in
[release-training-qc-audit-parity.json](release-training-qc-audit-parity.json).

The new independent NumPy covariance test initially used float64 throughout,
whereas the existing xarray path standardizes masked int8 calls in float32 and
then averages by node in float64. The discrepancy was about 2.8e-8. The reference
now follows those same dtypes; the production arithmetic was not changed to make
the test pass. Source parity checks the actual pre-refactor behavior separately.

This audit does not establish whether the fixed missing-data paths occurred in
any published real-data reanalysis. It does not change the packaged checkpoint,
nor does it show an improvement in model accuracy.

Reference source, scripts, logs, and test results are retained at
`/home/jonth/turbo/dreems/release_training_qc_audit_20260908/`.
Temporary worker outputs are under
`/scratch/jonth_root/jonth0/jonth/dreems/release_training_qc_audit_20260908/`.
Disposable pytest fixture directories are removed after verification; scripts,
source snapshots, result XML, and logs are preserved. No production corpus or
FEEMS results were regenerated, and the packaged weights are unchanged.

## Refinement audit verification

The refinement pass passed all 389 tests in 305 seconds. The 32 new cases cover
invalid config values and observations, fractional indices, failed solves,
nonfinite gradients, the clipped trust region, and independent dense-solve and
finite-difference references. The existing q-aware and pair-subset gradient tests
also passed. The suite retained the same 20 statistical/FEEMS warnings. Ruff,
workflow dry run, quickstart execution, and the installed-wheel check passed.

The source comparison ran 24 complete 50-step refinements on 9-, 25-, and 64-node
graphs, with sparse/dense observations, optional q covariance, and all pairs or
pair subsets. Initializer results were compared too. All 3,573 numerical checks
were exactly equal, excluding elapsed-time fields. Parameters/defaults and result
field order were preserved. The comparison script initially failed when computing
a difference between boolean diagnostics; fixing that script did not require a
production-code change or a repeat of the other completed quality checks.

The clipping regression was reproduced against the old source: a requested
0.75-SD per-edge limit allowed a 2.21-SD change and also exceeded the 0.5 RMS
radius. The fixed path respects both bounds. The default refinement path does
not enable per-edge clipping and retained exact parity on the tested cases.

Results are in [release-refinement-audit-parity.json](release-refinement-audit-parity.json).
Reference source, scripts, logs, and test results are retained at
`/home/jonth/turbo/dreems/release_refinement_audit_20260908/`.
Checks ran through Snakemake on Great Lakes CPU workers under `jonth0`.
No corpus, checkpoint, or FEEMS benchmark was regenerated. This is evidence of
compatibility and corrected failure handling, not improved predictive accuracy.

## Simulator audit verification

All 427 tests passed in 330 seconds, including 38 new simulator contract tests.
Ruff, the workflow dry run, the executed quickstart, and the installed-wheel
check passed through Snakemake on Great Lakes CPU workers under `jonth0`.

The source comparison covered 33 seeded outputs, six complete raw examples,
and the msprime sample-order helper. All 363 comparisons were exact. This
includes all three marker models, ploidies 1, 2, and 4, Rust covariance output,
and Rust output with more than 128 haploid copies. Public parameter names,
defaults, and result field order were unchanged. The Rust binary and packaged
checkpoint were not modified.

Python covariance output is deliberately excluded from exact comparison: its
usable-SNP count and ploidy handling were corrected. Separate tests check that
it rejects constant dosage columns, returns the requested number of usable
SNPs, and respects the attempt budget. Its RNG stream consequently changes.

A new independent check compares 5,000 conditioned markers against msprime
genealogies, with markers selected in proportion to branch length. Allele-count
means and the frequency spectrum passed the specified statistical tolerances.
Existing first-mutation, closed-form, and sample-order tests also passed. These
checks support the implemented simulation laws; they do not establish that a
law captures every feature of real genetic data.

Verification caught two typing mistakes in the refactor before this final run:
NumPy generic annotations were not accepted by beartype, and signed-only index
annotations incorrectly rejected valid unsigned indices. Both were corrected
and covered by regression tests. No simulation algorithm changed for those fixes.

The comparison summary is in
[release-simulation-audit-parity.json](release-simulation-audit-parity.json).
Source snapshots, scripts, logs, and test results are retained at
`/home/jonth/turbo/dreems/release_simulation_audit_20260908/`.
No production corpus, FEEMS result, or checkpoint was regenerated.

## Corpus and I/O audit verification

All 477 tests passed in 325 seconds. The 50 new cases cover seed precedence,
invalid configuration values, interrupted writes, failed renames, file formats,
and preserved permission bits. Those 50 tests also passed as a separate focused
job. Ruff, the workflow dry run, executed quickstart, and installed-wheel check
passed through Snakemake on Great Lakes CPU workers under `jonth0`.

Four recipes, including the production and SNP-range configs, retained identical
settings, 1,000 sequences of random draws, and final RNG states. Six complete
Rust examples were then generated and packed through the old and new corpus
workers. The six individual pickle files, two packs, and two metadata sidecars
were byte-identical. Public parameter names and defaults were preserved.

The end-to-end workflow test now exercises seed inheritance through the worker
snapshot. It verifies that an equivalent explicit seed schedules no new work,
along with the existing training-change reuse, core/gate, and publishing checks.

The first test run reached 470 passes but failed the unchanged-rerun assertion
because this audit added the permission safeguard while the workflow test was
running. Its log explicitly reports `workflow_io.py` as an updated input. The
final run used unchanged source throughout and passed without weakening that
assertion. This failure was caused by the audit procedure, not seed inheritance.

Results are in [release-corpus-audit-parity.json](release-corpus-audit-parity.json).
Source snapshots, scripts, logs, and test results are retained at
`/home/jonth/turbo/dreems/release_corpus_audit_20260908/`.
Temporary test outputs remain on scratch. The packaged checkpoint is unchanged,
and no production corpus or FEEMS benchmark was regenerated.

At the end of this pass, sampling-view selection still read source metadata while
parsing the Snakefile, and release configs contained historical absolute paths.
The release-DAG pass below addresses those remaining training dependencies.

## Fresh release training DAG

The previous sampling workflow could only start after its source corpus existed:
selection read every metadata sidecar while Snakemake was parsing the workflow.
It also retained fine-tuning stages that were not part of the released model.

`workflow/release.Snakefile` now combines the existing core/gate workflow with
sampling preparation. It loads the recorded scientific settings, replaces old
storage paths, and wires the prepared packs into both stages. Selection is a
Snakemake checkpoint, so its inputs can be generated earlier in the same run.
Both machine launchers default to this entry point. An existing corpus can still
be supplied through `raw_source_dir` without making its packs regeneration targets.

The sampling workflow now only selects examples and prepares paired masks and
ordinary replay packs. Feature preparation depends on feature settings, not an
optimizer or checkpoint. Source ordering, selection seeds, mask seeds, and the
paired-view definitions have not changed. The historical version, including
the removed fine-tuning stages, is retained outside the checkout in
`/home/jonth/turbo/dreems/release_dag_audit_20260908/retired_sampling_stages.tar`.
Its SHA-256 is
`5e343484b0a9f0a3ee09b6d6e537774623a2e7ec8a1344d2f38c68e479b8fb78`.

The launcher audit also found that passing a CLI `--config` override discarded
`DREEMS_RUN_ROOT`: argparse retained only the last of two config blocks. The
launchers now insert the environment default into an existing `--config` or
`-C` block. A later explicit CLI `outdir` still takes precedence.

The training diagnostic `eval/summary.json` includes early-stopping validation
examples. It is not an independent benchmark; the documentation now says so.
Independent H2H evaluation, SD calibration, and package promotion remain separate
workflows and still need a final dependency review. This entry point does not
silently replace the packaged model with a newly trained candidate.

### Verification

The focused integration test passed in 161 seconds on a Great Lakes CPU worker
under `jonth0`. Starting from an empty directory, a single invocation generated
six tiny Rust examples, selected and prepared paired sampling views, trained a
two-step core and two-step gate, evaluated the result, and published both stages.
It then verified that an unchanged rerun scheduled nothing, an optimizer change
reused raw and prepared data, and a mask-seed change rebuilt only the paired
views and their downstream work. The core started without a checkpoint.

The frozen-reference comparison retained both selection manifests and replay
selections exactly. All 88 arrays across eight paired views were identical when
prepared with the old full stage config and the new feature-only config. This
check used small lattice examples; it is not a large-corpus numerical benchmark.
Results are in [release-dag-audit-parity.json](release-dag-audit-parity.json).

A separate dry run planned the default production-sized DAG: 65,536 raw
simulation jobs and 78,315 total jobs before selection checkpoints expand their
inputs. Planning took 349 seconds and peaked at 2,427 MB RSS on the worker. No
production simulations, training, or FEEMS fits were executed by that dry run.

The first complete test run had 482 passes and one failed assertion about where
Snakemake puts its controller log. The database and job state were correctly
under the run directory, but the controller opens its log before reading the
workflow's `workdir` directive. The test now checks the intended invariant:
only the controller-log directory can remain outside the run directory. The
documentation explains how to move that log too with `--directory`.

The final full suite passed all 483 tests in 478 seconds. Ruff, the workflow smoke
test, executed quickstart, and installed-wheel check passed too. The 20 warnings
came from nearly constant test data in SciPy and numerical/deprecation warnings
in the official FEEMS package. Test execution used CPU workers under `jonth0`;
this pass did not exercise a full GPU training allocation.

Scripts, frozen references, logs, test results, and source snapshots are retained
at `/home/jonth/turbo/dreems/release_dag_audit_20260908/`. The archive excludes
temporary pytest directories and the production dry-run state. Those remain on
scratch. The packaged checkpoint retains SHA-256
`454cb50f4c7ef1213e001d700d1bf9e9f6a353b5c8be5619aae06e9d7a22cc15`.

## Calibration audit verification

All 528 tests passed in 616 seconds; the 48 focused calibration tests also passed
separately in 91 seconds. Ruff, the workflow smoke test, the executed quickstart,
and the installed-wheel check passed through Snakemake on `jonth0` CPU workers.
The full suite retained its 20 SciPy/FEEMS test warnings, with no new warnings.

The frozen-source check compared 300 fitted calibrations, 300 graph-balanced
metric summaries, and 2,655 scaled SD arrays. All comparisons were exact. A
36-example cached-prediction run produced four byte-identical reports and left
Equinox parameter bytes unchanged. Results are in
[release-calibration-audit-parity.json](release-calibration-audit-parity.json).

The audit then reused the actual 512 saved release predictions. The old and new
workers produced byte-identical fit, selection, verification, and CSV reports.
All three JSON reports also matched the published reports, and the selected
density scales matched the packaged checkpoint exactly. The source checkpoint
had raw SDs, so it was unaffected by the recalibration bug. This check took 17
seconds and performed no inference or FEEMS fits. Its results are in
[release-calibration-audit-published.json](release-calibration-audit-published.json).

The integration test runs the range-based calibration DAG from the current
packaged checkpoint and six small test examples. It checks disjoint case IDs,
unchanged canonical checkpoint bytes, unchanged Equinox parameter bytes after
calibration, verification reports, and an unchanged rerun that schedules no work.
Separate tests cover malformed ranges and offsets, invalid numerical inputs,
small-SD flooring, scalar NumPy inputs, and recalibration of a scaled source.

References, scripts, logs, source snapshots, and reports are retained at
`/home/jonth/turbo/dreems/release_calibration_audit_20260908/`. Temporary test
directories and checkpoint copies are excluded from the verification archive.
No production training, corpus generation, or FEEMS benchmark was rerun. The
released weights remain unchanged; the broader release audit is still open.

## Result and sampling audit verification

The final run passed 580 tests in 697 seconds, with 20 existing SciPy/FEEMS
warnings. All 86 focused result, CLI, and sampling tests passed. Ruff, the
workflow dry run, the executed quickstart, and the installed-wheel check passed
on CPU workers submitted through Snakemake under `jonth0`.

The frozen-source comparison reused all 512 saved release predictions. It
checked 6,144 result arrays and every exported edge record exactly. The same
comparison also covered 100 topology masks with shuffled edge/observation order,
1,600 arrays from sampling subsets, 12 balanced base selections, and refinement
observations from all input representations, including the omitted-SNP-count
fallback. Two FEEMS CLI tables, CSV and JSON, were byte-identical using a fixed
fit fixture. The table test does not rerun FEEMS; official FEEMS numerical parity
is covered separately by the full test suite.

Input/result field order and public parameter names, order, and defaults are
unchanged. No checkpoint conversion was needed. Results are recorded in
[release-result-audit-parity.json](release-result-audit-parity.json).

The first test controller was stopped when the FEEMS CLI's NaN-SD convention
was found during caller review. After repairing that path, source was frozen
again and the complete suite was restarted. The results above describe the
final source, not the interrupted run.

Reference code, source snapshots, test reports, and verification scripts are at
`/home/jonth/turbo/dreems/release_result_audit_20260908/`. Temporary pytest
directories and checkpoint copies are excluded from the durable archives.
No production training, corpus generation, or FEEMS benchmark was rerun. The
learned production gate, packaged checkpoint, and saved benchmark predictions
remain unchanged. The broader release audit is still open.

## Benchmark dependencies and public I/O verification

This pass passed 595 tests in 671.11 seconds, with 20 warnings. The warning
report includes warnings from the official FEEMS parity tests; those tests
passed. Ruff, the workflow dry run, the executed quickstart, and installation
from a built wheel also passed. All checks ran through Snakemake on Great Lakes
CPU workers under `jonth0`.

The focused suite passed 79 tests, including two workflow integration tests.
They import the actual rules into an isolated project and use Snakemake's touch
executor with placeholder outputs. They do not fit either method. The tests
check that:

- An unchanged rerun schedules no work.
- Changing a checkpoint or neural inference code schedules DREEMS only.
- Changing FEEMS worker or adapter code schedules its folds and final fit only.
- Unrelated plotting or simulation edits leave saved fits alone.
- Changing real-data grid resolution, zoom, topography limits, or DPI schedules
  rendering without changing prediction-file timestamps.
- The same workflows can be imported from a different working directory.

The frozen-source comparison rebuilt all three aggregate reports from 512 saved
H2H cases; the CSV, summary JSON, and protocol JSON were byte-identical. It also
checked 15 input arrays, three xarray results, three edge-record exports, six CLI
tables, and six FEEMS worker output files. The worker-file comparison used a
fixed fit fixture; the complete pytest suite tests numerical parity with the
official FEEMS implementation separately.

Input/result field order and public parameter names, order, and defaults remain
unchanged. The packaged checkpoint still has SHA-256
`454cb50f4c7ef1213e001d700d1bf9e9f6a353b5c8be5619aae06e9d7a22cc15`.
The comparison summary is in
[release-dependencies-audit-parity.json](release-dependencies-audit-parity.json).

The first harness used an unsupported Snakemake shell override. Replacing it
with the touch executor exposed the module-path and shell-variable problems
described above. Those were repaired before the final focused and full runs.
An overlapping quality launch was rejected by Snakemake's lock; it was restarted
after the focused controller exited normally. No locks were bypassed.

Source snapshots, reference code, comparison scripts, and test reports are at
`/home/jonth/turbo/dreems/release_dependencies_audit_20260908/`. Temporary pytest
directories, duplicate checkpoint files, and Snakemake database state are not
in the durable archives. No production corpus, training run, or FEEMS benchmark
was regenerated. The broader release audit remains open.

## Feature audit verification

The feature pass passed 251 tests in 308 seconds, including 38 new numerical
and input-validation cases. Ruff, the workflow dry run, the executed quickstart,
and the installed-wheel check also passed on CPU workers through Snakemake.

The frozen-source comparison covered 18 examples: chain, grid, and star graphs;
sparse and dense sampling; and genotype, sample-covariance, and node-covariance
inputs. All 1,443 compared arrays were bit-identical. This includes all three
supported spatial encodings, normalization and padding, all-pair initializer
outputs, and predictions from the packaged checkpoint. Elapsed times were
excluded. The test verifies compatibility, not biological accuracy.

On one CPU worker, five warm timing repetitions gave these medians:

| Graph nodes | Observed nodes | Operation | Before (s) | After (s) |
| --- | --- | --- | --- | --- |
| 25 | 25 | Features | 0.0138 | 0.0145 |
| 25 | 25 | Initializer | 0.00247 | 0.00342 |
| 144 | 36 | Features | 0.0581 | 0.0587 |
| 144 | 36 | Initializer | 0.00594 | 0.00631 |
| 1,225 | 61 | Features | 0.860 | 0.792 |
| 1,225 | 61 | Initializer | 0.145 | 0.116 |

These are small engineering timing checks on generated grid inputs, not real-data
or FEEMS benchmarks. Type checks have a modest fixed cost; the vectorized work
reduces time in the larger case. The checkpoint remains byte-identical.

The comparison summary is in
[release-feature-audit-parity.json](release-feature-audit-parity.json).
Reference source, scripts, timings, and logs are retained at
`/home/jonth/turbo/dreems/release_feature_audit_20260908/`.
Temporary test files are under
`/scratch/jonth_root/jonth0/jonth/dreems/release_feature_audit_20260908/`.

The first controller stalled because `sacct` did not report its completed jobs,
although the worker logs and `squeue --states=all` confirmed success. It was
stopped after all workers finished. Only this audit's completed-output metadata
was cleared with Snakemake's cleanup command; no production state was touched.
Restarting with `--slurm-status-command squeue` reused the completed outputs and
ran only the two remaining checks. The Great Lakes profile now uses that setting
by default; this site retains completed jobs in the controller for four hours.
After that profile change, the focused workflow tests, lint checks, and a dry run
with the Slurm profile passed in a separate worker. Its controller completed
normally using the default profile, without a command-line status override.

## Workflow audit verification

The final workflow pass passed 213 tests in 291 seconds on a Great Lakes CPU
worker, submitted by Snakemake under `jonth0`. Ruff, the workflow dry run, the
executed quickstart, and the separate installed-wheel check also passed.

The new integration test generates six small coalescent examples, prepares
features, trains and evaluates a two-step model, and copies the outputs to a
path containing spaces and a colon. It then prepares sampling views without a
checkpoint, fits a fresh core, and trains its surface gate. The test verifies:

- An unchanged rerun schedules no work.
- A learning-rate change reuses raw packs, training features, and normalization.
- Replay preparation handles packs smaller than 64 examples and counts validation
  examples correctly.
- Both stages publish byte-identical copies of their selected checkpoints.
- Gate training leaves every pre-existing core parameter exactly unchanged.
- Local training does not depend on `SLURM_JOB_ID` and removes its staging folder.

These are execution and compatibility checks, not a new model experiment.
Only the tiny test corpus was generated. No production corpus, checkpoint, or
FEEMS benchmark was regenerated. The packaged checkpoint still has SHA-256
`454cb50f4c7ef1213e001d700d1bf9e9f6a353b5c8be5619aae06e9d7a22cc15`.

Logs, test results, and workflow source snapshots are retained at
`/home/jonth/turbo/dreems/release_workflow_audit_20260908/`.
Temporary test artifacts are under
`/scratch/jonth_root/jonth0/jonth/dreems/release_workflow_audit_20260908/`.

## Simulation input audit verification

The final run passed 669 tests in 626 seconds, with 20 warnings from the existing
statistical and FEEMS tests. The focused input, simulation, and dependency checks
passed 277 tests in seven seconds. Ruff, the workflow dry run, the executed
quickstart, and the separate installed-wheel check also passed. All work ran on
CPU workers through Snakemake under `jonth0`.

The frozen-source comparison checked four complete recipes and 1,000 sequences
of parameter, family, SNP-count, and sampling draws. It also checked 2,413 masks,
322 triangular lattices, 180 seeded habitat motifs, ten landscape environments,
and 25 genotype standardizations. All 20,792 compared arrays were bit-identical;
so were the valid recipes, scalar metadata, and RNG states. This includes eight
small Rust examples, with both genotype and covariance output, and seven sparse
worker examples that already succeeded before the changes.

Three additional sparse seeds reproduced the old empty-sampling failure. They
now finish through the existing deterministic retry schedule. Tests also confirm
that the attempt limit is respected and that unrelated simulator errors are not
swallowed. The first comparison stopped on this bug; its log is retained.

Grid construction now uses SciPy component labeling and NumPy edge-index arrays
instead of Python traversal and nested edge-building loops. Vectorized Shapely
operations replace per-point membership and coastline-distance loops. The tests
check exact edge order, metadata, four-neighbor connectivity, component ties,
polygon holes, and boundary inclusion. No runtime improvement is claimed here;
this was a simplification and compatibility check.

One preliminary quality run was stopped through its Snakemake controller before
the retry and workflow-dependency fixes. The final source was frozen throughout
the complete successful run. The packaged checkpoint remains byte-identical.
No production corpus, training run, gallery, or FEEMS benchmark was regenerated.

The comparison summary is in
[release-simulation-inputs-audit-parity.json](release-simulation-inputs-audit-parity.json).
Source snapshots, comparison scripts, and test reports are retained at
`/home/jonth/turbo/dreems/release_simulation_inputs_audit_20260908/`.
Temporary fixtures remain on scratch and are excluded from the durable archive,
along with duplicate weights and Snakemake database state. The broader release
audit remains open.

## Benchmark archive audit verification

The final run passed 707 tests in 868 seconds, with the same 20 existing
SciPy/FEEMS warnings. The focused archive, pairing, and dependency checks passed
55 tests in 126 seconds. Ruff, the workflow dry run, the executed quickstart,
and the separate installed-wheel check also passed. All checks used CPU workers
through Snakemake under `jonth0`; no GPU was requested.

The retained 512-case FEEMS archive passed geometry, edge-order, generating-truth,
sample-count, SNP-count, and declared LOO-protocol checks against the retained
inputs. All 1,024 restored JSON/NPZ files were byte-identical to the saved results.
Repacking preserved every member's contents. Reaggregating the same saved FEEMS
and DREEMS results preserved the CSV bytes, numerical summaries, paired
comparisons, and bootstrap intervals. The genotype-identity limitation of these
pre-hash fits is recorded as `legacy_unverified`, not hidden by the checksum.

The archive comparison took 105 seconds and peaked at about 390 MB RAM. These
are file verification and aggregation costs, not inference or FEEMS fit timings.
The integration tests execute the actual archive rules from a relocated working
directory, including paths with spaces, renamed archives, in-place reuse, and
unchanged reruns. They also test malformed archives and mismatched inputs without
overwriting existing results. The first integration run exposed the literal
input-template expansion problem described above; it passed after that fix.

One quality launch was rejected while the completed focused controller still
held its lock. It was relaunched after that controller exited. No locks or
metadata were cleared. The final full run used unchanged source. The packaged
checkpoint remains byte-identical, and no production corpus, training run, or
FEEMS benchmark was regenerated.

The comparison summary is in
[release-benchmark-archive-audit-parity.json](release-benchmark-archive-audit-parity.json).
Source snapshots, comparison scripts, and test reports are retained at
`/home/jonth/turbo/dreems/release_benchmark_archive_audit_20260908/`. Temporary
pytest fixtures, duplicate weights, and Snakemake database state stay out of the
durable archives. The broader release audit remains open.

## Training worker audit verification

The final run passed 741 tests in 724 seconds, with the same 20 existing
SciPy/FEEMS warnings. The focused worker, report, and corpus checks passed 157
tests in 17 seconds. Ruff, the workflow dry run, the executed quickstart, and
the separate installed-wheel check also passed. All checks used CPU workers
through Snakemake under `jonth0`; no GPU was requested.

The numerical comparison used 96 temporary feature fixtures across 12 packs,
not a new population-genetic corpus. Against the frozen pre-edit code, it
checked 32 ordered or resampled normalization draws, four variance floors,
and 480 seeded batches with ordinary examples or groups of two or four views.
All 7,017 compared arrays and 234,083 scalar values matched exactly. The
public parameter names and defaults were unchanged.

Four fixtures also passed through the packaged checkpoint. The single-pack
evaluation JSON and CSV files were byte-identical, as were the numerical
prediction arrays. Progress records matched except for timings and output
paths. The shared H2H CSV writer preserved the original bytes for quoted
strings, newlines, missing values, and nonfinite values. The comparison took
18 seconds and peaked at about 980 MB RAM; this is a compatibility check,
not a population-genetic accuracy or performance benchmark.

The old empty-pack loop and mixed-group batch-size error were reproduced
before checking their fixes. Separate regression tests cover multi-pack
evaluation IDs, prediction offsets after size sorting, and a limit that ends
partway through the second pack. The main workflow still evaluates one pack
per job; no published H2H was recalculated. The full suite also exercised
fresh simulation, preparation, core and gate training, publishing, and reuse
through the small end-to-end Snakemake tests.

The full verification run used unchanged source. The released checkpoint,
saved object fields, and Equinox format remain unchanged. No production
corpus, training run, or FEEMS benchmark was regenerated.

The comparison summary is in
[release-pipeline-workers-audit-parity.json](release-pipeline-workers-audit-parity.json).
Source snapshots, comparison scripts, and test reports are retained at
`/home/jonth/turbo/dreems/release_pipeline_workers_audit_20260908/`. Temporary
feature packs, pytest fixtures, duplicate weights, and Snakemake database
state stay out of the durable archives. The broader release audit remains
open; these changes have not been committed or pushed.

## FEEMS adapter and public-input verification

The final run passed 801 tests in 732 seconds. Its 38 warnings were two existing
SciPy precision warnings and 36 FEEMS `row_stack` deprecation warnings from the
larger finite-score fixture. The focused adapter, input, and workflow checks
passed 196 tests in 154 seconds. Ruff, workflow dry runs, the executed quickstart,
and the separate installed-wheel check also passed. All work ran through
Snakemake on CPU workers under `jonth0`.

Against frozen pre-edit source, 32 paired runs with a recording test double
preserved the fitting calls and warm starts. Four additional paired runs used
the actual FEEMS implementation: free q, fixed q, scalar q, and SNP-holdout
selection on the same nine-deme fixture. The selected penalties, CV scores,
rates, and fit summaries were exact matches. Every CV score in these comparisons
was finite. The unit test separately compares official serial CV with the
process-parallel fold implementation. These checks establish implementation
agreement on a usable fixture, not predictive accuracy.

All 512 retained benchmark input files passed the shared validators with
unchanged arrays. The public genotype-dataset adapter produced an identical
xarray dataset, including its QC metadata and coordinates. Two retained cases
also passed through the packaged predictor with the old and new input objects;
their results matched exactly. In total, 4,838 arrays were byte-identical and
1,886 scalar comparisons passed. Uncompressed fold NPZ files were byte-identical
too. Public parameter names, defaults, dataclass fields, and checkpoint bytes
were preserved.

The old fractional-worker truncation and complex-covariance conversion were
reproduced before checking their fixes. Other regression tests cover malformed
graphs, duplicate coordinates, invalid fold files, interrupted writes, changed
input files during fitting, and truthful CV labels in the named helper APIs.

The first numerical comparison stopped on the all-NaN fixture described above.
A subsequent attempt stopped on a variable-shadowing error in the audit script.
Both logs are retained. After correcting the fixture and script, the complete
quality and numerical comparison run used unchanged source. The successful
comparison took 83 seconds and peaked at about 1.47 GB RAM. No production corpus,
training run, FEEMS benchmark, or gallery was regenerated.

The summary is in
[release-feems-inputs-audit-parity.json](release-feems-inputs-audit-parity.json).
Source snapshots, test reports, and comparison scripts are retained at
`/home/jonth/turbo/dreems/release_feems_inputs_audit_20260908/`. Temporary datasets,
duplicate weights, and scheduler database state are excluded from the archive.
The broader release audit remains open, and these changes are not committed
or pushed.

## Checkpoint and promotion verification

The full suite passed 849 tests in 921 seconds, with the same 38 existing
SciPy/FEEMS warnings. The focused checkpoint, promotion, calibration, inference,
and workflow checks passed 125 tests in 111 seconds. Ruff, workflow dry runs,
the executed quickstart, and the isolated installed-wheel check also passed.
All checks used CPU workers through Snakemake under `jonth0`.

The frozen pre-edit promoter and the revised promoter independently rebuilt the
release from its retained core, gate, calibration, training, and H2H artifacts.
Both outputs were byte-identical to the packaged checkpoint, with SHA-256
`454cb50f4c7ef1213e001d700d1bf9e9f6a353b5c8be5619aae06e9d7a22cc15`.
A second promotion produced the same bytes. These were temporary output copies;
the packaged checkpoint was not overwritten.

Loading four retained checkpoints preserved metadata, tree structure, and every
parameter array. Two retained benchmark examples also produced identical
predictions, including raw and calibrated SDs. All 528 compared arrays were
byte-identical, and sampling fractions matched on all 512 benchmark inputs.
Metadata-only calibration edits preserved each archive member's contents.
Public parameter names, defaults, and saved object fields were unchanged.

The old duplicate-split bug was reproduced using a copy of the published fit
report. New regression tests cover that case, inconsistent fitted scales,
invalid selection scores, another checkpoint's H2H reports, mismatched training
records, changed input files during promotion, duplicate ZIP members, invalid
metadata and sample assignments, and interrupted writes. The full suite also
exercised fresh simulation, core and gate training, publishing, and reuse.

The first focused run passed before an additional check was added for nondefault
early-stopping metrics. The final complete run used unchanged source. The
compatibility comparison took 40 seconds; its worker peaked at about 985 MB RAM.
No production corpus, training run, FEEMS benchmark, or gallery was regenerated.

The comparison summary is in
[release-promotion-audit-parity.json](release-promotion-audit-parity.json).
Source snapshots, comparison code, and test reports are retained at
`/home/jonth/turbo/dreems/release_promotion_audit_20260908/`. Temporary fixtures,
duplicate weights, and scheduler database files are excluded from the durable
archive. The broader workflow and renderer audit remains open. Nothing has
been committed or pushed.

## Release validation workflow verification

The full suite passed 859 tests in 983 seconds, with the same 38 existing
SciPy/FEEMS warnings. The focused checks passed 125 tests in 220 seconds. Ruff,
the workflow dry run, the executed quickstart, and the isolated installed-wheel
check passed. All computation used CPU workers through Snakemake under `jonth0`.
The runnable source was frozen before the full run and checked again before
archiving.

The fresh integration test built a candidate from an empty directory. It
generated six 16-SNP simulations using the Rust coalescent backend, prepared
sampling views, trained a small core and its frozen-core gate, evaluated nine
independent fixture inputs, rendered both galleries, calibrated SD, and verified
the candidate's lineage. The cached FEEMS estimates were synthetic test doubles;
they test orchestration and reuse, not statistical accuracy. The test checked
that an unchanged rerun did nothing and that changing the learning rate did not
repeat simulation or FEEMS restoration. The separate integration run took 353
seconds and peaked at about 1.75 GB RAM. The full suite repeated it successfully.

Against frozen pre-edit source, all calibration JSON and CSV reports on the 512
retained legacy predictions were byte-identical. Recalibration preserved the
Equinox parameter bytes; metadata differed only in report paths. Two retained
H2H inputs produced 16 byte-identical numerical arrays, including MAP estimates,
and 110 identical untimed report fields. The additional prediction hash was
checked separately. Public parameter names and defaults were preserved, and the
packaged checkpoint's SHA-256 remains
`454cb50f4c7ef1213e001d700d1bf9e9f6a353b5c8be5619aae06e9d7a22cc15`.

The first focused run exposed shell-formatting failures when the calibration
workflows were imported as modules. The first integration attempt also exposed
a Snakefile parse error from using `checkpoint` as a multiline keyword argument.
Those failed before producing a candidate and are retained in the audit logs.
After the fixes, focused checks, integration, and the full suite passed. No
production corpus, model training, FEEMS benchmark, or gallery was regenerated.

The comparison summary is in
[release-validation-workflow-audit-parity.json](release-validation-workflow-audit-parity.json).
Source snapshots, test reports, and readable integration logs are retained at
`/home/jonth/turbo/dreems/release_validation_workflow_audit_20260908/`. Temporary
datasets, duplicate weights, scheduler databases, and compiled caches are
excluded. The remaining review items above are still open; nothing has been
committed or pushed.

## Real-data worker verification

The full suite passed 893 tests in 963 seconds, with the same 38 existing
SciPy/FEEMS warnings. The focused checks passed 76 tests in 306 seconds. Ruff,
the workflow dry run, the executed quickstart, and the isolated installed-wheel
check passed. The full suite also repeated the fresh simulation, training,
evaluation, calibration, and candidate-packaging test. All computation used
CPU workers through Snakemake under `jonth0`, with runnable source held fixed.

Against frozen pre-edit source, all 15 retained real-data inputs and their
interpolated display grids matched exactly. Fresh predictions on the gray-wolf
and darter inputs preserved all five saved arrays and untimed report fields;
the new input hashes were checked separately. The wolf and cockatoo panels were
pixel-identical using offline stock basemaps. These checks preserve the existing
results; they do not establish new predictive accuracy or repeat genotype QC.

All 36 crossed scenario definitions preserved graph arrays, truth fields,
sampling masks, and seeds. Twelve pairs of small Rust simulations also matched
exactly. Public parameter names and defaults, dataclass field order, and the
packaged checkpoint were unchanged. The checkpoint's SHA-256 remains
`454cb50f4c7ef1213e001d700d1bf9e9f6a353b5c8be5619aae06e9d7a22cc15`.

New regression tests cover lossy numeric conversions, mismatched real-data
files, invalid posterior SDs, interrupted writes, nonstandard browser JSON,
HTTPS checks, downloaded-image validation, and network fallback warnings.
A direct Requests/Pillow check on the login node returned a valid 256-by-256
topographic JPEG with HTTP 200 and certificate verification enabled. The full
renderer comparisons were offline; no production gallery was overwritten.

The first focused run exposed two mistakes in the new test fixtures: a complex
scalar was supplied where the NPZ writer requires an array, and an interpolation
assertion assumed every floating-point triangle-boundary pixel was interior.
Both fixtures were corrected before the successful focused and full runs. The
failure log is retained. Numerical comparison took 54 seconds and the worker
peaked at about 1.21 GB RAM.

A separate raw-case calibration comparison used 36 synthetic residual/SD
fixtures. Fit, selection, and verification JSON, plus the verification CSV,
were byte-identical. Canonicalization copied the packaged archive byte for byte.
Attaching calibration preserved identical archive member contents; the resulting
ZIPs also matched after normalizing only their member timestamps. The first
attempt had incorrectly required new ZIP files to have identical timestamps.
That audit-script assertion was corrected without changing package code.

The comparison summary is in
[release-real-data-audit-parity.json](release-real-data-audit-parity.json).
Raw-calibration checks are recorded in
[release-real-data-calibration-parity.json](release-real-data-calibration-parity.json).
Source snapshots, comparison scripts, test reports, and failure logs are retained
at `/home/jonth/turbo/dreems/release_real_data_audit_20260908/`. Its manifest
records archive checksums and the 118-file source freeze. Temporary datasets,
duplicate weights, scheduler databases, and compiled caches are excluded.
The broader audit remains open. No production corpus, model training, FEEMS
benchmark, or gallery was regenerated, and nothing has been committed or pushed.

## Comparison and notebook verification

The full suite passed 937 tests in 1,080 seconds, with no failures, errors, or
skips. The focused checks passed 59 tests in 210 seconds. Ruff, the workflow
dry run, the executed quickstart, and the isolated installed-wheel check passed.
The full suite repeated both training-workflow integration tests, including
fresh candidate creation and reuse of existing artifacts. Those tests use small
synthetic fixtures, not a production retrain; their cached FEEMS numbers test
orchestration rather than accuracy. All workers used CPUs through Snakemake
under `jonth0`. The full test worker peaked at about 4.77 GiB RAM.

Against frozen reference source, three notebook demographies and all twelve
sampling views matched exactly. The views were rebuilt from three retained
msprime genotype matrices, without new simulations. All genotype, covariance,
graph, and report values were preserved. The unsigned-call integration failure
described above was caught on the first comparison attempt. The successful
reference run uses the archived genotype reader from before the signed-input
check; the current worker explicitly converts its binary calls at that boundary.

The twelve-case comparison bundle preserved all 31 arrays, its CSV bytes, and
its report numbers. Report prose now describes the reused sampling views. All
512 retained broad cases matched exactly, with 3,072 identical metric results
and unchanged deterministic example selections. Six figures were pixel-identical:
both aggregate plots, three notebook truth families, and a broad-example panel.
This checks preservation of existing results, not new predictive accuracy.

Public parameter names and defaults and saved dataclass field order were
preserved. The packaged checkpoint's SHA-256 remains
`454cb50f4c7ef1213e001d700d1bf9e9f6a353b5c8be5619aae06e9d7a22cc15`.
The comparison took 106 seconds and peaked at about 411 MiB RAM. The final source
freeze covers 118 runnable source, test, workflow, and documentation files;
audit notes are excluded so these results can be recorded after verification.

The summary is in
[release-comparison-audit-parity.json](release-comparison-audit-parity.json).
Source snapshots, reference code, comparison scripts, failure logs, and test
reports are retained at
`/home/jonth/turbo/dreems/release_comparison_audit_20260909/`. Temporary datasets,
duplicate weights, scheduler databases, and compiled caches are excluded.
The broader audit remains open. No production corpus, model training, FEEMS
benchmark, or published gallery was regenerated, and nothing has been committed
or pushed.

## General benchmark gallery audit

This pass covers `benchmark_gallery.py` and `snp_range_comparison.Snakefile`.
The checkpoint and router example renderers now share their drawing code;
three copies of the correlation/MSE scatter also use one implementation.
CSV, figure, and HTML outputs use the existing atomic writers. The shared CSV
integer parser rejects fractional counts and preserves IDs beyond float64's
exact integer range. Public call parameters and model fields are not changed.

Problems found and addressed:

- The general H2H renderer did not verify that DREEMS and FEEMS archives used
  the same graph, truth, and sample counts. It now checks those arrays before
  plotting. These checks do not establish historical genotype identity.
- Duplicate case IDs could silently overwrite rows. Packed prediction offsets
  could be truncated to integers or slice the wrong edges. Both are rejected.
- Filename-only caches could retain old contents after an artifact was replaced.
  Report readers now read the current file.
- Zero MSE and undefined correlations could break scatter plots or contaminate
  best/worst selections. Undefined metrics remain undefined, not zero; those
  cases are excluded from the relevant rankings but can still appear at random.
- A reused reference gallery could overwrite its destination before a malformed
  heading or provenance file was detected. Those checks now precede copying;
  source and destination must not overlap. The whole-directory copy is not an
  atomic transaction.
- SNP report text assumed particular counts and paths, and claimed an exact
  genotype prefix without reading the genotype matrices. Counts and links now
  come from the inputs; the report states that prefix identity is unverified.
- The SNP workflow relied on module-global shell variables and the parent
  workflow's directory. Imported rules now use their own source directory and
  explicit parameters. Code and environment dependencies include calibration
  and validation; reference-gallery files are tracked individually. Evaluation
  caps a reused final pack at the requested number of examples.

The corrected focused run passed all 75 tests in 262 seconds. The first run
passed 74 and failed one new workflow test because its pre-created outputs
correctly scheduled no work. The test fixture was corrected. Its imported
report rules execute, and dependency tests check reuse and invalidation without
running scientific fits. No production data or fits were rerun.

Against frozen pre-edit source, the 4,096 saved checkpoint-comparison rows,
their CSV bytes, router decisions, statistics, and deterministic selections
matched exactly. All 512 retained H2H cases and 24 low/high-SNP scenario
archives matched; the twelve paired SNP-count comparisons were unchanged.
The test also compared 128 slices from the packed prediction archives.
Ten figures were pixel-identical at a shared 70-DPI audit resolution. Production
output remains 190-DPI PNG plus PDF; the focused tests exercise that writer
separately. The frozen-source rerun took 102 seconds and used about 624 MiB RAM.

The full suite passed 973 tests in 1,271 seconds, with no failures, errors, or
skips. Ruff, the workflow dry run, the executed quickstart, and the isolated
installed-wheel check also passed. The full suite repeated both training
workflow integration tests, including the 66-step fresh release DAG and artifact
reuse. These use small synthetic fixtures; their cached FEEMS numbers test
orchestration, not predictive accuracy. All checks used CPU workers through
Snakemake under `jonth0`. The full test worker peaked at about 4.83 GiB RAM.

The 119-file source freeze and the packaged checkpoint remained unchanged.
The comparison summary is in
[release-gallery-audit-parity.json](release-gallery-audit-parity.json).
Verified source, scripts, test reports, and the initial failure log are retained
at `/home/jonth/turbo/dreems/release_gallery_audit_20260909/`. The archive excludes
temporary datasets, duplicate weights, scheduler databases, and compiled caches.
Temporary test copies from this pass are removed after archiving the evidence.
The broader audit remains open, and nothing has been committed or pushed.

## Final graph and API review

The path sampler solved the same shortest-path problem twice, once for
distances and again for predecessors. One solve now returns both. Its
fallback spatial-transect sampler is shared with the land sampler, retaining
their different quantile bounds and jitter scales. Duplicate removal no
longer rebuilds a set for every remaining candidate node.

Three behavior fixes are separate from that refactor:

- Zero, negative, nonfinite, or numerically nonadvancing hex-grid spacing is
  rejected before the enumeration loops. Bounds and graph indices are also
  checked before they can produce invalid meshes or wraparound indexing.
- An edgeless graph's largest component is one node, not all its disconnected
  nodes. Empty and single-node inputs remain supported. The production crop
  builder already rejects edgeless meshes, so this does not change accepted
  crops or consume additional randomness.
- Land-path sampling no longer catches every exception and silently changes
  the sampling design. Disconnected land subgraphs still use the existing
  spatial fallback; unexpected solver failures now propagate. Land masks must
  be boolean, match the node count, and contain an eligible node. Invalid path
  probabilities are rejected instead of clipped.

The remaining graph and motif functions now have explicit runtime type
checks and short descriptions of their role. Shape checks cover path and land
samplers, predecessor arrays, and mesh components. Documentation distinguishes
synthetic coast/corridor motifs from actual world-map crops, synthetic ridges
from measured elevation, and all-node sampling fractions from land-only ones.
FEEMS lattice surfaces preserve specified conductance ratios; other procedural
surfaces are rescaled to their requested log SD.

Legacy sampling conventions were retained, not silently redesigned. Site
counts are clipped to available nodes. In the wolf-like and global-landscape
samplers, the minimum-site requirement takes precedence over a conflicting
total-individual cap; recipes should not set a cap below minimum sites times
the largest per-site count. The earth-like generator still retries invalid
random masks, and the full recipe is validated before pipeline generation.
Changing these conventions would need an explicit simulation-design decision,
not a release cleanup.

The final public API review confirmed array/mapping and xarray entry points,
call-level genotype QC, precision counts, and output uncertainty semantics.
The dataset module imports integer validation from its owner rather than
through inference. Additional docstrings distinguish complete dosage input
from the MAF/missing-call QC convenience path and explain batch padding and
the different refinement defaults. No public parameters, defaults, or saved
fields changed.

All 220 focused tests passed in five seconds. Against the pre-edit source,
1,656 seeded calls produced identical values and subsequent RNG states:
4,259 arrays and 15,292 scalar values matched exactly. The checks cover float32
and float64 coordinates, connected/disconnected paths, uniform and transect
fallbacks, all twelve surface families, coalescence priors, lattice/grid/wolf-like
cases, migration-associated sampling, and eight projected landscape cases.
The landscape tests use a fixed synthetic coastline and lake to avoid network
access; they exercise the real projection and mesh code, not a world-map
download. All 72 reviewed public function signatures and six input/result
dataclass field layouts matched. The packaged checkpoint hash is unchanged.
See [the comparison record](release-graph-api-audit-parity.json).

The final full suite passed all 998 tests in 1,043 seconds, with no failures,
errors, or skips. Ruff, the workflow dry run, the executed quickstart notebook,
and the clean installed-wheel check also passed. The test worker peaked at
4,935 MiB (about 4.82 GiB). It repeated staged training and the fresh 66-step
release DAG, including independent evaluation, calibration, publication, and
reuse. These use tiny datasets and cached synthetic FEEMS results to test
orchestration, not scientific accuracy.

All checks ran through Snakemake on standard CPU workers under `jonth0`, with
no GPU jobs. The frozen 120-file source manifest and the packaged checkpoint
remained unchanged. The final source and verification bundle is at
`/home/jonth/turbo/dreems/release_graph_api_audit_20260909/`. Its manifest records
archive checksums, test results, and the checkpoint identity. This pass's
temporary test data are removed after the evidence is archived and checked.
The release audit is complete. At that point, nothing had been committed or pushed.
