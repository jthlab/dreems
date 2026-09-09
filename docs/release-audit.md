# Release Code Review

The refactor was verified on September 9, 2026. Array types, documentation,
and module cleanup are complete on `refactor`. The packaged checkpoint is
unchanged.

## Design

- `dreems.create_dataset` constructs an input from graph coordinates, edges,
  covariance, and sample counts. `dreems.infer` runs the packaged model.
- `dreems.from_genotype_dataset` filters allele calls, imputes missing dosages,
  assigns samples to graph nodes, and constructs the covariance.
- The model uses JAX, Equinox, and Optax. Checkpoints use Equinox serialization.
- Snakemake owns simulation, training, evaluation, calibration, and publishing.
  FEEMS fits have separate rules from neural inference and plotting.

## Corrections

The review found and corrected these problems:

- Invalid indices, counts, and complex arrays could be silently converted.
- Failed numerical solves and nonfinite training could appear successful.
- A failed write could damage an existing checkpoint or simulation pack.
- Sampling changes could leave covariance and sample metadata inconsistent.
- Genotype masks and completely missing samples were handled inconsistently.
- The Python simulator could misreport the number of usable SNPs.
- Calibration and comparison reports could combine mismatched artifacts.
- Workflow dependencies could repeat FEEMS fits or leave predictions stale.
- Mesh generation could hang on invalid spacing.
- Land sampling could hide an unexpected path-solver error.

Related training, inference, validation, checkpoint, and report code was
consolidated. Retired experiments were archived outside the repository.
The inventory and archive checksums are in
[the verification record](release-verification.json).

## Verification

At commit `4631246`:

- All 998 tests passed, with no failures, errors, or skips.
- Ruff, workflow dry runs, the executable notebook, and a clean CPU wheel
  installation passed.
- Small end-to-end tests exercised fresh simulations, core and gate training,
  evaluation, calibration, publication, and reuse of saved artifacts.
- Seeded graph and sampling comparisons matched 1,656 calls, including the
  subsequent RNG states.
- Earlier comparisons checked model outputs, gradients, Equinox serialization,
  512 saved H2H cases, and 4,096 saved checkpoint-comparison rows.
- The packaged checkpoint remained byte-identical.

These results apply to the recorded source revision. No production corpus
or FEEMS fit was regenerated during the review.

The [verification record](release-verification.json) preserves the earlier
JSON reports under their original filenames. The
[detailed review at that revision](https://github.com/jthlab/dreems/blob/463124663911453685e32023fd9e8e24bc58ad4e/docs/release-audit.md)
contains the individual findings and tests.

### Branch Checks

The next update consolidated simulation and benchmark modules and added
array dimensions to their interfaces. Its first focused test run passed
339 tests, including the fresh training and release workflows, and found
three annotation issues. Commit `31b0038` corrects those issues while
preserving scalar calibration inputs and existing shape-validation errors.
All 166 tests in the follow-up run passed, including a new regression test.
Ruff checks and formatting passed. The checkpoint checksum is unchanged.

All 24 earlier JSON records were preserved in the combined verification
file. Those checks covered the affected modules.

### Core Interfaces

Commit `c3a0aed` adds array layouts to feature and training data, removes
redundant FEEMS wrappers, and reads checkpoint dimensions from the model.
At that revision, the annotation test covered 18 core modules.

The full test run passed 1,022 tests and found one validation regression:
a FEEMS shape annotation raised a type error before the existing sample-count
check could raise `ValueError`. The commit preserves that check. All 179
follow-up tests passed, including the expanded FEEMS input tests, comparisons
with the official solver, and inference tests. The complete suite was not
rerun after this final annotation correction.

Ruff, workflow dry runs, the executable notebook, and a clean CPU wheel
installation passed again at this commit. All 66 arrays in the feature and
prediction comparison matched `d54a795` exactly. The checkpoint is unchanged.

### Final Checks

Commit `563ad02` completes dimension annotations across all 32 package modules
and five Python workflow workers. An AST test checks for bare array annotations
throughout those files. Shared archive readers replace duplicate helpers.
Commit `aa24c68` consolidates package documentation in the main README and
simplifies explanations in the workflow guide.

With the source held at `aa24c68` throughout the checks:

- All 1,051 tests passed, with no failures, errors, or skips, in 1,082.775 seconds.
- Ruff lint and format checks passed for the package, tests, and Python
  workflow workers.
- Workflow dry runs, launcher syntax checks, the quickstart notebook, and a
  clean CPU wheel installation passed.
- The fresh release workflow completed all 66 steps on small test fixtures.
  Subsequent invocations reused the saved outputs.
- All 231 seeded simulation cases matched `a06c4fb` exactly, including array
  dtypes and subsequent RNG states. These cover grid and hex meshes, masks,
  surface families, and the Rust and Python island simulators.
- The packaged checkpoint SHA256 remained
  `454cb50f4c7ef1213e001d700d1bf9e9f6a353b5c8be5619aae06e9d7a22cc15`.

Numerical checks ran through Snakemake on the standard Slurm partition using
`jonth0`. No production corpus, model training run, or FEEMS benchmark was
repeated for this refactor.

## Limits

Old prediction files without input or checkpoint hashes have incomplete
provenance. Agreement on graph geometry cannot recover that missing history.

Reported posterior SDs describe the neural estimate. MAP refinement supplies
a separate point estimate. Calibration on held-out simulations does not
guarantee calibration on every real dataset.

The wolf-like and global-landscape samplers give minimum site counts priority
over a conflicting total-sample cap. Set the cap high enough for the requested
minimum sites and maximum individuals per site. Changes to this convention
would change the simulation distribution.
