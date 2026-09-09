# Release Code Review

The first refactor was verified on September 9, 2026. The checkpoint is
unchanged. Further work on array types, documentation, and module layout is
in progress on `refactor`.

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
file. The complete test suite has not been rerun for this update.

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
