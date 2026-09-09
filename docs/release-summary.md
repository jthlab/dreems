# Release Audit Summary

The release code audit is complete as of September 9, 2026. It keeps the
released model and its public interface intact. It does not propose a new
architecture or claim improved predictive performance. The results below
describe the source verified at audit completion, before it was committed.

## Verification

- All 998 tests passed, with no failures, errors, or skips.
- Ruff, the workflow dry run, the executed quickstart, and installation into a
  clean CPU-only environment passed.
- The suite exercised both staged training and a fresh release DAG, including
  evaluation, calibration, publishing, and artifact reuse on small fixtures.
- The final graph/sampling comparison matched 1,656 seeded calls exactly,
  including subsequent RNG states. Earlier comparisons cover model outputs,
  gradients, Equinox serialization, 512 saved H2H cases, and 4,096 saved
  checkpoint-comparison rows.
- The packaged checkpoint remains byte-identical. No production corpus,
  training run, FEEMS fit, or gallery was regenerated for this audit.

The final checks ran through Snakemake on Great Lakes CPU workers under
`jonth0`. The full suite took 17.4 minutes and peaked at about 4.82 GiB RAM.
The source and verification bundle is at
`/home/jonth/turbo/dreems/release_graph_api_audit_20260909/`.

## What Changed

- Shared the repeated training, inference, checkpoint, validation, and report
  code. Model parameters still live in Equinox modules and use Equinox leaf
  serialization.
- Removed retired experimental workflows after archiving them outside the
  repository. The learned production surface gate is retained.
- Made runtime type checks explicit and added documentation for array order,
  sampling counts, covariance scaling, losses, uncertainty, and refinement.
- Connected fresh corpus generation, sampling views, core training, gate
  training, evaluation, calibration, and release packaging through Snakemake.
- Separated FEEMS work from model evaluation and rendering dependencies, so
  changing the neural checkpoint does not require fitting FEEMS again.

## Important Fixes

The audit found failures that ordinary successful runs would not expose:

- Nonfinite training or failed numerical solves could appear successful.
- Input conversion could silently truncate indices or discard imaginary parts.
- Failed writes could damage an existing checkpoint or corpus pack.
- Missing-call masks, unsupported samples, and sampling views could leave
  covariance and sample metadata inconsistent.
- The Python covariance simulator could report the wrong usable SNP count.
- Calibration and comparison reports could combine mismatched artifacts.
- Some workflow dependencies could rerun FEEMS unnecessarily or leave neural
  outputs stale.

Each finding, its correction, and its tests are recorded in the
[full audit](release-audit.md). The `release-*-parity.json` files summarize
comparisons against saved source and results. They are verification records,
not additional model benchmarks.

## Where To Start

- [User API and installation](../README.md#inference-api)
- [Executable quickstart notebook](../examples/dreems_quickstart.md)
- [Package layout](../src/dreems/README.md)
- [Training and evaluation workflows](../workflow/README.md)
- [Retired workflow inventory](retired-workflows.json)

Historical prediction files without input/checkpoint hashes cannot be given
retroactive provenance. Their limitations remain explicit in the reports.
Passing these checks does not establish calibration on arbitrary real data or
guarantee recovery in poorly sampled regions.
