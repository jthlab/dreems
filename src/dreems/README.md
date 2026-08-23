# DREEMS Package

This package is the publication-facing API for the DREEMS project. Code here
should be reusable, tested, and stable enough for paper figures and reproducible
evaluation scripts.

Modules:

- `dreems.metrics`: centered edge-log surface metrics used in DREEMS-vs-FEEMS
  comparisons.
- `dreems.inference`: user-facing checkpoint loading and prediction API:
  `load_model`, `DREEMSInput`, `DREEMSPredictor`, and `DREEMSResult`.
- `dreems.cli`: command-line wrapper around the user-facing inference API.
- `dreems.features`: topology-invariant covariance, graph-diffusion-band, and
  all-pair linearized-resistance feature construction.
- `dreems.model`: production GNN forward pass.
- `dreems.losses`: supervised NLL and shape losses.
- `dreems.corpus`: one-example MAF-conditioned simulation worker and
  deterministic raw-pack writer used by Snakemake; raw corpus artifacts are
  independent of training code.
- `dreems.pipeline`: artifact-oriented feature preparation, normalization,
  training, and evaluation CLI used by `workflow/Snakefile`.
- `dreems.island_sim`: sparse island-model coalescent simulator with genotype
  and direct sample-covariance outputs, an optional Rust/PyO3 backend, and a
  tested Python reference path.
- `dreems.train`: single-device production JAX training loop and checkpoint
  writer.
- `dreems.calibration`: explicit post-hoc SD calibration utilities.
- `dreems.checkpoints`: checkpoint loading and prediction API.
- `dreems.feems`: canonical FEEMS baseline protocols:
  - `free_q`: FEEMS v2, joint LOO CV over `lambda` and `lambda_q`.
  - `fixed_q`: FEEMS with LOO CV over `lambda` and fixed residual variance.
  - `scalar_q`: one learned global residual variance scalar.
