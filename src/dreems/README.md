# DREEMS Package

This package is the publication-facing API for Deep Recovery of Effective
Migration Surfaces (DREEMS). Code here
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
- `dreems.model`: production GNN forward pass, including the edgewise blend
  between its neural proposal and the linearized initializer and the final
  graph-level homogeneous-surface gate.
- `dreems.losses`: supervised NLL, shape, initializer-regret, and paired-mask
  consistency losses.
- `dreems.counterfactual`: deterministic sampling masks derived from existing
  covariance simulations for sampling-robustness training.
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

The packaged v3 checkpoint is the default for `dreems.infer`, `load_model`, and
the CLI. Its `surface_weight` diagnostic lies in `[0, 1]`: zero removes all
centered edge contrast, while one keeps the complete predicted surface.
