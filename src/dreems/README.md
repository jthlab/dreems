# DREEMS Package

This package is the publication-facing API for Deep Recovery of Effective
Migration Surfaces (DREEMS). Code here
should be reusable, tested, and stable enough for paper figures and reproducible
evaluation scripts.

Modules:

- `dreems.datasets`: the xarray input/output schema and the `create_dataset`
  and `infer` entry points.
- `dreems.genotypes`: sample-coordinate matching, genotype QC, and conversion
  from sgkit-compatible calls to node covariance.
- `dreems.metrics`: centered edge-log surface metrics used in DREEMS-vs-FEEMS
  comparisons.
- `dreems.inference`: user-facing checkpoint loading and prediction API:
  `load_model`, `DREEMSInput`, `DREEMSPredictor`, and `DREEMSResult`.
- `dreems.cli`: command-line wrapper around the user-facing inference API.
- `dreems.validation`: common graph, covariance, and index checks. These do not
  depend on the neural architecture.
- `dreems.h2h`: DREEMS benchmark inference, paired reports, and the combined CLI.
- `dreems.feems_benchmark`: FEEMS-only workers, separate from neural and rendering
  code so model changes do not invalidate expensive FEEMS fits.
- `dreems.benchmark_data`: common simulated-case inputs and report fields.
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
- `dreems.geography` and `dreems.motifs`: graph construction, procedural
  migration/coalescence fields, and site-sampling designs. These run before
  the simulator; they are not learned components of the model.
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

## FEEMS Inputs And Fitting

`dreems.feems` calls the official FEEMS package. Its input is a complete
diploid dosage matrix with samples in rows, zero-based sample-node IDs,
`graph={"edges": edges}`, and one coordinate pair per graph node. Returned log
rates follow the input edge order. Missing genotypes must be handled upstream;
the adapter removes monomorphic SNPs and uses FEEMS' heterozygosity scaling,
but does not apply MAF or LD filtering itself.

`fit_feems` uses the supplied `FEEMSConfig` to select the variance model and CV
method. The named helpers select the method in their name and use the config
for penalty grids and solver settings. `fit_feems_recommended_loo` fixes the
publication protocol, including the stricter default tolerance in the final
FEEMS refit. Worker count changes only how independent folds are scheduled.

Direct covariance inputs to DREEMS must already use the expected SNP scaling;
they cannot be filtered for MAF after the SNPs have been collapsed into a matrix.
Use `from_genotype_dataset` when QC and covariance construction are still needed.
