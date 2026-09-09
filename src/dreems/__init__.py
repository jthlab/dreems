"""Publication-facing DREEMS APIs."""

from dreems.calibration import SDCalibration
from dreems.datasets import (
    create_dataset,
    infer,
    validate_dataset,
)
from dreems.genotypes import (
    attach_sample_coordinates,
    from_genotype_dataset,
    from_vcf_zarr,
)
from dreems.inference import (
    DREEMSInput,
    DREEMSPredictor,
    DREEMSResult,
    InferenceExample,
    NodeCovarianceInput,
    genotype_covariance,
    load_model,
    predict,
)
from dreems.island_sim import (
    IslandCovarianceResult,
    IslandSimulationResult,
    require_rust_backend,
    rust_backend_available,
    simulate_island_covariance,
    simulate_island_model,
)
from dreems.linearized import (
    LinearizedConductanceEstimate,
    all_pair_adjoint_conductance_initializer,
)
from dreems.metrics import EdgeLogMetrics, centered, edge_log_metrics
from dreems.refinement import (
    CovarianceFit,
    CovarianceScales,
    DREEMSMapRefinementConfig,
    DREEMSMapRefinementResult,
    edge_line_roughness,
    q_covariance_at_observed_nodes,
    refine_edge_log_conductance,
)

__all__ = [
    "CovarianceFit",
    "CovarianceScales",
    "DREEMSPredictor",
    "DREEMSResult",
    "DREEMSInput",
    "DREEMSMapRefinementConfig",
    "DREEMSMapRefinementResult",
    "EdgeLogMetrics",
    "InferenceExample",
    "IslandCovarianceResult",
    "IslandSimulationResult",
    "LinearizedConductanceEstimate",
    "NodeCovarianceInput",
    "SDCalibration",
    "all_pair_adjoint_conductance_initializer",
    "attach_sample_coordinates",
    "centered",
    "create_dataset",
    "edge_line_roughness",
    "edge_log_metrics",
    "genotype_covariance",
    "from_genotype_dataset",
    "from_vcf_zarr",
    "infer",
    "load_model",
    "predict",
    "q_covariance_at_observed_nodes",
    "refine_edge_log_conductance",
    "require_rust_backend",
    "rust_backend_available",
    "simulate_island_covariance",
    "simulate_island_model",
    "validate_dataset",
]
