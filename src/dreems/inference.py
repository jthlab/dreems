"""Load a checkpoint and estimate migration from covariance or genotypes."""

from __future__ import annotations

# ruff: noqa: F722, F821, UP037
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
from beartype import beartype
from jaxtyping import Float, Int, Shaped, jaxtyped

from dreems import checkpoints, features, metrics, refinement
from dreems.validation import (
    _effective_snp_count,
    _integer_array,
    _real_array,
    validate_covariance_arrays,
    validate_genotype_arrays,
    validate_graph_arrays,
    validate_node_covariance_arrays,
)


@dataclass(frozen=True)
class DREEMSInput:
    """Sample covariance and geographic coordinates.

    Parameters
    ----------
    sample_covariance
        Covariance matrix between samples with shape ``(samples, samples)``.
    sample_coordinates
        Coordinates for rows/columns of ``sample_covariance`` with shape
        ``(samples, 2)``. Samples are assigned to their nearest graph node.
    node_coordinates
        Graph node coordinates with shape ``(nodes, 2)``.
    edges
        Undirected graph edges with shape ``(edges, 2)``. Edge endpoint indices
        must refer to rows in ``node_coordinates``.
    effective_snps
        Optional number of independent SNPs used to estimate the covariance.
        If omitted, DREEMS uses the sample count as a fallback. Covariance alone
        cannot determine this count; supply it for meaningful uncertainty.
    name
        Optional label propagated to prediction results.
    metadata
        Optional user metadata propagated to prediction results.
    """

    sample_covariance: Float[np.ndarray, "samples samples"]
    sample_coordinates: Float[np.ndarray, "samples 2"]
    node_coordinates: Float[np.ndarray, "nodes 2"]
    edges: Int[np.ndarray, "edges 2"]
    effective_snps: int | None = None
    name: str | None = None
    metadata: Mapping[str, Any] | None = None

    @beartype
    def __post_init__(self) -> None:
        """Convert array-like inputs before checking graph and observation dimensions."""
        sample_covariance = _real_array(self.sample_covariance, name="sample_covariance")
        sample_coordinates = _real_array(self.sample_coordinates, name="sample_coordinates")
        node_coordinates = _real_array(self.node_coordinates, name="node_coordinates")
        edges = _integer_array(np.asarray(self.edges), name="edges", dtype=np.int32)
        metadata = {} if self.metadata is None else dict(self.metadata)

        validate_graph_arrays(positions=node_coordinates, edges=edges)
        validate_covariance_arrays(
            sample_covariance=sample_covariance,
            sample_coordinates=sample_coordinates,
        )
        if self.effective_snps is not None:
            object.__setattr__(self, "effective_snps", _effective_snp_count(self.effective_snps))

        object.__setattr__(self, "sample_covariance", sample_covariance)
        object.__setattr__(self, "sample_coordinates", sample_coordinates)
        object.__setattr__(self, "node_coordinates", node_coordinates)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "metadata", metadata)

    @property
    @beartype
    def positions(self) -> Float[np.ndarray, "nodes 2"]:
        """Coordinates of graph nodes."""

        return self.node_coordinates

    @beartype
    def to_raw_example(self) -> dict[str, Any]:
        """Build the input mapping for feature construction."""

        payload: dict[str, Any] = {
            "graph": {
                "active_full": np.arange(self.node_coordinates.shape[0], dtype=np.int32),
                "edges": self.edges,
            },
            "active_positions": self.node_coordinates,
            "sample_coordinates": self.sample_coordinates,
            "sample_covariance": self.sample_covariance,
        }
        if self.effective_snps is not None:
            payload["effective_snps"] = self.effective_snps
        if self.name is not None:
            payload["name"] = self.name
        return payload


@dataclass(frozen=True)
class NodeCovarianceInput:
    """Covariance between sampled graph nodes.

    ``node_covariance`` contains covariance among the observed graph nodes,
    whose graph indices are given by ``observed_nodes``. ``node_sample_count``
    records how many samples contributed to every graph node.
    """

    node_covariance: Float[np.ndarray, "observed observed"]
    observed_nodes: Int[np.ndarray, "observed"]
    node_sample_count: Int[np.ndarray, "nodes"]
    node_coordinates: Float[np.ndarray, "nodes 2"]
    edges: Int[np.ndarray, "edges 2"]
    effective_snps: int
    name: str | None = None
    metadata: Mapping[str, Any] | None = None

    @beartype
    def __post_init__(self) -> None:
        """Preserve observed-node order and require matching sample counts."""
        node_covariance = _real_array(self.node_covariance, name="node_covariance")
        observed_nodes = _integer_array(
            np.asarray(self.observed_nodes), name="observed_nodes", dtype=np.int32
        )
        node_sample_count = _integer_array(
            np.asarray(self.node_sample_count), name="node_sample_count", dtype=np.int64
        )
        node_coordinates = _real_array(self.node_coordinates, name="node_coordinates")
        edges = _integer_array(np.asarray(self.edges), name="edges", dtype=np.int32)
        metadata = {} if self.metadata is None else dict(self.metadata)

        validate_graph_arrays(positions=node_coordinates, edges=edges)
        validate_node_covariance_arrays(
            node_covariance=node_covariance,
            observed_nodes=observed_nodes,
            node_sample_count=node_sample_count,
            num_nodes=node_coordinates.shape[0],
        )

        object.__setattr__(self, "node_covariance", node_covariance)
        object.__setattr__(self, "observed_nodes", observed_nodes)
        object.__setattr__(self, "node_sample_count", node_sample_count)
        object.__setattr__(self, "node_coordinates", node_coordinates)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "effective_snps", _effective_snp_count(self.effective_snps))
        object.__setattr__(self, "metadata", metadata)

    @property
    @beartype
    def positions(self) -> Float[np.ndarray, "nodes 2"]:
        """Coordinates of graph nodes."""

        return self.node_coordinates

    @beartype
    def to_raw_example(self) -> dict[str, Any]:
        """Expose the validated arrays to feature construction without reordering."""

        payload: dict[str, Any] = {
            "graph": {
                "active_full": np.arange(self.node_coordinates.shape[0], dtype=np.int32),
                "edges": self.edges,
            },
            "active_positions": self.node_coordinates,
            "node_covariance": self.node_covariance,
            "observed_nodes": self.observed_nodes,
            "node_sample_count": self.node_sample_count,
            "effective_snps": self.effective_snps,
        }
        if self.name is not None:
            payload["name"] = self.name
        return payload


@dataclass(frozen=True)
class InferenceExample:
    """Complete diploid dosages with a graph node assigned to each sample.

    Use ``dreems.from_genotype_dataset`` to filter and impute allele calls.
    """

    positions: Float[np.ndarray, "nodes 2"]
    edges: Int[np.ndarray, "edges 2"]
    sample_nodes: Int[np.ndarray, "samples"]
    genotypes: Float[np.ndarray, "samples snps"]
    name: str | None = None
    metadata: Mapping[str, Any] | None = None

    @beartype
    def __post_init__(self) -> None:
        """Validate genotype rows and graph assignments before constructing features."""
        positions = _real_array(self.positions, name="positions")
        edges = _integer_array(np.asarray(self.edges), name="edges", dtype=np.int32)
        sample_nodes = _integer_array(
            np.asarray(self.sample_nodes), name="sample_nodes", dtype=np.int32
        )
        genotypes = _real_array(self.genotypes, name="genotypes")
        metadata = {} if self.metadata is None else dict(self.metadata)

        validate_genotype_arrays(
            positions=positions,
            edges=edges,
            sample_nodes=sample_nodes,
            genotypes=genotypes,
        )

        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "sample_nodes", sample_nodes)
        object.__setattr__(self, "genotypes", genotypes)
        object.__setattr__(self, "metadata", metadata)

    @beartype
    def to_raw_example(self) -> dict[str, Any]:
        """Build the input mapping for feature construction."""

        payload: dict[str, Any] = {
            "graph": {
                "active_full": np.arange(self.positions.shape[0], dtype=np.int32),
                "edges": self.edges,
            },
            "active_positions": self.positions,
            "sample_nodes": self.sample_nodes,
            "genotypes": self.genotypes,
        }
        if self.name is not None:
            payload["name"] = self.name
        return payload


InferenceLike = DREEMSInput | NodeCovarianceInput | InferenceExample | Mapping[str, Any]


@dataclass(frozen=True)
class _RefinementObservation:
    node_covariance: Float[np.ndarray, "observed observed"]
    observed_nodes: Int[np.ndarray, "observed"]
    observed_sample_count: Float[np.ndarray, "observed"] | Int[np.ndarray, "observed"]
    effective_snps: int


@jaxtyped(typechecker=beartype)
def _edge_records(
    *,
    edge_log_mean: "Float[np.ndarray, 'edges']",
    edge_log_sd: "Float[np.ndarray, 'edges'] | None",
    edges: "Int[np.ndarray, 'edges 2']",
    positions: "Float[np.ndarray, 'nodes 2']",
    surface_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Export centered rates in the common DREEMS/FEEMS edge-table format.

    FEEMS has no posterior SD; ``None`` keeps the existing NaN table column.
    The caller validates the graph and centers the log rates before exporting.
    """
    relative = np.exp(edge_log_mean)
    return [
        {
            "edge_index": index,
            "left_node": int(left),
            "right_node": int(right),
            "left_x": float(positions[left, 0]),
            "left_y": float(positions[left, 1]),
            "right_x": float(positions[right, 0]),
            "right_y": float(positions[right, 1]),
            "edge_log_mean": float(edge_log_mean[index]),
            "edge_log_sd": np.nan if edge_log_sd is None else float(edge_log_sd[index]),
            "relative_migration_mean": float(relative[index]),
            "surface_weight": surface_weight,
        }
        for index, (left, right) in enumerate(edges)
    ]


@dataclass(frozen=True)
class DREEMSResult:
    """Edge log rates and pointwise neural uncertainty for one graph.

    Arrays follow the supplied edge and node order. Optional MAP rates are a
    separate point estimate; the SD fields still describe the neural output.
    """

    edge_log_mean: Float[np.ndarray, "edges"]
    edge_log_sd: Float[np.ndarray, "edges"]
    edges: Int[np.ndarray, "edges 2"]
    positions: Float[np.ndarray, "nodes 2"]
    raw_edge_log_sd: Float[np.ndarray, "edges"] | None = None
    surface_weight: float = 1.0
    node_log_q_mean: Float[np.ndarray, "nodes"] | None = None
    node_log_q_sd: Float[np.ndarray, "nodes"] | None = None
    name: str | None = None
    checkpoint_path: Path | None = None
    metadata: Mapping[str, Any] | None = None
    refined_edge_log_mean: Float[np.ndarray, "edges"] | None = None
    refinement_diagnostics: Mapping[str, Any] | None = None

    @beartype
    def __post_init__(self) -> None:
        """Coerce array-like inputs, then reject malformed or nonfinite results."""
        edges = _integer_array(np.asarray(self.edges), name="edges", dtype=np.int32)
        positions = _real_array(self.positions, name="positions")
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edges must have shape (edges, 2)")
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must have shape (nodes, 2)")
        if np.any(edges < 0) or np.any(edges >= positions.shape[0]):
            raise ValueError("edges must refer to rows in positions")
        if (self.node_log_q_mean is None) != (self.node_log_q_sd is None):
            raise ValueError("node_log_q_mean and node_log_q_sd must be supplied together")

        # Results can contain an edge subset, so connectivity is checked on input.
        for name, size in (
            ("edge_log_mean", edges.shape[0]),
            ("edge_log_sd", edges.shape[0]),
            ("raw_edge_log_sd", edges.shape[0]),
            ("refined_edge_log_mean", edges.shape[0]),
            ("node_log_q_mean", positions.shape[0]),
            ("node_log_q_sd", positions.shape[0]),
        ):
            value = getattr(self, name)
            if value is None and name not in {"edge_log_mean", "edge_log_sd"}:
                continue
            values = _real_array(value, name=name)
            if values.shape != (size,):
                raise ValueError(f"{name} must have shape ({size},), got {values.shape}")
            if name.endswith("_sd") and np.any(values < 0.0):
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, values)

        surface_weight = float(self.surface_weight)
        if not np.isfinite(surface_weight) or not 0.0 <= surface_weight <= 1.0:
            raise ValueError("surface_weight must be finite and between zero and one")
        object.__setattr__(self, "surface_weight", surface_weight)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "metadata", {} if self.metadata is None else dict(self.metadata))
        if self.refinement_diagnostics is not None:
            object.__setattr__(self, "refinement_diagnostics", dict(self.refinement_diagnostics))

    @property
    @jaxtyped(typechecker=beartype)
    def centered_edge_log_mean(self) -> "Float[np.ndarray, 'edges']":
        """Predicted edge log-conductances after removing the arbitrary offset."""

        return metrics.centered(self.edge_log_mean)

    @property
    @jaxtyped(typechecker=beartype)
    def centered_refined_edge_log_mean(self) -> "Float[np.ndarray, 'edges'] | None":
        """MAP-refined edge log-conductances, when refinement was requested."""

        if self.refined_edge_log_mean is None:
            return None
        return metrics.centered(self.refined_edge_log_mean)

    @property
    @jaxtyped(typechecker=beartype)
    def relative_migration_mean(self) -> "Float[np.ndarray, 'edges']":
        """Exponentiate the centered mean log rate, giving geometric mean one.

        This is not the arithmetic posterior mean of migration under a lognormal
        distribution. The property name is retained for API compatibility.
        """

        return np.exp(self.centered_edge_log_mean)

    @property
    @jaxtyped(typechecker=beartype)
    def refined_relative_migration_mean(self) -> "Float[np.ndarray, 'edges'] | None":
        """MAP-refined relative migration, when refinement was requested."""

        centered = self.centered_refined_edge_log_mean
        return None if centered is None else np.exp(centered)

    @property
    @beartype
    def refinement_applied(self) -> bool:
        """Whether a separate covariance MAP point estimate is available."""

        return self.refined_edge_log_mean is not None

    @jaxtyped(typechecker=beartype)
    def centered_edge_log_interval(
        self, level: Real = 0.95
    ) -> tuple["Float[np.ndarray, 'edges']", "Float[np.ndarray, 'edges']"]:
        """Neural-posterior pointwise interval for centered edge log-conductances."""

        z = normal_interval_z(level)
        mean = self.centered_edge_log_mean
        sd = self.edge_log_sd
        return mean - z * sd, mean + z * sd

    @jaxtyped(typechecker=beartype)
    def relative_migration_interval(
        self, level: Real = 0.95
    ) -> tuple["Float[np.ndarray, 'edges']", "Float[np.ndarray, 'edges']"]:
        """Pointwise interval for relative migration on the original scale."""

        lower, upper = self.centered_edge_log_interval(level=level)
        return np.exp(lower), np.exp(upper)

    @beartype
    def to_edge_records(self) -> list[dict[str, Any]]:
        """Return JSON-friendly per-edge prediction records."""

        records = _edge_records(
            edge_log_mean=self.centered_edge_log_mean,
            edge_log_sd=self.edge_log_sd,
            edges=self.edges,
            positions=self.positions,
            surface_weight=self.surface_weight,
        )
        refined = self.centered_refined_edge_log_mean
        refined_relative = None if refined is None else np.exp(refined)
        include_raw_sd = self.raw_edge_log_sd is not None and not np.array_equal(
            self.raw_edge_log_sd, self.edge_log_sd
        )
        for index, record in enumerate(records):
            if include_raw_sd:
                record["edge_log_sd_raw"] = float(self.raw_edge_log_sd[index])
            if self.refinement_applied:
                assert refined is not None and refined_relative is not None
                record["refined_edge_log_mean"] = float(refined[index])
                record["refined_relative_migration_mean"] = float(refined_relative[index])
            if self.name is not None:
                record["name"] = self.name
        return records


@beartype
@dataclass(frozen=True)
class DREEMSPredictor:
    """Loaded trained DREEMS model ready for inference."""

    checkpoint: checkpoints.DREEMSCheckpoint

    @classmethod
    def from_checkpoint(cls, path: Path | str | None = None) -> DREEMSPredictor:
        """Load once for repeated predictions; omit the path to use released weights."""

        return cls(checkpoints.load_checkpoint(path))

    def predict(
        self,
        example: InferenceLike,
        *,
        refinement_config: refinement.DREEMSMapRefinementConfig | bool | None = None,
    ) -> DREEMSResult:
        """Predict one example."""

        return self.predict_many([example], refinement_config=refinement_config)[0]

    def predict_many(
        self,
        examples: Sequence[InferenceLike],
        *,
        refinement_config: refinement.DREEMSMapRefinementConfig | bool | None = None,
    ) -> list[DREEMSResult]:
        """Predict examples in input order, padding the batch to its largest graph.

        Group similarly sized graphs to limit padding memory. Refinement is
        off by default here, unlike the higher-level ``dreems.infer`` API.
        When enabled, it adds a point estimate without replacing neural SDs.
        """

        inference_examples = [coerce_inference_example(example) for example in examples]
        if not inference_examples:
            raise ValueError("predict_many requires at least one example")
        raw_examples = [example.to_raw_example() for example in inference_examples]
        prediction = checkpoints.predict_examples(self.checkpoint, raw_examples)
        results = [
            DREEMSResult(
                edge_log_mean=mean,
                edge_log_sd=sd,
                raw_edge_log_sd=raw_sd,
                surface_weight=surface_weight,
                edges=example.edges,
                positions=example.positions,
                node_log_q_mean=q_mean,
                node_log_q_sd=q_sd,
                name=example.name,
                checkpoint_path=self.checkpoint.path,
                metadata=example.metadata,
            )
            for example, mean, sd, raw_sd, q_mean, q_sd, surface_weight in zip(
                inference_examples,
                prediction.edge_mean,
                prediction.edge_sd,
                prediction.raw_edge_sd,
                prediction.node_log_q_mean,
                prediction.node_log_q_sd,
                prediction.surface_weight,
                strict=True,
            )
        ]
        config = normalize_refinement_config(refinement_config)
        if config is None:
            return results
        return [
            refine_prediction_result(result=result, example=example, config=config)
            for result, example in zip(results, inference_examples, strict=True)
        ]


@beartype
def normalize_refinement_config(
    config: refinement.DREEMSMapRefinementConfig | bool | None,
) -> refinement.DREEMSMapRefinementConfig | None:
    """Resolve the public boolean shorthand without changing supplied settings."""

    if config is None or config is False:
        return None
    if config is True:
        return refinement.DREEMSMapRefinementConfig()
    return config


@jaxtyped(typechecker=beartype)
def genotype_covariance(
    genotypes: Shaped[np.ndarray, "samples snps"],
) -> tuple[Float[np.ndarray, "samples samples"], int]:
    """Standardize complete diploid dosages and return sample covariance.

    Monomorphic sites are removed, but no MAF, missing-call, or LD filtering
    is performed here. Use ``from_genotype_dataset`` for the call-level QC
    convenience interface, or apply those filters before supplying dosages.
    """

    standardized = features.standardized_polymorphic_genotypes(
        _real_array(genotypes, name="genotypes")
    )
    covariance = (standardized @ standardized.T) / float(standardized.shape[1])
    return covariance.astype(np.float64, copy=False), int(standardized.shape[1])


@beartype
def refinement_observation(
    example: DREEMSInput | NodeCovarianceInput | InferenceExample,
) -> _RefinementObservation:
    """Put all three input representations into the node-covariance refinement model."""

    if isinstance(example, NodeCovarianceInput):
        return _RefinementObservation(
            node_covariance=example.node_covariance,
            observed_nodes=example.observed_nodes,
            observed_sample_count=example.node_sample_count[example.observed_nodes],
            effective_snps=int(example.effective_snps),
        )
    if isinstance(example, DREEMSInput):
        sample_nodes = features.nearest_node_assignments(
            example.sample_coordinates,
            example.node_coordinates,
        )
        effective_snps = int(example.effective_snps or max(example.sample_covariance.shape[0], 2))
        sample_covariance = example.sample_covariance
    else:
        sample_covariance, effective_snps = genotype_covariance(example.genotypes)
        sample_nodes = example.sample_nodes
    covariance, observed_nodes, observed_sample_count = refinement.empirical_observed_covariance(
        sample_covariance=sample_covariance,
        sample_nodes=sample_nodes,
        num_nodes=example.positions.shape[0],
        effective_snps=effective_snps,
    )
    return _RefinementObservation(
        node_covariance=covariance,
        observed_nodes=observed_nodes,
        observed_sample_count=observed_sample_count,
        effective_snps=effective_snps,
    )


@beartype
def covariance_fit_metadata(fit: refinement.CovarianceFit) -> dict[str, float]:
    """Report covariance fit diagnostics as ordinary Python numbers."""
    return {
        "corr": float(fit.corr),
        "r2": float(fit.r2),
        "mse": float(fit.mse),
        "alpha": float(fit.alpha),
        "q_scale": float(fit.q_scale),
    }


@beartype
def refinement_metadata(
    result: refinement.DREEMSMapRefinementResult,
) -> dict[str, Any]:
    """Keep before/after fit statistics and the refinement optimization history."""
    return {
        "initial_fit": covariance_fit_metadata(result.initial_fit),
        "refined_fit": covariance_fit_metadata(result.refined_fit),
        "diagnostics": dict(result.diagnostics),
        "history": list(result.history),
    }


@beartype
def refine_prediction_result(
    *,
    result: DREEMSResult,
    example: DREEMSInput | NodeCovarianceInput | InferenceExample,
    config: refinement.DREEMSMapRefinementConfig,
) -> DREEMSResult:
    """Add a MAP point estimate while preserving neural means and uncertainty."""

    observation = refinement_observation(example)
    refined = refinement.refine_edge_log_conductance(
        node_covariance=observation.node_covariance,
        observed_nodes=observation.observed_nodes,
        observed_sample_count=observation.observed_sample_count,
        node_coordinates=result.positions,
        edges=result.edges,
        initial_edge_logs=result.edge_log_mean,
        # The MAP trust radius was selected against raw model SDs. Keep using
        # those SDs even when reported intervals have been recalibrated.
        initial_edge_sd=(
            result.edge_log_sd if result.raw_edge_log_sd is None else result.raw_edge_log_sd
        ),
        initial_node_log_q=result.node_log_q_mean,
        effective_snps=int(observation.effective_snps),
        config=config,
    )
    metadata = dict(result.metadata)
    metadata["map_refinement"] = dict(refined.diagnostics)
    return replace(
        result,
        metadata=metadata,
        refined_edge_log_mean=refined.edge_log_conductance,
        refinement_diagnostics=refinement_metadata(refined),
    )


@beartype
def load_model(path: Path | str | None = None) -> DREEMSPredictor:
    """Load a trained DREEMS checkpoint for inference.

    If ``path`` is omitted, the packaged diffusion-band checkpoint is loaded.
    """

    return DREEMSPredictor.from_checkpoint(path)


@beartype
def predict(
    checkpoint: DREEMSPredictor | checkpoints.DREEMSCheckpoint | Path | str | InferenceLike | None,
    example: InferenceLike | None = None,
    *,
    refinement_config: refinement.DREEMSMapRefinementConfig | bool | None = None,
) -> DREEMSResult:
    """Run ``predict(example)`` or ``predict(checkpoint, example)``.

    This array/mapping interface defaults to neural inference only. Pass
    ``refinement_config=True`` to add the default MAP refinement. For repeated
    calls, reuse a predictor returned by ``load_model``.
    """

    if example is None:
        example = checkpoint  # type: ignore[assignment]
        checkpoint = None
    if example is None:
        raise ValueError("predict requires an inference example")
    if isinstance(checkpoint, DREEMSPredictor):
        predictor = checkpoint
    elif isinstance(checkpoint, checkpoints.DREEMSCheckpoint):
        predictor = DREEMSPredictor(checkpoint)
    else:
        predictor = load_model(checkpoint)
    return predictor.predict(example, refinement_config=refinement_config)


@beartype
def coerce_inference_example(
    example: InferenceLike,
) -> DREEMSInput | NodeCovarianceInput | InferenceExample:
    """Accept the three input objects or their documented array mappings.

    Older workflow mappings use ``graph.edges`` and ``active_positions``. Those
    names remain accepted, but all paths use the same input-class validation.
    """

    if isinstance(example, DREEMSInput | NodeCovarianceInput | InferenceExample):
        return example
    if "graph" in example:
        graph = example["graph"]
        if not isinstance(graph, Mapping):
            raise ValueError("example['graph'] must be a mapping")
        edges = graph["edges"]
    else:
        edges = example["edges"]
    covariance_input = any(
        key in example for key in ("node_covariance", "sample_covariance", "covariance")
    )
    if not covariance_input and "genotypes" not in example:
        raise ValueError(
            "mapping input must contain node_covariance, sample_covariance, or genotypes"
        )
    coordinate_keys = (
        ("active_positions", "node_coordinates", "nodes")
        if covariance_input
        else ("active_positions", "positions", "node_coordinates")
    )
    coordinates = next((example[key] for key in coordinate_keys if key in example), None)
    if coordinates is None:
        raise ValueError(
            f"mapping input must provide graph coordinates as one of {coordinate_keys}"
        )
    common = {
        "edges": np.asarray(edges),
        "name": example.get("name") if isinstance(example.get("name"), str) else None,
        "metadata": inference_metadata(example),
    }
    if "node_covariance" in example:
        return NodeCovarianceInput(
            node_covariance=np.asarray(example["node_covariance"]),
            observed_nodes=np.asarray(example["observed_nodes"]),
            node_sample_count=np.asarray(example["node_sample_count"]),
            node_coordinates=np.asarray(coordinates),
            effective_snps=example["effective_snps"],
            **common,
        )
    if covariance_input:
        return DREEMSInput(
            sample_covariance=np.asarray(
                example["sample_covariance"]
                if "sample_covariance" in example
                else example["covariance"]
            ),
            sample_coordinates=np.asarray(example["sample_coordinates"]),
            node_coordinates=np.asarray(coordinates),
            effective_snps=example.get("effective_snps"),
            **common,
        )
    return InferenceExample(
        positions=np.asarray(coordinates),
        sample_nodes=np.asarray(example["sample_nodes"]),
        genotypes=np.asarray(example["genotypes"]),
        **common,
    )


@beartype
def inference_metadata(example: Mapping[str, Any]) -> dict[str, Any]:
    """Keep user annotations without copying data arrays or simulation truth into results."""
    reserved = {
        "active_positions",
        "positions",
        "node_coordinates",
        "nodes",
        "graph",
        "edges",
        "sample_coordinates",
        "sample_covariance",
        "covariance",
        "node_covariance",
        "observed_nodes",
        "node_sample_count",
        "effective_snps",
        "sample_nodes",
        "genotypes",
        "edge_log_rates",
        "node_log_q",
        "name",
    }
    return {key: value for key, value in example.items() if key not in reserved}


@beartype
def normal_interval_z(level: Real) -> float:
    """Two-sided standard normal quantile for a pointwise interval."""

    if not 0.0 < level < 1.0:
        raise ValueError("interval level must be between 0 and 1")
    return NormalDist().inv_cdf(0.5 + 0.5 * level)


__all__ = [
    "DREEMSInput",
    "DREEMSPredictor",
    "DREEMSResult",
    "InferenceExample",
    "NodeCovarianceInput",
    "genotype_covariance",
    "load_model",
    "predict",
]
