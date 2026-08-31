"""User-facing inference API for trained DREEMS checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from dreems import checkpoints, features, metrics, refinement


@dataclass(frozen=True)
class DREEMSInput:
    """Covariance-first input for DREEMS inference.

    Parameters
    ----------
    sample_covariance
        Sample-level covariance matrix with shape ``(samples, samples)``.
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
        If omitted, DREEMS uses a conservative default based on the number of
        samples.
    name
        Optional label propagated to prediction results.
    metadata
        Optional user metadata propagated to prediction results.
    """

    sample_covariance: np.ndarray
    sample_coordinates: np.ndarray
    node_coordinates: np.ndarray
    edges: np.ndarray
    effective_snps: int | None = None
    name: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        sample_covariance = np.asarray(self.sample_covariance, dtype=np.float64)
        sample_coordinates = np.asarray(self.sample_coordinates, dtype=np.float64)
        node_coordinates = np.asarray(self.node_coordinates, dtype=np.float64)
        edges = np.asarray(self.edges, dtype=np.int32)
        metadata = {} if self.metadata is None else dict(self.metadata)

        validate_graph_arrays(positions=node_coordinates, edges=edges)
        validate_covariance_arrays(
            sample_covariance=sample_covariance,
            sample_coordinates=sample_coordinates,
        )
        if self.effective_snps is not None and self.effective_snps < 2:
            raise ValueError("effective_snps must be at least 2")

        object.__setattr__(self, "sample_covariance", sample_covariance)
        object.__setattr__(self, "sample_coordinates", sample_coordinates)
        object.__setattr__(self, "node_coordinates", node_coordinates)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "metadata", metadata)

    @property
    def positions(self) -> np.ndarray:
        """Graph node coordinates, kept as ``positions`` for result construction."""

        return self.node_coordinates

    def to_raw_example(self) -> dict[str, Any]:
        """Return the internal raw-example mapping consumed by feature builders."""

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
            payload["effective_snps"] = int(self.effective_snps)
        if self.name is not None:
            payload["name"] = self.name
        return payload


@dataclass(frozen=True)
class NodeCovarianceInput:
    """Node-level covariance input used by the xarray public API.

    ``node_covariance`` contains covariance among the observed graph nodes,
    whose graph indices are given by ``observed_nodes``. ``node_sample_count``
    records how many samples contributed to every graph node.
    """

    node_covariance: np.ndarray
    observed_nodes: np.ndarray
    node_sample_count: np.ndarray
    node_coordinates: np.ndarray
    edges: np.ndarray
    effective_snps: int
    name: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        node_covariance = np.asarray(self.node_covariance, dtype=np.float64)
        observed_nodes = np.asarray(self.observed_nodes, dtype=np.int32)
        raw_sample_count = np.asarray(self.node_sample_count)
        if not np.all(np.isfinite(raw_sample_count)):
            raise ValueError("node_sample_count must be finite")
        if np.any(raw_sample_count < 0) or not np.allclose(
            raw_sample_count,
            np.round(raw_sample_count),
        ):
            raise ValueError("node_sample_count must contain nonnegative integers")
        node_sample_count = np.asarray(np.round(raw_sample_count), dtype=np.int64)
        node_coordinates = np.asarray(self.node_coordinates, dtype=np.float64)
        edges = np.asarray(self.edges, dtype=np.int32)
        metadata = {} if self.metadata is None else dict(self.metadata)

        validate_graph_arrays(positions=node_coordinates, edges=edges)
        validate_node_covariance_arrays(
            node_covariance=node_covariance,
            observed_nodes=observed_nodes,
            node_sample_count=node_sample_count,
            num_nodes=node_coordinates.shape[0],
        )
        if self.effective_snps < 2:
            raise ValueError("effective_snps must be at least 2")

        object.__setattr__(self, "node_covariance", node_covariance)
        object.__setattr__(self, "observed_nodes", observed_nodes)
        object.__setattr__(self, "node_sample_count", node_sample_count)
        object.__setattr__(self, "node_coordinates", node_coordinates)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "effective_snps", int(self.effective_snps))
        object.__setattr__(self, "metadata", metadata)

    @property
    def positions(self) -> np.ndarray:
        return self.node_coordinates

    def to_raw_example(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "graph": {
                "active_full": np.arange(self.node_coordinates.shape[0], dtype=np.int32),
                "edges": self.edges,
            },
            "active_positions": self.node_coordinates,
            "node_covariance": self.node_covariance,
            "observed_nodes": self.observed_nodes,
            "node_sample_count": self.node_sample_count,
            "effective_snps": int(self.effective_snps),
        }
        if self.name is not None:
            payload["name"] = self.name
        return payload


@dataclass(frozen=True)
class InferenceExample:
    """Genotype-based compatibility input for DREEMS inference.

    Prefer the xarray API in :mod:`dreems.datasets` for user-facing inference.
    This class is useful when tests or internal workflows still have genotypes
    and sample-node assignments rather than an already-computed covariance.
    """

    positions: np.ndarray
    edges: np.ndarray
    sample_nodes: np.ndarray
    genotypes: np.ndarray
    name: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=np.float64)
        edges = np.asarray(self.edges, dtype=np.int32)
        sample_nodes = np.asarray(self.sample_nodes, dtype=np.int32)
        genotypes = np.asarray(self.genotypes, dtype=np.float64)
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

    def to_raw_example(self) -> dict[str, Any]:
        """Return the internal raw-example mapping consumed by feature builders."""

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
    node_covariance: np.ndarray
    observed_nodes: np.ndarray
    observed_sample_count: np.ndarray
    effective_snps: int


@dataclass(frozen=True)
class DREEMSResult:
    """DREEMS prediction for one inference example."""

    edge_log_mean: np.ndarray
    edge_log_sd: np.ndarray
    edges: np.ndarray
    positions: np.ndarray
    raw_edge_log_sd: np.ndarray | None = None
    surface_weight: float = 1.0
    node_log_q_mean: np.ndarray | None = None
    node_log_q_sd: np.ndarray | None = None
    name: str | None = None
    checkpoint_path: Path | None = None
    metadata: Mapping[str, Any] | None = None
    refined_edge_log_mean: np.ndarray | None = None
    refinement_diagnostics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        edge_log_mean = np.asarray(self.edge_log_mean, dtype=np.float64)
        edge_log_sd = np.asarray(self.edge_log_sd, dtype=np.float64)
        raw_edge_log_sd = (
            None
            if self.raw_edge_log_sd is None
            else np.asarray(self.raw_edge_log_sd, dtype=np.float64)
        )
        edges = np.asarray(self.edges, dtype=np.int32)
        positions = np.asarray(self.positions, dtype=np.float64)
        node_log_q_mean = (
            None
            if self.node_log_q_mean is None
            else np.asarray(self.node_log_q_mean, dtype=np.float64)
        )
        node_log_q_sd = (
            None if self.node_log_q_sd is None else np.asarray(self.node_log_q_sd, dtype=np.float64)
        )
        metadata = {} if self.metadata is None else dict(self.metadata)
        refined_edge_log_mean = (
            None
            if self.refined_edge_log_mean is None
            else np.asarray(self.refined_edge_log_mean, dtype=np.float64)
        )
        refinement_diagnostics = (
            None if self.refinement_diagnostics is None else dict(self.refinement_diagnostics)
        )
        surface_weight = float(self.surface_weight)
        if edge_log_mean.ndim != 1:
            raise ValueError("edge_log_mean must be one-dimensional")
        if edge_log_sd.shape != edge_log_mean.shape:
            raise ValueError(
                f"edge_log_sd shape {edge_log_sd.shape} does not match "
                f"edge_log_mean shape {edge_log_mean.shape}"
            )
        if raw_edge_log_sd is not None and raw_edge_log_sd.shape != edge_log_mean.shape:
            raise ValueError(
                f"raw_edge_log_sd shape {raw_edge_log_sd.shape} does not match "
                f"edge_log_mean shape {edge_log_mean.shape}"
            )
        if refined_edge_log_mean is not None and refined_edge_log_mean.shape != edge_log_mean.shape:
            raise ValueError(
                f"refined_edge_log_mean shape {refined_edge_log_mean.shape} does not match "
                f"edge_log_mean shape {edge_log_mean.shape}"
            )
        if edges.shape != (edge_log_mean.shape[0], 2):
            raise ValueError(
                f"edges shape {edges.shape} does not match edge predictions {edge_log_mean.shape}"
            )
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must have shape (nodes, 2)")
        if node_log_q_mean is not None and node_log_q_mean.shape != (positions.shape[0],):
            raise ValueError("node_log_q_mean must have shape (nodes,)")
        if node_log_q_sd is not None and node_log_q_sd.shape != (positions.shape[0],):
            raise ValueError("node_log_q_sd must have shape (nodes,)")
        if (node_log_q_mean is None) != (node_log_q_sd is None):
            raise ValueError("node_log_q_mean and node_log_q_sd must be supplied together")
        if not np.isfinite(surface_weight) or not 0.0 <= surface_weight <= 1.0:
            raise ValueError("surface_weight must be finite and between zero and one")
        object.__setattr__(self, "edge_log_mean", edge_log_mean)
        object.__setattr__(self, "edge_log_sd", edge_log_sd)
        object.__setattr__(self, "raw_edge_log_sd", raw_edge_log_sd)
        object.__setattr__(self, "surface_weight", surface_weight)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "node_log_q_mean", node_log_q_mean)
        object.__setattr__(self, "node_log_q_sd", node_log_q_sd)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "refined_edge_log_mean", refined_edge_log_mean)
        object.__setattr__(self, "refinement_diagnostics", refinement_diagnostics)

    @property
    def centered_edge_log_mean(self) -> np.ndarray:
        """Predicted edge log-conductances after removing the arbitrary offset."""

        return metrics.centered(self.edge_log_mean)

    @property
    def centered_refined_edge_log_mean(self) -> np.ndarray | None:
        """MAP-refined edge log-conductances, when refinement was requested."""

        if self.refined_edge_log_mean is None:
            return None
        return metrics.centered(self.refined_edge_log_mean)

    @property
    def relative_migration_mean(self) -> np.ndarray:
        """Predicted relative migration, normalized to geometric mean one."""

        return np.exp(self.centered_edge_log_mean)

    @property
    def refined_relative_migration_mean(self) -> np.ndarray | None:
        """MAP-refined relative migration, when refinement was requested."""

        centered = self.centered_refined_edge_log_mean
        return None if centered is None else np.exp(centered)

    @property
    def refinement_applied(self) -> bool:
        """Whether a separate covariance MAP point estimate is available."""

        return self.refined_edge_log_mean is not None

    def centered_edge_log_interval(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Neural-posterior pointwise interval for centered edge log-conductances."""

        z = normal_interval_z(level)
        mean = self.centered_edge_log_mean
        sd = np.asarray(self.edge_log_sd, dtype=np.float64)
        return mean - z * sd, mean + z * sd

    def relative_migration_interval(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Pointwise interval for relative migration on the original scale."""

        lower, upper = self.centered_edge_log_interval(level=level)
        return np.exp(lower), np.exp(upper)

    def to_edge_records(self) -> list[dict[str, Any]]:
        """Return JSON-friendly per-edge prediction records."""

        relative = self.relative_migration_mean
        refined = self.centered_refined_edge_log_mean
        refined_relative = self.refined_relative_migration_mean
        records: list[dict[str, Any]] = []
        for index, (left, right) in enumerate(self.edges):
            left_index = int(left)
            right_index = int(right)
            record: dict[str, Any] = {
                "edge_index": index,
                "left_node": left_index,
                "right_node": right_index,
                "left_x": float(self.positions[left_index, 0]),
                "left_y": float(self.positions[left_index, 1]),
                "right_x": float(self.positions[right_index, 0]),
                "right_y": float(self.positions[right_index, 1]),
                "edge_log_mean": float(self.centered_edge_log_mean[index]),
                "edge_log_sd": float(self.edge_log_sd[index]),
                "relative_migration_mean": float(relative[index]),
                "surface_weight": self.surface_weight,
            }
            if self.raw_edge_log_sd is not None and not np.array_equal(
                self.raw_edge_log_sd,
                self.edge_log_sd,
            ):
                record["edge_log_sd_raw"] = float(self.raw_edge_log_sd[index])
            if self.refinement_applied:
                assert refined is not None and refined_relative is not None
                record["refined_edge_log_mean"] = float(refined[index])
                record["refined_relative_migration_mean"] = float(refined_relative[index])
            if self.name is not None:
                record["name"] = self.name
            records.append(record)
        return records


@dataclass(frozen=True)
class DREEMSPredictor:
    """Loaded trained DREEMS model ready for inference."""

    checkpoint: checkpoints.DREEMSCheckpoint

    @classmethod
    def from_checkpoint(cls, path: Path | str | None = None) -> DREEMSPredictor:
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
        """Predict one or more examples."""

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


def normalize_refinement_config(
    config: refinement.DREEMSMapRefinementConfig | bool | None,
) -> refinement.DREEMSMapRefinementConfig | None:
    if config is None or config is False:
        return None
    if config is True:
        return refinement.DREEMSMapRefinementConfig()
    return config


def genotype_covariance(genotypes: np.ndarray) -> tuple[np.ndarray, int]:
    """Return the SNP-standardized sample covariance used by DREEMS."""

    standardized = features.standardized_polymorphic_genotypes(
        np.asarray(genotypes, dtype=np.float64)
    )
    covariance = (standardized @ standardized.T) / float(standardized.shape[1])
    return covariance.astype(np.float64, copy=False), int(standardized.shape[1])


def refinement_observation(
    example: DREEMSInput | NodeCovarianceInput | InferenceExample,
) -> _RefinementObservation:
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
        covariance, observed_nodes, observed_sample_count = (
            refinement.empirical_observed_covariance(
                sample_covariance=example.sample_covariance,
                sample_nodes=sample_nodes,
                num_nodes=example.node_coordinates.shape[0],
                effective_snps=effective_snps,
            )
        )
        return _RefinementObservation(
            node_covariance=covariance,
            observed_nodes=observed_nodes,
            observed_sample_count=observed_sample_count,
            effective_snps=effective_snps,
        )
    sample_covariance, effective_snps = genotype_covariance(example.genotypes)
    covariance, observed_nodes, observed_sample_count = refinement.empirical_observed_covariance(
        sample_covariance=sample_covariance,
        sample_nodes=example.sample_nodes,
        num_nodes=example.positions.shape[0],
        effective_snps=effective_snps,
    )
    return _RefinementObservation(
        node_covariance=covariance,
        observed_nodes=observed_nodes,
        observed_sample_count=observed_sample_count,
        effective_snps=effective_snps,
    )


def covariance_fit_metadata(fit: refinement.CovarianceFit) -> dict[str, float]:
    return {
        "corr": float(fit.corr),
        "r2": float(fit.r2),
        "mse": float(fit.mse),
        "alpha": float(fit.alpha),
        "q_scale": float(fit.q_scale),
    }


def refinement_metadata(
    result: refinement.DREEMSMapRefinementResult,
) -> dict[str, Any]:
    return {
        "initial_fit": covariance_fit_metadata(result.initial_fit),
        "refined_fit": covariance_fit_metadata(result.refined_fit),
        "diagnostics": dict(result.diagnostics),
        "history": list(result.history),
    }


def refine_prediction_result(
    *,
    result: DREEMSResult,
    example: DREEMSInput | NodeCovarianceInput | InferenceExample,
    config: refinement.DREEMSMapRefinementConfig,
) -> DREEMSResult:
    observation = refinement_observation(example)
    refined = refinement.refine_edge_log_conductance(
        node_covariance=observation.node_covariance,
        observed_nodes=observation.observed_nodes,
        observed_sample_count=observation.observed_sample_count,
        node_coordinates=result.positions,
        edges=result.edges,
        initial_edge_logs=result.edge_log_mean,
        # Calibration changes reported intervals, not the frozen MAP point
        # estimator. Its trust radius was selected against raw model SDs.
        initial_edge_sd=(
            result.edge_log_sd if result.raw_edge_log_sd is None else result.raw_edge_log_sd
        ),
        initial_node_log_q=result.node_log_q_mean,
        effective_snps=int(observation.effective_snps),
        config=config,
    )
    metadata = dict(result.metadata)
    metadata["map_refinement"] = dict(refined.diagnostics)
    return DREEMSResult(
        edge_log_mean=result.edge_log_mean,
        edge_log_sd=result.edge_log_sd,
        raw_edge_log_sd=result.raw_edge_log_sd,
        surface_weight=result.surface_weight,
        edges=result.edges,
        positions=result.positions,
        node_log_q_mean=result.node_log_q_mean,
        node_log_q_sd=result.node_log_q_sd,
        name=result.name,
        checkpoint_path=result.checkpoint_path,
        metadata=metadata,
        refined_edge_log_mean=refined.edge_log_conductance,
        refinement_diagnostics=refinement_metadata(refined),
    )


def load_model(path: Path | str | None = None) -> DREEMSPredictor:
    """Load a trained DREEMS checkpoint for inference.

    If ``path`` is omitted, the packaged diffusion-band checkpoint is loaded.
    """

    return DREEMSPredictor.from_checkpoint(path)


def predict(
    checkpoint: DREEMSPredictor | checkpoints.DREEMSCheckpoint | Path | str | InferenceLike | None,
    example: InferenceLike | None = None,
    *,
    refinement_config: refinement.DREEMSMapRefinementConfig | bool | None = None,
) -> DREEMSResult:
    """Convenience function for one-off inference."""

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


def coerce_inference_example(
    example: InferenceLike,
) -> DREEMSInput | NodeCovarianceInput | InferenceExample:
    if isinstance(example, DREEMSInput | NodeCovarianceInput | InferenceExample):
        return example
    if not isinstance(example, Mapping):
        raise TypeError(
            "expected DREEMSInput, NodeCovarianceInput, InferenceExample, or mapping, "
            f"got {type(example).__name__}"
        )
    if "graph" in example:
        graph = example["graph"]
        if not isinstance(graph, Mapping):
            raise ValueError("example['graph'] must be a mapping")
        edges = graph["edges"]
    else:
        edges = example["edges"]
    if "node_covariance" in example:
        node_coordinates = (
            example["active_positions"]
            if "active_positions" in example
            else example["node_coordinates"]
            if "node_coordinates" in example
            else example["nodes"]
        )
        return NodeCovarianceInput(
            node_covariance=np.asarray(example["node_covariance"]),
            observed_nodes=np.asarray(example["observed_nodes"]),
            node_sample_count=np.asarray(example["node_sample_count"]),
            node_coordinates=np.asarray(node_coordinates),
            edges=np.asarray(edges),
            effective_snps=int(example["effective_snps"]),
            name=example.get("name") if isinstance(example.get("name"), str) else None,
            metadata=inference_metadata(example),
        )
    if "sample_covariance" in example or "covariance" in example:
        node_coordinates = (
            example["active_positions"]
            if "active_positions" in example
            else example["node_coordinates"]
            if "node_coordinates" in example
            else example["nodes"]
        )
        return DREEMSInput(
            sample_covariance=np.asarray(
                example["sample_covariance"]
                if "sample_covariance" in example
                else example["covariance"]
            ),
            sample_coordinates=np.asarray(example["sample_coordinates"]),
            node_coordinates=np.asarray(node_coordinates),
            edges=np.asarray(edges),
            effective_snps=(
                int(example["effective_snps"])
                if example.get("effective_snps") is not None
                else None
            ),
            name=example.get("name") if isinstance(example.get("name"), str) else None,
            metadata=inference_metadata(example),
        )
    if "genotypes" in example:
        return InferenceExample(
            positions=np.asarray(
                example["active_positions"]
                if "active_positions" in example
                else example["positions"]
                if "positions" in example
                else example["node_coordinates"]
            ),
            edges=np.asarray(edges),
            sample_nodes=np.asarray(example["sample_nodes"]),
            genotypes=np.asarray(example["genotypes"]),
            name=example.get("name") if isinstance(example.get("name"), str) else None,
            metadata=inference_metadata(example),
        )
    raise ValueError("mapping input must contain node_covariance, sample_covariance, or genotypes")


def inference_metadata(example: Mapping[str, Any]) -> dict[str, Any]:
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


def validate_graph_arrays(*, positions: np.ndarray, edges: np.ndarray) -> None:
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("node coordinates must have shape (nodes, 2)")
    if positions.shape[0] < 2:
        raise ValueError("node coordinates must contain at least two nodes")
    if not np.all(np.isfinite(positions)):
        raise ValueError("node coordinates must be finite")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (edges, 2)")
    if edges.shape[0] < 1:
        raise ValueError("edges must contain at least one edge")
    if np.any(edges < 0) or np.any(edges >= positions.shape[0]):
        raise ValueError("edge endpoints must refer to rows in node coordinates")
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("self edges are not supported")


def validate_covariance_arrays(
    *,
    sample_covariance: np.ndarray,
    sample_coordinates: np.ndarray,
) -> None:
    if sample_covariance.ndim != 2 or sample_covariance.shape[0] != sample_covariance.shape[1]:
        raise ValueError("sample_covariance must have shape (samples, samples)")
    if sample_covariance.shape[0] < 2:
        raise ValueError("sample_covariance must contain at least two samples")
    if not np.all(np.isfinite(sample_covariance)):
        raise ValueError("sample_covariance must be finite")
    if not np.allclose(sample_covariance, sample_covariance.T, rtol=1e-5, atol=1e-8):
        raise ValueError("sample_covariance must be symmetric")
    if sample_coordinates.ndim != 2 or sample_coordinates.shape[1] != 2:
        raise ValueError("sample_coordinates must have shape (samples, 2)")
    if sample_coordinates.shape[0] != sample_covariance.shape[0]:
        raise ValueError(
            f"sample_coordinates has {sample_coordinates.shape[0]} rows but "
            f"sample_covariance has {sample_covariance.shape[0]} samples"
        )
    if not np.all(np.isfinite(sample_coordinates)):
        raise ValueError("sample_coordinates must be finite")


def validate_node_covariance_arrays(
    *,
    node_covariance: np.ndarray,
    observed_nodes: np.ndarray,
    node_sample_count: np.ndarray,
    num_nodes: int,
) -> None:
    if node_covariance.ndim != 2 or node_covariance.shape[0] != node_covariance.shape[1]:
        raise ValueError("node_covariance must have shape (observed_nodes, observed_nodes)")
    if node_covariance.shape[0] < 2:
        raise ValueError("node_covariance must contain at least two observed nodes")
    if not np.all(np.isfinite(node_covariance)):
        raise ValueError("node_covariance must be finite")
    if not np.allclose(node_covariance, node_covariance.T, rtol=1e-5, atol=1e-8):
        raise ValueError("node_covariance must be symmetric")
    if observed_nodes.ndim != 1 or observed_nodes.shape[0] != node_covariance.shape[0]:
        raise ValueError("observed_nodes must index the rows of node_covariance")
    if np.unique(observed_nodes).shape[0] != observed_nodes.shape[0]:
        raise ValueError("observed_nodes must be unique")
    if np.any(observed_nodes < 0) or np.any(observed_nodes >= num_nodes):
        raise ValueError("observed_nodes must refer to graph nodes")
    if node_sample_count.shape != (num_nodes,):
        raise ValueError("node_sample_count must have shape (nodes,)")
    if np.any(node_sample_count < 0):
        raise ValueError("node_sample_count must be nonnegative")
    counted_nodes = np.flatnonzero(node_sample_count > 0)
    if not np.array_equal(np.sort(observed_nodes), counted_nodes):
        raise ValueError("nodes with positive node_sample_count must exactly match observed_nodes")


def validate_genotype_arrays(
    *,
    positions: np.ndarray,
    edges: np.ndarray,
    sample_nodes: np.ndarray,
    genotypes: np.ndarray,
) -> None:
    validate_graph_arrays(positions=positions, edges=edges)
    if sample_nodes.ndim != 1:
        raise ValueError("sample_nodes must be one-dimensional")
    if genotypes.ndim != 2:
        raise ValueError("genotypes must have shape (samples, snps)")
    if genotypes.shape[0] != sample_nodes.shape[0]:
        raise ValueError(
            f"genotypes have {genotypes.shape[0]} samples but sample_nodes has "
            f"{sample_nodes.shape[0]} entries"
        )
    if genotypes.shape[1] < 1:
        raise ValueError("genotypes must contain at least one SNP")
    if np.any(sample_nodes < 0) or np.any(sample_nodes >= positions.shape[0]):
        raise ValueError("sample_nodes must refer to rows in positions")
    if np.unique(sample_nodes).shape[0] < 2:
        raise ValueError("DREEMS inference requires samples from at least two graph nodes")
    if not np.all(np.isfinite(genotypes)):
        raise ValueError("genotypes must be finite")
    allele_sum = genotypes.sum(axis=0)
    polymorphic = (allele_sum > 0.0) & (allele_sum < 2.0 * genotypes.shape[0])
    if not np.any(polymorphic):
        raise ValueError("genotypes contain no polymorphic SNPs")


def normal_interval_z(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError("interval level must be between 0 and 1")
    return float(NormalDist().inv_cdf(0.5 + 0.5 * level))


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
