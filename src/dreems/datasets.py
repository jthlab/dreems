"""xarray datasets for user-facing DREEMS inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from dreems import checkpoints, refinement
from dreems.inference import (
    DREEMSPredictor,
    DREEMSResult,
    NodeCovarianceInput,
    predict,
)

SCHEMA_VERSION = "1.0"
NODE_DIM = "nodes"
EDGE_DIM = "edges"
EDGE_ENDPOINT_DIM = "edge_endpoints"
COORDINATE_DIM = "coordinate_axes"
OBSERVED_NODE_DIM = "observed_nodes"
OBSERVED_NODE_2_DIM = "observed_nodes_2"

NODE_POSITION = "node_position"
EDGE_NODE = "edge_node"
NODE_COVARIANCE = "node_covariance"
NODE_SAMPLE_COUNT = "node_sample_count"
EFFECTIVE_SNPS = "effective_snps"


def create_dataset(
    *,
    node_positions: np.ndarray,
    edges: np.ndarray,
    node_covariance: np.ndarray,
    observed_nodes: np.ndarray | None = None,
    node_sample_count: np.ndarray,
    effective_snps: int,
    coordinate_names: Sequence[str] = ("x", "y"),
    attrs: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """Create the canonical DREEMS input dataset from numpy arrays.

    ``node_covariance`` is ordered by ``observed_nodes`` rather than all graph
    nodes. This keeps sparse datasets compact while ``node_sample_count``
    records sampling precision on the complete graph.
    """

    positions = np.asarray(node_positions, dtype=np.float64)
    edge_nodes = np.asarray(edges, dtype=np.int32)
    covariance = np.asarray(node_covariance, dtype=np.float64)
    sample_count = np.asarray(node_sample_count)
    axes = tuple(str(name) for name in coordinate_names)
    if len(axes) != 2 or len(set(axes)) != 2:
        raise ValueError("coordinate_names must contain two distinct names")
    if sample_count.shape != (positions.shape[0],):
        raise ValueError("node_sample_count must have shape (nodes,)")
    if not np.all(np.isfinite(sample_count)):
        raise ValueError("node_sample_count must be finite")
    if np.any(sample_count < 0) or not np.allclose(sample_count, np.round(sample_count)):
        raise ValueError("node_sample_count must contain nonnegative integers")
    sample_count = np.asarray(np.round(sample_count), dtype=np.int64)
    if observed_nodes is None:
        positive_nodes = np.flatnonzero(sample_count > 0).astype(np.int32)
        if covariance.shape == (positive_nodes.size, positive_nodes.size):
            observed = positive_nodes
        elif covariance.shape == (positions.shape[0], positions.shape[0]):
            observed = np.arange(positions.shape[0], dtype=np.int32)
        else:
            raise ValueError(
                "observed_nodes cannot be inferred: covariance shape does not match "
                "positive node_sample_count entries or all graph nodes"
            )
    else:
        observed = np.asarray(observed_nodes, dtype=np.int32)

    dataset_attrs = {
        "dreems_schema": SCHEMA_VERSION,
        "coordinate_reference_system": "user supplied Cartesian coordinates",
    }
    if attrs is not None:
        dataset_attrs.update(dict(attrs))
    dataset_attrs["dreems_schema"] = SCHEMA_VERSION
    dataset = xr.Dataset(
        data_vars={
            NODE_POSITION: (
                (NODE_DIM, COORDINATE_DIM),
                positions,
                {"long_name": "graph node position"},
            ),
            EDGE_NODE: (
                (EDGE_DIM, EDGE_ENDPOINT_DIM),
                edge_nodes,
                {"long_name": "undirected graph edge endpoints"},
            ),
            NODE_COVARIANCE: (
                (OBSERVED_NODE_DIM, OBSERVED_NODE_2_DIM),
                covariance,
                {"long_name": "covariance of SNP-standardized observed-node dosages"},
            ),
            NODE_SAMPLE_COUNT: (
                (NODE_DIM,),
                sample_count,
                {"long_name": "number of samples assigned to each graph node"},
            ),
            EFFECTIVE_SNPS: (
                (),
                np.asarray(int(effective_snps), dtype=np.int64),
                {"long_name": "effective number of independent SNPs"},
            ),
        },
        coords={
            NODE_DIM: np.arange(positions.shape[0], dtype=np.int32),
            EDGE_DIM: np.arange(edge_nodes.shape[0], dtype=np.int32),
            EDGE_ENDPOINT_DIM: np.asarray(("source", "target"), dtype=str),
            COORDINATE_DIM: np.asarray(axes, dtype=str),
            OBSERVED_NODE_DIM: observed,
            OBSERVED_NODE_2_DIM: observed,
        },
        attrs=dataset_attrs,
    )
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: xr.Dataset) -> None:
    """Validate the canonical DREEMS dataset schema."""

    if not isinstance(dataset, xr.Dataset):
        raise TypeError(f"expected xarray.Dataset, got {type(dataset).__name__}")
    required = {
        NODE_POSITION,
        EDGE_NODE,
        NODE_COVARIANCE,
        NODE_SAMPLE_COUNT,
        EFFECTIVE_SNPS,
    }
    missing = sorted(required.difference(dataset.variables))
    if missing:
        raise ValueError(f"DREEMS dataset is missing variables: {', '.join(missing)}")
    expected_dims = {
        NODE_POSITION: (NODE_DIM, COORDINATE_DIM),
        EDGE_NODE: (EDGE_DIM, EDGE_ENDPOINT_DIM),
        NODE_COVARIANCE: (OBSERVED_NODE_DIM, OBSERVED_NODE_2_DIM),
        NODE_SAMPLE_COUNT: (NODE_DIM,),
        EFFECTIVE_SNPS: (),
    }
    for name, dims in expected_dims.items():
        if dataset[name].dims != dims:
            raise ValueError(f"{name} must have dimensions {dims}, got {dataset[name].dims}")

    positions = np.asarray(dataset[NODE_POSITION].values, dtype=np.float64)
    edges = np.asarray(dataset[EDGE_NODE].values, dtype=np.int32)
    covariance = np.asarray(dataset[NODE_COVARIANCE].values, dtype=np.float64)
    observed = np.asarray(dataset.coords[OBSERVED_NODE_DIM].values, dtype=np.int32)
    observed_2 = np.asarray(dataset.coords[OBSERVED_NODE_2_DIM].values, dtype=np.int32)
    raw_count = np.asarray(dataset[NODE_SAMPLE_COUNT].values)
    if not np.all(np.isfinite(raw_count)):
        raise ValueError("node_sample_count must be finite")
    if np.any(raw_count < 0) or not np.allclose(raw_count, np.round(raw_count)):
        raise ValueError("node_sample_count must contain nonnegative integers")
    sample_count = np.asarray(np.round(raw_count), dtype=np.int64)
    effective_snps = int(np.asarray(dataset[EFFECTIVE_SNPS].values).item())

    if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] < 2:
        raise ValueError("node_position must have shape (nodes, 2)")
    if not np.all(np.isfinite(positions)):
        raise ValueError("node_position must be finite")
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.shape[0] < 1:
        raise ValueError("edge_node must have shape (edges, 2)")
    if np.any(edges < 0) or np.any(edges >= positions.shape[0]):
        raise ValueError("edge_node endpoints must refer to graph nodes")
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("self edges are not supported")
    canonical_edges = np.sort(edges, axis=1)
    if np.unique(canonical_edges, axis=0).shape[0] != edges.shape[0]:
        raise ValueError("edge_node must not contain duplicate undirected edges")
    _validate_connected_graph(num_nodes=positions.shape[0], edges=edges)
    if not np.array_equal(observed, observed_2):
        raise ValueError("observed_nodes and observed_nodes_2 coordinates must match")

    NodeCovarianceInput(
        node_covariance=covariance,
        observed_nodes=observed,
        node_sample_count=sample_count,
        node_coordinates=positions,
        edges=edges,
        effective_snps=effective_snps,
    )
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -1.0e-7 * scale:
        raise ValueError("node_covariance must be positive semidefinite")


def infer(
    dataset: xr.Dataset,
    *,
    checkpoint: DREEMSPredictor | checkpoints.DREEMSCheckpoint | Path | str | None = None,
    refine: bool | refinement.DREEMSMapRefinementConfig = True,
) -> xr.Dataset:
    """Infer a migration surface and append predictions to an xarray dataset."""

    example = dataset_to_input(dataset)
    refinement_config: refinement.DREEMSMapRefinementConfig | bool | None
    refinement_config = refine if refine is not False else None
    result = predict(
        checkpoint,
        example,
        refinement_config=refinement_config,
    )
    return result_to_dataset(dataset, result)


def dataset_to_input(dataset: xr.Dataset) -> NodeCovarianceInput:
    """Convert a validated xarray dataset to the low-level inference input."""

    validate_dataset(dataset)
    metadata = {
        key: value for key, value in dataset.attrs.items() if key not in {"name", "dreems_schema"}
    }
    return NodeCovarianceInput(
        node_covariance=np.asarray(dataset[NODE_COVARIANCE].values, dtype=np.float64),
        observed_nodes=np.asarray(dataset.coords[OBSERVED_NODE_DIM].values, dtype=np.int32),
        node_sample_count=np.asarray(dataset[NODE_SAMPLE_COUNT].values, dtype=np.int64),
        node_coordinates=np.asarray(dataset[NODE_POSITION].values, dtype=np.float64),
        edges=np.asarray(dataset[EDGE_NODE].values, dtype=np.int32),
        effective_snps=int(np.asarray(dataset[EFFECTIVE_SNPS].values).item()),
        name=str(dataset.attrs["name"]) if "name" in dataset.attrs else None,
        metadata=metadata,
    )


def result_to_dataset(dataset: xr.Dataset, result: DREEMSResult) -> xr.Dataset:
    """Append one DREEMS result to its source dataset."""

    output = dataset.copy(deep=False)
    centered_mean = result.centered_edge_log_mean
    output["edge_log_migration_mean"] = (
        (EDGE_DIM,),
        centered_mean,
        {"long_name": "centered neural posterior mean edge log relative migration"},
    )
    output["edge_log_migration_sd"] = (
        (EDGE_DIM,),
        np.asarray(result.edge_log_sd, dtype=np.float64),
        {"long_name": "posterior standard deviation of edge log migration"},
    )
    if result.raw_edge_log_sd is not None and not np.array_equal(
        result.raw_edge_log_sd,
        result.edge_log_sd,
    ):
        output["edge_log_migration_sd_raw"] = (
            (EDGE_DIM,),
            np.asarray(result.raw_edge_log_sd, dtype=np.float64),
            {"long_name": "uncalibrated neural posterior standard deviation"},
        )
    output["edge_relative_migration_mean"] = (
        (EDGE_DIM,),
        np.exp(centered_mean),
        {
            "long_name": "neural posterior mean relative migration",
            "normalization": "geometric mean equals one",
        },
    )
    refined = result.centered_refined_edge_log_mean
    if refined is not None:
        output["edge_log_migration_refined"] = (
            (EDGE_DIM,),
            refined,
            {"long_name": "centered covariance MAP-refined edge log relative migration"},
        )
        output["edge_relative_migration_refined"] = (
            (EDGE_DIM,),
            np.exp(refined),
            {
                "long_name": "covariance MAP-refined relative migration",
                "normalization": "geometric mean equals one",
            },
        )
    if result.node_log_q_mean is not None:
        node_log_q = np.asarray(result.node_log_q_mean, dtype=np.float64)
        node_log_q = node_log_q - float(np.mean(node_log_q))
        output["node_log_q_mean"] = (
            (NODE_DIM,),
            node_log_q,
            {"long_name": "centered posterior mean relative node log coalescence rate"},
        )
        output["node_log_q_sd"] = (
            (NODE_DIM,),
            np.asarray(result.node_log_q_sd, dtype=np.float64),
            {"long_name": "posterior standard deviation of node log coalescence rate"},
        )

    output.attrs = dict(output.attrs)
    output.attrs.update(
        {
            "dreems_schema": SCHEMA_VERSION,
            "dreems_checkpoint": (
                "packaged default"
                if result.checkpoint_path is None
                else str(result.checkpoint_path)
            ),
            "dreems_refinement_applied": bool(result.refinement_applied),
        }
    )
    if result.refinement_diagnostics is not None:
        output.attrs["dreems_refinement"] = _json_safe(result.refinement_diagnostics)
    return output


def _validate_connected_graph(*, num_nodes: int, edges: np.ndarray) -> None:
    adjacency: list[list[int]] = [[] for _ in range(int(num_nodes))]
    for left, right in edges:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    seen = np.zeros(int(num_nodes), dtype=bool)
    stack = [0]
    seen[0] = True
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if seen[neighbor]:
                continue
            seen[neighbor] = True
            stack.append(neighbor)
    if not np.all(seen):
        raise ValueError("edge_node must define a connected graph")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "create_dataset",
    "dataset_to_input",
    "infer",
    "result_to_dataset",
    "validate_dataset",
]
