"""xarray datasets for user-facing DREEMS inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from beartype import beartype
from numpy.typing import ArrayLike

from dreems import checkpoints, refinement
from dreems.inference import (
    DREEMSPredictor,
    DREEMSResult,
    NodeCovarianceInput,
    predict,
)
from dreems.validation import _integer_array
from dreems.workflow_io import json_ready

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


@beartype
def create_dataset(
    *,
    node_positions: ArrayLike,
    edges: ArrayLike,
    node_covariance: ArrayLike,
    observed_nodes: ArrayLike | None = None,
    node_sample_count: ArrayLike,
    effective_snps: Any,
    coordinate_names: Sequence[str] = ("x", "y"),
    attrs: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """Create the canonical DREEMS input dataset from numpy arrays.

    ``node_covariance`` is ordered by ``observed_nodes`` rather than all graph
    nodes. This keeps sparse datasets compact while ``node_sample_count``
    records sampling precision on the complete graph.

    Counts are individuals, not allele copies. Coordinates must share one
    Cartesian system. The covariance must use standardized SNPs, as in
    ``from_genotype_dataset``; an unscaled dosage covariance is not equivalent.

    Array-like inputs are converted by ``NodeCovarianceInput``. The SNP count
    is validated there too, accepting scalar NumPy integers and rejecting
    fractional, boolean, or nonfinite values before conversion.
    """

    axes = tuple(str(name) for name in coordinate_names)
    if len(axes) != 2 or len(set(axes)) != 2:
        raise ValueError("coordinate_names must contain two distinct names")
    sample_count = _integer_array(
        np.asarray(node_sample_count), name="node_sample_count", dtype=np.int64
    )
    if observed_nodes is None:
        observed_nodes = np.flatnonzero(sample_count > 0)
    example = NodeCovarianceInput(
        node_covariance=node_covariance,
        observed_nodes=observed_nodes,
        node_sample_count=sample_count,
        node_coordinates=node_positions,
        edges=edges,
        effective_snps=effective_snps,
    )
    positions = example.positions
    edge_nodes = example.edges
    observed = example.observed_nodes

    dataset_attrs = {
        "dreems_schema": SCHEMA_VERSION,
        "coordinate_reference_system": "user supplied Cartesian coordinates",
    }
    if attrs is not None:
        dataset_attrs.update(dict(attrs))
    dataset_attrs["dreems_schema"] = SCHEMA_VERSION
    return xr.Dataset(
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
                example.node_covariance,
                {"long_name": "covariance of SNP-standardized observed-node dosages"},
            ),
            NODE_SAMPLE_COUNT: (
                (NODE_DIM,),
                sample_count,
                {"long_name": "number of samples assigned to each graph node"},
            ),
            EFFECTIVE_SNPS: (
                (),
                np.asarray(example.effective_snps, dtype=np.int64),
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


@beartype
def validate_dataset(dataset: xr.Dataset) -> None:
    """Check xarray dimensions and the same numeric constraints as NumPy inputs."""

    dataset_to_input(dataset)


@beartype
def dataset_to_input(dataset: xr.Dataset) -> NodeCovarianceInput:
    """Check the xarray schema and construct its validated inference input once."""

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

    for coordinate in (OBSERVED_NODE_DIM, OBSERVED_NODE_2_DIM):
        if coordinate not in dataset.coords or dataset.coords[coordinate].dims != (coordinate,):
            raise ValueError(
                f"{coordinate} must be an explicit coordinate indexing covariance rows"
            )
    observed = dataset.coords[OBSERVED_NODE_DIM].values
    observed_2 = dataset.coords[OBSERVED_NODE_2_DIM].values
    if not np.array_equal(observed, observed_2):
        raise ValueError("observed_nodes and observed_nodes_2 coordinates must match")

    metadata = {
        key: value for key, value in dataset.attrs.items() if key not in {"name", "dreems_schema"}
    }
    return NodeCovarianceInput(
        node_covariance=dataset[NODE_COVARIANCE].values,
        observed_nodes=observed,
        node_sample_count=dataset[NODE_SAMPLE_COUNT].values,
        node_coordinates=dataset[NODE_POSITION].values,
        edges=dataset[EDGE_NODE].values,
        effective_snps=dataset[EFFECTIVE_SNPS].item(),
        name=str(dataset.attrs["name"]) if "name" in dataset.attrs else None,
        metadata=metadata,
    )


@beartype
def infer(
    dataset: xr.Dataset,
    *,
    checkpoint: DREEMSPredictor | checkpoints.DREEMSCheckpoint | Path | str | None = None,
    refine: bool | refinement.DREEMSMapRefinementConfig = True,
) -> xr.Dataset:
    """Infer a migration surface, preserving input variables and replacing old fits.

    The packaged model is used unless ``checkpoint`` is supplied. Refinement is
    enabled by default here; ``refine=False`` returns only the neural estimate.
    Neural uncertainty is reported separately from the refined point estimate.
    """

    example = dataset_to_input(dataset)
    result = predict(
        checkpoint,
        example,
        refinement_config=refine,
    )
    return result_to_dataset(dataset, result)


@beartype
def result_to_dataset(dataset: xr.Dataset, result: DREEMSResult) -> xr.Dataset:
    """Replace model outputs without changing the source data or graph order."""

    if not np.array_equal(dataset[EDGE_NODE].values, result.edges) or not np.array_equal(
        dataset[NODE_POSITION].values, result.positions
    ):
        raise ValueError("result graph and node positions must match the dataset in the same order")
    # A second call may disable refinement or calibration. Do not retain outputs
    # from the first fit that the new result does not supply.
    output = dataset.drop_vars(
        [
            "edge_log_migration_mean",
            "edge_log_migration_sd",
            "edge_log_migration_sd_raw",
            "edge_relative_migration_mean",
            "edge_log_migration_refined",
            "edge_relative_migration_refined",
            "node_log_q_mean",
            "node_log_q_sd",
        ],
        errors="ignore",
    ).copy(deep=False)
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
            "long_name": "relative migration from the neural posterior mean log rate",
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
    output.attrs.pop("dreems_refinement", None)
    output.attrs.update(
        {
            "dreems_schema": SCHEMA_VERSION,
            "dreems_checkpoint": (
                "packaged default"
                if result.checkpoint_path is None
                else str(result.checkpoint_path)
            ),
            "dreems_refinement_applied": bool(result.refinement_applied),
            "dreems_surface_weight": float(result.surface_weight),
        }
    )
    if result.refinement_diagnostics is not None:
        output.attrs["dreems_refinement"] = json_ready(result.refinement_diagnostics)
    return output


__all__ = [
    "create_dataset",
    "dataset_to_input",
    "infer",
    "result_to_dataset",
    "validate_dataset",
]
