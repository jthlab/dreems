"""Array validation shared by inference inputs and benchmark workers."""

from __future__ import annotations

from typing import Any

import numpy as np
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components


@jaxtyped(typechecker=beartype)
def _real_array(values: Any, *, name: str) -> Float[np.ndarray, "*shape"]:  # noqa: F722
    """Convert real numeric inputs without silently discarding complex components."""
    array = np.asarray(values)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numbers")
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


@jaxtyped(typechecker=beartype)
def _integer_array(
    values: np.ndarray,
    *,
    name: str,
    dtype: type[np.signedinteger],
) -> Int[np.ndarray, "*shape"]:  # noqa: F722
    """Accept integer-valued numeric arrays without rounding or overflowing.

    CSV readers produce floats even for integer columns, so integral floats are
    valid. Bounds are compared as Python scalars to avoid rounding int64 limits
    to float64 before the comparison.
    """

    if values.dtype.kind not in "iuf" or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite integers")
    if values.dtype.kind == "f" and np.any(values != np.floor(values)):
        raise ValueError(f"{name} must contain integers, not fractional values")
    limits = np.iinfo(dtype)
    if values.size and (values.min().item() < limits.min or values.max().item() > limits.max):
        raise ValueError(f"{name} exceeds the range of {np.dtype(dtype).name}")
    return values.astype(dtype, copy=False)


@beartype
def _effective_snp_count(value: Any) -> int:
    """Validate the scalar precision count before converting it to an integer."""

    count = _integer_array(np.asarray(value), name="effective_snps", dtype=np.int64)
    if count.ndim != 0 or count.item() < 2:
        raise ValueError("effective_snps must be a scalar integer of at least 2")
    return count.item()


@beartype
def validate_graph_arrays(*, positions: np.ndarray, edges: np.ndarray) -> None:
    """Require finite coordinates and a simple, connected, undirected graph."""

    positions = _real_array(positions, name="node coordinates")
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("node coordinates must have shape (nodes, 2)")
    if positions.shape[0] < 2:
        raise ValueError("node coordinates must contain at least two nodes")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (edges, 2)")
    if edges.shape[0] < 1:
        raise ValueError("edges must contain at least one edge")
    edges = _integer_array(edges, name="edges", dtype=np.int32)
    if np.any(edges < 0) or np.any(edges >= positions.shape[0]):
        raise ValueError("edge endpoints must refer to rows in node coordinates")
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("self edges are not supported")
    if np.unique(np.sort(edges, axis=1), axis=0).shape[0] != edges.shape[0]:
        raise ValueError("edges must not contain duplicate undirected edges")
    adjacency = coo_array(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(positions.shape[0], positions.shape[0]),
    ).tocsr()
    if connected_components(adjacency, directed=False, return_labels=False) != 1:
        raise ValueError("edges must define a connected graph")


@beartype
def _validate_covariance(covariance: np.ndarray, *, name: str) -> None:
    """Check the shared matrix requirements for sample and node covariance."""

    covariance = _real_array(covariance, name=name)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if covariance.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two observations")
    if not np.allclose(covariance, covariance.T, rtol=1e-5, atol=1e-8):
        raise ValueError(f"{name} must be symmetric")


@beartype
def validate_covariance_arrays(
    *,
    sample_covariance: np.ndarray,
    sample_coordinates: np.ndarray,
) -> None:
    """Match covariance rows to finite sample coordinates in the same order."""

    _validate_covariance(sample_covariance, name="sample_covariance")
    sample_coordinates = _real_array(sample_coordinates, name="sample_coordinates")
    if sample_coordinates.ndim != 2 or sample_coordinates.shape[1] != 2:
        raise ValueError("sample_coordinates must have shape (samples, 2)")
    if sample_coordinates.shape[0] != sample_covariance.shape[0]:
        raise ValueError(
            f"sample_coordinates has {sample_coordinates.shape[0]} rows but "
            f"sample_covariance has {sample_covariance.shape[0]} samples"
        )


@beartype
def validate_node_covariance_arrays(
    *,
    node_covariance: np.ndarray,
    observed_nodes: np.ndarray,
    node_sample_count: np.ndarray,
    num_nodes: int,
) -> None:
    """Require one covariance row per sampled node and consistent sample counts."""

    _validate_covariance(node_covariance, name="node_covariance")
    observed_nodes = _integer_array(observed_nodes, name="observed_nodes", dtype=np.int32)
    node_sample_count = _integer_array(node_sample_count, name="node_sample_count", dtype=np.int64)
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
    eigenvalues = np.linalg.eigvalsh(0.5 * (node_covariance + node_covariance.T))
    scale = max(np.max(np.abs(eigenvalues)), 1.0)
    if np.min(eigenvalues) < -1.0e-7 * scale:
        raise ValueError("node_covariance must be positive semidefinite")


@beartype
def validate_diploid_dosages(genotypes: np.ndarray) -> None:
    """Require a complete sample-by-SNP matrix of diploid dosages.

    Fractional values from imputation are valid. Missing-call sentinels, NaNs,
    and counts outside [0, 2] are not; use the genotype-dataset QC path first.
    """
    genotypes = _real_array(genotypes, name="genotypes")
    if genotypes.ndim != 2 or genotypes.shape[0] < 2 or genotypes.shape[1] < 1:
        raise ValueError("genotypes must have shape (samples, snps), with >=2 samples and >=1 SNP")
    if np.any(genotypes < 0) or np.any(genotypes > 2):
        raise ValueError("diploid genotypes must contain dosages in [0, 2]")


@beartype
def validate_genotype_arrays(
    *,
    positions: np.ndarray,
    edges: np.ndarray,
    sample_nodes: np.ndarray,
    genotypes: np.ndarray,
) -> None:
    """Validate complete diploid dosages and their graph-node assignments."""

    validate_graph_arrays(positions=positions, edges=edges)
    validate_diploid_dosages(genotypes)
    if sample_nodes.ndim != 1:
        raise ValueError("sample_nodes must be one-dimensional")
    sample_nodes = _integer_array(sample_nodes, name="sample_nodes", dtype=np.int32)
    if genotypes.shape[0] != sample_nodes.shape[0]:
        raise ValueError(
            f"genotypes have {genotypes.shape[0]} samples but sample_nodes has "
            f"{sample_nodes.shape[0]} entries"
        )
    if np.any(sample_nodes < 0) or np.any(sample_nodes >= positions.shape[0]):
        raise ValueError("sample_nodes must refer to rows in positions")
    if np.unique(sample_nodes).shape[0] < 2:
        raise ValueError("inference requires samples from at least two graph nodes")
    allele_sum = genotypes.sum(axis=0)
    polymorphic = (allele_sum > 0.0) & (allele_sum < 2.0 * genotypes.shape[0])
    if not np.any(polymorphic):
        raise ValueError("genotypes contain no polymorphic SNPs")
