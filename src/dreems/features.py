"""Feature construction for the production DREEMS graph model."""

# ruff: noqa: F722,F821,UP037

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped

EDGE_CONDITIONER_FEATURE_NAMES = (
    "linearized_resistance_edge_log_centered",
    "linearized_resistance_valid",
)
DIFFUSION_BAND_TIMES = (0.25, 0.5, 1.0, 2.0, 4.0)
DIFFUSION_BAND_MAX_SCORE_CHANNELS = 8
SpatialEncoding = Literal["embedded", "topology", "topology_node_covariance"]
SPATIAL_ENCODINGS = ("embedded", "topology", "topology_node_covariance")


def centered(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array.astype(np.float64)
    finite = np.isfinite(array)
    if not np.any(finite):
        return np.zeros_like(array, dtype=np.float64)
    filled = array.copy()
    filled[~finite] = float(np.mean(filled[finite]))
    return filled - float(np.mean(filled))


def centered_edge_feature(values: np.ndarray) -> np.ndarray:
    return centered(np.asarray(values, dtype=np.float64)).astype(np.float32)


def symmetric_psd_eigh(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    covariance = np.asarray(matrix, dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance matrix contains non-finite values")
    try:
        return np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        scale = max(float(np.max(np.abs(covariance))) if covariance.size else 0.0, 1.0)
        identity = np.eye(covariance.shape[0], dtype=np.float64)
        for relative_jitter in (1e-12, 1e-10, 1e-8):
            try:
                return np.linalg.eigh(covariance + relative_jitter * scale * identity)
            except np.linalg.LinAlgError:
                continue
        vectors, values, _ = np.linalg.svd(covariance, full_matrices=False)
        return values, vectors


@jaxtyped(typechecker=beartype)
def standardized_polymorphic_genotypes(
    genotypes: "Float[np.ndarray, 'samples snps']",
) -> "Float[np.ndarray, 'samples polymorphic_snps']":
    allele_sum = genotypes.sum(axis=0)
    polymorphic = (allele_sum > 0.0) & (allele_sum < 2.0 * genotypes.shape[0])
    if not np.any(polymorphic):
        raise ValueError("no polymorphic SNPs")
    x = genotypes[:, polymorphic]
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, ddof=1, keepdims=True)
    keep = np.ravel(sd > 1e-8)
    if not np.any(keep):
        raise ValueError("no usable polymorphic SNPs")
    return (x[:, keep] / sd[:, keep]).astype(np.float64)


@jaxtyped(typechecker=beartype)
def standardized_deme_summaries(
    genotypes: "Float[np.ndarray, 'samples snps']",
    sample_nodes: "Int[np.ndarray, 'samples']",
    num_nodes: int,
) -> dict[str, np.ndarray | int]:
    standardized = standardized_polymorphic_genotypes(genotypes)
    effective_snps = int(standardized.shape[1])
    sample_count = np.bincount(sample_nodes, minlength=num_nodes).astype(np.float64)
    sum_x = np.zeros((num_nodes, effective_snps), dtype=np.float64)
    sum_x2 = np.zeros((num_nodes, effective_snps), dtype=np.float64)
    np.add.at(sum_x, sample_nodes, standardized)
    np.add.at(sum_x2, sample_nodes, standardized * standardized)
    mean_x = np.divide(
        sum_x,
        sample_count[:, None],
        out=np.zeros_like(sum_x),
        where=sample_count[:, None] > 0.0,
    )
    mean_x2 = np.divide(
        sum_x2,
        sample_count[:, None],
        out=np.zeros_like(sum_x2),
        where=sample_count[:, None] > 0.0,
    )
    return {
        "effective_snps": effective_snps,
        "sample_count": sample_count,
        "sum_x": sum_x,
        "sum_x2": sum_x2,
        "mean_x": mean_x,
        "mean_x2": mean_x2,
        "diag_mean": mean_x2.mean(axis=1),
    }


def nearest_node_assignments(
    sample_coordinates: np.ndarray,
    node_coordinates: np.ndarray,
) -> np.ndarray:
    samples = np.asarray(sample_coordinates, dtype=np.float64)
    nodes = np.asarray(node_coordinates, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("sample_coordinates must have shape (samples, 2)")
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("node_coordinates must have shape (nodes, 2)")
    diff = samples[:, None, :] - nodes[None, :, :]
    return np.argmin(np.sum(diff * diff, axis=2), axis=1).astype(np.int32)


def covariance_deme_summaries(
    sample_covariance: np.ndarray,
    sample_nodes: np.ndarray,
    num_nodes: int,
    *,
    effective_snps: int | None = None,
) -> dict[str, np.ndarray | int]:
    covariance = np.asarray(sample_covariance, dtype=np.float64)
    sample_nodes = np.asarray(sample_nodes, dtype=np.int32)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("sample_covariance must be square")
    if covariance.shape[0] != sample_nodes.shape[0]:
        raise ValueError(
            f"sample_covariance has {covariance.shape[0]} rows but sample_nodes has "
            f"{sample_nodes.shape[0]} entries"
        )
    covariance = 0.5 * (covariance + covariance.T)
    sample_count = np.bincount(sample_nodes, minlength=num_nodes).astype(np.float64)
    weights = np.zeros((num_nodes, sample_nodes.shape[0]), dtype=np.float64)
    for sample_index, node in enumerate(sample_nodes):
        count = sample_count[int(node)]
        if count > 0.0:
            weights[int(node), sample_index] = 1.0 / count
    effective = int(effective_snps or max(covariance.shape[0], 2))
    effective = max(effective, 2)
    values, vectors = symmetric_psd_eigh(covariance)
    keep = (
        values > max(float(np.max(values)) * 1e-10, 1e-12)
        if values.size
        else np.zeros((0,), dtype=bool)
    )
    if np.any(keep):
        sample_factors = vectors[:, keep] * np.sqrt(values[keep] * float(effective))[None, :]
        factors = weights @ sample_factors
    else:
        factors = np.zeros((num_nodes, 1), dtype=np.float64)
    diagonal = np.sum(factors * factors, axis=1) / float(effective)
    sum_x = factors * sample_count[:, None]
    sum_x2 = factors * factors * sample_count[:, None]
    return {
        "effective_snps": effective,
        "sample_count": sample_count,
        "sum_x": sum_x,
        "sum_x2": sum_x2,
        "mean_x": factors,
        "mean_x2": factors * factors,
        "diag_mean": diagonal,
    }


def node_covariance_deme_summaries(
    node_covariance: np.ndarray,
    observed_nodes: np.ndarray,
    node_sample_count: np.ndarray,
    num_nodes: int,
    *,
    effective_snps: int,
) -> dict[str, np.ndarray | int]:
    """Construct production features from observed-deme covariance.

    The factorization is equivalent to expanding each observed deme into
    ``node_sample_count`` identical virtual samples, but avoids materializing
    the resulting sample-by-sample covariance matrix.
    """

    covariance = np.asarray(node_covariance, dtype=np.float64)
    observed_nodes = np.asarray(observed_nodes, dtype=np.int32)
    sample_count = np.asarray(node_sample_count, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("node_covariance must be square")
    if observed_nodes.shape != (covariance.shape[0],):
        raise ValueError("observed_nodes must index the rows of node_covariance")
    if sample_count.shape != (num_nodes,):
        raise ValueError("node_sample_count must have shape (nodes,)")
    observed_count = sample_count[observed_nodes]
    if np.any(observed_count <= 0.0):
        raise ValueError("observed nodes must have positive sample counts")
    effective = max(int(effective_snps), 2)
    covariance = 0.5 * (covariance + covariance.T)

    root_count = np.sqrt(observed_count)
    expanded_gram = root_count[:, None] * covariance * root_count[None, :]
    values, vectors = symmetric_psd_eigh(expanded_gram)
    keep = (
        values > max(float(np.max(values)) * 1e-10, 1e-12)
        if values.size
        else np.zeros((0,), dtype=bool)
    )
    if np.any(keep):
        observed_factors = (
            vectors[:, keep]
            * np.sqrt(np.maximum(values[keep], 0.0) * float(effective))[None, :]
            / root_count[:, None]
        )
        factors = np.zeros((num_nodes, observed_factors.shape[1]), dtype=np.float64)
        factors[observed_nodes] = observed_factors
    else:
        factors = np.zeros((num_nodes, 1), dtype=np.float64)
    diagonal = np.sum(factors * factors, axis=1) / float(effective)
    return {
        "effective_snps": effective,
        "sample_count": sample_count,
        "sum_x": factors * sample_count[:, None],
        "sum_x2": factors * factors * sample_count[:, None],
        "mean_x": factors,
        "mean_x2": factors * factors,
        "diag_mean": diagonal,
    }


def sample_covariance_to_node_covariance(
    sample_covariance: np.ndarray,
    sample_nodes: np.ndarray,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse sample covariance to covariance among observed graph nodes."""

    covariance = np.asarray(sample_covariance, dtype=np.float64)
    sample_nodes = np.asarray(sample_nodes, dtype=np.int32)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("sample_covariance must be square")
    if sample_nodes.shape != (covariance.shape[0],):
        raise ValueError("sample_nodes must index sample_covariance rows")
    if np.any(sample_nodes < 0) or np.any(sample_nodes >= num_nodes):
        raise ValueError("sample_nodes contains an out-of-range graph node")

    covariance = 0.5 * (covariance + covariance.T)
    sample_count = np.bincount(sample_nodes, minlength=num_nodes).astype(np.float64)
    observed_nodes = np.flatnonzero(sample_count > 0.0).astype(np.int32)
    node_row = np.full(num_nodes, -1, dtype=np.int32)
    node_row[observed_nodes] = np.arange(observed_nodes.shape[0], dtype=np.int32)
    weights = np.zeros((observed_nodes.shape[0], sample_nodes.shape[0]), dtype=np.float64)
    sample_index = np.arange(sample_nodes.shape[0])
    weights[node_row[sample_nodes], sample_index] = 1.0 / sample_count[sample_nodes]
    node_covariance = weights @ covariance @ weights.T
    node_covariance = 0.5 * (node_covariance + node_covariance.T)
    return node_covariance, observed_nodes, sample_count


def example_sample_nodes(example: Mapping[str, Any], active_positions: np.ndarray) -> np.ndarray:
    if "node_covariance" in example:
        return np.asarray(example["observed_nodes"], dtype=np.int32)
    if "sample_nodes" in example:
        return np.asarray(example["sample_nodes"], dtype=np.int32)
    if "sample_coordinates" not in example:
        raise ValueError("example must contain sample_nodes or sample_coordinates")
    return nearest_node_assignments(
        np.asarray(example["sample_coordinates"], dtype=np.float64),
        active_positions,
    )


def example_deme_summaries(
    example: Mapping[str, Any],
    active_positions: np.ndarray,
    num_nodes: int,
    *,
    canonical_node_covariance: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray | int]]:
    sample_nodes = example_sample_nodes(example, active_positions)
    if canonical_node_covariance:
        if "node_covariance" in example:
            effective_snps = int(example["effective_snps"])
            node_covariance = np.asarray(example["node_covariance"], dtype=np.float64)
            observed_nodes = np.asarray(example["observed_nodes"], dtype=np.int32)
            node_sample_count = np.asarray(example["node_sample_count"], dtype=np.float64)
        else:
            if "sample_covariance" in example:
                sample_covariance = np.asarray(example["sample_covariance"], dtype=np.float64)
                effective_snps = int(
                    example.get("effective_snps") or max(sample_covariance.shape[0], 2)
                )
            elif "genotypes" in example:
                standardized = standardized_polymorphic_genotypes(
                    np.asarray(example["genotypes"], dtype=np.float64)
                )
                effective_snps = int(standardized.shape[1])
                sample_covariance = standardized @ standardized.T / float(effective_snps)
            else:
                raise ValueError(
                    "example must contain node_covariance, sample_covariance, or genotypes"
                )
            node_covariance, observed_nodes, node_sample_count = (
                sample_covariance_to_node_covariance(
                    sample_covariance,
                    sample_nodes,
                    num_nodes,
                )
            )
        return sample_nodes, node_covariance_deme_summaries(
            node_covariance,
            observed_nodes,
            node_sample_count,
            num_nodes,
            effective_snps=effective_snps,
        )
    if "node_covariance" in example:
        return sample_nodes, node_covariance_deme_summaries(
            np.asarray(example["node_covariance"], dtype=np.float64),
            sample_nodes,
            np.asarray(example["node_sample_count"], dtype=np.float64),
            num_nodes,
            effective_snps=int(example["effective_snps"]),
        )
    if "sample_covariance" in example:
        return sample_nodes, covariance_deme_summaries(
            np.asarray(example["sample_covariance"], dtype=np.float64),
            sample_nodes,
            num_nodes,
            effective_snps=(
                int(example["effective_snps"])
                if example.get("effective_snps") is not None
                else None
            ),
        )
    if "genotypes" in example:
        return sample_nodes, standardized_deme_summaries(
            np.asarray(example["genotypes"], dtype=np.float64),
            sample_nodes,
            num_nodes,
        )
    raise ValueError("example must contain node_covariance, sample_covariance, or genotypes")


def deme_covariance_scores(
    summaries: Mapping[str, np.ndarray | int],
    covariance_components: int,
) -> np.ndarray:
    sample_count = np.asarray(summaries["sample_count"], dtype=np.float64)
    mean_x = np.asarray(summaries["mean_x"], dtype=np.float64)
    effective_snps = int(summaries["effective_snps"])
    sampled_nodes = np.flatnonzero(sample_count > 0.0)
    scores = np.zeros((sample_count.shape[0], covariance_components), dtype=np.float64)
    if sampled_nodes.size == 0 or covariance_components <= 0:
        return scores

    deme_x = mean_x[sampled_nodes]
    covariance = (deme_x @ deme_x.T) / float(max(effective_snps, 1))
    values, vectors = symmetric_psd_eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    use = min(covariance_components, vectors.shape[1])
    if use > 0:
        scores[sampled_nodes, :use] = vectors[:, :use] * np.sqrt(np.maximum(values[:use], 0.0))
    for component in range(covariance_components):
        largest = int(np.argmax(np.abs(scores[:, component])))
        if scores[largest, component] < 0.0:
            scores[:, component] *= -1.0
    return scores


def scaled_log_support(values: np.ndarray) -> np.ndarray:
    logged = np.log1p(np.maximum(values, 0.0))
    max_value = float(logged.max()) if logged.size > 0 else 0.0
    if max_value <= 0.0:
        return np.zeros_like(logged, dtype=np.float64)
    return (logged / max_value).astype(np.float64)


def absolute_log_support(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(values, 0.0)).astype(np.float64)


def shortest_path_tree(
    start: int,
    adjacency: list[list[tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    parent = np.full(len(adjacency), -1, dtype=np.int32)
    parent_edge = np.full(len(adjacency), -1, dtype=np.int32)
    queue = [start]
    parent[start] = start
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        for neighbor, edge_index in adjacency[node]:
            if parent[neighbor] >= 0:
                continue
            parent[neighbor] = node
            parent_edge[neighbor] = edge_index
            queue.append(neighbor)
    return parent, parent_edge


def shortest_path_dag(
    start: int,
    adjacency: list[list[tuple[int, int]]],
) -> tuple[list[int], list[list[tuple[int, int]]], np.ndarray, np.ndarray]:
    """Build the directed acyclic graph containing every shortest path."""

    distance = np.full(len(adjacency), -1, dtype=np.int32)
    log_path_count = np.full(len(adjacency), -np.inf, dtype=np.float64)
    predecessors: list[list[tuple[int, int]]] = [[] for _ in adjacency]
    order = [start]
    distance[start] = 0
    log_path_count[start] = 0.0
    head = 0
    while head < len(order):
        node = order[head]
        head += 1
        next_distance = int(distance[node]) + 1
        for neighbor, edge_index in adjacency[node]:
            if distance[neighbor] < 0:
                distance[neighbor] = next_distance
                order.append(neighbor)
            if distance[neighbor] != next_distance:
                continue
            predecessors[neighbor].append((node, edge_index))
            log_path_count[neighbor] = np.logaddexp(
                log_path_count[neighbor],
                log_path_count[node],
            )
    return order, predecessors, distance, log_path_count


def distribute_over_shortest_paths(
    start: int,
    right_nodes: list[int],
    adjacency: list[list[tuple[int, int]]],
    pair_precision: Mapping[tuple[int, int], float],
    pair_cov_sum: Mapping[tuple[int, int], float],
    pair_resistance_sum: Mapping[tuple[int, int], float],
    sample_count: np.ndarray,
    edge_totals: np.ndarray,
) -> None:
    """Distribute pair summaries uniformly over all shortest paths."""

    order, predecessors, distance, log_path_count = shortest_path_dag(start, adjacency)
    node_totals = np.zeros((len(adjacency), edge_totals.shape[1]), dtype=np.float64)
    for right_node in right_nodes:
        if distance[right_node] <= 0:
            continue
        key = (start, right_node)
        total = pair_precision[key]
        if total <= 0.0:
            continue
        node_totals[right_node] += np.asarray(
            [
                total,
                pair_cov_sum[key],
                pair_resistance_sum[key],
                total * float(distance[right_node]),
                float(sample_count[start] * sample_count[right_node]),
            ],
            dtype=np.float64,
        )

    for node in reversed(order):
        if node == start or not predecessors[node]:
            continue
        for parent, edge_index in predecessors[node]:
            fraction = float(np.exp(log_path_count[parent] - log_path_count[node]))
            contribution = fraction * node_totals[node]
            edge_totals[edge_index] += contribution
            node_totals[parent] += contribution


def path_from_tree(
    start: int,
    target: int,
    parent: np.ndarray,
    parent_edge: np.ndarray,
) -> list[int]:
    if start == target:
        return []
    if parent[target] < 0:
        return []
    path = []
    node = target
    while node != start:
        edge_index = int(parent_edge[node])
        if edge_index < 0:
            return []
        path.append(edge_index)
        node = int(parent[node])
    return path


def deme_covariance_precision_summaries(
    summaries: Mapping[str, np.ndarray | int],
    edges: np.ndarray,
    *,
    path_routing: Literal["single_shortest", "all_shortest"] = "single_shortest",
) -> dict[str, np.ndarray]:
    if path_routing not in {"single_shortest", "all_shortest"}:
        raise ValueError(f"unsupported path routing: {path_routing!r}")

    sample_count = np.asarray(summaries["sample_count"], dtype=np.float64)
    sum_x = np.asarray(summaries["sum_x"], dtype=np.float64)
    sum_x2 = np.asarray(summaries["sum_x2"], dtype=np.float64)
    mean_x = np.asarray(summaries["mean_x"], dtype=np.float64)
    diag_mean = np.asarray(summaries["diag_mean"], dtype=np.float64)
    effective_snps = int(summaries["effective_snps"])
    num_nodes = sample_count.shape[0]
    edges = np.asarray(edges, dtype=np.int32)

    node_total = np.zeros(num_nodes, dtype=np.float64)
    node_within = np.zeros(num_nodes, dtype=np.float64)
    edge_direct = np.zeros(edges.shape[0], dtype=np.float64)
    path_precision = np.zeros(edges.shape[0], dtype=np.float64)
    path_cov_sum = np.zeros(edges.shape[0], dtype=np.float64)
    path_resistance_sum = np.zeros(edges.shape[0], dtype=np.float64)
    path_length_sum = np.zeros(edges.shape[0], dtype=np.float64)
    node_total_count = np.zeros(num_nodes, dtype=np.float64)
    node_within_count = np.zeros(num_nodes, dtype=np.float64)
    edge_direct_count = np.zeros(edges.shape[0], dtype=np.float64)
    path_count = np.zeros(edges.shape[0], dtype=np.float64)
    edge_lookup = {
        (int(min(left, right)), int(max(left, right))): index
        for index, (left, right) in enumerate(edges)
    }
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for edge_index, (left, right) in enumerate(edges):
        left = int(left)
        right = int(right)
        adjacency[left].append((right, edge_index))
        adjacency[right].append((left, edge_index))
    if path_routing == "all_shortest":
        for neighbors in adjacency:
            neighbors.sort(key=lambda item: item[0])

    sampled_nodes = np.flatnonzero(sample_count > 0.0)
    pair_precision: dict[tuple[int, int], float] = {}
    pair_cov_sum: dict[tuple[int, int], float] = {}
    pair_resistance_sum: dict[tuple[int, int], float] = {}
    if sampled_nodes.size >= 1 and effective_snps >= 2:
        deme_x = mean_x[sampled_nodes]
        counts = sample_count[sampled_nodes]
        covariance = (deme_x @ deme_x.T) / float(effective_snps)
        mean_square = deme_x * deme_x
        second = (mean_square @ mean_square.T) / float(effective_snps)
        entry_var = np.maximum(second - covariance * covariance, 1e-12)
        entry_var = entry_var / float(max(effective_snps - 1, 1))
        pair_counts = counts[:, None] * counts[None, :]
        total_precision = np.minimum(pair_counts / entry_var, 1e18)

        for node in sampled_nodes:
            count = sample_count[node]
            if count < 2.0:
                continue
            pair_count = 0.5 * count * (count - 1.0)
            within_per_snp = (sum_x[node] * sum_x[node] - sum_x2[node]) / max(
                count * (count - 1.0),
                1.0,
            )
            within_cov = float(np.mean(within_per_snp))
            within_second = float(np.mean(within_per_snp * within_per_snp))
            within_var = max(within_second - within_cov * within_cov, 1e-12)
            within_var /= float(max(effective_snps - 1, 1))
            value = min(pair_count / within_var, 1e18)
            node_total[node] += value
            node_within[node] += value
            node_total_count[node] += pair_count
            node_within_count[node] += pair_count

        for left_index in range(sampled_nodes.size - 1):
            left_node = int(sampled_nodes[left_index])
            for right_index in range(left_index + 1, sampled_nodes.size):
                right_node = int(sampled_nodes[right_index])
                value = float(total_precision[left_index, right_index])
                if value <= 0.0:
                    continue
                pair_count = float(counts[left_index] * counts[right_index])
                node_total[left_node] += value
                node_total[right_node] += value
                node_total_count[left_node] += pair_count
                node_total_count[right_node] += pair_count
                key = (left_node, right_node)
                pair_precision[key] = value
                pair_cov = float(covariance[left_index, right_index])
                pair_cov_sum[key] = value * pair_cov
                resistance = float(diag_mean[left_node] + diag_mean[right_node] - 2.0 * pair_cov)
                pair_resistance_sum[key] = value * resistance
                edge_index = edge_lookup.get(key)
                if edge_index is not None:
                    edge_direct[edge_index] += value
                    edge_direct_count[edge_index] += pair_count

        targets_by_source: dict[int, list[int]] = {}
        for left_node, right_node in pair_precision:
            targets_by_source.setdefault(left_node, []).append(right_node)
        for left_node, right_nodes in targets_by_source.items():
            if path_routing == "all_shortest":
                edge_totals = np.zeros((edges.shape[0], 5), dtype=np.float64)
                distribute_over_shortest_paths(
                    left_node,
                    right_nodes,
                    adjacency,
                    pair_precision,
                    pair_cov_sum,
                    pair_resistance_sum,
                    sample_count,
                    edge_totals,
                )
                path_precision += edge_totals[:, 0]
                path_cov_sum += edge_totals[:, 1]
                path_resistance_sum += edge_totals[:, 2]
                path_length_sum += edge_totals[:, 3]
                path_count += edge_totals[:, 4]
                continue

            parent, parent_edge = shortest_path_tree(left_node, adjacency)
            for right_node in right_nodes:
                key = (left_node, right_node)
                total = pair_precision[key]
                if total <= 0.0:
                    continue
                path = path_from_tree(left_node, right_node, parent, parent_edge)
                if not path:
                    continue
                pair_cov = pair_cov_sum[key] / total
                pair_resistance = pair_resistance_sum[key] / total
                path_length = float(len(path))
                count_total = float(sample_count[left_node] * sample_count[right_node])
                for edge_index in path:
                    path_precision[edge_index] += total
                    path_cov_sum[edge_index] += total * pair_cov
                    path_resistance_sum[edge_index] += total * pair_resistance
                    path_length_sum[edge_index] += total * path_length
                    path_count[edge_index] += count_total

    node_total_support = scaled_log_support(node_total)
    node_within_support = scaled_log_support(node_within)
    edge_direct_support = scaled_log_support(edge_direct)
    edge_path_support = scaled_log_support(path_precision)
    node_total_log_support = absolute_log_support(node_total_count)
    node_within_log_support = absolute_log_support(node_within_count)
    edge_direct_log_support = absolute_log_support(edge_direct_count)
    edge_path_log_support = absolute_log_support(path_count)
    edge_path_cov = np.divide(
        path_cov_sum,
        np.maximum(path_precision, 1.0),
        out=np.zeros_like(path_cov_sum),
        where=path_precision > 0.0,
    )
    edge_path_resistance = np.divide(
        path_resistance_sum,
        np.maximum(path_precision, 1.0),
        out=np.zeros_like(path_resistance_sum),
        where=path_precision > 0.0,
    )
    edge_path_resistance = np.log1p(np.maximum(edge_path_resistance, 0.0))
    edge_path_length = np.divide(
        path_length_sum,
        np.maximum(path_precision, 1.0),
        out=np.zeros_like(path_length_sum),
        where=path_precision > 0.0,
    )
    max_path_length = float(edge_path_length.max()) if edge_path_length.size else 0.0
    if max_path_length > 0.0:
        edge_path_length = edge_path_length / max_path_length
    endpoint_support = 0.5 * (node_total_support[edges[:, 0]] + node_total_support[edges[:, 1]])
    endpoint_log_support = 0.5 * (
        node_total_log_support[edges[:, 0]] + node_total_log_support[edges[:, 1]]
    )
    return {
        "node_total_support": node_total_support,
        "node_within_support": node_within_support,
        "node_total_log_support": node_total_log_support,
        "node_within_log_support": node_within_log_support,
        "edge_direct_support": edge_direct_support,
        "edge_endpoint_support": endpoint_support,
        "edge_path_support": edge_path_support,
        "edge_direct_log_support": edge_direct_log_support,
        "edge_endpoint_log_support": endpoint_log_support,
        "edge_path_log_support": edge_path_log_support,
        "edge_path_cov": edge_path_cov,
        "edge_path_resistance": edge_path_resistance,
        "edge_path_length": edge_path_length,
    }


def standardized_signal_columns(values: np.ndarray) -> np.ndarray:
    signals = np.asarray(values, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals[:, None]
    if signals.size == 0:
        return np.zeros_like(signals, dtype=np.float64)
    out = signals.copy()
    for column in range(out.shape[1]):
        values_column = out[:, column]
        finite = np.isfinite(values_column)
        if not np.any(finite):
            out[:, column] = 0.0
            continue
        mean = float(np.mean(values_column[finite]))
        filled = values_column.copy()
        filled[~finite] = mean
        centered_column = filled - mean
        sd = float(np.std(centered_column))
        if sd <= 1e-8:
            out[:, column] = 0.0
        else:
            out[:, column] = np.clip(centered_column / sd, -8.0, 8.0)
    return out.astype(np.float64)


def geometry_normalized_laplacian(
    positions: np.ndarray,
    edges: np.ndarray,
) -> sp.csr_matrix:
    positions = np.asarray(positions, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int32)
    num_nodes = int(positions.shape[0])
    if num_nodes == 0:
        return sp.csr_matrix((0, 0), dtype=np.float64)
    if edges.shape[0] == 0:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float64)

    left = edges[:, 0].astype(np.int64)
    right = edges[:, 1].astype(np.int64)
    lengths = np.linalg.norm(positions[right] - positions[left], axis=1)
    positive = lengths[np.isfinite(lengths) & (lengths > 0.0)]
    reference_length = float(np.median(positive)) if positive.size else 1.0
    weights = reference_length / np.maximum(lengths, max(reference_length * 1e-6, 1e-12))
    weights = np.clip(weights, 0.25, 4.0)

    degree = np.bincount(
        np.concatenate([left, right]),
        weights=np.concatenate([weights, weights]),
        minlength=num_nodes,
    ).astype(np.float64)
    inv_sqrt_degree = np.zeros(num_nodes, dtype=np.float64)
    positive_degree = degree > 0.0
    inv_sqrt_degree[positive_degree] = 1.0 / np.sqrt(degree[positive_degree])
    scaled = weights * inv_sqrt_degree[left] * inv_sqrt_degree[right]
    rows = np.concatenate([np.arange(num_nodes), left, right])
    cols = np.concatenate([np.arange(num_nodes), right, left])
    data = np.concatenate(
        [
            positive_degree.astype(np.float64),
            -scaled,
            -scaled,
        ]
    )
    return sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()


def diffusion_band_summaries(
    positions: np.ndarray,
    edges: np.ndarray,
    signals: np.ndarray,
    *,
    times: tuple[float, ...] = DIFFUSION_BAND_TIMES,
) -> dict[str, np.ndarray]:
    positions = np.asarray(positions, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int32)
    signals = standardized_signal_columns(signals)
    num_nodes = int(positions.shape[0])
    num_edges = int(edges.shape[0])
    signal_count = int(signals.shape[1]) if signals.ndim == 2 else 0
    band_count = len(times) + 1
    if num_nodes == 0 or signal_count == 0:
        empty_node = np.zeros((num_nodes, 0), dtype=np.float64)
        empty_edge = np.zeros((num_edges, 0), dtype=np.float64)
        return {
            "node_bands": empty_node,
            "edge_abs_contrast": empty_edge,
            "edge_squared_contrast": empty_edge,
            "edge_endpoint_energy": empty_edge,
        }
    if num_edges == 0:
        node_width = signal_count * band_count
        edge_width = signal_count * band_count
        return {
            "node_bands": np.zeros((num_nodes, node_width), dtype=np.float64),
            "edge_abs_contrast": np.zeros((num_edges, edge_width), dtype=np.float64),
            "edge_squared_contrast": np.zeros((num_edges, edge_width), dtype=np.float64),
            "edge_endpoint_energy": np.zeros((num_edges, edge_width), dtype=np.float64),
        }

    laplacian = geometry_normalized_laplacian(positions, edges)
    lowpasses: list[np.ndarray] = []
    for time_value in times:
        smoothed = spla.expm_multiply((-float(time_value)) * laplacian, signals)
        lowpasses.append(np.asarray(smoothed, dtype=np.float64))

    bands = []
    previous = signals
    for smoothed in lowpasses:
        bands.append(previous - smoothed)
        previous = smoothed
    bands.append(previous)
    node_bands = np.concatenate(bands, axis=1)
    left = edges[:, 0].astype(np.int64)
    right = edges[:, 1].astype(np.int64)
    edge_abs_blocks = []
    edge_squared_blocks = []
    edge_energy_blocks = []
    for band in bands:
        contrast = band[right] - band[left]
        abs_contrast = np.abs(contrast)
        edge_abs_blocks.append(abs_contrast)
        edge_squared_blocks.append(np.log1p(contrast * contrast))
        edge_energy_blocks.append(0.5 * (np.abs(band[left]) + np.abs(band[right])))
    return {
        "node_bands": np.nan_to_num(node_bands, copy=False).astype(np.float64),
        "edge_abs_contrast": np.nan_to_num(
            np.concatenate(edge_abs_blocks, axis=1),
            copy=False,
        ).astype(np.float64),
        "edge_squared_contrast": np.nan_to_num(
            np.concatenate(edge_squared_blocks, axis=1),
            copy=False,
        ).astype(np.float64),
        "edge_endpoint_energy": np.nan_to_num(
            np.concatenate(edge_energy_blocks, axis=1),
            copy=False,
        ).astype(np.float64),
    }


@beartype
def graph_features(
    example: Mapping[str, Any],
    covariance_components: int,
    *,
    spatial_encoding: SpatialEncoding = "embedded",
) -> tuple[np.ndarray, np.ndarray]:
    if spatial_encoding not in SPATIAL_ENCODINGS:
        raise ValueError(f"unsupported spatial encoding: {spatial_encoding!r}")
    topology = spatial_encoding in {"topology", "topology_node_covariance"}

    graph = example["graph"]
    active_positions = np.asarray(example["active_positions"], dtype=np.float64)
    num_nodes = int(np.asarray(graph["active_full"]).shape[0])
    edges = np.asarray(graph["edges"], dtype=np.int32)

    _sample_nodes, summaries = example_deme_summaries(
        example,
        active_positions,
        num_nodes,
        canonical_node_covariance=spatial_encoding == "topology_node_covariance",
    )
    sample_count = np.asarray(summaries["sample_count"], dtype=np.float64)
    effective_snps = int(summaries["effective_snps"])
    score_mean = deme_covariance_scores(summaries, covariance_components)

    coords = active_positions - active_positions.mean(axis=0, keepdims=True)
    coord_scale = float(np.max(np.linalg.norm(coords, axis=1)))
    if coord_scale > 0.0:
        coords = coords / coord_scale
    if topology:
        coords = np.zeros_like(coords)

    global_se = 1.0 / np.sqrt(effective_snps)
    node_se = np.zeros_like(sample_count)
    sampled = sample_count > 0.0
    node_se[sampled] = 1.0 / np.sqrt(effective_snps * sample_count[sampled])
    log_count = np.log1p(sample_count)
    log_count = log_count / max(float(log_count.max()), 1.0)

    precision = deme_covariance_precision_summaries(
        summaries,
        edges,
        path_routing="all_shortest" if topology else "single_shortest",
    )
    score_channels = min(int(score_mean.shape[1]), DIFFUSION_BAND_MAX_SCORE_CHANNELS)
    diffusion_signal_blocks = []
    if score_channels > 0:
        diffusion_signal_blocks.append(score_mean[:, :score_channels])
    diffusion_signal_blocks.extend(
        [
            log_count[:, None],
            sampled[:, None].astype(np.float64),
            precision["node_total_support"][:, None],
            precision["node_within_support"][:, None],
            np.asarray(summaries["diag_mean"], dtype=np.float64)[:, None],
        ]
    )
    diffusion = diffusion_band_summaries(
        active_positions,
        edges,
        np.concatenate(diffusion_signal_blocks, axis=1),
    )
    node_feature_blocks = [
        score_mean,
        coords,
        log_count[:, None],
        sampled[:, None].astype(np.float64),
        np.full((num_nodes, 1), np.log1p(effective_snps), dtype=np.float64),
        np.full((num_nodes, 1), global_se, dtype=np.float64),
        node_se[:, None],
        precision["node_total_support"][:, None],
        precision["node_within_support"][:, None],
        precision["node_total_log_support"][:, None],
        precision["node_within_log_support"][:, None],
    ]
    node_feature_blocks.append(diffusion["node_bands"])
    node_x = np.concatenate(
        node_feature_blocks,
        axis=1,
    ).astype(np.float32)

    delta = active_positions[edges[:, 1]] - active_positions[edges[:, 0]]
    length = np.linalg.norm(delta, axis=1, keepdims=True)
    maximum_length = float(length.max()) if length.size else 0.0
    max_length = (
        max(maximum_length, np.finfo(np.float64).eps) if topology else max(maximum_length, 1.0)
    )
    left_count = sample_count[edges[:, 0]]
    right_count = sample_count[edges[:, 1]]
    left_se = np.zeros_like(left_count)
    right_se = np.zeros_like(right_count)
    left_se[left_count > 0.0] = 1.0 / np.sqrt(effective_snps * left_count[left_count > 0.0])
    right_se[right_count > 0.0] = 1.0 / np.sqrt(effective_snps * right_count[right_count > 0.0])
    delta_features = np.zeros_like(delta) if topology else delta / max_length
    left_log_count = np.log1p(left_count)
    right_log_count = np.log1p(right_count)
    if topology:
        count_features = np.column_stack(
            [0.5 * (left_log_count + right_log_count), np.abs(left_log_count - right_log_count)]
        )
        se_features = np.column_stack([0.5 * (left_se + right_se), np.abs(left_se - right_se)])
    else:
        count_features = np.column_stack([left_log_count, right_log_count])
        se_features = np.column_stack([left_se, right_se])
    edge_feature_blocks = [
        delta_features,
        length / max_length,
        count_features,
        se_features,
        np.full((edges.shape[0], 1), global_se, dtype=np.float64),
        precision["edge_direct_support"][:, None],
        precision["edge_endpoint_support"][:, None],
        precision["edge_direct_log_support"][:, None],
        precision["edge_endpoint_log_support"][:, None],
        precision["edge_path_support"][:, None],
        precision["edge_path_cov"][:, None],
        precision["edge_path_resistance"][:, None],
        precision["edge_path_length"][:, None],
        precision["edge_path_log_support"][:, None],
    ]
    edge_feature_blocks.extend(
        [
            diffusion["edge_abs_contrast"],
            diffusion["edge_squared_contrast"],
            diffusion["edge_endpoint_energy"],
        ]
    )
    edge_x = np.concatenate(edge_feature_blocks, axis=1).astype(np.float32)
    return node_x, edge_x


def prepare_examples(
    examples: list[Mapping[str, Any]],
    *,
    covariance_components: int,
    spatial_encoding: SpatialEncoding = "embedded",
) -> list[dict[str, np.ndarray]]:
    out: list[dict[str, np.ndarray]] = []
    for example in examples:
        node_x, edge_x = graph_features(
            example,
            covariance_components,
            spatial_encoding=spatial_encoding,
        )
        edges = np.asarray(example["graph"]["edges"], dtype=np.int32)
        y = np.asarray(example.get("edge_log_rates", np.zeros(edges.shape[0])), dtype=np.float32)
        q = np.asarray(
            example.get("node_log_q", np.zeros(node_x.shape[0], dtype=np.float32)),
            dtype=np.float32,
        )
        out.append(
            {
                "node_x": node_x,
                "edge_x": edge_x,
                "edges": edges,
                "y": y,
                "q": q,
                "sampled_node_fraction": np.asarray(
                    np.unique(
                        example_sample_nodes(
                            example,
                            np.asarray(example["active_positions"], dtype=np.float64),
                        )
                    ).size
                    / max(node_x.shape[0], 1),
                    dtype=np.float32,
                ),
            }
        )
    return out


def apply_normalization(
    items: list[dict[str, np.ndarray]],
    stats: Mapping[str, Any],
) -> list[dict[str, np.ndarray]]:
    node_mean = np.asarray(stats["node_mean"], dtype=np.float32)[None, :]
    node_std = np.asarray(stats["node_std"], dtype=np.float32)[None, :]
    edge_mean = np.asarray(stats["edge_mean"], dtype=np.float32)[None, :]
    edge_std = np.asarray(stats["edge_std"], dtype=np.float32)[None, :]
    y_scale = float(stats["y_scale"])
    q_y_scale = float(stats["q_y_scale"])
    out = []
    for item in items:
        y = item["y"] - float(item["y"].mean())
        q = item["q"] - float(item["q"].mean())
        edge_count = int(np.asarray(item["edges"]).shape[0])
        initializer_values = np.asarray(
            item.get(
                "linearized_resistance_edge_log",
                np.zeros(edge_count, dtype=np.float32),
            ),
            dtype=np.float64,
        )
        initializer_valid = np.asarray(
            item.get(
                "linearized_resistance_valid",
                np.zeros(edge_count, dtype=np.float32),
            ),
            dtype=np.float32,
        )
        if initializer_values.shape != (edge_count,) or initializer_valid.shape != (edge_count,):
            raise ValueError("linearized initializer fields must match the number of edges")
        finite_valid = (initializer_valid > 0.0) & np.isfinite(initializer_values)
        initializer_valid = finite_valid.astype(np.float32)
        if np.any(finite_valid):
            initializer_center = float(np.mean(initializer_values[finite_valid]))
            initializer = np.where(
                finite_valid,
                initializer_values - initializer_center,
                0.0,
            )
        else:
            initializer = np.zeros((edge_count,), dtype=np.float64)
        normalized: dict[str, np.ndarray] = {
            "node_x": ((item["node_x"] - node_mean) / node_std).astype(np.float32),
            "edge_x": ((item["edge_x"] - edge_mean) / edge_std).astype(np.float32),
            "edges": item["edges"],
            "y": (y / y_scale).astype(np.float32),
            "q": (q / q_y_scale).astype(np.float32),
            "edge_initializer": (initializer / y_scale).astype(np.float32),
            "edge_initializer_valid": initializer_valid,
        }
        if "counterfactual_group" in item:
            normalized["counterfactual_group"] = np.asarray(
                item["counterfactual_group"],
                dtype=np.int32,
            )
        if "counterfactual_anchor" in item:
            normalized["counterfactual_anchor"] = np.asarray(
                item["counterfactual_anchor"],
                dtype=np.float32,
            )
        if "counterfactual_mask" in item:
            normalized["counterfactual_mask"] = np.asarray(
                item["counterfactual_mask"],
                dtype=np.int32,
            )
        if "sampled_node_fraction" in item:
            normalized["sampled_node_fraction"] = np.asarray(
                item["sampled_node_fraction"],
                dtype=np.float32,
            )
        out.append(normalized)
    return out


def normalize_prepared(
    train: list[dict[str, np.ndarray]],
    validation: list[dict[str, np.ndarray]] | None = None,
    *,
    std_floor: float = 0.2,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, np.ndarray]] | None, dict[str, object]]:
    if not train:
        raise ValueError("cannot normalize an empty training set")
    train_node = np.concatenate([x["node_x"] for x in train], axis=0)
    train_edge = np.concatenate([x["edge_x"] for x in train], axis=0)
    y_all = np.concatenate([x["y"] for x in train])
    q_all = np.concatenate([x["q"] for x in train])
    floor = max(std_floor, 1e-12)
    node_mean = train_node.mean(axis=0, keepdims=True)
    node_std = np.maximum(train_node.std(axis=0, keepdims=True), floor)
    edge_mean = train_edge.mean(axis=0, keepdims=True)
    edge_std = np.maximum(train_edge.std(axis=0, keepdims=True), floor)
    stats = {
        "node_mean": np.ravel(node_mean).astype(float).tolist(),
        "node_std": np.ravel(node_std).astype(float).tolist(),
        "edge_mean": np.ravel(edge_mean).astype(float).tolist(),
        "edge_std": np.ravel(edge_std).astype(float).tolist(),
        "normalization_std_floor": floor,
        "y_scale": float(y_all.std() + 1e-6),
        "q_y_scale": float(q_all.std() + 1e-6),
    }
    normalized_train = apply_normalization(train, stats)
    normalized_validation = (
        apply_normalization(validation, stats) if validation is not None else None
    )
    return normalized_train, normalized_validation, stats


def pad_examples(
    items: list[dict[str, np.ndarray]],
    max_nodes: int | None = None,
    max_edges: int | None = None,
) -> dict[str, np.ndarray]:
    if not items:
        raise ValueError("cannot pad an empty example list")
    max_nodes = max_nodes or max(x["node_x"].shape[0] for x in items)
    max_edges = max_edges or max(x["edges"].shape[0] for x in items)
    batch = len(items)
    node_dim = items[0]["node_x"].shape[1]
    edge_dim = items[0]["edge_x"].shape[1]
    node_x = np.zeros((batch, max_nodes, node_dim), dtype=np.float32)
    edge_x = np.zeros((batch, max_edges, edge_dim), dtype=np.float32)
    edges = np.zeros((batch, max_edges, 2), dtype=np.int32)
    y = np.zeros((batch, max_edges), dtype=np.float32)
    q = np.zeros((batch, max_nodes), dtype=np.float32)
    node_mask = np.zeros((batch, max_nodes), dtype=np.float32)
    edge_mask = np.zeros((batch, max_edges), dtype=np.float32)
    edge_initializer = np.zeros((batch, max_edges), dtype=np.float32)
    edge_initializer_valid = np.zeros((batch, max_edges), dtype=np.float32)
    counterfactual_group = np.full((batch,), -1, dtype=np.int32)
    counterfactual_anchor = np.zeros((batch,), dtype=np.float32)
    counterfactual_mask = np.full((batch,), -1, dtype=np.int32)
    sampled_node_fraction = np.full((batch,), np.nan, dtype=np.float32)
    for index, item in enumerate(items):
        n = item["node_x"].shape[0]
        e = item["edges"].shape[0]
        node_x[index, :n] = item["node_x"]
        edge_x[index, :e] = item["edge_x"]
        edges[index, :e] = item["edges"]
        y[index, :e] = item["y"]
        q[index, :n] = item["q"]
        node_mask[index, :n] = 1.0
        edge_mask[index, :e] = 1.0
        if "edge_initializer" in item:
            edge_initializer[index, :e] = item["edge_initializer"]
        if "edge_initializer_valid" in item:
            edge_initializer_valid[index, :e] = item["edge_initializer_valid"]
        if "counterfactual_group" in item:
            counterfactual_group[index] = int(np.asarray(item["counterfactual_group"]))
        if "counterfactual_anchor" in item:
            counterfactual_anchor[index] = float(np.asarray(item["counterfactual_anchor"]))
        if "counterfactual_mask" in item:
            counterfactual_mask[index] = int(np.asarray(item["counterfactual_mask"]))
        if "sampled_node_fraction" in item:
            sampled_node_fraction[index] = float(np.asarray(item["sampled_node_fraction"]))
    return {
        "node_x": node_x,
        "edge_x": edge_x,
        "edges": edges,
        "y": y,
        "q": q,
        "node_mask": node_mask,
        "edge_mask": edge_mask,
        "edge_initializer": edge_initializer,
        "edge_initializer_valid": edge_initializer_valid,
        "counterfactual_group": counterfactual_group,
        "counterfactual_anchor": counterfactual_anchor,
        "counterfactual_mask": counterfactual_mask,
        "sampled_node_fraction": sampled_node_fraction,
    }


def sparse_laplacian(
    num_nodes: int,
    edges: np.ndarray,
    edge_log_rates: np.ndarray,
) -> sp.csr_matrix:
    rates = np.exp(np.asarray(edge_log_rates, dtype=np.float64))
    rows = []
    cols = []
    data = []
    for (left, right), rate in zip(np.asarray(edges, dtype=np.int32), rates, strict=True):
        left = int(left)
        right = int(right)
        rows.extend([left, right, left, right])
        cols.extend([left, right, right, left])
        data.extend([rate, rate, -rate, -rate])
    return sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()


def edge_conditioner_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    edge_conditioner = config.get("edge_conditioner")
    if not isinstance(edge_conditioner, Mapping):
        raise ValueError("model metadata is missing edge_conditioner")
    return edge_conditioner


def _linearized_conditioner_observation(
    example: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, int | None, np.ndarray | None]:
    active_positions = np.asarray(example["active_positions"], dtype=np.float64)
    sample_nodes = example_sample_nodes(example, active_positions)
    if "node_covariance" in example:
        return (
            np.asarray(example["node_covariance"], dtype=np.float64),
            sample_nodes,
            int(example["effective_snps"]),
            np.asarray(example["node_sample_count"], dtype=np.float64),
        )
    if "sample_covariance" in example:
        effective_snps = (
            int(example["effective_snps"]) if example.get("effective_snps") is not None else None
        )
        return (
            np.asarray(example["sample_covariance"], dtype=np.float64),
            sample_nodes,
            effective_snps,
            None,
        )
    if "genotypes" in example:
        standardized = standardized_polymorphic_genotypes(
            np.asarray(example["genotypes"], dtype=np.float64)
        )
        effective_snps = int(standardized.shape[1])
        sample_covariance = (standardized @ standardized.T) / float(max(effective_snps, 1))
        return sample_covariance.astype(np.float64), sample_nodes, effective_snps, None
    raise ValueError("example must contain node_covariance, sample_covariance, or genotypes")


def _fit_all_pair_diagonal_resistance_proxy(
    example: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, object]]:
    from dreems.linearized import all_pair_adjoint_conductance_initializer

    graph = example["graph"]
    edges = np.asarray(graph["edges"], dtype=np.int32)
    node_coordinates = np.asarray(example["active_positions"], dtype=np.float64)
    conditioner = edge_conditioner_config(config)
    sample_covariance, sample_nodes, effective_snps, node_sample_count = (
        _linearized_conditioner_observation(example)
    )
    estimate = all_pair_adjoint_conductance_initializer(
        sample_covariance=sample_covariance,
        sample_nodes=sample_nodes,
        node_coordinates=node_coordinates,
        edges=edges,
        effective_snps=effective_snps,
        node_sample_count=node_sample_count,
        smoothness=float(conditioner.get("smoothness", 0.01)),
        ridge_scale=float(conditioner.get("ridge_scale", 0.001)),
        pair_weight_power=float(conditioner.get("pair_weight_power", 0.0)),
        length_weighted_baseline=bool(conditioner.get("length_weighted_baseline", True)),
        output_sd=(
            None
            if conditioner.get("output_sd") is None
            else float(conditioner.get("output_sd", 1.0))
        ),
        batch_size=int(conditioner.get("batch_size", 256)),
    )
    diagnostics = dict(estimate.diagnostics)
    diagnostics["method"] = "all_pair_diagonal_adjoint"
    diagnostics["n_pairs"] = int(diagnostics.get("observed_pairs", 0))
    diagnostics["n_observed_nodes"] = int(diagnostics.get("observed_nodes", 0))
    return estimate.edge_log_conductance.astype(np.float32), diagnostics


def fit_linearized_resistance_proxy(
    example: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, object]]:
    conditioner = edge_conditioner_config(config)
    if conditioner.get("method") != "all_pair_diagonal_adjoint":
        raise ValueError("DREEMS requires the all-pair diagonal-adjoint conditioner")
    return _fit_all_pair_diagonal_resistance_proxy(example, config)


def append_edge_conditioner_features_to_item(
    item: dict[str, np.ndarray],
    edge_values: np.ndarray | None,
    *,
    valid: bool,
) -> dict[str, np.ndarray]:
    edge_count = int(np.asarray(item["edges"]).shape[0])
    if valid and edge_values is not None:
        values = np.asarray(edge_values, dtype=np.float64)
        if values.shape != (edge_count,):
            raise ValueError(
                f"linearized edge conditioner shape {values.shape} does not match "
                f"edge count {edge_count}"
            )
        feature = centered_edge_feature(values)
        valid_column = np.ones((edge_count,), dtype=np.float32)
    else:
        values = np.full((edge_count,), np.nan, dtype=np.float64)
        feature = np.zeros((edge_count,), dtype=np.float32)
        valid_column = np.zeros((edge_count,), dtype=np.float32)
    out = dict(item)
    out["edge_x"] = np.concatenate(
        [np.asarray(item["edge_x"], dtype=np.float32), feature[:, None], valid_column[:, None]],
        axis=1,
    ).astype(np.float32)
    out["linearized_resistance_edge_log"] = values.astype(np.float32)
    out["linearized_resistance_valid"] = valid_column
    return out


def append_edge_conditioner_features_for_checkpoint(
    prepared: list[dict[str, np.ndarray]],
    examples: list[Mapping[str, Any]],
    model: Mapping[str, Any],
    stats: Mapping[str, Any],
    *,
    fail_on_error: bool = True,
) -> list[dict[str, np.ndarray]]:
    if not prepared:
        return prepared
    conditioner_kind = str(edge_conditioner_config(model).get("kind"))
    if conditioner_kind != "linearized_resistance":
        raise ValueError("DREEMS requires the linearized-resistance edge conditioner")
    feature_count = len(EDGE_CONDITIONER_FEATURE_NAMES)
    current_dims = {int(np.asarray(item["edge_x"]).shape[1]) for item in prepared}
    if len(current_dims) != 1:
        raise ValueError(f"prepared examples have inconsistent edge_x dimensions: {current_dims}")
    current_edge_dim = next(iter(current_dims))
    stats_edge_dim = (
        len(stats.get("edge_mean", [])) if isinstance(stats.get("edge_mean"), list) else 0
    )
    expected_edge_dim = int(model.get("edge_dim", stats_edge_dim or current_edge_dim))
    if current_edge_dim == expected_edge_dim:
        return prepared
    if expected_edge_dim != current_edge_dim + feature_count:
        raise ValueError(
            "checkpoint edge_dim does not match prepared examples and is not the "
            f"linearized-conditioner +{feature_count} case: "
            f"prepared edge_dim={current_edge_dim}, checkpoint edge_dim={expected_edge_dim}"
        )
    if len(prepared) != len(examples):
        raise ValueError(
            f"cannot append edge conditioning: {len(prepared)} prepared examples for "
            f"{len(examples)} raw examples"
        )
    out = []
    failures = 0
    failure_examples: list[dict[str, object]] = []
    seconds = 0.0
    methods: set[str] = set()
    observed_pairs: list[int] = []
    observed_nodes: list[int] = []
    for item, example in zip(prepared, examples, strict=True):
        edge_log = None
        valid = True
        try:
            edge_log, info = fit_linearized_resistance_proxy(example, model)
            seconds += float(info.get("seconds", 0.0))
            methods.add(str(info.get("method", "")))
            if info.get("n_pairs") is not None:
                observed_pairs.append(int(info["n_pairs"]))
            if info.get("n_observed_nodes") is not None:
                observed_nodes.append(int(info["n_observed_nodes"]))
        except Exception as exc:  # noqa: BLE001
            failures += 1
            valid = False
            index = len(out)
            if len(failure_examples) < 10:
                failure_examples.append(
                    {
                        "index": index,
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
            if fail_on_error:
                raise ValueError(
                    "linearized edge conditioner failed for "
                    f"example {index}: {type(exc).__name__}: {exc}"
                ) from exc
        out.append(append_edge_conditioner_features_to_item(item, edge_log, valid=valid))
    print(
        json.dumps(
            {
                "stage": "append_edge_conditioner_features",
                "edge_conditioner": conditioner_kind,
                "edge_conditioner_methods": sorted(method for method in methods if method),
                "examples": len(out),
                "conditioner_failures": failures,
                "conditioner_failure_examples": failure_examples,
                "conditioner_seconds_total": seconds,
                "conditioner_observed_pairs_mean": (
                    None if not observed_pairs else float(np.mean(observed_pairs))
                ),
                "conditioner_observed_nodes_mean": (
                    None if not observed_nodes else float(np.mean(observed_nodes))
                ),
                "edge_dim_before": current_edge_dim,
                "edge_dim_after": expected_edge_dim,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return out
