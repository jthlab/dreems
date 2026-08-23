from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from dreems import features


@dataclass(frozen=True)
class LinearizedConductanceEstimate:
    """All-pair linearized resistance initializer for edge log-conductances."""

    edge_log_conductance: np.ndarray
    edge_information: np.ndarray
    observed_nodes: np.ndarray
    observed_pair_log_resistance: np.ndarray
    baseline_pair_log_resistance: np.ndarray
    pair_residual: np.ndarray
    diagnostics: dict[str, Any]


def _centered(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    return array - float(np.mean(array))


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = _centered(left)
    right = _centered(right)
    denom = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(left, right) / denom)


def _baseline_conductance(
    node_coordinates: np.ndarray,
    edges: np.ndarray,
    *,
    length_weighted: bool,
) -> np.ndarray:
    if not length_weighted:
        return np.ones(edges.shape[0], dtype=np.float64)
    delta = node_coordinates[edges[:, 1]] - node_coordinates[edges[:, 0]]
    length = np.linalg.norm(delta, axis=1)
    positive = length[np.isfinite(length) & (length > 0.0)]
    reference = float(np.median(positive)) if positive.size else 1.0
    return np.clip(
        reference / np.maximum(length, max(reference * 1e-6, 1e-12)),
        0.25,
        4.0,
    ).astype(np.float64)


def _weighted_laplacian(
    num_nodes: int,
    edges: np.ndarray,
    conductance: np.ndarray,
) -> sp.csr_matrix:
    left = edges[:, 0].astype(np.int64)
    right = edges[:, 1].astype(np.int64)
    rows = np.concatenate([left, right, left, right])
    cols = np.concatenate([left, right, right, left])
    data = np.concatenate([conductance, conductance, -conductance, -conductance])
    return sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()


def _grounded_source_potentials(
    num_nodes: int,
    laplacian: sp.csr_matrix,
    source_nodes: np.ndarray,
) -> np.ndarray:
    if num_nodes < 2:
        return np.zeros((num_nodes, source_nodes.shape[0]), dtype=np.float64)
    ground = num_nodes - 1
    reduced = laplacian[:ground, :ground].tocsc()
    rhs = np.zeros((ground, source_nodes.shape[0]), dtype=np.float64)
    for column, node in enumerate(source_nodes.astype(np.int64, copy=False)):
        if int(node) != ground:
            rhs[int(node), column] = 1.0
    solved = np.asarray(spla.spsolve(reduced, rhs), dtype=np.float64)
    if solved.ndim == 1:
        solved = solved[:, None]
    potentials = np.zeros((num_nodes, source_nodes.shape[0]), dtype=np.float64)
    potentials[:ground, :] = solved
    return potentials


def _line_graph_laplacian(num_nodes: int, edges: np.ndarray) -> sp.csr_matrix:
    num_edges = int(edges.shape[0])
    incident: list[list[int]] = [[] for _node in range(num_nodes)]
    for edge_index, (left, right) in enumerate(edges):
        incident[int(left)].append(edge_index)
        incident[int(right)].append(edge_index)

    left_pairs: list[int] = []
    right_pairs: list[int] = []
    for edge_indices in incident:
        for outer in range(len(edge_indices)):
            for inner in range(outer + 1, len(edge_indices)):
                left_pairs.append(edge_indices[outer])
                right_pairs.append(edge_indices[inner])
    if not left_pairs:
        return sp.csr_matrix((num_edges, num_edges), dtype=np.float64)
    left = np.asarray(left_pairs, dtype=np.int64)
    right = np.asarray(right_pairs, dtype=np.int64)
    adjacency = sp.coo_matrix(
        (
            np.ones(left.size * 2, dtype=np.float64),
            (np.concatenate([left, right]), np.concatenate([right, left])),
        ),
        shape=(num_edges, num_edges),
    ).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return (sp.diags(degree) - adjacency).tocsr()


def _deme_pair_log_resistance(
    sample_covariance: np.ndarray,
    sample_nodes: np.ndarray,
    num_nodes: int,
    effective_snps: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    summaries = features.covariance_deme_summaries(
        sample_covariance,
        sample_nodes,
        num_nodes,
        effective_snps=effective_snps,
    )
    sample_count = np.asarray(summaries["sample_count"], dtype=np.float64)
    observed_nodes = np.flatnonzero(sample_count > 0.0).astype(np.int64)
    if observed_nodes.shape[0] < 2:
        raise ValueError("at least two observed graph nodes are required")

    mean_x = np.asarray(summaries["mean_x"], dtype=np.float64)[observed_nodes]
    diagonal = np.asarray(summaries["diag_mean"], dtype=np.float64)[observed_nodes]
    effective = float(int(summaries["effective_snps"]))
    covariance = mean_x @ mean_x.T / max(effective, 1.0)
    resistance = diagonal[:, None] + diagonal[None, :] - 2.0 * covariance
    resistance = 0.5 * (resistance + resistance.T)
    left_index, right_index = np.triu_indices(observed_nodes.shape[0], k=1)
    observed_log = np.log(np.maximum(resistance[left_index, right_index], 1e-8))
    return (
        observed_nodes,
        sample_count[observed_nodes],
        left_index.astype(np.int64, copy=False),
        right_index.astype(np.int64, copy=False),
        _centered(observed_log),
    )


def all_pair_adjoint_conductance_initializer(
    *,
    sample_covariance: np.ndarray,
    sample_nodes: np.ndarray,
    node_coordinates: np.ndarray,
    edges: np.ndarray,
    effective_snps: int | None = None,
    node_sample_count: np.ndarray | None = None,
    smoothness: float = 0.01,
    ridge_scale: float = 0.001,
    pair_weight_power: float = 0.0,
    length_weighted_baseline: bool = True,
    output_sd: float | None = 1.0,
    batch_size: int = 256,
) -> LinearizedConductanceEstimate:
    """Estimate an edge log-conductance field from all observed resistance pairs.

    This is a deliberately approximate FEEMS-like initializer. It linearizes
    effective-resistance observations around a fixed baseline graph, accumulates
    the all-pair adjoint score on edges, and solves a smoothed diagonal-normal
    system on the line graph.
    """

    started = time.perf_counter()
    node_coordinates = np.asarray(node_coordinates, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int32)
    sample_covariance = np.asarray(sample_covariance, dtype=np.float64)
    sample_nodes = np.asarray(sample_nodes, dtype=np.int32)
    if node_coordinates.ndim != 2 or node_coordinates.shape[1] != 2:
        raise ValueError("node_coordinates must have shape (nodes, 2)")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (edges, 2)")
    if sample_covariance.ndim != 2 or sample_covariance.shape[0] != sample_covariance.shape[1]:
        raise ValueError("sample_covariance must be square")
    if sample_covariance.shape[0] != sample_nodes.shape[0]:
        raise ValueError("sample_covariance rows must match sample_nodes")
    if edges.shape[0] == 0:
        raise ValueError("at least one graph edge is required")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    num_nodes = int(node_coordinates.shape[0])
    num_edges = int(edges.shape[0])
    observed_nodes, counts, pair_left, pair_right, observed_log = _deme_pair_log_resistance(
        sample_covariance,
        sample_nodes,
        num_nodes,
        effective_snps,
    )
    if node_sample_count is not None:
        full_count = np.asarray(node_sample_count, dtype=np.float64)
        if full_count.shape != (num_nodes,):
            raise ValueError("node_sample_count must have shape (nodes,)")
        counts = full_count[observed_nodes]
        if np.any(~np.isfinite(counts)) or np.any(counts <= 0.0):
            raise ValueError("observed node_sample_count values must be finite and positive")
    conductance = _baseline_conductance(
        node_coordinates,
        edges,
        length_weighted=length_weighted_baseline,
    )
    laplacian = _weighted_laplacian(num_nodes, edges, conductance)
    potentials = _grounded_source_potentials(num_nodes, laplacian, observed_nodes)

    pair_resistance = np.zeros((pair_left.shape[0],), dtype=np.float64)
    for start in range(0, pair_left.shape[0], batch_size):
        stop = min(start + batch_size, pair_left.shape[0])
        local_left = pair_left[start:stop]
        local_right = pair_right[start:stop]
        local_potential = potentials[:, local_left] - potentials[:, local_right]
        local_columns = np.arange(stop - start)
        pair_resistance[start:stop] = (
            local_potential[observed_nodes[local_left], local_columns]
            - local_potential[observed_nodes[local_right], local_columns]
        )
    pair_resistance = np.maximum(pair_resistance, 1e-10)
    baseline_log = _centered(np.log(pair_resistance))
    residual = observed_log - baseline_log

    if pair_weight_power == 0.0:
        pair_weights = np.ones_like(residual)
    else:
        pair_weights = np.power(
            np.maximum(counts[pair_left] * counts[pair_right], 1.0),
            float(pair_weight_power),
        )
        pair_weights = pair_weights / max(float(np.mean(pair_weights)), 1e-12)

    edge_left = edges[:, 0].astype(np.int64)
    edge_right = edges[:, 1].astype(np.int64)
    score = np.zeros(num_edges, dtype=np.float64)
    information = np.zeros(num_edges, dtype=np.float64)
    for start in range(0, residual.shape[0], batch_size):
        stop = min(start + batch_size, residual.shape[0])
        local_potential = (
            potentials[:, pair_left[start:stop]] - potentials[:, pair_right[start:stop]]
        )
        edge_drop = local_potential[edge_left, :] - local_potential[edge_right, :]
        sensitivity = -(
            conductance[:, None] * edge_drop * edge_drop / pair_resistance[None, start:stop]
        )
        weights = pair_weights[start:stop]
        score += sensitivity @ (weights * residual[start:stop])
        information += np.sum((sensitivity * sensitivity) * weights[None, :], axis=1)

    positive_info = information[information > 0.0]
    ridge = float(ridge_scale) * (float(np.median(positive_info)) if positive_info.size else 1.0)
    ridge = max(ridge, 1e-12)
    line_laplacian = _line_graph_laplacian(num_nodes, edges)
    system = sp.diags(information + ridge) + float(smoothness) * line_laplacian
    edge_log = np.asarray(spla.spsolve(system.tocsc(), score), dtype=np.float64)
    edge_log = _centered(edge_log)
    if output_sd is not None:
        sd = float(np.std(edge_log))
        target_sd = float(output_sd)
        if sd > 1e-12 and target_sd >= 0.0:
            edge_log = edge_log * (target_sd / sd)

    diagnostics = {
        "seconds": float(time.perf_counter() - started),
        "nodes": num_nodes,
        "edges": num_edges,
        "observed_nodes": int(observed_nodes.shape[0]),
        "observed_pairs": int(residual.shape[0]),
        "smoothness": float(smoothness),
        "ridge": float(ridge),
        "ridge_scale": float(ridge_scale),
        "pair_weight_power": float(pair_weight_power),
        "length_weighted_baseline": bool(length_weighted_baseline),
        "baseline_pair_log_corr": _safe_corr(observed_log, baseline_log),
        "pair_residual_sd": float(np.std(residual)),
        "edge_information_median": float(np.median(information)),
        "edge_information_max": float(np.max(information)) if information.size else 0.0,
        "edge_log_sd": float(np.std(edge_log)),
    }
    return LinearizedConductanceEstimate(
        edge_log_conductance=edge_log.astype(np.float64, copy=False),
        edge_information=information.astype(np.float64, copy=False),
        observed_nodes=observed_nodes.astype(np.int32, copy=False),
        observed_pair_log_resistance=observed_log.astype(np.float64, copy=False),
        baseline_pair_log_resistance=baseline_log.astype(np.float64, copy=False),
        pair_residual=residual.astype(np.float64, copy=False),
        diagnostics=diagnostics,
    )
