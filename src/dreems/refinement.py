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
class CovarianceFit:
    """Scale-fitted off-diagonal covariance agreement."""

    corr: float
    r2: float
    mse: float
    alpha: float
    q_scale: float = 0.0

    def to_prefixed_dict(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_corr": self.corr,
            f"{prefix}_r2": self.r2,
            f"{prefix}_mse": self.mse,
            f"{prefix}_alpha": self.alpha,
            f"{prefix}_q_scale": self.q_scale,
        }


@dataclass(frozen=True)
class CovarianceScales:
    """Non-negative scales for migration and local-diversity covariance."""

    alpha: float
    q_scale: float = 0.0


@dataclass(frozen=True)
class DREEMSMapRefinementConfig:
    """Configuration for covariance MAP refinement of edge log-conductances."""

    steps: int = 50
    learning_rate: float = 0.15
    smoothness_weight: float = 1.0e-4
    sd_floor: float = 0.25
    trust_region_radius: float = 0.5
    edge_batch_size: int = 256
    log_every: int = 10


@dataclass(frozen=True)
class DREEMSMapRefinementResult:
    """Result of DREEMS-prior MAP refinement."""

    edge_log_conductance: np.ndarray
    observed_nodes: np.ndarray
    empirical_centered_covariance: np.ndarray
    initial_fit: CovarianceFit
    refined_fit: CovarianceFit
    diagnostics: dict[str, Any]
    history: list[dict[str, float | int]]
    node_log_q: np.ndarray | None = None


def centered(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    finite = np.isfinite(array)
    if not np.any(finite):
        return np.zeros_like(array, dtype=np.float64)
    out = array.astype(np.float64, copy=True)
    out[~finite] = float(np.mean(out[finite]))
    return out - float(np.mean(out))


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (matrix + matrix.T)


def center_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = symmetrize(matrix)
    count = int(matrix.shape[0])
    projector = np.eye(count, dtype=np.float64) - np.ones((count, count)) / float(count)
    return symmetrize(projector @ matrix @ projector)


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = centered(np.ravel(left))
    right = centered(np.ravel(right))
    finite = np.isfinite(left) & np.isfinite(right)
    if int(np.sum(finite)) < 3:
        return float("nan")
    left = left[finite]
    right = right[finite]
    denom = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denom) if denom > 0.0 else float("nan")


def line_graph_laplacian(num_nodes: int, edges: np.ndarray) -> sp.csr_matrix:
    num_edges = int(edges.shape[0])
    incident: list[list[int]] = [[] for _node in range(int(num_nodes))]
    for edge_index, (left, right) in enumerate(np.asarray(edges, dtype=np.int64)):
        incident[int(left)].append(edge_index)
        incident[int(right)].append(edge_index)

    rows: list[int] = []
    cols: list[int] = []
    for edge_indices in incident:
        for outer in range(len(edge_indices)):
            for inner in range(outer + 1, len(edge_indices)):
                left = edge_indices[outer]
                right = edge_indices[inner]
                rows.extend([left, right])
                cols.extend([right, left])
    if not rows:
        return sp.csr_matrix((num_edges, num_edges), dtype=np.float64)
    adjacency = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(num_edges, num_edges),
    ).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return (sp.diags(degree) - adjacency).tocsr()


def edge_line_roughness(
    edge_logs: np.ndarray,
    *,
    num_nodes: int,
    edges: np.ndarray,
) -> float:
    edge_logs = centered(edge_logs)
    line_laplacian = line_graph_laplacian(int(num_nodes), np.asarray(edges, dtype=np.int32))
    energy = float(edge_logs @ (line_laplacian @ edge_logs))
    return math.sqrt(max(energy, 0.0) / float(max(edge_logs.shape[0], 1)))


def posterior_sd_scale(prior_sd: np.ndarray, sd_floor: float) -> np.ndarray:
    sd = np.asarray(prior_sd, dtype=np.float64)
    sd = np.where(np.isfinite(sd), sd, 1.0)
    return np.maximum(sd, float(sd_floor))


def trust_region_diagnostics(
    edge_logs: np.ndarray,
    *,
    initial_logs: np.ndarray,
    prior_sd: np.ndarray,
    sd_floor: float,
) -> dict[str, float]:
    sd = posterior_sd_scale(prior_sd, sd_floor)
    delta = centered(np.asarray(edge_logs, dtype=np.float64) - np.asarray(initial_logs))
    scaled = delta / sd
    return {
        "trust_region_rms": float(np.sqrt(np.mean(scaled * scaled))),
        "trust_region_max_abs": float(np.max(np.abs(scaled))) if scaled.size else 0.0,
        "trust_region_delta_rms": float(np.sqrt(np.mean(delta * delta))),
    }


def project_edge_log_trust_region(
    edge_logs: np.ndarray,
    *,
    initial_logs: np.ndarray,
    prior_sd: np.ndarray,
    sd_floor: float,
    radius: float,
    max_edge_radius: float,
) -> np.ndarray:
    """Project an edge field into the uncertainty-scaled DREEMS trust region."""

    candidate = centered(np.asarray(edge_logs, dtype=np.float64))
    radius = float(radius)
    max_edge_radius = float(max_edge_radius)
    if radius <= 0.0 and max_edge_radius <= 0.0:
        return candidate

    initial = centered(np.asarray(initial_logs, dtype=np.float64))
    sd = posterior_sd_scale(prior_sd, sd_floor)
    scaled = centered(candidate - initial) / sd
    if max_edge_radius > 0.0:
        scaled = np.clip(scaled, -max_edge_radius, max_edge_radius)
    if radius > 0.0:
        rms = float(np.sqrt(np.mean(scaled * scaled)))
        if rms > radius:
            scaled *= radius / max(rms, 1.0e-12)
    return centered(initial + scaled * sd)


def empirical_observed_covariance(
    *,
    sample_covariance: np.ndarray,
    sample_nodes: np.ndarray,
    num_nodes: int,
    effective_snps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    summaries = features.covariance_deme_summaries(
        sample_covariance,
        sample_nodes,
        int(num_nodes),
        effective_snps=effective_snps,
    )
    sample_count = np.asarray(summaries["sample_count"], dtype=np.float64)
    observed_nodes = np.flatnonzero(sample_count > 0.0).astype(np.int32)
    if observed_nodes.shape[0] < 2:
        raise ValueError("at least two observed graph nodes are required")
    mean_x = np.asarray(summaries["mean_x"], dtype=np.float64)[observed_nodes]
    effective = float(int(summaries["effective_snps"]))
    covariance = symmetrize(mean_x @ mean_x.T / max(effective, 1.0))
    return covariance, observed_nodes, sample_count[observed_nodes]


def validate_observed_covariance(
    *,
    node_covariance: np.ndarray,
    observed_nodes: np.ndarray,
    observed_sample_count: np.ndarray,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    covariance = symmetrize(np.asarray(node_covariance, dtype=np.float64))
    nodes = np.asarray(observed_nodes, dtype=np.int32)
    counts = np.asarray(observed_sample_count, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("node_covariance must be square")
    if covariance.shape[0] < 2:
        raise ValueError("at least two observed graph nodes are required")
    if nodes.shape != (covariance.shape[0],):
        raise ValueError("observed_nodes must index node_covariance")
    if counts.shape != nodes.shape:
        raise ValueError("observed_sample_count must match observed_nodes")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("node_covariance must be finite")
    if np.unique(nodes).shape[0] != nodes.shape[0]:
        raise ValueError("observed_nodes must be unique")
    if np.any(nodes < 0) or np.any(nodes >= int(num_nodes)):
        raise ValueError("observed_nodes are outside the graph")
    if np.any(~np.isfinite(counts)) or np.any(counts <= 0.0):
        raise ValueError("observed_sample_count must be finite and positive")
    return covariance, nodes, counts


def source_potentials(
    *,
    num_nodes: int,
    edges: np.ndarray,
    edge_logs: np.ndarray,
    source_nodes: np.ndarray,
) -> np.ndarray:
    laplacian = features.sparse_laplacian(
        int(num_nodes), np.asarray(edges, dtype=np.int32), edge_logs
    )
    if int(num_nodes) < 2:
        return np.zeros((int(num_nodes), int(source_nodes.shape[0])), dtype=np.float64)
    ground = int(num_nodes) - 1
    reduced = laplacian[:ground, :ground].tocsc()
    rhs = np.zeros((ground, int(source_nodes.shape[0])), dtype=np.float64)
    for column, node in enumerate(source_nodes.astype(np.int64, copy=False)):
        if int(node) != ground:
            rhs[int(node), column] = 1.0
    solved = np.asarray(spla.spsolve(reduced, rhs), dtype=np.float64)
    if solved.ndim == 1:
        solved = solved[:, None]
    potentials = np.zeros((int(num_nodes), int(source_nodes.shape[0])), dtype=np.float64)
    potentials[:ground, :] = solved
    return potentials


def model_covariance_at_observed_nodes(
    *,
    num_nodes: int,
    edges: np.ndarray,
    edge_logs: np.ndarray,
    observed_nodes: np.ndarray,
) -> np.ndarray:
    potentials = source_potentials(
        num_nodes=int(num_nodes),
        edges=edges,
        edge_logs=edge_logs,
        source_nodes=observed_nodes,
    )
    return symmetrize(potentials[observed_nodes])


def q_covariance_at_observed_nodes(
    *,
    node_log_q: np.ndarray,
    observed_nodes: np.ndarray,
    observed_sample_count: np.ndarray,
) -> np.ndarray:
    """Return the relative local-diversity covariance for observed deme means.

    ``node_log_q`` is a relative log coalescence-rate field. Its arbitrary
    global offset is absorbed by the fitted ``q_scale``. Averaging over
    ``n_i`` samples at a deme contributes relative variance
    ``exp(node_log_q_i) / n_i``.
    """

    node_log_q = np.asarray(node_log_q, dtype=np.float64)
    observed_nodes = np.asarray(observed_nodes, dtype=np.int64)
    observed_sample_count = np.asarray(observed_sample_count, dtype=np.float64)
    if node_log_q.ndim != 1:
        raise ValueError("node_log_q must be one-dimensional")
    if observed_nodes.shape != observed_sample_count.shape:
        raise ValueError("observed_nodes and observed_sample_count must have matching shapes")
    if np.any(observed_nodes < 0) or np.any(observed_nodes >= node_log_q.shape[0]):
        raise ValueError("observed_nodes are outside node_log_q")
    if np.any(observed_sample_count <= 0.0):
        raise ValueError("observed_sample_count must be positive")

    observed_log_q = node_log_q[observed_nodes]
    finite = np.isfinite(observed_log_q)
    if not np.all(finite):
        replacement = float(np.mean(observed_log_q[finite])) if np.any(finite) else 0.0
        observed_log_q = np.where(finite, observed_log_q, replacement)
    observed_log_q = observed_log_q - float(np.mean(observed_log_q))
    relative_variance = np.exp(np.clip(observed_log_q, -30.0, 30.0))
    relative_variance /= observed_sample_count
    return np.diag(relative_variance)


def offdiag_indices(count: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(int(count), k=1)


def normalized_pair_indices(
    pair_indices: tuple[np.ndarray, np.ndarray] | None,
    *,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if pair_indices is None:
        return offdiag_indices(int(count))
    row = np.asarray(pair_indices[0], dtype=np.int64)
    col = np.asarray(pair_indices[1], dtype=np.int64)
    if row.shape != col.shape:
        raise ValueError("pair index arrays must have the same shape")
    if row.ndim != 1:
        row = np.ravel(row)
        col = np.ravel(col)
    if row.size == 0:
        raise ValueError("pair_indices must include at least one pair")
    if np.any(row < 0) or np.any(col < 0) or np.any(row >= int(count)) or np.any(col >= int(count)):
        raise ValueError("pair_indices are outside the observed covariance shape")
    if np.any(row == col):
        raise ValueError("pair_indices must select off-diagonal pairs")
    return row.astype(np.int64, copy=False), col.astype(np.int64, copy=False)


def fit_nonnegative_scale(target: np.ndarray, basis: np.ndarray) -> float:
    denom = float(np.dot(basis, basis))
    if denom <= 0.0:
        return 0.0
    return max(float(np.dot(target, basis) / denom), 0.0)


def fit_covariance_scales(
    target: np.ndarray,
    model_basis: np.ndarray,
    q_basis: np.ndarray | None = None,
) -> CovarianceScales:
    """Solve the one- or two-component non-negative least-squares problem."""

    target = np.asarray(target, dtype=np.float64)
    model_basis = np.asarray(model_basis, dtype=np.float64)
    if q_basis is None:
        return CovarianceScales(alpha=fit_nonnegative_scale(target, model_basis))

    q_basis = np.asarray(q_basis, dtype=np.float64)
    if target.shape != model_basis.shape or target.shape != q_basis.shape:
        raise ValueError("target and covariance bases must have matching shapes")

    candidates = [
        CovarianceScales(alpha=0.0, q_scale=0.0),
        CovarianceScales(
            alpha=fit_nonnegative_scale(target, model_basis),
            q_scale=0.0,
        ),
        CovarianceScales(
            alpha=0.0,
            q_scale=fit_nonnegative_scale(target, q_basis),
        ),
    ]
    design = np.column_stack((model_basis, q_basis))
    coefficients, _residuals, rank, _singular_values = np.linalg.lstsq(
        design,
        target,
        rcond=None,
    )
    if rank == 2 and np.all(coefficients >= 0.0):
        candidates.append(
            CovarianceScales(
                alpha=float(coefficients[0]),
                q_scale=float(coefficients[1]),
            )
        )

    def residual_energy(scales: CovarianceScales) -> float:
        residual = target - scales.alpha * model_basis - scales.q_scale * q_basis
        return float(np.dot(residual, residual))

    return min(candidates, key=residual_energy)


def covariance_fit(
    *,
    empirical_covariance: np.ndarray,
    model_covariance: np.ndarray,
    q_covariance: np.ndarray | None = None,
    pair_indices: tuple[np.ndarray, np.ndarray] | None = None,
    scales: CovarianceScales | None = None,
) -> CovarianceFit:
    empirical_centered = center_matrix(empirical_covariance)
    model_centered = center_matrix(model_covariance)
    q_centered = None if q_covariance is None else center_matrix(q_covariance)
    row, col = normalized_pair_indices(pair_indices, count=empirical_centered.shape[0])
    target = empirical_centered[row, col]
    model = model_centered[row, col]
    q_basis = None if q_centered is None else q_centered[row, col]
    fitted_scales = scales or fit_covariance_scales(target, model, q_basis)
    if fitted_scales.alpha < 0.0 or fitted_scales.q_scale < 0.0:
        raise ValueError("covariance scales must be non-negative")
    if q_basis is None and fitted_scales.q_scale != 0.0:
        raise ValueError("q_scale must be zero when q_covariance is not supplied")
    prediction = fitted_scales.alpha * model
    if q_basis is not None:
        prediction = prediction + fitted_scales.q_scale * q_basis
    residual = target - prediction
    target_energy = float(np.dot(target, target))
    residual_energy = float(np.dot(residual, residual))
    return CovarianceFit(
        corr=safe_corr(target, prediction),
        r2=1.0 - residual_energy / target_energy if target_energy > 0.0 else float("nan"),
        mse=float(np.mean(residual * residual)) if residual.size else float("nan"),
        alpha=float(fitted_scales.alpha),
        q_scale=float(fitted_scales.q_scale),
    )


def covariance_objective_and_gradient(
    *,
    edge_logs: np.ndarray,
    initial_logs: np.ndarray,
    prior_sd: np.ndarray,
    num_nodes: int,
    edges: np.ndarray,
    observed_nodes: np.ndarray,
    empirical_centered_covariance: np.ndarray,
    line_laplacian: sp.csr_matrix,
    q_centered_covariance: np.ndarray | None = None,
    pair_indices: tuple[np.ndarray, np.ndarray] | None = None,
    prior_weight: float,
    smoothness_weight: float,
    sd_floor: float,
    edge_batch_size: int,
) -> tuple[float, np.ndarray, dict[str, float]]:
    edge_logs = centered(edge_logs)
    initial_logs = centered(initial_logs)
    potentials = source_potentials(
        num_nodes=int(num_nodes),
        edges=edges,
        edge_logs=edge_logs,
        source_nodes=observed_nodes,
    )
    green = symmetrize(potentials[observed_nodes])
    model_centered = center_matrix(green)
    row, col = normalized_pair_indices(pair_indices, count=observed_nodes.shape[0])
    target = empirical_centered_covariance[row, col]
    model = model_centered[row, col]
    q_basis = (
        None
        if q_centered_covariance is None
        else np.asarray(q_centered_covariance, dtype=np.float64)[row, col]
    )
    scales = fit_covariance_scales(target, model, q_basis)
    prediction = scales.alpha * model
    if q_basis is not None:
        prediction = prediction + scales.q_scale * q_basis
    residual = target - prediction
    pair_count = max(int(target.shape[0]), 1)
    data_loss = 0.5 * float(np.mean(residual * residual))

    rates = np.exp(edge_logs)
    edge_left = np.asarray(edges[:, 0], dtype=np.int64)
    edge_right = np.asarray(edges[:, 1], dtype=np.int64)
    edge_drop = potentials[edge_left, :] - potentials[edge_right, :]
    edge_drop = edge_drop - np.mean(edge_drop, axis=1, keepdims=True)
    data_grad = np.zeros(edge_logs.shape[0], dtype=np.float64)
    for start in range(0, edge_logs.shape[0], int(edge_batch_size)):
        stop = min(start + int(edge_batch_size), edge_logs.shape[0])
        local = edge_drop[start:stop]
        quad = np.sum(local[:, row] * local[:, col] * residual[None, :], axis=1)
        data_grad[start:stop] = scales.alpha * rates[start:stop] * quad / float(pair_count)

    sd = np.maximum(np.asarray(prior_sd, dtype=np.float64), float(sd_floor))
    prior_delta = centered(edge_logs - initial_logs)
    prior_terms = prior_delta / sd
    prior_loss = 0.5 * float(prior_weight) * float(np.mean(prior_terms * prior_terms))
    prior_grad = float(prior_weight) * prior_delta / (sd * sd) / float(edge_logs.shape[0])

    smooth_vec = np.asarray(line_laplacian @ edge_logs, dtype=np.float64)
    smooth_loss = (
        0.5 * float(smoothness_weight) * float(edge_logs @ smooth_vec) / float(edge_logs.shape[0])
    )
    smooth_grad = float(smoothness_weight) * smooth_vec / float(edge_logs.shape[0])

    total_grad = centered(data_grad + prior_grad + smooth_grad)
    total_loss = data_loss + prior_loss + smooth_loss
    stats = {
        "loss": float(total_loss),
        "data_loss": float(data_loss),
        "prior_loss": float(prior_loss),
        "smooth_loss": float(smooth_loss),
        "alpha": float(scales.alpha),
        "q_scale": float(scales.q_scale),
        "offdiag_corr": safe_corr(target, prediction),
        "offdiag_r2": 1.0
        - float(np.dot(residual, residual)) / max(float(np.dot(target, target)), 1e-300),
        "edge_log_sd": float(np.std(edge_logs)),
        "grad_rms": float(np.sqrt(np.mean(total_grad * total_grad))),
        "data_grad_rms": float(np.sqrt(np.mean(data_grad * data_grad))),
        "prior_grad_rms": float(np.sqrt(np.mean(prior_grad * prior_grad))),
        "smooth_grad_rms": float(np.sqrt(np.mean(smooth_grad * smooth_grad))),
    }
    return float(total_loss), total_grad, stats


def refine_edge_log_conductance(
    *,
    sample_covariance: np.ndarray | None = None,
    sample_nodes: np.ndarray | None = None,
    node_covariance: np.ndarray | None = None,
    observed_nodes: np.ndarray | None = None,
    observed_sample_count: np.ndarray | None = None,
    node_coordinates: np.ndarray,
    edges: np.ndarray,
    initial_edge_logs: np.ndarray,
    initial_edge_sd: np.ndarray | None = None,
    initial_node_log_q: np.ndarray | None = None,
    effective_snps: int | None = None,
    config: DREEMSMapRefinementConfig | None = None,
    fit_pair_indices: tuple[np.ndarray, np.ndarray] | None = None,
) -> DREEMSMapRefinementResult:
    """Refine edge log-conductances against the observed covariance.

    The objective is a MAP-style criterion:

    data = offdiag least squares between centered empirical covariance and
        alpha * centered Green covariance plus an optional scaled local
        diversity covariance derived from the DREEMS q field, with both
        non-negative scales fitted at each step;
    smooth = optional quadratic penalty on the line-graph Laplacian.

    This implementation uses analytic sparse-solve gradients and Adam updates.
    It is intentionally CPU-friendly and does not depend on JAX autodiff.
    """

    cfg = config or DREEMSMapRefinementConfig()
    if cfg.steps < 0:
        raise ValueError("steps must be non-negative")
    if cfg.learning_rate < 0.0:
        raise ValueError("learning_rate must be non-negative")
    if cfg.trust_region_radius < 0.0:
        raise ValueError("trust_region_radius must be non-negative")
    if cfg.edge_batch_size <= 0:
        raise ValueError("edge_batch_size must be positive")

    started = time.perf_counter()
    node_coordinates = np.asarray(node_coordinates, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int32)
    initial_logs = centered(np.asarray(initial_edge_logs, dtype=np.float64))
    if initial_logs.shape != (edges.shape[0],):
        raise ValueError("initial_edge_logs must have shape (edges,)")
    if initial_edge_sd is None:
        prior_sd = np.ones_like(initial_logs, dtype=np.float64)
    else:
        prior_sd = np.asarray(initial_edge_sd, dtype=np.float64)
        if prior_sd.shape != initial_logs.shape:
            raise ValueError("initial_edge_sd must have shape (edges,)")
    node_log_q = None
    if initial_node_log_q is not None:
        node_log_q = centered(np.asarray(initial_node_log_q, dtype=np.float64))
        if node_log_q.shape != (node_coordinates.shape[0],):
            raise ValueError("initial_node_log_q must have shape (nodes,)")

    if node_covariance is not None:
        if sample_covariance is not None or sample_nodes is not None:
            raise ValueError(
                "provide either node_covariance or sample_covariance/sample_nodes, not both"
            )
        if observed_nodes is None or observed_sample_count is None:
            raise ValueError("node_covariance requires observed_nodes and observed_sample_count")
        empirical, observed_nodes, observed_sample_count = validate_observed_covariance(
            node_covariance=node_covariance,
            observed_nodes=observed_nodes,
            observed_sample_count=observed_sample_count,
            num_nodes=int(node_coordinates.shape[0]),
        )
    else:
        if sample_covariance is None or sample_nodes is None:
            raise ValueError(
                "sample_covariance and sample_nodes are required when node_covariance is omitted"
            )
        empirical, observed_nodes, observed_sample_count = empirical_observed_covariance(
            sample_covariance=np.asarray(sample_covariance, dtype=np.float64),
            sample_nodes=np.asarray(sample_nodes, dtype=np.int32),
            num_nodes=int(node_coordinates.shape[0]),
            effective_snps=effective_snps,
        )
    empirical_centered = center_matrix(empirical)
    q_covariance = (
        None
        if node_log_q is None
        else q_covariance_at_observed_nodes(
            node_log_q=node_log_q,
            observed_nodes=observed_nodes,
            observed_sample_count=observed_sample_count,
        )
    )
    q_centered = None if q_covariance is None else center_matrix(q_covariance)
    pair_indices = normalized_pair_indices(fit_pair_indices, count=observed_nodes.shape[0])
    line_lap = line_graph_laplacian(int(node_coordinates.shape[0]), edges)
    initial_model = model_covariance_at_observed_nodes(
        num_nodes=int(node_coordinates.shape[0]),
        edges=edges,
        edge_logs=initial_logs,
        observed_nodes=observed_nodes,
    )
    initial_fit = covariance_fit(
        empirical_covariance=empirical,
        model_covariance=initial_model,
        q_covariance=q_covariance,
        pair_indices=pair_indices,
    )

    logs = initial_logs.copy()
    first_moment = np.zeros_like(logs)
    second_moment = np.zeros_like(logs)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    history: list[dict[str, float | int]] = []
    for step in range(1, int(cfg.steps) + 1):
        _loss, grad, stats = covariance_objective_and_gradient(
            edge_logs=logs,
            initial_logs=initial_logs,
            prior_sd=prior_sd,
            num_nodes=int(node_coordinates.shape[0]),
            edges=edges,
            observed_nodes=observed_nodes,
            empirical_centered_covariance=empirical_centered,
            line_laplacian=line_lap,
            q_centered_covariance=q_centered,
            pair_indices=pair_indices,
            prior_weight=0.0,
            smoothness_weight=float(cfg.smoothness_weight),
            sd_floor=float(cfg.sd_floor),
            edge_batch_size=int(cfg.edge_batch_size),
        )
        first_moment = beta1 * first_moment + (1.0 - beta1) * grad
        second_moment = beta2 * second_moment + (1.0 - beta2) * (grad * grad)
        first_hat = first_moment / (1.0 - beta1**step)
        second_hat = second_moment / (1.0 - beta2**step)
        update = float(cfg.learning_rate) * first_hat / (np.sqrt(second_hat) + eps)
        logs = project_edge_log_trust_region(
            logs - update,
            initial_logs=initial_logs,
            prior_sd=prior_sd,
            sd_floor=float(cfg.sd_floor),
            radius=float(cfg.trust_region_radius),
            max_edge_radius=0.0,
        )
        if step == 1 or step == int(cfg.steps) or step % int(cfg.log_every) == 0:
            trust_stats = trust_region_diagnostics(
                logs,
                initial_logs=initial_logs,
                prior_sd=prior_sd,
                sd_floor=float(cfg.sd_floor),
            )
            history.append(
                {
                    "step": int(step),
                    "seconds": float(time.perf_counter() - started),
                    "update_rms": float(np.sqrt(np.mean(update * update))),
                    **stats,
                    **trust_stats,
                }
            )

    refined_model = model_covariance_at_observed_nodes(
        num_nodes=int(node_coordinates.shape[0]),
        edges=edges,
        edge_logs=logs,
        observed_nodes=observed_nodes,
    )
    refined_fit = covariance_fit(
        empirical_covariance=empirical,
        model_covariance=refined_model,
        q_covariance=q_covariance,
        pair_indices=pair_indices,
    )
    diagnostics = {
        "seconds": float(time.perf_counter() - started),
        "nodes": int(node_coordinates.shape[0]),
        "edges": int(edges.shape[0]),
        "observed_nodes": int(observed_nodes.shape[0]),
        "observed_pairs": int(observed_nodes.shape[0] * (observed_nodes.shape[0] - 1) // 2),
        "observed_sample_count_mean": float(np.mean(observed_sample_count)),
        "observed_sample_count_max": float(np.max(observed_sample_count)),
        "q_aware": bool(node_log_q is not None),
        "initial_q_scale": float(initial_fit.q_scale),
        "refined_q_scale": float(refined_fit.q_scale),
        "steps": int(cfg.steps),
        "learning_rate": float(cfg.learning_rate),
        "smoothness_weight": float(cfg.smoothness_weight),
        "sd_floor": float(cfg.sd_floor),
        "trust_region_radius": float(cfg.trust_region_radius),
        "edge_batch_size": int(cfg.edge_batch_size),
        "initial_edge_log_sd": float(np.std(initial_logs)),
        "refined_edge_log_sd": float(np.std(logs)),
        **trust_region_diagnostics(
            logs,
            initial_logs=initial_logs,
            prior_sd=prior_sd,
            sd_floor=float(cfg.sd_floor),
        ),
        "initial_roughness": edge_line_roughness(
            initial_logs,
            num_nodes=int(node_coordinates.shape[0]),
            edges=edges,
        ),
        "refined_roughness": edge_line_roughness(
            logs,
            num_nodes=int(node_coordinates.shape[0]),
            edges=edges,
        ),
    }
    return DREEMSMapRefinementResult(
        edge_log_conductance=logs.astype(np.float64, copy=False),
        observed_nodes=observed_nodes.astype(np.int32, copy=False),
        empirical_centered_covariance=empirical_centered.astype(np.float64, copy=False),
        initial_fit=initial_fit,
        refined_fit=refined_fit,
        diagnostics=diagnostics,
        history=history,
        node_log_q=node_log_q,
    )
