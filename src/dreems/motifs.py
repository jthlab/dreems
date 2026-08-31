"""Synthetic graph motifs, sample designs, and landscape cases."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree

from dreems.geography import (
    build_global_landscape_mesh,
    draw_q_surface,
    draw_rff_params,
    draw_surface,
    edge_log_rates_from_surface,
    eval_rff_params,
    feems_triangular_lattice_graph,
    graph_from_positions_edges,
    masked_grid_graph,
    node_log_q_from_surface,
    points_in_geometry,
    random_mask,
    standardize,
)
from dreems.simulation_config import (
    choose_weighted_name,
    draw_local_diploids_per_node,
    draw_weighted_positive_float,
    draw_weighted_positive_int,
    local_simulation_metadata,
    string_list,
)


def sample_grid_dimensions(
    rng: np.random.Generator,
    sim: Mapping[str, Any],
) -> tuple[int, int]:
    height = int(rng.integers(int(sim["min_height"]), int(sim["max_height"]) + 1))
    width = int(rng.integers(int(sim["min_width"]), int(sim["max_width"]) + 1))
    return height, width


def choose_sample_regime(
    rng: np.random.Generator,
    sim: Mapping[str, Any],
    local: dict[str, Any],
) -> dict[str, Any]:
    regimes = sim.get("sample_regimes") or []
    if regimes:
        weights = np.asarray(
            [float(regime.get("weight", 1.0)) for regime in regimes],
            dtype=np.float64,
        )
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
        if float(weights.sum()) <= 0.0:
            weights = np.ones(len(regimes), dtype=np.float64)
        weights /= float(weights.sum())
        regime = dict(regimes[int(rng.choice(len(regimes), p=weights))])
        local["sample_regime"] = str(regime.get("name", "sample_regime"))
        local["sample_design"] = str(
            regime.get(
                "sample_design",
                regime.get("design", string_list(sim.get("sample_designs"), default=["sites"])[0]),
            )
        )
        for key in (
            "min_samples",
            "max_samples",
            "min_sample_fraction",
            "max_sample_fraction",
            "min_diploids_per_site",
            "max_diploids_per_site",
        ):
            if key in regime:
                local[key] = regime[key]
        if "diploids_per_node" in regime:
            local["diploids_per_node"] = max(int(regime["diploids_per_node"]), 1)
        elif "diploids_per_site" in regime:
            local["diploids_per_node"] = max(int(regime["diploids_per_site"]), 1)
        else:
            local["diploids_per_node"] = draw_local_diploids_per_node(rng, local)
        return local

    designs = string_list(sim.get("sample_designs"), default=["sites"])
    local["sample_design"] = choose_weighted_name(rng, designs, None)
    local["sample_regime"] = str(local["sample_design"])
    return local


def draw_sample_nodes_from_positions(
    rng: np.random.Generator,
    positions: np.ndarray,
    local: Mapping[str, Any],
) -> np.ndarray:
    num_nodes = int(positions.shape[0])
    sample_design = str(local.get("sample_design", "sites"))
    diploids_per_node = max(int(local.get("diploids_per_node", 1)), 1)
    min_fraction = local.get("min_sample_fraction")
    max_fraction = local.get("max_sample_fraction")
    min_samples = int(local.get("min_samples", 0) or 0)
    max_samples = int(local.get("max_samples", 0) or 0)
    if min_fraction is not None and max_fraction is not None:
        low_fraction = max(0.0, min(float(min_fraction), float(max_fraction)))
        high_fraction = min(1.0, max(float(min_fraction), float(max_fraction)))
        num_samples = int(round(float(rng.uniform(low_fraction, high_fraction)) * num_nodes))
        if min_samples > 0:
            num_samples = max(num_samples, min_samples)
        if max_samples > 0:
            num_samples = min(num_samples, max_samples)
        num_samples = int(np.clip(num_samples, 1, num_nodes))
    else:
        if min_samples <= 0 or max_samples <= 0:
            fraction = float(local.get("sampled_node_fraction", 0.25))
            num_samples = int(round(fraction * num_nodes))
            num_samples = max(int(local.get("min_sampled_nodes", 2)), num_samples)
            if local.get("max_sampled_nodes") is not None:
                num_samples = min(num_samples, int(local["max_sampled_nodes"]))
            num_samples = int(np.clip(num_samples, 1, num_nodes))
        else:
            num_samples = int(rng.integers(min_samples, max_samples + 1))
            num_samples = int(np.clip(num_samples, 1, num_nodes))

    def repair_sample_nodes(sample_nodes: np.ndarray) -> np.ndarray:
        samples = np.asarray(sample_nodes, dtype=np.int32).reshape(-1)
        if num_nodes <= 0:
            raise ValueError("cannot draw samples from an empty graph")
        if samples.size < 2:
            needed = 2 - samples.size
            if num_nodes == 1:
                extra = np.zeros(needed, dtype=np.int32)
            else:
                present = set(int(value) for value in samples.tolist())
                candidates = [node for node in range(num_nodes) if node not in present]
                if len(candidates) >= needed:
                    extra = rng.choice(candidates, size=needed, replace=False).astype(np.int32)
                else:
                    extra = rng.integers(0, num_nodes, size=needed, dtype=np.int32)
            samples = np.concatenate([samples, extra])
        unique = np.unique(samples)
        if num_nodes >= 2 and unique.shape[0] < 2:
            replacement = int(rng.integers(0, num_nodes - 1))
            if replacement >= int(unique[0]):
                replacement += 1
            samples[-1] = replacement
        return samples.astype(np.int32)

    def cap_total_samples(sample_nodes: np.ndarray) -> np.ndarray:
        samples = np.asarray(sample_nodes, dtype=np.int32).reshape(-1)
        if max_samples <= 0:
            return repair_sample_nodes(samples)
        target = max(int(max_samples), 2)
        if samples.size <= target:
            return repair_sample_nodes(samples)
        unique, counts = np.unique(samples, return_counts=True)
        if unique.shape[0] >= target:
            selected = rng.choice(unique, size=target, replace=False).astype(np.int32)
            return repair_sample_nodes(selected)
        extra_counts = np.maximum(counts - 1, 0)
        extras = np.repeat(unique, extra_counts).astype(np.int32)
        selected = [unique.astype(np.int32)]
        need = target - int(unique.shape[0])
        if need > 0 and extras.size > 0:
            extra_indices = rng.choice(extras.size, size=min(need, extras.size), replace=False)
            selected.append(extras[extra_indices].astype(np.int32))
        return repair_sample_nodes(np.concatenate(selected))

    if sample_design == "full":
        return cap_total_samples(np.repeat(np.arange(num_nodes, dtype=np.int32), diploids_per_node))
    if sample_design == "uniform":
        return cap_total_samples(rng.integers(0, num_nodes, size=num_samples, dtype=np.int32))
    if sample_design == "poisson_sites":
        mean_sites = max(float(num_samples) - 1.0, 0.0)
        num_sites = int(1 + rng.poisson(mean_sites))
        if max_samples > 0:
            num_sites = min(num_sites, max_samples)
        num_sites = int(np.clip(num_sites, 1, num_nodes))
        sites = rng.choice(num_nodes, size=num_sites, replace=False).astype(np.int32)
        mean_copies = max(float(diploids_per_node) - 1.0, 0.0)
        copies = 1 + rng.poisson(mean_copies, size=num_sites)
        return cap_total_samples(np.repeat(sites, copies).astype(np.int32))
    if sample_design == "sites":
        num_sites = min(num_samples, num_nodes)
        sites = rng.choice(num_nodes, size=num_sites, replace=False).astype(np.int32)
        return cap_total_samples(np.repeat(sites, diploids_per_node).astype(np.int32))
    if sample_design != "clustered":
        raise ValueError(f"unknown sample design: {sample_design}")

    num_clusters = int(rng.integers(1, min(8, num_samples) + 1))
    center_indices = rng.choice(num_nodes, size=num_clusters, replace=num_nodes < num_clusters)
    centers = positions[center_indices]
    extent = max(float(np.ptp(positions[:, 0])), float(np.ptp(positions[:, 1])), 1.0)
    sigma = rng.uniform(0.04, 0.14) * extent
    cluster = rng.integers(0, num_clusters, size=num_samples)
    targets = centers[cluster] + rng.normal(scale=sigma, size=(num_samples, 2))
    _distances, nearest = cKDTree(positions).query(targets, k=1)
    return cap_total_samples(np.asarray(nearest, dtype=np.int32))


def node_mean_edge_values(
    num_nodes: int,
    edges: np.ndarray,
    edge_values: np.ndarray,
) -> np.ndarray:
    """Average an undirected edge field onto incident nodes."""

    edge_array = np.asarray(edges, dtype=np.int32)
    values = np.asarray(edge_values, dtype=np.float64)
    sums = np.zeros(int(num_nodes), dtype=np.float64)
    counts = np.zeros(int(num_nodes), dtype=np.float64)
    np.add.at(sums, edge_array[:, 0], values)
    np.add.at(sums, edge_array[:, 1], values)
    np.add.at(counts, edge_array[:, 0], 1.0)
    np.add.at(counts, edge_array[:, 1], 1.0)
    return sums / np.maximum(counts, 1.0)


def node_barrier_proximity(
    num_nodes: int,
    edges: np.ndarray,
    edge_log_rates: np.ndarray,
    *,
    quantile: float,
) -> np.ndarray:
    """Score nodes adjacent to unusually low-conductance edges."""

    edge_array = np.asarray(edges, dtype=np.int32)
    rates = np.asarray(edge_log_rates, dtype=np.float64)
    threshold = float(np.quantile(rates, float(quantile)))
    low_edge_score = np.maximum(threshold - rates, 0.0)
    direct = np.zeros(int(num_nodes), dtype=np.float64)
    np.maximum.at(direct, edge_array[:, 0], low_edge_score)
    np.maximum.at(direct, edge_array[:, 1], low_edge_score)
    adjacency = sp.csr_matrix(
        (
            np.ones(2 * edge_array.shape[0], dtype=np.float64),
            (
                np.r_[edge_array[:, 0], edge_array[:, 1]],
                np.r_[edge_array[:, 1], edge_array[:, 0]],
            ),
        ),
        shape=(int(num_nodes), int(num_nodes)),
    )
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    neighboring = np.asarray(adjacency @ direct).reshape(-1) / np.maximum(degree, 1.0)
    return standardize(direct + 0.5 * neighboring)


def sampling_correlation(values: np.ndarray, score: np.ndarray) -> float:
    left = standardize(np.asarray(values, dtype=np.float64))
    right = standardize(np.asarray(score, dtype=np.float64))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else float("nan")


def apply_migration_associated_sampling(
    rng: np.random.Generator,
    case: dict[str, Any],
    sim: Mapping[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    """Resample site identities while preserving sample counts and graph truth."""

    supported = {
        "independent",
        "migration_positive",
        "migration_negative",
        "barrier_proximity",
    }
    if mode not in supported:
        raise ValueError(f"unsupported sampling association mode: {mode!r}")

    positions = np.asarray(case["active_positions"], dtype=np.float64)
    edges = np.asarray(case["graph"]["edges"], dtype=np.int32)
    edge_log_rates = np.asarray(case["edge_log_rates"], dtype=np.float64)
    original_samples = np.asarray(case["sample_nodes"], dtype=np.int32).reshape(-1)
    num_nodes = int(positions.shape[0])
    migration_score = standardize(node_mean_edge_values(num_nodes, edges, edge_log_rates))
    if "node_land_mask" in case:
        eligible = np.flatnonzero(np.asarray(case["node_land_mask"], dtype=bool))
    else:
        eligible = np.arange(num_nodes, dtype=np.int32)
    if eligible.size < 2:
        eligible = np.arange(num_nodes, dtype=np.int32)
    if eligible.size < 2:
        raise ValueError("migration-associated sampling requires at least two eligible nodes")

    original_sites, original_multiplicities = np.unique(original_samples, return_counts=True)
    total_samples = int(original_samples.size)
    if total_samples < 2:
        raise ValueError("migration-associated sampling requires at least two samples")
    target_sites = min(max(int(original_sites.size), 2), int(eligible.size))
    if target_sites == original_multiplicities.size:
        multiplicities = np.asarray(original_multiplicities, dtype=np.int32)
    else:
        quotient, remainder = divmod(total_samples, target_sites)
        multiplicities = np.full(target_sites, quotient, dtype=np.int32)
        multiplicities[:remainder] += 1

    strength = float(
        rng.uniform(
            float(sim["sampling_association_min_strength"]),
            float(sim["sampling_association_max_strength"]),
        )
    )
    accessibility = eval_rff_params(
        draw_rff_params(
            rng,
            features_count=max(int(sim.get("gp_features", 64)), 32),
            length_scale=float(sim["sampling_spatial_length_scale"]),
        ),
        positions,
    )
    logits = float(sim["sampling_spatial_noise_scale"]) * accessibility
    if mode == "migration_positive":
        logits = logits + strength * migration_score
    elif mode == "migration_negative":
        logits = logits - strength * migration_score
    elif mode == "barrier_proximity":
        logits = logits + strength * node_barrier_proximity(
            num_nodes,
            edges,
            edge_log_rates,
            quantile=float(sim["sampling_barrier_quantile"]),
        )

    eligible_logits = logits[eligible]
    eligible_logits = eligible_logits - float(np.max(eligible_logits))
    probabilities = np.exp(np.clip(eligible_logits, -40.0, 0.0))
    probabilities /= float(np.sum(probabilities))
    selected_sites = np.asarray(
        rng.choice(
            eligible,
            size=target_sites,
            replace=False,
            p=probabilities,
        ),
        dtype=np.int32,
    )
    multiplicities = multiplicities[rng.permutation(multiplicities.shape[0])]

    samples = np.repeat(selected_sites, multiplicities).astype(np.int32)
    sample_count = np.bincount(samples, minlength=num_nodes)
    inclusion = (sample_count > 0).astype(np.float64)
    original_design = str(case.get("sample_design", ""))
    case.update(
        {
            "sample_nodes": samples,
            "sample_design": f"{original_design}+{mode}" if original_design else mode,
            "sampling_base_design": original_design,
            "sampling_association_mode": mode,
            "sampling_association_strength": strength,
            "sampling_spatial_noise_scale": float(sim["sampling_spatial_noise_scale"]),
            "sampling_inclusion_migration_correlation": sampling_correlation(
                inclusion[eligible], migration_score[eligible]
            ),
            "sampling_count_migration_correlation": sampling_correlation(
                sample_count[eligible], migration_score[eligible]
            ),
            "sampling_sampled_mean_migration": float(np.mean(migration_score[selected_sites])),
            "sampling_eligible_mean_migration": float(np.mean(migration_score[eligible])),
        }
    )
    return case


def choose_path_sample_sites(
    rng: np.random.Generator,
    positions: np.ndarray,
    edges: np.ndarray,
    sampled_sites: int,
) -> np.ndarray:
    num_nodes = int(positions.shape[0])
    sampled_sites = min(max(int(sampled_sites), 1), num_nodes)
    if sampled_sites >= num_nodes:
        return np.arange(num_nodes, dtype=np.int32)
    weights = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    graph = sp.csr_matrix(
        (
            np.r_[weights, weights],
            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
        ),
        shape=(num_nodes, num_nodes),
    )
    start = int(rng.integers(0, num_nodes))
    distances = shortest_path(graph, directed=False, indices=start, unweighted=False)
    finite = np.flatnonzero(np.isfinite(distances))
    if finite.shape[0] >= sampled_sites:
        far = finite[np.argsort(distances[finite])[-max(8, sampled_sites) :]]
        end = int(rng.choice(far))
        _distance, predecessors = shortest_path(
            graph,
            directed=False,
            indices=start,
            unweighted=False,
            return_predecessors=True,
        )
        path = reconstruct_shortest_path(predecessors, start, end)
        if path.shape[0] >= sampled_sites:
            selected = path[np.rint(np.linspace(0, path.shape[0] - 1, sampled_sites)).astype(int)]
            selected = np.unique(selected).astype(np.int32)
            if selected.shape[0] >= min(sampled_sites, 2):
                return selected

    coords = positions - positions.mean(axis=0, keepdims=True)
    direction = rng.normal(size=2)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    projection = coords @ direction
    chosen = []
    for value in np.quantile(projection, np.linspace(0.04, 0.96, sampled_sites)):
        jitter = rng.normal(scale=0.04 * max(float(np.ptp(projection)), 1.0))
        chosen.append(int(np.argmin(np.abs(projection - value - jitter))))
    chosen = list(dict.fromkeys(chosen))
    if len(chosen) < sampled_sites:
        remaining = [int(idx) for idx in np.argsort(projection) if int(idx) not in set(chosen)]
        chosen.extend(remaining[: sampled_sites - len(chosen)])
    return np.asarray(chosen[:sampled_sites], dtype=np.int32)


def reconstruct_shortest_path(predecessors: np.ndarray, start: int, end: int) -> np.ndarray:
    path = [int(end)]
    current = int(end)
    while current != int(start):
        current = int(predecessors[current])
        if current < 0:
            return np.asarray([], dtype=np.int32)
        path.append(current)
        if len(path) > predecessors.shape[0] + 1:
            return np.asarray([], dtype=np.int32)
    return np.asarray(path[::-1], dtype=np.int32)


def draw_feems_lattice_sample_nodes(
    rng: np.random.Generator,
    positions: np.ndarray,
    local: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    start = float(local["feems_lattice_barrier_startpt"])
    end = float(local["feems_lattice_barrier_endpt"])
    sample_prob = draw_weighted_positive_float(
        rng,
        choices=local.get("feems_lattice_sample_prob_choices"),
        weights=local.get("feems_lattice_sample_prob_weights"),
        fallback=1.0,
    )
    n_samples_per_node = max(int(local["feems_lattice_n_samples_per_node"]), 1)
    sample_nodes: list[int] = []
    node_diploids = np.zeros(positions.shape[0], dtype=np.int32)
    for node, (x, _y) in enumerate(positions):
        if x <= start:
            probability = float(local["feems_lattice_corridor_left_prob"])
        elif x >= end:
            probability = float(local["feems_lattice_corridor_right_prob"])
        else:
            probability = float(local["feems_lattice_barrier_prob"])
        probability = float(np.clip(probability, 0.0, 1.0))
        diploids = int(rng.binomial(n_samples_per_node, probability))
        if diploids > 0 and rng.random() > sample_prob:
            diploids = 0
        node_diploids[node] = diploids
        sample_nodes.extend([node] * diploids)
    sample_nodes_array = np.asarray(sample_nodes, dtype=np.int32)
    uncapped_samples = int(sample_nodes_array.shape[0])
    raw_sample_cap = local.get("max_samples")
    if raw_sample_cap is not None:
        sample_cap = max(int(raw_sample_cap), 2)
        if sample_nodes_array.shape[0] > sample_cap:
            keep = rng.choice(sample_nodes_array.shape[0], size=sample_cap, replace=False)
            sample_nodes_array = sample_nodes_array[keep]
            node_diploids = np.bincount(
                sample_nodes_array,
                minlength=positions.shape[0],
            ).astype(np.int32)
    else:
        sample_cap = None
    metadata = {
        "feems_lattice_sample_prob": float(sample_prob),
        "feems_lattice_n_samples_per_node": int(n_samples_per_node),
        "feems_lattice_uncapped_samples": uncapped_samples,
        "feems_lattice_sample_cap": sample_cap,
        "feems_lattice_samples_capped": bool(sample_nodes_array.shape[0] < uncapped_samples),
        "feems_lattice_barrier_prob": float(local["feems_lattice_barrier_prob"]),
        "feems_lattice_corridor_left_prob": float(local["feems_lattice_corridor_left_prob"]),
        "feems_lattice_corridor_right_prob": float(local["feems_lattice_corridor_right_prob"]),
        "feems_lattice_zero_sample_nodes": int(np.sum(node_diploids == 0)),
    }
    return sample_nodes_array, metadata


def draw_feems_lattice_case(
    rng: np.random.Generator,
    sim: Mapping[str, Any],
    local: Mapping[str, Any],
) -> dict[str, Any]:
    rows = draw_weighted_positive_int(
        rng,
        value=sim["feems_lattice_rows"],
        choices=sim.get("feems_lattice_row_choices"),
        weights=sim.get("feems_lattice_row_weights"),
        fallback=int(sim["feems_lattice_rows"]),
    )
    columns = draw_weighted_positive_int(
        rng,
        value=sim["feems_lattice_columns"],
        choices=sim.get("feems_lattice_column_choices"),
        weights=sim.get("feems_lattice_column_weights"),
        fallback=int(sim["feems_lattice_columns"]),
    )
    graph, positions = feems_triangular_lattice_graph(rows, columns)
    reference_columns = max(int(sim["feems_lattice_columns"]), 2)
    barrier_scale = (float(columns) - 0.5) / (float(reference_columns) - 0.5)
    case_local = dict(local)
    case_local["feems_lattice_barrier_startpt"] = (
        float(local["feems_lattice_barrier_startpt"]) * barrier_scale
    )
    case_local["feems_lattice_barrier_endpt"] = (
        float(local["feems_lattice_barrier_endpt"]) * barrier_scale
    )
    surface_kind = choose_weighted_name(
        rng,
        string_list(sim.get("feems_lattice_surface_kinds")),
        sim.get("feems_lattice_surface_weights"),
    )
    surface = draw_surface(rng, surface_kind=surface_kind, local=case_local)
    q_surface = draw_q_surface(rng, local)
    sample_nodes, sample_metadata = draw_feems_lattice_sample_nodes(rng, positions, case_local)
    return {
        "graph": graph,
        "active_positions": positions,
        "edge_log_rates": edge_log_rates_from_surface(surface, positions, graph["edges"]),
        "node_log_q": node_log_q_from_surface(q_surface, positions),
        "sample_nodes": sample_nodes,
        "surface_kind": str(surface_kind),
        "mask_kind": "feems_triangular_lattice",
        "graph_source": "feems_triangular_lattice",
        "height": int(rows),
        "width": int(columns),
        "sample_design": "feems_setup_graph",
        "sample_regime": (
            "dense_feems"
            if float(sample_metadata["feems_lattice_sample_prob"]) >= 0.999
            else "sparse_feems"
        ),
        "feems_lattice_training": True,
        "feems_lattice_rows": int(rows),
        "feems_lattice_columns": int(columns),
        "feems_lattice_barrier_startpt": float(case_local["feems_lattice_barrier_startpt"]),
        "feems_lattice_barrier_endpt": float(case_local["feems_lattice_barrier_endpt"]),
        "feems_lattice_barrier_w": float(local["feems_lattice_barrier_w"]),
        "feems_lattice_corridor_w": float(local["feems_lattice_corridor_w"]),
        "feems_lattice_anisotropy_scaler": float(local["feems_lattice_anisotropy_scaler"]),
        **sample_metadata,
        **local_simulation_metadata(local),
    }


def draw_grid_case(
    rng: np.random.Generator,
    sim: Mapping[str, Any],
    local: Mapping[str, Any],
    *,
    surface_kind: str,
    mask_kind: str | None = None,
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    if height is None or width is None:
        height, width = sample_grid_dimensions(rng, sim)
    if mask_kind is None:
        mask_kind = choose_weighted_name(
            rng,
            string_list(sim.get("mask_kinds"), default=["full"]),
            None,
        )
    mask = random_mask(rng, int(height), int(width), str(mask_kind))
    graph, positions = masked_grid_graph(mask)
    surface = draw_surface(rng, surface_kind=surface_kind, local=local)
    q_surface = draw_q_surface(rng, local)
    return {
        "graph": graph,
        "active_positions": positions,
        "edge_log_rates": edge_log_rates_from_surface(surface, positions, graph["edges"]),
        "node_log_q": node_log_q_from_surface(q_surface, positions),
        "sample_nodes": draw_sample_nodes_from_positions(rng, positions, local),
        "surface_kind": str(surface_kind),
        "mask_kind": str(mask_kind),
        "height": int(height),
        "width": int(width),
        "sample_design": str(local.get("sample_design", "")),
        "sample_regime": str(local.get("sample_regime", "")),
        **local_simulation_metadata(local),
    }


def draw_target_grid_dimensions(
    rng: np.random.Generator,
    target_nodes: int,
) -> tuple[int, int]:
    aspect = float(rng.uniform(0.65, 1.45))
    height = max(4, int(round(math.sqrt(target_nodes / aspect))))
    width = max(4, int(round(height * aspect)))
    return height, width


def draw_earth_like_case(
    rng: np.random.Generator,
    sim: Mapping[str, Any],
    local: Mapping[str, Any],
    *,
    surface_kind: str,
) -> dict[str, Any]:
    target_nodes = int(rng.integers(int(sim["earth_min_nodes"]), int(sim["earth_max_nodes"]) + 1))
    for _attempt in range(50):
        height, width = draw_target_grid_dimensions(rng, target_nodes)
        mask_kind = choose_weighted_name(rng, ["coast", "islands", "bay", "lake"], None)
        try:
            case = draw_grid_case(
                rng,
                sim,
                local,
                surface_kind=surface_kind,
                mask_kind=mask_kind,
                height=height,
                width=width,
            )
        except ValueError:
            continue
        if case["active_positions"].shape[0] >= max(20, int(0.35 * target_nodes)):
            case.update(
                {
                    "mask_kind": "earth-land",
                    "earth_backend": str(sim.get("earth_backend", "synthetic_grid")),
                    "earth_target_nodes": int(target_nodes),
                }
            )
            return case
    raise RuntimeError("could not build synthetic earth-like graph")


def draw_weighted_diploids(
    rng: np.random.Generator,
    *,
    low: int,
    high: int,
    weights: object,
) -> int:
    candidates = [str(value) for value in range(max(int(low), 1), max(int(high), 1) + 1)]
    return int(choose_weighted_name(rng, candidates, weights))


def global_landscape_environment(
    rng: np.random.Generator,
    positions: np.ndarray,
    lonlat: np.ndarray,
    land_mask: np.ndarray,
    land_xy: Any,
    local: Mapping[str, Any],
) -> dict[str, Any]:
    from shapely.geometry import Point

    boundary = land_xy.boundary
    coast_distance = np.asarray(
        [boundary.distance(Point(float(x), float(y))) for x, y in positions],
        dtype=np.float64,
    )
    extent = max(float(np.ptp(positions[:, 0])), float(np.ptp(positions[:, 1])), 1.0)
    coast_scale = float(rng.uniform(0.04, 0.16) * extent)
    coast_proximity = np.exp(-coast_distance / max(coast_scale, 1e-6))
    latitude = standardize(lonlat[:, 1])
    habitat_gp = eval_rff_params(
        draw_rff_params(
            rng,
            features_count=max(int(local["gp_features"]), 64),
            length_scale=float(rng.uniform(0.18, 0.48)),
        ),
        positions,
    )
    water_indicator = (~np.asarray(land_mask, dtype=bool)).astype(np.float64)
    coast_weight = float(rng.normal(0.0, 0.7))
    lat_weight = float(rng.normal(0.0, 0.45))
    water_weight = float(rng.uniform(0.8, 1.8))
    habitat = standardize(
        habitat_gp
        + coast_weight * standardize(coast_proximity)
        + lat_weight * latitude
        - water_weight * water_indicator
    )
    return {
        "coast_distance_km": coast_distance,
        "coast_proximity": standardize(coast_proximity),
        "habitat": habitat,
        "latitude": latitude,
        "coast_scale_km": coast_scale,
        "coast_weight": coast_weight,
        "lat_weight": lat_weight,
        "water_weight": water_weight,
    }


def global_landscape_ridge(
    rng: np.random.Generator,
    positions: np.ndarray,
    strength_scale: float,
) -> dict[str, Any]:
    center = np.mean(positions, axis=0)
    extent = max(float(np.ptp(positions[:, 0])), float(np.ptp(positions[:, 1])), 1.0)
    theta = float(rng.uniform(0.0, math.pi))
    along_vec = np.asarray([math.cos(theta), math.sin(theta)])
    normal_vec = np.asarray([-math.sin(theta), math.cos(theta)])
    amplitude = float(rng.uniform(0.02, 0.11) * extent)
    wavelength = float(rng.uniform(0.45, 1.30) * extent)
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    width = float(rng.uniform(0.018, 0.070) * extent)
    shifted = positions - center[None, :]
    along = shifted @ along_vec
    across = shifted @ normal_vec
    signed = across - amplitude * np.sin(2.0 * math.pi * along / max(wavelength, 1e-6) + phase)
    return {
        "signed": signed,
        "theta": theta,
        "amplitude_km": amplitude,
        "wavelength_km": wavelength,
        "phase": phase,
        "width_km": width,
        "strength": float(strength_scale * rng.uniform(0.55, 1.35)),
    }


def global_landscape_surfaces(
    rng: np.random.Generator,
    *,
    positions: np.ndarray,
    lonlat: np.ndarray,
    edges: np.ndarray,
    land_mask: np.ndarray,
    land_xy: Any,
    local: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    env = global_landscape_environment(rng, positions, lonlat, land_mask, land_xy, local)
    ridge = global_landscape_ridge(
        rng,
        positions,
        float(local.get("global_landscape_ridge_strength", 1.4)),
    )
    left = edges[:, 0]
    right = edges[:, 1]
    midpoint = 0.5 * (positions[left] + positions[right])
    midpoint_land = points_in_geometry(midpoint, land_xy)
    land_mask = np.asarray(land_mask, dtype=bool)
    endpoint_water = 1.0 - 0.5 * (
        land_mask[left].astype(np.float64) + land_mask[right].astype(np.float64)
    )
    water_penalty = np.maximum(endpoint_water, (~midpoint_land).astype(np.float64))

    migration_gp = eval_rff_params(
        draw_rff_params(
            rng,
            features_count=max(int(local["gp_features"]), 64),
            length_scale=float(local["migration_gp_length_scale"]),
        ),
        positions,
    )
    edge_gp = 0.5 * (migration_gp[left] + migration_gp[right])
    habitat = np.asarray(env["habitat"], dtype=np.float64)
    edge_habitat = 0.5 * (habitat[left] + habitat[right])
    habitat_mismatch = np.abs(habitat[left] - habitat[right])
    coast = np.asarray(env["coast_proximity"], dtype=np.float64)
    edge_coast = 0.5 * (coast[left] + coast[right])
    edge_lengths = np.linalg.norm(positions[left] - positions[right], axis=1)
    length_z = standardize(edge_lengths)
    signed = np.asarray(ridge["signed"], dtype=np.float64)
    crosses_ridge = signed[left] * signed[right] < 0.0
    min_abs_signed = np.minimum(np.abs(signed[left]), np.abs(signed[right]))
    near_ridge = np.exp(-0.5 * (min_abs_signed / max(float(ridge["width_km"]), 1e-6)) ** 2)
    ridge_penalty = np.where(crosses_ridge, 1.0, near_ridge)

    habitat_weight = float(rng.normal(0.55, 0.30))
    mismatch_weight = float(rng.uniform(0.10, 0.55))
    length_weight = float(rng.uniform(0.04, 0.20))
    water_strength = float(local.get("global_landscape_water_strength", 2.8))
    coast_strength = float(local.get("global_landscape_coast_strength", 1.2))
    values = (
        edge_gp
        + habitat_weight * edge_habitat
        - mismatch_weight * habitat_mismatch
        - water_strength * water_penalty
        - coast_strength * edge_coast
        - float(ridge["strength"]) * ridge_penalty
        - length_weight * length_z
    )
    edge_log_rates = (standardize(values) * float(local["log_rate_std"])).astype(np.float32)

    if float(local["q_log_std"]) <= 0.0 or str(local.get("q_prior", "gp")) == "constant":
        node_log_q = np.zeros(positions.shape[0], dtype=np.float32)
    else:
        q_gp = eval_rff_params(
            draw_rff_params(
                rng,
                features_count=max(int(local["gp_features"]), 64),
                length_scale=float(local["q_gp_length_scale"]),
            ),
            positions,
        )
        q_values = (
            q_gp
            + float(rng.normal(-0.45, 0.25)) * habitat
            + float(rng.normal(0.0, 0.35)) * coast
            + float(rng.uniform(0.20, 0.70)) * (~land_mask).astype(np.float64)
        )
        node_log_q = (standardize(q_values) * float(local["q_log_std"])).astype(np.float32)

    metadata = {
        "global_landscape_surface_model": (
            "GP + habitat + water/coast/ridge penalties on a random world-map crop"
        ),
        "global_landscape_coast_scale_km": float(env["coast_scale_km"]),
        "global_landscape_migration_habitat_weight": habitat_weight,
        "global_landscape_migration_mismatch_weight": mismatch_weight,
        "global_landscape_migration_length_weight": length_weight,
        "global_landscape_water_edge_fraction": float(np.mean(water_penalty > 0.0)),
        "global_landscape_midpoint_water_edge_fraction": float(np.mean(~midpoint_land)),
        "global_landscape_ridge_strength": float(ridge["strength"]),
        "global_landscape_ridge_width_km": float(ridge["width_km"]),
    }
    return edge_log_rates, node_log_q, metadata


def choose_global_landscape_sample_sites(
    rng: np.random.Generator,
    positions: np.ndarray,
    edges: np.ndarray,
    land_mask: np.ndarray,
    sampled_sites: int,
    *,
    path_probability: float = 1.0,
) -> np.ndarray:
    land_nodes = np.flatnonzero(np.asarray(land_mask, dtype=bool)).astype(np.int32)
    sampled_sites = min(max(int(sampled_sites), 1), land_nodes.shape[0])
    if sampled_sites >= land_nodes.shape[0]:
        return land_nodes

    old_to_land = -np.ones(positions.shape[0], dtype=np.int32)
    old_to_land[land_nodes] = np.arange(land_nodes.shape[0], dtype=np.int32)
    land_edge_mask = land_mask[edges[:, 0]] & land_mask[edges[:, 1]]
    land_edges = old_to_land[edges[land_edge_mask]]
    path_probability = float(np.clip(path_probability, 0.0, 1.0))
    if land_edges.shape[0] >= max(sampled_sites - 1, 1) and rng.random() < path_probability:
        try:
            selected = choose_path_sample_sites(
                rng,
                positions[land_nodes],
                land_edges.astype(np.int32, copy=False),
                sampled_sites,
            )
            if selected.shape[0] >= min(sampled_sites, 2):
                return land_nodes[selected[:sampled_sites]]
        except Exception:  # noqa: BLE001
            pass

    if rng.random() < 0.35:
        return rng.choice(land_nodes, size=sampled_sites, replace=False).astype(np.int32)

    coords = positions[land_nodes] - np.mean(positions[land_nodes], axis=0, keepdims=True)
    direction = rng.normal(size=2)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    projection = coords @ direction
    chosen: list[int] = []
    for value in np.quantile(projection, np.linspace(0.03, 0.97, sampled_sites)):
        jitter = rng.normal(scale=0.035 * max(float(np.ptp(projection)), 1.0))
        chosen.append(int(np.argmin(np.abs(projection - value - jitter))))
    chosen = list(dict.fromkeys(chosen))
    if len(chosen) < sampled_sites:
        remaining = [idx for idx in np.argsort(projection) if int(idx) not in set(chosen)]
        chosen.extend(int(idx) for idx in remaining[: sampled_sites - len(chosen)])
    return land_nodes[np.asarray(chosen[:sampled_sites], dtype=np.int32)]


def draw_global_landscape_case(
    rng: np.random.Generator,
    sim: Mapping[str, Any],
    local: Mapping[str, Any],
) -> dict[str, Any]:
    region = build_global_landscape_mesh(rng, sim)
    positions = np.asarray(region["positions"], dtype=np.float64)
    lonlat = np.asarray(region["lonlat"], dtype=np.float64)
    edges = np.asarray(region["edges"], dtype=np.int32)
    land_mask = np.asarray(region["land_mask"], dtype=bool)
    graph = graph_from_positions_edges(positions, edges)[0]

    surface_kinds = string_list(
        sim.get("global_landscape_surface_kinds"),
        default=["global_landscape"],
    )
    surface_kind = choose_weighted_name(
        rng,
        surface_kinds,
        sim.get("global_landscape_surface_weights"),
    )
    if surface_kind == "global_landscape":
        edge_log_rates, node_log_q, surface_metadata = global_landscape_surfaces(
            rng,
            positions=positions,
            lonlat=lonlat,
            edges=edges,
            land_mask=land_mask,
            land_xy=region["land_xy"],
            local=local,
        )
    else:
        surface = draw_surface(rng, surface_kind=surface_kind, local=local)
        q_surface = draw_q_surface(rng, local)
        edge_log_rates = edge_log_rates_from_surface(surface, positions, edges)
        node_log_q = node_log_q_from_surface(q_surface, positions)
        surface_metadata = {"global_landscape_surface_model": f"generic:{surface_kind}"}

    diploids = draw_weighted_diploids(
        rng,
        low=int(sim["global_landscape_min_diploids_per_site"]),
        high=int(sim["global_landscape_max_diploids_per_site"]),
        weights=sim.get("global_landscape_diploid_weights"),
    )
    land_nodes = int(land_mask.sum())
    min_sites = max(int(sim["global_landscape_min_sampled_sites"]), 1)
    max_sites_raw = sim.get("global_landscape_max_sampled_sites")
    max_sites = land_nodes if max_sites_raw is None else max(int(max_sites_raw), min_sites)
    max_sites = min(max_sites, land_nodes)
    max_total_samples = sim.get("global_landscape_max_total_samples")
    if max_total_samples is not None:
        max_sites = min(max_sites, max(int(max_total_samples) // max(diploids, 1), min_sites))
    if sim.get("global_landscape_sampled_fraction_choices") is not None:
        requested_fraction = min(
            draw_weighted_positive_float(
                rng,
                choices=sim.get("global_landscape_sampled_fraction_choices"),
                weights=sim.get("global_landscape_sampled_fraction_weights"),
                fallback=float(sim["global_landscape_sampled_sites"]) / max(float(land_nodes), 1.0),
            ),
            1.0,
        )
        sampled_sites = int(round(requested_fraction * land_nodes))
    else:
        sampled_sites = draw_weighted_positive_int(
            rng,
            value=sim["global_landscape_sampled_sites"],
            choices=sim.get("global_landscape_sampled_site_choices"),
            weights=sim.get("global_landscape_sampled_site_weights"),
            fallback=int(sim["global_landscape_sampled_sites"]),
        )
        requested_fraction = float(sampled_sites) / max(float(land_nodes), 1.0)
    sampled_sites = min(max(sampled_sites, min_sites), max_sites)
    sample_sites = choose_global_landscape_sample_sites(
        rng,
        positions,
        edges,
        land_mask,
        sampled_sites,
        path_probability=float(sim.get("global_landscape_path_sample_probability", 1.0)),
    )
    sample_nodes = np.repeat(sample_sites, diploids).astype(np.int32)
    bbox = region["bbox"]
    return {
        "graph": graph,
        "active_positions": positions,
        "node_lonlat": lonlat,
        "node_land_mask": land_mask,
        "edge_log_rates": edge_log_rates,
        "node_log_q": node_log_q,
        "sample_nodes": sample_nodes,
        "surface_kind": str(surface_kind),
        "height": 0,
        "width": 0,
        "mask_kind": "global_landscape_world_map",
        "graph_source": "global_landscape",
        "sample_design": "global_landscape_land_sites",
        "sample_regime": "global_landscape_land_sites",
        "diploids_per_site": int(diploids),
        "global_landscape_training": True,
        "global_landscape_resolution": str(sim["global_landscape_resolution"]),
        "global_landscape_center_lon": float(region["center_lon"]),
        "global_landscape_center_lat": float(region["center_lat"]),
        "global_landscape_bbox_lon_min": float(bbox.bounds[0]),
        "global_landscape_bbox_lat_min": float(bbox.bounds[1]),
        "global_landscape_bbox_lon_max": float(bbox.bounds[2]),
        "global_landscape_bbox_lat_max": float(bbox.bounds[3]),
        "global_landscape_box_width_deg": float(region["box_width_deg"]),
        "global_landscape_box_height_deg": float(region["box_height_deg"]),
        "global_landscape_land_fraction": float(region["land_fraction"]),
        "global_landscape_land_nodes": land_nodes,
        "global_landscape_water_nodes": int((~land_mask).sum()),
        "global_landscape_spacing_km": float(region["spacing_km"]),
        "global_landscape_sampled_land_fraction": float(sample_sites.shape[0] / land_nodes),
        "global_landscape_sampled_node_fraction": float(sample_sites.shape[0] / positions.shape[0]),
        "global_landscape_requested_sampled_fraction": float(requested_fraction),
        **surface_metadata,
        **local_simulation_metadata(local),
    }


def draw_known_graph_case(
    rng: np.random.Generator,
    sim: Mapping[str, Any],
    local: Mapping[str, Any],
    *,
    graph: dict[str, np.ndarray],
    positions: np.ndarray,
    sample_sites: np.ndarray,
    surface_kind: str,
    diploids_per_site: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    surface = draw_surface(rng, surface_kind=surface_kind, local=local)
    q_surface = draw_q_surface(rng, local)
    sample_nodes = np.repeat(
        np.asarray(sample_sites, dtype=np.int32),
        max(int(diploids_per_site), 1),
    )
    return {
        "graph": graph,
        "active_positions": positions,
        "edge_log_rates": edge_log_rates_from_surface(surface, positions, graph["edges"]),
        "node_log_q": node_log_q_from_surface(q_surface, positions),
        "sample_nodes": sample_nodes.astype(np.int32),
        "surface_kind": str(surface_kind),
        "height": 0,
        "width": 0,
        "sample_design": str(metadata.get("sample_design", local.get("sample_design", ""))),
        "sample_regime": str(metadata.get("sample_regime", local.get("sample_regime", ""))),
        "diploids_per_site": int(diploids_per_site),
        **dict(metadata),
        **local_simulation_metadata(local),
    }


def draw_wolflike_case(
    rng: np.random.Generator,
    sim: Mapping[str, Any],
    local: Mapping[str, Any],
    *,
    surface_kinds: Sequence[str],
) -> dict[str, Any]:
    source = choose_weighted_name(
        rng,
        string_list(sim.get("wolflike_graph_sources"), default=["synthetic-corridor", "land"]),
        sim.get("wolflike_graph_source_weights"),
    )
    target_nodes = draw_weighted_positive_int(
        rng,
        value=sim["wolflike_target_nodes"],
        choices=sim.get("wolflike_target_node_choices"),
        weights=sim.get("wolflike_target_node_weights"),
        fallback=240,
    )
    mask_kind = "corridor" if source in {"corridor", "synthetic-corridor"} else "coast"
    grid_target_nodes = int(round(target_nodes * (2.5 if mask_kind == "corridor" else 1.5)))
    height, width = draw_target_grid_dimensions(rng, grid_target_nodes)
    for _attempt in range(50):
        try:
            mask = random_mask(rng, height, width, mask_kind)
            graph, positions = masked_grid_graph(mask)
        except ValueError:
            height, width = draw_target_grid_dimensions(rng, grid_target_nodes)
            continue
        if positions.shape[0] >= max(20, int(0.35 * target_nodes)):
            break
        height, width = draw_target_grid_dimensions(rng, grid_target_nodes)
    else:
        raise RuntimeError("could not build wolflike graph")

    diploids = draw_weighted_diploids(
        rng,
        low=int(sim["wolflike_min_diploids_per_site"]),
        high=int(sim["wolflike_max_diploids_per_site"]),
        weights=sim.get("wolflike_diploid_weights"),
    )
    min_sites = max(int(sim["wolflike_min_sampled_sites"]), 1)
    max_sites_raw = sim.get("wolflike_max_sampled_sites")
    max_sites = positions.shape[0] if max_sites_raw is None else max(int(max_sites_raw), min_sites)
    max_sites = min(max_sites, positions.shape[0])
    max_total_samples = sim.get("wolflike_max_total_samples")
    if max_total_samples is not None:
        max_sites = min(max_sites, max(int(max_total_samples) // max(diploids, 1), min_sites))
    if sim.get("wolflike_sampled_fraction_choices") is not None:
        fraction = min(
            draw_weighted_positive_float(
                rng,
                choices=sim.get("wolflike_sampled_fraction_choices"),
                weights=sim.get("wolflike_sampled_fraction_weights"),
                fallback=float(sim["wolflike_sampled_sites"]) / max(float(target_nodes), 1.0),
            ),
            1.0,
        )
        sampled_sites = int(round(fraction * positions.shape[0]))
    else:
        sampled_sites = draw_weighted_positive_int(
            rng,
            value=sim["wolflike_sampled_sites"],
            choices=sim.get("wolflike_sampled_site_choices"),
            weights=sim.get("wolflike_sampled_site_weights"),
            fallback=15,
        )
        fraction = float(sampled_sites) / max(float(positions.shape[0]), 1.0)
    sampled_sites = min(max(sampled_sites, min_sites), max_sites)
    sample_sites = choose_path_sample_sites(rng, positions, graph["edges"], sampled_sites)
    wolflike_surfaces = string_list(sim.get("wolflike_surface_kinds"), default=surface_kinds)
    if source != "land":
        wolflike_surfaces = [kind for kind in wolflike_surfaces if kind != "earth"] or [
            "smooth",
            "barrier",
            "gp",
        ]
    surface_kind = choose_weighted_name(
        rng,
        wolflike_surfaces,
        sim.get("wolflike_surface_weights"),
    )
    return draw_known_graph_case(
        rng,
        sim,
        local,
        graph=graph,
        positions=positions,
        sample_sites=sample_sites,
        surface_kind=surface_kind,
        diploids_per_site=diploids,
        metadata={
            "graph_source": "wolflike",
            "mask_kind": f"wolflike_{source}",
            "sample_design": "wolflike_path",
            "sample_regime": "wolflike_path",
            "wolflike_graph_training": True,
            "wolflike_graph_source": str(source),
            "wolflike_sampled_node_fraction": float(sample_sites.shape[0] / positions.shape[0]),
            "wolflike_requested_sampled_fraction": float(fraction),
            "wolflike_target_nodes": int(target_nodes),
        },
    )
