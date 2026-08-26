"""Validated synthetic-corpus configuration and random mixture draws."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def simulation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "seed": None,
        "simulator": "auto",
        "min_height": 10,
        "max_height": 18,
        "min_width": 10,
        "max_width": 18,
        "case_weights": {"grid": 1.0},
        "earth_min_nodes": 80,
        "earth_max_nodes": 900,
        "earth_backend": "synthetic_grid",
        "global_landscape_resolution": "50m",
        "global_landscape_cartopy_data_dir": "artifacts/cartopy",
        "global_landscape_min_nodes": 300,
        "global_landscape_max_nodes": 1800,
        "global_landscape_target_nodes": 900,
        "global_landscape_target_node_choices": None,
        "global_landscape_target_node_weights": None,
        "global_landscape_min_box_deg": 3.0,
        "global_landscape_max_box_deg": 24.0,
        "global_landscape_box_width_choices": None,
        "global_landscape_box_width_weights": None,
        "global_landscape_box_aspect_min": 0.65,
        "global_landscape_box_aspect_max": 1.35,
        "global_landscape_min_land_fraction": 0.05,
        "global_landscape_min_lat": -55.0,
        "global_landscape_max_lat": 72.0,
        "global_landscape_max_region_tries": 500,
        "global_landscape_preferred_center_lon": None,
        "global_landscape_preferred_center_lat": None,
        "global_landscape_center_jitter_deg": 0.0,
        "global_landscape_surface_kinds": ["global_landscape"],
        "global_landscape_surface_weights": None,
        "global_landscape_sampled_fraction_choices": [0.03, 0.05, 0.10, 0.25, 0.50],
        "global_landscape_sampled_fraction_weights": {
            "0.03": 2.0,
            "0.05": 2.0,
            "0.1": 2.0,
            "0.25": 1.0,
            "0.5": 0.5,
        },
        "global_landscape_sampled_sites": 25,
        "global_landscape_sampled_site_choices": None,
        "global_landscape_sampled_site_weights": None,
        "global_landscape_min_sampled_sites": 2,
        "global_landscape_max_sampled_sites": None,
        "global_landscape_max_total_samples": None,
        "global_landscape_path_sample_probability": 1.0,
        "global_landscape_min_diploids_per_site": 1,
        "global_landscape_max_diploids_per_site": 16,
        "global_landscape_diploid_weights": None,
        "global_landscape_water_strength": 2.8,
        "global_landscape_coast_strength": 1.2,
        "global_landscape_ridge_strength": 1.4,
        "feems_lattice_rows": 8,
        "feems_lattice_columns": 12,
        "feems_lattice_row_choices": None,
        "feems_lattice_column_choices": None,
        "feems_lattice_row_weights": None,
        "feems_lattice_column_weights": None,
        "feems_lattice_surface_kinds": [
            "feems_homogeneous",
            "feems_heterogeneous_barrier",
            "feems_anisotropic",
        ],
        "feems_lattice_surface_weights": None,
        "feems_lattice_barrier_startpt": 2.5,
        "feems_lattice_barrier_endpt": 8.5,
        "feems_lattice_barrier_w": 0.1,
        "feems_lattice_corridor_w": 0.5,
        "feems_lattice_anisotropy_scaler": 5.0,
        "feems_lattice_n_samples_per_node": 10,
        "feems_lattice_sample_prob_choices": [1.0, 0.2],
        "feems_lattice_sample_prob_weights": None,
        "feems_lattice_barrier_prob": 1.0,
        "feems_lattice_corridor_left_prob": 1.0,
        "feems_lattice_corridor_right_prob": 1.0,
        "mask_kinds": ["full"],
        "wolflike_graph_sources": ["synthetic-corridor", "land"],
        "wolflike_graph_source_weights": None,
        "wolflike_target_nodes": 240,
        "wolflike_target_node_choices": None,
        "wolflike_target_node_weights": None,
        "wolflike_sampled_sites": 15,
        "wolflike_sampled_site_choices": None,
        "wolflike_sampled_site_weights": None,
        "wolflike_sampled_fraction_choices": None,
        "wolflike_sampled_fraction_weights": None,
        "wolflike_min_sampled_sites": 2,
        "wolflike_max_sampled_sites": None,
        "wolflike_max_total_samples": None,
        "wolflike_min_diploids_per_site": 1,
        "wolflike_max_diploids_per_site": 16,
        "wolflike_diploid_weights": None,
        "wolflike_surface_kinds": None,
        "wolflike_surface_weights": None,
        "num_snps": 4000,
        "min_num_snps": None,
        "max_num_snps": None,
        "num_snps_log_bins": None,
        "sample_designs": ["sites"],
        "sample_regimes": None,
        "sampling_counterfactual_modes": None,
        "sampling_counterfactual_group_splits": ["train", "test"],
        "sampling_association_min_strength": 0.75,
        "sampling_association_max_strength": 2.50,
        "sampling_spatial_noise_scale": 0.75,
        "sampling_spatial_length_scale": 0.35,
        "sampling_barrier_quantile": 0.20,
        "min_samples": 8,
        "max_samples": 45,
        "min_sample_fraction": None,
        "max_sample_fraction": None,
        "sampled_node_fraction": 0.25,
        "min_sampled_nodes": 6,
        "max_sampled_nodes": None,
        "diploids_per_node": 2,
        "min_diploids_per_node": None,
        "max_diploids_per_node": None,
        "min_diploids_per_site": None,
        "max_diploids_per_site": None,
        "surface_kinds": ["smooth", "barrier", "corridor"],
        "surface_weights": None,
        "landscape_min_features": 3,
        "landscape_max_features": 6,
        "landscape_water_strength": 2.7,
        "landscape_mountain_strength": 1.8,
        "landscape_coast_strength": 1.5,
        "landscape_feature_jitter": 0.06,
        "log_rate_std": 1.0,
        "min_log_rate_std": None,
        "max_log_rate_std": None,
        "q_log_std": 0.35,
        "min_q_log_std": None,
        "max_q_log_std": None,
        "q_prior": "gp",
        "gp_features": 64,
        "migration_gp_length_scale": 0.35,
        "min_migration_gp_length_scale": None,
        "max_migration_gp_length_scale": None,
        "q_gp_length_scale": 0.35,
        "min_q_gp_length_scale": None,
        "max_q_gp_length_scale": None,
        "msprime_max_nodes": 20,
        "coalescent_migration_scale": 0.5,
        "coalescent_effective_size": 1.0,
        "coalescent_mutation_rate": 0.01,
        "coalescent_sequence_length": 10.0,
        "coalescent_batch_replicates": 128,
        "coalescent_max_batches": 64,
        "island_ploidy": 2,
        "island_mutation_rate": 0.01,
        "island_migration_scale": 1.0,
        "island_coalescence_scale": 1.0,
        "island_symmetric_migration": True,
        "island_max_attempts": None,
        "island_use_rust": True,
        "island_require_rust": False,
        "island_emit_covariance": False,
        "island_min_allele_frequency": 0.05,
        "island_max_allele_frequency": 0.95,
        "island_marker_model": "conditioned_random_marker",
        "max_conditionable_attempts": 32,
    }
    user_value = config.get("simulation", {})
    if user_value is None:
        user_value = {}
    if not isinstance(user_value, dict):
        raise ValueError("expected mapping for config key 'simulation'")
    user = dict(user_value)
    unknown = sorted(set(user) - set(defaults))
    if unknown:
        raise ValueError(f"unsupported simulation config keys: {', '.join(unknown)}")
    defaults.update(user)
    modes = string_list(defaults["sampling_counterfactual_modes"])
    supported_modes = {
        "independent",
        "migration_positive",
        "migration_negative",
        "barrier_proximity",
    }
    unsupported_modes = sorted(set(modes) - supported_modes)
    if unsupported_modes:
        raise ValueError(
            "unsupported sampling counterfactual modes: " + ", ".join(unsupported_modes)
        )
    grouped_splits = string_list(defaults["sampling_counterfactual_group_splits"])
    unsupported_splits = sorted(set(grouped_splits) - {"train", "test"})
    if unsupported_splits:
        raise ValueError(
            "unsupported sampling counterfactual group splits: " + ", ".join(unsupported_splits)
        )
    min_strength = float(defaults["sampling_association_min_strength"])
    max_strength = float(defaults["sampling_association_max_strength"])
    if min_strength < 0.0 or max_strength < min_strength:
        raise ValueError("sampling association strengths must satisfy 0 <= min <= max")
    if float(defaults["sampling_spatial_noise_scale"]) < 0.0:
        raise ValueError("sampling_spatial_noise_scale must be nonnegative")
    if float(defaults["sampling_spatial_length_scale"]) <= 0.0:
        raise ValueError("sampling_spatial_length_scale must be positive")
    barrier_quantile = float(defaults["sampling_barrier_quantile"])
    if not 0.0 < barrier_quantile < 1.0:
        raise ValueError("sampling_barrier_quantile must lie strictly between zero and one")
    raw_snp_bins = defaults["num_snps_log_bins"]
    if raw_snp_bins is not None:
        if isinstance(raw_snp_bins, (str, bytes)) or not isinstance(raw_snp_bins, Sequence):
            raise ValueError("simulation.num_snps_log_bins must be a sequence of [low, high] pairs")
        snp_bins: list[list[int]] = []
        for raw_bin in raw_snp_bins:
            if isinstance(raw_bin, (str, bytes)) or not isinstance(raw_bin, Sequence):
                raise ValueError(
                    "simulation.num_snps_log_bins must be a sequence of [low, high] pairs"
                )
            if len(raw_bin) != 2:
                raise ValueError(
                    "simulation.num_snps_log_bins must be a sequence of [low, high] pairs"
                )
            low, high = (int(raw_bin[0]), int(raw_bin[1]))
            if low <= 0 or high < low:
                raise ValueError("SNP bins must satisfy 0 < low <= high")
            snp_bins.append([low, high])
        if not snp_bins:
            raise ValueError("simulation.num_snps_log_bins cannot be empty")
        defaults["num_snps_log_bins"] = snp_bins
    return defaults


def selected_simulator(sim: Mapping[str, Any], num_nodes: int) -> str:
    simulator = str(sim.get("simulator", "auto"))
    aliases = {
        "coalescent": "msprime_coalescent",
        "msprime": "msprime_coalescent",
        "island": "island_coalescent",
    }
    simulator = aliases.get(simulator, simulator)
    if simulator == "auto":
        if num_nodes <= int(sim["msprime_max_nodes"]):
            return "msprime_coalescent"
        return "island_coalescent"
    if simulator not in {
        "msprime_coalescent",
        "island_coalescent",
    }:
        raise ValueError(
            "simulation.simulator must be one of auto, msprime_coalescent, or island_coalescent"
        )
    return simulator


def string_list(value: object, *, default: Sequence[str] = ()) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [piece for chunk in value.split(",") for piece in chunk.split() if piece]
    return [str(item) for item in value]


def parse_weight_mapping(value: object) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        raw_items = value.items()
    else:
        raw_items = []
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            if isinstance(item, Mapping):
                key = item.get("name", item.get("key", item.get("value")))
                weight = item.get("weight", 1.0)
            else:
                text = str(item)
                if "=" not in text:
                    continue
                key, weight = text.split("=", 1)
            raw_items.append((key, weight))
    weights: dict[str, float] = {}
    for key, weight in raw_items:
        try:
            numeric = float(weight)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            weights[str(key)] = numeric
    return weights


def choose_weighted_name(
    rng: np.random.Generator,
    names: Sequence[str],
    weight_config: object,
) -> str:
    candidates = [str(name) for name in names]
    if not candidates:
        raise ValueError("at least one candidate is required")
    weights_map = parse_weight_mapping(weight_config)
    if not weights_map:
        return candidates[int(rng.integers(0, len(candidates)))]
    weights = np.asarray([weights_map.get(name, 1.0) for name in candidates], dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(candidates), dtype=np.float64)
    weights /= float(weights.sum())
    return candidates[int(rng.choice(len(candidates), p=weights))]


CASE_KINDS = ("grid", "earth_like", "wolflike", "global_landscape", "feems_lattice")


def choose_case_kind(rng: np.random.Generator, value: object) -> str:
    weights = parse_weight_mapping(value)
    unknown = sorted(set(weights) - set(CASE_KINDS))
    if unknown:
        raise ValueError(f"unsupported simulation case kinds: {', '.join(unknown)}")
    names = [name for name in CASE_KINDS if weights.get(name, 0.0) > 0.0]
    if not names:
        raise ValueError("simulation.case_weights must assign positive weight to a case kind")
    probabilities = np.asarray([weights[name] for name in names], dtype=np.float64)
    probabilities /= probabilities.sum()
    return names[int(rng.choice(len(names), p=probabilities))]


def draw_float_range(
    rng: np.random.Generator,
    default_value: float,
    min_value: object,
    max_value: object,
) -> float:
    if min_value is None or max_value is None:
        return float(default_value)
    low = min(float(min_value), float(max_value))
    high = max(float(min_value), float(max_value))
    return float(rng.uniform(low, high))


def draw_num_snps(rng: np.random.Generator, sim: Mapping[str, Any]) -> int:
    bins = sim.get("num_snps_log_bins")
    if bins:
        position = float(rng.random()) * len(bins)
        bin_index = min(int(position), len(bins) - 1)
        within_bin = position - bin_index
        low, high = bins[bin_index]
        low = int(low)
        high = int(high)
        if low == high:
            return low
        log_value = np.log(low) + within_bin * (np.log(high + 1) - np.log(low))
        value = int(np.floor(np.exp(log_value)))
        return int(np.clip(value, low, high))
    if sim.get("min_num_snps") is not None and sim.get("max_num_snps") is not None:
        low = min(int(sim["min_num_snps"]), int(sim["max_num_snps"]))
        high = max(int(sim["min_num_snps"]), int(sim["max_num_snps"]))
        return int(rng.integers(low, high + 1))
    return int(sim["num_snps"])


def local_simulation_parameters(
    rng: np.random.Generator,
    sim: Mapping[str, Any],
) -> dict[str, Any]:
    local = dict(sim)
    local["log_rate_std"] = draw_float_range(
        rng,
        float(sim["log_rate_std"]),
        sim.get("min_log_rate_std"),
        sim.get("max_log_rate_std"),
    )
    local["q_log_std"] = draw_float_range(
        rng,
        float(sim["q_log_std"]),
        sim.get("min_q_log_std"),
        sim.get("max_q_log_std"),
    )
    local["migration_gp_length_scale"] = draw_float_range(
        rng,
        float(sim["migration_gp_length_scale"]),
        sim.get("min_migration_gp_length_scale"),
        sim.get("max_migration_gp_length_scale"),
    )
    local["q_gp_length_scale"] = draw_float_range(
        rng,
        float(sim["q_gp_length_scale"]),
        sim.get("min_q_gp_length_scale"),
        sim.get("max_q_gp_length_scale"),
    )
    local["diploids_per_node"] = draw_local_diploids_per_node(rng, local)
    return local


def draw_weighted_positive_int(
    rng: np.random.Generator,
    *,
    value: object,
    choices: object,
    weights: object,
    fallback: int,
) -> int:
    if choices is None:
        try:
            return max(int(value), 1)
        except (TypeError, ValueError):
            return max(int(fallback), 1)
    candidates = []
    values = choices if isinstance(choices, (list, tuple)) else [choices]
    for raw in values:
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            candidates.append(str(parsed))
    if not candidates:
        return max(int(fallback), 1)
    return int(choose_weighted_name(rng, candidates, weights))


def draw_weighted_positive_float(
    rng: np.random.Generator,
    *,
    choices: object,
    weights: object,
    fallback: float,
) -> float:
    if choices is None:
        return max(float(fallback), 0.0)
    candidates = []
    values = choices if isinstance(choices, (list, tuple)) else [choices]
    for raw in values:
        try:
            parsed = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(parsed) and parsed > 0.0:
            candidates.append(str(parsed))
    if not candidates:
        return max(float(fallback), 0.0)
    return float(choose_weighted_name(rng, candidates, weights))


def draw_local_diploids_per_node(
    rng: np.random.Generator,
    sim: Mapping[str, Any],
) -> int:
    if (
        sim.get("min_diploids_per_site") is not None
        and sim.get("max_diploids_per_site") is not None
    ):
        low = min(int(sim["min_diploids_per_site"]), int(sim["max_diploids_per_site"]))
        high = max(int(sim["min_diploids_per_site"]), int(sim["max_diploids_per_site"]))
        return int(rng.integers(low, high + 1))
    if (
        sim.get("min_diploids_per_node") is not None
        and sim.get("max_diploids_per_node") is not None
    ):
        low = min(int(sim["min_diploids_per_node"]), int(sim["max_diploids_per_node"]))
        high = max(int(sim["min_diploids_per_node"]), int(sim["max_diploids_per_node"]))
        return int(rng.integers(low, high + 1))
    return max(int(sim["diploids_per_node"]), 1)


def local_simulation_metadata(local: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "local_log_rate_std": float(local["log_rate_std"]),
        "local_q_log_std": float(local["q_log_std"]),
        "local_migration_gp_length_scale": float(local["migration_gp_length_scale"]),
        "local_q_gp_length_scale": float(local["q_gp_length_scale"]),
        "local_diploids_per_node": int(local["diploids_per_node"]),
        "local_min_samples": int(local.get("min_samples", 0) or 0),
        "local_max_samples": int(local.get("max_samples", 0) or 0),
        "local_min_sample_fraction": (
            None
            if local.get("min_sample_fraction") is None
            else float(local["min_sample_fraction"])
        ),
        "local_max_sample_fraction": (
            None
            if local.get("max_sample_fraction") is None
            else float(local["max_sample_fraction"])
        ),
    }
