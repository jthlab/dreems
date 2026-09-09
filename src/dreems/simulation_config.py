"""Validate simulation settings and draw from the configured mixtures."""

from __future__ import annotations

# ruff: noqa: F722, F821, UP037
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np
from beartype import beartype
from jaxtyping import Float, Shaped, jaxtyped

from dreems.validation import _integer_array


@beartype
def _integer_setting(value: object, name: str, minimum: int = 1) -> int:
    """Read a scalar count without truncation, clipping, or a fallback on error."""
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError as error:
            raise ValueError(f"{name} must be an integer") from error
    array = _integer_array(np.asarray(value), name=name, dtype=np.int64)
    if array.ndim != 0 or array.item() < minimum:
        raise ValueError(f"{name} must be a scalar integer >= {minimum}")
    return array.item()


@beartype
def _finite_setting(value: object, name: str) -> float:
    """Read a finite scalar number, rejecting booleans."""
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite scalar number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite scalar number") from error
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@beartype
def _positive_choices(choices: object, *, integer: bool) -> list[str]:
    """Use the same numeric spelling as the existing named-weight lookup."""
    if isinstance(choices, np.ndarray):
        if choices.ndim != 1:
            raise ValueError("choices must be one-dimensional")
        values = list(choices)
    else:
        values = choices if isinstance(choices, (list, tuple)) else [choices]
    if not values:
        raise ValueError("choices must be a nonempty one-dimensional sequence")
    convert = _integer_setting if integer else _finite_setting
    numbers = [convert(value, "choice") for value in values]
    if any(number <= 0 for number in numbers):
        raise ValueError("choices must be positive")
    return [str(number) for number in numbers]


@jaxtyped(typechecker=beartype)
def _weight_probabilities(
    weights: Shaped[np.ndarray, "*weight_axes"],
) -> Float[np.ndarray, "*weight_axes"]:
    """Normalize finite, nonnegative weights with a positive total."""
    total = weights.sum()
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or not np.isfinite(total)
        or total <= 0
    ):
        raise ValueError("weights must be finite, nonnegative, and have a positive finite sum")
    return weights / total


@beartype
def _integer_bounds(low: object, high: object, name: str) -> tuple[int, int] | None:
    """Require both optional bounds, retaining the existing endpoint-order convention."""
    if low is None and high is None:
        return None
    if low is None or high is None:
        raise ValueError(f"{name} requires both lower and upper bounds")
    values = (_integer_setting(low, name), _integer_setting(high, name))
    return min(values), max(values)


@beartype
def simulation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply simulation defaults without modifying the caller's config.

    The seed comes from ``simulation.seed``, then top-level ``seed``, then 12001.
    Explicit null seeds use the same fallback as omitted seeds. The remaining
    settings describe the mixture of graphs, surfaces, and sampling designs.
    """
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
    seed = defaults["seed"]
    if seed is None:
        seed = config.get("seed")
    if seed is None:
        seed = 12001
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("simulation seed must be a nonnegative integer")
    defaults["seed"] = int(seed)
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
    if (
        not np.isfinite([min_strength, max_strength]).all()
        or min_strength < 0.0
        or max_strength < min_strength
    ):
        raise ValueError("sampling association strengths must satisfy 0 <= min <= max")
    noise_scale = float(defaults["sampling_spatial_noise_scale"])
    if not np.isfinite(noise_scale) or noise_scale < 0.0:
        raise ValueError("sampling_spatial_noise_scale must be finite and nonnegative")
    length_scale = float(defaults["sampling_spatial_length_scale"])
    if not np.isfinite(length_scale) or length_scale <= 0.0:
        raise ValueError("sampling_spatial_length_scale must be finite and positive")
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
            if any(isinstance(bound, bool) or not isinstance(bound, Integral) for bound in raw_bin):
                raise ValueError("SNP bins must contain integer bounds")
            low, high = (int(raw_bin[0]), int(raw_bin[1]))
            if low <= 0 or high < low:
                raise ValueError("SNP bins must satisfy 0 < low <= high")
            snp_bins.append([low, high])
        if not snp_bins:
            raise ValueError("simulation.num_snps_log_bins cannot be empty")
        defaults["num_snps_log_bins"] = snp_bins
    validate_simulation_settings(defaults)
    return defaults


@beartype
def validate_sampling_settings(local: Mapping[str, Any]) -> None:
    """Check sampling overrides before selecting a node or drawing a count.

    For the generic sampling designs, zero min/max_samples select the node
    fraction and omit that count bound. The FEEMS-lattice sampler has its own
    cap convention. max_samples also caps total individuals in both samplers.
    """
    for key in ("min_samples", "max_samples"):
        if local.get(key) is not None:
            _integer_setting(local[key], key, minimum=0)
    for key in ("diploids_per_node", "diploids_per_site", "min_sampled_nodes", "max_sampled_nodes"):
        if local.get(key) is not None:
            _integer_setting(local[key], key)
    for key in ("sampled_node_fraction", "min_sample_fraction", "max_sample_fraction"):
        if local.get(key) is not None and not 0 <= _finite_setting(local[key], key) <= 1:
            raise ValueError(f"{key} must lie in [0, 1]")
    low, high = local.get("min_sample_fraction"), local.get("max_sample_fraction")
    if (low is None) != (high is None):
        raise ValueError(
            "sample fractions require both min_sample_fraction and max_sample_fraction"
        )
    for name in ("diploids_per_site", "diploids_per_node"):
        _integer_bounds(local.get(f"min_{name}"), local.get(f"max_{name}"), name)
    low, high = local.get("min_samples", 0), local.get("max_samples", 0)
    if low and high and int(low) > int(high):
        raise ValueError("min_samples must not exceed max_samples")
    design = local.get("sample_design", local.get("design"))
    if design is not None and design not in {
        "sites",
        "poisson_sites",
        "uniform",
        "clustered",
        "full",
    }:
        raise ValueError(f"unknown sample design: {design!r}")


@beartype
def validate_simulation_settings(sim: Mapping[str, Any]) -> None:
    """Validate the complete recipe without drawing randomness or rewriting it."""
    from dreems.island_sim import _check_ascertainment

    validate_sampling_settings(sim)
    for key in (
        "min_height",
        "max_height",
        "min_width",
        "max_width",
        "earth_min_nodes",
        "earth_max_nodes",
        "global_landscape_min_nodes",
        "global_landscape_max_nodes",
        "global_landscape_target_nodes",
        "global_landscape_max_region_tries",
        "global_landscape_sampled_sites",
        "global_landscape_min_sampled_sites",
        "global_landscape_max_sampled_sites",
        "global_landscape_max_total_samples",
        "global_landscape_min_diploids_per_site",
        "global_landscape_max_diploids_per_site",
        "wolflike_target_nodes",
        "wolflike_sampled_sites",
        "wolflike_min_sampled_sites",
        "wolflike_max_sampled_sites",
        "wolflike_max_total_samples",
        "wolflike_min_diploids_per_site",
        "wolflike_max_diploids_per_site",
        "feems_lattice_n_samples_per_node",
        "num_snps",
        "landscape_min_features",
        "landscape_max_features",
        "gp_features",
        "coalescent_batch_replicates",
        "coalescent_max_batches",
        "island_ploidy",
        "island_max_attempts",
        "max_conditionable_attempts",
    ):
        if sim[key] is not None:
            _integer_setting(sim[key], key)
    for key in ("feems_lattice_rows", "feems_lattice_columns"):
        _integer_setting(sim[key], key, minimum=2)
    _integer_setting(sim["msprime_max_nodes"], "msprime_max_nodes", minimum=0)
    _integer_bounds(sim["min_num_snps"], sim["max_num_snps"], "num_snps")
    for key in (
        "log_rate_std",
        "q_log_std",
        "global_landscape_center_jitter_deg",
        "global_landscape_water_strength",
        "global_landscape_coast_strength",
        "global_landscape_ridge_strength",
        "landscape_water_strength",
        "landscape_mountain_strength",
        "landscape_coast_strength",
        "landscape_feature_jitter",
    ):
        if _finite_setting(sim[key], key) < 0:
            raise ValueError(f"{key} must be nonnegative")
    for key in (
        "global_landscape_min_box_deg",
        "global_landscape_max_box_deg",
        "global_landscape_box_aspect_min",
        "global_landscape_box_aspect_max",
        "migration_gp_length_scale",
        "q_gp_length_scale",
        "coalescent_migration_scale",
        "coalescent_effective_size",
        "coalescent_mutation_rate",
        "coalescent_sequence_length",
        "island_mutation_rate",
        "island_migration_scale",
        "island_coalescence_scale",
        "feems_lattice_barrier_w",
        "feems_lattice_corridor_w",
        "feems_lattice_anisotropy_scaler",
    ):
        if _finite_setting(sim[key], key) <= 0:
            raise ValueError(f"{key} must be positive")
    for key in (
        "global_landscape_min_land_fraction",
        "global_landscape_path_sample_probability",
        "feems_lattice_barrier_prob",
        "feems_lattice_corridor_left_prob",
        "feems_lattice_corridor_right_prob",
        "island_min_allele_frequency",
        "island_max_allele_frequency",
    ):
        if not 0 <= _finite_setting(sim[key], key) <= 1:
            raise ValueError(f"{key} must lie in [0, 1]")
    for name in ("log_rate_std", "q_log_std", "migration_gp_length_scale", "q_gp_length_scale"):
        low, high = sim[f"min_{name}"], sim[f"max_{name}"]
        if low is None and high is None:
            continue
        if low is None or high is None:
            raise ValueError(f"{name} requires both range bounds")
        values = [_finite_setting(value, name) for value in (low, high)]
        if min(values) < 0 or ("length_scale" in name and min(values) == 0):
            raise ValueError(f"invalid bounds for {name}")
    for low, high in (
        ("min_height", "max_height"),
        ("min_width", "max_width"),
        ("earth_min_nodes", "earth_max_nodes"),
        ("global_landscape_min_nodes", "global_landscape_max_nodes"),
        ("global_landscape_min_diploids_per_site", "global_landscape_max_diploids_per_site"),
        ("wolflike_min_diploids_per_site", "wolflike_max_diploids_per_site"),
        ("landscape_min_features", "landscape_max_features"),
        ("global_landscape_min_box_deg", "global_landscape_max_box_deg"),
        ("island_min_allele_frequency", "island_max_allele_frequency"),
    ):
        if _finite_setting(sim[low], low) > _finite_setting(sim[high], high):
            raise ValueError(f"{low} must not exceed {high}")
    for key in (
        "global_landscape_min_lat",
        "global_landscape_max_lat",
        "global_landscape_preferred_center_lat",
    ):
        if sim[key] is not None and not -90 <= _finite_setting(sim[key], key) <= 90:
            raise ValueError(f"{key} must lie in [-90, 90]")
    lon, lat = (
        sim["global_landscape_preferred_center_lon"],
        sim["global_landscape_preferred_center_lat"],
    )
    if (lon is None) != (lat is None):
        raise ValueError("a preferred landscape center needs both longitude and latitude")
    if lon is not None and not -180 <= _finite_setting(lon, "preferred longitude") <= 180:
        raise ValueError("preferred longitude must lie in [-180, 180]")
    if _finite_setting(sim["global_landscape_min_lat"], "minimum latitude") >= _finite_setting(
        sim["global_landscape_max_lat"], "maximum latitude"
    ):
        raise ValueError("global_landscape_min_lat must be less than global_landscape_max_lat")
    if _finite_setting(sim["feems_lattice_barrier_startpt"], "barrier start") >= _finite_setting(
        sim["feems_lattice_barrier_endpt"], "barrier end"
    ):
        raise ValueError("FEEMS lattice barrier start must precede its end")
    for key in (
        "island_symmetric_migration",
        "island_use_rust",
        "island_require_rust",
        "island_emit_covariance",
    ):
        if not isinstance(sim[key], (bool, np.bool_)):
            raise ValueError(f"{key} must be a boolean")
    if sim["q_prior"] not in {"constant", "smooth", "gp"}:
        raise ValueError("q_prior must be constant, smooth, or gp")
    if sim["earth_backend"] != "synthetic_grid":
        raise ValueError("earth_backend must be synthetic_grid")
    if sim["global_landscape_resolution"] not in {"110m", "50m", "10m"}:
        raise ValueError("global_landscape_resolution must be 110m, 50m, or 10m")
    selected_simulator(sim, 2)
    _check_ascertainment(
        min_allele_frequency=_finite_setting(
            sim["island_min_allele_frequency"], "minimum frequency"
        ),
        max_allele_frequency=_finite_setting(
            sim["island_max_allele_frequency"], "maximum frequency"
        ),
        marker_model=sim["island_marker_model"],
    )

    # Extra weight names are allowed: a graph family may use a subset of a
    # shared surface mixture. Missing surface weights default to one.
    for key, value in sim.items():
        if key.endswith("_weights"):
            weights = parse_weight_mapping(value)
            if weights:
                _weight_probabilities(np.asarray(list(weights.values())))
        if key.endswith("_choices") and value is not None:
            integer = key not in {
                "global_landscape_box_width_choices",
                "global_landscape_sampled_fraction_choices",
                "wolflike_sampled_fraction_choices",
                "feems_lattice_sample_prob_choices",
            }
            candidates = _positive_choices(value, integer=integer)
            if ("fraction" in key or "sample_prob" in key) and max(map(float, candidates)) > 1:
                raise ValueError(f"{key} must lie in (0, 1]")
            if (
                key in {"feems_lattice_row_choices", "feems_lattice_column_choices"}
                and min(map(int, candidates)) < 2
            ):
                raise ValueError(f"{key} must contain counts >=2")
    case_weights = parse_weight_mapping(sim["case_weights"])
    if set(case_weights) - set(CASE_KINDS):
        raise ValueError("unsupported simulation case kinds")
    _weight_probabilities(np.asarray([case_weights.get(name, 0.0) for name in CASE_KINDS]))
    surfaces = {
        "constant",
        "smooth",
        "barrier",
        "corridor",
        "gp",
        "earth",
        "landscape",
        "north_america_landscape",
        "multiscale",
        "feems_homogeneous",
        "feems_heterogeneous_barrier",
        "feems_anisotropic",
    }
    for key, allowed in (
        ("mask_kinds", {"full", "bay", "lake", "coast", "islands", "corridor"}),
        ("sample_designs", {"sites", "poisson_sites", "uniform", "clustered", "full"}),
        ("surface_kinds", surfaces),
        ("wolflike_surface_kinds", surfaces),
        ("feems_lattice_surface_kinds", surfaces),
        ("global_landscape_surface_kinds", surfaces | {"global_landscape"}),
        ("wolflike_graph_sources", {"synthetic-corridor", "corridor", "land"}),
    ):
        if sim[key] is not None:
            names = string_list(sim[key])
            if not names or set(names) - allowed:
                raise ValueError(f"invalid {key}: {names}")
    regimes = sim["sample_regimes"]
    if regimes is not None and not isinstance(regimes, (list, tuple)):
        raise ValueError("sample_regimes must be a list of mappings")
    if regimes:
        if not isinstance(regimes, (list, tuple)) or not all(
            isinstance(row, Mapping) for row in regimes
        ):
            raise ValueError("sample_regimes must be a list of mappings")
        allowed = {
            "name",
            "weight",
            "sample_design",
            "design",
            "min_samples",
            "max_samples",
            "min_sample_fraction",
            "max_sample_fraction",
            "min_diploids_per_site",
            "max_diploids_per_site",
            "diploids_per_node",
            "diploids_per_site",
        }
        for regime in regimes:
            if set(regime) - allowed:
                raise ValueError(f"unknown sample-regime fields: {set(regime) - allowed}")
            validate_sampling_settings({**sim, **regime})
        _weight_probabilities(
            np.asarray([_finite_setting(row.get("weight", 1), "regime weight") for row in regimes])
        )


@beartype
def selected_simulator(sim: Mapping[str, Any], num_nodes: Integral) -> str:
    """Resolve backend aliases; auto uses msprime only on small graphs."""
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


@beartype
def string_list(value: object, *, default: Sequence[str] = ()) -> list[str]:
    """Accept a config list or names separated by commas or whitespace."""
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [piece for chunk in value.split(",") for piece in chunk.split() if piece]
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError("names must be a one-dimensional sequence")
        value = value.tolist()
    if (
        not isinstance(value, Sequence)
        or isinstance(value, bytes)
        or not all(isinstance(item, str) for item in value)
    ):
        raise ValueError("names must be strings, supplied as a sequence or comma-separated text")
    return [str(item) for item in value]


@beartype
def parse_weight_mapping(value: object) -> dict[str, float]:
    """Read named weights from a mapping or the historical ``name=weight`` form.

    Invalid entries raise an error.
    Callers decide whether absent names receive zero or the default weight one.
    """
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
                    raise ValueError(f"expected name=weight, got {item!r}")
                key, weight = text.split("=", 1)
            raw_items.append((key, weight))
    weights: dict[str, float] = {}
    for key, weight in raw_items:
        if key is None or not str(key).strip() or str(key) in weights:
            raise ValueError("weight names must be nonempty and unique")
        numeric = _finite_setting(weight, f"weight for {key!r}")
        if numeric < 0:
            raise ValueError(f"weight for {key!r} must be nonnegative")
        weights[str(key)] = numeric
    return weights


@beartype
def choose_weighted_name(
    rng: np.random.Generator,
    names: Sequence[str],
    weight_config: object,
) -> str:
    """Draw a name, using weight one for unspecified candidates.

    Zero excludes a candidate. Empty weights mean a uniform draw, but explicitly
    assigning zero to every candidate is an error.
    """
    candidates = [str(name) for name in names]
    if not candidates:
        raise ValueError("at least one candidate is required")
    weights_map = parse_weight_mapping(weight_config)
    if not weights_map:
        return candidates[int(rng.integers(0, len(candidates)))]
    weights = np.asarray([weights_map.get(name, 1.0) for name in candidates], dtype=np.float64)
    return candidates[int(rng.choice(len(candidates), p=_weight_probabilities(weights)))]


CASE_KINDS = ("grid", "earth_like", "wolflike", "global_landscape", "feems_lattice")


@beartype
def choose_case_kind(rng: np.random.Generator, value: object) -> str:
    """Draw a graph family; unlike surface weights, absent families are excluded."""
    weights = parse_weight_mapping(value)
    unknown = sorted(set(weights) - set(CASE_KINDS))
    if unknown:
        raise ValueError(f"unsupported simulation case kinds: {', '.join(unknown)}")
    names = [name for name in CASE_KINDS if weights.get(name, 0.0) > 0.0]
    if not names:
        raise ValueError("simulation.case_weights must assign positive weight to a case kind")
    probabilities = np.asarray([weights[name] for name in names], dtype=np.float64)
    return names[int(rng.choice(len(names), p=_weight_probabilities(probabilities)))]


@beartype
def draw_float_range(
    rng: np.random.Generator,
    default_value: Real,
    min_value: object,
    max_value: object,
) -> float:
    """Draw between the bounds; use the default only when both are absent."""
    if min_value is None and max_value is None:
        return _finite_setting(default_value, "default_value")
    if min_value is None or max_value is None:
        raise ValueError("a floating-point range requires both bounds")
    values = (_finite_setting(min_value, "min_value"), _finite_setting(max_value, "max_value"))
    low, high = min(values), max(values)
    return float(rng.uniform(low, high))


@beartype
def draw_num_snps(rng: np.random.Generator, sim: Mapping[str, Any]) -> int:
    """Draw SNP count from equal-weight log bins, an integer range, or a fixed count.

    A single uniform draw chooses both the log bin and the position within it.
    Keeping that draw order preserves the seeds of existing corpus examples.
    """
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
    bounds = _integer_bounds(sim.get("min_num_snps"), sim.get("max_num_snps"), "num_snps")
    if bounds is not None:
        low, high = bounds
        return int(rng.integers(low, high + 1))
    return _integer_setting(sim["num_snps"], "num_snps")


@beartype
def local_simulation_parameters(
    rng: np.random.Generator,
    sim: Mapping[str, Any],
) -> dict[str, Any]:
    """Draw per-example surface scales and sample counts in a fixed RNG order."""
    local = dict(sim)
    for name in ("log_rate_std", "q_log_std", "migration_gp_length_scale", "q_gp_length_scale"):
        local[name] = draw_float_range(
            rng, sim[name], sim.get(f"min_{name}"), sim.get(f"max_{name}")
        )
    local["diploids_per_node"] = draw_local_diploids_per_node(rng, local)
    return local


@beartype
def draw_weighted_positive_int(
    rng: np.random.Generator,
    *,
    value: object,
    choices: object,
    weights: object,
    fallback: Integral,
) -> int:
    """Draw a positive integer; an omitted value uses the explicit fallback."""
    if choices is None:
        return _integer_setting(fallback if value is None else value, "value")
    candidates = _positive_choices(choices, integer=True)
    return int(choose_weighted_name(rng, candidates, weights))


@beartype
def draw_weighted_positive_float(
    rng: np.random.Generator,
    *,
    choices: object,
    weights: object,
    fallback: Real,
) -> float:
    """Draw a positive number; omitted choices use the explicit fallback."""
    if choices is None:
        number = _finite_setting(fallback, "fallback")
        if number <= 0:
            raise ValueError("fallback must be positive")
        return number
    candidates = _positive_choices(choices, integer=False)
    return float(choose_weighted_name(rng, candidates, weights))


@beartype
def draw_local_diploids_per_node(
    rng: np.random.Generator,
    sim: Mapping[str, Any],
) -> int:
    """Prefer per-site bounds, then per-node bounds, then the fixed sample count."""
    for name in ("diploids_per_site", "diploids_per_node"):
        bounds = _integer_bounds(sim.get(f"min_{name}"), sim.get(f"max_{name}"), name)
        if bounds is not None:
            return int(rng.integers(bounds[0], bounds[1] + 1))
    return _integer_setting(sim["diploids_per_node"], "diploids_per_node")


@beartype
def local_simulation_metadata(local: Mapping[str, Any]) -> dict[str, Any]:
    """Record the sampled parameters alongside each raw example."""
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
