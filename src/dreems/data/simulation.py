"""Generate one conditioned coalescent example from a corpus configuration."""

# ruff: noqa: F722, F821, UP037

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from beartype import beartype
from jaxtyping import Bool, Float, Integer, Real, jaxtyped

from dreems.data import island_sim
from dreems.data.motifs import (
    apply_migration_associated_sampling,
    choose_sample_regime,
    draw_earth_like_case,
    draw_feems_lattice_case,
    draw_global_landscape_case,
    draw_grid_case,
    draw_wolflike_case,
)
from dreems.data.simulation_config import (
    choose_case_kind,
    choose_weighted_name,
    draw_num_snps,
    local_simulation_parameters,
    selected_simulator,
    string_list,
)


class _InsufficientSampling(ValueError):
    """A random sampling draw cannot supply a pair of observed demes."""


@beartype
def simulate_case_genotypes(
    rng: np.random.Generator,
    sim: Mapping[str, Any],
    case: dict[str, Any],
    *,
    num_snps: int,
) -> str:
    """Attach genotype dosages or covariance to a pre-drawn graph and surface."""
    graph = case["graph"]
    positions = np.asarray(case["active_positions"])
    edges = np.asarray(graph["edges"], dtype=np.int32)
    edge_log_rates = np.asarray(case["edge_log_rates"], dtype=np.float64)
    node_log_q = np.asarray(case["node_log_q"], dtype=np.float64)
    sample_nodes = np.asarray(case["sample_nodes"], dtype=np.int32)
    simulator = selected_simulator(sim, positions.shape[0])
    genotypes: Float[np.ndarray, "sample snp"] | None = None
    sample_covariance: Float[np.ndarray, "sample sample"] | None = None
    effective_snps: int | None = None
    if simulator in {"msprime_coalescent", "island_coalescent"}:
        case["island_min_allele_frequency"] = float(sim["island_min_allele_frequency"])
        case["island_max_allele_frequency"] = float(sim["island_max_allele_frequency"])
        case["island_marker_model"] = str(sim["island_marker_model"])
    if simulator == "msprime_coalescent":
        genotypes = simulate_msprime_genotypes(
            num_nodes=positions.shape[0],
            edges=edges,
            edge_log_rates=edge_log_rates,
            node_log_q=node_log_q,
            sample_nodes=sample_nodes,
            num_snps=num_snps,
            rng=rng,
            migration_scale=float(sim["coalescent_migration_scale"]),
            effective_size=float(sim["coalescent_effective_size"]),
            mutation_rate=float(sim["coalescent_mutation_rate"]),
            sequence_length=float(sim["coalescent_sequence_length"]),
            batch_replicates=int(sim["coalescent_batch_replicates"]),
            max_batches=int(sim["coalescent_max_batches"]),
            min_allele_frequency=float(sim["island_min_allele_frequency"]),
            max_allele_frequency=float(sim["island_max_allele_frequency"]),
        )
    elif simulator == "island_coalescent":
        if bool(sim["island_require_rust"]):
            if not bool(sim["island_use_rust"]):
                raise ValueError("island_require_rust requires island_use_rust")
            island_sim.require_rust_backend()
        island_kwargs = {
            "sample_nodes": sample_nodes,
            "num_demes": positions.shape[0],
            "q": float(sim["island_coalescence_scale"]) * np.exp(node_log_q),
            "migration_edges": edges,
            "migration_rates": float(sim["island_migration_scale"]) * np.exp(edge_log_rates),
            "num_snps": num_snps,
            "ploidy": int(sim["island_ploidy"]),
            "mutation_rate": float(sim["island_mutation_rate"]),
            "symmetric": bool(sim["island_symmetric_migration"]),
            "seed": int(rng.integers(1, np.iinfo(np.int32).max)),
            "max_attempts": (
                None if sim.get("island_max_attempts") is None else int(sim["island_max_attempts"])
            ),
            "use_rust": bool(sim["island_use_rust"]),
            "min_allele_frequency": float(sim["island_min_allele_frequency"]),
            "max_allele_frequency": float(sim["island_max_allele_frequency"]),
            "marker_model": str(sim["island_marker_model"]),
        }
        if bool(sim["island_emit_covariance"]):
            covariance_result = island_sim.simulate_island_covariance(**island_kwargs)
            case["island_covariance_estimator"] = "finite"
            sample_covariance = covariance_result.sample_covariance.astype(np.float64)
            effective_snps = int(covariance_result.effective_snps)
            case["island_attempts"] = int(covariance_result.attempts)
            case["island_used_rust"] = bool(covariance_result.used_rust)
        else:
            result = island_sim.simulate_island_model(**island_kwargs)
            genotypes = result.genotypes.astype(np.float64)
            case["island_attempts"] = int(result.attempts)
            case["island_used_rust"] = bool(result.used_rust)
    else:
        raise ValueError(f"unsupported simulator after validation: {simulator!r}")
    if genotypes is not None:
        case["genotypes"] = genotypes
        case.pop("sample_covariance", None)
        case.pop("effective_snps", None)
    else:
        if sample_covariance is None or effective_snps is None:
            raise RuntimeError("simulator did not produce genotypes or sample covariance")
        case["sample_covariance"] = sample_covariance
        case["effective_snps"] = int(effective_snps)
        case.pop("genotypes", None)
    return simulator


@beartype
def simulate_raw_example(
    rng: np.random.Generator,
    *,
    sim: Mapping[str, Any],
    split: str,
    shard: int,
    index: int,
) -> dict[str, Any]:
    """Draw geometry, migration, q, sampling, and then the genetic observations.

    Keep this draw order stable: changing it changes every downstream random draw
    for a given seed, even when the marginal simulation recipe is unchanged.
    """
    local = choose_sample_regime(rng, sim, local_simulation_parameters(rng, sim))
    num_snps = draw_num_snps(rng, sim)
    surface_kinds = string_list(sim.get("surface_kinds"), default=["smooth", "barrier", "gp"])
    grid_surface_kinds = [kind for kind in surface_kinds if kind != "earth"] or [
        "smooth",
        "barrier",
        "gp",
    ]
    case_kind = choose_case_kind(rng, sim["case_weights"])

    if case_kind == "feems_lattice":
        case = draw_feems_lattice_case(rng, sim, local)
        surface_kind = str(case["surface_kind"])
    elif case_kind == "global_landscape":
        case = draw_global_landscape_case(rng, sim, local)
        surface_kind = str(case["surface_kind"])
    elif case_kind == "wolflike":
        case = draw_wolflike_case(rng, sim, local, surface_kinds=surface_kinds)
        surface_kind = str(case["surface_kind"])
    elif case_kind == "earth_like":
        surface_kind = choose_weighted_name(rng, surface_kinds, sim.get("surface_weights"))
        case = draw_earth_like_case(rng, sim, local, surface_kind=surface_kind)
        case["graph_source"] = "earth_like"
    else:
        surface_kind = choose_weighted_name(rng, grid_surface_kinds, sim.get("surface_weights"))
        case = draw_grid_case(rng, sim, local, surface_kind=surface_kind)
        case["graph_source"] = "grid"

    sampling_mode = sim.get("sampling_association_mode_override")
    if sampling_mode is not None:
        case = apply_migration_associated_sampling(
            rng,
            case,
            sim,
            mode=str(sampling_mode),
        )

    if not pairwise_conditionable_example(case):
        # Check before simulation: an empty draw fails inside the simulator,
        # before the seeded worker would otherwise get a chance to retry it.
        raise _InsufficientSampling("fewer than two graph nodes were sampled")
    simulator = simulate_case_genotypes(rng, sim, case, num_snps=num_snps)
    case.update(
        {
            "num_snps": int(num_snps),
            "simulator": simulator,
            "surface_kind": str(surface_kind),
            "split": split,
            "shard": int(shard),
            "index": int(index),
        }
    )
    return case


@beartype
def counterfactual_sampling_assignment(
    sim: Mapping[str, Any],
    example_id: int,
    *,
    split: str,
) -> tuple[int, str | None]:
    """Give related sampling views a shared graph seed and different masks."""
    modes = string_list(sim.get("sampling_counterfactual_modes"))
    if not modes:
        return int(example_id), None
    mode = modes[int(example_id) % len(modes)]
    grouped_splits = set(string_list(sim.get("sampling_counterfactual_group_splits")))
    seed_shard = int(example_id) // len(modes) if split in grouped_splits else int(example_id)
    return seed_shard, mode


SPLIT_SEED_OFFSETS = {"train": 0, "test": 10_000_000}


@beartype
def raw_example_seed(*, base_seed: int, split: str, shard: int, index: int) -> int:
    """Retain the historical seed formula for reproducing existing raw packs."""
    if split not in SPLIT_SEED_OFFSETS:
        raise ValueError(f"unknown raw generation split {split!r}")
    return int(base_seed + SPLIT_SEED_OFFSETS[split] + 1009 * shard + 104_729 * index)


@beartype
def simulate_raw_example_from_seed(
    payload: tuple[int, dict[str, Any], str, int, int],
) -> dict[str, Any]:
    """Retry an example only when fewer than two graph nodes were sampled."""
    seed, sim, split, shard, index = payload
    max_attempts = int(sim.get("max_conditionable_attempts", 32))
    for attempt in range(max(max_attempts, 1)):
        try:
            example = simulate_raw_example(
                np.random.default_rng(seed + 1_000_003 * attempt),
                sim=sim,
                split=split,
                shard=shard,
                index=index,
            )
        except _InsufficientSampling:
            continue
        if pairwise_conditionable_example(example):
            if attempt:
                example["conditionable_retry_attempt"] = int(attempt)
            return example
    raise ValueError(
        f"failed to generate a pairwise-conditionable raw example after {max_attempts} "
        f"attempts for split={split}, shard={shard}, index={index}"
    )


@beartype
def pairwise_conditionable_example(example: Mapping[str, Any]) -> bool:
    """A resistance conditioner needs at least one pair of observed demes."""
    sample_nodes = np.asarray(example.get("sample_nodes", []), dtype=np.int32)
    return bool(np.unique(sample_nodes).shape[0] >= 2)


@jaxtyped(typechecker=beartype)
def allele_frequency_columns(
    genotypes: Real[np.ndarray, "sample snp"],
    *,
    ploidy: int = 2,
) -> Float[np.ndarray, "snp"]:
    """Calculate allele frequency from dosage and ploidy."""
    values = np.asarray(genotypes, dtype=np.float64)
    return values.sum(axis=0) / float(int(ploidy) * values.shape[0])


@jaxtyped(typechecker=beartype)
def ascertained_columns(
    genotypes: Real[np.ndarray, "sample snp"],
    *,
    min_allele_frequency: float,
    max_allele_frequency: float,
    ploidy: int = 2,
) -> Bool[np.ndarray, "snp"]:
    """Select segregating SNPs in the requested frequency interval."""
    values = np.asarray(genotypes)
    allele_sum = values.sum(axis=0)
    polymorphic = (allele_sum > 0.0) & (allele_sum < float(int(ploidy) * values.shape[0]))
    frequency = allele_frequency_columns(values, ploidy=ploidy)
    return polymorphic & (frequency >= min_allele_frequency) & (frequency <= max_allele_frequency)


@jaxtyped(typechecker=beartype)
def genotypes_from_grouped_haplotypes(
    haplotypes: Real[np.ndarray, "snp haploid"],
    *,
    sample_nodes: Integer[np.ndarray, "sample"],
    num_nodes: int,
    ploidy: int = 2,
) -> Real[np.ndarray, "sample snp"]:
    """Sum haplotypes into dosages and restore the caller's sample order."""

    samples = np.asarray(sample_nodes)
    values = np.asarray(haplotypes)
    if ploidy <= 0:
        raise ValueError("ploidy must be positive")
    if np.any(samples < 0) or np.any(samples >= num_nodes):
        raise ValueError("sample_nodes are outside the graph")
    if len(samples) * ploidy != values.shape[1]:
        raise ValueError("haplotype count does not match sample_nodes and ploidy")
    grouped_dosages = values.reshape(values.shape[0], len(samples), ploidy).sum(axis=2).T
    genotypes = np.empty(grouped_dosages.shape, dtype=values.dtype)
    # msprime groups samples by deme; stable sorting preserves within-deme order.
    genotypes[np.argsort(samples, kind="stable")] = grouped_dosages
    return genotypes


@jaxtyped(typechecker=beartype)
def simulate_msprime_genotypes(
    *,
    num_nodes: int,
    edges: Integer[np.ndarray, "edge 2"],
    edge_log_rates: Real[np.ndarray, "edge"],
    node_log_q: Real[np.ndarray, "node"],
    sample_nodes: Integer[np.ndarray, "sample"],
    num_snps: int,
    rng: np.random.Generator,
    migration_scale: float,
    effective_size: float,
    mutation_rate: float,
    sequence_length: float,
    batch_replicates: int,
    max_batches: int,
    min_allele_frequency: float = 0.0,
    max_allele_frequency: float = 1.0,
) -> Float[np.ndarray, "sample retained_snps"]:
    """Select one segregating site from each msprime replicate with mutations.

    This is diploid and uses Ne=effective_size*exp(-node_log_q), clipped to
    [1e-3, 1e6]. Each tree contributes at most one site; retained sites then
    pass the derived-frequency filter. This simulator weights trees through
    their probability of carrying mutations. The Rust ``conditioned_marker``
    mode samples a branch on each tree in proportion to its eligible length.
    """
    try:
        import msprime
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "msprime is required for simulation.simulator='msprime_coalescent'"
        ) from exc

    sample_nodes = np.asarray(sample_nodes, dtype=np.int32)
    sample_counts = np.bincount(sample_nodes, minlength=num_nodes).astype(np.int32)
    node_ne = float(effective_size) * np.exp(-np.asarray(node_log_q, dtype=np.float64))
    node_ne = np.clip(node_ne, 1e-3, 1e6)
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=2 * int(sample_counts[node]),
            initial_size=float(node_ne[node]),
        )
        for node in range(num_nodes)
    ]
    migration_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    rates = float(migration_scale) * np.exp(np.asarray(edge_log_rates, dtype=np.float64))
    for (left, right), rate in zip(edges, rates, strict=True):
        migration_matrix[int(left), int(right)] = float(rate)
        migration_matrix[int(right), int(left)] = float(rate)

    chunks: list[Float[np.ndarray, "sample retained_snps"]] = []
    remaining = int(num_snps)
    for _batch in range(int(max_batches)):
        if remaining <= 0:
            break
        replicates = max(int(batch_replicates), remaining)
        seed = int(rng.integers(1, np.iinfo(np.int32).max))
        tree_sequences = msprime.simulate(
            population_configurations=population_configurations,
            migration_matrix=migration_matrix,
            length=float(sequence_length),
            mutation_rate=float(mutation_rate),
            num_replicates=replicates,
            Ne=1.0,
            random_seed=seed,
        )
        haplotypes = []
        for tree_sequence in tree_sequences:
            matrix = tree_sequence.genotype_matrix()
            if matrix.shape[0] == 0:
                continue
            chosen = matrix[int(rng.integers(0, matrix.shape[0])), :]
            if np.all(chosen == chosen[0]):
                continue
            haplotypes.append(chosen.astype(np.float64, copy=False))
            if len(haplotypes) >= remaining:
                break
        if haplotypes:
            haplotype_matrix = np.vstack(haplotypes)
            genotypes = genotypes_from_grouped_haplotypes(
                haplotype_matrix,
                sample_nodes=sample_nodes,
                num_nodes=num_nodes,
                ploidy=2,
            )
            keep = ascertained_columns(
                genotypes,
                min_allele_frequency=float(min_allele_frequency),
                max_allele_frequency=float(max_allele_frequency),
                ploidy=2,
            )
            if np.any(keep):
                kept = genotypes[:, keep][:, :remaining]
                chunks.append(kept)
                remaining -= kept.shape[1]
    if remaining > 0:
        raise RuntimeError(
            f"msprime simulator produced {num_snps - remaining} polymorphic SNPs "
            f"after {max_batches} batches; requested {num_snps}"
        )
    return np.concatenate(chunks, axis=1)[:, :num_snps]
