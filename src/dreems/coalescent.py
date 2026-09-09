"""Small-graph msprime simulation and SNP ascertainment."""

# ruff: noqa: F722,F821,UP037

from __future__ import annotations

import numpy as np
from beartype import beartype
from jaxtyping import Integer, Real, jaxtyped


@jaxtyped(typechecker=beartype)
def allele_frequency_columns(
    genotypes: Real[np.ndarray, "sample snp"],
    *,
    ploidy: int = 2,
) -> np.ndarray:
    """Derived-allele frequency for each fixed-ploidy dosage column."""
    values = np.asarray(genotypes, dtype=np.float64)
    return values.sum(axis=0) / float(int(ploidy) * values.shape[0])


@jaxtyped(typechecker=beartype)
def ascertained_columns(
    genotypes: Real[np.ndarray, "sample snp"],
    *,
    min_allele_frequency: float,
    max_allele_frequency: float,
    ploidy: int = 2,
) -> np.ndarray:
    """Keep segregating columns within the inclusive derived-frequency interval."""
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
) -> np.ndarray:
    """Convert msprime population-grouped haplotypes to caller sample order."""

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
) -> np.ndarray:
    """Select one segregating site per mutation-bearing msprime replicate.

    This is diploid and uses Ne=effective_size*exp(-node_log_q), clipped to
    [1e-3, 1e6]. Each tree contributes at most one site; retained sites then
    pass the derived-frequency filter. It is not the always-marked-tree mode
    used by the production Rust corpus.
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

    chunks: list[np.ndarray] = []
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
