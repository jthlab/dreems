"""Small-graph msprime simulation and SNP ascertainment."""

from __future__ import annotations

import numpy as np


def allele_frequency_columns(
    genotypes: np.ndarray,
    *,
    ploidy: int = 2,
) -> np.ndarray:
    values = np.asarray(genotypes, dtype=np.float64)
    return values.sum(axis=0) / float(int(ploidy) * values.shape[0])


def ascertained_columns(
    genotypes: np.ndarray,
    *,
    min_allele_frequency: float,
    max_allele_frequency: float,
    ploidy: int = 2,
) -> np.ndarray:
    values = np.asarray(genotypes)
    allele_sum = values.sum(axis=0)
    polymorphic = (allele_sum > 0.0) & (allele_sum < float(int(ploidy) * values.shape[0]))
    frequency = allele_frequency_columns(values, ploidy=ploidy)
    return polymorphic & (frequency >= min_allele_frequency) & (frequency <= max_allele_frequency)


def genotypes_from_grouped_haplotypes(
    haplotypes: np.ndarray,
    *,
    sample_nodes: np.ndarray,
    num_nodes: int,
    ploidy: int = 2,
) -> np.ndarray:
    """Convert msprime population-grouped haplotypes to caller sample order."""

    samples = np.asarray(sample_nodes, dtype=np.int32).reshape(-1)
    values = np.asarray(haplotypes)
    if values.ndim != 2:
        raise ValueError("haplotypes must have shape (num_snps, num_haplotypes)")
    sample_counts = np.bincount(samples, minlength=int(num_nodes)).astype(np.int64)
    haploid_counts = sample_counts * int(ploidy)
    offsets = np.concatenate([[0], np.cumsum(haploid_counts[:-1])]).astype(np.int64)
    if int(haploid_counts.sum()) != values.shape[1]:
        raise ValueError("haplotype count does not match sample_nodes and ploidy")
    cursors = np.zeros(int(num_nodes), dtype=np.int64)
    genotypes = np.empty((samples.shape[0], values.shape[0]), dtype=values.dtype)
    for sample_index, deme_value in enumerate(samples):
        deme = int(deme_value)
        start = int(offsets[deme] + cursors[deme])
        end = start + int(ploidy)
        genotypes[sample_index] = values[:, start:end].sum(axis=1)
        cursors[deme] += int(ploidy)
    return genotypes


def simulate_msprime_genotypes(
    *,
    num_nodes: int,
    edges: np.ndarray,
    edge_log_rates: np.ndarray,
    node_log_q: np.ndarray,
    sample_nodes: np.ndarray,
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
