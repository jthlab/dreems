"""Structured island-model coalescent simulator for independent SNPs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

try:  # pragma: no cover - exercised when the optional Rust extension is built.
    from dreems import _island_sim
except ImportError:  # pragma: no cover - normal in source-only test environments.
    _island_sim = None


@dataclass(frozen=True)
class IslandSimulationResult:
    """Independent segregating SNPs simulated under a structured coalescent."""

    genotypes: NDArray[np.uint8]
    allele_counts: NDArray[np.uint32]
    attempts: int
    used_rust: bool


MarkerModel = Literal["first_mutation", "random_marker", "conditioned_random_marker"]


def rust_backend_available() -> bool:
    """Return whether the compiled island-simulator backend can be imported."""

    return _island_sim is not None


def require_rust_backend() -> None:
    """Fail before simulation when a workflow requires the Rust backend."""

    if not rust_backend_available():
        raise RuntimeError(
            "the Rust island simulator is required but dreems._island_sim is not installed; "
            "build it with `uv run maturin develop --release "
            "--manifest-path rust/island_sim/Cargo.toml`"
        )


@dataclass(frozen=True)
class IslandCovarianceResult:
    """Sample covariance accumulated directly from independent segregating SNPs."""

    sample_covariance: NDArray[np.float64]
    effective_snps: int
    attempts: int
    used_rust: bool


def _check_inputs(
    *,
    sample_nodes: NDArray[np.integer[Any]],
    num_demes: int,
    q: NDArray[np.floating[Any]],
    migration_edges: NDArray[np.integer[Any]],
    migration_rates: NDArray[np.floating[Any]],
    num_snps: int,
    ploidy: int,
    mutation_rate: float,
) -> tuple[NDArray[np.int32], NDArray[np.float64], NDArray[np.int32], NDArray[np.float64]]:
    samples = np.asarray(sample_nodes, dtype=np.int32)
    if samples.ndim != 1:
        raise ValueError("sample_nodes must be a one-dimensional integer array")
    if samples.size == 0:
        raise ValueError("sample_nodes must not be empty")
    if int(num_demes) <= 0:
        raise ValueError("num_demes must be positive")
    if np.any(samples < 0) or np.any(samples >= int(num_demes)):
        raise ValueError("sample_nodes entries must be in [0, num_demes)")
    if int(num_snps) <= 0:
        raise ValueError("num_snps must be positive")
    if int(ploidy) <= 0:
        raise ValueError("ploidy must be positive")
    if float(mutation_rate) <= 0.0 or not np.isfinite(mutation_rate):
        raise ValueError("mutation_rate must be finite and positive")

    q_array = np.asarray(q, dtype=np.float64)
    if q_array.shape != (int(num_demes),):
        raise ValueError("q must have shape (num_demes,)")
    if np.any(~np.isfinite(q_array)) or np.any(q_array < 0.0):
        raise ValueError("q must contain finite non-negative coalescence rates")

    edges = np.asarray(migration_edges, dtype=np.int32)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("migration_edges must have shape (num_edges, 2)")
    if np.any(edges < 0) or np.any(edges >= int(num_demes)):
        raise ValueError("migration_edges entries must be in [0, num_demes)")
    rates = np.asarray(migration_rates, dtype=np.float64)
    if rates.shape != (edges.shape[0],):
        raise ValueError("migration_rates must have shape (num_edges,)")
    if np.any(~np.isfinite(rates)) or np.any(rates < 0.0):
        raise ValueError("migration_rates must contain finite non-negative values")
    return samples, q_array, edges, rates


def _check_ascertainment(
    *,
    min_allele_frequency: float,
    max_allele_frequency: float,
    marker_model: str,
) -> tuple[float, float, MarkerModel]:
    min_frequency = float(min_allele_frequency)
    max_frequency = float(max_allele_frequency)
    if (
        not np.isfinite(min_frequency)
        or not np.isfinite(max_frequency)
        or min_frequency < 0.0
        or max_frequency > 1.0
        or min_frequency > max_frequency
    ):
        raise ValueError("allele-frequency thresholds must satisfy 0 <= min <= max <= 1")
    aliases: dict[str, MarkerModel] = {
        "first": "first_mutation",
        "first_mutation": "first_mutation",
        "random_marker": "random_marker",
        "uniform_marker": "random_marker",
        "feems_paper": "random_marker",
        "branch_marker": "conditioned_random_marker",
        "conditioned_first_mutation": "conditioned_random_marker",
        "conditioned_marker": "conditioned_random_marker",
        "conditioned_random_marker": "conditioned_random_marker",
        "segregating_marker": "conditioned_random_marker",
    }
    try:
        resolved = aliases[str(marker_model)]
    except KeyError as error:
        raise ValueError(
            "marker_model must be first_mutation, random_marker, or conditioned_random_marker"
        ) from error
    return min_frequency, max_frequency, resolved


def _counts_pass_allele_frequency(
    counts: NDArray[np.uint32],
    *,
    total_haploids: int,
    min_allele_frequency: float,
    max_allele_frequency: float,
) -> bool:
    frequency = float(np.sum(counts, dtype=np.uint64)) / float(total_haploids)
    return min_allele_frequency <= frequency <= max_allele_frequency


def _covariance_from_genotypes(
    genotypes: NDArray[np.integer[Any]],
) -> tuple[NDArray[np.float64], int]:
    values = np.asarray(genotypes, dtype=np.float64)
    allele_sum = values.sum(axis=0)
    polymorphic = (allele_sum > 0.0) & (allele_sum < 2.0 * values.shape[0])
    if not np.any(polymorphic):
        raise ValueError("no polymorphic SNPs")
    standardized = values[:, polymorphic] - values[:, polymorphic].mean(axis=0, keepdims=True)
    sd = standardized.std(axis=0, ddof=1, keepdims=True)
    keep = np.ravel(sd > 1e-8)
    if not np.any(keep):
        raise ValueError("no usable polymorphic SNPs")
    standardized = standardized[:, keep] / sd[:, keep]
    return (standardized @ standardized.T) / float(standardized.shape[1]), int(
        standardized.shape[1]
    )


def _directed_adjacency(
    *,
    num_demes: int,
    migration_edges: NDArray[np.int32],
    migration_rates: NDArray[np.float64],
    symmetric: bool,
) -> list[list[tuple[int, float]]]:
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(num_demes)]
    for (left, right), rate in zip(migration_edges, migration_rates, strict=True):
        if rate <= 0.0 or int(left) == int(right):
            continue
        adjacency[int(left)].append((int(right), float(rate)))
        if symmetric:
            adjacency[int(right)].append((int(left), float(rate)))
    return adjacency


def _merge_descendant_counts(left: dict[int, int], right: dict[int, int]) -> dict[int, int]:
    if len(left) < len(right):
        left, right = right, left
    for key, value in right.items():
        left[key] = left.get(key, 0) + value
    return left


def _remove_active(active: list[int], active_pos: list[int], lineage: int) -> None:
    pos = active_pos[lineage]
    last = active.pop()
    if pos < len(active):
        active[pos] = last
        active_pos[last] = pos
    active_pos[lineage] = -1


def _simulate_one_allele_count_python(
    *,
    sample_nodes: NDArray[np.int32],
    num_demes: int,
    q: NDArray[np.float64],
    adjacency: list[list[tuple[int, float]]],
    ploidy: int,
    mutation_rate: float,
    rng: np.random.Generator,
) -> NDArray[np.uint32] | None:
    deme_lineages: list[list[int]] = [[] for _ in range(num_demes)]
    lineage_deme: list[int] = []
    descendants: list[dict[int, int] | None] = []
    active: list[int] = []
    active_pos: list[int] = []

    for sample_deme in sample_nodes:
        deme = int(sample_deme)
        for _copy in range(ploidy):
            lineage = len(lineage_deme)
            lineage_deme.append(deme)
            descendants.append({deme: 1})
            deme_lineages[deme].append(lineage)
            active_pos.append(len(active))
            active.append(lineage)

    total_haploids = len(active)
    if total_haploids < 2:
        return None

    while len(active) > 1:
        coalescence_rates = np.asarray(
            [
                float(q[deme]) * len(items) * (len(items) - 1) * 0.5
                for deme, items in enumerate(deme_lineages)
            ],
            dtype=np.float64,
        )
        migration_rates = np.asarray(
            [
                len(items) * sum(rate for _target, rate in adjacency[deme])
                for deme, items in enumerate(deme_lineages)
            ],
            dtype=np.float64,
        )
        total_coalescence = float(coalescence_rates.sum())
        total_migration = float(migration_rates.sum())
        total_mutation = float(mutation_rate) * len(active)
        total_rate = total_coalescence + total_migration + total_mutation
        if total_rate <= 0.0 or not np.isfinite(total_rate):
            return None

        event = float(rng.random() * total_rate)
        if event < total_mutation:
            lineage = active[int(rng.integers(0, len(active)))]
            counts = np.zeros(num_demes, dtype=np.uint32)
            for deme, count in descendants[lineage].items():  # type: ignore[union-attr]
                counts[int(deme)] = int(count)
            total = int(counts.sum())
            if 0 < total < total_haploids:
                return counts
            return None
        event -= total_mutation

        if event < total_coalescence:
            deme = int(rng.choice(num_demes, p=coalescence_rates / total_coalescence))
            items = deme_lineages[deme]
            first, second = rng.choice(len(items), size=2, replace=False)
            if first > second:
                first, second = second, first
            keep = items[first]
            remove = items[second]
            items.pop(second)
            descendants[keep] = _merge_descendant_counts(
                descendants[keep],  # type: ignore[arg-type]
                descendants[remove],  # type: ignore[arg-type]
            )
            descendants[remove] = None
            lineage_deme[remove] = -1
            _remove_active(active, active_pos, remove)
            continue

        if total_migration <= 0.0:
            return None
        deme = int(rng.choice(num_demes, p=migration_rates / total_migration))
        items = deme_lineages[deme]
        pos = int(rng.integers(0, len(items)))
        lineage = items.pop(pos)
        outgoing = adjacency[deme]
        outgoing_rates = np.asarray([rate for _target, rate in outgoing], dtype=np.float64)
        target_index = int(rng.choice(len(outgoing), p=outgoing_rates / outgoing_rates.sum()))
        target = outgoing[target_index][0]
        lineage_deme[lineage] = int(target)
        deme_lineages[int(target)].append(lineage)

    return None


def _simulate_one_random_marker_allele_count_python(
    *,
    sample_nodes: NDArray[np.int32],
    num_demes: int,
    q: NDArray[np.float64],
    adjacency: list[list[tuple[int, float]]],
    ploidy: int,
    mutation_rate: float,
    condition_on_mutation: bool,
    rng: np.random.Generator,
) -> NDArray[np.uint32] | None:
    deme_lineages: list[list[int]] = [[] for _ in range(num_demes)]
    lineage_deme: list[int] = []
    descendants: list[dict[int, int] | None] = []
    active: list[int] = []
    active_pos: list[int] = []

    for sample_deme in sample_nodes:
        deme = int(sample_deme)
        for _copy in range(ploidy):
            lineage = len(lineage_deme)
            lineage_deme.append(deme)
            descendants.append({deme: 1})
            deme_lineages[deme].append(lineage)
            active_pos.append(len(active))
            active.append(lineage)

    total_haploids = len(active)
    if total_haploids < 2:
        return None

    branch_mass = 0.0
    selected: dict[int, int] | None = None

    while len(active) > 1:
        coalescence_rates = np.asarray(
            [
                float(q[deme]) * len(items) * (len(items) - 1) * 0.5
                for deme, items in enumerate(deme_lineages)
            ],
            dtype=np.float64,
        )
        migration_rates = np.asarray(
            [
                len(items) * sum(rate for _target, rate in adjacency[deme])
                for deme, items in enumerate(deme_lineages)
            ],
            dtype=np.float64,
        )
        total_coalescence = float(coalescence_rates.sum())
        total_migration = float(migration_rates.sum())
        total_rate = total_coalescence + total_migration
        if total_rate <= 0.0 or not np.isfinite(total_rate):
            return None

        interval_mass = float(rng.exponential(1.0 / total_rate)) * len(active)
        if interval_mass > 0.0 and np.isfinite(interval_mass):
            next_mass = branch_mass + interval_mass
            if float(rng.random() * next_mass) < interval_mass:
                lineage = active[int(rng.integers(0, len(active)))]
                selected = dict(descendants[lineage] or {})
            branch_mass = next_mass

        event = float(rng.random() * total_rate)
        if event < total_coalescence:
            deme = int(rng.choice(num_demes, p=coalescence_rates / total_coalescence))
            items = deme_lineages[deme]
            first, second = rng.choice(len(items), size=2, replace=False)
            if first > second:
                first, second = second, first
            keep = items[first]
            remove = items[second]
            items.pop(second)
            descendants[keep] = _merge_descendant_counts(
                descendants[keep],  # type: ignore[arg-type]
                descendants[remove],  # type: ignore[arg-type]
            )
            descendants[remove] = None
            lineage_deme[remove] = -1
            _remove_active(active, active_pos, remove)
            continue

        if total_migration <= 0.0:
            return None
        deme = int(rng.choice(num_demes, p=migration_rates / total_migration))
        items = deme_lineages[deme]
        pos = int(rng.integers(0, len(items)))
        lineage = items.pop(pos)
        outgoing = adjacency[deme]
        outgoing_rates = np.asarray([rate for _target, rate in outgoing], dtype=np.float64)
        target_index = int(rng.choice(len(outgoing), p=outgoing_rates / outgoing_rates.sum()))
        target = outgoing[target_index][0]
        lineage_deme[lineage] = int(target)
        deme_lineages[int(target)].append(lineage)

    if selected is None or branch_mass <= 0.0 or not np.isfinite(branch_mass):
        return None
    if not condition_on_mutation:
        mutation_probability = 1.0 - float(np.exp(-float(mutation_rate) * branch_mass))
        if float(rng.random()) >= mutation_probability:
            return None
    counts = np.zeros(num_demes, dtype=np.uint32)
    for deme, count in selected.items():
        counts[int(deme)] = int(count)
    total = int(counts.sum())
    if 0 < total < total_haploids:
        return counts
    return None


def _counts_to_genotypes(
    *,
    allele_counts: NDArray[np.uint32],
    sample_nodes: NDArray[np.int32],
    ploidy: int,
    rng: np.random.Generator,
) -> NDArray[np.uint8]:
    genotypes = np.zeros((sample_nodes.shape[0], allele_counts.shape[0]), dtype=np.uint8)
    for snp_index in range(allele_counts.shape[0]):
        counts = allele_counts[snp_index]
        for deme in np.flatnonzero(np.bincount(sample_nodes, minlength=counts.shape[0]) > 0):
            sample_indices = np.flatnonzero(sample_nodes == int(deme))
            haploid_count = sample_indices.shape[0] * int(ploidy)
            derived = int(counts[int(deme)])
            if derived < 0 or derived > haploid_count:
                raise ValueError("allele_counts are inconsistent with sample_nodes and ploidy")
            haplotypes = np.zeros(haploid_count, dtype=np.uint8)
            haplotypes[:derived] = 1
            rng.shuffle(haplotypes)
            reshaped = haplotypes.reshape(sample_indices.shape[0], int(ploidy))
            genotypes[sample_indices, snp_index] = reshaped.sum(axis=1)
    return genotypes


def _simulate_island_model_python(
    *,
    sample_nodes: NDArray[np.int32],
    num_demes: int,
    q: NDArray[np.float64],
    migration_edges: NDArray[np.int32],
    migration_rates: NDArray[np.float64],
    num_snps: int,
    ploidy: int,
    mutation_rate: float,
    symmetric: bool,
    seed: int | None,
    max_attempts: int,
    min_allele_frequency: float,
    max_allele_frequency: float,
    marker_model: MarkerModel,
) -> IslandSimulationResult:
    rng = np.random.default_rng(seed)
    adjacency = _directed_adjacency(
        num_demes=num_demes,
        migration_edges=migration_edges,
        migration_rates=migration_rates,
        symmetric=symmetric,
    )
    allele_counts = np.zeros((num_snps, num_demes), dtype=np.uint32)
    accepted = 0
    attempts = 0
    while accepted < num_snps and attempts < max_attempts:
        attempts += 1
        if marker_model in {"random_marker", "conditioned_random_marker"}:
            counts = _simulate_one_random_marker_allele_count_python(
                sample_nodes=sample_nodes,
                num_demes=num_demes,
                q=q,
                adjacency=adjacency,
                ploidy=ploidy,
                mutation_rate=mutation_rate,
                condition_on_mutation=marker_model == "conditioned_random_marker",
                rng=rng,
            )
        else:
            counts = _simulate_one_allele_count_python(
                sample_nodes=sample_nodes,
                num_demes=num_demes,
                q=q,
                adjacency=adjacency,
                ploidy=ploidy,
                mutation_rate=mutation_rate,
                rng=rng,
            )
        if counts is None:
            continue
        if not _counts_pass_allele_frequency(
            counts,
            total_haploids=sample_nodes.shape[0] * int(ploidy),
            min_allele_frequency=min_allele_frequency,
            max_allele_frequency=max_allele_frequency,
        ):
            continue
        allele_counts[accepted] = counts
        accepted += 1
    if accepted < num_snps:
        raise RuntimeError(
            f"island simulator accepted {accepted} segregating SNPs after {attempts} attempts; "
            f"requested {num_snps}"
        )
    genotypes = _counts_to_genotypes(
        allele_counts=allele_counts,
        sample_nodes=sample_nodes,
        ploidy=ploidy,
        rng=rng,
    )
    return IslandSimulationResult(
        genotypes=genotypes,
        allele_counts=allele_counts,
        attempts=attempts,
        used_rust=False,
    )


def simulate_island_model(
    *,
    sample_nodes: NDArray[np.integer[Any]],
    num_demes: int,
    q: NDArray[np.floating[Any]],
    migration_edges: NDArray[np.integer[Any]],
    migration_rates: NDArray[np.floating[Any]],
    num_snps: int,
    ploidy: int = 2,
    mutation_rate: float = 0.01,
    symmetric: bool = True,
    seed: int | None = None,
    max_attempts: int | None = None,
    use_rust: bool = True,
    min_allele_frequency: float = 0.0,
    max_allele_frequency: float = 1.0,
    marker_model: str = "first_mutation",
) -> IslandSimulationResult:
    """Simulate independent segregating SNPs under a sparse island coalescent.

    `sample_nodes` indexes sampled polyploid individuals by deme. The simulator
    traces haploid lineages backward in time. A SNP is accepted only if a mutation
    hits before the sample reaches its MRCA; the derived allele count per sampled
    deme is then converted to individual genotype counts by sampling haploids
    without replacement inside each deme.
    """

    samples, q_array, edges, rates = _check_inputs(
        sample_nodes=sample_nodes,
        num_demes=int(num_demes),
        q=q,
        migration_edges=migration_edges,
        migration_rates=migration_rates,
        num_snps=int(num_snps),
        ploidy=int(ploidy),
        mutation_rate=float(mutation_rate),
    )
    min_frequency, max_frequency, resolved_marker_model = _check_ascertainment(
        min_allele_frequency=min_allele_frequency,
        max_allele_frequency=max_allele_frequency,
        marker_model=marker_model,
    )
    attempts = int(max_attempts or max(1000, int(num_snps) * 2000))
    if use_rust and _island_sim is not None:
        genotypes, allele_counts, rust_attempts = _island_sim.simulate_island_model(
            samples,
            int(num_demes),
            q_array,
            edges,
            rates,
            int(num_snps),
            int(ploidy),
            float(mutation_rate),
            bool(symmetric),
            seed,
            attempts,
            min_frequency,
            max_frequency,
            resolved_marker_model,
        )
        return IslandSimulationResult(
            genotypes=np.asarray(genotypes, dtype=np.uint8),
            allele_counts=np.asarray(allele_counts, dtype=np.uint32),
            attempts=int(rust_attempts),
            used_rust=True,
        )
    return _simulate_island_model_python(
        sample_nodes=samples,
        num_demes=int(num_demes),
        q=q_array,
        migration_edges=edges,
        migration_rates=rates,
        num_snps=int(num_snps),
        ploidy=int(ploidy),
        mutation_rate=float(mutation_rate),
        symmetric=bool(symmetric),
        seed=seed,
        max_attempts=attempts,
        min_allele_frequency=min_frequency,
        max_allele_frequency=max_frequency,
        marker_model=resolved_marker_model,
    )


def simulate_island_covariance(
    *,
    sample_nodes: NDArray[np.integer[Any]],
    num_demes: int,
    q: NDArray[np.floating[Any]],
    migration_edges: NDArray[np.integer[Any]],
    migration_rates: NDArray[np.floating[Any]],
    num_snps: int,
    ploidy: int = 2,
    mutation_rate: float = 0.01,
    symmetric: bool = True,
    seed: int | None = None,
    max_attempts: int | None = None,
    use_rust: bool = True,
    min_allele_frequency: float = 0.0,
    max_allele_frequency: float = 1.0,
    marker_model: str = "first_mutation",
) -> IslandCovarianceResult:
    """Simulate directly to a sample covariance matrix.

    The returned covariance uses the same convention as DREEMS inference:
    genotype dosage columns are centered, divided by their sample standard
    deviation with `ddof=1`, and accumulated as `X @ X.T / effective_snps`.
    `num_snps` is the requested number of usable standardized SNP columns.
    """

    samples, q_array, edges, rates = _check_inputs(
        sample_nodes=sample_nodes,
        num_demes=int(num_demes),
        q=q,
        migration_edges=migration_edges,
        migration_rates=migration_rates,
        num_snps=int(num_snps),
        ploidy=int(ploidy),
        mutation_rate=float(mutation_rate),
    )
    min_frequency, max_frequency, resolved_marker_model = _check_ascertainment(
        min_allele_frequency=min_allele_frequency,
        max_allele_frequency=max_allele_frequency,
        marker_model=marker_model,
    )
    attempts = int(max_attempts or max(1000, int(num_snps) * 2000))
    if use_rust and _island_sim is not None:
        covariance, effective_snps, rust_attempts = _island_sim.simulate_island_covariance(
            samples,
            int(num_demes),
            q_array,
            edges,
            rates,
            int(num_snps),
            int(ploidy),
            float(mutation_rate),
            bool(symmetric),
            seed,
            attempts,
            min_frequency,
            max_frequency,
            resolved_marker_model,
        )
        return IslandCovarianceResult(
            sample_covariance=np.asarray(covariance, dtype=np.float64),
            effective_snps=int(effective_snps),
            attempts=int(rust_attempts),
            used_rust=True,
        )

    result = _simulate_island_model_python(
        sample_nodes=samples,
        num_demes=int(num_demes),
        q=q_array,
        migration_edges=edges,
        migration_rates=rates,
        num_snps=int(num_snps),
        ploidy=int(ploidy),
        mutation_rate=float(mutation_rate),
        symmetric=bool(symmetric),
        seed=seed,
        max_attempts=attempts,
        min_allele_frequency=min_frequency,
        max_allele_frequency=max_frequency,
        marker_model=resolved_marker_model,
    )
    covariance, effective_snps = _covariance_from_genotypes(result.genotypes)
    return IslandCovarianceResult(
        sample_covariance=covariance,
        effective_snps=effective_snps,
        attempts=result.attempts,
        used_rust=False,
    )
