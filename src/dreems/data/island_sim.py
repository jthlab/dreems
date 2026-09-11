"""Structured coalescent simulation, with an optional fast Rust backend.

Migration rates run backward in time, per lineage. In deme i, each pair of
lineages coalesces at rate q[i]. Output dosages retain the caller's sample order.
"""

# ruff: noqa: F722,F821,UP037

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np
from beartype import beartype
from jaxtyping import Float64, Int32, Integer, UInt8, UInt32, jaxtyped
from jaxtyping import Real as RealArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

try:  # pragma: no cover - exercised when the optional Rust extension is built.
    from dreems import _island_sim
except ImportError:  # pragma: no cover - normal in source-only test environments.
    _island_sim = None


@jaxtyped(typechecker=beartype)
@dataclass(frozen=True)
class IslandSimulationResult:
    """Independent segregating SNPs simulated under a structured coalescent."""

    genotypes: UInt8[np.ndarray, "sample snp"]
    allele_counts: UInt32[np.ndarray, "snp deme"]
    attempts: int
    used_rust: bool


MarkerModel = Literal["first_mutation", "random_marker", "conditioned_random_marker"]


@beartype
def rust_backend_available() -> bool:
    """Return whether the compiled island-simulator backend can be imported."""

    return _island_sim is not None


@beartype
def require_rust_backend() -> None:
    """Fail before simulation when a workflow requires the Rust backend."""

    if not rust_backend_available():
        raise RuntimeError(
            "the Rust island simulator is required but dreems._island_sim is not installed; "
            "build it with `uv run maturin develop --release "
            "--manifest-path rust/island_sim/Cargo.toml`"
        )


@jaxtyped(typechecker=beartype)
@dataclass(frozen=True)
class IslandCovarianceResult:
    """Sample covariance accumulated directly from independent segregating SNPs."""

    sample_covariance: Float64[np.ndarray, "sample sample"]
    effective_snps: int
    attempts: int
    used_rust: bool


@jaxtyped(typechecker=beartype)
def _check_inputs(
    *,
    sample_nodes: Integer[np.ndarray, "sample"],
    num_demes: Integral,
    q: RealArray[np.ndarray, "deme"],
    migration_edges: Integer[np.ndarray, "edge 2"],
    migration_rates: RealArray[np.ndarray, "edge"],
    num_snps: Integral,
    ploidy: Integral,
    mutation_rate: Real,
) -> tuple[
    Int32[np.ndarray, "sample"],
    Float64[np.ndarray, "deme"],
    Int32[np.ndarray, "edge 2"],
    Float64[np.ndarray, "edge"],
]:
    """Check indices before narrowing them to the extension's int32 arrays."""
    if sample_nodes.size == 0:
        raise ValueError("sample_nodes must not be empty")
    if num_demes <= 0 or num_demes > np.iinfo(np.int32).max:
        raise ValueError("num_demes must be positive and fit in int32")
    if np.any(sample_nodes < 0) or np.any(sample_nodes >= num_demes):
        raise ValueError("sample_nodes entries must be in [0, num_demes)")
    if num_snps <= 0:
        raise ValueError("num_snps must be positive")
    if ploidy <= 0 or ploidy > np.iinfo(np.uint8).max:
        raise ValueError("ploidy must be in [1, 255] for uint8 genotype dosages")
    if sample_nodes.size * int(ploidy) < 2:
        raise ValueError("at least two sampled haploid copies are required")
    if mutation_rate <= 0.0 or not np.isfinite(mutation_rate):
        raise ValueError("mutation_rate must be finite and positive")

    q_array = np.asarray(q, dtype=np.float64)
    if q_array.shape != (int(num_demes),):
        raise ValueError("q must have shape (num_demes,)")
    if np.any(~np.isfinite(q_array)) or np.any(q_array < 0.0):
        raise ValueError("q must contain finite non-negative coalescence rates")

    if np.any(migration_edges < 0) or np.any(migration_edges >= num_demes):
        raise ValueError("migration_edges entries must be in [0, num_demes)")
    rates = np.asarray(migration_rates, dtype=np.float64)
    if np.any(~np.isfinite(rates)) or np.any(rates < 0.0):
        raise ValueError("migration_rates must contain finite non-negative values")
    return (
        np.ascontiguousarray(sample_nodes, dtype=np.int32),
        np.ascontiguousarray(q_array),
        np.ascontiguousarray(migration_edges, dtype=np.int32),
        np.ascontiguousarray(rates),
    )


@beartype
def _check_ascertainment(
    *,
    min_allele_frequency: Real,
    max_allele_frequency: Real,
    marker_model: str,
) -> tuple[float, float, MarkerModel]:
    """Resolve saved marker aliases without changing their sampling law."""
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


@jaxtyped(typechecker=beartype)
def _counts_pass_allele_frequency(
    counts: UInt32[np.ndarray, "deme"],
    *,
    total_haploids: int,
    min_allele_frequency: float,
    max_allele_frequency: float,
) -> bool:
    """Apply the derived-allele interval before assigning copies to individuals."""
    frequency = float(np.sum(counts, dtype=np.uint64)) / float(total_haploids)
    return min_allele_frequency <= frequency <= max_allele_frequency


@jaxtyped(typechecker=beartype)
def _directed_adjacency(
    *,
    num_demes: int,
    migration_edges: Int32[np.ndarray, "edge 2"],
    migration_rates: Float64[np.ndarray, "edge"],
    symmetric: bool,
) -> list[list[tuple[int, float]]]:
    """Collect outgoing rates, adding reverse edges only in symmetric mode."""
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(num_demes)]
    for (left, right), rate in zip(migration_edges, migration_rates, strict=True):
        if rate <= 0.0 or int(left) == int(right):
            continue
        adjacency[int(left)].append((int(right), float(rate)))
        if symmetric:
            adjacency[int(right)].append((int(left), float(rate)))
    return adjacency


@beartype
def _merge_descendant_counts(left: dict[int, int], right: dict[int, int]) -> dict[int, int]:
    """Merge the smaller descendant map into the larger one in place."""
    if len(left) < len(right):
        left, right = right, left
    for key, value in right.items():
        left[key] = left.get(key, 0) + value
    return left


@beartype
def _remove_active(active: list[int], active_pos: list[int], lineage: int) -> None:
    """Remove a lineage in constant time; maintain its reverse position map."""
    pos = active_pos[lineage]
    last = active.pop()
    if pos < len(active):
        active[pos] = last
        active_pos[last] = pos
    active_pos[lineage] = -1


@jaxtyped(typechecker=beartype)
def _simulate_counts_python(
    *,
    sample_nodes: Int32[np.ndarray, "sample"],
    num_demes: int,
    q: Float64[np.ndarray, "deme"],
    adjacency: list[list[tuple[int, float]]],
    ploidy: int,
    mutation_rate: float,
    marker_model: MarkerModel,
    rng: np.random.Generator,
) -> UInt32[np.ndarray, "deme"] | None:
    """Run one genealogy attempt, returning descendant counts by sampling deme.

    First-mutation mode samples the competing event directly. The other modes
    keep one branch in a reservoir, weighted by length, while completing the tree.
    Event and RNG order match the original separate Python implementations.
    """
    deme_lineages: list[list[int]] = [[] for _ in range(num_demes)]
    descendants: list[dict[int, int]] = []
    active: list[int] = []
    active_pos: list[int] = []

    for sample_deme in sample_nodes:
        deme = int(sample_deme)
        for _copy in range(ploidy):
            lineage = len(descendants)
            descendants.append({deme: 1})
            deme_lineages[deme].append(lineage)
            active_pos.append(len(active))
            active.append(lineage)

    total_haploids = len(active)
    if total_haploids < 2:
        return None

    branch_mass = 0.0
    selected: dict[int, int] | None = None
    first_mutation = marker_model == "first_mutation"

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
        total_mutation = float(mutation_rate) * len(active) if first_mutation else 0.0
        total_rate = total_coalescence + total_migration + total_mutation
        if total_rate <= 0.0 or not np.isfinite(total_rate):
            return None

        if not first_mutation:
            interval_mass = float(rng.exponential(1.0 / total_rate)) * len(active)
            if interval_mass > 0.0 and np.isfinite(interval_mass):
                next_mass = branch_mass + interval_mass
                if float(rng.random() * next_mass) < interval_mass:
                    lineage = active[int(rng.integers(0, len(active)))]
                    selected = dict(descendants[lineage])
                branch_mass = next_mass

        event = float(rng.random() * total_rate)
        if first_mutation:
            if event < total_mutation:
                lineage = active[int(rng.integers(0, len(active)))]
                selected = descendants[lineage]
                break
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
                descendants[keep],
                descendants[remove],
            )
            descendants[remove] = {}
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
        deme_lineages[int(target)].append(lineage)

    if selected is None:
        return None
    if not first_mutation and (branch_mass <= 0.0 or not np.isfinite(branch_mass)):
        return None
    if marker_model == "random_marker":
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


@jaxtyped(typechecker=beartype)
def _counts_to_genotypes(
    *,
    allele_counts: UInt32[np.ndarray, "snp deme"],
    sample_nodes: Int32[np.ndarray, "sample"],
    ploidy: int,
    rng: np.random.Generator,
) -> UInt8[np.ndarray, "sample snp"]:
    """Assign allele copies without replacement within each sampling deme."""
    genotypes = np.zeros((sample_nodes.shape[0], allele_counts.shape[0]), dtype=np.uint8)
    samples_by_deme = [
        (int(deme), np.flatnonzero(sample_nodes == deme)) for deme in np.unique(sample_nodes)
    ]
    for snp_index in range(allele_counts.shape[0]):
        counts = allele_counts[snp_index]
        for deme, sample_indices in samples_by_deme:
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


@jaxtyped(typechecker=beartype)
def _simulate_python(
    *,
    sample_nodes: Int32[np.ndarray, "sample"],
    num_demes: int,
    q: Float64[np.ndarray, "deme"],
    migration_edges: Int32[np.ndarray, "edge 2"],
    migration_rates: Float64[np.ndarray, "edge"],
    num_snps: int,
    ploidy: int,
    mutation_rate: float,
    symmetric: bool,
    seed: int | None,
    max_attempts: int,
    min_allele_frequency: float,
    max_allele_frequency: float,
    marker_model: MarkerModel,
    covariance: bool,
) -> IslandSimulationResult | IslandCovarianceResult:
    """Collect segregating genotypes or stream exactly num_snps usable columns.

    Genotype output delays within-deme shuffling until after all trees, preserving
    its seeded sequence. Covariance output shuffles each accepted marker before
    checking its dosage variance, so rejected constant dosages do not count.
    """
    rng = np.random.default_rng(seed)
    adjacency = _directed_adjacency(
        num_demes=num_demes,
        migration_edges=migration_edges,
        migration_rates=migration_rates,
        symmetric=symmetric,
    )
    allele_counts = None if covariance else np.zeros((num_snps, num_demes), dtype=np.uint32)
    covariance_sum = np.zeros((len(sample_nodes), len(sample_nodes))) if covariance else None
    accepted = 0
    attempts = 0
    while accepted < num_snps and attempts < max_attempts:
        attempts += 1
        counts = _simulate_counts_python(
            sample_nodes=sample_nodes,
            num_demes=num_demes,
            q=q,
            adjacency=adjacency,
            ploidy=ploidy,
            mutation_rate=mutation_rate,
            marker_model=marker_model,
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
        if covariance_sum is not None:
            dosage = _counts_to_genotypes(
                allele_counts=counts[None, :], sample_nodes=sample_nodes, ploidy=ploidy, rng=rng
            )[:, 0].astype(np.float64)
            centered = dosage - dosage.mean()
            sum_squares = np.dot(centered, centered)
            if sum_squares <= 1e-16:
                continue
            standardized = centered / np.sqrt(sum_squares / (len(sample_nodes) - 1))
            covariance_sum += np.outer(standardized, standardized)
        else:
            assert allele_counts is not None
            allele_counts[accepted] = counts
        accepted += 1
    if accepted < num_snps:
        raise RuntimeError(
            f"island simulator accepted {accepted} {'usable' if covariance else 'segregating'} SNPs after {attempts} attempts; "
            f"requested {num_snps}"
        )
    if covariance_sum is not None:
        return IslandCovarianceResult(
            sample_covariance=covariance_sum / accepted,
            effective_snps=accepted,
            attempts=attempts,
            used_rust=False,
        )
    assert allele_counts is not None
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


@jaxtyped(typechecker=beartype)
def _check_coalescence(
    sample_nodes: Int32[np.ndarray, "sample"],
    q: Float64[np.ndarray, "deme"],
    edges: Int32[np.ndarray, "edge 2"],
    rates: Float64[np.ndarray, "edge"],
    *,
    symmetric: bool,
) -> None:
    """Require eventual common ancestry for modes that complete each genealogy.

    Every sampled lineage must reach the same closed communicating class, which
    must allow coalescence. Otherwise an attempt can run forever while lineages
    migrate without ever meeting; max_attempts does not limit events in a tree.
    """
    active = edges[rates > 0.0]
    adjacency = coo_matrix(
        (np.ones(len(active)), (active[:, 0], active[:, 1])), shape=(len(q), len(q))
    ).tocsr()
    if symmetric:
        adjacency = adjacency + adjacency.T
    count, labels = connected_components(adjacency, directed=True, connection="strong")
    row, col = adjacency.nonzero()
    cross = labels[row] != labels[col]
    components = coo_matrix(
        (np.ones(cross.sum()), (labels[row[cross]], labels[col[cross]])), shape=(count, count)
    ).tocsr()
    terminal = np.diff(components.indptr) == 0
    reachable_sinks: set[int] = set()
    for label in np.unique(labels[sample_nodes]):
        reached = breadth_first_order(
            components, int(label), directed=True, return_predecessors=False
        )
        reachable_sinks.update(reached[terminal[reached]].tolist())
    if len(reachable_sinks) != 1 or not np.any(q[labels == next(iter(reachable_sinks))] > 0.0):
        raise ValueError(
            "full-genealogy marker modes require all sampled lineages to reach one "
            "closed migration class with a positive coalescence rate"
        )


@jaxtyped(typechecker=beartype)
def _simulate(
    *,
    sample_nodes: Integer[np.ndarray, "sample"],
    num_demes: Integral,
    q: RealArray[np.ndarray, "deme"],
    migration_edges: Integer[np.ndarray, "edge 2"],
    migration_rates: RealArray[np.ndarray, "edge"],
    num_snps: Integral,
    ploidy: Integral = 2,
    mutation_rate: Real = 0.01,
    symmetric: bool = True,
    seed: Integral | None = None,
    max_attempts: Integral | None = None,
    use_rust: bool = True,
    min_allele_frequency: Real = 0.0,
    max_allele_frequency: Real = 1.0,
    marker_model: str = "first_mutation",
    covariance: bool,
) -> IslandSimulationResult | IslandCovarianceResult:
    """Validate once, choose the backend, and wrap its three output values."""

    samples, q_array, edges, rates = _check_inputs(
        sample_nodes=sample_nodes,
        num_demes=num_demes,
        q=q,
        migration_edges=migration_edges,
        migration_rates=migration_rates,
        num_snps=num_snps,
        ploidy=ploidy,
        mutation_rate=mutation_rate,
    )
    min_frequency, max_frequency, resolved_marker_model = _check_ascertainment(
        min_allele_frequency=min_allele_frequency,
        max_allele_frequency=max_allele_frequency,
        marker_model=marker_model,
    )
    if covariance and samples.size < 2:
        raise ValueError("covariance requires at least two samples")
    if max_attempts is not None and max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if seed is not None and not 0 <= seed <= np.iinfo(np.uint64).max:
        raise ValueError("seed must fit in uint64")
    seed = None if seed is None else int(seed)
    attempts = int(max(1000, int(num_snps) * 2000) if max_attempts is None else max_attempts)
    if resolved_marker_model != "first_mutation":
        _check_coalescence(samples, q_array, edges, rates, symmetric=symmetric)
    if use_rust and _island_sim is not None:
        backend = (
            _island_sim.simulate_island_covariance
            if covariance
            else _island_sim.simulate_island_model
        )
        values, counts, rust_attempts = backend(
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
        if covariance:
            return IslandCovarianceResult(
                sample_covariance=np.asarray(values, dtype=np.float64),
                effective_snps=int(counts),
                attempts=int(rust_attempts),
                used_rust=True,
            )
        return IslandSimulationResult(
            genotypes=np.asarray(values, dtype=np.uint8),
            allele_counts=np.asarray(counts, dtype=np.uint32),
            attempts=int(rust_attempts),
            used_rust=True,
        )
    return _simulate_python(
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
        covariance=covariance,
    )


@jaxtyped(typechecker=beartype)
def simulate_island_model(
    *,
    sample_nodes: Integer[np.ndarray, "sample"],
    num_demes: Integral,
    q: RealArray[np.ndarray, "deme"],
    migration_edges: Integer[np.ndarray, "edge 2"],
    migration_rates: RealArray[np.ndarray, "edge"],
    num_snps: Integral,
    ploidy: Integral = 2,
    mutation_rate: Real = 0.01,
    symmetric: bool = True,
    seed: Integral | None = None,
    max_attempts: Integral | None = None,
    use_rust: bool = True,
    min_allele_frequency: Real = 0.0,
    max_allele_frequency: Real = 1.0,
    marker_model: str = "first_mutation",
) -> IslandSimulationResult:
    """Simulate segregating markers and assign copies within sampling demes.

    ``first_mutation`` stops at the first mutation backward in time and rejects
    trees that coalesce first. ``random_marker`` completes a tree, accepts it
    with probability 1-exp(-mutation_rate * total_branch_length), then uses a
    uniformly placed marker on its branches. ``conditioned_random_marker`` uses
    one such marker on every completed tree, without that acceptance weighting;
    its distribution does not depend on mutation_rate. These are distinct laws.

    Allele-frequency bounds apply to derived counts; symmetric bounds such as
    [0.05, 0.95] implement a MAF filter. Samples may have fixed ploidy 1-255.
    Each deme's copies are assigned to individuals without replacement, preserving
    sample order. A segregating marker can still have constant individual dosage.
    Missing Rust uses the slower Python backend unless require_rust_backend was
    called first. Seeds are reproducible within a backend, not between backends.
    """
    return _simulate(
        sample_nodes=sample_nodes,
        num_demes=num_demes,
        q=q,
        migration_edges=migration_edges,
        migration_rates=migration_rates,
        num_snps=num_snps,
        ploidy=ploidy,
        mutation_rate=mutation_rate,
        symmetric=symmetric,
        seed=seed,
        max_attempts=max_attempts,
        use_rust=use_rust,
        min_allele_frequency=min_allele_frequency,
        max_allele_frequency=max_allele_frequency,
        marker_model=marker_model,
        covariance=False,
    )


@jaxtyped(typechecker=beartype)
def simulate_island_covariance(
    *,
    sample_nodes: Integer[np.ndarray, "sample"],
    num_demes: Integral,
    q: RealArray[np.ndarray, "deme"],
    migration_edges: Integer[np.ndarray, "edge 2"],
    migration_rates: RealArray[np.ndarray, "edge"],
    num_snps: Integral,
    ploidy: Integral = 2,
    mutation_rate: Real = 0.01,
    symmetric: bool = True,
    seed: Integral | None = None,
    max_attempts: Integral | None = None,
    use_rust: bool = True,
    min_allele_frequency: Real = 0.0,
    max_allele_frequency: Real = 1.0,
    marker_model: str = "first_mutation",
) -> IslandCovarianceResult:
    """Simulate directly to a sample covariance matrix.

    The returned covariance uses the same convention as DREEMS inference:
    genotype dosage columns are centered, divided by their sample standard
    deviation with `ddof=1`, and accumulated as `X @ X.T / effective_snps`.
    `num_snps` is the requested number of usable standardized SNP columns.
    Both backends reject constant dosage columns and continue until this count
    or max_attempts is reached. At least two individuals are required. Marker
    modes and frequency filters are the same as in simulate_island_model.
    """
    return _simulate(
        sample_nodes=sample_nodes,
        num_demes=num_demes,
        q=q,
        migration_edges=migration_edges,
        migration_rates=migration_rates,
        num_snps=num_snps,
        ploidy=ploidy,
        mutation_rate=mutation_rate,
        symmetric=symmetric,
        seed=seed,
        max_attempts=max_attempts,
        use_rust=use_rust,
        min_allele_frequency=min_allele_frequency,
        max_allele_frequency=max_allele_frequency,
        marker_model=marker_model,
        covariance=True,
    )
