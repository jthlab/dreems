"""Simulator contracts that do not depend on a trained checkpoint."""

from __future__ import annotations

import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from jaxtyping import TypeCheckError

from dreems import coalescent, island_sim


@pytest.fixture
def arguments() -> dict:
    return dict(
        sample_nodes=np.asarray([2, 0, 1, 0], dtype=np.int32),
        num_demes=3,
        q=np.asarray([1.0, 0.7, 1.2]),
        migration_edges=np.asarray([[0, 1], [1, 2]], dtype=np.int32),
        migration_rates=np.asarray([0.4, 0.7]),
        num_snps=10,
        ploidy=2,
        seed=7291,
        marker_model="conditioned_random_marker",
        use_rust=False,
    )


@pytest.mark.parametrize("covariance", [False, True])
@pytest.mark.parametrize("field", ["sample_nodes", "migration_edges"])
def test_fractional_indices_are_not_truncated(
    arguments: dict, covariance: bool, field: str
) -> None:
    arguments[field] = arguments[field].astype(float) + 0.3
    simulate = (
        island_sim.simulate_island_covariance if covariance else island_sim.simulate_island_model
    )
    with pytest.raises((BeartypeCallHintParamViolation, TypeCheckError)):
        simulate(**arguments)


@pytest.mark.parametrize("field", ["sample_nodes", "migration_edges"])
def test_large_indices_are_checked_before_int32_cast(arguments: dict, field: str) -> None:
    arguments[field] = arguments[field].astype(np.uint64)
    arguments[field].flat[0] = 2**32
    with pytest.raises(ValueError, match=field):
        island_sim.simulate_island_model(**arguments)


@pytest.mark.parametrize("field", ["num_demes", "num_snps", "ploidy", "max_attempts"])
def test_fractional_counts_are_rejected(arguments: dict, field: str) -> None:
    arguments[field] = 2.3
    with pytest.raises((BeartypeCallHintParamViolation, TypeCheckError)):
        island_sim.simulate_island_model(**arguments)


@pytest.mark.parametrize(
    ("field", "value"),
    [("ploidy", 256), ("max_attempts", 0), ("max_attempts", -1), ("seed", -1), ("seed", 2**64)],
)
def test_invalid_bounds_fail_before_simulation(arguments: dict, field: str, value: int) -> None:
    arguments[field] = value
    with pytest.raises(ValueError, match=field):
        island_sim.simulate_island_model(**arguments)


@pytest.mark.parametrize("use_rust", [False, True])
@pytest.mark.parametrize("index_dtype", [np.int32, np.uint64])
def test_numpy_scalars_and_strided_inputs(
    arguments: dict, use_rust: bool, index_dtype: type
) -> None:
    if use_rust and not island_sim.rust_backend_available():
        pytest.skip("Rust extension is not installed")
    arguments["use_rust"] = use_rust
    expected = island_sim.simulate_island_model(**arguments)
    arguments["sample_nodes"] = arguments["sample_nodes"].astype(index_dtype)
    arguments["migration_edges"] = arguments["migration_edges"].astype(index_dtype)
    for field in ("sample_nodes", "q", "migration_rates"):
        arguments[field] = np.repeat(arguments[field], 2)[::2]
        assert not arguments[field].flags.c_contiguous
    for field in ("num_demes", "num_snps", "ploidy", "seed"):
        arguments[field] = np.int64(arguments[field])
    actual = island_sim.simulate_island_model(**arguments)
    np.testing.assert_array_equal(actual.genotypes, expected.genotypes)
    np.testing.assert_array_equal(actual.allele_counts, expected.allele_counts)
    assert actual.attempts == expected.attempts


@pytest.mark.parametrize("ploidy", [2, 4])
def test_python_covariance_counts_only_variable_dosages(
    monkeypatch: pytest.MonkeyPatch, ploidy: int
) -> None:
    # The tetraploid columns also expose the old diploid-only polymorphism test.
    base = ploidy - 1
    counts = iter(
        [
            np.asarray([base, base], dtype=np.uint32),
            np.asarray([base - 1, base + 1], dtype=np.uint32),
            np.asarray([base + 1, base - 1], dtype=np.uint32),
        ]
    )
    monkeypatch.setattr(island_sim, "_simulate_counts_python", lambda **kwargs: next(counts))
    result = island_sim.simulate_island_covariance(
        sample_nodes=np.asarray([0, 1], dtype=np.int32),
        num_demes=2,
        q=np.ones(2),
        migration_edges=np.asarray([[0, 1]], dtype=np.int32),
        migration_rates=np.ones(1),
        num_snps=2,
        ploidy=ploidy,
        seed=12,
        max_attempts=3,
        use_rust=False,
    )
    assert result.attempts == 3
    assert result.effective_snps == 2
    np.testing.assert_allclose(result.sample_covariance, [[0.5, -0.5], [-0.5, 0.5]], atol=1e-14)


def test_python_covariance_honors_attempt_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        island_sim, "_simulate_counts_python", lambda **kwargs: np.asarray([1, 1], dtype=np.uint32)
    )
    with pytest.raises(RuntimeError, match="accepted 0 usable SNPs after 3 attempts"):
        island_sim.simulate_island_covariance(
            sample_nodes=np.asarray([0, 1], dtype=np.int32),
            num_demes=2,
            q=np.ones(2),
            migration_edges=np.asarray([[0, 1]], dtype=np.int32),
            migration_rates=np.ones(1),
            num_snps=2,
            max_attempts=3,
            use_rust=False,
        )


@pytest.mark.parametrize("use_rust", [False, True])
def test_covariance_requires_two_individuals(arguments: dict, use_rust: bool) -> None:
    arguments.update(sample_nodes=np.asarray([0], dtype=np.int32), use_rust=use_rust)
    with pytest.raises(ValueError, match="at least two samples"):
        island_sim.simulate_island_covariance(**arguments)


@pytest.mark.parametrize("kind", ["zero_q", "disconnected", "two_sinks", "sink_without_q"])
@pytest.mark.parametrize("use_rust", [False, True])
def test_full_genealogy_rejects_noncoalescing_migration(
    arguments: dict, kind: str, use_rust: bool
) -> None:
    arguments["use_rust"] = use_rust
    if kind == "zero_q":
        arguments["q"] = np.zeros(3)
    elif kind == "disconnected":
        arguments["migration_rates"] = np.zeros(2)
    elif kind == "two_sinks":
        arguments.update(
            migration_edges=np.asarray([[0, 1], [0, 2]], dtype=np.int32), symmetric=False
        )
    else:
        arguments.update(q=np.asarray([1.0, 1.0, 0.0]), symmetric=False)
    with pytest.raises(ValueError, match="closed migration class"):
        island_sim.simulate_island_model(**arguments)


def test_directed_migration_with_one_coalescing_sink_is_valid(arguments: dict) -> None:
    arguments["symmetric"] = False
    result = island_sim.simulate_island_model(**arguments)
    assert result.genotypes.shape == (4, 10)


def test_first_mutation_does_not_require_eventual_coalescence(arguments: dict) -> None:
    arguments.update(q=np.zeros(3), migration_rates=np.zeros(2), marker_model="first_mutation")
    result = island_sim.simulate_island_model(**arguments)
    assert result.attempts == 10
    np.testing.assert_array_equal(result.genotypes.sum(axis=0), np.ones(10))


@pytest.mark.parametrize("ploidy", [1, 2, 4])
def test_grouped_haplotypes_preserve_sample_order(ploidy: int) -> None:
    rng = np.random.default_rng(501)
    samples = np.asarray([2, 0, 2, 1, 0], dtype=np.int32)
    values = rng.integers(0, 2, size=(17, len(samples) * ploidy), dtype=np.int8)
    expected = np.zeros((len(samples), 17), dtype=np.int8)
    cursor = 0
    for deme in range(3):
        for index in np.flatnonzero(samples == deme):
            expected[index] = values[:, cursor : cursor + ploidy].sum(axis=1)
            cursor += ploidy
    result = coalescent.genotypes_from_grouped_haplotypes(
        values, sample_nodes=samples, num_nodes=3, ploidy=ploidy
    )
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == expected.dtype


def test_conditioned_marker_rust_matches_msprime_branch_sampling() -> None:
    if not island_sim.rust_backend_available():
        pytest.skip("Rust extension is not installed")
    msprime = pytest.importorskip("msprime")
    snps = 5000
    q = np.asarray([1.0, 0.7])
    samples = np.asarray([1, 0, 1, 0], dtype=np.int32)
    result = island_sim.simulate_island_model(
        sample_nodes=samples,
        num_demes=2,
        q=q,
        migration_edges=np.asarray([[0, 1]], dtype=np.int32),
        migration_rates=np.asarray([0.4]),
        num_snps=snps,
        ploidy=2,
        seed=2381,
        marker_model="conditioned_random_marker",
        use_rust=True,
    )
    populations = [
        msprime.PopulationConfiguration(sample_size=4, initial_size=1 / (2 * rate)) for rate in q
    ]
    trees = msprime.simulate(
        population_configurations=populations,
        migration_matrix=[[0.0, 0.4], [0.4, 0.0]],
        length=1.0,
        mutation_rate=0.0,
        num_replicates=snps,
        Ne=1.0,
        random_seed=416,
    )
    rng = np.random.default_rng(417)
    reference = np.zeros((snps, 2), dtype=np.uint32)
    for index, ts in enumerate(trees):
        tree = ts.first()
        branches = [node for node in tree.nodes() if tree.parent(node) != -1]
        length = np.asarray([tree.branch_length(node) for node in branches])
        chosen = rng.choice(branches, p=length / length.sum())
        for sample in tree.samples(chosen):
            reference[index, ts.node(sample).population] += 1
    # Compare deme means and the site-frequency spectrum, not seeded realizations.
    np.testing.assert_allclose(result.allele_counts.mean(axis=0), reference.mean(axis=0), atol=0.12)
    actual_hist = np.bincount(result.allele_counts.sum(axis=1).astype(int), minlength=9) / snps
    expected_hist = np.bincount(reference.sum(axis=1).astype(int), minlength=9) / snps
    np.testing.assert_allclose(actual_hist, expected_hist, atol=0.035)
