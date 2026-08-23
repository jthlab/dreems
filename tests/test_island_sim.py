from __future__ import annotations

import os

import numpy as np
import pytest

from dreems import features, island_sim, simulation, simulation_config


def _require_rust_extension() -> None:
    if island_sim._island_sim is None:
        if os.environ.get("DREEMS_REQUIRE_RUST") == "1":
            pytest.fail("Rust island simulator extension is required but not installed")
        pytest.skip("Rust island simulator extension is not installed")


def test_required_rust_backend_fails_before_simulation(monkeypatch) -> None:
    monkeypatch.setattr(island_sim, "_island_sim", None)
    sim = simulation_config.simulation_config(
        {
            "simulation": {
                "simulator": "island_coalescent",
                "island_use_rust": True,
                "island_require_rust": True,
                "island_emit_covariance": True,
            }
        }
    )
    case = {
        "graph": {"edges": np.asarray([[0, 1]], dtype=np.int32)},
        "active_positions": np.zeros((2, 2), dtype=np.float64),
        "edge_log_rates": np.zeros(1, dtype=np.float64),
        "node_log_q": np.zeros(2, dtype=np.float64),
        "sample_nodes": np.asarray([0, 1], dtype=np.int32),
    }

    with pytest.raises(RuntimeError, match="Rust island simulator is required"):
        simulation.simulate_case_genotypes(
            np.random.default_rng(17),
            sim,
            case,
            num_snps=10,
        )


def symmetric_complete_island_edges(num_demes: int) -> np.ndarray:
    return np.asarray(
        [(left, right) for left in range(num_demes) for right in range(left + 1, num_demes)],
        dtype=np.int32,
    )


def two_lineage_mutation_before_coalescence_probability(
    *,
    num_demes: int,
    coalescence_rate: float,
    per_target_migration_rate: float,
    mutation_rate: float,
    start_same_deme: bool,
) -> float:
    """Closed-form absorption probability for a symmetric complete island model."""

    migration_out = 2.0 * float(num_demes - 1) * float(per_target_migration_rate)
    to_same = 2.0 * float(per_target_migration_rate)
    system = np.asarray(
        [
            [
                float(coalescence_rate) + 2.0 * float(mutation_rate) + migration_out,
                -migration_out,
            ],
            [-to_same, 2.0 * float(mutation_rate) + to_same],
        ],
        dtype=np.float64,
    )
    right_hand_side = np.asarray([2.0 * mutation_rate, 2.0 * mutation_rate], dtype=np.float64)
    same_probability, different_probability = np.linalg.solve(system, right_hand_side)
    return float(same_probability if start_same_deme else different_probability)


def msprime_first_mutation_counts(
    *,
    sample_nodes: np.ndarray,
    num_demes: int,
    q: np.ndarray,
    migration_edges: np.ndarray,
    migration_rates: np.ndarray,
    num_snps: int,
    ploidy: int,
    mutation_rate: float,
    seed: int,
) -> np.ndarray:
    msprime = pytest.importorskip("msprime")
    rng = np.random.default_rng(seed)
    sample_counts = np.bincount(sample_nodes, minlength=num_demes)
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=int(ploidy * sample_counts[deme]),
            initial_size=float(1.0 / (2.0 * q[deme])),
        )
        for deme in range(num_demes)
    ]
    migration_matrix = np.zeros((num_demes, num_demes), dtype=np.float64)
    for (left, right), rate in zip(migration_edges, migration_rates, strict=True):
        migration_matrix[int(left), int(right)] = float(rate)
        migration_matrix[int(right), int(left)] = float(rate)

    total_haploids = int(sample_nodes.shape[0] * ploidy)
    counts: list[np.ndarray] = []
    for _batch in range(100):
        if len(counts) >= num_snps:
            break
        tree_sequences = msprime.simulate(
            population_configurations=population_configurations,
            migration_matrix=migration_matrix,
            length=1.0,
            mutation_rate=0.0,
            num_replicates=max(256, num_snps - len(counts)),
            Ne=1.0,
            random_seed=int(rng.integers(1, np.iinfo(np.int32).max)),
        )
        for tree_sequence in tree_sequences:
            tree = tree_sequence.first()
            total_branch_length = float(tree.total_branch_length)
            if total_branch_length <= 0.0:
                continue
            mutation_distance = float(rng.exponential(1.0 / mutation_rate))
            if mutation_distance >= total_branch_length:
                continue
            branches = []
            times = {0.0, float(tree.time(tree.root))}
            for node in tree.nodes():
                parent = tree.parent(node)
                if parent == -1:
                    continue
                child_time = float(tree.time(node))
                parent_time = float(tree.time(parent))
                if parent_time > child_time:
                    branches.append((node, child_time, parent_time))
                    times.add(child_time)
                    times.add(parent_time)
            remaining = mutation_distance
            chosen = None
            sorted_times = sorted(times)
            for low, high in zip(sorted_times[:-1], sorted_times[1:], strict=True):
                width = high - low
                if width <= 0.0:
                    continue
                active = [
                    node
                    for node, child_time, parent_time in branches
                    if child_time <= low + 1e-12 and parent_time >= high - 1e-12
                ]
                interval_mass = len(active) * width
                if interval_mass <= 0.0:
                    continue
                if remaining < interval_mass:
                    chosen = int(rng.choice(active))
                    break
                remaining -= interval_mass
            if chosen is None:
                continue
            deme_counts = np.zeros(num_demes, dtype=np.uint32)
            for sample in tree.samples(chosen):
                deme_counts[int(tree_sequence.node(sample).population)] += 1
            total = int(deme_counts.sum())
            if 0 < total < total_haploids:
                counts.append(deme_counts)
                if len(counts) >= num_snps:
                    break
    if len(counts) < num_snps:
        raise RuntimeError(f"msprime accepted {len(counts)} SNPs; requested {num_snps}")
    return np.asarray(counts, dtype=np.uint32)


def test_island_simulator_conditions_on_segregation_and_samples_without_replacement() -> None:
    sample_nodes = np.asarray([0, 0, 1, 1], dtype=np.int32)
    result = island_sim.simulate_island_model(
        sample_nodes=sample_nodes,
        num_demes=2,
        q=np.asarray([1.0, 1.2], dtype=np.float64),
        migration_edges=np.asarray([[0, 1]], dtype=np.int32),
        migration_rates=np.asarray([0.2], dtype=np.float64),
        num_snps=12,
        ploidy=2,
        mutation_rate=2.0,
        symmetric=True,
        seed=101,
        use_rust=False,
    )
    assert result.genotypes.shape == (4, 12)
    assert result.allele_counts.shape == (12, 2)
    assert np.all(result.genotypes.sum(axis=0) > 0)
    assert np.all(result.genotypes.sum(axis=0) < 2 * sample_nodes.shape[0])

    for snp in range(result.allele_counts.shape[0]):
        for deme in range(2):
            observed = int(result.genotypes[sample_nodes == deme, snp].sum())
            assert observed == int(result.allele_counts[snp, deme])


def test_island_simulator_accepts_directed_rates() -> None:
    result = island_sim.simulate_island_model(
        sample_nodes=np.asarray([0, 1, 1], dtype=np.int32),
        num_demes=2,
        q=np.asarray([1.0, 1.0], dtype=np.float64),
        migration_edges=np.asarray([[0, 1], [1, 0]], dtype=np.int32),
        migration_rates=np.asarray([0.05, 0.4], dtype=np.float64),
        num_snps=6,
        ploidy=3,
        mutation_rate=1.5,
        symmetric=False,
        seed=202,
        use_rust=False,
    )
    assert result.genotypes.shape == (3, 6)
    assert result.genotypes.max() <= 3
    assert np.all(result.genotypes.sum(axis=0) > 0)
    assert np.all(result.genotypes.sum(axis=0) < 3 * 3)


@pytest.mark.parametrize(
    "marker_model",
    ["first_mutation", "random_marker", "conditioned_random_marker"],
)
@pytest.mark.parametrize("use_rust", [False, True])
def test_island_simulator_can_ascertain_common_markers(
    use_rust: bool,
    marker_model: str,
) -> None:
    if use_rust:
        _require_rust_extension()
    result = island_sim.simulate_island_model(
        sample_nodes=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32),
        num_demes=3,
        q=np.asarray([1.0, 0.9, 1.1], dtype=np.float64),
        migration_edges=np.asarray([[0, 1], [1, 2]], dtype=np.int32),
        migration_rates=np.asarray([0.4, 0.2], dtype=np.float64),
        num_snps=20,
        ploidy=2,
        mutation_rate=1.0,
        symmetric=True,
        seed=404,
        max_attempts=200_000,
        use_rust=use_rust,
        min_allele_frequency=0.2,
        max_allele_frequency=0.8,
        marker_model=marker_model,
    )
    assert result.used_rust is use_rust
    allele_frequency = result.genotypes.sum(axis=0) / float(2 * result.genotypes.shape[0])
    assert np.all(allele_frequency >= 0.2)
    assert np.all(allele_frequency <= 0.8)
    assert result.attempts >= 20


@pytest.mark.parametrize("use_rust", [False, True])
def test_conditioned_random_marker_skips_no_mutation_rejection(use_rust: bool) -> None:
    if use_rust:
        _require_rust_extension()

    result = island_sim.simulate_island_model(
        sample_nodes=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32),
        num_demes=3,
        q=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        migration_edges=np.asarray([[0, 1], [1, 2]], dtype=np.int32),
        migration_rates=np.asarray([0.2, 0.2], dtype=np.float64),
        num_snps=25,
        ploidy=2,
        mutation_rate=1.0e-12,
        symmetric=True,
        seed=505,
        max_attempts=25,
        use_rust=use_rust,
        marker_model="conditioned_random_marker",
    )

    assert result.used_rust is use_rust
    assert result.attempts == 25
    assert result.allele_counts.shape == (25, 3)
    assert np.all(result.allele_counts.sum(axis=1) > 0)
    assert np.all(result.allele_counts.sum(axis=1) < 2 * 6)


def test_symmetric_complete_island_acceptance_matches_closed_form() -> None:
    _require_rust_extension()
    num_demes = 5
    coalescence_rate = 1.0
    migration_rate = 0.3
    mutation_rate = 0.2
    num_snps = 4_000
    edges = symmetric_complete_island_edges(num_demes)
    edge_rates = np.full(edges.shape[0], migration_rate, dtype=np.float64)
    q = np.full(num_demes, coalescence_rate, dtype=np.float64)

    for start_same_deme, sample_nodes, seed in [
        (True, np.asarray([0, 0], dtype=np.int32), 707),
        (False, np.asarray([0, 1], dtype=np.int32), 808),
    ]:
        result = island_sim.simulate_island_model(
            sample_nodes=sample_nodes,
            num_demes=num_demes,
            q=q,
            migration_edges=edges,
            migration_rates=edge_rates,
            num_snps=num_snps,
            ploidy=1,
            mutation_rate=mutation_rate,
            symmetric=True,
            seed=seed,
            use_rust=True,
        )
        assert result.used_rust is True
        observed = float(num_snps / result.attempts)
        expected = two_lineage_mutation_before_coalescence_probability(
            num_demes=num_demes,
            coalescence_rate=coalescence_rate,
            per_target_migration_rate=migration_rate,
            mutation_rate=mutation_rate,
            start_same_deme=start_same_deme,
        )
        assert abs(observed - expected) < 0.04
        assert np.all(result.allele_counts.sum(axis=1) == 1)


def test_symmetric_complete_island_msprime_tmrca_matches_closed_form() -> None:
    msprime = pytest.importorskip("msprime")

    num_demes = 5
    coalescence_rate = 1.0
    migration_rate = 0.3
    replicates = 2_000
    migration_matrix = np.full((num_demes, num_demes), migration_rate, dtype=np.float64)
    np.fill_diagonal(migration_matrix, 0.0)

    for sample_sizes, expected_tmrca, seed in [
        ([2, 0, 0, 0, 0], num_demes / coalescence_rate, 909),
        ([1, 1, 0, 0, 0], num_demes / coalescence_rate + 1.0 / (2.0 * migration_rate), 1001),
    ]:
        population_configurations = [
            msprime.PopulationConfiguration(
                sample_size=sample_size,
                initial_size=1.0 / (2.0 * coalescence_rate),
            )
            for sample_size in sample_sizes
        ]
        tree_sequences = msprime.simulate(
            population_configurations=population_configurations,
            migration_matrix=migration_matrix,
            length=1.0,
            mutation_rate=0.0,
            num_replicates=replicates,
            Ne=1.0,
            random_seed=seed,
        )
        tmrcas = np.asarray(
            [
                tree_sequence.first().time(tree_sequence.first().root)
                for tree_sequence in tree_sequences
            ],
            dtype=np.float64,
        )
        assert abs(float(tmrcas.mean()) - expected_tmrca) < 0.5


def test_rust_island_allele_count_summaries_match_msprime() -> None:
    _require_rust_extension()
    sample_nodes = np.asarray([0, 0, 0, 1, 1, 2, 2, 2], dtype=np.int32)
    num_demes = 3
    q = np.asarray([1.0, 0.8, 1.3], dtype=np.float64)
    edges = np.asarray([[0, 1], [1, 2], [0, 2]], dtype=np.int32)
    rates = np.asarray([0.4, 0.25, 0.1], dtype=np.float64)
    num_snps = 1_500
    ploidy = 2
    mutation_rate = 0.3

    rust_result = island_sim.simulate_island_model(
        sample_nodes=sample_nodes,
        num_demes=num_demes,
        q=q,
        migration_edges=edges,
        migration_rates=rates,
        num_snps=num_snps,
        ploidy=ploidy,
        mutation_rate=mutation_rate,
        symmetric=True,
        seed=111,
        use_rust=True,
    )
    assert rust_result.used_rust is True
    rust = rust_result.allele_counts
    reference = msprime_first_mutation_counts(
        sample_nodes=sample_nodes,
        num_demes=num_demes,
        q=q,
        migration_edges=edges,
        migration_rates=rates,
        num_snps=num_snps,
        ploidy=ploidy,
        mutation_rate=mutation_rate,
        seed=222,
    )

    rust_total = rust.sum(axis=1)
    reference_total = reference.sum(axis=1)
    assert abs(float(rust_total.mean()) - float(reference_total.mean())) < 0.35
    assert np.max(np.abs(rust.mean(axis=0) - reference.mean(axis=0))) < 0.25
    rust_hist = np.bincount(rust_total, minlength=ploidy * sample_nodes.shape[0] + 1) / num_snps
    reference_hist = (
        np.bincount(reference_total, minlength=ploidy * sample_nodes.shape[0] + 1) / num_snps
    )
    assert float(np.abs(rust_hist - reference_hist).sum()) < 0.25


def test_rust_multiword_bitset_summaries_match_msprime() -> None:
    _require_rust_extension()
    sample_nodes = np.repeat(np.arange(3, dtype=np.int32), [24, 22, 20])
    num_demes = 3
    q = np.asarray([1.0, 0.9, 1.2], dtype=np.float64)
    edges = np.asarray([[0, 1], [1, 2], [0, 2]], dtype=np.int32)
    rates = np.asarray([0.35, 0.2, 0.12], dtype=np.float64)
    num_snps = 300
    ploidy = 2
    mutation_rate = 0.2

    assert sample_nodes.shape[0] * ploidy > 128
    rust_result = island_sim.simulate_island_model(
        sample_nodes=sample_nodes,
        num_demes=num_demes,
        q=q,
        migration_edges=edges,
        migration_rates=rates,
        num_snps=num_snps,
        ploidy=ploidy,
        mutation_rate=mutation_rate,
        symmetric=True,
        seed=333,
        use_rust=True,
    )
    assert rust_result.used_rust is True
    rust = rust_result.allele_counts
    reference = msprime_first_mutation_counts(
        sample_nodes=sample_nodes,
        num_demes=num_demes,
        q=q,
        migration_edges=edges,
        migration_rates=rates,
        num_snps=num_snps,
        ploidy=ploidy,
        mutation_rate=mutation_rate,
        seed=444,
    )

    rust_total = rust.sum(axis=1)
    reference_total = reference.sum(axis=1)
    assert abs(float(rust_total.mean()) - float(reference_total.mean())) < 3.0
    assert np.max(np.abs(rust.mean(axis=0) - reference.mean(axis=0))) < 2.5


def test_island_simulator_uses_rust_backend_when_installed() -> None:
    _require_rust_extension()
    result = island_sim.simulate_island_model(
        sample_nodes=np.asarray([0, 0, 1, 1], dtype=np.int32),
        num_demes=2,
        q=np.asarray([1.0, 1.0], dtype=np.float64),
        migration_edges=np.asarray([[0, 1]], dtype=np.int32),
        migration_rates=np.asarray([0.3], dtype=np.float64),
        num_snps=5,
        ploidy=2,
        mutation_rate=2.0,
        symmetric=True,
        seed=303,
        use_rust=True,
    )
    assert result.used_rust is True
    assert result.genotypes.shape == (4, 5)
    assert np.all(result.genotypes.sum(axis=0) > 0)
    assert np.all(result.genotypes.sum(axis=0) < 8)


def test_island_simulator_can_emit_sample_covariance() -> None:
    _require_rust_extension()
    result = island_sim.simulate_island_covariance(
        sample_nodes=np.asarray([0, 0, 1, 1], dtype=np.int32),
        num_demes=2,
        q=np.asarray([1.0, 1.0], dtype=np.float64),
        migration_edges=np.asarray([[0, 1]], dtype=np.int32),
        migration_rates=np.asarray([0.3], dtype=np.float64),
        num_snps=300,
        ploidy=2,
        mutation_rate=2.0,
        symmetric=True,
        seed=404,
        use_rust=True,
    )
    covariance = result.sample_covariance
    assert result.used_rust is True
    assert result.effective_snps == 300
    assert result.attempts >= 300
    assert covariance.shape == (4, 4)
    assert np.allclose(covariance, covariance.T, atol=1e-10)
    assert np.isclose(float(np.trace(covariance)), 3.0, atol=1e-10)
    assert np.linalg.eigvalsh(covariance).min() > -1e-8


def test_pipeline_can_use_island_coalescent_simulator() -> None:
    sim = simulation_config.simulation_config(
        {
            "simulation": {
                "seed": 31,
                "simulator": "island_coalescent",
                "min_height": 3,
                "max_height": 3,
                "min_width": 3,
                "max_width": 3,
                "num_snps": 6,
                "sampled_node_fraction": 0.5,
                "min_sampled_nodes": 4,
                "diploids_per_node": 1,
                "surface_kinds": ["smooth"],
                "q_log_std": 0.1,
                "island_ploidy": 2,
                "island_mutation_rate": 2.0,
                "island_use_rust": False,
            }
        }
    )
    example = simulation.simulate_raw_example(
        np.random.default_rng(31),
        sim=sim,
        split="train",
        shard=0,
        index=0,
    )
    assert example["simulator"] == "island_coalescent"
    assert example["genotypes"].shape == (example["sample_nodes"].shape[0], 6)
    assert int(example["island_attempts"]) >= 6


def test_pipeline_can_emit_island_covariance_examples() -> None:
    _require_rust_extension()
    sim = simulation_config.simulation_config(
        {
            "simulation": {
                "seed": 37,
                "simulator": "island_coalescent",
                "min_height": 3,
                "max_height": 3,
                "min_width": 3,
                "max_width": 3,
                "num_snps": 20,
                "sampled_node_fraction": 0.5,
                "min_sampled_nodes": 4,
                "diploids_per_node": 1,
                "surface_kinds": ["smooth"],
                "q_log_std": 0.1,
                "island_ploidy": 2,
                "island_mutation_rate": 2.0,
                "island_use_rust": True,
                "island_emit_covariance": True,
            }
        }
    )
    example = simulation.simulate_raw_example(
        np.random.default_rng(37),
        sim=sim,
        split="train",
        shard=0,
        index=0,
    )
    assert example["simulator"] == "island_coalescent"
    assert example["island_used_rust"] is True
    assert "genotypes" not in example
    assert example["sample_covariance"].shape == (
        example["sample_nodes"].shape[0],
        example["sample_nodes"].shape[0],
    )
    assert int(example["effective_snps"]) == 20
    prepared = features.prepare_examples(
        [example],
        covariance_components=2,
    )
    assert prepared[0]["edge_x"].shape[0] == example["graph"]["edges"].shape[0]


def island_covariance_config_payload() -> dict[str, object]:
    return {
        "simulation": {
            "seed": 41,
            "simulator": "island_coalescent",
            "min_height": 3,
            "max_height": 3,
            "min_width": 3,
            "max_width": 3,
            "num_snps": 8,
            "sampled_node_fraction": 0.5,
            "min_sampled_nodes": 4,
            "diploids_per_node": 1,
            "surface_kinds": ["smooth"],
            "q_log_std": 0.1,
            "island_ploidy": 2,
            "island_mutation_rate": 2.0,
            "island_use_rust": False,
            "island_emit_covariance": True,
        }
    }


def test_pipeline_covariance_config_emits_train_covariance() -> None:
    sim = simulation_config.simulation_config(
        {
            "simulation": {
                **island_covariance_config_payload()["simulation"],
            }
        }
    )
    train_example = simulation.simulate_raw_example(
        np.random.default_rng(41),
        sim=sim,
        split="train",
        shard=0,
        index=0,
    )
    assert "sample_covariance" in train_example
    assert "genotypes" not in train_example
