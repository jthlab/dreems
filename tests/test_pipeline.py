from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dreems.data import corpus, geography, motifs, simulation, simulation_config
from dreems.inference import calibration
from dreems.neural import checkpoints, features
from dreems.training import pipeline, train
from dreems.utils import workflow_io


def test_load_config_preserves_json_exponent_numbers(tmp_path) -> None:
    path = tmp_path / "training.json"
    path.write_text('{"dreems":{"optimizer":{"learning_rate":5e-05}}}\n')

    loaded = workflow_io.load_config(path)

    learning_rate = loaded["dreems"]["optimizer"]["learning_rate"]
    assert learning_rate == 5.0e-5
    assert isinstance(learning_rate, float)


def test_normalization_sanitizes_invalid_linearized_initializer() -> None:
    item = {
        "node_x": np.zeros((2, 1), dtype=np.float32),
        "edge_x": np.zeros((1, 2), dtype=np.float32),
        "edges": np.asarray([[0, 1]], dtype=np.int32),
        "y": np.zeros((1,), dtype=np.float32),
        "q": np.zeros((2,), dtype=np.float32),
        "linearized_resistance_edge_log": np.asarray([np.nan], dtype=np.float32),
        "linearized_resistance_valid": np.asarray([0.0], dtype=np.float32),
    }
    stats = {
        "node_mean": [0.0],
        "node_std": [1.0],
        "edge_mean": [0.0, 0.0],
        "edge_std": [1.0, 1.0],
        "y_scale": 1.0,
        "q_y_scale": 1.0,
    }

    [normalized] = features.apply_normalization([item], stats)

    np.testing.assert_array_equal(normalized["edge_initializer"], np.zeros(1))
    np.testing.assert_array_equal(normalized["edge_initializer_valid"], np.zeros(1))


def small_raw_example(offset: float = 0.0) -> dict[str, object]:
    positions = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
        dtype=np.float64,
    )
    edges = np.asarray(
        [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4], [1, 4]],
        dtype=np.int32,
    )
    genotypes = np.asarray(
        [
            [0, 1, 2, 0, 1, 2],
            [1, 1, 1, 0, 2, 0],
            [2, 0, 1, 1, 0, 1],
            [0, 2, 0, 2, 1, 1],
            [1, 0, 2, 1, 2, 0],
            [2, 1, 0, 0, 0, 2],
            [0, 2, 1, 2, 1, 0],
            [1, 0, 0, 1, 2, 1],
        ],
        dtype=np.float64,
    )
    return {
        "graph": {"active_full": np.arange(positions.shape[0]), "edges": edges},
        "active_positions": positions,
        "sample_nodes": np.asarray([0, 1, 1, 2, 3, 4, 4, 0], dtype=np.int32),
        "genotypes": genotypes,
        "edge_log_rates": np.linspace(-0.5 + offset, 0.6 + offset, 6, dtype=np.float32),
        "node_log_q": np.linspace(-0.2 + offset, 0.2 + offset, 5, dtype=np.float32),
        "surface_kind": "toy",
    }


def test_corpus_worker_generates_exactly_one_deterministic_example(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "simulation.json"
    config_path.write_text('{"simulation":{"seed":17}}\n')
    output = tmp_path / "raw.pkl"
    received: list[tuple[int, dict[str, object], str, int, int]] = []

    def fake_simulate(payload):
        received.append(payload)
        return small_raw_example()

    monkeypatch.setattr(corpus.simulation, "simulate_raw_example_from_seed", fake_simulate)
    corpus.generate_example(
        config_path=config_path,
        split="test",
        example_id=3,
        output=output,
    )

    assert len(workflow_io.read_pickle(output)) == 1
    assert received[0][0] == 17 + simulation.SPLIT_SEED_OFFSETS["test"] + 1009 * 3
    assert received[0][2:] == ("test", 3, 0)


def test_corpus_packs_ordered_worker_outputs_with_one_audit_sidecar(tmp_path: Path) -> None:
    inputs = []
    for index in range(3):
        path = tmp_path / "staging" / f"chunk_{index}.pkl"
        workflow_io.write_pickle(path, [small_raw_example(float(index))])
        inputs.append(path)
    output = tmp_path / "raw" / "pack_2.pkl"
    metadata = tmp_path / "raw" / "pack_2.metadata.json"
    config = tmp_path / "simulation.json"
    config.write_text('{"simulation":{"seed":17}}\n')

    corpus.pack_examples(
        inputs=inputs,
        output=output,
        metadata=metadata,
        config_path=config,
        split="train",
        pack_index=2,
        start_index=8,
    )

    packed = workflow_io.read_pickle(output)
    assert len(packed) == 3
    np.testing.assert_allclose(
        [example["edge_log_rates"][0] for example in packed],
        [-0.5, 0.5, 1.5],
    )
    sidecar = json.loads(metadata.read_text())
    assert sidecar["pack_index"] == 2
    assert sidecar["start_index"] == 8
    assert sidecar["stop_index"] == 11
    assert sidecar["base_seed"] == 17
    assert [row["index"] for row in sidecar["rows"]] == [8, 9, 10]


def test_raw_example_window_selects_across_pack_boundaries(tmp_path: Path) -> None:
    first = tmp_path / "pack_0.pkl"
    second = tmp_path / "pack_1.pkl"
    workflow_io.write_pickle(
        first,
        [small_raw_example(float(index)) for index in range(3)],
    )
    workflow_io.write_pickle(
        second,
        [small_raw_example(float(index)) for index in range(3, 6)],
    )

    windows = pipeline.raw_example_window(
        [first, second],
        start_index=2,
        max_examples=3,
    )

    assert [(path.name, offset, len(items)) for path, offset, items in windows] == [
        ("pack_0.pkl", 2, 1),
        ("pack_1.pkl", 0, 2),
    ]
    np.testing.assert_allclose(
        [item["edge_log_rates"][0] for _path, _offset, items in windows for item in items],
        [1.5, 2.5, 3.5],
    )


def write_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def test_msprime_haplotypes_are_restored_to_requested_sample_order() -> None:
    sample_nodes = np.asarray([2, 0, 2, 1], dtype=np.int32)
    haplotypes = np.asarray(
        [[1, 0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0, 1, 0]],
        dtype=np.int8,
    )

    genotypes = simulation.genotypes_from_grouped_haplotypes(
        haplotypes,
        sample_nodes=sample_nodes,
        num_nodes=3,
        ploidy=2,
    )

    np.testing.assert_array_equal(
        genotypes,
        np.asarray([[2, 1], [1, 0], [0, 1], [1, 2]], dtype=np.int8),
    )


def test_simulation_defaults_use_maf_conditioned_island_markers() -> None:
    sim = simulation_config.simulation_config({})

    assert sim["island_marker_model"] == "conditioned_random_marker"
    assert sim["island_min_allele_frequency"] == 0.05
    assert sim["island_max_allele_frequency"] == 0.95


def test_num_snps_log_bins_are_balanced_and_log_uniform() -> None:
    bins = [[50, 128], [129, 512], [513, 2_000], [2_001, 10_000], [10_001, 50_000]]
    sim = simulation_config.simulation_config({"simulation": {"num_snps_log_bins": bins}})
    rng = np.random.default_rng(29)
    draws = np.asarray([simulation_config.draw_num_snps(rng, sim) for _ in range(50_000)])

    counts = np.asarray([np.sum((draws >= low) & (draws <= high)) for low, high in bins])
    np.testing.assert_allclose(counts / draws.size, 0.2, atol=0.01)
    for low, high in bins:
        values = draws[(draws >= low) & (draws <= high)]
        assert np.isclose(np.median(np.log(values)), np.log(np.sqrt(low * high)), atol=0.04)


@pytest.mark.parametrize(
    "bins",
    [[], [[0, 10]], [[20, 10]], [[10]], "10-20"],
)
def test_num_snps_log_bins_reject_invalid_ranges(bins) -> None:
    with pytest.raises(ValueError, match="SNP bins|num_snps_log_bins"):
        simulation_config.simulation_config({"simulation": {"num_snps_log_bins": bins}})


def test_feems_lattice_and_reference_surfaces_match_feems_protocol() -> None:
    graph, positions = geography.feems_triangular_lattice_graph(8, 12)
    assert positions.shape == (96, 2)
    assert graph["edges"].shape == (249, 2)

    sim = simulation_config.simulation_config({})
    local = simulation_config.local_simulation_parameters(np.random.default_rng(13), sim)
    for name in ("feems_heterogeneous_barrier", "feems_anisotropic"):
        surface = geography.draw_surface(
            np.random.default_rng(13),
            surface_kind=name,
            local=local,
        )
        log_rates = geography.edge_log_rates_from_surface(
            surface,
            positions,
            graph["edges"],
        )
        assert np.isclose(float(np.ptp(log_rates)), np.log(5.0), atol=1e-6)


def test_multiscale_surface_is_reproducible_and_normalized() -> None:
    graph, positions = geography.masked_grid_graph(np.ones((14, 17), dtype=bool))
    sim = simulation_config.simulation_config({})
    local = simulation_config.local_simulation_parameters(np.random.default_rng(3), sim)
    local["log_rate_std"] = 1.25

    surface = geography.draw_surface(
        np.random.default_rng(41),
        surface_kind="multiscale",
        local=local,
    )
    repeated = geography.draw_surface(
        np.random.default_rng(41),
        surface_kind="multiscale",
        local=local,
    )
    rates = geography.edge_log_rates_from_surface(surface, positions, graph["edges"])
    repeated_rates = geography.edge_log_rates_from_surface(
        repeated,
        positions,
        graph["edges"],
    )

    assert [band["length_scale"] for band in surface["bands"]] == pytest.approx(
        [band["length_scale"] for band in repeated["bands"]]
    )
    assert len(surface["bands"]) == 3
    assert 1 <= len(surface["regions"]) <= 3
    assert 1 <= len(surface["primitives"]) <= 4
    assert 0.65 <= surface["bands"][0]["length_scale"] <= 1.25
    assert 0.22 <= surface["bands"][1]["length_scale"] <= 0.55
    assert 0.07 <= surface["bands"][2]["length_scale"] <= 0.20
    np.testing.assert_array_equal(rates, repeated_rates)
    assert np.isclose(float(np.mean(rates)), 0.0, atol=1e-6)
    assert np.isclose(float(np.std(rates)), 1.25, atol=1e-6)


def test_feems_lattice_choices_vary_graph_size_and_scale_reference_barrier() -> None:
    sim = simulation_config.simulation_config(
        {
            "simulation": {
                "feems_lattice_row_choices": [8, 16],
                "feems_lattice_column_choices": [9, 12],
                "feems_lattice_surface_kinds": ["feems_heterogeneous_barrier"],
                "feems_lattice_sample_prob_choices": [1.0],
                "max_samples": 100,
                "q_log_std": 0.0,
            }
        }
    )
    shapes = set()
    for seed in range(32):
        rng = np.random.default_rng(seed)
        local = simulation_config.local_simulation_parameters(rng, sim)
        case = motifs.draw_feems_lattice_case(rng, sim, local)
        shapes.add((case["feems_lattice_rows"], case["feems_lattice_columns"]))
        assert np.isclose(float(np.ptp(case["edge_log_rates"])), np.log(5.0), atol=1e-6)
        assert case["sample_nodes"].shape[0] == 100
        assert case["feems_lattice_samples_capped"] is True

    assert {rows for rows, _columns in shapes} == {8, 16}
    assert {columns for _rows, columns in shapes} == {9, 12}


def test_sample_designs_always_observe_two_nodes_and_respect_sample_cap() -> None:
    positions = np.column_stack([np.arange(100, dtype=np.float64), np.zeros(100, dtype=np.float64)])
    for design in ("uniform", "poisson_sites", "sites", "clustered", "full"):
        sample_nodes = motifs.draw_sample_nodes_from_positions(
            np.random.default_rng(23),
            positions,
            {
                "sample_design": design,
                "diploids_per_node": 20,
                "min_samples": 1,
                "max_samples": 120,
                "min_sample_fraction": 0.01,
                "max_sample_fraction": 0.01,
            },
        )
        assert sample_nodes.shape[0] <= 120
        assert np.unique(sample_nodes).shape[0] >= 2


def migration_sampling_case(num_nodes: int = 400) -> dict[str, object]:
    positions = np.column_stack(
        [np.arange(num_nodes, dtype=np.float64), np.zeros(num_nodes, dtype=np.float64)]
    )
    edges = np.column_stack(
        [np.arange(num_nodes - 1, dtype=np.int32), np.arange(1, num_nodes, dtype=np.int32)]
    )
    return {
        "graph": {"edges": edges},
        "active_positions": positions,
        "edge_log_rates": np.linspace(-2.0, 2.0, edges.shape[0]),
        "sample_nodes": np.repeat(np.arange(100, dtype=np.int32), 2),
        "sample_design": "sites",
    }


def test_migration_associated_sampling_preserves_counts_and_varies_sign() -> None:
    sim = simulation_config.simulation_config(
        {
            "simulation": {
                "sampling_association_min_strength": 5.0,
                "sampling_association_max_strength": 5.0,
                "sampling_spatial_noise_scale": 0.0,
            }
        }
    )
    independent = motifs.apply_migration_associated_sampling(
        np.random.default_rng(41),
        migration_sampling_case(),
        sim,
        mode="independent",
    )
    positive = motifs.apply_migration_associated_sampling(
        np.random.default_rng(41),
        migration_sampling_case(),
        sim,
        mode="migration_positive",
    )
    negative = motifs.apply_migration_associated_sampling(
        np.random.default_rng(41),
        migration_sampling_case(),
        sim,
        mode="migration_negative",
    )

    for example in (independent, positive, negative):
        assert example["sample_nodes"].shape == (200,)
        _nodes, counts = np.unique(example["sample_nodes"], return_counts=True)
        assert counts.shape == (100,)
        assert np.all(counts == 2)
    assert abs(independent["sampling_inclusion_migration_correlation"]) < 0.15
    assert positive["sampling_inclusion_migration_correlation"] > 0.5
    assert negative["sampling_inclusion_migration_correlation"] < -0.5
    assert positive["sampling_sampled_mean_migration"] > 0.0
    assert negative["sampling_sampled_mean_migration"] < 0.0


def test_barrier_proximity_sampling_targets_low_edge_and_respects_land_mask() -> None:
    case = migration_sampling_case()
    edge_rates = np.zeros(399, dtype=np.float64)
    edge_rates[199] = -5.0
    case["edge_log_rates"] = edge_rates
    land_mask = np.ones(400, dtype=bool)
    land_mask[-20:] = False
    case["node_land_mask"] = land_mask
    case["sample_nodes"] = np.arange(40, dtype=np.int32)
    sim = simulation_config.simulation_config(
        {
            "simulation": {
                "sampling_association_min_strength": 8.0,
                "sampling_association_max_strength": 8.0,
                "sampling_spatial_noise_scale": 0.0,
            }
        }
    )

    sampled = motifs.apply_migration_associated_sampling(
        np.random.default_rng(53),
        case,
        sim,
        mode="barrier_proximity",
    )

    sites = np.unique(sampled["sample_nodes"])
    assert 199 in sites
    assert 200 in sites
    assert np.all(land_mask[sites])


def test_migration_associated_sampling_conserves_samples_when_eligibility_shrinks() -> None:
    case = migration_sampling_case()
    land_mask = np.zeros(400, dtype=bool)
    land_mask[:50] = True
    case["node_land_mask"] = land_mask
    sim = simulation_config.simulation_config(
        {
            "simulation": {
                "sampling_association_min_strength": 2.0,
                "sampling_association_max_strength": 2.0,
            }
        }
    )

    sampled = motifs.apply_migration_associated_sampling(
        np.random.default_rng(59),
        case,
        sim,
        mode="migration_positive",
    )

    samples = np.asarray(sampled["sample_nodes"])
    assert samples.shape == (200,)
    assert np.unique(samples).shape == (50,)
    assert np.all(land_mask[samples])


def test_counterfactual_sampling_assignment_groups_surface_seeds() -> None:
    sim = simulation_config.simulation_config(
        {
            "simulation": {
                "sampling_counterfactual_modes": [
                    "independent",
                    "migration_positive",
                    "migration_negative",
                    "barrier_proximity",
                ]
            }
        }
    )

    assignments = [
        simulation.counterfactual_sampling_assignment(sim, index, split="train")
        for index in range(8)
    ]

    assert assignments[:4] == [
        (0, "independent"),
        (0, "migration_positive"),
        (0, "migration_negative"),
        (0, "barrier_proximity"),
    ]
    assert assignments[4:] == [
        (1, "independent"),
        (1, "migration_positive"),
        (1, "migration_negative"),
        (1, "barrier_proximity"),
    ]


def test_counterfactual_sampling_can_keep_training_truths_distinct() -> None:
    sim = simulation_config.simulation_config(
        {
            "simulation": {
                "sampling_counterfactual_modes": [
                    "independent",
                    "migration_positive",
                    "migration_negative",
                    "barrier_proximity",
                ],
                "sampling_counterfactual_group_splits": ["test"],
            }
        }
    )

    train_assignments = [
        simulation.counterfactual_sampling_assignment(sim, index, split="train")
        for index in range(8)
    ]
    test_assignments = [
        simulation.counterfactual_sampling_assignment(sim, index, split="test")
        for index in range(8)
    ]

    assert [seed for seed, _mode in train_assignments] == list(range(8))
    assert [mode for _seed, mode in train_assignments] == [
        "independent",
        "migration_positive",
        "migration_negative",
        "barrier_proximity",
    ] * 2
    assert [seed for seed, _mode in test_assignments] == [0, 0, 0, 0, 1, 1, 1, 1]


def test_counterfactual_modes_share_graph_surface_and_q(monkeypatch) -> None:
    sim = simulation_config.simulation_config(
        {
            "simulation": {
                "case_weights": {"grid": 1.0},
                "min_height": 12,
                "max_height": 12,
                "min_width": 12,
                "max_width": 12,
                "mask_kinds": ["full"],
                "surface_kinds": ["barrier"],
                "sample_designs": ["sites"],
                "min_samples": 24,
                "max_samples": 24,
            }
        }
    )

    def skip_genotypes(*_args, **_kwargs):
        return "test"

    monkeypatch.setattr(simulation, "simulate_case_genotypes", skip_genotypes)
    examples = []
    for mode in (
        "independent",
        "migration_positive",
        "migration_negative",
        "barrier_proximity",
    ):
        local_sim = {**sim, "sampling_association_mode_override": mode}
        examples.append(
            simulation.simulate_raw_example(
                np.random.default_rng(71),
                sim=local_sim,
                split="train",
                shard=0,
                index=0,
            )
        )

    reference = examples[0]
    for example in examples[1:]:
        assert np.array_equal(example["graph"]["edges"], reference["graph"]["edges"])
        assert np.array_equal(example["active_positions"], reference["active_positions"])
        assert np.array_equal(example["edge_log_rates"], reference["edge_log_rates"])
        assert np.array_equal(example["node_log_q"], reference["node_log_q"])
        assert example["num_snps"] == reference["num_snps"]
    assert len({tuple(example["sample_nodes"]) for example in examples}) == 4


def test_sampling_config_rejects_unknown_counterfactual_mode() -> None:
    with pytest.raises(ValueError, match="unsupported sampling counterfactual"):
        simulation_config.simulation_config(
            {"simulation": {"sampling_counterfactual_modes": ["not-a-mode"]}}
        )


def test_sampling_config_rejects_unknown_counterfactual_group_split() -> None:
    with pytest.raises(ValueError, match="counterfactual group splits"):
        simulation_config.simulation_config(
            {"simulation": {"sampling_counterfactual_group_splits": ["validation"]}}
        )


def test_manifest_paths_are_relative_to_the_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "inputs.txt"
    absolute = tmp_path / "absolute.pkl"
    manifest.write_text(f"# comment\nrelative.pkl\n{absolute}\n\n")

    paths = workflow_io.required_input_paths(None, manifest, argument_name="--inputs")

    assert paths == [tmp_path / "relative.pkl", absolute]


def test_raw_generation_retries_an_unconditionable_draw(monkeypatch) -> None:
    calls = 0

    def fake_simulate(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        example = small_raw_example()
        if calls == 1:
            example["sample_nodes"] = np.zeros(8, dtype=np.int32)
        return example

    monkeypatch.setattr(simulation, "simulate_raw_example", fake_simulate)
    example = simulation.simulate_raw_example_from_seed(
        (123, {"max_conditionable_attempts": 2}, "train", 0, 0)
    )

    assert calls == 2
    assert example["conditionable_retry_attempt"] == 1


def test_covariance_factorization_recovers_from_eigh_failure(monkeypatch) -> None:
    original_eigh = np.linalg.eigh
    calls = 0

    def flaky_eigh(matrix):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise np.linalg.LinAlgError("did not converge")
        return original_eigh(matrix)

    monkeypatch.setattr(features.np.linalg, "eigh", flaky_eigh)
    summaries = features.covariance_deme_summaries(
        np.asarray([[1.0, 0.2], [0.2, 1.0]]),
        np.asarray([0, 1], dtype=np.int32),
        2,
        effective_snps=8,
    )

    assert calls == 2
    assert np.all(np.isfinite(summaries["mean_x"]))


def test_production_features_are_diffusion_bands_plus_all_pair_conditioner() -> None:
    example = small_raw_example()
    prepared = features.prepare_examples([example], covariance_components=3)
    item = prepared[0]
    assert item["node_x"].shape == (5, 62)
    assert item["edge_x"].shape == (6, 161)
    assert np.max(np.abs(item["edge_x"][:, -144:])) > 0.0

    config = train.DREEMSTrainConfig(features=train.DREEMSFeatureConfig(covariance_components=3))
    conditioned = train.append_training_edge_conditioner_features(
        prepared,
        [example],
        config,
    )[0]
    assert conditioned["edge_x"].shape == (6, 163)
    assert np.all(conditioned["linearized_resistance_valid"] == 1.0)


def test_node_covariance_features_are_input_representation_invariant() -> None:
    genotype_example = small_raw_example()
    sample_nodes = np.asarray(genotype_example["sample_nodes"], dtype=np.int32)
    standardized = features.standardized_polymorphic_genotypes(
        np.asarray(genotype_example["genotypes"], dtype=np.float64)
    )
    effective_snps = int(standardized.shape[1])
    sample_covariance = standardized @ standardized.T / float(effective_snps)
    node_covariance, observed_nodes, node_sample_count = (
        features.sample_covariance_to_node_covariance(
            sample_covariance,
            sample_nodes,
            num_nodes=np.asarray(genotype_example["active_positions"]).shape[0],
        )
    )

    sample_example = {key: value for key, value in genotype_example.items() if key != "genotypes"}
    sample_example["sample_covariance"] = sample_covariance
    sample_example["effective_snps"] = effective_snps
    node_example = {
        key: value
        for key, value in genotype_example.items()
        if key not in {"genotypes", "sample_nodes"}
    }
    node_example.update(
        {
            "node_covariance": node_covariance,
            "observed_nodes": observed_nodes,
            "node_sample_count": node_sample_count,
            "effective_snps": effective_snps,
        }
    )

    prepared = features.prepare_examples(
        [genotype_example, sample_example, node_example],
        covariance_components=3,
        spatial_encoding="topology_node_covariance",
    )
    for candidate in prepared[1:]:
        np.testing.assert_allclose(candidate["node_x"], prepared[0]["node_x"], atol=2e-6)
        np.testing.assert_allclose(candidate["edge_x"], prepared[0]["edge_x"], atol=2e-6)

    config = train.DREEMSTrainConfig(
        features=train.DREEMSFeatureConfig(
            covariance_components=3,
            spatial_encoding="topology_node_covariance",
        )
    )
    conditioned = train.append_training_edge_conditioner_features(
        prepared,
        [genotype_example, sample_example, node_example],
        config,
    )
    for candidate in conditioned[1:]:
        np.testing.assert_allclose(candidate["edge_x"], conditioned[0]["edge_x"], atol=2e-6)


def test_canonical_features_match_old_topology_features_for_node_covariance() -> None:
    genotype_example = small_raw_example()
    sample_nodes = np.asarray(genotype_example["sample_nodes"], dtype=np.int32)
    standardized = features.standardized_polymorphic_genotypes(
        np.asarray(genotype_example["genotypes"], dtype=np.float64)
    )
    sample_covariance = standardized @ standardized.T / standardized.shape[1]
    node_covariance, observed_nodes, node_sample_count = (
        features.sample_covariance_to_node_covariance(
            sample_covariance,
            sample_nodes,
            num_nodes=np.asarray(genotype_example["active_positions"]).shape[0],
        )
    )
    node_example = {
        key: value
        for key, value in genotype_example.items()
        if key not in {"genotypes", "sample_nodes"}
    }
    node_example.update(
        {
            "node_covariance": node_covariance,
            "observed_nodes": observed_nodes,
            "node_sample_count": node_sample_count,
            "effective_snps": standardized.shape[1],
        }
    )

    old = features.prepare_examples(
        [node_example],
        covariance_components=3,
        spatial_encoding="topology",
    )[0]
    canonical = features.prepare_examples(
        [node_example],
        covariance_components=3,
        spatial_encoding="topology_node_covariance",
    )[0]
    for name in ("node_x", "edge_x", "edges", "y", "q"):
        np.testing.assert_array_equal(canonical[name], old[name])


@pytest.mark.parametrize(
    "coordinate_transform",
    [
        np.asarray([[0.0, -1.0], [1.0, 0.0]]),
        np.asarray([[0.0, 3.0], [3.0, 0.0]]),
        np.asarray([[0.01, 0.0], [0.0, 0.01]]),
    ],
    ids=["rotation", "reflection-and-scale", "downscale"],
)
def test_topology_features_ignore_edge_representation_and_coordinate_frame(
    coordinate_transform: np.ndarray,
) -> None:
    example = small_raw_example()
    edges = np.asarray(example["graph"]["edges"])
    permutation = np.asarray([5, 2, 0, 4, 1, 3], dtype=np.int64)
    represented_differently = small_raw_example()
    represented_differently["graph"] = {
        **represented_differently["graph"],
        "edges": edges[permutation, ::-1],
    }
    represented_differently["edge_log_rates"] = np.asarray(example["edge_log_rates"])[permutation]
    positions = np.asarray(example["active_positions"])
    represented_differently["active_positions"] = positions @ coordinate_transform.T + np.asarray(
        [17.0, -4.0]
    )

    reference = features.prepare_examples(
        [example],
        covariance_components=3,
        spatial_encoding="topology",
    )[0]
    changed = features.prepare_examples(
        [represented_differently],
        covariance_components=3,
        spatial_encoding="topology",
    )[0]
    changed_lookup = {
        tuple(sorted(map(int, edge))): index for index, edge in enumerate(changed["edges"])
    }
    alignment = [changed_lookup[tuple(sorted(map(int, edge)))] for edge in reference["edges"]]

    np.testing.assert_allclose(changed["node_x"], reference["node_x"], atol=2e-6)
    np.testing.assert_allclose(changed["edge_x"][alignment], reference["edge_x"], atol=2e-6)
    np.testing.assert_allclose(changed["y"][alignment], reference["y"], atol=2e-6)

    embedded_changed = features.prepare_examples(
        [represented_differently],
        covariance_components=3,
        spatial_encoding="embedded",
    )[0]
    assert not np.allclose(embedded_changed["node_x"], reference["node_x"])

    old_to_new = np.asarray([3, 0, 4, 1, 2], dtype=np.int32)
    relabeled = small_raw_example()
    relabeled_positions = np.empty_like(positions)
    relabeled_positions[old_to_new] = positions
    relabeled_q = np.empty_like(np.asarray(example["node_log_q"]))
    relabeled_q[old_to_new] = np.asarray(example["node_log_q"])
    relabeled["active_positions"] = relabeled_positions
    relabeled["sample_nodes"] = old_to_new[np.asarray(example["sample_nodes"])]
    relabeled["node_log_q"] = relabeled_q
    relabeled["graph"] = {
        **relabeled["graph"],
        "active_full": np.arange(old_to_new.size),
        "edges": old_to_new[edges],
    }
    relabeled_features = features.prepare_examples(
        [relabeled],
        covariance_components=3,
        spatial_encoding="topology",
    )[0]
    np.testing.assert_allclose(
        relabeled_features["node_x"][old_to_new],
        reference["node_x"],
        atol=2e-6,
    )
    np.testing.assert_allclose(relabeled_features["edge_x"], reference["edge_x"], atol=2e-6)


def test_conditioner_fails_instead_of_silently_zero_filling() -> None:
    example = small_raw_example()
    example["sample_nodes"] = np.zeros(8, dtype=np.int32)
    prepared = features.prepare_examples([example], covariance_components=3)
    model_metadata = {
        "edge_conditioner": {
            "kind": "linearized_resistance",
            "method": "all_pair_diagonal_adjoint",
        },
        "edge_dim": prepared[0]["edge_x"].shape[1] + 2,
    }

    with pytest.raises(ValueError, match="linearized edge conditioner failed"):
        features.append_edge_conditioner_features_for_checkpoint(
            prepared,
            [example],
            model_metadata,
            stats={},
        )


def test_streaming_batches_span_one_example_shards(tmp_path: Path) -> None:
    paths = []
    for index, marker in enumerate((1.0, 2.0)):
        path = tmp_path / f"shard_{index}.pkl"
        write_pickle(
            path,
            [
                {
                    "node_x": np.asarray([[marker]], dtype=np.float32),
                    "edge_x": np.asarray([[0.0]], dtype=np.float32),
                    "edges": np.asarray([[0, 0]], dtype=np.int32),
                    "y": np.asarray([0.0], dtype=np.float32),
                    "q": np.asarray([0.0], dtype=np.float32),
                }
            ],
        )
        paths.append(path)
    stats = {
        "node_mean": [0.0],
        "node_std": [1.0],
        "edge_mean": [0.0],
        "edge_std": [1.0],
        "y_scale": 1.0,
        "q_y_scale": 1.0,
    }
    dims = pipeline.scan_prepared_dimensions(paths)

    batch = next(
        iter(
            pipeline.iter_training_batches(
                paths,
                stats,
                dims=dims,
                batch_size=2,
                steps=1,
                seed=7,
            )
        )
    )

    assert set(batch["node_x"][:, 0, 0]) == {1.0, 2.0}


def test_stage_input_paths_copies_to_unique_local_names(tmp_path: Path) -> None:
    first = tmp_path / "one" / "pack.pkl"
    second = tmp_path / "two" / "pack.pkl"
    write_pickle(first, [1])
    write_pickle(second, [2])

    staged = pipeline.stage_input_paths(
        [first, second],
        tmp_path / "cache",
        label="train",
    )

    assert [path.name for path in staged] == ["00000_pack.pkl", "00001_pack.pkl"]
    assert [workflow_io.read_pickle(path) for path in staged] == [[1], [2]]


def test_stage_input_paths_reuses_duplicate_sources(tmp_path: Path) -> None:
    source = tmp_path / "pack.pkl"
    write_pickle(source, [1])

    staged = pipeline.stage_input_paths(
        [source, source, source],
        tmp_path / "cache",
        label="train",
    )

    assert staged == [staged[0], staged[0], staged[0]]
    assert [path.name for path in (tmp_path / "cache" / "train").iterdir()] == ["00000_pack.pkl"]
    assert workflow_io.read_pickle(staged[0]) == [1]


def test_stage_input_paths_fails_before_copy_when_local_space_is_insufficient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "pack.pkl"
    write_pickle(source, [1])
    monkeypatch.setattr(pipeline.shutil, "disk_usage", lambda _path: SimpleNamespace(free=0))

    with pytest.raises(OSError, match="insufficient local space"):
        pipeline.stage_input_paths([source], tmp_path / "cache", label="train")

    assert not (tmp_path / "cache" / "train" / "00000_pack.pkl").exists()


def test_prepared_dimensions_use_sidecar_without_loading_pickle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "chunk_0.pkl"
    path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "prepared_dimensions": {
                    "examples": 1,
                    "max_nodes": 7,
                    "max_edges": 12,
                    "node_dim": 109,
                    "edge_dim": 253,
                }
            }
        )
    )
    monkeypatch.setattr(
        pipeline,
        "read_pickle",
        lambda _path: pytest.fail("sidecar should avoid loading the shard"),
    )

    assert pipeline.scan_prepared_dimensions([path]) == {
        "examples": 1,
        "max_nodes": 7,
        "max_edges": 12,
        "node_dim": 109,
        "edge_dim": 253,
    }


def test_pack_prepared_selects_manifest_window_and_writes_dimensions(tmp_path: Path) -> None:
    inputs = []
    for index, nodes in enumerate((2, 3, 4)):
        path = tmp_path / "prepared" / f"chunk_{index}.pkl"
        item = {
            "node_x": np.full((nodes, 2), index, dtype=np.float32),
            "edge_x": np.full((nodes - 1, 3), index, dtype=np.float32),
            "edges": np.column_stack(
                [np.arange(nodes - 1, dtype=np.int32), np.arange(1, nodes, dtype=np.int32)]
            ),
            "y": np.zeros(nodes - 1, dtype=np.float32),
            "q": np.zeros(nodes, dtype=np.float32),
        }
        write_pickle(path, [item])
        inputs.append(path)
    manifest = tmp_path / "prepared" / "manifest.txt"
    manifest.write_text("\n".join(path.name for path in inputs) + "\n")
    output = tmp_path / "packs" / "pack_0.pkl"
    metadata = output.with_suffix(".metadata.json")

    pipeline.main(
        [
            "pack-prepared",
            "--input-manifest",
            str(manifest),
            "--start-index",
            "1",
            "--max-inputs",
            "2",
            "--output",
            str(output),
            "--metadata",
            str(metadata),
        ]
    )

    packed = workflow_io.read_pickle(output)
    assert [item["node_x"].shape[0] for item in packed] == [3, 4]
    sidecar = json.loads(metadata.read_text())
    assert sidecar["source_shards"] == 2
    assert sidecar["source_start_index"] == 1
    assert sidecar["source_stop_index"] == 3
    assert sidecar["prepared_dimensions"] == {
        "examples": 2,
        "max_nodes": 4,
        "max_edges": 3,
        "node_dim": 2,
        "edge_dim": 3,
    }


def write_smoke_config(path: Path, outdir: Path) -> None:
    payload = {
        "outdir": str(outdir),
        "train_validation_examples": 1,
        "simulation": {
            "seed": 17,
            "simulator": "msprime_coalescent",
            "min_height": 3,
            "max_height": 3,
            "min_width": 3,
            "max_width": 3,
            "num_snps": 16,
            "sampled_node_fraction": 0.5,
            "min_sampled_nodes": 4,
            "diploids_per_node": 1,
            "surface_kinds": ["smooth"],
            "coalescent_mutation_rate": 0.05,
            "coalescent_batch_replicates": 16,
            "coalescent_max_batches": 8,
        },
        "dreems": {
            "architecture": {"hidden_channels": 8, "layers": 1, "context_layers": 1},
            "features": {"covariance_components": 3, "normalization_std_floor": 0.2},
            "edge_conditioner": {"batch_size": 32},
            "optimizer": {
                "seed": 11,
                "steps": 1,
                "batch_size": 1,
                "learning_rate": 5.0e-5,
                "log_every": 1,
            },
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True))


def test_pipeline_cli_prepares_trains_and_evaluates(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    outdir = tmp_path / "out"
    write_smoke_config(config, outdir)
    train_raw = tmp_path / "raw" / "train.pkl"
    test_raw = tmp_path / "raw" / "test.pkl"
    write_pickle(train_raw, [small_raw_example()])
    write_pickle(test_raw, [small_raw_example(0.1)])

    prepared_batch = tmp_path / "prepared" / "batch.pkl"
    pipeline.main(
        [
            "prepare-batch",
            "--config",
            str(config),
            "--inputs",
            str(train_raw),
            str(test_raw),
            "--output",
            str(prepared_batch),
            "--metadata",
            str(prepared_batch.with_suffix(".metadata.json")),
        ]
    )
    batch_rows = workflow_io.read_pickle(prepared_batch)
    assert len(batch_rows) == 2
    assert all(float(row["sampled_node_fraction"]) == 1.0 for row in batch_rows)
    np.testing.assert_allclose(batch_rows[0]["y"], small_raw_example()["edge_log_rates"])
    np.testing.assert_allclose(batch_rows[1]["y"], small_raw_example(0.1)["edge_log_rates"])
    batch_metadata = json.loads(prepared_batch.with_suffix(".metadata.json").read_text())
    assert batch_metadata["source_shards"] == 2
    assert batch_metadata["examples"] == 2
    assert batch_metadata["skipped_unconditionable_examples"] == 0

    prepared = []
    for split, raw in (("train", train_raw), ("test", test_raw)):
        output = tmp_path / "prepared" / f"{split}.pkl"
        pipeline.main(
            [
                "prepare-shard",
                "--config",
                str(config),
                "--input",
                str(raw),
                "--output",
                str(output),
                "--metadata",
                str(output.with_suffix(".metadata.json")),
            ]
        )
        prepared.append(output)

    shard_stats = tmp_path / "normalization.stats.json"
    normalization = tmp_path / "normalization.json"
    pipeline.main(
        [
            "normalization-shard-stats",
            "--inputs",
            str(prepared[0]),
            "--output",
            str(shard_stats),
        ]
    )
    pipeline.main(
        [
            "merge-normalization",
            "--config",
            str(config),
            "--inputs",
            str(shard_stats),
            "--output",
            str(normalization),
        ]
    )

    checkpoint = tmp_path / "model" / "checkpoint.dreems.zip"
    summary = tmp_path / "model" / "training_summary.json"
    pipeline.main(
        [
            "train",
            "--config",
            str(config),
            "--normalization",
            str(normalization),
            "--inputs",
            str(prepared[0]),
            "--validation-inputs",
            str(prepared[1]),
            "--output",
            str(checkpoint),
            "--summary",
            str(summary),
        ]
    )

    loaded = checkpoints.load_checkpoint(checkpoint)
    assert loaded.model["features"]["kind"] == "diffusion_bands_v1"
    evaluation = tmp_path / "evaluation"
    predictions = evaluation / "predictions.npz"
    pipeline.main(
        [
            "evaluate",
            "--checkpoint",
            str(checkpoint),
            "--inputs",
            str(prepared[1]),
            "--output-dir",
            str(evaluation),
            "--batch-size",
            "1",
            "--predictions-output",
            str(predictions),
        ]
    )
    assert json.loads((evaluation / "summary.json").read_text())["examples"] == 1
    with np.load(predictions) as archive:
        expected_truth = small_raw_example(0.1)["edge_log_rates"]
        expected_truth = expected_truth - np.mean(expected_truth)
        np.testing.assert_array_equal(archive["replicate"], np.array([0]))
        np.testing.assert_array_equal(archive["edge_offsets"], np.array([0, expected_truth.size]))
        np.testing.assert_allclose(archive["edge_truth"], expected_truth)
        assert archive["edge_mean"].shape == expected_truth.shape
        assert archive["edge_sd"].shape == expected_truth.shape
        raw_edge_sd = archive["edge_sd"].copy()

    calibrated_checkpoint = tmp_path / "model" / "checkpoint.calibrated.dreems.zip"
    checkpoints.attach_edge_sd_calibration(
        checkpoint,
        calibrated_checkpoint,
        calibration.SDCalibration(method="global", global_scale=2.0),
    )
    calibrated_evaluation = tmp_path / "evaluation_calibrated"
    calibrated_predictions = calibrated_evaluation / "predictions.npz"
    pipeline.main(
        [
            "evaluate",
            "--checkpoint",
            str(calibrated_checkpoint),
            "--inputs",
            str(prepared[1]),
            "--output-dir",
            str(calibrated_evaluation),
            "--batch-size",
            "1",
            "--predictions-output",
            str(calibrated_predictions),
        ]
    )
    with np.load(calibrated_predictions) as archive:
        np.testing.assert_allclose(archive["edge_sd"], 2.0 * raw_edge_sd)


def test_global_landscape_samples_land_but_keeps_water_edges(monkeypatch) -> None:
    shapely = pytest.importorskip("shapely.geometry")
    synthetic_coast = shapely.box(10.0, 20.0, 24.0, 55.0)
    monkeypatch.setattr(
        geography,
        "load_world_land_geometry",
        lambda _resolution, _cartopy_data_dir: synthetic_coast,
    )
    sim = simulation_config.simulation_config(
        {
            "simulation": {
                "global_landscape_target_node_choices": [140],
                "global_landscape_min_nodes": 60,
                "global_landscape_max_nodes": 260,
                "global_landscape_min_box_deg": 8.0,
                "global_landscape_max_box_deg": 8.0,
                "global_landscape_min_land_fraction": 0.10,
                "global_landscape_preferred_center_lon": 23.7,
                "global_landscape_preferred_center_lat": 38.5,
                "global_landscape_sampled_fraction_choices": [0.15],
                "global_landscape_min_diploids_per_site": 2,
                "global_landscape_max_diploids_per_site": 2,
            }
        }
    )
    rng = np.random.default_rng(31)
    local = motifs.choose_sample_regime(
        rng,
        sim,
        simulation_config.local_simulation_parameters(rng, sim),
    )

    case = motifs.draw_global_landscape_case(rng, sim, local)
    land = np.asarray(case["node_land_mask"], dtype=bool)
    sampled = np.unique(np.asarray(case["sample_nodes"], dtype=np.int32))
    edges = np.asarray(case["graph"]["edges"], dtype=np.int32)

    assert np.all(land[sampled])
    assert np.any(~(land[edges[:, 0]] & land[edges[:, 1]]))
    assert np.std(case["edge_log_rates"]) > 0.5
