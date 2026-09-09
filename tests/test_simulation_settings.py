"""Invalid recipes must fail before changing the intended simulation mixture."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from shapely.geometry import Polygon

from dreems import geography, motifs, simulation
from dreems import simulation_config as settings


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_snps": 12.5},
        {"num_snps": True},
        {"num_snps": 2**64},
        {"min_num_snps": 50},
        {"max_num_snps": 50},
        {"min_log_rate_std": 0.1},
        {"max_diploids_per_site": 4},
        {"min_sample_fraction": 0.1},
        {"sampled_node_fraction": 1.1},
        {"log_rate_std": -1},
        {"q_log_std": np.nan},
        {"migration_gp_length_scale": 0},
        {"island_min_allele_frequency": 0.7, "island_max_allele_frequency": 0.2},
        {"island_marker_model": "typo"},
        {"island_use_rust": "false"},
        {"q_prior": "typo"},
        {"earth_backend": "not_a_backend"},
        {"case_weights": {"grid": 0}},
        {"case_weights": {"grid": 1, "wolflike": -1}},
        {"case_weights": {"grid": np.inf}},
        {"surface_weights": {"smooth": "typo"}},
        {"surface_weights": {"smooth": True}},
        {"surface_kinds": ["typo"]},
        {"surface_kinds": []},
        {"sample_designs": ["typo"]},
        {"mask_kinds": ["typo"]},
        {"feems_lattice_rows": 1},
        {"feems_lattice_row_choices": [1, 4]},
        {"feems_lattice_row_choices": [True, 4]},
        {"wolflike_target_node_choices": []},
        {"wolflike_target_node_choices": [10.5]},
        {"wolflike_sampled_fraction_choices": [1.5]},
        {"global_landscape_preferred_center_lon": 10},
        {"global_landscape_min_lat": -100},
        {"global_landscape_resolution": "1m"},
        {"sample_regimes": {"name": "typo"}},
        {"sample_regimes": [{"name": "x", "weight": 0}]},
        {"sample_regimes": [{"name": "x", "weight": np.nan}]},
        {"sample_regimes": [{"name": "x", "samplez": 10}]},
        {"sample_regimes": [{"name": "x", "diploids_per_site": 0}]},
        {"sample_regimes": [{"name": "x", "design": "typo"}]},
    ],
)
def test_invalid_recipes_raise(overrides: dict) -> None:
    with pytest.raises(ValueError):
        settings.simulation_config({"simulation": overrides})


def test_supported_boundary_values_and_reversed_draw_ranges() -> None:
    overrides = {
        "min_samples": 0,
        "max_samples": 0,
        "sampled_node_fraction": 0,
        "q_log_std": 0,
        "log_rate_std": 0,
        "msprime_max_nodes": 0,
        "island_min_allele_frequency": 0.5,
        "island_max_allele_frequency": 0.5,
        "min_num_snps": 100,
        "max_num_snps": 50,
        "min_log_rate_std": 2.0,
        "max_log_rate_std": 1.0,
        "min_diploids_per_site": 4,
        "max_diploids_per_site": 2,
        "min_sample_fraction": 1.0,
        "max_sample_fraction": 0.1,
        "surface_kinds": ["smooth"],
        "surface_weights": {"smooth": 1, "barrier": 2},
    }
    sim = settings.simulation_config({"simulation": overrides})
    for key, value in overrides.items():
        assert sim[key] == value
    rng = np.random.default_rng(5)
    assert 50 <= settings.draw_num_snps(rng, sim) <= 100
    assert 2 <= settings.draw_local_diploids_per_node(rng, sim) <= 4
    assert 1 <= settings.local_simulation_parameters(rng, sim)["log_rate_std"] <= 2


@pytest.mark.parametrize(
    "name", ["config.example.yaml", "config.production.yaml", "config.snp_range_experiment.yaml"]
)
def test_existing_complete_recipes_remain_valid(name: str) -> None:
    root = Path(__file__).resolve().parents[1]
    recipe = yaml.safe_load((root / "workflow" / name).read_text())
    sim = settings.simulation_config(recipe)
    assert sim["seed"] == recipe["simulation"]["seed"]


@pytest.mark.parametrize(
    "weights", [{"a": -1}, {"a": np.nan}, {"a": 0, "b": 0}, ["a=1", "a=2"], ["a=typo"]]
)
def test_bad_weights_do_not_consume_randomness(weights: object) -> None:
    rng = np.random.default_rng(100)
    before = rng.bit_generator.state
    with pytest.raises(ValueError):
        settings.choose_weighted_name(rng, ["a", "b"], weights)
    assert rng.bit_generator.state == before


def test_unspecified_weights_still_use_one_and_zero_excludes() -> None:
    rng = np.random.default_rng(12)
    assert settings.choose_weighted_name(rng, ["a", "b"], {"a": 0, "unrelated": 2}) == "b"
    assert settings.parse_weight_mapping([{"name": "a", "weight": 2}, "b=3"]) == {
        "a": 2.0,
        "b": 3.0,
    }


@pytest.mark.parametrize("choices", [[], [0], [-2], [1.5], [True, 2], [[2, 3]]])
def test_bad_integer_choices_cannot_be_clipped_or_ignored(choices: object) -> None:
    with pytest.raises(ValueError):
        settings.draw_weighted_positive_int(
            np.random.default_rng(1), value=4, choices=choices, weights=None, fallback=4
        )


def test_invalid_regime_and_sample_count_fail_on_direct_calls() -> None:
    rng = np.random.default_rng(20)
    before = rng.bit_generator.state
    with pytest.raises(ValueError):
        motifs.choose_sample_regime(rng, {"sample_regimes": [{"weight": -1}]}, {})
    with pytest.raises(ValueError):
        motifs.draw_sample_nodes_from_positions(rng, np.zeros((3, 2)), {"diploids_per_node": 1.5})
    assert rng.bit_generator.state == before


def test_mask_edges_keep_the_saved_order_and_metadata() -> None:
    mask = np.asarray([[True, True, False], [True, True, True]])
    graph, positions = geography.masked_grid_graph(mask)
    np.testing.assert_array_equal(graph["source_active_full"], [0, 1, 3, 4, 5])
    np.testing.assert_array_equal(graph["edges"], [[0, 1], [2, 3], [3, 4], [0, 2], [1, 3]])
    np.testing.assert_array_equal(graph["edge_kind"], [0, 0, 0, 1, 1])
    np.testing.assert_array_equal(graph["edge_row"], [0, 1, 1, 0, 0])
    np.testing.assert_array_equal(graph["edge_col"], [0, 0, 1, 0, 1])
    np.testing.assert_array_equal(positions, [[0, 0], [0.5, 0], [0, -1], [0.5, -1], [1, -1]])
    assert graph["edges"].dtype == np.int32


def test_component_mask_uses_four_neighbors() -> None:
    mask = np.asarray([[True, False, True], [False, True, True]])
    np.testing.assert_array_equal(
        geography.largest_component_mask(mask), [[False, False, True], [False, True, True]]
    )
    with pytest.raises(ValueError, match="no active"):
        geography.largest_component_mask(np.zeros((3, 3), dtype=bool))


def test_geometry_covers_boundaries_but_excludes_hole_interior() -> None:
    polygon = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)], holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]])
    points = np.asarray([[0, 0], [0.5, 0.5], [1, 1], [1.5, 1.5], [4, 0]])
    np.testing.assert_array_equal(
        geography.points_in_geometry(points, polygon), [True, True, True, False, False]
    )


def test_graph_builder_does_not_truncate_endpoints() -> None:
    with pytest.raises(ValueError, match="fractional"):
        geography.graph_from_positions_edges(np.zeros((2, 2)), np.asarray([[0, 1.5]]))


def test_lattice_does_not_silently_change_its_requested_size() -> None:
    with pytest.raises(ValueError, match="n_rows"):
        geography.feems_triangular_lattice_graph(1, 4)


def sparse_lattice_recipe() -> dict:
    """The small sparse fixture that exposed an empty first sampling draw."""
    return settings.simulation_config(
        {
            "simulation": {
                "case_weights": {"feems_lattice": 1},
                "num_snps": 16,
                "feems_lattice_rows": 4,
                "feems_lattice_columns": 4,
                "feems_lattice_n_samples_per_node": 1,
                "feems_lattice_surface_kinds": ["feems_homogeneous", "feems_heterogeneous_barrier"],
                "feems_lattice_barrier_startpt": 1.0,
                "feems_lattice_barrier_endpt": 2.0,
                "min_samples": 4,
                "max_samples": 6,
                "diploids_per_node": 1,
            }
        }
    )


def test_empty_draw_retries_before_running_the_simulator(monkeypatch) -> None:
    sim = sparse_lattice_recipe()
    calls = []

    def genetic_data(rng, sim, case, *, num_snps):
        assert np.unique(case["sample_nodes"]).size >= 2
        calls.append(case["sample_nodes"].copy())
        return "test"

    monkeypatch.setattr(simulation, "simulate_case_genotypes", genetic_data)
    with pytest.raises(ValueError, match="fewer than two"):
        simulation.simulate_raw_example(
            np.random.default_rng(1133), sim=sim, split="test", shard=0, index=0
        )
    assert not calls
    first = simulation.simulate_raw_example_from_seed((1133, sim, "test", 0, 0))
    second = simulation.simulate_raw_example_from_seed((1133, sim, "test", 0, 0))
    assert first["conditionable_retry_attempt"] > 0
    assert first["conditionable_retry_attempt"] == second["conditionable_retry_attempt"]
    np.testing.assert_array_equal(first["sample_nodes"], second["sample_nodes"])
    assert len(calls) == 2


def test_retry_does_not_hide_a_simulator_error(monkeypatch) -> None:
    sim = sparse_lattice_recipe()
    sim["feems_lattice_sample_prob_choices"] = [1.0]

    def broken(*args, **kwargs):
        raise ValueError("genuine simulator failure")

    monkeypatch.setattr(simulation, "simulate_case_genotypes", broken)
    with pytest.raises(ValueError, match="genuine simulator failure"):
        simulation.simulate_raw_example_from_seed((1133, sim, "test", 0, 0))


def test_retry_budget_is_honored_for_an_unsampleable_design(monkeypatch) -> None:
    sim = sparse_lattice_recipe()
    sim.update(
        feems_lattice_barrier_prob=0,
        feems_lattice_corridor_left_prob=0,
        feems_lattice_corridor_right_prob=0,
        max_conditionable_attempts=3,
    )
    calls = []
    monkeypatch.setattr(
        simulation, "simulate_case_genotypes", lambda *args, **kwargs: calls.append(1)
    )
    with pytest.raises(ValueError, match="after 3 attempts"):
        simulation.simulate_raw_example_from_seed((1133, sim, "test", 0, 0))
    assert not calls
