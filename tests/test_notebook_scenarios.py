from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import yaml

from dreems import notebook_scenarios, workflow_io


def _config() -> dict[str, object]:
    return {"notebook_scenarios": {}}


@pytest.mark.parametrize("value", [2.5, -1, True, "1000", np.nan, [2]])
@pytest.mark.parametrize("key", ["num_snps", "retained_snps"])
def test_notebook_settings_reject_invalid_counts(key, value):
    with pytest.raises(ValueError):
        notebook_scenarios.scenario_count({"notebook_scenarios": {key: value}})


@pytest.mark.parametrize("value", [-1, 0.25, 1j])
def test_notebook_maf_filter_rejects_incomplete_or_noninteger_calls(value):
    calls = np.full((3, 4, 2), value)
    with pytest.raises(ValueError, match="integer"):
        notebook_scenarios.maf_filtered_dosages(calls)


def test_notebook_haplotype_layout_must_include_complete_diploids():
    with pytest.raises(ValueError, match="deme order"):
        notebook_scenarios._calls_for_nodes(np.zeros((3, 12), dtype=np.uint8), np.asarray([0]))


def test_notebook_graph_preserves_networkx_edge_order() -> None:
    graph, positions = notebook_scenarios.notebook_graph()
    expected = nx.generators.lattice.triangular_lattice_graph(7, 22, with_positions=True)
    expected = nx.convert_node_labels_to_integers(expected)
    np.testing.assert_array_equal(
        positions,
        np.asarray(list(nx.get_node_attributes(expected, "pos").values())),
    )
    np.testing.assert_array_equal(graph["edges"], np.asarray(expected.edges, dtype=np.int32))
    assert positions.shape == (96, 2)
    assert graph["edges"].shape == (249, 2)


def test_notebook_specs_and_masks_match_source_cells() -> None:
    config = _config()
    assert notebook_scenarios.scenario_count(config) == 12
    expected_counts = {
        "barrier": (96, 9, 48, 72),
        "anisotropy": (96, 19, 48, 72),
        "homogeneous": (96, 9, 48, 72),
    }
    for example_id in range(12):
        spec = notebook_scenarios.scenario_spec(config, example_id)
        mask_index = notebook_scenarios.MASKS.index(spec.mask)
        nodes = notebook_scenarios.sampled_nodes(spec.truth, spec.mask)
        assert nodes.size == expected_counts[spec.truth][mask_index]
        assert notebook_scenarios.observed_node_count(config, example_id) == nodes.size

    expected_random_10 = np.random.default_rng(6).choice(96, size=9, replace=False)
    expected_random_20 = np.random.default_rng(6).choice(96, size=19, replace=False)
    np.testing.assert_array_equal(
        notebook_scenarios.sampled_nodes("barrier", "random_10pct"),
        expected_random_10,
    )
    np.testing.assert_array_equal(
        notebook_scenarios.sampled_nodes("anisotropy", "random_10pct"),
        expected_random_20,
    )


def test_notebook_truths_match_barrier_and_anisotropy_rules() -> None:
    graph, positions = notebook_scenarios.notebook_graph()
    edges = graph["edges"]
    horizontal = positions[edges[:, 0], 1] == positions[edges[:, 1], 1]

    barrier = notebook_scenarios.edge_rates("barrier", positions, edges)
    anisotropy = notebook_scenarios.edge_rates("anisotropy", positions, edges)
    homogeneous = notebook_scenarios.edge_rates("homogeneous", positions, edges)
    np.testing.assert_allclose(np.unique(barrier), [1.0e-5, 5.0e-5])
    np.testing.assert_allclose(anisotropy[horizontal], 5.0e-5)
    np.testing.assert_allclose(anisotropy[~horizontal], 1.0e-5)
    np.testing.assert_allclose(homogeneous, 1.0e-5)


def test_maf_filter_is_applied_after_sample_subsetting() -> None:
    calls = np.zeros((4, 10, 2), dtype=np.uint8)
    calls[0, :2, 0] = 1  # MAF 0.10: retained.
    calls[1, 0, 0] = 1  # MAF 0.05: retained at the boundary.
    calls[2, :, :] = 1  # Fixed derived: removed.
    calls[3, 0, 0] = 2  # Multiallelic state: removed.
    dosages, keep = notebook_scenarios.maf_filtered_dosages(calls)
    np.testing.assert_array_equal(keep, [True, True, False, False])
    assert dosages.shape == (10, 2)


def test_model_snp_target_is_applied_after_maf_filtering() -> None:
    calls = np.zeros((4, 10, 2), dtype=np.uint8)
    calls[0, :2, 0] = 1
    calls[1, :3, 0] = 1
    calls[2, 0, 0] = 1
    calls[3, :, :] = 1

    selected_calls, dosages, available = notebook_scenarios.select_model_snps(
        calls,
        target_snps=2,
    )

    assert available == 3
    np.testing.assert_array_equal(selected_calls, calls[:2])
    assert dosages.shape == (10, 2)
    with pytest.raises(ValueError, match="only 3 pass"):
        notebook_scenarios.select_model_snps(calls, target_snps=4)


def test_unsigned_notebook_calls_use_the_public_genotype_path() -> None:
    graph, positions = notebook_scenarios.notebook_graph()
    nodes = np.asarray([0, 1], dtype=np.int32)
    calls = np.random.default_rng(3).integers(0, 2, size=(12, 20, 2), dtype=np.uint8)
    data = notebook_scenarios._notebook_dataset(
        calls, selected_nodes=nodes, raw_positions=positions, edges=graph["edges"]
    )
    dosages, _ = notebook_scenarios.maf_filtered_dosages(calls)
    standardized = (dosages - dosages.mean(axis=0)) / dosages.std(axis=0, ddof=1)
    deme_means = standardized.reshape(2, 10, -1).mean(axis=1)
    expected = deme_means @ deme_means.T / standardized.shape[1]
    np.testing.assert_allclose(data.node_covariance.values, expected, rtol=1e-6, atol=1e-8)
    np.testing.assert_array_equal(data.observed_nodes.values, nodes)
    assert data.attrs["retained_variants"] == standardized.shape[1]
    assert calls.dtype == np.uint8


def test_exact_h2h_config_records_source_checksum() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "workflow" / "config.feems_scenarios_exact_h2h.yaml").read_text()
    )
    settings = config["notebook_scenarios"]
    source = Path(settings["source_notebook"])
    assert source.is_file()
    assert workflow_io.sha256(source) == settings["source_notebook_sha256"]
    assert config["example_count"] == 12
    assert config["feems_fold_runtime"] == "4h"
