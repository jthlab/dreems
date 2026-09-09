from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from dreems import counterfactual
from dreems.geography import feems_triangular_lattice_graph


def dense_example() -> dict[str, object]:
    graph, positions = feems_triangular_lattice_graph(4, 4)
    sample_nodes = np.repeat(np.arange(positions.shape[0], dtype=np.int32), 2)
    rng = np.random.default_rng(4)
    factors = rng.normal(size=(sample_nodes.size, 6))
    covariance = factors @ factors.T / factors.shape[1]
    return {
        "graph": graph,
        "active_positions": positions,
        "sample_nodes": sample_nodes,
        "sample_covariance": covariance.astype(np.float32),
        "effective_snps": 100,
        "edge_log_rates": np.linspace(-1.0, 1.0, len(graph["edges"]), dtype=np.float32),
        "node_log_q": np.zeros(positions.shape[0], dtype=np.float32),
        "sample_design": "full",
    }


def test_counterfactual_variants_share_truth_and_subset_covariance() -> None:
    example = dense_example()

    variants = counterfactual.counterfactual_variants(example, group=9, seed=13)

    assert [item["sampling_counterfactual_mask"] for item in variants] == list(
        counterfactual.COUNTERFACTUAL_MASKS
    )
    assert sum(bool(item["sampling_counterfactual_anchor"]) for item in variants) == 1
    assert all(item["sampling_counterfactual_group"] == 9 for item in variants)
    assert counterfactual.COUNTERFACTUAL_MASK_INDEX == {
        "anchor": 0,
        "random_10pct": 1,
        "outer_regions": 2,
        "one_side_missing": 3,
    }
    assert all(
        np.array_equal(item["edge_log_rates"], example["edge_log_rates"]) for item in variants
    )
    observed_counts = [np.unique(item["sample_nodes"]).size for item in variants]
    assert observed_counts[0] == 16
    assert observed_counts[1] == 2
    assert observed_counts[2] == 8
    assert observed_counts[3] == 12
    for item in variants:
        sample_count = len(item["sample_nodes"])
        assert item["sample_covariance"].shape == (sample_count, sample_count)


def test_topology_masks_do_not_depend_on_coordinate_axes() -> None:
    example = dense_example()
    reference = counterfactual.counterfactual_node_sets(example, seed=21)
    transformed = dict(example)
    positions = np.asarray(example["active_positions"])
    transformed["active_positions"] = np.column_stack([-positions[:, 1], positions[:, 0]])

    changed = counterfactual.counterfactual_node_sets(transformed, seed=21)

    for name in counterfactual.COUNTERFACTUAL_MASKS:
        np.testing.assert_array_equal(changed[name], reference[name])


def test_base_selection_balances_available_surface_and_snp_strata(tmp_path: Path) -> None:
    rows = []
    for index, (surface, snps) in enumerate(
        [
            ("feems_homogeneous", 100),
            ("feems_homogeneous", 1_000),
            ("feems_heterogeneous_barrier", 100),
            ("feems_heterogeneous_barrier", 1_000),
            ("feems_anisotropic", 100),
            ("feems_anisotropic", 1_000),
            ("gp", 100),
            ("gp", 1_000),
        ]
    ):
        rows.append(
            {
                "index": index,
                "surface_kind": surface,
                "graph_source": "test",
                "num_nodes": 16,
                "num_sampled_nodes": 16,
                "sampled_node_fraction": 1.0,
                "effective_snps": snps,
            }
        )
    metadata = tmp_path / "pack_0.metadata.json"
    metadata.write_text(json.dumps({"rows": rows}))
    with (tmp_path / "pack_0.pkl").open("wb") as handle:
        pickle.dump([], handle)

    selected = counterfactual.select_base_examples(
        [metadata],
        count=8,
        seed=5,
    )

    assert len(selected) == 8
    assert {item.surface_kind for item in selected} == {
        surface
        for surface, _snps in [
            ("feems_homogeneous", 100),
            ("feems_heterogeneous_barrier", 100),
            ("feems_anisotropic", 100),
            ("gp", 100),
        ]
    }
    assert [item.slot for item in selected] == list(range(8))


def test_subset_keeps_all_sample_representations_aligned() -> None:
    example = dense_example()
    sample_nodes = example["sample_nodes"]
    example["genotypes"] = np.arange(sample_nodes.size * 5, dtype=np.int8).reshape(-1, 5)
    example["sample_coordinates"] = example["active_positions"][sample_nodes]
    selected = np.asarray([2, 3, 7], dtype=np.int32)
    subset = counterfactual.subset_example_to_nodes(example, selected)
    keep = np.flatnonzero(np.isin(sample_nodes, selected))
    for name in ("sample_nodes", "genotypes", "sample_coordinates"):
        np.testing.assert_array_equal(subset[name], example[name][keep])
        assert subset[name].dtype == example[name].dtype
    np.testing.assert_array_equal(
        subset["sample_covariance"], example["sample_covariance"][np.ix_(keep, keep)]
    )
    assert subset["sample_covariance"].dtype == np.float32
    assert subset["graph"] is example["graph"]
    assert subset["edge_log_rates"] is example["edge_log_rates"]
    assert subset["effective_snps"] == example["effective_snps"]
    assert len(example["genotypes"]) == sample_nodes.size


def test_subset_preserves_covariance_dtype_and_values_without_float_roundtrip() -> None:
    example = dense_example()
    size = example["sample_nodes"].size
    covariance = np.arange(size * size, dtype=np.int64).reshape(size, size) + 2**60
    example["sample_covariance"] = covariance
    subset = counterfactual.subset_example_to_nodes(example, np.asarray([0, 1]))
    np.testing.assert_array_equal(subset["sample_covariance"], covariance[:4, :4])


@pytest.mark.parametrize("selected", [[0.5, 1.0], [-1, 1, 2], [0, 1, 16], [0, 2**32 + 1], [[0, 1]]])
def test_subset_rejects_invalid_selected_nodes(selected) -> None:
    with pytest.raises(ValueError, match="selected_nodes"):
        counterfactual.subset_example_to_nodes(dense_example(), np.asarray(selected))


@pytest.mark.parametrize("name", ["sample_covariance", "genotypes", "sample_coordinates"])
def test_subset_checks_sample_axes(name) -> None:
    example = dense_example()
    example[name] = np.ones((2, 2))
    with pytest.raises(ValueError, match=name):
        counterfactual.subset_example_to_nodes(example, np.asarray([0, 1]))


def test_topology_rejects_disconnected_graph_with_some_reachable_samples() -> None:
    with pytest.raises(ValueError, match="connected"):
        counterfactual.topology_axis_scores(4, np.asarray([[0, 1], [2, 3]]), np.arange(4), seed=1)
    # An unsampled disconnected component is invalid too.
    with pytest.raises(ValueError, match="connected"):
        counterfactual.topology_axis_scores(
            4, np.asarray([[0, 1], [2, 3]]), np.asarray([0, 1]), seed=1
        )


@pytest.mark.parametrize(
    ("edges", "observed", "message"),
    [
        ([[0, 1.5], [1, 2]], [0, 2], "edges"),
        ([[0, -1], [1, 2]], [0, 2], "edges"),
        ([[0, 1], [1, 3]], [0, 2], "edges"),
        ([[0, 1], [1, 2]], [0, 2.5], "observed_nodes"),
        ([[0, 1], [1, 2]], [0, -1], "observed_nodes"),
        ([[0, 1], [1, 2]], [], "observed_nodes"),
    ],
)
def test_topology_checks_indices_before_conversion(edges, observed, message) -> None:
    with pytest.raises(ValueError, match=message):
        counterfactual.topology_axis_scores(3, np.asarray(edges), np.asarray(observed), seed=1)


@pytest.mark.parametrize("fraction", [np.nan, np.inf, -0.1, 1.1])
def test_base_selection_rejects_invalid_density_metadata(tmp_path, fraction) -> None:
    path = tmp_path / "pack.metadata.json"
    path.write_text(json.dumps({"rows": [{"sampled_node_fraction": fraction}]}))
    path.with_suffix("").with_suffix(".pkl").touch()
    with pytest.raises(ValueError, match="sampled_node_fraction"):
        counterfactual.select_base_examples([path], count=1, seed=1)
