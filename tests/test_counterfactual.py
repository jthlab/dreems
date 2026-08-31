from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

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
