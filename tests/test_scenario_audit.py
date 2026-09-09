from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest

from dreems import benchmark_gallery, scenario_audit


@pytest.mark.parametrize(
    "key,value",
    [
        ("replicates", 1.5),
        ("seed", -1),
        ("rows", 4.5),
        ("columns", True),
        ("num_snps", 3.5),
        ("ploidy", 2.5),
        ("diploids_per_sampled_node", 0),
        ("random_sample_fraction", 0.0),
        ("random_sample_fraction", 1.2),
        ("barrier_rate", -1.0),
        ("anisotropy_scaler", float("nan")),
    ],
)
def test_scenario_settings_reject_silent_conversions(scenario_config, key, value):
    scenario_config["scenario_audit"][key] = value
    with pytest.raises(ValueError):
        scenario_audit.scenario_components(scenario_config, 0)


@pytest.fixture
def scenario_config() -> dict[str, object]:
    return {
        "scenario_audit": {
            "seed": 20260830,
            "replicates": 3,
            "truths": ["barrier", "anisotropy", "homogeneous"],
            "masks": ["full", "random_10pct", "outer_slabs", "left_missing"],
            "rows": 8,
            "columns": 12,
            "barrier_start": 2.5,
            "barrier_end": 8.5,
            "barrier_rate": 0.1,
            "corridor_rate": 0.5,
            "anisotropy_scaler": 5.0,
            "random_sample_fraction": 0.10,
            "diploids_per_sampled_node": 10,
        }
    }


def test_crossed_design_has_stable_ids(scenario_config: dict[str, object]) -> None:
    assert scenario_audit.scenario_count(scenario_config) == 36
    first = scenario_audit.scenario_spec(scenario_config, 0)
    last = scenario_audit.scenario_spec(scenario_config, 35)
    assert (first.replicate, first.truth, first.mask) == (0, "barrier", "full")
    assert (last.replicate, last.truth, last.mask) == (2, "homogeneous", "left_missing")
    with pytest.raises(ValueError, match="example_id"):
        scenario_audit.scenario_spec(scenario_config, 36)


def test_masks_match_notebook_and_are_crossed_with_truth(
    scenario_config: dict[str, object],
) -> None:
    expected_counts = {
        "full": 96,
        "random_10pct": 9,
        "outer_slabs": 48,
        "left_missing": 72,
    }
    masks_by_truth: dict[str, dict[str, np.ndarray]] = {}
    for example_id in range(12):
        spec, _graph, _positions, _truth, sample_nodes = scenario_audit.scenario_components(
            scenario_config, example_id
        )
        observed = np.unique(sample_nodes)
        assert observed.size == expected_counts[spec.mask]
        assert sample_nodes.size == 10 * expected_counts[spec.mask]
        masks_by_truth.setdefault(spec.truth, {})[spec.mask] = observed

    for mask in expected_counts:
        np.testing.assert_array_equal(
            masks_by_truth["barrier"][mask],
            masks_by_truth["anisotropy"][mask],
        )
        np.testing.assert_array_equal(
            masks_by_truth["barrier"][mask],
            masks_by_truth["homogeneous"][mask],
        )


def test_truths_have_the_intended_edge_patterns(scenario_config: dict[str, object]) -> None:
    truths: dict[str, tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]] = {}
    for example_id in (0, 4, 8):
        spec, graph, positions, truth, _sample_nodes = scenario_audit.scenario_components(
            scenario_config, example_id
        )
        truths[spec.truth] = (graph, positions, truth)

    barrier = truths["barrier"][2]
    assert np.unique(np.round(barrier, 6)).size == 2

    graph, positions, anisotropy = truths["anisotropy"]
    edges = np.asarray(graph["edges"], dtype=np.int32)
    horizontal = np.isclose(positions[edges[:, 0], 1], positions[edges[:, 1], 1])
    assert float(anisotropy[horizontal].mean()) > float(anisotropy[~horizontal].mean())

    homogeneous = truths["homogeneous"][2]
    np.testing.assert_allclose(homogeneous, 0.0)


def test_scenario_diagnostics_cross_truth_and_edge_observability(tmp_path, monkeypatch) -> None:
    positions = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    edges = np.asarray([[0, 1], [0, 2], [1, 3], [2, 3]], dtype=np.int32)
    rows = []
    example = 0
    for truth_index, truth in enumerate(benchmark_gallery.SCENARIO_TRUTHS):
        for mask in benchmark_gallery.SCENARIO_MASKS:
            sample_count = np.asarray([1, 1, 0, 0], dtype=np.int32)
            values = np.asarray([1.0, -1.0, 0.5, -0.5]) + 0.1 * truth_index
            np.savez_compressed(
                tmp_path / f"dreems_{example}.npz",
                positions=positions,
                edges=edges,
                truth=values,
                sample_count=sample_count,
                edge_sd=np.ones(4),
                conditioner=values,
                dreems=values,
                dreems_map=values,
            )
            np.savez_compressed(
                tmp_path / f"feems_{example}.npz",
                positions=positions,
                edges=edges,
                truth=values,
                sample_count=sample_count,
                feems=values,
            )
            rows.append(
                {
                    "index": example,
                    "scenario_replicate": 0,
                    "scenario_truth": truth,
                    "scenario_mask": mask,
                    "num_snps": 1000,
                    "posterior_sd_mean": 0.2,
                    "neural_95_coverage": 0.95,
                    **{
                        f"{prefix}_{metric}": 0.1
                        for _key, _label, prefix in benchmark_gallery.SCENARIO_METHODS
                        for metric in ("corr", "mse", "rms")
                    },
                }
            )
            example += 1

    lookup = benchmark_gallery._scenario_row_lookup(rows)
    similarity = benchmark_gallery._scenario_similarity_rows(
        lookup,
        dreems_pattern=str(tmp_path / "dreems_{example}.npz"),
        feems_pattern=str(tmp_path / "feems_{example}.npz"),
    )
    observability = benchmark_gallery._scenario_observability_rows(
        rows,
        dreems_pattern=str(tmp_path / "dreems_{example}.npz"),
        feems_pattern=str(tmp_path / "feems_{example}.npz"),
    )
    assert len(similarity) == 4 * 4 * 3
    assert {row["sampled_endpoints"] for row in observability} == {0, 1, 2}

    rows_path = tmp_path / "rows.csv"
    output_dir = tmp_path / "gallery"
    benchmark_gallery.write_rows(rows_path, rows)

    def fake_save_figure(_figure, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test figure")

    monkeypatch.setattr(benchmark_gallery, "save_figure", fake_save_figure)
    benchmark_gallery.command_scenario_gallery(
        Namespace(
            rows=rows_path,
            dreems_pattern=str(tmp_path / "dreems_{example}.npz"),
            feems_pattern=str(tmp_path / "feems_{example}.npz"),
            output_dir=output_dir,
        )
    )
    assert (output_dir / "index.html").is_file()
    assert (output_dir / "scenario_summary.json").is_file()
