from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from dreems.evaluation import benchmark_gallery


def evaluation_rows(offset: float) -> list[dict[str, float | int]]:
    return [
        {
            "replicate": index,
            "replicate_global": index,
            "edge_log_corr": 0.4 + offset + 0.05 * index,
            "edge_log_mse": 0.8 - offset - 0.05 * index,
            "edge_log_rmse": 0.8,
            "edge_log_slope": 0.5,
            "edge_log_scale_ratio": 0.6,
            "edge_sd_mean": 0.7,
            "edge_50_coverage": 0.5,
            "edge_80_coverage": 0.8,
            "edge_95_coverage": 0.95,
            "q_log_corr": 0.2,
            "q_log_mse": 0.3,
            "q_80_coverage": 0.8,
        }
        for index in range(4)
    ]


def test_merge_checkpoint_rows_and_fixed_snp_bins(tmp_path: Path) -> None:
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    metadata = tmp_path / "metadata.json"
    old.write_text(json.dumps({"rows": evaluation_rows(0.0)}))
    new.write_text(json.dumps({"rows": evaluation_rows(0.1)}))
    counts = [50, 129, 513, 2001]
    metadata.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "index": index,
                        "num_snps": count,
                        "num_samples": 12,
                        "sampled_node_fraction": 0.2,
                    }
                    for index, count in enumerate(counts)
                ]
            }
        )
    )

    rows = benchmark_gallery.merge_checkpoint_rows(old, new, [metadata])

    assert [row["snp_bin"] for row in rows] == [
        "50-128",
        "129-512",
        "513-2,000",
        "2,001-10,000",
    ]
    assert all(row["corr_delta"] == pytest.approx(0.1) for row in rows)
    assert all(row["mse_advantage"] == pytest.approx(0.1) for row in rows)
    summary = benchmark_gallery.paired_stats(rows, label="all")
    assert summary["n"] == 4
    assert summary["n_corr"] == 4
    assert summary["n_mse"] == 4
    assert summary["corr_win_fraction"] == 1.0
    assert summary["mse_win_fraction"] == 1.0


def test_paired_stats_keeps_mse_when_correlation_is_undefined() -> None:
    rows = [
        {
            "old_edge_log_corr": np.nan,
            "new_edge_log_corr": np.nan,
            "old_edge_log_mse": 0.8,
            "new_edge_log_mse": 0.6,
        },
        {
            "old_edge_log_corr": 0.4,
            "new_edge_log_corr": 0.5,
            "old_edge_log_mse": 0.7,
            "new_edge_log_mse": 0.5,
        },
    ]

    summary = benchmark_gallery.paired_stats(rows, label="test")

    assert summary["n"] == 2
    assert summary["n_corr"] == 1
    assert summary["n_mse"] == 2
    assert summary["mse_advantage"] == pytest.approx(0.2)


def test_merge_scenario_snp_rows_requires_a_paired_design(tmp_path: Path) -> None:
    low = tmp_path / "low.csv"
    high = tmp_path / "high.csv"
    common = {
        "index": 0,
        "scenario_truth": "barrier",
        "scenario_mask": "full",
        "scenario_replicate": 0,
        "scenario_seed": 10,
        "num_nodes": 96,
        "num_edges": 249,
        "num_samples": 960,
        "num_sampled_nodes": 96,
        "dreems_edge_log_corr": 0.8,
        "dreems_edge_log_mse": 0.3,
        "dreems_edge_log_rms": 0.5,
        "dreems_map_edge_log_corr": 0.75,
        "dreems_map_edge_log_mse": 0.25,
        "dreems_map_edge_log_rms": 0.45,
        "feems_edge_log_corr": 0.6,
        "feems_edge_log_mse": 0.4,
        "feems_edge_log_rms": 0.4,
    }
    benchmark_gallery.write_rows(low, [{**common, "num_snps": 75}])
    benchmark_gallery.write_rows(
        high,
        [
            {
                **common,
                "num_snps": 1000,
                "dreems_edge_log_corr": 0.9,
                "dreems_edge_log_mse": 0.2,
            }
        ],
    )

    rows = benchmark_gallery.merge_scenario_snp_rows(low, high)

    assert rows[0]["dreems_corr_gain"] == pytest.approx(0.1)
    assert rows[0]["dreems_mse_reduction"] == pytest.approx(0.1)
    assert rows[0]["low_snps"] == 75
    assert rows[0]["high_snps"] == 1000

    benchmark_gallery.write_rows(
        high,
        [{**common, "scenario_mask": "left_missing", "num_snps": 1000}],
    )
    with pytest.raises(ValueError, match="scenario_mask"):
        benchmark_gallery.merge_scenario_snp_rows(low, high)


def test_prediction_slice_uses_replicate_offsets(tmp_path: Path) -> None:
    path = tmp_path / "predictions.npz"
    np.savez_compressed(
        path,
        replicate=np.array([65, 64]),
        edge_offsets=np.array([0, 2, 5]),
        edge_truth=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        edge_mean=np.array([1.1, 2.1, 3.1, 4.1, 5.1]),
        edge_sd=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    )

    truth, mean, sd = benchmark_gallery.prediction_slice(path, 64)

    np.testing.assert_array_equal(truth, np.array([3.0, 4.0, 5.0]))
    np.testing.assert_allclose(mean, np.array([3.1, 4.1, 5.1]))
    np.testing.assert_allclose(sd, np.array([0.3, 0.4, 0.5]))


def test_router_selects_checkpoint_from_snp_count() -> None:
    rows = [
        {
            "replicate": 0,
            "num_snps": 512,
            "old_edge_log_corr": 0.4,
            "new_edge_log_corr": 0.6,
            "old_edge_log_mse": 0.8,
            "new_edge_log_mse": 0.5,
        },
        {
            "replicate": 1,
            "num_snps": 513,
            "old_edge_log_corr": 0.7,
            "new_edge_log_corr": 0.5,
            "old_edge_log_mse": 0.4,
            "new_edge_log_mse": 0.6,
        },
    ]

    routed = benchmark_gallery.build_router_rows(rows, threshold=512)
    summary = benchmark_gallery.router_stats(routed, label="all")

    assert [row["router_checkpoint"] for row in routed] == ["broad", "released"]
    assert [row["router_edge_log_corr"] for row in routed] == [0.6, 0.7]
    assert [row["router_edge_log_mse"] for row in routed] == [0.5, 0.4]
    assert summary["router_corr"] == pytest.approx(0.65)
    assert summary["router_mse"] == pytest.approx(0.45)
    assert summary["router_mse_win_fraction_vs_alternative"] == 1.0


def test_router_h2h_reference_requires_released_route(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "index.html").write_text(
        "<h1>DREEMS + fixed-radius MAP vs FEEMS recommended LOO</h1><h3>DREEMS + MAP best</h3>"
    )
    (reference / "panel.png").write_bytes(b"panel")
    rows = tmp_path / "rows.csv"
    benchmark_gallery.write_rows(
        rows,
        [{"index": 0, "num_snps": 2_000}, {"index": 1, "num_snps": 10_000}],
    )
    output = tmp_path / "output"
    benchmark_gallery.command_router_h2h_gallery(
        Namespace(
            reference_gallery=reference,
            reference_rows=rows,
            threshold=512,
            output_dir=output,
        )
    )

    assert (output / "panel.png").read_bytes() == b"panel"
    page = (output / "index.html").read_text()
    assert "DREEMS router + fixed-radius MAP" in page
    assert "selected Released for every case" in page
    selections = benchmark_gallery.read_csv_rows(output / "router_selections.csv")
    assert {row["router_checkpoint"] for row in selections} == {"released"}

    benchmark_gallery.write_rows(rows, [{"index": 0, "num_snps": 128}])
    with pytest.raises(ValueError, match="select Broad-SNP"):
        benchmark_gallery.command_router_h2h_gallery(
            Namespace(
                reference_gallery=reference,
                reference_rows=rows,
                threshold=512,
                output_dir=tmp_path / "invalid",
            )
        )


def test_h2h_example_figure_smoke(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    edges = np.array([[0, 1], [1, 2], [2, 0]])
    truth = np.array([-0.5, 0.0, 0.5])
    dreems_path = tmp_path / "dreems_0.npz"
    feems_path = tmp_path / "feems_0.npz"
    np.savez_compressed(
        dreems_path,
        positions=positions,
        edges=edges,
        truth=truth,
        dreems_map=truth * 0.8,
        sample_count=np.array([1.0, 0.0, 2.0]),
    )
    np.savez_compressed(
        feems_path,
        feems=truth * 0.6,
        positions=positions,
        edges=edges,
        truth=truth,
        sample_count=np.array([1.0, 0.0, 2.0]),
    )
    rows = {
        0: {
            "index": "0",
            "num_snps": "100",
            "num_samples": "3",
            "sampled_node_fraction": "0.667",
            "dreems_map_edge_log_corr": "1.0",
            "dreems_map_edge_log_mse": "0.01",
            "feems_edge_log_corr": "1.0",
            "feems_edge_log_mse": "0.04",
        }
    }
    output = tmp_path / "example.png"

    benchmark_gallery.h2h_example_figure(
        rows,
        [0],
        dreems_pattern=str(tmp_path / "dreems_{example}.npz"),
        feems_pattern=str(tmp_path / "feems_{example}.npz"),
        output=output,
    )

    assert output.exists()
    assert output.with_suffix(".pdf").exists()
