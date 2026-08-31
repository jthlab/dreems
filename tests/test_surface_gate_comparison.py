from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dreems.surface_gate_comparison import (
    build_scenario_bundle,
    edge_metrics,
    load_npz,
    projected_alpha,
    select_broad_examples,
    sha256,
)


def test_edge_metrics_and_projection_use_centered_surfaces() -> None:
    truth = np.asarray([-1.0, 0.0, 1.0])
    estimate = 0.5 * truth + 12.0

    metrics = edge_metrics(truth, estimate)

    assert np.isclose(metrics.correlation, 1.0)
    assert np.isclose(metrics.mse, np.mean((0.5 * truth) ** 2))
    assert np.isclose(projected_alpha(truth, estimate), 0.5)


def test_build_scenario_bundle_matches_core_gate_and_feems(tmp_path: Path) -> None:
    examples = 12
    positions = np.broadcast_to(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        (examples, 3, 2),
    ).copy()
    edges = np.broadcast_to(np.asarray([[0, 1], [1, 2]]), (examples, 2, 2)).copy()
    truth = np.tile(np.asarray([-1.0, 1.0]), (examples, 1))
    truth[8:] = 0.0
    core_map = 0.8 * truth + np.asarray([0.2, -0.2])
    core_neural = 0.9 * truth + np.asarray([0.3, -0.3])
    feems = 0.7 * truth + np.asarray([0.1, -0.1])
    truth_names = np.repeat(np.asarray(["barrier", "anisotropy", "homogeneous"]), 4)
    mask_names = np.tile(
        np.asarray(["full", "random_10pct", "outer_slabs", "left_missing"]),
        3,
    )
    sample_count = np.ones((examples, 3), dtype=np.float64)

    core_bundle = tmp_path / "core.npz"
    np.savez_compressed(
        core_bundle,
        index=np.arange(examples),
        truth_name=truth_names,
        mask_name=mask_names,
        sample_design=mask_names,
        num_snps=np.full(examples, 1000),
        positions=positions,
        edges=edges,
        truth=truth,
        dreems=core_neural,
        dreems_map=core_map,
        edge_sd=np.ones_like(truth),
        feems=feems,
        sample_count=sample_count,
        checkpoint_sha256=np.asarray("core-checkpoint-hash"),
        source_notebook_sha256=np.asarray("source-hash"),
        feems_protocol=np.asarray("loo"),
    )

    for example in range(examples):
        alpha = 0.1 if example >= 8 else 1.0
        np.savez_compressed(
            tmp_path / f"gate_{example}.npz",
            positions=positions[example],
            edges=edges[example],
            truth=truth[example],
            sample_count=sample_count[example],
            dreems=alpha * core_neural[example],
            dreems_map=alpha * core_map[example],
            edge_sd=np.ones_like(truth[example]),
        )

    checkpoint = tmp_path / "gate.zip"
    checkpoint.write_bytes(b"gate checkpoint")
    output = tmp_path / "comparison.npz"
    rows = tmp_path / "rows.csv"
    report = tmp_path / "report.json"
    build_scenario_bundle(
        core_bundle_path=core_bundle,
        gated_pattern=str(tmp_path / "gate_{example}.npz"),
        gated_checkpoint=checkpoint,
        output=output,
        rows_output=rows,
        report_output=report,
    )

    bundle = load_npz(output)
    payload = json.loads(report.read_text())
    assert bundle["index"].shape == (12,)
    assert np.allclose(bundle["gate_alpha_neural"][:8], 1.0)
    assert np.allclose(bundle["gate_alpha_neural"][8:], 0.1)
    assert bundle["gated_checkpoint_sha256"].item() == sha256(checkpoint)
    assert payload["by_truth"]["homogeneous"]["gated_vs_core"]["mean_mse_advantage"] > 0.0
    assert rows.exists()


def test_broad_selection_is_deterministic() -> None:
    rows = [
        {
            "example": index,
            "alpha": 1.0 - index / 1000.0,
            "baseline_map_mse": 1.0,
            "gated_map_mse": 1.0 + (index - 10) / 100.0,
        }
        for index in range(20)
    ]

    first = select_broad_examples(rows, count=4)
    second = select_broad_examples(rows, count=4)

    assert first == second
    assert first["Strongest gate shrinkage"] == [19, 18, 17, 16]
    assert first["Largest gate regressions"] == [19, 18, 17, 16]
