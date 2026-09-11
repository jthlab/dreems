from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dreems.evaluation import surface_gate_comparison as comparison
from dreems.evaluation.surface_gate_comparison import (
    build_scenario_bundle,
    edge_metrics,
    projected_alpha,
    select_broad_examples,
    sha256,
)
from dreems.utils import workflow_io


def test_edge_metrics_and_projection_use_centered_surfaces() -> None:
    truth = np.asarray([-1.0, 0.0, 1.0])
    estimate = 0.5 * truth + 12.0

    metrics = edge_metrics(truth, estimate)

    assert np.isclose(metrics.correlation, 1.0)
    assert np.isclose(metrics.mse, np.mean((0.5 * truth) ** 2))
    assert np.isclose(projected_alpha(truth, estimate), 0.5)


@pytest.fixture
def scenario_inputs(tmp_path: Path) -> dict:
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
    return dict(
        core_bundle_path=core_bundle,
        gated_pattern=str(tmp_path / "gate_{example}.npz"),
        gated_checkpoint=checkpoint,
        output=output,
        rows_output=rows,
        report_output=report,
    )


def test_build_scenario_bundle_matches_core_gate_and_feems(scenario_inputs: dict) -> None:
    build_scenario_bundle(**scenario_inputs)
    bundle = workflow_io.read_npz(scenario_inputs["output"])
    payload = json.loads(scenario_inputs["report_output"].read_text())
    assert bundle["index"].shape == (12,)
    assert np.allclose(bundle["gate_alpha_neural"][:8], 1.0)
    assert np.allclose(bundle["gate_alpha_neural"][8:], 0.1)
    assert bundle["gated_checkpoint_sha256"].item() == sha256(scenario_inputs["gated_checkpoint"])
    assert payload["by_truth"]["homogeneous"]["gated_vs_core"]["mean_mse_advantage"] > 0.0
    assert scenario_inputs["rows_output"].exists()


def test_comparison_notebook_runs_with_current_api(scenario_inputs: dict) -> None:
    jupytext = pytest.importorskip("jupytext")
    import matplotlib.pyplot as plt

    build_scenario_bundle(**scenario_inputs)
    source = Path(__file__).parents[1] / "workflow/notebooks/feems_comparison_surface_gate.md"
    notebook = jupytext.read(source)
    namespace = {"__name__": "__main__"}
    try:
        for cell in notebook.cells:
            if cell.cell_type == "code":
                code = cell.source.replace(
                    'Path("__SCENARIO_BUNDLE__")', f"Path({str(scenario_inputs['output'])!r})"
                )
                exec(compile(code, str(source), "exec"), namespace)
        np.testing.assert_array_equal(namespace["bundle"]["index"], np.arange(12))
    finally:
        plt.close("all")


@pytest.mark.parametrize(
    "field", ["index", "edges", "num_snps", "truth", "edge_sd", "positions", "truth_name"]
)
def test_scenario_bundle_rejects_invalid_core_arrays(scenario_inputs, field):
    path = scenario_inputs["core_bundle_path"]
    arrays = workflow_io.read_npz(path)
    if field in {"index", "edges", "num_snps"}:
        arrays[field] = arrays[field].astype(float) + 0.25
    elif field == "truth":
        arrays[field] = arrays[field].astype(complex) + 1j
    elif field == "edge_sd":
        arrays[field] *= -1
    elif field == "positions":
        arrays[field] = np.asarray(1.0)
    else:
        arrays[field] = arrays[field][::-1]
    workflow_io.write_npz(path, **arrays)
    scenario_inputs["output"].write_bytes(b"previous bundle")
    with pytest.raises(ValueError):
        build_scenario_bundle(**scenario_inputs)
    assert scenario_inputs["output"].read_bytes() == b"previous bundle"


@pytest.mark.parametrize("change", ["shape", "negative_sd", "checkpoint"])
def test_scenario_bundle_rejects_invalid_gated_prediction(scenario_inputs, change):
    path = Path(scenario_inputs["gated_pattern"].format(example=0))
    arrays = workflow_io.read_npz(path)
    if change == "shape":
        arrays["dreems"] = np.zeros(3)
    elif change == "negative_sd":
        arrays["edge_sd"] *= -1
    else:
        arrays["checkpoint_sha256"] = np.asarray("another checkpoint")
    workflow_io.write_npz(path, **arrays)
    with pytest.raises(ValueError):
        build_scenario_bundle(**scenario_inputs)
    assert not scenario_inputs["output"].exists()


def test_writers_use_the_shared_atomic_implementation(tmp_path):
    assert comparison.write_npz is workflow_io.write_npz
    assert comparison.sha256 is workflow_io.sha256
    path = tmp_path / "report.json"
    path.write_bytes(b"old report")
    with pytest.raises(TypeError):
        comparison.write_json(path, {"invalid": object()})
    assert path.read_bytes() == b"old report"


@pytest.mark.parametrize("value", ["1.5", "NaN", "inf", -1, True, None])
def test_row_ids_must_be_exact_nonnegative_integers(value):
    with pytest.raises(ValueError):
        comparison._as_int({"example": value})


def test_large_csv_id_does_not_round_through_float():
    assert comparison._as_int({"example": "9007199254740993"}) == 9007199254740993
    assert comparison._as_int({"example": "12.0"}) == 12


def test_duplicate_case_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        select_broad_examples([{"example": 0}, {"example": 0}], count=1)


def test_homogeneous_perfect_fits_can_be_plotted():
    row = {key: 0.0 for key in ("baseline_map_mse", "gated_map_mse", "feems_mse")}
    row.update({key: np.nan for key in ("baseline_map_corr", "gated_map_corr", "feems_corr")})
    row.update(alpha=0.0, sampled_node_fraction=0.1)
    figure = comparison.broad_aggregate_figure([row])
    try:
        figure.canvas.draw()
        assert figure.axes[1].get_xscale() == "linear"
        assert "1 structured cases" in figure.axes[0].get_title()
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


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
