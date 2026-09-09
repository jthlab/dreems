"""Shared benchmark input checks and public report serialization."""

from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dreems import benchmark_data, cli, datasets, feems_benchmark, h2h, workflow_io


@pytest.fixture
def raw() -> dict:
    return {
        "graph": {"edges": np.asarray([[0, 1], [1, 2]], dtype=np.int32)},
        "active_positions": np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        "sample_nodes": np.asarray([0, 1, 2], dtype=np.int32),
        "genotypes": np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]]),
        "edge_log_rates": np.asarray([-0.5, 0.5]),
    }


@pytest.mark.parametrize("field", ["edges", "sample_nodes"])
@pytest.mark.parametrize("bad", [-1.0, 0.5, 2.0**32 + 1])
def test_benchmark_indices_are_checked_before_cast(raw: dict, field: str, bad: float) -> None:
    target = raw["graph"] if field == "edges" else raw
    target[field] = target[field].astype(np.float64)
    target[field].flat[0] = bad
    with pytest.raises(ValueError):
        benchmark_data._example_arrays(raw)


def test_shared_benchmark_loader_preserves_order_and_truth(raw: dict, tmp_path: Path) -> None:
    path = tmp_path / "raw.pkl"
    workflow_io.write_pickle(path, [raw])
    loaded = benchmark_data._load_example(path)
    positions, edges, nodes, genotypes, truth = benchmark_data._example_arrays(loaded)
    for actual, expected in zip(
        (positions, edges, nodes, genotypes, truth),
        (
            raw["active_positions"],
            raw["graph"]["edges"],
            raw["sample_nodes"],
            raw["genotypes"],
            raw["edge_log_rates"],
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    assert workflow_io.sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("writer", ["csv", "npz"])
def test_failed_report_write_preserves_existing_output(
    tmp_path: Path, monkeypatch, writer: str
) -> None:
    path = tmp_path / f"report.{writer}"
    path.write_bytes(b"previous report")
    if writer == "csv":
        with pytest.raises(ValueError, match="fields not in fieldnames"):
            cli.write_records(path, [{"x": 1}, {"extra": 2}])
    else:

        def fail(handle, **arrays):
            handle.write(b"partial archive")
            raise OSError("interrupted write")

        monkeypatch.setattr(np, "savez_compressed", fail)
        with pytest.raises(OSError, match="interrupted write"):
            workflow_io.write_npz(path, data=np.ones(2))
    assert path.read_bytes() == b"previous report"
    assert list(tmp_path.iterdir()) == [path]


def test_numpy_boolean_and_path_metadata_are_json_safe(tmp_path: Path) -> None:
    value = {"converged": np.bool_(True), "path": tmp_path, "history": np.asarray([1.0, 2.0])}
    converted = workflow_io.json_ready(value)
    assert json.loads(json.dumps(converted)) == {
        "converged": True,
        "path": str(tmp_path),
        "history": [1.0, 2.0],
    }
    assert json.loads(json.dumps(value, default=cli.json_default)) == converted
    with pytest.raises(TypeError, match="not JSON serializable"):
        cli.json_default(object())


def test_dataset_builder_keeps_array_like_inputs(raw: dict) -> None:
    dataset = datasets.create_dataset(
        node_positions=raw["active_positions"].tolist(),
        edges=raw["graph"]["edges"].tolist(),
        node_covariance=np.eye(3).tolist(),
        node_sample_count=[1, 1, 1],
        effective_snps=np.int64(100),
    )
    datasets.validate_dataset(dataset)
    assert dataset.effective_snps.item() == 100


def test_combined_cli_uses_standalone_feems_workers() -> None:
    parser = h2h.build_parser()
    args = parser.parse_args(
        ["feems-fold", "--input", "example.pkl", "--fold", "1", "--output-npz", "fold.npz"]
    )
    assert args.handler is feems_benchmark.run_feems_fold
    assert h2h.run_feems_fold is feems_benchmark.run_feems_fold
    assert h2h.run_feems_case is feems_benchmark.run_feems_case


def test_feems_worker_uses_validated_arrays_and_official_loo(raw, tmp_path, monkeypatch) -> None:
    path = tmp_path / "case.pkl"
    workflow_io.write_pickle(path, raw)
    calls = []

    def fit(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            edge_log_rates=np.asarray([-0.2, 0.2]),
            cv_mode="leave_one_out",
            cv_folds=3,
            protocol="v2_joint_lambda_lambda_q",
            best_lambda=1.0,
            best_lambda_q=0.1,
            n_snps_after_filter=2,
            train_loss=0.5,
            mean_cv_error=np.asarray([1.0, 0.5]),
        )

    monkeypatch.setattr(feems_benchmark.feems, "fit_feems_recommended_loo", fit)
    output = tmp_path / "fit.json"
    feems_benchmark.run_feems_case(
        Namespace(
            input=path,
            index=0,
            cv_jobs=1,
            cv_checkpoint_dir=None,
            output_json=output,
            output_npz=tmp_path / "fit.npz",
        )
    )
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0]["sample_nodes"], raw["sample_nodes"])
    assert calls[0]["cv_checkpoint_key"].endswith(workflow_io.sha256(path))
    report = json.loads(output.read_text())
    assert report["input_sha256"] == workflow_io.sha256(path)
    assert report["cv_folds"] == 3
    assert report["method"] == "feems"


def test_fold_worker_refuses_to_label_old_data_with_a_new_hash(raw, tmp_path, monkeypatch) -> None:
    path = tmp_path / "case.pkl"
    workflow_io.write_pickle(path, raw)
    output = tmp_path / "fold.npz"
    output.write_bytes(b"previous fold")

    def fit(**kwargs):
        path.write_bytes(b"replaced while fitting")
        return np.ones((5, 10))

    monkeypatch.setattr(feems_benchmark.feems, "recommended_loo_fold_errors", fit)
    with pytest.raises(RuntimeError, match="input changed during fitting"):
        feems_benchmark.run_feems_fold(Namespace(input=path, fold=0, output_npz=output))
    assert output.read_bytes() == b"previous fold"
