"""Prepared-pack contracts and identity preservation in training workers."""

from __future__ import annotations

import copy
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from jaxtyping import TypeCheckError

from dreems import pipeline, workflow_io

STATS = {
    "node_mean": [0.0, 0.0],
    "node_std": [1.0, 1.0],
    "edge_mean": [0.0, 0.0, 0.0],
    "edge_std": [1.0, 1.0, 1.0],
    "y_scale": 1.0,
    "q_y_scale": 1.0,
}
DIMS = {"max_nodes": 6, "max_edges": 5}


def prepared(nodes: int = 3, marker: float = 1.0) -> dict[str, np.ndarray]:
    return {
        "node_x": np.full((nodes, 2), marker, dtype=np.float32),
        "edge_x": np.full((nodes - 1, 3), marker, dtype=np.float32),
        "edges": np.column_stack([np.arange(nodes - 1), np.arange(1, nodes)]).astype(np.int32),
        "y": np.arange(nodes - 1, dtype=np.float32) * marker,
        "q": np.arange(nodes, dtype=np.float32) * marker,
    }


def write_pack(tmp_path: Path, items: list[dict], name: str = "pack.pkl") -> Path:
    path = tmp_path / name
    workflow_io.write_pickle(path, items)
    return path


def stream(paths: list[Path], *, batch_size: int = 4, steps: int = 1):
    return iter(
        pipeline.iter_training_batches(
            paths, STATS, dims=DIMS, batch_size=batch_size, steps=steps, seed=7
        )
    )


@pytest.mark.parametrize("max_examples", [0, -1, True])
def test_prepared_limit_is_validated(tmp_path: Path, max_examples) -> None:
    path = write_pack(tmp_path, [prepared()])
    with pytest.raises(ValueError):
        pipeline.load_prepared_shards([path], max_examples=max_examples)


@pytest.mark.parametrize("payload", [{}, [0], [prepared(), None]])
def test_prepared_readers_reject_malformed_packs(tmp_path: Path, payload) -> None:
    path = write_pack(tmp_path, payload)
    for read in (
        lambda: pipeline.load_prepared_shards([path]),
        lambda: pipeline.normalized_shard(path, STATS),
        lambda: pipeline.normalization_accumulator_from_prepared([path]),
    ):
        with pytest.raises(ValueError, match="list of dictionaries"):
            read()


@pytest.mark.parametrize("field", ["node_x", "edge_x"])
def test_width_mismatch_cannot_broadcast_into_statistics(tmp_path: Path, field: str) -> None:
    second = prepared()
    second[field] = second[field][:, :1]
    items = [prepared(), second]
    with pytest.raises(ValueError, match="widths"):
        pipeline.prepared_dimensions(items)
    path = write_pack(tmp_path, items)
    with pytest.raises(ValueError, match="widths"):
        pipeline.normalization_accumulator_from_prepared([path])


@pytest.mark.parametrize("value", [0, -1, True, 1.5, float("nan"), [2]])
def test_dimension_metadata_does_not_truncate_counts(tmp_path: Path, value) -> None:
    path = tmp_path / "pack.pkl"
    dimensions = pipeline.prepared_dimensions([prepared()])
    dimensions["examples"] = value
    workflow_io.write_json(path.with_suffix(".metadata.json"), {"prepared_dimensions": dimensions})
    with pytest.raises(ValueError):
        pipeline.prepared_dimensions_from_metadata(path)


@pytest.fixture
def moments(tmp_path: Path) -> dict:
    return pipeline.normalization_accumulator_from_prepared([write_pack(tmp_path, [prepared()])])


@pytest.mark.parametrize(
    "field,value",
    [
        ("node_count", 1.5),
        ("y_count", 0),
        ("q_count", True),
        ("node_sum", [1.0]),
        ("edge_sumsq", [1.0]),
        ("y_sum", [1.0]),
        ("q_sumsq", -1.0),
        ("edge_sum", [1.0, float("nan"), 1.0]),
    ],
)
def test_invalid_moments_fail_before_normalization(moments: dict, field: str, value) -> None:
    bad = copy.deepcopy(moments)
    bad[field] = value
    for consume in (
        lambda: pipeline.merge_normalization_accumulators([moments, bad]),
        lambda: pipeline.finalize_normalization_stats(bad, std_floor=0.2),
    ):
        with pytest.raises(ValueError):
            consume()


@pytest.mark.parametrize("floor", [-1.0, float("nan"), float("inf")])
def test_invalid_normalization_floor_is_rejected(moments: dict, floor: float) -> None:
    with pytest.raises(ValueError, match="std_floor"):
        pipeline.finalize_normalization_stats(moments, std_floor=floor)


def test_moments_remain_row_weighted_and_uncentered(tmp_path: Path) -> None:
    a, b = prepared(2, 1.0), prepared(6, 3.0)
    result = pipeline.finalize_normalization_stats(
        pipeline.normalization_accumulator_from_prepared([write_pack(tmp_path, [a, b])]),
        std_floor=0.2,
    )
    assert result["node_count"] == 8
    np.testing.assert_array_equal(result["node_mean"], [2.5, 2.5])
    y = np.concatenate([a["y"], b["y"]]).astype(np.float64)
    assert result["y_scale"] == pytest.approx(y.std() + 1e-6)


def test_streaming_rejects_empty_inputs_and_all_empty_packs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no training shards"):
        next(stream([]))
    empty = write_pack(tmp_path, [])
    with pytest.raises(ValueError, match="all training shards are empty"):
        next(stream([empty]))
    assert list(stream([], steps=0)) == []


@pytest.mark.parametrize("size", [0, -1, True])
def test_invalid_streaming_batch_size_fails(tmp_path: Path, size) -> None:
    with pytest.raises(ValueError):
        next(stream([write_pack(tmp_path, [prepared()])], batch_size=size))


def test_mixed_counterfactual_group_sizes_cannot_change_batch_shape(tmp_path: Path) -> None:
    items = []
    for group, size in enumerate((2, 4)):
        for index in range(size):
            item = prepared(marker=float(index))
            item["counterfactual_group"] = np.asarray(group, dtype=np.int32)
            item["counterfactual_anchor"] = np.asarray(index == 0, dtype=np.float32)
            items.append(item)
    with pytest.raises(ValueError, match="same size"):
        next(stream([write_pack(tmp_path, items)], batch_size=8))


def test_runtime_types_and_coverage_shapes_are_enforced(tmp_path: Path) -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        pipeline.load_examples(str(tmp_path / "pack.pkl"))
    with pytest.raises(TypeCheckError):
        pipeline.edge_coverage(np.zeros(3), np.zeros(3), np.ones(1), 1.0)


def test_multi_shard_evaluation_keeps_unique_ids_and_prediction_alignment(
    tmp_path: Path, monkeypatch
) -> None:
    items = [prepared(n, float(i + 1)) for i, n in enumerate((5, 2, 4, 3, 6))]
    paths = [write_pack(tmp_path, items[:3], "a.pkl"), write_pack(tmp_path, items[3:], "b.pkl")]
    loaded = SimpleNamespace(
        params=None,
        normalization=STATS,
        model={},
        edge_sd_calibration=SimpleNamespace(apply=lambda sd, **_: sd),
    )
    monkeypatch.setattr(pipeline.checkpoints, "load_checkpoint", lambda _: loaded)
    monkeypatch.setattr(
        pipeline.dreems_model,
        "predict",
        lambda _, arrays, **kw: (
            arrays["y"],
            np.zeros_like(arrays["y"]),
            arrays["q"],
            np.zeros_like(arrays["q"]),
        ),
    )
    output = tmp_path / "evaluation"
    prediction = output / "predictions.npz"
    pipeline.command_evaluate(
        Namespace(
            inputs=paths,
            input_manifest=None,
            checkpoint=tmp_path / "fixture.zip",
            batch_size=2,
            replicate_offset=13,
            max_examples=4,
            output_dir=output,
            predictions_output=prediction,
        )
    )
    rows = json.loads((output / "summary.json").read_text())["rows"]
    assert [row["replicate"] for row in rows] == [13, 14, 15, 16]
    assert [row["replicate_within_shard"] for row in rows] == [0, 1, 2, 0]
    with np.load(prediction) as data:
        assert data["replicate"].tolist() == [14, 15, 13, 16]
        for index, replicate in enumerate(data["replicate"]):
            start, stop = data["edge_offsets"][index : index + 2]
            expected = items[replicate - 13]["y"]
            expected = expected - expected.mean()
            np.testing.assert_array_equal(data["edge_truth"][start:stop], expected)
            np.testing.assert_array_equal(data["edge_mean"][start:stop], expected)


def test_csv_failure_preserves_previous_file(tmp_path: Path) -> None:
    class BrokenValue:
        def __str__(self):
            raise ValueError("bad value")

    path = tmp_path / "report.csv"
    path.write_text("existing report")
    with pytest.raises(ValueError, match="bad value"):
        workflow_io.write_csv(path, [{"x": 1}, {"x": BrokenValue()}])
    assert path.read_text() == "existing report"
