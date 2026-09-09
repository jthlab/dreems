"""Corpus seeds and file failures must not silently change a training run."""

from __future__ import annotations

import json
import pickle
import stat
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dreems import corpus, simulation_config, workflow_io


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, 12001),
        ({"simulation": None, "seed": None}, 12001),
        ({"seed": 29}, 29),
        ({"simulation": {"seed": None}, "seed": 29}, 29),
        ({"simulation": {"seed": 17}, "seed": 29}, 17),
        ({"simulation": {"seed": 0}, "seed": 29}, 0),
    ],
)
def test_corpus_seed_is_shared_by_generation_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: dict[str, Any], expected: int
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    original = config_path.read_bytes()
    received = []

    def simulate(payload):
        received.append(payload)
        return {
            "graph": {"edges": np.array([[0, 1]], dtype=np.int32)},
            "active_positions": np.array([[0.0, 0.0], [1.0, 0.0]]),
            "sample_nodes": np.array([0, 1], dtype=np.int32),
            "sample_covariance": np.array([[1.0, -1.0], [-1.0, 1.0]]),
            "effective_snps": 10,
        }

    monkeypatch.setattr(corpus.simulation, "simulate_raw_example_from_seed", simulate)
    raw = tmp_path / "example.pkl"
    corpus.generate_example(config_path=config_path, split="test", example_id=3, output=raw)
    assert received[0][0] == expected + 10_000_000 + 1009 * 3
    assert received[0][1]["seed"] == expected
    packed = tmp_path / "pack.pkl"
    metadata = tmp_path / "metadata.json"
    corpus.pack_examples(
        inputs=[raw],
        output=packed,
        metadata=metadata,
        config_path=config_path,
        split="test",
        pack_index=0,
        start_index=3,
    )
    assert json.loads(metadata.read_text())["base_seed"] == expected
    assert config_path.read_bytes() == original
    assert len(workflow_io.read_pickle(packed)) == 1


@pytest.mark.parametrize("seed", [-1, 1.5, "17", True, np.nan, np.inf])
def test_invalid_corpus_seed_is_not_cast(seed: Any) -> None:
    with pytest.raises(ValueError, match="seed must be a nonnegative integer"):
        simulation_config.simulation_config({"simulation": {"seed": seed}})


def test_numpy_seed_is_supported_without_mutating_input() -> None:
    config = {"simulation": {"seed": np.uint64(123)}}
    result = simulation_config.simulation_config(config)
    assert result["seed"] == 123
    assert type(result["seed"]) is int
    assert isinstance(config["simulation"]["seed"], np.uint64)


@pytest.mark.parametrize("bound", [10.5, "10", True, np.inf])
def test_snp_bin_bounds_are_not_truncated(bound: Any) -> None:
    with pytest.raises(ValueError, match="SNP bins must contain integer bounds"):
        simulation_config.simulation_config({"simulation": {"num_snps_log_bins": [[bound, 20]]}})


@pytest.mark.parametrize(
    "key",
    [
        "sampling_association_min_strength",
        "sampling_association_max_strength",
        "sampling_spatial_noise_scale",
        "sampling_spatial_length_scale",
    ],
)
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_sampling_scales_must_be_finite(key: str, value: float) -> None:
    with pytest.raises(ValueError, match="sampling"):
        simulation_config.simulation_config({"simulation": {key: value}})


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
@pytest.mark.parametrize("contents", ["[]", "0", "false", '""'])
def test_falsy_nonmapping_config_is_not_treated_as_empty(
    tmp_path: Path, suffix: str, contents: str
) -> None:
    path = tmp_path / f"config{suffix}"
    path.write_text(contents)
    with pytest.raises(ValueError, match="expected a config mapping"):
        workflow_io.load_config(path)


@pytest.mark.parametrize("suffix,contents", [(".json", "null"), (".yaml", ""), (".yaml", "null")])
def test_empty_config_stays_supported(tmp_path: Path, suffix: str, contents: str) -> None:
    path = tmp_path / f"config{suffix}"
    path.write_text(contents)
    assert workflow_io.load_config(path) == {}


def test_yaml_config_requires_string_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("42: value\n")
    with pytest.raises(ValueError, match="expected string config keys"):
        workflow_io.load_config(path)


@pytest.mark.parametrize("existing", [False, True])
def test_interrupted_pickle_write_leaves_previous_pack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    path = tmp_path / "pack.pkl"
    if existing:
        workflow_io.write_pickle(path, ["previous"])
    before = path.read_bytes() if existing else None

    def broken_dump(payload, handle, *, protocol):
        handle.write(b"partial pickle")
        assert (path.read_bytes() if path.exists() else None) == before
        raise OSError("simulated write failure")

    monkeypatch.setattr(workflow_io.pickle, "dump", broken_dump)
    with pytest.raises(OSError, match="simulated write failure"):
        workflow_io.write_pickle(path, ["replacement"])
    assert (path.read_bytes() if path.exists() else None) == before
    assert set(tmp_path.iterdir()) == ({path} if existing else set())


@pytest.mark.parametrize("kind", ["pickle", "json"])
def test_failed_output_rename_leaves_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    writer = workflow_io.write_pickle if kind == "pickle" else workflow_io.write_json
    path = tmp_path / f"output.{kind}"
    writer(path, {"value": "previous"})
    before = path.read_bytes()

    def fail_replace(source, destination):
        assert destination == path
        raise OSError("simulated rename failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated rename failure"):
        writer(path, {"value": "new"})
    assert path.read_bytes() == before
    assert set(tmp_path.iterdir()) == {path}


def test_failed_json_conversion_preserves_existing_report(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    workflow_io.write_json(path, {"value": "previous"})
    before = path.read_bytes()
    with pytest.raises(TypeError):
        workflow_io.write_json(path, {"value": object()})
    assert path.read_bytes() == before
    assert set(tmp_path.iterdir()) == {path}


def test_writers_preserve_format_and_array_dtypes(tmp_path: Path) -> None:
    payload = {"z": np.array([1.0, 2.0], dtype=np.float32), "a": np.int64(5)}
    pack = tmp_path / "pack.pkl"
    report = tmp_path / "report.json"
    workflow_io.write_pickle(pack, payload)
    workflow_io.write_json(report, payload)
    assert pack.read_bytes() == pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    assert (
        report.read_text() == json.dumps({"a": 5, "z": [1.0, 2.0]}, indent=2, sort_keys=True) + "\n"
    )
    assert workflow_io.read_pickle(pack)["z"].dtype == np.float32
    assert set(tmp_path.iterdir()) == {pack, report}


@pytest.mark.parametrize("kind", ["pickle", "json"])
@pytest.mark.parametrize("mode", [0o600, 0o640, 0o664])
def test_atomic_writers_preserve_existing_file_permissions(
    tmp_path: Path, kind: str, mode: int
) -> None:
    writer = workflow_io.write_pickle if kind == "pickle" else workflow_io.write_json
    path = tmp_path / f"output.{kind}"
    writer(path, {"old": 1})
    path.chmod(mode)
    writer(path, {"new": 2})
    assert stat.S_IMODE(path.stat().st_mode) == mode


def test_invalid_pack_config_is_rejected_before_writing(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"simulation":{"seed":-1}}')
    output = tmp_path / "pack.pkl"
    workflow_io.write_pickle(output, ["previous"])
    before = output.read_bytes()
    with pytest.raises(ValueError, match="seed"):
        corpus.pack_examples(
            inputs=[tmp_path / "unused.pkl"],
            output=output,
            metadata=tmp_path / "metadata.json",
            config_path=config,
            split="train",
            pack_index=0,
            start_index=0,
        )
    assert output.read_bytes() == before
