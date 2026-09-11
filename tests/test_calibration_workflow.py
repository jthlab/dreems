"""Exercise calibration without training a new model or fitting FEEMS."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from dreems.inference.calibration import SDCalibration
from dreems.neural import checkpoints
from dreems.utils.workflow_io import sha256, write_pickle

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "src/dreems/data/dreems_diffusion_bands.dreems.zip"


@pytest.fixture(scope="module")
def gate_worker():
    spec = importlib.util.spec_from_file_location(
        "gate_calibration_worker", ROOT / "workflow/calibrate_surface_gate.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_workflow(directory: Path, config: dict, *options: str) -> subprocess.CompletedProcess:
    config_path = directory / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    result = subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "--no-sync",
            "--project",
            str(ROOT),
            "snakemake",
            "all",
            "-s",
            str(ROOT / "workflow/calibration.Snakefile"),
            "--configfile",
            str(config_path),
            "--profile",
            str(ROOT / "workflow/profiles/local"),
            "--directory",
            str(directory),
            "--cores",
            "2",
            "--resources",
            "mem_mb=8000",
            "gpu=0",
            "--notemp",
            *options,
        ],
        cwd=ROOT,
        env={**os.environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=600,
    )
    (directory / f"run_{len(list(directory.glob('run_*.log')))}.log").write_text(
        result.stdout + result.stderr
    )
    return result


def config_for(directory: Path) -> dict:
    return {
        "outdir": str(directory / "output"),
        "checkpoint": str(WEIGHTS),
        "raw_source_dir": str(directory / "raw"),
        "raw_source_layout": "packs",
        "raw_pack_size": 3,
        "fit_start": 0,
        "fit_stop": 2,
        "selection_start": 2,
        "selection_stop": 4,
        "verification_start": 4,
        "verification_stop": 6,
    }


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"selection_start": 1}, "overlap"),
        ({"verification_start": 3}, "overlap"),
        ({"fit_start": -1}, "nonnegative"),
        ({"fit_stop": 0}, "nonempty"),
        ({"fit_start": 0.5}, "integers"),
        ({"fit_start": True}, "integers"),
        ({"raw_pack_size": 1.5}, "raw_pack_size"),
    ],
)
def test_invalid_splits_fail_before_scheduling(tmp_path, changes, message) -> None:
    config = config_for(tmp_path)
    config.update(changes)
    result = run_workflow(tmp_path, config, "--dry-run")
    assert result.returncode != 0
    assert message in result.stdout + result.stderr
    assert not (tmp_path / "output").exists()


def test_calibration_dag_accepts_current_checkpoint_and_reuses_outputs(tmp_path) -> None:
    rng = np.random.default_rng(91)
    positions = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    edges = np.asarray([[0, 1], [0, 2], [1, 3], [2, 3], [0, 3]], dtype=np.int32)
    examples = [
        {
            "graph": {"active_full": np.arange(4), "edges": edges},
            "active_positions": positions,
            "sample_nodes": np.repeat(np.arange(4), 2),
            "genotypes": rng.integers(0, 3, (8, 40)),
            "edge_log_rates": rng.normal(size=5),
        }
        for _ in range(6)
    ]
    for pack in range(2):
        write_pickle(tmp_path / f"raw/test/pack_{pack}.pkl", examples[3 * pack : 3 * pack + 3])
    config = config_for(tmp_path)
    result = run_workflow(tmp_path, config)
    assert result.returncode == 0, (result.stdout + result.stderr)[-15000:]
    output = tmp_path / "output"
    assert (output / "checkpoint.canonical.dreems.zip").read_bytes() == WEIGHTS.read_bytes()
    with (
        zipfile.ZipFile(WEIGHTS) as original,
        zipfile.ZipFile(output / "checkpoint.calibrated.dreems.zip") as fitted,
    ):
        assert original.read("params.eqx") == fitted.read("params.eqx")
    report = json.loads((output / "verification/report.json").read_text())
    assert report["verification_examples"] == 2
    assert all(row["graphs"] == 2 for row in report["rows"])
    for role, indices in (("fit", (0, 1)), ("selection", (2, 3)), ("verification", (4, 5))):
        for index in indices:
            with np.load(output / f"cases/{role}/{index}.npz") as case:
                assert case["role"].item() == role
                assert case["index"].item() == index
    planned = run_workflow(tmp_path, config, "--dry-run")
    assert planned.returncode == 0
    assert "Nothing to be done" in planned.stdout + planned.stderr


@pytest.mark.parametrize("offsets", [[0, 1], [1, 2, 3], [0, 2, 2], [0, 2, 4], [0.0, 1.5, 3.0]])
def test_invalid_prediction_pack_offsets_fail(tmp_path, gate_worker, offsets) -> None:
    path = tmp_path / "pack.npz"
    np.savez(
        path,
        replicate=np.asarray([0, 1]),
        edge_offsets=np.asarray(offsets),
        edge_truth=np.ones(3),
        edge_mean=np.ones(3),
        edge_sd=np.ones(3),
    )
    with pytest.raises(ValueError, match="offsets"):
        gate_worker.load_predictions([path])


def test_cached_calibration_undoes_source_scale(tmp_path, gate_worker) -> None:
    source = tmp_path / "source.dreems.zip"
    checkpoints.attach_edge_sd_calibration(
        WEIGHTS, source, SDCalibration(method="global", global_scale=2.0)
    )
    rows = tmp_path / "rows.csv"
    rows.write_text(
        "example,family,sampled_node_fraction\n" + "".join(f"{i},barrier,0.2\n" for i in range(9))
    )
    predictions = [tmp_path / f"example_{i}.npz" for i in range(9)]
    for path in predictions:
        np.savez(
            path, truth=np.asarray([-2.0, 0.0, 2.0]), dreems=np.zeros(3), edge_sd=np.full(3, 2.0)
        )
    gate_worker.calibrate(
        argparse.Namespace(
            checkpoint=source,
            predictions=predictions,
            included_rows=rows,
            seed=5,
            fit_output=tmp_path / "fit.json",
            selection_output=tmp_path / "selection.json",
            verification_output=tmp_path / "verification.json",
            verification_csv=tmp_path / "verification.csv",
            output_checkpoint=tmp_path / "calibrated.dreems.zip",
        )
    )
    selected = json.loads((tmp_path / "selection.json").read_text())
    assert selected["calibration"]["global_scale"] == pytest.approx(np.sqrt(8 / 3))
    splits = [
        json.loads((tmp_path / f"{name}.json").read_text())["split_examples"]
        for name in ("fit", "selection", "verification")
    ]
    assert sorted(index for split in splits for index in split) == list(range(9))
    assert all(len(split) == 3 for split in splits)


@pytest.mark.parametrize("hash_kind", ["missing", "wrong", "nonscalar"])
def test_strict_calibration_rejects_unverified_predictions(
    tmp_path, gate_worker, hash_kind
) -> None:
    path = tmp_path / "example_0.npz"
    contents = {"truth": np.ones(3), "dreems": np.zeros(3), "edge_sd": np.ones(3)}
    if hash_kind != "missing":
        contents["checkpoint_sha256"] = np.asarray(
            "0" * 64 if hash_kind == "wrong" else [sha256(WEIGHTS)]
        )
    np.savez(path, **contents)
    with pytest.raises(ValueError, match="checkpoint"):
        gate_worker.calibrate(
            argparse.Namespace(checkpoint=WEIGHTS, predictions=[path], require_checkpoint_hash=True)
        )


@pytest.mark.parametrize("key", ["truth", "dreems", "edge_sd"])
def test_complex_calibration_arrays_cannot_lose_their_imaginary_part(
    tmp_path, gate_worker, key
) -> None:
    values = {"truth": np.ones(3), "dreems": np.zeros(3), "edge_sd": np.ones(3)}
    values[key] = values[key].astype(complex) + 1j
    path = tmp_path / "example_0.npz"
    np.savez(path, **values)
    with pytest.raises(ValueError, match="real numbers"):
        gate_worker.load_predictions([path])
