"""Promotion must verify the candidate without changing its Equinox bytes."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pytest

from dreems.inference.calibration import SDCalibration
from dreems.neural import checkpoints, model
from dreems.utils.workflow_io import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def promoter():
    spec = importlib.util.spec_from_file_location(
        "promotion_worker", ROOT / "workflow/promote_surface_gate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def core_model():
    return model.DREEMS(
        jax.random.PRNGKey(19),
        node_dim=3,
        edge_dim=4,
        hidden_channels=4,
        layers=1,
        context_layers=1,
        initializer_gate=True,
    )


@pytest.fixture
def release_inputs(tmp_path, core_model):
    """A core, a frozen-core gate, and all reports needed for a tiny release."""
    record = {
        "step": 3,
        "validation_loss": -0.2,
        "validation_best": True,
        "validation_checks_since_best": 0,
    }
    run = {
        "best_step": 3,
        "best_validation_loss": -0.2,
        "early_stopping_metric": "validation_loss",
        "best_validation_metric": -0.2,
        "best_validation_metrics": record,
        "restored_best_validation": False,
        "history": [record],
        "final_step": 3,
    }
    architecture = {
        "kind": "diffusion_band_initializer_gate_v2",
        "hidden_channels": 4,
        "layers": 1,
        "context_layers": 1,
        "initializer_gate": True,
        "surface_gate": False,
    }
    metadata = {
        "format": "dreems-equinox-v1",
        "normalization": {"y_scale": 1.0},
        "model": {
            "schema_version": 2,
            "node_dim": 3,
            "edge_dim": 4,
            "architecture": architecture,
            "features": {"kind": "topology_diffusion_bands_v2", "covariance_components": 3},
            "edge_conditioner": {"method": "all_pair_diagonal_adjoint"},
        },
        "training_run": run,
        "training_config": {"optimizer": {"trainable_scope": "surface_gate"}},
    }
    gate = model.DREEMS(
        jax.random.PRNGKey(19),
        node_dim=3,
        edge_dim=4,
        hidden_channels=4,
        layers=1,
        context_layers=1,
        initializer_gate=True,
        surface_gate=True,
    )
    arguments = {}
    for key, params in (("core", core_model), ("surface_gate", gate)):
        if key == "surface_gate":
            metadata["model"]["architecture"]["kind"] = "diffusion_band_surface_gate_v3"
            metadata["model"]["architecture"]["surface_gate"] = True
        buffer = io.BytesIO()
        eqx.tree_serialise_leaves(buffer, params)
        arguments[key] = tmp_path / f"{key}.dreems.zip"
        checkpoints._write_checkpoint_archive(
            arguments[key], {"params.eqx": buffer.getvalue()}, metadata
        )
        arguments[f"{key}_history"] = tmp_path / f"{key}_history.jsonl"
        history = [{"step": 1, "validation_loss": 0.3}, record, {"step": 4, "validation_loss": 0.1}]
        arguments[f"{key}_history"].write_text("".join(json.dumps(row) + "\n" for row in history))
        arguments[f"{key}_summary"] = tmp_path / f"{key}_summary.json"
        write_json(
            arguments[f"{key}_summary"],
            {
                "training_run": {
                    **run,
                    "history": history,
                    "final_step": 4,
                    "restored_best_validation": True,
                },
                "losses": [0.3, -0.2, 0.1],
            },
        )
    arguments["calibrated_surface_gate"] = tmp_path / "calibrated.dreems.zip"
    calibration = SDCalibration(global_scale=1.2, method="global")
    checkpoints.attach_edge_sd_calibration(
        arguments["surface_gate"], arguments["calibrated_surface_gate"], calibration
    )
    for label in ("broad", "hard"):
        arguments[f"{label}_protocol"] = tmp_path / f"{label}_protocol.json"
        arguments[f"{label}_report"] = tmp_path / f"{label}_report.json"
        protocol = {"dreems": {"checkpoint_sha256": sha256(arguments["surface_gate"])}}
        write_json(arguments[f"{label}_protocol"], protocol)
        write_json(arguments[f"{label}_report"], {"protocol": protocol, "examples": 6})
        arguments[f"expected_{label}_examples"] = 6
    fitted = calibration.to_dict()
    reports = {
        "fit": {"candidates": {"global": {"calibration": fitted}}, "split_examples": [0, 1]},
        "selection": {
            "selected": "global",
            "calibration": fitted,
            "split_examples": [2, 3],
            "candidate_metrics": {"global": {"graph_balanced_nll": -0.2}},
        },
        "verification": {"selected": "global", "calibration": fitted, "split_examples": [4, 5]},
    }
    for stage, report in reports.items():
        arguments[f"calibration_{stage}"] = tmp_path / f"{stage}.json"
        write_json(arguments[f"calibration_{stage}"], report)
    config = tmp_path / "training.yaml"
    config.write_text("seed: 19\n")
    arguments["training_configs"] = [config]
    arguments["output"] = tmp_path / "release.dreems.zip"
    return arguments


def test_promotion_is_deterministic_and_copies_parameter_bytes(release_inputs, promoter) -> None:
    promoter.promote(**release_inputs)
    output = release_inputs["output"]
    before = output.read_bytes()
    promoter.promote(**release_inputs)
    assert output.read_bytes() == before
    loaded = checkpoints.load_checkpoint(output)
    assert loaded.payload["provenance"]["frozen_core_verified"] is True
    assert loaded.payload["training_run"]["final_step"] == 4
    assert loaded.edge_sd_calibration.global_scale == 1.2
    with (
        zipfile.ZipFile(output) as result,
        zipfile.ZipFile(release_inputs["surface_gate"]) as source,
    ):
        assert result.read("params.eqx") == source.read("params.eqx")
        assert result.namelist() == ["metadata.json", "params.eqx"]
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in result.infolist())


def test_calibration_hash_must_name_the_candidate(release_inputs, promoter) -> None:
    path = release_inputs["calibration_fit"]
    report = json.loads(path.read_text())
    report["checkpoint_sha256"] = "another checkpoint"
    write_json(path, report)
    with pytest.raises(ValueError, match="calibration reports"):
        promoter.promote(**release_inputs)


@pytest.mark.parametrize("part", ["hash", "protocol", "examples"])
def test_promotion_rejects_a_wrong_benchmark(release_inputs, promoter, part) -> None:
    path = release_inputs["broad_report"]
    report = json.loads(path.read_text())
    if part == "hash":
        report["protocol"]["dreems"]["checkpoint_sha256"] = "another checkpoint"
        write_json(release_inputs["broad_protocol"], report["protocol"])
    elif part == "protocol":
        report["protocol"]["changed"] = True
    else:
        report["examples"] = 5
    write_json(path, report)
    with pytest.raises(ValueError, match="broad H2H"):
        promoter.promote(**release_inputs)
    assert not release_inputs["output"].exists()


@pytest.mark.parametrize("indices", [[0, 0, 1], [0, True], [0, 1.0], [0, -1], [], [0, 2]])
def test_invalid_calibration_split_cannot_be_promoted(release_inputs, promoter, indices) -> None:
    path = release_inputs["calibration_fit"]
    report = json.loads(path.read_text())
    report["split_examples"] = indices
    write_json(path, report)
    with pytest.raises(ValueError, match="SD calibration"):
        promoter.promote(**release_inputs)


@pytest.mark.parametrize("stage", ["fit", "selection", "verification"])
def test_calibration_must_match_each_report(release_inputs, promoter, stage) -> None:
    path = release_inputs[f"calibration_{stage}"]
    report = json.loads(path.read_text())
    calibration = (
        report["candidates"]["global"]["calibration"] if stage == "fit" else report["calibration"]
    )
    calibration["global_scale"] = 8.0
    write_json(path, report)
    with pytest.raises(ValueError, match="calibration|verification"):
        promoter.promote(**release_inputs)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), None, True])
def test_nonfinite_or_missing_selection_score_fails(release_inputs, promoter, score) -> None:
    path = release_inputs["calibration_selection"]
    report = json.loads(path.read_text())
    report["candidate_metrics"]["global"]["graph_balanced_nll"] = score
    write_json(path, report)
    with pytest.raises(ValueError, match="NLL"):
        promoter.promote(**release_inputs)


def test_chosen_calibration_must_minimize_selection_loss(release_inputs, promoter) -> None:
    path = release_inputs["calibration_fit"]
    fit = json.loads(path.read_text())
    fit["candidates"]["raw"] = {"calibration": SDCalibration().to_dict()}
    write_json(path, fit)
    path = release_inputs["calibration_selection"]
    selection = json.loads(path.read_text())
    selection["candidate_metrics"]["raw"] = {"graph_balanced_nll": -0.5}
    write_json(path, selection)
    with pytest.raises(ValueError, match="minimize"):
        promoter.promote(**release_inputs)


@pytest.mark.parametrize(
    "change", ["missing", "nonfinite", "different", "unrestored", "history", "metric"]
)
def test_training_records_are_checked(release_inputs, promoter, change) -> None:
    path = release_inputs["core_summary"]
    summary = json.loads(path.read_text())
    if change == "missing":
        summary["training_run"]["history"] = []
    elif change == "history":
        release_inputs["core_history"].write_text("[]\n")
    elif change == "unrestored":
        summary["training_run"]["restored_best_validation"] = "true"
    elif change == "metric":
        summary["training_run"]["early_stopping_metric"] = "validation_correlation"
    else:
        summary["training_run"]["best_validation_loss"] = (
            float("nan") if change == "nonfinite" else -0.1
        )
    write_json(path, summary)
    with pytest.raises(ValueError):
        promoter.promote(**release_inputs)


def test_frozen_core_comparison_checks_static_tree_structure(core_model, promoter) -> None:
    left = checkpoints.DREEMSCheckpoint(Path("left"), core_model, {}, {})
    changed = eqx.tree_at(lambda value: value.self_layers, core_model, list(core_model.self_layers))
    right = replace(left, path=Path("right"), params=changed)
    with pytest.raises(ValueError, match="tree structure"):
        promoter.assert_equal_model_fields(left, right, excluded=set())


def test_training_verification_uses_the_configured_stopping_metric(
    release_inputs, promoter
) -> None:
    summary_path = release_inputs["core_summary"]
    summary = json.loads(summary_path.read_text())
    run = summary["training_run"]
    run["early_stopping_metric"] = "validation_edge_nll"
    for record in run["history"]:
        record["validation_edge_nll"] = record["validation_loss"]
        record["validation_loss"] += 3.0
    run["best_validation_metrics"] = run["history"][1]
    write_json(summary_path, summary)
    release_inputs["core_history"].write_text(
        "".join(json.dumps(row) + "\n" for row in run["history"])
    )
    _, metadata = checkpoints._read_checkpoint_archive(release_inputs["core"])
    metadata["training_run"] = copy.deepcopy(run)
    # Final checkpoints omit the two bookkeeping fields present in best snapshots.
    metadata["training_run"]["best_validation_metrics"].pop("validation_best")
    metadata["training_run"]["best_validation_metrics"].pop("validation_checks_since_best")
    result = promoter.verified_training_stage(
        checkpoint_metadata=metadata,
        summary_path=summary_path,
        history_path=release_inputs["core_history"],
        name="core",
    )
    assert result["training_run"]["best_validation_loss"] == -0.2


def test_matching_loss_alone_is_not_enough_to_verify_a_training_run(
    release_inputs, promoter
) -> None:
    _, metadata = checkpoints._read_checkpoint_archive(release_inputs["core"])
    metadata["training_run"]["best_validation_metrics"]["validation_edge_nll"] = -0.9
    with pytest.raises(ValueError, match="best metrics differ"):
        promoter.verified_training_stage(
            checkpoint_metadata=metadata,
            summary_path=release_inputs["core_summary"],
            history_path=release_inputs["core_history"],
            name="core",
        )


def test_frozen_core_comparison_checks_parameter_values(core_model, promoter) -> None:
    left = checkpoints.DREEMSCheckpoint(Path("left"), core_model, {}, {})
    changed = eqx.tree_at(lambda value: value.input.weight, core_model, core_model.input.weight + 1)
    with pytest.raises(ValueError, match="changed"):
        promoter.assert_equal_model_fields(left, replace(left, params=changed), excluded=set())


def test_changed_input_during_promotion_preserves_existing_output(
    release_inputs, promoter, monkeypatch
) -> None:
    output = release_inputs["output"]
    output.write_bytes(b"previous release")
    original = promoter.verify_calibration_reports

    def change_input(**kwargs):
        sizes = original(**kwargs)
        release_inputs["training_configs"][0].write_text("seed: 20\n")
        return sizes

    monkeypatch.setattr(promoter, "verify_calibration_reports", change_input)
    with pytest.raises(ValueError, match="changed during verification"):
        promoter.promote(**release_inputs)
    assert output.read_bytes() == b"previous release"
    assert not list(output.parent.glob(f".{output.name}.*"))


def test_promotion_cannot_overwrite_an_input(release_inputs, promoter) -> None:
    release_inputs["output"] = release_inputs["core"]
    before = sha256(release_inputs["core"])
    with pytest.raises(ValueError, match="overwrite"):
        promoter.promote(**release_inputs)
    assert sha256(release_inputs["core"]) == before


def test_duplicate_archive_members_are_rejected(tmp_path) -> None:
    path = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("metadata.json", "{}")
        archive.writestr("params.eqx", b"weights")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("metadata.json", "{}")
    with pytest.raises(ValueError, match="duplicate"):
        checkpoints._read_checkpoint_archive(path)


def test_atomic_archive_retains_extra_members_and_existing_permissions(tmp_path) -> None:
    path = tmp_path / "checkpoint.zip"
    path.write_bytes(b"previous")
    path.chmod(0o640)
    members = {"params.eqx": b"saved bytes", "notes.txt": b"notes"}
    checkpoints._write_checkpoint_archive(path, members, {"format": "example"})
    saved, payload = checkpoints._read_checkpoint_archive(path)
    assert saved["params.eqx"] == members["params.eqx"]
    assert saved["notes.txt"] == b"notes"
    assert payload == {"format": "example"}
    assert path.stat().st_mode & 0o777 == 0o640


def test_failed_archive_write_leaves_original_file(tmp_path, monkeypatch) -> None:
    path = tmp_path / "checkpoint.zip"
    path.write_bytes(b"previous")

    def fail(*args, **kwargs):
        raise OSError("test write failure")

    monkeypatch.setattr(zipfile.ZipFile, "writestr", fail)
    with pytest.raises(OSError, match="test write failure"):
        checkpoints._write_checkpoint_archive(path, {"params.eqx": b"weights"}, {})
    assert path.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".checkpoint.zip.*"))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("node_dim", 2.5),
        ("edge_dim", True),
        ("hidden_channels", "4"),
        ("layers", -1),
        ("context_layers", float("inf")),
        ("initializer_gate", "false"),
        ("surface_gate", 1),
    ],
)
def test_metadata_cannot_silently_change_architecture(release_inputs, name, value) -> None:
    _, metadata = checkpoints._read_checkpoint_archive(release_inputs["core"])
    model_metadata = copy.deepcopy(metadata["model"])
    target = model_metadata if name in {"node_dim", "edge_dim"} else model_metadata["architecture"]
    target[name] = value
    with pytest.raises(ValueError, match=name):
        checkpoints.model_from_metadata(model_metadata)


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("node_sample_count", [-1, 3]),
        ("node_sample_count", [0.5, 2]),
        ("node_sample_count", [float("nan"), 2]),
        ("sample_nodes", [0, 2]),
        ("sample_nodes", [0, -1]),
        ("sample_nodes", [0.5]),
        ("sample_nodes", [[0, 1]]),
    ],
)
def test_invalid_sample_assignment_cannot_change_calibration(name, values) -> None:
    example = {"positions": np.array([[0, 0], [1, 1]]), name: values}
    with pytest.raises(ValueError, match=name):
        checkpoints._sampled_node_fraction(example)


def test_sampling_fraction_counts_nodes_not_samples() -> None:
    example = {"positions": np.array([[0, 0], [1, 1]]), "node_sample_count": [0, 4]}
    assert checkpoints._sampled_node_fraction(example) == 0.5
    example.pop("node_sample_count")
    example["sample_nodes"] = [1, 1, 1, 1]
    assert checkpoints._sampled_node_fraction(example) == 0.5
