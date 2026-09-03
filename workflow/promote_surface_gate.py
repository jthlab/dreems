"""Promote a staged full retrain to the packaged DREEMS checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import jax
import numpy as np

from dreems import checkpoints

ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
SURFACE_GATE_FIELDS = {"surface_gate_hidden", "surface_gate_out"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_archive(path: Path) -> tuple[dict[str, Any], bytes]:
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read(checkpoints.CHECKPOINT_METADATA))
        params = archive.read(checkpoints.CHECKPOINT_PARAMS)
    if not isinstance(metadata, dict):
        raise TypeError(f"checkpoint metadata in {path} is not an object")
    return metadata, params


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def assert_equal_normalization(
    left: checkpoints.DREEMSCheckpoint,
    right: checkpoints.DREEMSCheckpoint,
) -> None:
    if left.normalization != right.normalization:
        raise ValueError(f"normalization differs between {left.path} and {right.path}")


def assert_equal_model_fields(
    left: checkpoints.DREEMSCheckpoint,
    right: checkpoints.DREEMSCheckpoint,
    *,
    excluded: set[str],
) -> None:
    """Require exact equality for every model field outside the named gates."""

    for field in dataclasses.fields(left.params):
        if field.name in excluded:
            continue
        left_leaves = jax.tree.leaves(getattr(left.params, field.name))
        right_leaves = jax.tree.leaves(getattr(right.params, field.name))
        if len(left_leaves) != len(right_leaves):
            raise ValueError(f"model field {field.name} has a different tree structure")
        for index, (left_leaf, right_leaf) in enumerate(
            zip(left_leaves, right_leaves, strict=True)
        ):
            if not np.array_equal(np.asarray(left_leaf), np.asarray(right_leaf)):
                raise ValueError(f"model field {field.name} leaf {index} changed")


def verified_training_stage(
    *,
    checkpoint_metadata: dict[str, Any],
    summary_path: Path,
    history_path: Path,
    name: str,
) -> dict[str, Any]:
    summary = read_json(summary_path)
    history = read_json_lines(history_path)
    training_run = summary.get("training_run")
    if not isinstance(training_run, dict):
        raise ValueError(f"{name} summary has no training_run")
    if training_run.get("history") != history:
        raise ValueError(f"{name} summary and JSON-lines history disagree")
    if not training_run.get("restored_best_validation", False):
        raise ValueError(f"{name} did not restore its best validation checkpoint")

    checkpoint_run = checkpoint_metadata.get("training_run")
    if not isinstance(checkpoint_run, dict):
        raise ValueError(f"{name} checkpoint has no training_run")
    for field in ("best_step", "best_validation_loss"):
        if checkpoint_run.get(field) != training_run.get(field):
            raise ValueError(f"{name} checkpoint and summary disagree on {field}")
    return {
        "losses": summary.get("losses"),
        "training_config": checkpoint_metadata.get("training_config"),
        "training_run": training_run,
    }


def verify_h2h_report(
    *,
    checkpoint: Path,
    protocol_path: Path,
    report_path: Path,
    expected_examples: int,
    label: str,
) -> None:
    protocol = read_json(protocol_path)
    report = read_json(report_path)
    if report.get("protocol") != protocol:
        raise ValueError(f"{label} report and protocol file disagree")
    recorded_hash = protocol.get("dreems", {}).get("checkpoint_sha256")
    if recorded_hash != sha256(checkpoint):
        raise ValueError(f"{label} report does not use the promoted checkpoint")
    if report.get("examples") != expected_examples:
        raise ValueError(
            f"{label} report has {report.get('examples')} examples; expected {expected_examples}"
        )


def verify_calibration_reports(
    *,
    fit_path: Path,
    selection_path: Path,
    verification_path: Path,
    calibration: dict[str, Any],
    expected_examples: int,
) -> dict[str, int]:
    fit = read_json(fit_path)
    selection = read_json(selection_path)
    verification = read_json(verification_path)
    selected = selection.get("selected")
    if selected not in fit.get("candidates", {}):
        raise ValueError("selected SD calibration is absent from the fit report")
    numerical_calibration = dict(calibration)
    numerical_calibration.pop("provenance", None)
    if selection.get("calibration") != numerical_calibration:
        raise ValueError("selected SD calibration differs from checkpoint metadata")
    if (
        verification.get("calibration") != numerical_calibration
        or verification.get("selected") != selected
    ):
        raise ValueError("verification report does not describe the selected calibration")

    split_sets = [
        set(fit.get("split_examples", [])),
        set(selection.get("split_examples", [])),
        set(verification.get("split_examples", [])),
    ]
    if any(
        left & right for index, left in enumerate(split_sets) for right in split_sets[index + 1 :]
    ):
        raise ValueError("SD calibration fit, selection, and verification splits overlap")
    if len(set().union(*split_sets)) != expected_examples:
        raise ValueError("SD calibration splits do not cover the full independent benchmark")
    return {
        "fit_examples": len(split_sets[0]),
        "selection_examples": len(split_sets[1]),
        "verification_examples": len(split_sets[2]),
    }


def zip_member(name: str, contents: bytes) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    return info, contents


def promote(
    *,
    core: Path,
    surface_gate: Path,
    calibrated_surface_gate: Path,
    training_configs: list[Path],
    surface_gate_summary: Path,
    surface_gate_history: Path,
    core_summary: Path,
    core_history: Path,
    calibration_fit: Path,
    calibration_selection: Path,
    calibration_verification: Path,
    broad_protocol: Path,
    broad_report: Path,
    hard_protocol: Path,
    hard_report: Path,
    expected_broad_examples: int,
    expected_hard_examples: int,
    output: Path,
) -> None:
    core_checkpoint = checkpoints.load_checkpoint(core)
    gated = checkpoints.load_checkpoint(surface_gate)
    calibrated = checkpoints.load_checkpoint(calibrated_surface_gate)

    if core_checkpoint.model["architecture"].get("kind") != "diffusion_band_initializer_gate_v2":
        raise ValueError("core checkpoint does not use the v2 initializer gate")
    if gated.model["architecture"].get("kind") != "diffusion_band_surface_gate_v3":
        raise ValueError("surface checkpoint does not use the v3 surface gate")
    if gated.payload["training_config"]["optimizer"].get("trainable_scope") != "surface_gate":
        raise ValueError("surface checkpoint was not trained with a frozen core")

    assert_equal_normalization(core_checkpoint, gated)
    assert_equal_normalization(gated, calibrated)
    assert_equal_model_fields(core_checkpoint, gated, excluded=SURFACE_GATE_FIELDS)
    assert_equal_model_fields(gated, calibrated, excluded=set())

    core_metadata, _ = read_archive(core)
    gated_metadata, gated_params = read_archive(surface_gate)
    calibrated_metadata, calibrated_params = read_archive(calibrated_surface_gate)
    if gated_params != calibrated_params:
        raise ValueError("SD calibration changed serialized model parameters")
    calibration = calibrated_metadata.get("edge_sd_calibration")
    if not isinstance(calibration, dict):
        raise ValueError("calibrated surface checkpoint has no edge SD calibration")

    core_stage = verified_training_stage(
        checkpoint_metadata=core_metadata,
        summary_path=core_summary,
        history_path=core_history,
        name="core",
    )
    gate_stage = verified_training_stage(
        checkpoint_metadata=gated_metadata,
        summary_path=surface_gate_summary,
        history_path=surface_gate_history,
        name="surface gate",
    )
    verify_h2h_report(
        checkpoint=surface_gate,
        protocol_path=broad_protocol,
        report_path=broad_report,
        expected_examples=expected_broad_examples,
        label="broad H2H",
    )
    verify_h2h_report(
        checkpoint=surface_gate,
        protocol_path=hard_protocol,
        report_path=hard_report,
        expected_examples=expected_hard_examples,
        label="hard-case H2H",
    )
    calibration_split_sizes = verify_calibration_reports(
        fit_path=calibration_fit,
        selection_path=calibration_selection,
        verification_path=calibration_verification,
        calibration=calibration,
        expected_examples=expected_broad_examples,
    )

    calibration = dict(calibration)
    calibration["provenance"] = calibration_split_sizes
    gated_metadata["edge_sd_calibration"] = calibration
    gated_metadata["train_losses"] = gate_stage["losses"]
    gated_metadata["training_run"] = gate_stage["training_run"]
    gated_metadata["training_stages"] = {
        "core": core_stage,
        "surface_gate": gate_stage,
    }
    gated_metadata["provenance"] = {
        "broad_h2h_protocol_sha256": sha256(broad_protocol),
        "broad_h2h_report_sha256": sha256(broad_report),
        "calibration_fit_report_sha256": sha256(calibration_fit),
        "calibration_selection_report_sha256": sha256(calibration_selection),
        "calibration_verification_report_sha256": sha256(calibration_verification),
        "core_checkpoint_sha256": sha256(core),
        "core_training_history_sha256": sha256(core_history),
        "core_training_summary_sha256": sha256(core_summary),
        "frozen_core_verified": True,
        "hard_h2h_protocol_sha256": sha256(hard_protocol),
        "hard_h2h_report_sha256": sha256(hard_report),
        "posterior_sd_calibration_reused": False,
        "surface_gate_checkpoint_sha256": sha256(surface_gate),
        "surface_gate_training_history_sha256": sha256(surface_gate_history),
        "surface_gate_training_summary_sha256": sha256(surface_gate_summary),
        "training_configs": [
            {"path": str(path), "sha256": sha256(path)} for path in training_configs
        ],
    }

    metadata_bytes = json.dumps(
        gated_metadata,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(temporary_path, "w") as archive:
            archive.writestr(*zip_member(checkpoints.CHECKPOINT_METADATA, metadata_bytes))
            archive.writestr(*zip_member(checkpoints.CHECKPOINT_PARAMS, gated_params))
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--surface-gate", type=Path, required=True)
    parser.add_argument("--calibrated-surface-gate", type=Path, required=True)
    parser.add_argument("--training-configs", type=Path, nargs="+", required=True)
    parser.add_argument("--surface-gate-summary", type=Path, required=True)
    parser.add_argument("--surface-gate-history", type=Path, required=True)
    parser.add_argument("--core-summary", type=Path, required=True)
    parser.add_argument("--core-history", type=Path, required=True)
    parser.add_argument("--calibration-fit", type=Path, required=True)
    parser.add_argument("--calibration-selection", type=Path, required=True)
    parser.add_argument("--calibration-verification", type=Path, required=True)
    parser.add_argument("--broad-protocol", type=Path, required=True)
    parser.add_argument("--broad-report", type=Path, required=True)
    parser.add_argument("--hard-protocol", type=Path, required=True)
    parser.add_argument("--hard-report", type=Path, required=True)
    parser.add_argument("--expected-broad-examples", type=int, required=True)
    parser.add_argument("--expected-hard-examples", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    promote(
        core=args.core,
        surface_gate=args.surface_gate,
        calibrated_surface_gate=args.calibrated_surface_gate,
        training_configs=args.training_configs,
        surface_gate_summary=args.surface_gate_summary,
        surface_gate_history=args.surface_gate_history,
        core_summary=args.core_summary,
        core_history=args.core_history,
        calibration_fit=args.calibration_fit,
        calibration_selection=args.calibration_selection,
        calibration_verification=args.calibration_verification,
        broad_protocol=args.broad_protocol,
        broad_report=args.broad_report,
        hard_protocol=args.hard_protocol,
        hard_report=args.hard_report,
        expected_broad_examples=args.expected_broad_examples,
        expected_hard_examples=args.expected_hard_examples,
        output=args.output,
    )


if __name__ == "__main__":
    main()
