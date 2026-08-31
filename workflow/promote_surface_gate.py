"""Promote the audited surface-gated model to the packaged release checkpoint."""

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


def zip_member(name: str, contents: bytes) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    return info, contents


def promote(
    *,
    sampling_robustness: Path,
    surface_gate: Path,
    calibrated_surface_gate: Path,
    surface_gate_config: Path,
    surface_gate_summary: Path,
    surface_gate_history: Path,
    sampling_robustness_summary: Path,
    sampling_robustness_history: Path,
    calibration_fit: Path,
    calibration_selection: Path,
    calibration_verification: Path,
    independent_report: Path,
    broad_report: Path,
    scenario_report: Path,
    output: Path,
) -> None:
    initializer = checkpoints.load_checkpoint(sampling_robustness)
    gated = checkpoints.load_checkpoint(surface_gate)
    calibrated = checkpoints.load_checkpoint(calibrated_surface_gate)

    if initializer.model["architecture"].get("kind") != "diffusion_band_initializer_gate_v2":
        raise ValueError("initializer checkpoint does not use the v2 initializer gate")
    if gated.model["architecture"].get("kind") != "diffusion_band_surface_gate_v3":
        raise ValueError("surface checkpoint does not use the v3 surface gate")
    if gated.payload["training_config"]["optimizer"].get("trainable_scope") != "surface_gate":
        raise ValueError("surface checkpoint was not trained with a frozen core")

    assert_equal_normalization(initializer, gated)
    assert_equal_normalization(gated, calibrated)
    assert_equal_model_fields(initializer, gated, excluded=SURFACE_GATE_FIELDS)
    assert_equal_model_fields(gated, calibrated, excluded=set())

    gated_metadata, gated_params = read_archive(surface_gate)
    calibrated_metadata, calibrated_params = read_archive(calibrated_surface_gate)
    if gated_params != calibrated_params:
        raise ValueError("SD calibration changed serialized model parameters")
    calibration = calibrated_metadata.get("edge_sd_calibration")
    if not isinstance(calibration, dict):
        raise ValueError("calibrated surface checkpoint has no edge SD calibration")

    scenario = read_json(scenario_report)
    recorded_gate = scenario.get("provenance", {}).get("gated_checkpoint_sha256")
    if recorded_gate != sha256(surface_gate):
        raise ValueError("scenario report does not describe the promoted surface gate")

    gate_summary = read_json(surface_gate_summary)
    gate_history = read_json_lines(surface_gate_history)
    if gate_summary["training_run"]["history"] != gate_history:
        raise ValueError("surface-gate summary and JSON-lines history disagree")
    if gated_metadata["training_run"]["best_step"] != gate_summary["training_run"]["best_step"]:
        raise ValueError("surface-gate checkpoint is not the recorded best step")
    if (
        gated_metadata["training_run"]["best_validation_loss"]
        != gate_summary["training_run"]["best_validation_loss"]
    ):
        raise ValueError("surface-gate checkpoint and summary losses disagree")

    sampling_summary = read_json(sampling_robustness_summary)
    sampling_history = read_json_lines(sampling_robustness_history)
    if sampling_summary["training_run"]["history"] != sampling_history:
        raise ValueError("sampling-robustness summary and JSON-lines history disagree")

    gated_metadata["edge_sd_calibration"] = calibration
    gated_metadata["train_losses"] = gate_summary["losses"]
    gated_metadata["training_run"] = gate_summary["training_run"]
    gated_metadata["provenance"] = {
        "broad_h2h_report_sha256": sha256(broad_report),
        "calibration_fit_report_sha256": sha256(calibration_fit),
        "calibration_selection_report_sha256": sha256(calibration_selection),
        "calibration_verification_report_sha256": sha256(calibration_verification),
        "frozen_pre_gate_model_verified": True,
        "independent_gate_report_sha256": sha256(independent_report),
        "posterior_sd_calibration_reused": False,
        "sampling_robustness_checkpoint_sha256": sha256(sampling_robustness),
        "sampling_robustness_training_history_sha256": sha256(sampling_robustness_history),
        "sampling_robustness_training_summary_sha256": sha256(sampling_robustness_summary),
        "scenario_h2h_report_sha256": sha256(scenario_report),
        "surface_gate_checkpoint_sha256": sha256(surface_gate),
        "surface_gate_config_sha256": sha256(surface_gate_config),
        "surface_gate_training_history_sha256": sha256(surface_gate_history),
        "surface_gate_training_summary_sha256": sha256(surface_gate_summary),
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
    parser.add_argument("--sampling-robustness", type=Path, required=True)
    parser.add_argument("--surface-gate", type=Path, required=True)
    parser.add_argument("--calibrated-surface-gate", type=Path, required=True)
    parser.add_argument("--surface-gate-config", type=Path, required=True)
    parser.add_argument("--surface-gate-summary", type=Path, required=True)
    parser.add_argument("--surface-gate-history", type=Path, required=True)
    parser.add_argument("--sampling-robustness-summary", type=Path, required=True)
    parser.add_argument("--sampling-robustness-history", type=Path, required=True)
    parser.add_argument("--calibration-fit", type=Path, required=True)
    parser.add_argument("--calibration-selection", type=Path, required=True)
    parser.add_argument("--calibration-verification", type=Path, required=True)
    parser.add_argument("--independent-report", type=Path, required=True)
    parser.add_argument("--broad-report", type=Path, required=True)
    parser.add_argument("--scenario-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    promote(
        sampling_robustness=args.sampling_robustness,
        surface_gate=args.surface_gate,
        calibrated_surface_gate=args.calibrated_surface_gate,
        surface_gate_config=args.surface_gate_config,
        surface_gate_summary=args.surface_gate_summary,
        surface_gate_history=args.surface_gate_history,
        sampling_robustness_summary=args.sampling_robustness_summary,
        sampling_robustness_history=args.sampling_robustness_history,
        calibration_fit=args.calibration_fit,
        calibration_selection=args.calibration_selection,
        calibration_verification=args.calibration_verification,
        independent_report=args.independent_report,
        broad_report=args.broad_report,
        scenario_report=args.scenario_report,
        output=args.output,
    )


if __name__ == "__main__":
    main()
