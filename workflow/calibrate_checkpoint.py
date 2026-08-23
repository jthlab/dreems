"""Fit and verify edge posterior-SD calibration on held-out simulations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from dreems import calibration, checkpoints, corpus, metrics
from dreems.workflow_io import json_ready, read_pickle, write_json

INTERVAL_LEVELS = {
    "50": 0.6744897501960817,
    "80": 1.2815515655446004,
    "95": 1.959963984540054,
}


def load_raw_example(path: Path, *, example_index: int = 0) -> dict[str, Any]:
    payload = read_pickle(path)
    if isinstance(payload, list):
        if not 0 <= example_index < len(payload):
            raise IndexError(
                f"example index {example_index} is outside {path} with {len(payload)} examples"
            )
        payload = payload[example_index]
    elif example_index != 0:
        raise IndexError(f"cannot select example {example_index} from unbatched input {path}")
    if not isinstance(payload, dict):
        raise TypeError(f"expected a dictionary in {path}")
    return payload


def truth_edge_logs(example: Mapping[str, Any]) -> np.ndarray:
    value = example.get("edge_log_rates", example.get("truth"))
    if value is None:
        raise ValueError("calibration example has no edge-log truth")
    return np.asarray(value, dtype=np.float64)


def command_case(args: argparse.Namespace) -> None:
    example = load_raw_example(args.input, example_index=int(args.example_index))
    prediction = checkpoints.predict_examples(args.checkpoint, [example])
    truth = metrics.centered(truth_edge_logs(example))
    estimate = metrics.centered(prediction.edge_mean[0])
    raw_sd = np.asarray(prediction.raw_edge_sd[0], dtype=np.float64)
    if truth.shape != estimate.shape or truth.shape != raw_sd.shape:
        raise ValueError("truth, prediction, and posterior SD must align by edge")
    metadata = corpus.example_metadata(example, int(args.index))
    density = float(metadata["sampled_node_fraction"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        error=truth - estimate,
        raw_sd=raw_sd,
        sampled_node_fraction=np.asarray(density, dtype=np.float64),
        num_nodes=np.asarray(metadata["num_nodes"], dtype=np.int64),
        num_edges=np.asarray(metadata["num_edges"], dtype=np.int64),
        index=np.asarray(args.index, dtype=np.int64),
        role=np.asarray(args.role),
    )
    print(
        json.dumps(
            {
                "stage": "calibration_case",
                "role": args.role,
                "index": int(args.index),
                "edges": int(truth.size),
                "sampled_node_fraction": density,
            }
        ),
        flush=True,
    )


def command_canonicalize(args: argparse.Namespace) -> None:
    with zipfile.ZipFile(args.checkpoint) as archive:
        params_sha256 = hashlib.sha256(archive.read(checkpoints.CHECKPOINT_PARAMS)).hexdigest()
    checkpoints.migrate_node_covariance_checkpoint(
        args.checkpoint,
        args.output,
        provenance={
            "source_checkpoint": str(args.checkpoint),
            "source_checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "source_params_sha256": params_sha256,
        },
    )


def load_cases(paths: Sequence[Path]) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    errors: list[np.ndarray] = []
    raw_sd: list[np.ndarray] = []
    density: list[float] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            errors.append(np.asarray(payload["error"], dtype=np.float64))
            raw_sd.append(np.asarray(payload["raw_sd"], dtype=np.float64))
            density.append(float(np.asarray(payload["sampled_node_fraction"]).item()))
    return errors, raw_sd, np.asarray(density, dtype=np.float64)


def calibration_metrics(
    errors: Sequence[np.ndarray],
    raw_sd: Sequence[np.ndarray],
    density: np.ndarray,
    fitted: calibration.SDCalibration,
) -> dict[str, float | int]:
    nll: list[float] = []
    coverage = {level: [] for level in INTERVAL_LEVELS}
    edge_count = 0
    for error, sd, local_density in zip(errors, raw_sd, density, strict=True):
        calibrated_sd = fitted.apply(sd, sampled_node_fraction=float(local_density))
        finite = np.isfinite(error) & np.isfinite(calibrated_sd) & (calibrated_sd > 0.0)
        if not np.any(finite):
            continue
        local_error = np.asarray(error[finite], dtype=np.float64)
        local_sd = np.asarray(calibrated_sd[finite], dtype=np.float64)
        standardized = local_error / local_sd
        nll.append(float(np.mean(np.log(local_sd) + 0.5 * standardized * standardized)))
        for level, z_value in INTERVAL_LEVELS.items():
            coverage[level].append(float(np.mean(np.abs(standardized) <= z_value)))
        edge_count += int(local_error.size)
    if not nll:
        raise ValueError("no finite calibration cases")
    result: dict[str, float | int] = {
        "graphs": len(nll),
        "edges": edge_count,
        "graph_balanced_nll": float(np.mean(nll)),
    }
    for level, values in coverage.items():
        result[f"graph_balanced_coverage_{level}"] = float(np.mean(values))
    return result


def calibration_candidates(
    errors: Sequence[np.ndarray],
    raw_sd: Sequence[np.ndarray],
    density: np.ndarray,
) -> dict[str, calibration.SDCalibration]:
    density_fit = calibration.SDCalibration.from_graphs(
        errors,
        raw_sd,
        density,
        density_shrinkage=25.0,
    )
    return {
        "raw": calibration.SDCalibration(),
        "global": calibration.SDCalibration(
            method="global",
            global_scale=density_fit.global_scale,
        ),
        "density": density_fit,
    }


def command_fit(args: argparse.Namespace) -> None:
    errors, raw_sd, density = load_cases(args.inputs)
    candidates = calibration_candidates(errors, raw_sd, density)
    write_json(
        args.output,
        {
            "stage": "edge_sd_calibration_fit",
            "fit_examples": len(errors),
            "candidates": {
                name: {
                    "calibration": fitted.to_dict(),
                    "fit_metrics": calibration_metrics(errors, raw_sd, density, fitted),
                }
                for name, fitted in candidates.items()
            },
        },
    )


def load_candidates(path: Path) -> dict[str, calibration.SDCalibration]:
    payload = json.loads(path.read_text())
    candidates = payload.get("candidates")
    if not isinstance(candidates, dict):
        raise ValueError(f"{path} has no calibration candidates")
    return {
        str(name): calibration.SDCalibration.from_dict(value["calibration"])
        for name, value in candidates.items()
    }


def command_select(args: argparse.Namespace) -> None:
    errors, raw_sd, density = load_cases(args.inputs)
    candidates = load_candidates(args.fit)
    candidate_metrics = {
        name: calibration_metrics(errors, raw_sd, density, fitted)
        for name, fitted in candidates.items()
    }
    selected_name = min(
        candidate_metrics,
        key=lambda name: float(candidate_metrics[name]["graph_balanced_nll"]),
    )
    write_json(
        args.output,
        {
            "stage": "edge_sd_calibration_selection",
            "selection_examples": len(errors),
            "selected": selected_name,
            "calibration": candidates[selected_name].to_dict(),
            "candidate_metrics": candidate_metrics,
            "fit": str(args.fit),
        },
    )


def density_label(value: float) -> str:
    if value <= 0.10:
        return "sparse <=10%"
    if value <= 0.25:
        return "medium 10-25%"
    return "dense >25%"


def verification_rows(
    errors: Sequence[np.ndarray],
    raw_sd: Sequence[np.ndarray],
    density: np.ndarray,
    selected: calibration.SDCalibration,
) -> list[dict[str, Any]]:
    methods = {"raw": calibration.SDCalibration(), "selected": selected}
    rows: list[dict[str, Any]] = []
    labels = np.asarray([density_label(value) for value in density], dtype=object)
    for method, fitted in methods.items():
        rows.append(
            {
                "method": method,
                "sampling": "all",
                **calibration_metrics(errors, raw_sd, density, fitted),
            }
        )
        for label in ("sparse <=10%", "medium 10-25%", "dense >25%"):
            selected_indices = np.flatnonzero(labels == label)
            if selected_indices.size == 0:
                continue
            rows.append(
                {
                    "method": method,
                    "sampling": label,
                    **calibration_metrics(
                        [errors[index] for index in selected_indices],
                        [raw_sd[index] for index in selected_indices],
                        density[selected_indices],
                        fitted,
                    ),
                }
            )
    return rows


def command_verify(args: argparse.Namespace) -> None:
    errors, raw_sd, density = load_cases(args.inputs)
    selection = json.loads(args.selection.read_text())
    selected = calibration.SDCalibration.from_dict(selection["calibration"])
    if args.checkpoint is not None:
        loaded = checkpoints.load_checkpoint(args.checkpoint)
        if loaded.edge_sd_calibration != selected:
            raise ValueError("calibrated checkpoint does not contain the selected SD calibration")
    rows = verification_rows(errors, raw_sd, density, selected)
    write_json(
        args.output_json,
        {
            "stage": "edge_sd_calibration_verification",
            "verification_examples": len(errors),
            "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
            "selected": selection["selected"],
            "calibration": selected.to_dict(),
            "rows": rows,
        },
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(json_ready(rows))


def command_attach(args: argparse.Namespace) -> None:
    selection = json.loads(args.selection.read_text())
    fitted = calibration.SDCalibration.from_dict(selection["calibration"])
    checkpoints.attach_edge_sd_calibration(
        args.checkpoint,
        args.output,
        fitted,
        provenance={
            "fit_report": str(args.fit),
            "selection_report": str(args.selection),
            "fit_examples": int(json.loads(args.fit.read_text())["fit_examples"]),
            "selection_examples": int(selection["selection_examples"]),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    canonicalize = subparsers.add_parser("canonicalize")
    canonicalize.add_argument("--checkpoint", type=Path, required=True)
    canonicalize.add_argument("--output", type=Path, required=True)
    canonicalize.set_defaults(func=command_canonicalize)

    case = subparsers.add_parser("case")
    case.add_argument("--checkpoint", type=Path, required=True)
    case.add_argument("--input", type=Path, required=True)
    case.add_argument("--role", choices=("fit", "selection", "verification"), required=True)
    case.add_argument("--index", type=int, required=True)
    case.add_argument("--example-index", type=int, default=0)
    case.add_argument("--output", type=Path, required=True)
    case.set_defaults(func=command_case)

    fit = subparsers.add_parser("fit")
    fit.add_argument("--inputs", type=Path, nargs="+", required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.set_defaults(func=command_fit)

    select = subparsers.add_parser("select")
    select.add_argument("--fit", type=Path, required=True)
    select.add_argument("--inputs", type=Path, nargs="+", required=True)
    select.add_argument("--output", type=Path, required=True)
    select.set_defaults(func=command_select)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--checkpoint", type=Path)
    verify.add_argument("--selection", type=Path, required=True)
    verify.add_argument("--inputs", type=Path, nargs="+", required=True)
    verify.add_argument("--output-json", type=Path, required=True)
    verify.add_argument("--output-csv", type=Path, required=True)
    verify.set_defaults(func=command_verify)

    attach = subparsers.add_parser("attach")
    attach.add_argument("--checkpoint", type=Path, required=True)
    attach.add_argument("--fit", type=Path, required=True)
    attach.add_argument("--selection", type=Path, required=True)
    attach.add_argument("--output", type=Path, required=True)
    attach.set_defaults(func=command_attach)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
