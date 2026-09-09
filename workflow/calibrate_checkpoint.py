"""Fit and verify edge posterior-SD calibration on held-out simulations."""

# ruff: noqa: F722, F821, UP037

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped

from dreems import calibration, checkpoints, corpus, metrics
from dreems.validation import _real_array
from dreems.workflow_io import _atomic_output, json_ready, read_pickle, write_csv, write_json


@beartype
def load_raw_example(path: Path, *, example_index: int = 0) -> dict[str, Any]:
    """Select one trusted corpus example, with an explicit index for packed files."""
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


@jaxtyped(typechecker=beartype)
def truth_edge_logs(example: Mapping[str, Any]) -> Float[np.ndarray, "edges"]:
    """Read the simulator's edge log-conductances, including the older truth key."""
    value = example.get("edge_log_rates", example.get("truth"))
    if value is None:
        raise ValueError("calibration example has no edge-log truth")
    return _real_array(value, name="edge-log truth")


@beartype
def command_case(args: argparse.Namespace) -> None:
    """Save centered edge residuals and uncalibrated SDs for one simulation."""
    example = load_raw_example(args.input, example_index=int(args.example_index))
    prediction = checkpoints.predict_examples(args.checkpoint, [example])
    truth = metrics.centered(truth_edge_logs(example))
    estimate = metrics.centered(prediction.edge_mean[0])
    raw_sd = np.asarray(prediction.raw_edge_sd[0], dtype=np.float64)
    if truth.shape != estimate.shape or truth.shape != raw_sd.shape:
        raise ValueError("truth, prediction, and posterior SD must align by edge")
    metadata = corpus.example_metadata(example, int(args.index))
    density = float(metadata["sampled_node_fraction"])

    with _atomic_output(args.output) as stream:
        np.savez_compressed(
            stream,
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


@beartype
def command_canonicalize(args: argparse.Namespace) -> None:
    """Normalize checkpoint metadata while retaining its Equinox parameter bytes."""
    members, metadata = checkpoints._read_checkpoint_archive(args.checkpoint)
    params_sha256 = hashlib.sha256(members[checkpoints.CHECKPOINT_PARAMS]).hexdigest()
    if metadata["model"]["features"]["kind"] == "topology_diffusion_bands_v2":
        with args.checkpoint.open("rb") as source, _atomic_output(args.output) as destination:
            shutil.copyfileobj(source, destination)
        return
    checkpoints.migrate_node_covariance_checkpoint(
        args.checkpoint,
        args.output,
        provenance={
            "source_checkpoint": str(args.checkpoint),
            "source_checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "source_params_sha256": params_sha256,
        },
    )


@jaxtyped(typechecker=beartype)
def load_cases(
    paths: Sequence[Path],
) -> tuple[
    list[Float[np.ndarray, "_edges"]], list[Float[np.ndarray, "_edges"]], Float[np.ndarray, "cases"]
]:
    """Read residuals and SDs for each graph, then one sampling fraction per graph."""
    if len(paths) != len({path.resolve() for path in paths}):
        raise ValueError("duplicate calibration case paths")
    errors: list[Float[np.ndarray, "_edges"]] = []
    raw_sd: list[Float[np.ndarray, "_edges"]] = []
    density: list[float] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            errors.append(_real_array(payload["error"], name="calibration error"))
            raw_sd.append(_real_array(payload["raw_sd"], name="raw_sd"))
            fraction = _real_array(payload["sampled_node_fraction"], name="sampled_node_fraction")
            if fraction.ndim != 0:
                raise ValueError("sampled_node_fraction must be a scalar")
            density.append(fraction.item())
    return errors, raw_sd, np.asarray(density, dtype=np.float64)


@beartype
def command_fit(args: argparse.Namespace) -> None:
    """Fit candidate scales without seeing the selection or verification cases."""
    errors, raw_sd, density = load_cases(args.inputs)
    candidates = calibration.candidates(errors, raw_sd, density)
    write_json(
        args.output,
        {
            "stage": "edge_sd_calibration_fit",
            "fit_examples": len(errors),
            "candidates": {
                name: {
                    "calibration": fitted.to_dict(),
                    "fit_metrics": calibration.summarize(errors, raw_sd, density, fitted),
                }
                for name, fitted in candidates.items()
            },
        },
    )


@beartype
def load_candidates(path: Path) -> dict[str, calibration.SDCalibration]:
    """Read the fitted scale options, retaining their order for deterministic ties."""
    payload = json.loads(path.read_text())
    candidates = payload.get("candidates")
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError(f"{path} has no calibration candidates")
    return {
        str(name): calibration.SDCalibration.from_dict(value["calibration"])
        for name, value in candidates.items()
    }


@beartype
def command_select(args: argparse.Namespace) -> None:
    """Choose the fitted option with the lowest graph-balanced selection NLL."""
    errors, raw_sd, density = load_cases(args.inputs)
    candidates = load_candidates(args.fit)
    candidate_metrics = {
        name: calibration.summarize(errors, raw_sd, density, fitted)
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


@beartype
def density_label(value: float) -> str:
    """Use the report's existing labels and closed upper bin boundaries."""
    if value <= 0.10:
        return "sparse <=10%"
    if value <= 0.25:
        return "medium 10-25%"
    return "dense >25%"


@jaxtyped(typechecker=beartype)
def verification_rows(
    errors: Sequence[Float[np.ndarray, "_edges"]],
    raw_sd: Sequence[Float[np.ndarray, "_edges"]],
    density: Float[np.ndarray, "cases"],
    selected: calibration.SDCalibration,
) -> list[dict[str, Any]]:
    """Report held-out coverage overall and in each nonempty sampling bin."""
    methods = {"raw": calibration.SDCalibration(), "selected": selected}
    rows: list[dict[str, Any]] = []
    labels = np.asarray([density_label(value) for value in density], dtype=object)
    for method, fitted in methods.items():
        rows.append(
            {
                "method": method,
                "sampling": "all",
                **calibration.summarize(errors, raw_sd, density, fitted),
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
                    **calibration.summarize(
                        [errors[index] for index in selected_indices],
                        [raw_sd[index] for index in selected_indices],
                        density[selected_indices],
                        fitted,
                    ),
                }
            )
    return rows


@beartype
def command_verify(args: argparse.Namespace) -> None:
    """Evaluate the selected calibration without refitting on verification data."""
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
    write_csv(args.output_csv, json_ready(rows))


@beartype
def command_attach(args: argparse.Namespace) -> None:
    """Attach scale metadata to a separate checkpoint, leaving all weights intact."""
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


@beartype
def build_parser() -> argparse.ArgumentParser:
    """Expose each calibration stage as its own Snakemake worker command."""
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


@beartype
def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
