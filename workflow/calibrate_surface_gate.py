"""Calibrate the production surface gate from cached independent predictions."""

# ruff: noqa: F722, F821, UP037

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped

from dreems.evaluation import metrics
from dreems.inference import calibration
from dreems.neural import checkpoints
from dreems.utils.validation import _real_array
from dreems.utils.workflow_io import json_ready, sha256, write_csv, write_json


@beartype
def density_label(value: float) -> str:
    """Use the short sampling-bin labels retained in the release reports."""
    if value <= 0.10:
        return "sparse"
    if value <= 0.25:
        return "medium"
    return "dense"


@beartype
def load_included_rows(path: Path) -> dict[int, dict[str, Any]]:
    """Index the selected independent cases without silently overwriting duplicates."""
    with path.open(newline="") as stream:
        rows = [dict(row) for row in csv.DictReader(stream)]
    if not rows:
        raise ValueError(f"{path} has no independent examples")
    result = {}
    for row in rows:
        index_value = row.get("example", row.get("index"))
        family = row.get("family", row.get("surface_kind"))
        if index_value is None or family is None:
            raise ValueError(f"{path} must contain example/index and family/surface_kind")
        index = int(index_value)
        if index < 0:
            raise ValueError("example indices cannot be negative")
        if index in result:
            raise ValueError(f"duplicate row for example {index}")
        row["family"] = family
        density = float(row["sampled_node_fraction"])
        if not np.isfinite(density) or not 0.0 <= density <= 1.0:
            raise ValueError(f"invalid sampled_node_fraction for example {index}")
        result[index] = row
    return result


@beartype
def prediction_index(path: Path) -> int:
    """Read the example ID from a single-example prediction filename."""
    match = re.search(r"example_(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"cannot infer example index from {path}")
    return int(match.group(1))


@jaxtyped(typechecker=beartype)
def load_predictions(paths: list[Path]) -> dict[int, dict[str, Float[np.ndarray, "_edges"]]]:
    """Read packed or individual edge predictions; offsets must partition the arrays."""
    result: dict[int, dict[str, Float[np.ndarray, "_edges"]]] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            if {"replicate", "edge_offsets", "edge_truth", "edge_mean", "edge_sd"} <= set(
                archive.files
            ):
                replicates = archive["replicate"]
                offsets = archive["edge_offsets"]
                truth = _real_array(archive["edge_truth"], name="edge_truth")
                mean = _real_array(archive["edge_mean"], name="edge_mean")
                sd = _real_array(archive["edge_sd"], name="edge_sd")
            elif {"truth", "dreems", "edge_sd"} <= set(archive.files):
                replicates = np.asarray([prediction_index(path)], dtype=np.int64)
                truth = _real_array(archive["truth"], name="truth")
                mean = _real_array(archive["dreems"], name="dreems")
                sd = _real_array(archive["edge_sd"], name="edge_sd")
                offsets = np.asarray([0, truth.size], dtype=np.int64)
            else:
                raise ValueError(f"{path} is not a recognized prediction archive")
        if truth.shape != mean.shape or truth.shape != sd.shape:
            raise ValueError(f"truth, mean, and SD shapes disagree in {path}")
        if (
            truth.ndim != 1
            or replicates.ndim != 1
            or not replicates.size
            or replicates.dtype.kind not in "iu"
            or np.any(replicates < 0)
            or offsets.shape != (replicates.size + 1,)
            or offsets.dtype.kind not in "iu"
            or offsets[0] != 0
            or offsets[-1] != truth.size
            or np.any(offsets[1:] <= offsets[:-1])
        ):
            raise ValueError(f"invalid replicate IDs or edge offsets in {path}")
        for local_index, replicate in enumerate(replicates):
            start, stop = int(offsets[local_index]), int(offsets[local_index + 1])
            index = int(replicate)
            if index in result:
                raise ValueError(f"duplicate prediction for example {index}")
            result[index] = {
                "error": metrics.centered(truth[start:stop]) - metrics.centered(mean[start:stop]),
                "raw_sd": sd[start:stop],
            }
    return result


@beartype
def split_examples(rows: dict[int, dict[str, Any]], seed: int) -> dict[str, list[int]]:
    """Split each surface family and sampling bin into three reproducible groups."""

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in rows.items():
        groups[(row["family"], density_label(float(row["sampled_node_fraction"])))].append(index)

    rng = np.random.default_rng(seed)
    splits = {"fit": [], "selection": [], "verification": []}
    names = tuple(splits)
    for group in sorted(groups):
        indices = np.asarray(sorted(groups[group]), dtype=np.int64)
        rng.shuffle(indices)
        start = int(rng.integers(0, len(names)))
        for offset, index in enumerate(indices):
            splits[names[(start + offset) % len(names)]].append(int(index))
    for name in names:
        splits[name].sort()
        if not splits[name]:
            raise ValueError(f"calibration split {name} is empty")
    return splits


@jaxtyped(typechecker=beartype)
def arrays_for(
    indices: list[int],
    rows: dict[int, dict[str, Any]],
    predictions: dict[int, dict[str, Float[np.ndarray, "_edges"]]],
) -> tuple[
    list[Float[np.ndarray, "_edges"]], list[Float[np.ndarray, "_edges"]], Float[np.ndarray, "cases"]
]:
    """Keep residuals, SDs, and sampling fractions aligned by example ID."""
    errors = [predictions[index]["error"] for index in indices]
    raw_sd = [predictions[index]["raw_sd"] for index in indices]
    density = np.asarray(
        [float(rows[index]["sampled_node_fraction"]) for index in indices],
        dtype=np.float64,
    )
    return errors, raw_sd, density


@beartype
def write_verification_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the same column order as the JSON report."""
    if not rows:
        raise ValueError("verification rows must be nonempty")
    write_csv(path, json_ready(rows))


@beartype
def calibrate(args: argparse.Namespace) -> None:
    """Fit, select, and verify cached predictions on three disjoint subsets."""
    checkpoint_digest = sha256(args.checkpoint)
    verified = True
    for path in args.predictions:
        with np.load(path, allow_pickle=False) as archive:
            if "checkpoint_sha256" not in archive:
                verified = False
                if getattr(args, "require_checkpoint_hash", False):
                    raise ValueError(
                        f"{path} has no checkpoint hash; regenerate its neural prediction"
                    )
                continue
            recorded = archive["checkpoint_sha256"]
            if (
                recorded.shape != ()
                or recorded.dtype.kind != "U"
                or recorded.item() != checkpoint_digest
            ):
                raise ValueError(f"{path} was not produced by the requested checkpoint")
    provenance = {"checkpoint_sha256": checkpoint_digest} if verified else {}
    rows = load_included_rows(args.included_rows)
    predictions = load_predictions(args.predictions)
    missing = sorted(set(rows).difference(predictions))
    if missing:
        raise ValueError(f"missing cached predictions for examples {missing[:10]}")
    predictions = {index: predictions[index] for index in rows}
    # H2H stores the source checkpoint's reported SD, which may already be
    # calibrated. Fit replacement metadata against its underlying raw SD.
    _, metadata = checkpoints._read_checkpoint_archive(args.checkpoint)
    source_calibration = calibration.SDCalibration.from_dict(metadata.get("edge_sd_calibration"))
    for index, prediction in predictions.items():
        source_scale = source_calibration.scale_for_density(
            float(rows[index]["sampled_node_fraction"])
        )
        prediction["raw_sd"] = prediction["raw_sd"] / source_scale
    splits = split_examples(rows, int(args.seed))

    fit_arrays = arrays_for(splits["fit"], rows, predictions)
    fitted_candidates = calibration.candidates(*fit_arrays)
    fit_payload = {
        **provenance,
        "stage": "surface_gate_sd_calibration_fit",
        "seed": int(args.seed),
        "split_examples": splits["fit"],
        "candidates": {
            name: {
                "calibration": fitted.to_dict(),
                "fit_metrics": calibration.summarize(*fit_arrays, fitted),
            }
            for name, fitted in fitted_candidates.items()
        },
    }
    write_json(args.fit_output, fit_payload)

    selection_arrays = arrays_for(splits["selection"], rows, predictions)
    selection_metrics = {
        name: calibration.summarize(*selection_arrays, fitted)
        for name, fitted in fitted_candidates.items()
    }
    selected_name = min(
        selection_metrics,
        key=lambda name: float(selection_metrics[name]["graph_balanced_nll"]),
    )
    selected = fitted_candidates[selected_name]
    selection_payload = {
        **provenance,
        "stage": "surface_gate_sd_calibration_selection",
        "selected": selected_name,
        "calibration": selected.to_dict(),
        "split_examples": splits["selection"],
        "candidate_metrics": selection_metrics,
    }
    write_json(args.selection_output, selection_payload)

    verification_arrays = arrays_for(splits["verification"], rows, predictions)
    verification_rows = []
    for method_name, fitted in (("raw", calibration.SDCalibration()), ("selected", selected)):
        verification_rows.append(
            {
                "method": method_name,
                "sampling": "all",
                **calibration.summarize(*verification_arrays, fitted),
            }
        )
        labels = np.asarray(
            [density_label(value) for value in verification_arrays[2]],
            dtype=object,
        )
        for label in ("sparse", "medium", "dense"):
            selected_indices = np.flatnonzero(labels == label)
            if selected_indices.size == 0:
                continue
            verification_rows.append(
                {
                    "method": method_name,
                    "sampling": label,
                    **calibration.summarize(
                        [verification_arrays[0][index] for index in selected_indices],
                        [verification_arrays[1][index] for index in selected_indices],
                        verification_arrays[2][selected_indices],
                        fitted,
                    ),
                }
            )
    verification_payload = {
        **provenance,
        "stage": "surface_gate_sd_calibration_verification",
        "selected": selected_name,
        "calibration": selected.to_dict(),
        "split_examples": splits["verification"],
        "rows": verification_rows,
    }
    write_json(args.verification_output, verification_payload)
    write_verification_csv(args.verification_csv, verification_rows)
    if sha256(args.checkpoint) != checkpoint_digest:
        raise ValueError("source checkpoint changed during calibration")
    checkpoints.attach_edge_sd_calibration(
        args.checkpoint,
        args.output_checkpoint,
        selected,
        provenance={
            "fit_report": str(args.fit_output),
            "selection_report": str(args.selection_output),
            "verification_report": str(args.verification_output),
            "fit_examples": len(splits["fit"]),
            "selection_examples": len(splits["selection"]),
            "verification_examples": len(splits["verification"]),
        },
    )


@beartype
def main() -> None:
    """Fit SD metadata without running inference or FEEMS again."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--require-checkpoint-hash", action="store_true")
    parser.add_argument("--included-rows", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--fit-output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    parser.add_argument("--verification-csv", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    calibrate(parser.parse_args())


if __name__ == "__main__":
    main()
