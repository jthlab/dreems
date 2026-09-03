"""Calibrate the production surface gate from cached independent predictions."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from dreems import calibration, checkpoints, metrics
from dreems.workflow_io import json_ready, write_json

INTERVALS = {
    "50": 0.6744897501960817,
    "80": 1.2815515655446004,
    "95": 1.959963984540054,
}


def density_label(value: float) -> str:
    if value <= 0.10:
        return "sparse"
    if value <= 0.25:
        return "medium"
    return "dense"


def load_included_rows(path: Path) -> dict[int, dict[str, Any]]:
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
        if index in result:
            raise ValueError(f"duplicate row for example {index}")
        row["family"] = family
        result[index] = row
    return result


def prediction_index(path: Path) -> int:
    match = re.search(r"example_(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"cannot infer example index from {path}")
    return int(match.group(1))


def load_predictions(paths: list[Path]) -> dict[int, dict[str, np.ndarray]]:
    result: dict[int, dict[str, np.ndarray]] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            if {"replicate", "edge_offsets", "edge_truth", "edge_mean", "edge_sd"} <= set(
                archive.files
            ):
                replicates = np.asarray(archive["replicate"], dtype=np.int64)
                offsets = np.asarray(archive["edge_offsets"], dtype=np.int64)
                truth = np.asarray(archive["edge_truth"], dtype=np.float64)
                mean = np.asarray(archive["edge_mean"], dtype=np.float64)
                sd = np.asarray(archive["edge_sd"], dtype=np.float64)
            elif {"truth", "dreems", "edge_sd"} <= set(archive.files):
                replicates = np.asarray([prediction_index(path)], dtype=np.int64)
                truth = np.asarray(archive["truth"], dtype=np.float64)
                mean = np.asarray(archive["dreems"], dtype=np.float64)
                sd = np.asarray(archive["edge_sd"], dtype=np.float64)
                offsets = np.asarray([0, truth.size], dtype=np.int64)
            else:
                raise ValueError(f"{path} is not a recognized prediction archive")
        if truth.shape != mean.shape or truth.shape != sd.shape:
            raise ValueError(f"truth, mean, and SD shapes disagree in {path}")
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


def split_examples(rows: dict[int, dict[str, Any]], seed: int) -> dict[str, list[int]]:
    """Make deterministic, family- and density-balanced thirds."""

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


def arrays_for(
    indices: list[int],
    rows: dict[int, dict[str, Any]],
    predictions: dict[int, dict[str, np.ndarray]],
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    errors = [predictions[index]["error"] for index in indices]
    raw_sd = [predictions[index]["raw_sd"] for index in indices]
    density = np.asarray(
        [float(rows[index]["sampled_node_fraction"]) for index in indices],
        dtype=np.float64,
    )
    return errors, raw_sd, density


def calibration_metrics(
    errors: list[np.ndarray],
    raw_sd: list[np.ndarray],
    density: np.ndarray,
    fitted: calibration.SDCalibration,
) -> dict[str, float | int]:
    graph_nll: list[float] = []
    coverage = {name: [] for name in INTERVALS}
    edge_count = 0
    for error, sd, local_density in zip(errors, raw_sd, density, strict=True):
        local_sd = fitted.apply(sd, sampled_node_fraction=float(local_density))
        finite = np.isfinite(error) & np.isfinite(local_sd) & (local_sd > 0.0)
        standardized = error[finite] / local_sd[finite]
        if standardized.size == 0:
            continue
        graph_nll.append(
            float(np.mean(np.log(local_sd[finite]) + 0.5 * standardized * standardized))
        )
        for name, z_value in INTERVALS.items():
            coverage[name].append(float(np.mean(np.abs(standardized) <= z_value)))
        edge_count += int(standardized.size)
    if not graph_nll:
        raise ValueError("no finite calibration observations")
    result: dict[str, float | int] = {
        "graphs": len(graph_nll),
        "edges": edge_count,
        "graph_balanced_nll": float(np.mean(graph_nll)),
    }
    result.update(
        {
            f"graph_balanced_coverage_{name}": float(np.mean(values))
            for name, values in coverage.items()
        }
    )
    return result


def candidates(
    errors: list[np.ndarray],
    raw_sd: list[np.ndarray],
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


def write_verification_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(json_ready(rows))


def calibrate(args: argparse.Namespace) -> None:
    rows = load_included_rows(args.included_rows)
    predictions = load_predictions(args.predictions)
    missing = sorted(set(rows).difference(predictions))
    if missing:
        raise ValueError(f"missing cached predictions for examples {missing[:10]}")
    predictions = {index: predictions[index] for index in rows}
    splits = split_examples(rows, int(args.seed))

    fit_arrays = arrays_for(splits["fit"], rows, predictions)
    fitted_candidates = candidates(*fit_arrays)
    fit_payload = {
        "stage": "surface_gate_sd_calibration_fit",
        "seed": int(args.seed),
        "split_examples": splits["fit"],
        "candidates": {
            name: {
                "calibration": fitted.to_dict(),
                "fit_metrics": calibration_metrics(*fit_arrays, fitted),
            }
            for name, fitted in fitted_candidates.items()
        },
    }
    write_json(args.fit_output, fit_payload)

    selection_arrays = arrays_for(splits["selection"], rows, predictions)
    selection_metrics = {
        name: calibration_metrics(*selection_arrays, fitted)
        for name, fitted in fitted_candidates.items()
    }
    selected_name = min(
        selection_metrics,
        key=lambda name: float(selection_metrics[name]["graph_balanced_nll"]),
    )
    selected = fitted_candidates[selected_name]
    selection_payload = {
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
                **calibration_metrics(*verification_arrays, fitted),
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
                    **calibration_metrics(
                        [verification_arrays[0][index] for index in selected_indices],
                        [verification_arrays[1][index] for index in selected_indices],
                        verification_arrays[2][selected_indices],
                        fitted,
                    ),
                }
            )
    verification_payload = {
        "stage": "surface_gate_sd_calibration_verification",
        "selected": selected_name,
        "calibration": selected.to_dict(),
        "split_examples": splits["verification"],
        "rows": verification_rows,
    }
    write_json(args.verification_output, verification_payload)
    write_verification_csv(args.verification_csv, verification_rows)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
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
