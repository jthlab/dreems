#!/usr/bin/env python3
"""Test whether covariance fit distinguishes uniform from structured surfaces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from dreems import metrics, refinement
from dreems.inference import InferenceExample, genotype_covariance, load_model
from dreems.workflow_io import json_ready

NULL_FAMILIES = {"constant", "feems_homogeneous"}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        return pickle.load(stream)


def _single_example(path: Path) -> dict[str, Any]:
    value = _load_pickle(path)
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"expected one example in {path}, found {len(value)}")
        value = value[0]
    if not isinstance(value, dict):
        raise TypeError(f"expected an example dictionary in {path}")
    return value


def _prediction_map(path: Path) -> dict[int, dict[str, np.ndarray]]:
    with np.load(path) as archive:
        replicate = np.asarray(archive["replicate"], dtype=np.int64)
        edge_offsets = np.asarray(archive["edge_offsets"], dtype=np.int64)
        edge_mean = np.asarray(archive["edge_mean"], dtype=np.float64)
        q_offsets = np.asarray(archive["q_offsets"], dtype=np.int64)
        q_mean = np.asarray(archive["q_mean"], dtype=np.float64)
    if edge_offsets.shape != (replicate.size + 1,) or q_offsets.shape != (replicate.size + 1,):
        raise ValueError(f"invalid prediction offsets in {path}")
    result = {}
    for local, index in enumerate(replicate):
        edge_start, edge_stop = int(edge_offsets[local]), int(edge_offsets[local + 1])
        q_start, q_stop = int(q_offsets[local]), int(q_offsets[local + 1])
        result[int(index)] = {
            "edge": edge_mean[edge_start:edge_stop],
            "q": q_mean[q_start:q_stop],
        }
    if len(result) != replicate.size:
        raise ValueError(f"duplicate replicate IDs in {path}")
    return result


def _observed_covariance(
    example: Mapping[str, Any],
    *,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_nodes = np.asarray(example["sample_nodes"], dtype=np.int32)
    if "sample_covariance" in example:
        sample_covariance = np.asarray(example["sample_covariance"], dtype=np.float64)
        effective_snps = int(example.get("effective_snps", sample_covariance.shape[0]))
    elif "genotypes" in example:
        sample_covariance, effective_snps = genotype_covariance(
            np.asarray(example["genotypes"], dtype=np.float64)
        )
    else:
        raise ValueError("example has neither sample_covariance nor genotypes")
    return refinement.empirical_observed_covariance(
        sample_covariance=sample_covariance,
        sample_nodes=sample_nodes,
        num_nodes=num_nodes,
        effective_snps=effective_snps,
    )


def _fit(
    *,
    empirical: np.ndarray,
    observed_nodes: np.ndarray,
    observed_count: np.ndarray,
    edges: np.ndarray,
    edge_logs: np.ndarray,
    node_log_q: np.ndarray,
    num_nodes: int,
) -> refinement.CovarianceFit:
    model_covariance = refinement.model_covariance_at_observed_nodes(
        num_nodes=num_nodes,
        edges=edges,
        edge_logs=edge_logs,
        observed_nodes=observed_nodes,
    )
    q_covariance = refinement.q_covariance_at_observed_nodes(
        node_log_q=node_log_q,
        observed_nodes=observed_nodes,
        observed_sample_count=observed_count,
    )
    return refinement.covariance_fit(
        empirical_covariance=empirical,
        model_covariance=model_covariance,
        q_covariance=q_covariance,
    )


def _fit_fields(prefix: str, fit: refinement.CovarianceFit) -> dict[str, float]:
    return {
        f"{prefix}_corr": float(fit.corr),
        f"{prefix}_r2": float(fit.r2),
        f"{prefix}_mse": float(fit.mse),
        f"{prefix}_alpha": float(fit.alpha),
        f"{prefix}_q_scale": float(fit.q_scale),
    }


def _evidence_row(
    example: Mapping[str, Any],
    *,
    edge_prediction: np.ndarray,
    q_prediction: np.ndarray,
    example_id: int,
) -> dict[str, Any]:
    positions = np.asarray(example["active_positions"], dtype=np.float64)
    graph = example["graph"]
    if not isinstance(graph, Mapping):
        raise TypeError("example graph must be a mapping")
    edges = np.asarray(graph["edges"], dtype=np.int32)
    truth = np.asarray(example["edge_log_rates"], dtype=np.float64)
    q_truth = np.asarray(example["node_log_q"], dtype=np.float64)
    if edge_prediction.shape != truth.shape:
        raise ValueError(f"example {example_id} edge prediction shape does not match truth")
    if q_prediction.shape != q_truth.shape:
        raise ValueError(f"example {example_id} q prediction shape does not match truth")
    empirical, observed_nodes, observed_count = _observed_covariance(
        example,
        num_nodes=positions.shape[0],
    )
    uniform_fit = _fit(
        empirical=empirical,
        observed_nodes=observed_nodes,
        observed_count=observed_count,
        edges=edges,
        edge_logs=np.zeros(edges.shape[0], dtype=np.float64),
        node_log_q=q_prediction,
        num_nodes=positions.shape[0],
    )
    predicted_fit = _fit(
        empirical=empirical,
        observed_nodes=observed_nodes,
        observed_count=observed_count,
        edges=edges,
        edge_logs=metrics.centered(edge_prediction),
        node_log_q=q_prediction,
        num_nodes=positions.shape[0],
    )
    oracle_fit = _fit(
        empirical=empirical,
        observed_nodes=observed_nodes,
        observed_count=observed_count,
        edges=edges,
        edge_logs=metrics.centered(truth),
        node_log_q=q_truth,
        num_nodes=positions.shape[0],
    )
    floor = np.finfo(np.float64).tiny
    log_mse_ratio = math.log(max(uniform_fit.mse, floor) / max(predicted_fit.mse, floor))
    family = str(example.get("surface_kind", "unknown"))
    return {
        "example": int(example_id),
        "family": family,
        "is_null": family in NULL_FAMILIES,
        "sample_design": str(example.get("sample_design", "")),
        "num_nodes": int(positions.shape[0]),
        "num_edges": int(edges.shape[0]),
        "num_observed_nodes": int(observed_nodes.size),
        "sampled_node_fraction": float(observed_nodes.size / positions.shape[0]),
        "num_samples": int(np.asarray(example["sample_nodes"]).size),
        "num_snps": int(example.get("effective_snps", example.get("num_snps", 0))),
        "truth_rms": float(np.sqrt(np.mean(metrics.centered(truth) ** 2))),
        "prediction_rms": float(np.sqrt(np.mean(metrics.centered(edge_prediction) ** 2))),
        "edge_mse": float(metrics.edge_log_metrics(truth, edge_prediction).mse),
        "forward_log_mse_ratio": float(log_mse_ratio),
        "forward_relative_mse_improvement": float(
            (uniform_fit.mse - predicted_fit.mse) / max(uniform_fit.mse, floor)
        ),
        "forward_r2_improvement": float(predicted_fit.r2 - uniform_fit.r2),
        **_fit_fields("uniform", uniform_fit),
        **_fit_fields("predicted", predicted_fit),
        **_fit_fields("oracle", oracle_fit),
    }


def command_pack(args: argparse.Namespace) -> None:
    examples = _load_pickle(args.raw_pack)
    if not isinstance(examples, list):
        raise TypeError(f"expected a list in {args.raw_pack}")
    predictions = _prediction_map(args.predictions)
    rows = []
    for local, example in enumerate(examples):
        example_id = 64 * int(args.pack_id) + local
        if example_id not in predictions:
            raise KeyError(f"prediction {example_id} is missing")
        prediction = predictions[example_id]
        rows.append(
            _evidence_row(
                example,
                edge_prediction=prediction["edge"],
                q_prediction=prediction["q"],
                example_id=example_id,
            )
        )
    if set(predictions) != {int(row["example"]) for row in rows}:
        raise ValueError("raw examples and prediction IDs do not match")
    _write_json(
        args.output,
        {"stage": "forward_null_evidence_pack", "pack_id": args.pack_id, "rows": rows},
    )


def command_exact(args: argparse.Namespace) -> None:
    example = _single_example(args.input)
    graph = example["graph"]
    if not isinstance(graph, Mapping):
        raise TypeError("example graph must be a mapping")
    inference_example = InferenceExample(
        positions=np.asarray(example["active_positions"], dtype=np.float64),
        edges=np.asarray(graph["edges"], dtype=np.int32),
        sample_nodes=np.asarray(example["sample_nodes"], dtype=np.int32),
        genotypes=np.asarray(example["genotypes"], dtype=np.float64),
        name=f"forward_null_exact_{args.example_id}",
    )
    prediction = load_model(args.checkpoint).predict(inference_example)
    if prediction.node_log_q_mean is None:
        raise RuntimeError("checkpoint did not return node q estimates")
    row = _evidence_row(
        example,
        edge_prediction=prediction.edge_log_mean,
        q_prediction=prediction.node_log_q_mean,
        example_id=args.example_id,
    )
    row.update(
        {
            "scenario_truth": str(example.get("scenario_truth", "")),
            "scenario_mask": str(example.get("scenario_mask", "")),
        }
    )
    _write_json(args.output, {"stage": "forward_null_evidence_exact", "row": row})


def _rank_auc(null_scores: np.ndarray, structured_scores: np.ndarray) -> float:
    values = np.concatenate((null_scores, structured_scores))
    ranks = stats.rankdata(values, method="average")
    structured_rank_sum = float(np.sum(ranks[null_scores.size :]))
    count = float(null_scores.size * structured_scores.size)
    return (
        structured_rank_sum - structured_scores.size * (structured_scores.size + 1) / 2.0
    ) / count


def _best_threshold(rows: Sequence[Mapping[str, Any]]) -> float:
    scores = np.asarray([row["forward_log_mse_ratio"] for row in rows], dtype=np.float64)
    labels = np.asarray([not bool(row["is_null"]) for row in rows], dtype=bool)
    unique = np.unique(scores)
    candidates = np.concatenate(
        (
            [np.nextafter(unique[0], -np.inf)],
            (unique[:-1] + unique[1:]) / 2.0,
            [np.nextafter(unique[-1], np.inf)],
        )
    )
    best = None
    for threshold in candidates:
        predicted = scores > threshold
        sensitivity = float(np.mean(predicted[labels]))
        specificity = float(np.mean(~predicted[~labels]))
        candidate = (0.5 * (sensitivity + specificity), specificity, -abs(float(threshold)))
        if best is None or candidate > best[0]:
            best = (candidate, float(threshold))
    if best is None:
        raise RuntimeError("no threshold candidates")
    return best[1]


def _classification(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, float | int]:
    scores = np.asarray([row["forward_log_mse_ratio"] for row in rows], dtype=np.float64)
    labels = np.asarray([not bool(row["is_null"]) for row in rows], dtype=bool)
    predicted = scores > threshold
    return {
        "examples": len(rows),
        "structured_sensitivity": float(np.mean(predicted[labels])),
        "null_specificity": float(np.mean(~predicted[~labels])),
        "balanced_accuracy": float(
            0.5 * (np.mean(predicted[labels]) + np.mean(~predicted[~labels]))
        ),
        "structured_false_negative_count": int(np.sum(~predicted[labels])),
        "null_false_positive_count": int(np.sum(predicted[~labels])),
    }


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scores = np.asarray([row["forward_log_mse_ratio"] for row in rows], dtype=np.float64)
    return {
        "examples": len(rows),
        "forward_log_mse_ratio_mean": float(np.mean(scores)),
        "forward_log_mse_ratio_median": float(np.median(scores)),
        "forward_log_mse_ratio_q05": float(np.quantile(scores, 0.05)),
        "forward_log_mse_ratio_q95": float(np.quantile(scores, 0.95)),
        "uniform_mse_mean": float(np.mean([row["uniform_mse"] for row in rows])),
        "predicted_mse_mean": float(np.mean([row["predicted_mse"] for row in rows])),
        "oracle_mse_mean": float(np.mean([row["oracle_mse"] for row in rows])),
    }


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(json_ready(rows))


def _draw_figure(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    exact: Sequence[Mapping[str, Any]],
    threshold: float,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), constrained_layout=True)
    null = [row for row in rows if bool(row["is_null"])]
    structured = [row for row in rows if not bool(row["is_null"])]
    bins = np.linspace(
        min(float(row["forward_log_mse_ratio"]) for row in rows),
        max(float(row["forward_log_mse_ratio"]) for row in rows),
        40,
    )
    axes[0].hist(
        [row["forward_log_mse_ratio"] for row in null],
        bins=bins,
        alpha=0.65,
        color="#bd5b46",
        label="homogeneous",
    )
    axes[0].hist(
        [row["forward_log_mse_ratio"] for row in structured],
        bins=bins,
        alpha=0.55,
        color="#278a67",
        label="structured",
    )
    axes[0].axvline(threshold, color="#222222", linewidth=1.2, linestyle="--")
    axes[0].set_xlabel("log(uniform MSE / DREEMS MSE)")
    axes[0].set_ylabel("Examples")
    axes[0].set_title("Independent simulations")
    axes[0].legend(frameon=False)

    colors = ["#bd5b46" if row["is_null"] else "#278a67" for row in rows]
    axes[1].scatter(
        [row["uniform_mse"] for row in rows],
        [row["predicted_mse"] for row in rows],
        c=colors,
        s=12,
        alpha=0.65,
    )
    maximum = max(
        max(float(row["uniform_mse"]) for row in rows),
        max(float(row["predicted_mse"]) for row in rows),
    )
    axes[1].plot([0.0, maximum], [0.0, maximum], color="#555555", linewidth=1)
    axes[1].set_xlabel("Uniform forward-fit MSE")
    axes[1].set_ylabel("DREEMS forward-fit MSE")
    axes[1].set_title("Same covariance and fitted q")

    labels = [f"{row['scenario_truth']}\n{row['scenario_mask']}" for row in exact]
    values = [float(row["forward_log_mse_ratio"]) for row in exact]
    axes[2].bar(
        np.arange(len(exact)),
        values,
        color=["#bd5b46" if row["is_null"] else "#278a67" for row in exact],
    )
    axes[2].axhline(threshold, color="#222222", linewidth=1.2, linestyle="--")
    axes[2].set_xticks(np.arange(len(exact)), labels, rotation=55, ha="right")
    axes[2].set_ylabel("log(uniform MSE / DREEMS MSE)")
    axes[2].set_title("Exact notebook cases")

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    figure.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(figure)


def command_aggregate(args: argparse.Namespace) -> None:
    rows = []
    for path in args.packs:
        rows.extend(json.loads(path.read_text())["rows"])
    selected = {
        int(row["original_index"]) for row in json.loads(args.selection.read_text())["examples"]
    }
    before_exclusion = len(rows)
    rows = [row for row in rows if int(row["example"]) not in selected]
    exact = [json.loads(path.read_text())["row"] for path in args.exact]
    rows.sort(key=lambda row: int(row["example"]))
    exact.sort(key=lambda row: int(row["example"]))

    calibration = [row for row in rows if int(row["example"]) % 2 == 0]
    verification = [row for row in rows if int(row["example"]) % 2 == 1]
    threshold = _best_threshold(calibration)
    null_scores = np.asarray(
        [row["forward_log_mse_ratio"] for row in verification if row["is_null"]],
        dtype=np.float64,
    )
    structured_scores = np.asarray(
        [row["forward_log_mse_ratio"] for row in verification if not row["is_null"]],
        dtype=np.float64,
    )
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups["null" if row["is_null"] else "structured"].append(row)
        groups[f"family:{row['family']}"].append(row)
    report = {
        "stage": "forward_null_evidence_audit",
        "interpretation": "positive scores favor the DREEMS heterogeneous surface over uniform conductance",
        "examples_before_gate_selection_exclusion": before_exclusion,
        "gate_selection_examples_excluded": before_exclusion - len(rows),
        "examples_analyzed": len(rows),
        "calibration_examples": len(calibration),
        "verification_examples": len(verification),
        "calibrated_threshold": threshold,
        "verification_auc": _rank_auc(null_scores, structured_scores),
        "calibration_classification": _classification(calibration, threshold),
        "verification_classification": _classification(verification, threshold),
        "summary": {label: _group_summary(group) for label, group in sorted(groups.items())},
        "exact": [
            {
                **row,
                "classified_structured": bool(row["forward_log_mse_ratio"] > threshold),
            }
            for row in exact
        ],
    }
    _write_json(args.output, report)
    _write_rows(args.rows_output, [*rows, *exact])
    _draw_figure(args.figure, rows, exact, threshold)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack = subparsers.add_parser("pack")
    pack.add_argument("--raw-pack", type=Path, required=True)
    pack.add_argument("--predictions", type=Path, required=True)
    pack.add_argument("--pack-id", type=int, required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.set_defaults(function=command_pack)

    exact = subparsers.add_parser("exact")
    exact.add_argument("--input", type=Path, required=True)
    exact.add_argument("--checkpoint", type=Path, required=True)
    exact.add_argument("--example-id", type=int, required=True)
    exact.add_argument("--output", type=Path, required=True)
    exact.set_defaults(function=command_exact)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--packs", type=Path, nargs="+", required=True)
    aggregate.add_argument("--exact", type=Path, nargs="+", required=True)
    aggregate.add_argument("--selection", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--rows-output", type=Path, required=True)
    aggregate.add_argument("--figure", type=Path, required=True)
    aggregate.set_defaults(function=command_aggregate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
