#!/usr/bin/env python3
"""Measure whether a learned surface gate helps on the immutable 512-case H2H."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from dreems.metrics import centered, edge_log_metrics
from dreems.workflow_io import json_ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--example-count", type=int, required=True)
    parser.add_argument("--gated-pattern", required=True)
    parser.add_argument("--baseline-json-pattern", required=True)
    parser.add_argument("--baseline-npz-pattern", required=True)
    parser.add_argument("--feems-json-pattern", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    return parser.parse_args()


def formatted(pattern: str, example: int, suffix: str | None = None) -> Path:
    value = pattern.format(example=example)
    if suffix is not None:
        value += suffix
    return Path(value)


def projected_alpha(baseline: np.ndarray, gated: np.ndarray) -> tuple[float, float]:
    baseline = centered(baseline)
    gated = centered(gated)
    denominator = float(np.dot(baseline, baseline))
    if denominator <= 1e-12:
        return float("nan"), float("nan")
    alpha = float(np.dot(baseline, gated) / denominator)
    residual = gated - alpha * baseline
    relative_residual = float(np.linalg.norm(residual) / max(float(np.linalg.norm(gated)), 1e-12))
    return alpha, relative_residual


def read_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in range(args.example_count):
        gated_json = json.loads(formatted(args.gated_pattern, example, ".json").read_text())
        baseline_json = json.loads(formatted(args.baseline_json_pattern, example).read_text())
        feems_json = json.loads(formatted(args.feems_json_pattern, example).read_text())
        with np.load(formatted(args.gated_pattern, example, ".npz")) as archive:
            gated_truth = np.asarray(archive["truth"], dtype=np.float64)
            gated_neural = np.asarray(archive["dreems"], dtype=np.float64)
            gated_map = np.asarray(archive["dreems_map"], dtype=np.float64)
        with np.load(formatted(args.baseline_npz_pattern, example)) as archive:
            baseline_truth = np.asarray(archive["truth"], dtype=np.float64)
            baseline_neural = np.asarray(archive["dreems"], dtype=np.float64)
            baseline_map = np.asarray(archive["dreems_map"], dtype=np.float64)
        if not np.allclose(gated_truth, baseline_truth, rtol=0.0, atol=1e-10):
            raise ValueError(f"example {example} has mismatched truth arrays")
        if int(gated_json["index"]) != example or int(baseline_json["index"]) != example:
            raise ValueError(f"example {example} has mismatched result metadata")

        alpha, pattern_residual = projected_alpha(baseline_neural, gated_neural)
        before_neural = edge_log_metrics(gated_truth, baseline_neural)
        after_neural = edge_log_metrics(gated_truth, gated_neural)
        before_map = edge_log_metrics(gated_truth, baseline_map)
        after_map = edge_log_metrics(gated_truth, gated_map)
        rows.append(
            {
                "example": example,
                "num_nodes": int(gated_json["num_nodes"]),
                "num_edges": int(gated_json["num_edges"]),
                "num_sampled_nodes": int(gated_json["num_sampled_nodes"]),
                "sampled_node_fraction": float(gated_json["sampled_node_fraction"]),
                "num_snps": int(gated_json["num_snps"]),
                "truth_rms": float(gated_json["truth_edge_log_rms"]),
                "alpha": alpha,
                "gate_pattern_residual": pattern_residual,
                "baseline_neural_mse": before_neural.mse,
                "gated_neural_mse": after_neural.mse,
                "baseline_neural_corr": before_neural.corr,
                "gated_neural_corr": after_neural.corr,
                "baseline_map_mse": before_map.mse,
                "gated_map_mse": after_map.mse,
                "baseline_map_corr": before_map.corr,
                "gated_map_corr": after_map.corr,
                "feems_mse": float(feems_json["feems_edge_log_mse"]),
                "feems_corr": float(feems_json["feems_edge_log_corr"]),
            }
        )
    return rows


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def paired_delta(
    rows: list[dict[str, Any]],
    candidate: str,
    baseline: str,
    *,
    higher_is_better: bool,
) -> dict[str, Any]:
    candidate_values = np.asarray([row[candidate] for row in rows], dtype=np.float64)
    baseline_values = np.asarray([row[baseline] for row in rows], dtype=np.float64)
    finite = np.isfinite(candidate_values) & np.isfinite(baseline_values)
    candidate_values = candidate_values[finite]
    baseline_values = baseline_values[finite]
    advantage = candidate_values - baseline_values
    if not higher_is_better:
        advantage = -advantage
    if advantage.size < 2:
        low = high = p_value = float("nan")
    else:
        standard_error = float(np.std(advantage, ddof=1) / math.sqrt(advantage.size))
        critical = float(stats.t.ppf(0.975, advantage.size - 1))
        mean = float(np.mean(advantage))
        low, high = mean - critical * standard_error, mean + critical * standard_error
        p_value = float(stats.ttest_1samp(advantage, 0.0).pvalue)
    return {
        "n": int(advantage.size),
        "baseline_mean": finite_mean(baseline_values),
        "candidate_mean": finite_mean(candidate_values),
        "mean_advantage": finite_mean(advantage),
        "mean_advantage_ci95_low": low,
        "mean_advantage_ci95_high": high,
        "two_sided_p": p_value,
        "win_fraction": float(np.mean(advantage > 0.0)) if advantage.size else float("nan"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    alpha = np.asarray([row["alpha"] for row in rows], dtype=np.float64)
    residual = np.asarray([row["gate_pattern_residual"] for row in rows], dtype=np.float64)
    finite_alpha = alpha[np.isfinite(alpha)]
    return {
        "examples": len(rows),
        "alpha": {
            "mean": finite_mean(finite_alpha),
            "quantiles": {
                str(quantile): float(np.quantile(finite_alpha, quantile))
                for quantile in (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
            },
            "fraction_below_0.5": float(np.mean(finite_alpha < 0.5)),
            "fraction_below_0.9": float(np.mean(finite_alpha < 0.9)),
            "maximum_pattern_residual": float(np.nanmax(residual)),
        },
        "gated_neural_vs_baseline": {
            "mse": paired_delta(
                rows, "gated_neural_mse", "baseline_neural_mse", higher_is_better=False
            ),
            "corr": paired_delta(
                rows, "gated_neural_corr", "baseline_neural_corr", higher_is_better=True
            ),
        },
        "gated_map_vs_baseline": {
            "mse": paired_delta(rows, "gated_map_mse", "baseline_map_mse", higher_is_better=False),
            "corr": paired_delta(
                rows, "gated_map_corr", "baseline_map_corr", higher_is_better=True
            ),
        },
        "gated_map_vs_feems": {
            "mse": paired_delta(rows, "gated_map_mse", "feems_mse", higher_is_better=False),
            "corr": paired_delta(rows, "gated_map_corr", "feems_corr", higher_is_better=True),
        },
    }


def binned_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    specifications = {
        "sampled_node_fraction": (0.0, 0.05, 0.1, 0.2, 1.01),
        "num_snps": (0, 1_000, 3_000, 10_000, math.inf),
    }
    for key, breaks in specifications.items():
        groups = {}
        for low, high in zip(breaks[:-1], breaks[1:], strict=True):
            selected = [row for row in rows if low <= float(row[key]) < high]
            if selected:
                label = f"[{low:g}, {high:g})"
                groups[label] = summarize(selected)
        result[key] = groups
    return result


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def draw_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    alpha = np.asarray([row["alpha"] for row in rows], dtype=np.float64)
    baseline_mse = np.asarray([row["baseline_map_mse"] for row in rows], dtype=np.float64)
    gated_mse = np.asarray([row["gated_map_mse"] for row in rows], dtype=np.float64)
    fractions = np.asarray([row["sampled_node_fraction"] for row in rows], dtype=np.float64)
    delta = baseline_mse - gated_mse

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    axes[0].hist(alpha[np.isfinite(alpha)], bins=np.linspace(0.0, 1.0, 31), color="#278a67")
    axes[0].axvline(1.0, color="#444444", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Learned surface amplitude")
    axes[0].set_ylabel("Examples")
    axes[0].set_title("Gate on structured landscapes")

    positive = (baseline_mse > 0.0) & (gated_mse > 0.0)
    limit_low = float(min(np.min(baseline_mse[positive]), np.min(gated_mse[positive])))
    limit_high = float(max(np.max(baseline_mse[positive]), np.max(gated_mse[positive])))
    axes[1].scatter(baseline_mse, gated_mse, s=12, alpha=0.5, color="#278a67")
    axes[1].plot([limit_low, limit_high], [limit_low, limit_high], color="#555555", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Baseline MAP edge MSE")
    axes[1].set_ylabel("Gated MAP edge MSE")
    axes[1].set_title("Paired 512-case audit")

    axes[2].scatter(fractions, delta, s=12, alpha=0.5, color="#315c9b")
    axes[2].axhline(0.0, color="#555555", linewidth=1)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Sampled-node fraction")
    axes[2].set_ylabel("Baseline MSE - gated MSE")
    axes[2].set_title("Positive means shrinkage helps")

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    figure.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rows = read_rows(args)
    report = {
        "stage": "broad_null_surface_gate_audit",
        "overall": summarize(rows),
        "strata": binned_summaries(rows),
        "interpretation": (
            "All 512 cases are structured global-landscape simulations. Positive paired "
            "MSE advantage means that the gate improved the estimate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_ready(report), indent=2, sort_keys=True) + "\n")
    write_rows(args.rows_output, rows)
    draw_figure(args.figure, rows)


if __name__ == "__main__":
    main()
