#!/usr/bin/env python3
"""Audit a surface gate on test examples untouched by gate model selection."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from dreems.metrics import centered, edge_log_metrics
from dreems.workflow_io import json_ready

NULL_FAMILIES = {"constant", "feems_homogeneous"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--gated", type=Path, nargs="+", required=True)
    parser.add_argument("--metadata", type=Path, nargs="+", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--included-packs", type=int, nargs="+", required=True)
    parser.add_argument("--excluded-replay-packs", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    return parser.parse_args()


def load_predictions(paths: list[Path]) -> dict[int, dict[str, np.ndarray]]:
    result: dict[int, dict[str, np.ndarray]] = {}
    for path in paths:
        with np.load(path) as archive:
            replicates = np.asarray(archive["replicate"], dtype=np.int64)
            offsets = np.asarray(archive["edge_offsets"], dtype=np.int64)
            truth = np.asarray(archive["edge_truth"], dtype=np.float64)
            mean = np.asarray(archive["edge_mean"], dtype=np.float64)
        for local, replicate in enumerate(replicates):
            start, stop = int(offsets[local]), int(offsets[local + 1])
            result[int(replicate)] = {
                "truth": truth[start:stop],
                "mean": mean[start:stop],
            }
    return result


def load_metadata(paths: list[Path], pack_ids: list[int]) -> dict[int, dict[str, Any]]:
    if len(paths) != len(pack_ids):
        raise ValueError("metadata paths and pack IDs must have the same length")
    result: dict[int, dict[str, Any]] = {}
    for path, pack_id in zip(paths, pack_ids, strict=True):
        payload = json.loads(path.read_text())
        for row in payload["rows"]:
            # Prepared-pack row indices are local to each 64-example raw pack.
            index = 64 * pack_id + int(row["index"])
            if index in result:
                raise ValueError(f"duplicate metadata index {index}")
            result[index] = {**row, "index": index}
    return result


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


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], int]:
    baseline = load_predictions(args.baseline)
    gated = load_predictions(args.gated)
    metadata = load_metadata(args.metadata, args.included_packs)
    selection = json.loads(args.selection.read_text())
    selected_indices = {int(row["original_index"]) for row in selection["examples"]}
    if baseline.keys() != gated.keys() or baseline.keys() != metadata.keys():
        raise ValueError("baseline, gated, and metadata example IDs do not match")

    rows = []
    excluded = 0
    for index in sorted(metadata):
        if index in selected_indices:
            excluded += 1
            continue
        before = baseline[index]
        after = gated[index]
        if not np.allclose(before["truth"], after["truth"], rtol=0.0, atol=1e-10):
            raise ValueError(f"example {index} has mismatched truth arrays")
        family = str(metadata[index]["surface_kind"])
        before_metrics = edge_log_metrics(before["truth"], before["mean"])
        after_metrics = edge_log_metrics(after["truth"], after["mean"])
        alpha, pattern_residual = projected_alpha(before["mean"], after["mean"])
        rows.append(
            {
                "example": index,
                "family": family,
                "is_null": family in NULL_FAMILIES,
                "sample_design": str(metadata[index]["sample_design"]),
                "num_nodes": int(metadata[index]["num_nodes"]),
                "num_sampled_nodes": int(metadata[index]["num_sampled_nodes"]),
                "sampled_node_fraction": float(metadata[index]["sampled_node_fraction"]),
                "num_snps": int(metadata[index]["num_snps"]),
                "truth_rms": float(np.sqrt(np.mean(centered(before["truth"]) ** 2))),
                "alpha": alpha,
                "gate_pattern_residual": pattern_residual,
                "baseline_mse": before_metrics.mse,
                "gated_mse": after_metrics.mse,
                "baseline_corr": before_metrics.corr,
                "gated_corr": after_metrics.corr,
            }
        )
    return rows, excluded


def paired_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    before = np.asarray([row["baseline_mse"] for row in rows], dtype=np.float64)
    after = np.asarray([row["gated_mse"] for row in rows], dtype=np.float64)
    advantage = before - after
    if advantage.size < 2:
        low = high = p_value = float("nan")
    else:
        standard_error = float(np.std(advantage, ddof=1) / math.sqrt(advantage.size))
        critical = float(stats.t.ppf(0.975, advantage.size - 1))
        mean = float(np.mean(advantage))
        low, high = mean - critical * standard_error, mean + critical * standard_error
        p_value = float(stats.ttest_1samp(advantage, 0.0).pvalue)
    alpha = np.asarray([row["alpha"] for row in rows], dtype=np.float64)
    return {
        "examples": len(rows),
        "baseline_mse": float(np.mean(before)),
        "gated_mse": float(np.mean(after)),
        "mean_mse_advantage": float(np.mean(advantage)),
        "mean_mse_advantage_ci95_low": low,
        "mean_mse_advantage_ci95_high": high,
        "mse_advantage_two_sided_p": p_value,
        "mse_win_fraction": float(np.mean(advantage > 0.0)),
        "median_alpha": float(np.nanmedian(alpha)),
        "alpha_q05": float(np.nanquantile(alpha, 0.05)),
        "alpha_q95": float(np.nanquantile(alpha, 0.95)),
        "fraction_alpha_below_0.5": float(np.mean(alpha < 0.5)),
        "fraction_alpha_below_0.9": float(np.mean(alpha < 0.9)),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups["null" if row["is_null"] else "structured"].append(row)
        groups[f"family:{row['family']}"].append(row)
    return {label: paired_summary(selected) for label, selected in sorted(groups.items())}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    colors = {"null": "#bd5b46", "structured": "#278a67"}
    for label, is_null in (("null", True), ("structured", False)):
        selected = [row for row in rows if row["is_null"] is is_null]
        axes[0].hist(
            [row["alpha"] for row in selected],
            bins=np.linspace(0.0, 1.0, 31),
            alpha=0.6,
            label=label,
            color=colors[label],
        )
    axes[0].set_xlabel("Learned surface amplitude")
    axes[0].set_ylabel("Examples")
    axes[0].set_title("Independent test examples")
    axes[0].legend(frameon=False)

    categories = ["null", "structured"]
    x = np.arange(2)
    width = 0.36
    for offset, method, color in (
        (-0.5, "baseline", "#8f8f8f"),
        (0.5, "gated", "#278a67"),
    ):
        values = []
        for category in categories:
            selected = [
                row for row in rows if ("null" if row["is_null"] else "structured") == category
            ]
            values.append(np.mean([row[f"{method}_mse"] for row in selected]))
        axes[1].bar(x + offset * width, values, width, color=color, label=method)
    axes[1].set_xticks(x, categories)
    axes[1].set_ylabel("Mean edge MSE")
    axes[1].set_title("Null gain versus structured cost")
    axes[1].legend(frameon=False)

    anisotropy = [row for row in rows if row["family"] == "feems_anisotropic"]
    axes[2].scatter(
        [row["baseline_mse"] for row in anisotropy],
        [row["gated_mse"] for row in anisotropy],
        c=[row["alpha"] for row in anisotropy],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=25,
    )
    limits = axes[2].get_xlim()
    axes[2].plot(limits, limits, color="#555555", linewidth=1)
    axes[2].set_xlim(limits)
    axes[2].set_ylim(limits)
    axes[2].set_xlabel("Baseline anisotropy MSE")
    axes[2].set_ylabel("Gated anisotropy MSE")
    axes[2].set_title("Anisotropy guardrail")

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    figure.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rows, excluded = build_rows(args)
    report = {
        "stage": "independent_null_surface_gate_audit",
        "included_prepared_packs": args.included_packs,
        "excluded_replay_validation_packs": args.excluded_replay_packs,
        "examples_before_selection_exclusion": len(rows) + excluded,
        "gate_selection_examples_excluded": excluded,
        "examples_analyzed": len(rows),
        "maximum_pattern_residual": float(
            np.nanmax([row["gate_pattern_residual"] for row in rows])
        ),
        "summary": summarize(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_ready(report), indent=2, sort_keys=True) + "\n")
    write_rows(args.rows_output, rows)
    draw_figure(args.figure, rows)


if __name__ == "__main__":
    main()
