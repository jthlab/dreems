#!/usr/bin/env python3
"""Audit a trained graph-level surface gate against its unchanged parent model."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from dreems.metrics import centered
from dreems.workflow_io import json_ready

NULL_FAMILIES = {"constant", "feems_homogeneous", "homogeneous"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--baseline-validation", type=Path, nargs="+", required=True)
    parser.add_argument("--gated-validation", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-exact", type=Path, nargs="+", required=True)
    parser.add_argument("--gated-exact", type=Path, nargs="+", required=True)
    parser.add_argument("--gated-exact-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    return parser.parse_args()


def load_prediction_archives(paths: list[Path]) -> dict[int, dict[str, np.ndarray]]:
    result: dict[int, dict[str, np.ndarray]] = {}
    for path in paths:
        with np.load(path) as archive:
            replicates = np.asarray(archive["replicate"], dtype=np.int64)
            offsets = np.asarray(archive["edge_offsets"], dtype=np.int64)
            truth = np.asarray(archive["edge_truth"], dtype=np.float64)
            mean = np.asarray(archive["edge_mean"], dtype=np.float64)
            sd = np.asarray(archive["edge_sd"], dtype=np.float64)
        for index, replicate in enumerate(replicates):
            start, stop = int(offsets[index]), int(offsets[index + 1])
            result[int(replicate)] = {
                "truth": truth[start:stop],
                "mean": mean[start:stop],
                "sd": sd[start:stop],
            }
    return result


def projected_alpha(baseline: np.ndarray, candidate: np.ndarray) -> float:
    baseline = centered(np.asarray(baseline, dtype=np.float64))
    candidate = centered(np.asarray(candidate, dtype=np.float64))
    denominator = float(np.dot(baseline, baseline))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.dot(baseline, candidate) / denominator)


def mse(truth: np.ndarray, estimate: np.ndarray) -> float:
    residual = centered(np.asarray(estimate)) - centered(np.asarray(truth))
    return float(np.mean(residual * residual))


def validation_rows(
    baseline_paths: list[Path],
    gated_paths: list[Path],
    selection_path: Path,
) -> list[dict[str, Any]]:
    baseline = load_prediction_archives(baseline_paths)
    gated = load_prediction_archives(gated_paths)
    selection = json.loads(selection_path.read_text())
    family_by_group = {
        int(example["slot"]): str(example["surface_kind"]) for example in selection["examples"]
    }
    rows = []
    for replicate in sorted(set(baseline) & set(gated)):
        group = replicate // 4
        family = family_by_group[group]
        before = baseline[replicate]
        after = gated[replicate]
        rows.append(
            {
                "dataset": "counterfactual_validation",
                "replicate": replicate,
                "group": group,
                "mask": replicate % 4,
                "family": family,
                "is_null": family in NULL_FAMILIES,
                "baseline_mse": mse(before["truth"], before["mean"]),
                "gated_mse": mse(after["truth"], after["mean"]),
                "alpha": projected_alpha(before["mean"], after["mean"]),
            }
        )
    return rows


def exact_rows(
    baseline_paths: list[Path],
    gated_paths: list[Path],
    gated_json_paths: list[Path],
) -> list[dict[str, Any]]:
    metadata = {
        int(json.loads(path.read_text())["index"]): json.loads(path.read_text())
        for path in gated_json_paths
    }
    rows = []
    for index, (baseline_path, gated_path) in enumerate(
        zip(baseline_paths, gated_paths, strict=True)
    ):
        with np.load(baseline_path) as archive:
            truth = np.asarray(archive["truth"], dtype=np.float64)
            baseline_neural = np.asarray(archive["dreems"], dtype=np.float64)
            baseline_map = np.asarray(archive["dreems_map"], dtype=np.float64)
        with np.load(gated_path) as archive:
            gated_neural = np.asarray(archive["dreems"], dtype=np.float64)
            gated_map = np.asarray(archive["dreems_map"], dtype=np.float64)
        row_metadata = metadata[index]
        rows.append(
            {
                "dataset": "exact_notebook",
                "replicate": index,
                "group": index // 4,
                "mask": index % 4,
                "family": str(row_metadata["scenario_truth"]),
                "sample_design": str(row_metadata["scenario_mask"]),
                "is_null": str(row_metadata["scenario_truth"]) == "homogeneous",
                "baseline_mse": mse(truth, baseline_neural),
                "gated_mse": mse(truth, gated_neural),
                "baseline_map_mse": mse(truth, baseline_map),
                "gated_map_mse": mse(truth, gated_map),
                "alpha": projected_alpha(baseline_neural, gated_neural),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = "null" if row["is_null"] else "structured"
        groups[label].append(row)
        groups[f"family:{row['family']}"].append(row)
    for label, selected in sorted(groups.items()):
        summary = {
            "examples": len(selected),
            "baseline_mse": float(np.mean([row["baseline_mse"] for row in selected])),
            "gated_mse": float(np.mean([row["gated_mse"] for row in selected])),
            "median_alpha": float(np.nanmedian([row["alpha"] for row in selected])),
        }
        if all("baseline_map_mse" in row for row in selected):
            summary["baseline_map_mse"] = float(
                np.mean([row["baseline_map_mse"] for row in selected])
            )
            summary["gated_map_mse"] = float(np.mean([row["gated_map_mse"] for row in selected]))
        summaries[label] = summary
    return summaries


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def draw_figure(
    path: Path,
    validation: list[dict[str, Any]],
    exact: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    colors = {"baseline": "#8f8f8f", "gated": "#278a67"}

    categories = ["null", "structured"]
    x = np.arange(len(categories))
    width = 0.36
    for offset, method in ((-0.5, "baseline"), (0.5, "gated")):
        values = []
        for category in categories:
            selected = [
                row
                for row in validation
                if ("null" if row["is_null"] else "structured") == category
            ]
            values.append(np.mean([row[f"{method}_mse"] for row in selected]))
        axes[0].bar(x + offset * width, values, width, label=method, color=colors[method])
    axes[0].set_xticks(x, categories)
    axes[0].set_ylabel("Mean edge MSE")
    axes[0].set_title("Counterfactual validation")
    axes[0].legend(frameon=False)

    families = sorted({str(row["family"]) for row in validation})
    alpha_values = [
        [row["alpha"] for row in validation if row["family"] == family] for family in families
    ]
    axes[1].boxplot(alpha_values, tick_labels=families, showfliers=False)
    axes[1].axhline(1.0, color="#777777", linewidth=1, linestyle="--")
    axes[1].set_ylim(-0.05, 1.08)
    axes[1].set_ylabel("Learned surface amplitude")
    axes[1].set_title("Gate by generating family")
    axes[1].tick_params(axis="x", rotation=55)

    exact_families = ["barrier", "anisotropy", "homogeneous"]
    x = np.arange(len(exact_families))
    for offset, method in ((-0.5, "baseline_map"), (0.5, "gated_map")):
        values = [
            np.mean([row[f"{method}_mse"] for row in exact if row["family"] == family])
            for family in exact_families
        ]
        axes[2].bar(
            x + offset * width,
            values,
            width,
            label=method.replace("_map", " + MAP"),
            color=colors[method.removesuffix("_map")],
        )
    axes[2].set_xticks(x, exact_families)
    axes[2].set_ylabel("Mean edge MSE")
    axes[2].set_title("Exact notebook scenarios")
    axes[2].legend(frameon=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    figure.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    validation = validation_rows(
        args.baseline_validation,
        args.gated_validation,
        args.selection,
    )
    exact = exact_rows(args.baseline_exact, args.gated_exact, args.gated_exact_json)
    training = json.loads(args.training_summary.read_text())
    report = {
        "stage": "null_surface_gate_audit",
        "training": {
            key: training.get(key)
            for key in (
                "best_step",
                "best_validation_loss",
                "final_loss",
                "stopped_early",
                "stop_reason",
            )
        },
        "counterfactual_validation": summarize(validation),
        "exact_notebook": summarize(exact),
        "interpretation_guardrail": (
            "The gate can change only global centered-surface amplitude; it cannot alter "
            "edge ordering or spatial pattern."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_ready(report), indent=2, sort_keys=True) + "\n")
    write_rows(args.rows_output, validation + exact)
    draw_figure(args.figure, validation, exact)


if __name__ == "__main__":
    main()
