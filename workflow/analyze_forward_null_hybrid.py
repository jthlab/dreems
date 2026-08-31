#!/usr/bin/env python3
"""Combine the learned surface gate with a forward-model guardrail."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from dreems.metrics import edge_log_metrics
from dreems.workflow_io import json_ready


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _paired_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline = np.asarray([row["baseline_mse"] for row in rows], dtype=np.float64)
    gated = np.asarray([row["gated_mse"] for row in rows], dtype=np.float64)
    hybrid = np.asarray([row["hybrid_mse"] for row in rows], dtype=np.float64)
    advantage = baseline - hybrid
    if advantage.size < 2:
        low = high = p_value = float("nan")
    else:
        standard_error = float(np.std(advantage, ddof=1) / math.sqrt(advantage.size))
        critical = float(stats.t.ppf(0.975, advantage.size - 1))
        mean = float(np.mean(advantage))
        low = mean - critical * standard_error
        high = mean + critical * standard_error
        p_value = float(stats.ttest_1samp(advantage, 0.0).pvalue)
    return {
        "examples": len(rows),
        "baseline_mse": float(np.mean(baseline)),
        "learned_gate_mse": float(np.mean(gated)),
        "guarded_gate_mse": float(np.mean(hybrid)),
        "guarded_mse_advantage": float(np.mean(advantage)),
        "guarded_mse_advantage_ci95_low": low,
        "guarded_mse_advantage_ci95_high": high,
        "guarded_mse_advantage_two_sided_p": p_value,
        "guarded_mse_win_fraction": float(np.mean(advantage > 0.0)),
        "forward_guard_restore_fraction": float(
            np.mean([row["forward_guard_restored_baseline"] for row in rows])
        ),
        "learned_alpha_median": float(np.median([row["learned_alpha"] for row in rows])),
        "guarded_alpha_median": float(np.median([row["guarded_alpha"] for row in rows])),
    }


def _independent_rows(
    *,
    forward_rows: Sequence[Mapping[str, str]],
    gate_rows: Sequence[Mapping[str, str]],
    threshold: float,
    uniform_r2_threshold: float | None = None,
) -> list[dict[str, Any]]:
    forward = {
        int(row["example"]): row
        for row in forward_rows
        if not str(row.get("scenario_truth", "")).strip()
    }
    gate = {int(row["example"]): row for row in gate_rows}
    if forward.keys() != gate.keys():
        missing_forward = sorted(gate.keys() - forward.keys())
        missing_gate = sorted(forward.keys() - gate.keys())
        raise ValueError(
            "forward and learned-gate examples differ: "
            f"missing_forward={missing_forward[:8]}, missing_gate={missing_gate[:8]}"
        )
    rows = []
    for example in sorted(forward):
        forward_row = forward[example]
        gate_row = gate[example]
        score = float(forward_row["forward_log_mse_ratio"])
        uniform_r2 = float(forward_row["uniform_r2"])
        relative_guard = score > threshold
        absolute_fit_guard = uniform_r2_threshold is not None and (
            not np.isfinite(uniform_r2) or uniform_r2 < uniform_r2_threshold
        )
        restore = relative_guard or absolute_fit_guard
        learned_alpha = float(gate_row["alpha"])
        rows.append(
            {
                "dataset": "independent",
                "example": example,
                "family": gate_row["family"],
                "is_null": gate_row["is_null"].lower() == "true",
                "forward_log_mse_ratio": score,
                "uniform_r2": uniform_r2,
                "relative_evidence_guard": relative_guard,
                "absolute_fit_guard": absolute_fit_guard,
                "forward_guard_restored_baseline": restore,
                "learned_alpha": learned_alpha,
                "guarded_alpha": 1.0 if restore else learned_alpha,
                "baseline_mse": float(gate_row["baseline_mse"]),
                "gated_mse": float(gate_row["gated_mse"]),
                "hybrid_mse": float(gate_row["baseline_mse"] if restore else gate_row["gated_mse"]),
            }
        )
    return rows


def _exact_rows(
    *,
    exact_evidence: Sequence[Mapping[str, Any]],
    baseline_paths: Sequence[Path],
    gated_paths: Sequence[Path],
    threshold: float,
    uniform_r2_threshold: float | None = None,
) -> list[dict[str, Any]]:
    if len(exact_evidence) != len(baseline_paths) or len(exact_evidence) != len(gated_paths):
        raise ValueError("exact evidence and prediction path counts differ")
    rows = []
    for evidence, baseline_path, gated_path in zip(
        exact_evidence, baseline_paths, gated_paths, strict=True
    ):
        with np.load(baseline_path) as baseline_archive:
            truth = np.asarray(baseline_archive["truth"], dtype=np.float64)
            baseline = np.asarray(baseline_archive["dreems"], dtype=np.float64)
            baseline_map = np.asarray(baseline_archive["dreems_map"], dtype=np.float64)
        with np.load(gated_path) as gated_archive:
            np.testing.assert_allclose(truth, gated_archive["truth"], rtol=0.0, atol=1e-10)
            gated = np.asarray(gated_archive["dreems"], dtype=np.float64)
            gated_map = np.asarray(gated_archive["dreems_map"], dtype=np.float64)
        score = float(evidence["forward_log_mse_ratio"])
        uniform_r2 = float(evidence["uniform_r2"])
        relative_guard = score > threshold
        absolute_fit_guard = uniform_r2_threshold is not None and (
            not np.isfinite(uniform_r2) or uniform_r2 < uniform_r2_threshold
        )
        restore = relative_guard or absolute_fit_guard
        denominator = float(np.dot(baseline, baseline))
        learned_alpha = (
            float(np.dot(baseline, gated) / denominator) if denominator > 1e-12 else float("nan")
        )
        hybrid = baseline if restore else gated
        hybrid_map = baseline_map if restore else gated_map
        family = str(evidence["scenario_truth"])
        rows.append(
            {
                "dataset": "exact_notebook",
                "example": int(evidence["example"]),
                "family": family,
                "mask": str(evidence["scenario_mask"]),
                "is_null": bool(evidence["is_null"]),
                "forward_log_mse_ratio": score,
                "uniform_r2": uniform_r2,
                "relative_evidence_guard": relative_guard,
                "absolute_fit_guard": absolute_fit_guard,
                "forward_guard_restored_baseline": restore,
                "learned_alpha": learned_alpha,
                "guarded_alpha": 1.0 if restore else learned_alpha,
                "baseline_mse": edge_log_metrics(truth, baseline).mse,
                "gated_mse": edge_log_metrics(truth, gated).mse,
                "hybrid_mse": edge_log_metrics(truth, hybrid).mse,
                "baseline_map_mse": edge_log_metrics(truth, baseline_map).mse,
                "gated_map_mse": edge_log_metrics(truth, gated_map).mse,
                "hybrid_map_mse": edge_log_metrics(truth, hybrid_map).mse,
            }
        )
    return rows


def _summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups["null" if row["is_null"] else "structured"].append(row)
        groups[f"family:{row['family']}"].append(row)
    return {label: _paired_summary(group) for label, group in sorted(groups.items())}


def _exact_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups["null" if row["is_null"] else "structured"].append(row)
        groups[f"family:{row['family']}"].append(row)
    result = {}
    for label, group in sorted(groups.items()):
        result[label] = {
            **_paired_summary(group),
            "baseline_map_mse": float(np.mean([row["baseline_map_mse"] for row in group])),
            "learned_gate_map_mse": float(np.mean([row["gated_map_mse"] for row in group])),
            "guarded_gate_map_mse": float(np.mean([row["hybrid_map_mse"] for row in group])),
        }
    return result


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(json_ready(rows))


def _draw_figure(
    path: Path,
    independent: Sequence[Mapping[str, Any]],
    exact: Sequence[Mapping[str, Any]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    methods = ("baseline", "gated", "hybrid")
    method_labels = ("baseline", "learned gate", "guarded gate")
    colors = ("#8f8f8f", "#278a67", "#2568a6")
    x = np.arange(2)
    width = 0.24
    for method_index, (method, label, color) in enumerate(
        zip(methods, method_labels, colors, strict=True)
    ):
        values = []
        for is_null in (True, False):
            selected = [row for row in independent if row["is_null"] is is_null]
            values.append(float(np.mean([row[f"{method}_mse"] for row in selected])))
        axes[0].bar(x + (method_index - 1) * width, values, width, color=color, label=label)
    axes[0].set_xticks(x, ("homogeneous", "structured"))
    axes[0].set_ylabel("Mean edge MSE")
    axes[0].set_title("Independent simulations")
    axes[0].legend(frameon=False)

    labels = [f"{row['family']}\n{row['mask']}" for row in exact]
    exact_x = np.arange(len(exact))
    for method_index, (method, label, color) in enumerate(
        zip(methods, method_labels, colors, strict=True)
    ):
        axes[1].bar(
            exact_x + (method_index - 1) * width,
            [row[f"{method}_mse"] for row in exact],
            width,
            color=color,
            label=label,
        )
    axes[1].set_xticks(exact_x, labels, rotation=55, ha="right")
    axes[1].set_ylabel("Edge MSE")
    axes[1].set_title("Exact notebook guardrail")

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    figure.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-report", type=Path, required=True)
    parser.add_argument("--forward-rows", type=Path, required=True)
    parser.add_argument("--gate-rows", type=Path, required=True)
    parser.add_argument("--baseline-exact", type=Path, nargs="+", required=True)
    parser.add_argument("--gated-exact", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument(
        "--uniform-r2-null-quantile",
        type=float,
        default=None,
        help=(
            "Also restore the baseline when uniform-conductance R2 falls below "
            "this quantile of calibration-set null examples"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    forward_report = json.loads(args.forward_report.read_text())
    threshold = float(forward_report["calibrated_threshold"])
    forward_rows = _read_csv(args.forward_rows)
    uniform_r2_threshold = None
    if args.uniform_r2_null_quantile is not None:
        quantile = args.uniform_r2_null_quantile
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("uniform R2 null quantile must lie between zero and one")
        calibration_null_r2 = np.asarray(
            [
                float(row["uniform_r2"])
                for row in forward_rows
                if not str(row.get("scenario_truth", "")).strip()
                and row["is_null"].lower() == "true"
                and int(row["example"]) % 2 == 0
                and np.isfinite(float(row["uniform_r2"]))
            ],
            dtype=np.float64,
        )
        if calibration_null_r2.size == 0:
            raise ValueError("no finite calibration-set null R2 values were found")
        uniform_r2_threshold = float(np.quantile(calibration_null_r2, quantile))
    independent = _independent_rows(
        forward_rows=forward_rows,
        gate_rows=_read_csv(args.gate_rows),
        threshold=threshold,
        uniform_r2_threshold=uniform_r2_threshold,
    )
    exact = _exact_rows(
        exact_evidence=forward_report["exact"],
        baseline_paths=args.baseline_exact,
        gated_paths=args.gated_exact,
        threshold=threshold,
        uniform_r2_threshold=uniform_r2_threshold,
    )
    rule = "use the learned gate only when relative evidence does not favor structure"
    if uniform_r2_threshold is not None:
        rule += " and uniform conductance passes the absolute-fit guard"
    report = {
        "stage": "guarded_null_surface_gate_audit",
        "rule": rule,
        "forward_log_mse_ratio_threshold": threshold,
        "uniform_r2_null_quantile": args.uniform_r2_null_quantile,
        "uniform_r2_threshold": uniform_r2_threshold,
        "independent": _summaries(independent),
        "independent_calibration": _summaries(
            [row for row in independent if row["example"] % 2 == 0]
        ),
        "independent_verification": _summaries(
            [row for row in independent if row["example"] % 2 == 1]
        ),
        "exact_notebook": _exact_summary(exact),
        "exact_rows": exact,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_ready(report), indent=2, sort_keys=True) + "\n")
    _write_rows(args.rows_output, [*independent, *exact])
    _draw_figure(args.figure, independent, exact)


if __name__ == "__main__":
    main()
