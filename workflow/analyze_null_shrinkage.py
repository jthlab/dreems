"""Select and audit graph-level shrinkage toward a constant surface."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dreems.metrics import edge_log_metrics
from dreems.null_shrinkage import NullShrinkageRule, centered, mask_stability, rms
from dreems.workflow_io import json_ready, write_json

NULL_SURFACES = {"constant", "feems_homogeneous"}


@dataclass(frozen=True)
class GroupPrediction:
    group: int
    surface_kind: str
    family: str
    effective_snps: int
    sampled_node_fraction: float
    truth: np.ndarray
    edge_mean: np.ndarray
    edge_sd: np.ndarray
    stability: float

    @property
    def is_null(self) -> bool:
        return self.surface_kind in NULL_SURFACES


def surface_family(surface_kind: str) -> str:
    if surface_kind in NULL_SURFACES:
        return "null"
    if surface_kind in {"barrier", "feems_heterogeneous_barrier"}:
        return "barrier"
    if surface_kind == "feems_anisotropic":
        return "anisotropy"
    return "broad"


def array_slices(
    archive: np.lib.npyio.NpzFile,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    replicates = np.asarray(archive["replicate"], dtype=np.int64)
    offsets = np.asarray(archive["edge_offsets"], dtype=np.int64)
    if offsets.shape != (replicates.size + 1,):
        raise ValueError("prediction offsets do not align with replicate IDs")
    result: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for index, replicate in enumerate(replicates):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        result[int(replicate)] = (
            np.asarray(archive["edge_truth"][start:stop], dtype=np.float64),
            np.asarray(archive["edge_mean"][start:stop], dtype=np.float64),
            np.asarray(archive["edge_sd"][start:stop], dtype=np.float64),
        )
    return result


def load_group_predictions(
    prediction_paths: list[Path],
    summary_paths: list[Path],
    selection_path: Path,
) -> list[GroupPrediction]:
    if len(prediction_paths) != len(summary_paths):
        raise ValueError("prediction and summary paths must align")
    selection = json.loads(selection_path.read_text())["examples"]
    grouped: dict[int, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]]] = {}
    for prediction_path, summary_path in zip(prediction_paths, summary_paths, strict=True):
        summary = json.loads(summary_path.read_text())
        rows = {int(row["replicate"]): row for row in summary["rows"]}
        with np.load(prediction_path, allow_pickle=False) as archive:
            predictions = array_slices(archive)
        if predictions.keys() != rows.keys():
            raise ValueError(f"prediction IDs do not match summary rows in {prediction_path}")
        for replicate, arrays in predictions.items():
            row = rows[replicate]
            group = int(row["counterfactual_group"])
            mask = int(row["counterfactual_mask"])
            grouped.setdefault(group, {})[mask] = (*arrays, row)

    result: list[GroupPrediction] = []
    for group, masks in sorted(grouped.items()):
        if set(masks) != {0, 1, 2, 3}:
            raise ValueError(f"counterfactual group {group} does not contain four masks")
        anchor_truth, anchor_mean, anchor_sd, anchor_row = masks[0]
        for mask in (1, 2, 3):
            truth, mean, _sd, _row = masks[mask]
            np.testing.assert_allclose(truth, anchor_truth, atol=0.0, rtol=0.0)
            if mean.shape != anchor_mean.shape:
                raise ValueError(f"counterfactual group {group} changes edge count")
        metadata = selection[group]
        kind = str(metadata["surface_kind"])
        result.append(
            GroupPrediction(
                group=group,
                surface_kind=kind,
                family=surface_family(kind),
                effective_snps=int(metadata["effective_snps"]),
                sampled_node_fraction=float(anchor_row["sampled_node_fraction"]),
                truth=centered(anchor_truth),
                edge_mean=centered(anchor_mean),
                edge_sd=np.asarray(anchor_sd, dtype=np.float64),
                stability=mask_stability(anchor_mean, [masks[index][1] for index in (1, 2, 3)]),
            )
        )
    return result


def stratified_split(
    examples: list[GroupPrediction],
    *,
    seed: int,
) -> dict[int, str]:
    rng = np.random.default_rng(seed)
    result: dict[int, str] = {}
    for family in sorted({example.family for example in examples}):
        indices = np.asarray(
            [example.group for example in examples if example.family == family],
            dtype=np.int64,
        )
        rng.shuffle(indices)
        midpoint = int(math.ceil(indices.size / 2.0))
        result.update({int(index): "selection" for index in indices[:midpoint]})
        result.update({int(index): "verification" for index in indices[midpoint:]})
    return result


def fit_sd_scale(examples: list[GroupPrediction]) -> float:
    scores = []
    for example in examples:
        finite = np.isfinite(example.edge_sd) & (example.edge_sd > 1e-8)
        if not np.any(finite):
            continue
        error = example.truth[finite] - example.edge_mean[finite]
        scores.append(float(np.mean((error / example.edge_sd[finite]) ** 2)))
    if not scores:
        return 1.0
    return float(np.sqrt(np.mean(scores)))


def quadratic_terms(example: GroupPrediction) -> tuple[float, float, float]:
    truth = centered(example.truth)
    prediction = centered(example.edge_mean)
    return (
        float(np.mean(truth * truth)),
        float(np.mean(prediction * prediction)),
        float(np.mean(truth * prediction)),
    )


def mse_at_alpha(example: GroupPrediction, alpha: float) -> float:
    truth_ss, prediction_ss, cross = quadratic_terms(example)
    return truth_ss + alpha * alpha * prediction_ss - 2.0 * alpha * cross


def rule_metrics(
    examples: list[GroupPrediction],
    rule: NullShrinkageRule,
) -> dict[str, float | int]:
    rows = []
    for example in examples:
        alpha = rule.alpha(
            example.edge_mean,
            example.edge_sd,
            stability=example.stability if rule.stability_weight > 0.0 else None,
        )
        rows.append((example.is_null, mse_at_alpha(example, alpha), alpha))
    result: dict[str, float | int] = {"examples": len(rows)}
    for label, is_null in (("null", True), ("nonnull", False), ("all", None)):
        selected = [row for row in rows if is_null is None or row[0] is is_null]
        result[f"{label}_examples"] = len(selected)
        result[f"{label}_mse"] = float(np.mean([row[1] for row in selected]))
        result[f"{label}_mean_alpha"] = float(np.mean([row[2] for row in selected]))
        result[f"{label}_median_alpha"] = float(np.median([row[2] for row in selected]))
    return result


def scalar_arrays(
    examples: list[GroupPrediction],
    *,
    sd_scale: float,
) -> dict[str, np.ndarray]:
    truth_ss = []
    prediction_ss = []
    cross = []
    log_ratio = []
    for example in examples:
        local_truth_ss, local_prediction_ss, local_cross = quadratic_terms(example)
        truth_ss.append(local_truth_ss)
        prediction_ss.append(local_prediction_ss)
        cross.append(local_cross)
        sd_rms = float(np.sqrt(np.mean(example.edge_sd**2))) * sd_scale
        log_ratio.append(
            float(np.log(max(math.sqrt(local_prediction_ss) / max(sd_rms, 1e-8), 1e-8)))
        )
    return {
        "is_null": np.asarray([example.is_null for example in examples], dtype=bool),
        "stability": np.asarray([example.stability for example in examples], dtype=np.float64),
        "truth_ss": np.asarray(truth_ss, dtype=np.float64),
        "prediction_ss": np.asarray(prediction_ss, dtype=np.float64),
        "cross": np.asarray(cross, dtype=np.float64),
        "log_ratio": np.asarray(log_ratio, dtype=np.float64),
    }


def vector_rule_metrics(
    arrays: dict[str, np.ndarray],
    rule: NullShrinkageRule,
) -> dict[str, float | int]:
    score = (
        rule.stability_weight * arrays["stability"]
        + (1.0 - rule.stability_weight) * arrays["log_ratio"]
    )
    logits = np.clip(rule.slope * (score - rule.threshold), -60.0, 60.0)
    probability = 1.0 / (1.0 + np.exp(-logits))
    alpha = rule.minimum_alpha + (1.0 - rule.minimum_alpha) * probability
    mse = (
        arrays["truth_ss"] + alpha * alpha * arrays["prediction_ss"] - 2.0 * alpha * arrays["cross"]
    )
    result: dict[str, float | int] = {"examples": int(mse.size)}
    for label, mask in (
        ("null", arrays["is_null"]),
        ("nonnull", ~arrays["is_null"]),
        ("all", np.ones_like(arrays["is_null"], dtype=bool)),
    ):
        result[f"{label}_examples"] = int(np.sum(mask))
        result[f"{label}_mse"] = float(np.mean(mse[mask]))
        result[f"{label}_mean_alpha"] = float(np.mean(alpha[mask]))
        result[f"{label}_median_alpha"] = float(np.median(alpha[mask]))
    return result


def baseline_metrics(examples: list[GroupPrediction]) -> dict[str, float | int]:
    return rule_metrics(
        examples,
        NullShrinkageRule(stability_weight=0.0, threshold=-100.0, slope=1.0),
    )


def candidate_rules(
    examples: list[GroupPrediction],
    *,
    sd_scale: float,
    stability_weights: tuple[float, ...],
) -> list[NullShrinkageRule]:
    rules: list[NullShrinkageRule] = []
    for weight in stability_weights:
        raw_scores = []
        probe = NullShrinkageRule(
            stability_weight=weight,
            threshold=0.0,
            slope=1.0,
            sd_scale=sd_scale,
        )
        for example in examples:
            raw_scores.append(
                probe.score(
                    example.edge_mean,
                    example.edge_sd,
                    stability=example.stability if weight > 0.0 else None,
                )
            )
        low, high = np.quantile(np.asarray(raw_scores), [0.02, 0.98])
        thresholds = np.linspace(float(low) - 0.2, float(high) + 0.2, 61)
        for threshold in thresholds:
            for slope in (2.0, 4.0, 8.0, 16.0, 32.0):
                for minimum_alpha in (0.0, 0.05, 0.10, 0.25):
                    rules.append(
                        NullShrinkageRule(
                            stability_weight=weight,
                            threshold=float(threshold),
                            slope=slope,
                            minimum_alpha=minimum_alpha,
                            sd_scale=sd_scale,
                        )
                    )
    return rules


def select_rule(
    examples: list[GroupPrediction],
    *,
    sd_scale: float,
    stability_weights: tuple[float, ...],
    max_nonnull_increase: float,
) -> tuple[NullShrinkageRule, list[dict[str, Any]]]:
    baseline = baseline_metrics(examples)
    baseline_nonnull = float(baseline["nonnull_mse"])
    arrays = scalar_arrays(examples, sd_scale=sd_scale)
    candidates = []
    for rule in candidate_rules(
        examples,
        sd_scale=sd_scale,
        stability_weights=stability_weights,
    ):
        metrics = vector_rule_metrics(arrays, rule)
        nonnull_change = float(metrics["nonnull_mse"]) / baseline_nonnull - 1.0
        candidates.append(
            {
                "rule": rule.to_dict(),
                "nonnull_relative_mse_change": nonnull_change,
                **metrics,
            }
        )
    eligible = [
        row
        for row in candidates
        if float(row["nonnull_relative_mse_change"]) <= max_nonnull_increase
    ]
    if not eligible:
        raise RuntimeError("no shrinkage rule satisfies the nonnull risk constraint")
    eligible.sort(
        key=lambda row: (
            float(row["null_mse"]),
            float(row["all_mse"]),
            -float(row["nonnull_mean_alpha"]),
        )
    )
    return NullShrinkageRule.from_dict(eligible[0]["rule"]), eligible[:25]


def rank_auc(negative: np.ndarray, positive: np.ndarray) -> float:
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean((comparisons > 0.0) + 0.5 * (comparisons == 0.0)))


def diagnostic_rows(
    examples: list[GroupPrediction],
    split: dict[int, str],
    rule: NullShrinkageRule,
    one_pass_rule: NullShrinkageRule,
) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        alpha = rule.alpha(example.edge_mean, example.edge_sd, stability=example.stability)
        one_pass_alpha = one_pass_rule.alpha(example.edge_mean, example.edge_sd)
        rows.append(
            {
                "group": example.group,
                "split": split[example.group],
                "surface_kind": example.surface_kind,
                "family": example.family,
                "is_null": example.is_null,
                "effective_snps": example.effective_snps,
                "sampled_node_fraction": example.sampled_node_fraction,
                "prediction_rms": rms(example.edge_mean),
                "posterior_sd_rms": float(np.sqrt(np.mean(example.edge_sd**2))),
                "stability": example.stability,
                "alpha": alpha,
                "one_pass_alpha": one_pass_alpha,
                "baseline_mse": mse_at_alpha(example, 1.0),
                "shrunk_mse": mse_at_alpha(example, alpha),
                "one_pass_mse": mse_at_alpha(example, one_pass_alpha),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(json_ready(rows))


def exact_notebook_metrics(
    path: Path,
    hybrid_rule: NullShrinkageRule,
    one_pass_rule: NullShrinkageRule,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    rows = []
    for truth_name in ("barrier", "anisotropy", "homogeneous"):
        indices = np.flatnonzero(arrays["truth_name"] == truth_name)
        anchor = int(indices[0])
        stability = mask_stability(
            arrays["dreems"][anchor],
            [arrays["dreems"][int(index)] for index in indices[1:]],
        )
        hybrid_alpha = hybrid_rule.alpha(
            arrays["dreems"][anchor],
            arrays["edge_sd"][anchor],
            stability=stability,
        )
        for index in indices:
            local = int(index)
            truth = centered(arrays["truth"][local])
            prediction = centered(arrays["dreems_map"][local])
            one_pass_alpha = one_pass_rule.alpha(arrays["dreems"][local], arrays["edge_sd"][local])
            for method, alpha in (
                ("baseline", 1.0),
                ("hybrid_group", hybrid_alpha),
                ("one_pass", one_pass_alpha),
            ):
                metrics = edge_log_metrics(truth, alpha * prediction)
                rows.append(
                    {
                        "index": local,
                        "truth": truth_name,
                        "sample_design": str(arrays["sample_design"][local]),
                        "method": method,
                        "alpha": alpha,
                        "stability": stability,
                        "mse": metrics.mse,
                        "corr": metrics.corr,
                    }
                )
    return summarize_method_rows(rows)


def summarize_method_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = sorted({str(row["method"]) for row in rows})
    summaries = {}
    for method in methods:
        local = [row for row in rows if row["method"] == method]
        summaries[method] = {
            "examples": len(local),
            "mean_mse": float(np.mean([float(row["mse"]) for row in local])),
            "mean_alpha": float(np.mean([float(row["alpha"]) for row in local])),
            "rows": local,
        }
    return summaries


def broad_h2h_metrics(paths: list[Path], rule: NullShrinkageRule) -> dict[str, Any]:
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            truth = centered(archive["truth"])
            prediction = centered(archive["dreems_map"])
            alpha = rule.alpha(archive["dreems"], archive["edge_sd"])
        rows.append(
            {
                "method": "baseline",
                "alpha": 1.0,
                "mse": edge_log_metrics(truth, prediction).mse,
            }
        )
        rows.append(
            {
                "method": "one_pass",
                "alpha": alpha,
                "mse": edge_log_metrics(truth, alpha * prediction).mse,
            }
        )
    return summarize_method_rows(rows)


def plot_diagnostics(
    rows: list[dict[str, Any]],
    report: dict[str, Any],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    for is_null, label, color in (
        (True, "homogeneous", "#b23a2b"),
        (False, "structured", "#2468a2"),
    ):
        local = [row for row in rows if bool(row["is_null"]) is is_null]
        axes[0].scatter(
            [row["stability"] for row in local],
            [
                np.log(
                    max(
                        float(row["prediction_rms"]) / max(float(row["posterior_sd_rms"]), 1e-8),
                        1e-8,
                    )
                )
                for row in local
            ],
            s=24,
            alpha=0.7,
            label=label,
            color=color,
        )
        axes[1].scatter(
            [row["stability"] for row in local],
            [row["alpha"] for row in local],
            s=24,
            alpha=0.7,
            label=label,
            color=color,
        )
    axes[0].set_xlabel("cross-mask edge correlation")
    axes[0].set_ylabel("log posterior signal/noise")
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("cross-mask edge correlation")
    axes[1].set_ylabel("selected shrinkage alpha")

    verification = report["verification"]
    labels = ["homogeneous", "structured"]
    baseline = [verification["baseline"]["null_mse"], verification["baseline"]["nonnull_mse"]]
    shrunk = [verification["hybrid"]["null_mse"], verification["hybrid"]["nonnull_mse"]]
    x = np.arange(2)
    axes[2].bar(x - 0.18, baseline, 0.36, label="baseline", color="#8c8c8c")
    axes[2].bar(x + 0.18, shrunk, 0.36, label="shrinkage", color="#2b8c6b")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("mean edge MSE")
    axes[2].legend(frameon=False)
    figure.suptitle("Null-surface shrinkage audit")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def command_analyze(args: argparse.Namespace) -> None:
    examples = load_group_predictions(args.predictions, args.summaries, args.selection)
    split = stratified_split(examples, seed=int(args.seed))
    selection = [example for example in examples if split[example.group] == "selection"]
    verification = [example for example in examples if split[example.group] == "verification"]
    sd_scale = fit_sd_scale(selection)
    hybrid_rule, hybrid_candidates = select_rule(
        selection,
        sd_scale=sd_scale,
        stability_weights=(0.25, 0.5, 0.75, 1.0),
        max_nonnull_increase=float(args.max_nonnull_increase),
    )
    one_pass_rule, one_pass_candidates = select_rule(
        selection,
        sd_scale=sd_scale,
        stability_weights=(0.0,),
        max_nonnull_increase=float(args.max_nonnull_increase),
    )
    rows = diagnostic_rows(examples, split, hybrid_rule, one_pass_rule)
    selection_baseline = baseline_metrics(selection)
    verification_baseline = baseline_metrics(verification)

    stability_null = np.asarray(
        [example.stability for example in verification if example.is_null], dtype=np.float64
    )
    stability_nonnull = np.asarray(
        [example.stability for example in verification if not example.is_null], dtype=np.float64
    )
    report = {
        "stage": "null_surface_shrinkage_audit",
        "selection_path": str(args.selection),
        "examples": len(examples),
        "split_seed": int(args.seed),
        "sd_scale": sd_scale,
        "selected_hybrid_rule": hybrid_rule.to_dict(),
        "selected_one_pass_rule": one_pass_rule.to_dict(),
        "selection": {
            "baseline": selection_baseline,
            "hybrid": rule_metrics(selection, hybrid_rule),
            "one_pass": rule_metrics(selection, one_pass_rule),
        },
        "verification": {
            "baseline": verification_baseline,
            "hybrid": rule_metrics(verification, hybrid_rule),
            "one_pass": rule_metrics(verification, one_pass_rule),
            "stability_auc": rank_auc(stability_null, stability_nonnull),
        },
        "top_hybrid_candidates": hybrid_candidates,
        "top_one_pass_candidates": one_pass_candidates,
        "exact_notebook": exact_notebook_metrics(
            args.exact_notebook,
            hybrid_rule,
            one_pass_rule,
        ),
        "broad_h2h": broad_h2h_metrics(args.broad_predictions, one_pass_rule),
        "limitations": [
            "The grouped feasibility set was used for early stopping of the sampling-robustness checkpoint.",
            "Mask-stability shrinkage requires additional DREEMS evaluations on deliberate node subsets.",
            "The exact-notebook hybrid alpha is estimated from all four notebook masks and is therefore diagnostic, not a deployable sparse-case protocol.",
        ],
    }
    write_json(args.output, report)
    write_csv(args.rows_output, rows)
    plot_diagnostics(rows, report, args.figure)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--predictions", type=Path, nargs="+", required=True)
    result.add_argument("--summaries", type=Path, nargs="+", required=True)
    result.add_argument("--selection", type=Path, required=True)
    result.add_argument("--exact-notebook", type=Path, required=True)
    result.add_argument("--broad-predictions", type=Path, nargs="+", required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--rows-output", type=Path, required=True)
    result.add_argument("--figure", type=Path, required=True)
    result.add_argument("--seed", type=int, default=20260831)
    result.add_argument("--max-nonnull-increase", type=float, default=0.01)
    return result


if __name__ == "__main__":
    command_analyze(parser().parse_args())
