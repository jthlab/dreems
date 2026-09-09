"""Reproducible per-example workers for DREEMS versus FEEMS comparisons."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from beartype import beartype
from scipy.stats import binomtest

from dreems import features, feems, feems_benchmark, metrics
from dreems.benchmark_data import (
    _base_row,
    _centered_rms,
    _example_arrays,
    _load_example,
    _metric_fields,
)
from dreems.feems_benchmark import FEEMS_PINNED_COMMIT
from dreems.feems_benchmark import run_feems_case as run_feems_case
from dreems.feems_benchmark import run_feems_fold as run_feems_fold
from dreems.inference import InferenceExample, load_model
from dreems.refinement import DREEMSMapRefinementConfig
from dreems.workflow_io import json_ready, sha256, write_csv, write_json, write_npz

FEEMS_CV_NOTEBOOK = (
    "https://github.com/NovembreLab/feems/blob/main/docsrc/notebooks/cross-validation.ipynb"
)
FEEMS_CV_SOURCE = (
    "https://github.com/NovembreLab/feems/blob/"
    "aa890b0e8ef583dd657a6a696af591b732d1d8be/feems/cross_validation.py"
)


@beartype
def _coverage(truth: np.ndarray, mean: np.ndarray, sd: np.ndarray, z: float) -> float:
    """Score pointwise neural intervals against the centered generating surface."""
    centered_truth = metrics.centered(truth)
    centered_mean = metrics.centered(mean)
    sd = np.asarray(sd, dtype=np.float64)
    finite = np.isfinite(centered_truth) & np.isfinite(centered_mean) & np.isfinite(sd) & (sd > 0.0)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(np.abs(centered_truth[finite] - centered_mean[finite]) <= z * sd[finite]))


@beartype
def run_dreems_case(args: argparse.Namespace) -> None:
    """Save neural and MAP estimates from the same simulated observation."""
    input_digest = sha256(args.input)
    checkpoint_digest = sha256(args.checkpoint)
    payload = _load_example(args.input)
    positions, edges, sample_nodes, genotypes, truth = _example_arrays(payload)
    example = InferenceExample(
        positions=positions,
        edges=edges,
        sample_nodes=sample_nodes,
        genotypes=genotypes,
        name=f"h2h_{args.index}",
    )
    started = time.perf_counter()
    predictor = load_model(args.checkpoint)
    loaded_seconds = time.perf_counter() - started
    conditioner_started = time.perf_counter()
    conditioner, conditioner_info = features.fit_linearized_resistance_proxy(
        payload,
        predictor.checkpoint.model,
    )
    conditioner_seconds = time.perf_counter() - conditioner_started
    refinement_config = DREEMSMapRefinementConfig()
    prediction_started = time.perf_counter()
    result = predictor.predict(example, refinement_config=refinement_config)
    prediction_seconds = time.perf_counter() - prediction_started
    neural = result.centered_edge_log_mean
    refined = result.centered_refined_edge_log_mean
    if refined is None:
        raise RuntimeError("DREEMS MAP refinement did not return an edge estimate")
    edge_sd = np.asarray(result.edge_log_sd, dtype=np.float64)

    row = {
        **_base_row(
            index=args.index,
            payload=payload,
            positions=positions,
            edges=edges,
            sample_nodes=sample_nodes,
        ),
        "method": "dreems",
        "input_sha256": input_digest,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_load_seconds": float(loaded_seconds),
        "conditioner_seconds": float(conditioner_seconds),
        "conditioner_observed_nodes": int(conditioner_info.get("n_observed_nodes", 0)),
        "conditioner_observed_pairs": int(conditioner_info.get("n_pairs", 0)),
        "conditioner_baseline_pair_log_corr": float(
            conditioner_info.get("baseline_pair_log_corr", float("nan"))
        ),
        "conditioner_pair_residual_sd": float(
            conditioner_info.get("pair_residual_sd", float("nan"))
        ),
        "prediction_seconds": float(prediction_seconds),
        "elapsed_seconds": float(time.perf_counter() - started),
        "posterior_sd_mean": float(np.mean(edge_sd)),
        "posterior_sd_median": float(np.median(edge_sd)),
        "truth_edge_log_rms": _centered_rms(truth),
        "conditioner_edge_log_rms": _centered_rms(conditioner),
        "dreems_edge_log_rms": _centered_rms(neural),
        "dreems_map_edge_log_rms": _centered_rms(refined),
        "neural_50_coverage": _coverage(truth, neural, edge_sd, 0.67448975),
        "neural_80_coverage": _coverage(truth, neural, edge_sd, 1.28155157),
        "neural_95_coverage": _coverage(truth, neural, edge_sd, 1.95996398),
        **_metric_fields("conditioner_edge_log", truth, conditioner),
        **_metric_fields("dreems_edge_log", truth, neural),
        **_metric_fields("dreems_map_edge_log", truth, refined),
    }
    if sha256(args.input) != input_digest:
        raise RuntimeError("DREEMS input changed during inference; refusing to publish the result")
    if sha256(args.checkpoint) != checkpoint_digest:
        raise RuntimeError(
            "DREEMS checkpoint changed during inference; refusing to publish the result"
        )
    diagnostics = result.refinement_diagnostics or {}
    for fit_name in ("initial_fit", "refined_fit"):
        fit = diagnostics.get(fit_name, {})
        if isinstance(fit, Mapping):
            for key in ("corr", "r2", "mse", "alpha", "q_scale"):
                if key in fit:
                    row[f"covariance_{fit_name.removesuffix('_fit')}_{key}"] = float(fit[key])

    sample_count = np.bincount(sample_nodes, minlength=positions.shape[0]).astype(np.int32)
    write_json(args.output_json, row)
    write_npz(
        args.output_npz,
        positions=positions,
        edges=edges,
        truth=metrics.centered(truth),
        conditioner=metrics.centered(conditioner),
        dreems=neural,
        dreems_map=refined,
        edge_sd=edge_sd,
        sample_count=sample_count,
        checkpoint_sha256=np.asarray(checkpoint_digest),
    )
    print(json.dumps(json_ready({"stage": "dreems_h2h_case", **row}), sort_keys=True), flush=True)


@beartype
def _read_rows(paths: Sequence[Path], expected_method: str) -> dict[int, dict[str, Any]]:
    """Index reports without coercing case IDs or accepting malformed input hashes."""
    rows: dict[int, dict[str, Any]] = {}
    for path in paths:
        row = json.loads(path.read_text())
        if not isinstance(row, dict) or row.get("method") != expected_method:
            raise ValueError(f"{path} is not a {expected_method} result")
        index = row.get("index")
        if type(index) is not int or index < 0:
            raise ValueError(f"{path} needs a nonnegative integer case ID")
        digest = row.get("input_sha256")
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(f"{path} has an invalid input SHA-256 digest")
        if index in rows:
            raise ValueError(f"duplicate {expected_method} result for example {index}")
        rows[index] = row
    return rows


@beartype
def _paired_bootstrap(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float]:
    """Resample cases, not edges, to obtain an interval for the mean advantage."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    draws = rng.integers(0, finite.size, size=(replicates, finite.size))
    means = np.mean(finite[draws], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


@beartype
def _paired_comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
    baseline: str,
    higher_is_better: bool,
    rng: np.random.Generator,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    """Summarize paired advantages using the metric's direction of improvement."""
    candidate_values = np.asarray([row[candidate] for row in rows], dtype=np.float64)
    baseline_values = np.asarray([row[baseline] for row in rows], dtype=np.float64)
    finite = np.isfinite(candidate_values) & np.isfinite(baseline_values)
    delta = candidate_values[finite] - baseline_values[finite]
    advantage = delta if higher_is_better else -delta
    wins = int(np.sum(advantage > 0.0))
    losses = int(np.sum(advantage < 0.0))
    non_ties = wins + losses
    low, high = _paired_bootstrap(
        advantage,
        rng=rng,
        replicates=bootstrap_replicates,
    )
    return {
        "candidate": candidate,
        "baseline": baseline,
        "positive_advantage_means_candidate_is_better": True,
        "n": int(advantage.size),
        "mean_advantage": float(np.mean(advantage)) if advantage.size else float("nan"),
        "median_advantage": float(np.median(advantage)) if advantage.size else float("nan"),
        "mean_advantage_bootstrap_95_low": low,
        "mean_advantage_bootstrap_95_high": high,
        "wins": wins,
        "losses": losses,
        "ties": int(advantage.size - non_ties),
        "win_rate_excluding_ties": float(wins / non_ties) if non_ties else float("nan"),
        "two_sided_sign_test_p": (
            float(binomtest(wins, non_ties, 0.5).pvalue) if non_ties else float("nan")
        ),
    }


@beartype
def _numeric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Report finite case counts and summary statistics for numeric columns."""
    keys = sorted({key for row in rows for key in row})
    summary: dict[str, dict[str, float | int]] = {}
    for key in keys:
        values = [row.get(key) for row in rows]
        numeric = [float(value) for value in values if isinstance(value, int | float)]
        if not numeric:
            continue
        array = np.asarray(numeric, dtype=np.float64)
        finite = array[np.isfinite(array)]
        if finite.size:
            summary[key] = {
                "n": int(finite.size),
                "mean": float(np.mean(finite)),
                "median": float(np.median(finite)),
                "sd": float(np.std(finite)),
            }
    return summary


@beartype
def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Replace a complete CSV report, preserving first-seen column order."""
    write_csv(path, rows)


@beartype
def protocol_manifest(
    *,
    cv_execution: str,
    checkpoint: Path,
    source_commit: str,
    input_pattern: str,
    metadata_pattern: str,
    feems_result_source: str = "computed",
) -> dict[str, Any]:
    """Record the checkpoint and the precise FEEMS cross-validation protocol."""
    config = feems.recommended_loo_config()
    feems.require_recommended_loo(config)
    return {
        "comparison": "paired synthetic edge-log migration recovery",
        "dreems_source_commit": source_commit,
        "input_pattern": input_pattern,
        "metadata_pattern": metadata_pattern,
        "feems": {
            "protocol": feems.FEEMS_RECOMMENDED_LOO_PROTOCOL,
            "mode": config.mode,
            "cv_unit": "observed deme",
            "cv_folds": "one per observed deme",
            "selection_score": "FEEMS held-out allele-frequency L2 error",
            "lambda_grid": list(config.lambdas),
            "lambda_q_grid": list(config.lambdas_q),
            "cv_factr": config.factr,
            "final_refit_factr": feems.FEEMS_DEFAULT_FINAL_FACTR,
            "final_refit_maxiter": feems.FEEMS_RECOMMENDED_MAXITER,
            "final_refit_log_weight_bounds": "unbounded",
            "random_state": config.random_state,
            "fold_execution": cv_execution,
            "result_source": feems_result_source,
            "fold_parallelism_only": True,
            "final_refit": "all samples at jointly selected lambda and lambda_q",
            "pinned_commit": FEEMS_PINNED_COMMIT,
            "official_notebook": FEEMS_CV_NOTEBOOK,
            "official_source": FEEMS_CV_SOURCE,
        },
        "dreems": {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "neural_estimate": True,
            "map_estimate": {
                "enabled": True,
                **vars(DREEMSMapRefinementConfig()),
            },
        },
    }


@beartype
def aggregate(args: argparse.Namespace) -> None:
    """Join matching case reports and retain the documented FEEMS tuning protocol."""
    feems_rows = _read_rows(args.feems_json, "feems")
    dreems_rows = _read_rows(args.dreems_json, "dreems")
    if feems_rows.keys() != dreems_rows.keys():
        raise ValueError("FEEMS and DREEMS result sets contain different examples")
    if any("checkpoint_sha256" in row for row in dreems_rows.values()):
        checkpoint_digest = sha256(args.checkpoint)
        for index, row in dreems_rows.items():
            if row.get("checkpoint_sha256") != checkpoint_digest:
                raise ValueError(
                    f"example {index} is not verified against the requested checkpoint"
                )

    rows: list[dict[str, Any]] = []
    for index in sorted(feems_rows):
        feems_row = feems_rows[index]
        dreems_row = dreems_rows[index]
        if (
            feems_row.get("input_sha256") is not None
            and dreems_row.get("input_sha256") is not None
            and feems_row["input_sha256"] != dreems_row["input_sha256"]
        ):
            raise ValueError(f"example {index} uses different input files for FEEMS and DREEMS")
        for key in ("num_nodes", "num_edges", "num_samples", "num_sampled_nodes"):
            if feems_row[key] != dreems_row[key]:
                raise ValueError(f"example {index} disagrees on {key}")
        if feems_row["protocol"] != feems.FEEMS_RECOMMENDED_LOO_PROTOCOL:
            raise ValueError(f"example {index} did not use documented FEEMS LOO")
        row = dict(dreems_row)
        row.update(
            {
                key: value
                for key, value in feems_row.items()
                if key.startswith("feems_")
                or key
                in {
                    "best_lambda",
                    "best_lambda_q",
                    "cv_folds",
                    "cv_mode",
                    "n_snps_after_filter",
                    "train_loss",
                }
            }
        )
        row["feems_elapsed_seconds"] = feems_row["elapsed_seconds"]
        row["feems_cv_jobs"] = feems_row["cv_jobs"]
        row["feems_protocol"] = feems_row["protocol"]
        rows.append(row)

    rng = np.random.default_rng(args.bootstrap_seed)
    comparisons = {
        "dreems_neural_corr_vs_feems": _paired_comparison(
            rows,
            candidate="dreems_edge_log_corr",
            baseline="feems_edge_log_corr",
            higher_is_better=True,
            rng=rng,
            bootstrap_replicates=args.bootstrap_replicates,
        ),
        "dreems_neural_mse_vs_feems": _paired_comparison(
            rows,
            candidate="dreems_edge_log_mse",
            baseline="feems_edge_log_mse",
            higher_is_better=False,
            rng=rng,
            bootstrap_replicates=args.bootstrap_replicates,
        ),
        "dreems_map_corr_vs_feems": _paired_comparison(
            rows,
            candidate="dreems_map_edge_log_corr",
            baseline="feems_edge_log_corr",
            higher_is_better=True,
            rng=rng,
            bootstrap_replicates=args.bootstrap_replicates,
        ),
        "dreems_map_mse_vs_feems": _paired_comparison(
            rows,
            candidate="dreems_map_edge_log_mse",
            baseline="feems_edge_log_mse",
            higher_is_better=False,
            rng=rng,
            bootstrap_replicates=args.bootstrap_replicates,
        ),
    }
    manifest = protocol_manifest(
        cv_execution=args.cv_execution,
        checkpoint=args.checkpoint,
        source_commit=args.source_commit,
        input_pattern=args.input_pattern,
        metadata_pattern=args.metadata_pattern,
        feems_result_source=getattr(args, "feems_result_source", "computed"),
    )
    manifest["paired_input_identity"] = (
        "sha256"
        if all(
            feems_rows[index].get("input_sha256") is not None
            and dreems_rows[index].get("input_sha256") is not None
            for index in feems_rows
        )
        else "not_recorded_for_all_cases"
    )
    manifest["dreems"]["prediction_checkpoint_identity"] = (
        "sha256"
        if all(row.get("checkpoint_sha256") is not None for row in dreems_rows.values())
        else "not_recorded_for_all_cases"
    )
    verification_path = getattr(args, "feems_archive_verification", None)
    if (
        getattr(args, "feems_result_source", "computed") in {"verified_archive", "legacy_archive"}
        and verification_path is None
    ):
        raise ValueError("archive reuse requires its verification report")
    if verification_path is not None:
        verification = json.loads(verification_path.read_text())
        if (
            getattr(args, "feems_result_source", "computed") == "verified_archive"
            and verification["input_identity"] != "sha256"
        ):
            raise ValueError("legacy archive input identity cannot be reported as verified")
        cases = verification["cases"]
        if len(cases) != len(rows) or {case["index"] for case in cases} != set(feems_rows):
            raise ValueError("archive verification covers different examples")
        for case in cases:
            digest = dreems_rows[case["index"]].get("input_sha256")
            if digest is not None and digest != case["current_input_sha256"]:
                raise ValueError("archive verification and DREEMS used different inputs")
        manifest["feems"]["archive_verification"] = {
            key: verification[key] for key in ("archive_sha256", "examples", "input_identity")
        }
    summary = {
        "examples": len(rows),
        "protocol": manifest,
        "numeric_summary": _numeric_summary(rows),
        "paired_comparisons": comparisons,
    }
    _write_csv(args.output_csv, rows)
    write_json(args.output_json, summary)
    write_json(args.output_protocol, manifest)
    print(
        json.dumps(
            {
                "stage": "aggregate_h2h",
                "examples": len(rows),
                "paired_comparisons": comparisons,
            },
            sort_keys=True,
        ),
        flush=True,
    )


@beartype
def build_parser() -> argparse.ArgumentParser:
    """Expose separate fitting and aggregation commands with stable arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    feems_benchmark.add_commands(subparsers)

    dreems_case = subparsers.add_parser("dreems-case")
    dreems_case.add_argument("--input", type=Path, required=True)
    dreems_case.add_argument("--index", type=int, required=True)
    dreems_case.add_argument("--checkpoint", type=Path, required=True)
    dreems_case.add_argument("--output-json", type=Path, required=True)
    dreems_case.add_argument("--output-npz", type=Path, required=True)
    dreems_case.set_defaults(handler=run_dreems_case)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--feems-json", type=Path, nargs="+", required=True)
    aggregate_parser.add_argument("--dreems-json", type=Path, nargs="+", required=True)
    aggregate_parser.add_argument("--checkpoint", type=Path, required=True)
    aggregate_parser.add_argument("--cv-execution", required=True)
    aggregate_parser.add_argument(
        "--feems-result-source",
        choices=("computed", "verified_archive", "legacy_archive"),
        default="computed",
    )
    aggregate_parser.add_argument("--feems-archive-verification", type=Path)
    aggregate_parser.add_argument("--source-commit", required=True)
    aggregate_parser.add_argument("--input-pattern", required=True)
    aggregate_parser.add_argument("--metadata-pattern", required=True)
    aggregate_parser.add_argument("--bootstrap-seed", type=int, default=20260822)
    aggregate_parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    aggregate_parser.add_argument("--output-json", type=Path, required=True)
    aggregate_parser.add_argument("--output-csv", type=Path, required=True)
    aggregate_parser.add_argument("--output-protocol", type=Path, required=True)
    aggregate_parser.set_defaults(handler=aggregate)
    return parser


@beartype
def main() -> None:
    """Dispatch one benchmark task selected on the command line."""
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
