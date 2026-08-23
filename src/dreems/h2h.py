"""Reproducible per-example workers for DREEMS versus FEEMS comparisons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from dreems import feems, metrics
from dreems.inference import InferenceExample, load_model
from dreems.refinement import DREEMSMapRefinementConfig
from dreems.workflow_io import json_ready

FEEMS_CV_NOTEBOOK = (
    "https://github.com/NovembreLab/feems/blob/main/docsrc/notebooks/cross-validation.ipynb"
)
FEEMS_CV_SOURCE = (
    "https://github.com/NovembreLab/feems/blob/"
    "aa890b0e8ef583dd657a6a696af591b732d1d8be/feems/cross_validation.py"
)
FEEMS_PINNED_COMMIT = "aa890b0e8ef583dd657a6a696af591b732d1d8be"

CASE_METADATA_KEYS = (
    "surface_kind",
    "sample_design",
    "sample_regime",
    "graph_source",
    "mask_kind",
    "num_snps",
    "effective_snps",
    "island_marker_model",
    "sampling_association_mode",
    "sampling_association_strength",
    "sampling_count_migration_correlation",
    "sampling_inclusion_migration_correlation",
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _load_example(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError(f"expected one example in {path}, found {len(payload)}")
        payload = payload[0]
    if not isinstance(payload, dict):
        raise TypeError(f"expected an example dictionary in {path}")
    return payload


def _example_arrays(
    payload: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    graph = payload.get("graph")
    if not isinstance(graph, Mapping):
        raise TypeError("H2H example must contain a graph mapping")
    positions = np.asarray(payload["active_positions"], dtype=np.float64)
    edges = np.asarray(graph["edges"], dtype=np.int32)
    sample_nodes = np.asarray(payload["sample_nodes"], dtype=np.int32)
    genotypes = np.asarray(payload["genotypes"], dtype=np.float64)
    truth = np.asarray(payload.get("edge_log_rates", payload.get("truth")), dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("active_positions must have shape (nodes, 2)")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("graph edges must have shape (edges, 2)")
    if truth.shape != (edges.shape[0],):
        raise ValueError("truth must contain one log migration value per edge")
    if genotypes.ndim != 2 or genotypes.shape[0] != sample_nodes.shape[0]:
        raise ValueError("genotypes and sample_nodes disagree on sample count")
    return positions, edges, sample_nodes, genotypes, truth


def _case_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: payload[key] for key in CASE_METADATA_KEYS if key in payload}


def _base_row(
    *,
    index: int,
    payload: Mapping[str, Any],
    positions: np.ndarray,
    edges: np.ndarray,
    sample_nodes: np.ndarray,
) -> dict[str, Any]:
    observed = np.unique(sample_nodes)
    return {
        "index": int(index),
        "num_nodes": int(positions.shape[0]),
        "num_edges": int(edges.shape[0]),
        "num_samples": int(sample_nodes.shape[0]),
        "num_sampled_nodes": int(observed.size),
        "sampled_node_fraction": float(observed.size / positions.shape[0]),
        **_case_metadata(payload),
    }


def _metric_fields(prefix: str, truth: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    return metrics.edge_log_metrics(truth, estimate).to_prefixed_dict(prefix)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feems_cv_checkpoint_key(input_path: Path) -> str:
    return f"{FEEMS_PINNED_COMMIT}:{feems.FEEMS_RECOMMENDED_LOO_PROTOCOL}:{_sha256(input_path)}"


def run_feems_fold(args: argparse.Namespace) -> None:
    payload = _load_example(args.input)
    positions, _edges, sample_nodes, genotypes, _truth = _example_arrays(payload)
    started = time.perf_counter()
    errors = feems.recommended_loo_fold_errors(
        genotypes=genotypes,
        sample_nodes=sample_nodes,
        graph=dict(payload["graph"]),
        active_positions=positions,
        fold=int(args.fold),
    )
    elapsed = time.perf_counter() - started
    feems.write_cv_fold_checkpoint(
        args.output_npz,
        fold=int(args.fold),
        checkpoint_key=_feems_cv_checkpoint_key(args.input),
        errors=errors,
    )
    print(
        json.dumps(
            {
                "stage": "feems_h2h_fold",
                "fold": int(args.fold),
                "elapsed_seconds": float(elapsed),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def run_feems_case(args: argparse.Namespace) -> None:
    payload = _load_example(args.input)
    positions, edges, sample_nodes, genotypes, truth = _example_arrays(payload)
    config = feems.recommended_loo_config(cv_jobs=int(args.cv_jobs))
    feems.require_recommended_loo(config)
    observed_nodes = np.unique(sample_nodes)
    started = time.perf_counter()
    result = feems.fit_feems_recommended_loo(
        genotypes=genotypes,
        sample_nodes=sample_nodes,
        graph=dict(payload["graph"]),
        active_positions=positions,
        cv_jobs=config.parallel_cv_jobs,
        cv_checkpoint_dir=args.cv_checkpoint_dir,
        cv_checkpoint_key=_feems_cv_checkpoint_key(args.input),
    )
    elapsed = time.perf_counter() - started
    estimate = np.asarray(result.edge_log_rates, dtype=np.float64)
    if result.cv_mode != "leave_one_out" or result.cv_folds != observed_nodes.size:
        raise RuntimeError(
            "FEEMS did not execute one held-out fold per observed deme: "
            f"mode={result.cv_mode!r}, folds={result.cv_folds}, "
            f"observed_demes={observed_nodes.size}"
        )
    if result.protocol != "v2_joint_lambda_lambda_q":
        raise RuntimeError(f"unexpected FEEMS fit protocol: {result.protocol!r}")
    if estimate.shape != truth.shape or not np.all(np.isfinite(estimate)):
        raise RuntimeError("FEEMS returned an invalid edge estimate")

    row = {
        **_base_row(
            index=args.index,
            payload=payload,
            positions=positions,
            edges=edges,
            sample_nodes=sample_nodes,
        ),
        "method": "feems",
        "protocol": feems.FEEMS_RECOMMENDED_LOO_PROTOCOL,
        "elapsed_seconds": float(elapsed),
        "cv_jobs": int(config.parallel_cv_jobs),
        "best_lambda": float(result.best_lambda),
        "best_lambda_q": float(result.best_lambda_q),
        "cv_folds": int(result.cv_folds),
        "cv_mode": result.cv_mode,
        "n_snps_after_filter": int(result.n_snps_after_filter),
        "train_loss": float(result.train_loss),
        **_metric_fields("feems_edge_log", truth, estimate),
    }
    sample_count = np.bincount(sample_nodes, minlength=positions.shape[0]).astype(np.int32)
    _atomic_json(args.output_json, row)
    _atomic_npz(
        args.output_npz,
        positions=positions,
        edges=edges,
        truth=metrics.centered(truth),
        feems=metrics.centered(estimate),
        sample_count=sample_count,
        mean_cv_error=np.asarray(result.mean_cv_error, dtype=np.float64),
    )
    print(json.dumps(json_ready({"stage": "feems_h2h_case", **row}), sort_keys=True), flush=True)


def _coverage(truth: np.ndarray, mean: np.ndarray, sd: np.ndarray, z: float) -> float:
    centered_truth = metrics.centered(truth)
    centered_mean = metrics.centered(mean)
    sd = np.asarray(sd, dtype=np.float64)
    finite = np.isfinite(centered_truth) & np.isfinite(centered_mean) & np.isfinite(sd) & (sd > 0.0)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(np.abs(centered_truth[finite] - centered_mean[finite]) <= z * sd[finite]))


def run_dreems_case(args: argparse.Namespace) -> None:
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
    refinement_config = DREEMSMapRefinementConfig()
    result = predictor.predict(example, refinement_config=refinement_config)
    prediction_seconds = time.perf_counter() - started - loaded_seconds
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
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_load_seconds": float(loaded_seconds),
        "prediction_seconds": float(prediction_seconds),
        "elapsed_seconds": float(time.perf_counter() - started),
        "posterior_sd_mean": float(np.mean(edge_sd)),
        "posterior_sd_median": float(np.median(edge_sd)),
        "neural_50_coverage": _coverage(truth, neural, edge_sd, 0.67448975),
        "neural_80_coverage": _coverage(truth, neural, edge_sd, 1.28155157),
        "neural_95_coverage": _coverage(truth, neural, edge_sd, 1.95996398),
        **_metric_fields("dreems_edge_log", truth, neural),
        **_metric_fields("dreems_map_edge_log", truth, refined),
    }
    diagnostics = result.refinement_diagnostics or {}
    for fit_name in ("initial_fit", "refined_fit"):
        fit = diagnostics.get(fit_name, {})
        if isinstance(fit, Mapping):
            for key in ("corr", "r2", "mse", "alpha", "q_scale"):
                if key in fit:
                    row[f"covariance_{fit_name.removesuffix('_fit')}_{key}"] = float(fit[key])

    sample_count = np.bincount(sample_nodes, minlength=positions.shape[0]).astype(np.int32)
    _atomic_json(args.output_json, row)
    _atomic_npz(
        args.output_npz,
        positions=positions,
        edges=edges,
        truth=metrics.centered(truth),
        dreems=neural,
        dreems_map=refined,
        edge_sd=edge_sd,
        sample_count=sample_count,
    )
    print(json.dumps(json_ready({"stage": "dreems_h2h_case", **row}), sort_keys=True), flush=True)


def _read_rows(paths: Sequence[Path], expected_method: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for path in paths:
        row = json.loads(path.read_text())
        if row.get("method") != expected_method:
            raise ValueError(f"{path} is not a {expected_method} result")
        index = int(row["index"])
        if index in rows:
            raise ValueError(f"duplicate {expected_method} result for example {index}")
        rows[index] = row
    return rows


def _paired_bootstrap(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    draws = rng.integers(0, finite.size, size=(replicates, finite.size))
    means = np.mean(finite[draws], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _paired_comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
    baseline: str,
    higher_is_better: bool,
    rng: np.random.Generator,
    bootstrap_replicates: int,
) -> dict[str, Any]:
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


def _numeric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(key for key in row if key not in fieldnames)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def protocol_manifest(
    *,
    cv_execution: str,
    checkpoint: Path,
    source_commit: str,
    input_pattern: str,
    metadata_pattern: str,
) -> dict[str, Any]:
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
            "fold_parallelism_only": True,
            "final_refit": "all samples at jointly selected lambda and lambda_q",
            "pinned_commit": FEEMS_PINNED_COMMIT,
            "official_notebook": FEEMS_CV_NOTEBOOK,
            "official_source": FEEMS_CV_SOURCE,
        },
        "dreems": {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "neural_estimate": True,
            "map_estimate": {
                "enabled": True,
                **vars(DREEMSMapRefinementConfig()),
            },
        },
    }


def aggregate(args: argparse.Namespace) -> None:
    feems_rows = _read_rows(args.feems_json, "feems")
    dreems_rows = _read_rows(args.dreems_json, "dreems")
    if feems_rows.keys() != dreems_rows.keys():
        raise ValueError("FEEMS and DREEMS result sets contain different examples")

    rows: list[dict[str, Any]] = []
    for index in sorted(feems_rows):
        feems_row = feems_rows[index]
        dreems_row = dreems_rows[index]
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
    )
    summary = {
        "examples": len(rows),
        "protocol": manifest,
        "numeric_summary": _numeric_summary(rows),
        "paired_comparisons": comparisons,
    }
    _write_csv(args.output_csv, rows)
    _atomic_json(args.output_json, summary)
    _atomic_json(args.output_protocol, manifest)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    feems_fold = subparsers.add_parser("feems-fold")
    feems_fold.add_argument("--input", type=Path, required=True)
    feems_fold.add_argument("--fold", type=int, required=True)
    feems_fold.add_argument("--output-npz", type=Path, required=True)
    feems_fold.set_defaults(handler=run_feems_fold)

    feems_case = subparsers.add_parser("feems-case")
    feems_case.add_argument("--input", type=Path, required=True)
    feems_case.add_argument("--index", type=int, required=True)
    feems_case.add_argument("--cv-jobs", type=int, required=True)
    feems_case.add_argument("--cv-checkpoint-dir", type=Path)
    feems_case.add_argument("--output-json", type=Path, required=True)
    feems_case.add_argument("--output-npz", type=Path, required=True)
    feems_case.set_defaults(handler=run_feems_case)

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


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
