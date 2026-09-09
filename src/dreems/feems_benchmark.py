"""Independent FEEMS workers for the paired simulation benchmark.

The workflow calls this module directly so DREEMS code and plotting changes do
not invalidate fitted FEEMS cases. The combined ``dreems.h2h`` CLI also exposes
these commands.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from beartype import beartype

from dreems import feems, metrics
from dreems.benchmark_data import (
    _base_row,
    _centered_rms,
    _example_arrays,
    _load_example,
    _metric_fields,
)
from dreems.workflow_io import json_ready, sha256, write_json, write_npz

FEEMS_PINNED_COMMIT = "aa890b0e8ef583dd657a6a696af591b732d1d8be"


@beartype
def _feems_cv_checkpoint_key(input_path: Path) -> str:
    """Tie resumable folds to the input bytes and the recorded FEEMS protocol."""
    return f"{FEEMS_PINNED_COMMIT}:{feems.FEEMS_RECOMMENDED_LOO_PROTOCOL}:{sha256(input_path)}"


@beartype
def run_feems_fold(args: argparse.Namespace) -> None:
    """Fit all prescribed penalty values for one held-out deme."""
    checkpoint_key = _feems_cv_checkpoint_key(args.input)
    payload = _load_example(args.input)
    positions, _edges, sample_nodes, genotypes, _truth = _example_arrays(payload)
    started = time.perf_counter()
    errors = feems.recommended_loo_fold_errors(
        genotypes=genotypes,
        sample_nodes=sample_nodes,
        graph=dict(payload["graph"]),
        active_positions=positions,
        fold=args.fold,
    )
    elapsed = time.perf_counter() - started
    if _feems_cv_checkpoint_key(args.input) != checkpoint_key:
        raise RuntimeError("FEEMS input changed during fitting; refusing to publish the fold")
    feems.write_cv_fold_checkpoint(
        args.output_npz,
        fold=args.fold,
        checkpoint_key=checkpoint_key,
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


@beartype
def run_feems_case(args: argparse.Namespace) -> None:
    """Choose penalties from saved LOO folds, refit, and score the generating surface."""
    input_digest = sha256(args.input)
    payload = _load_example(args.input)
    positions, edges, sample_nodes, genotypes, truth = _example_arrays(payload)
    config = feems.recommended_loo_config(cv_jobs=args.cv_jobs)
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
        "input_sha256": input_digest,
        "protocol": feems.FEEMS_RECOMMENDED_LOO_PROTOCOL,
        "elapsed_seconds": float(elapsed),
        "cv_jobs": int(config.parallel_cv_jobs),
        "best_lambda": float(result.best_lambda),
        "best_lambda_q": float(result.best_lambda_q),
        "cv_folds": int(result.cv_folds),
        "cv_mode": result.cv_mode,
        "n_snps_after_filter": int(result.n_snps_after_filter),
        "train_loss": float(result.train_loss),
        "truth_edge_log_rms": _centered_rms(truth),
        "feems_edge_log_rms": _centered_rms(estimate),
        **_metric_fields("feems_edge_log", truth, estimate),
    }
    if sha256(args.input) != input_digest:
        raise RuntimeError("FEEMS input changed during fitting; refusing to publish the result")
    sample_count = np.bincount(sample_nodes, minlength=positions.shape[0]).astype(np.int32)
    write_json(args.output_json, row)
    write_npz(
        args.output_npz,
        positions=positions,
        edges=edges,
        truth=metrics.centered(truth),
        feems=metrics.centered(estimate),
        sample_count=sample_count,
        mean_cv_error=np.asarray(result.mean_cv_error, dtype=np.float64),
    )
    print(json.dumps(json_ready({"stage": "feems_h2h_case", **row}), sort_keys=True), flush=True)


@beartype
def add_commands(subparsers: argparse._SubParsersAction) -> None:
    """Use the same FEEMS arguments in standalone and combined benchmark CLIs."""
    fold = subparsers.add_parser("feems-fold")
    fold.add_argument("--input", type=Path, required=True)
    fold.add_argument("--fold", type=int, required=True)
    fold.add_argument("--output-npz", type=Path, required=True)
    fold.set_defaults(handler=run_feems_fold)
    case = subparsers.add_parser("feems-case")
    case.add_argument("--input", type=Path, required=True)
    case.add_argument("--index", type=int, required=True)
    case.add_argument("--cv-jobs", type=int, required=True)
    case.add_argument("--cv-checkpoint-dir", type=Path)
    case.add_argument("--output-json", type=Path, required=True)
    case.add_argument("--output-npz", type=Path, required=True)
    case.set_defaults(handler=run_feems_case)


@beartype
def main() -> None:
    """Run one LOO fold or the final fit using the official FEEMS protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_commands(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
