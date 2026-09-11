#!/usr/bin/env python3
"""Run the packaged DREEMS checkpoint on one archived real-data input."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from beartype import beartype

from dreems import DREEMSInput, DREEMSMapRefinementConfig, load_model
from dreems.neural.checkpoints import _read_checkpoint_archive
from dreems.utils.workflow_io import json_finite, sha256, write_json_finite, write_npz


@beartype
def parse_args() -> argparse.Namespace:
    """Expose prediction and checkpoint provenance as separate workflow tasks."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    predict = commands.add_parser("predict")
    predict.add_argument("--prepared-npz", type=Path, required=True)
    predict.add_argument("--prepared-json", type=Path, required=True)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--output-npz", type=Path, required=True)
    predict.add_argument("--output-json", type=Path, required=True)

    provenance = commands.add_parser("provenance")
    provenance.add_argument("--checkpoint", type=Path, required=True)
    provenance.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@beartype
def checkpoint_metadata(path: Path) -> dict[str, Any]:
    """Record the architecture and weights without running feature construction."""
    digest = sha256(path)
    members, metadata = _read_checkpoint_archive(path)
    if sha256(path) != digest:
        raise ValueError("checkpoint changed while reading its metadata")
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": digest,
        "params_sha256": hashlib.sha256(members["params.eqx"]).hexdigest(),
        "model": metadata["model"],
        "edge_sd_calibration": metadata.get("edge_sd_calibration"),
        "training_provenance": metadata.get("provenance"),
    }


@beartype
def sha256_member(path: Path, member: str) -> str:
    """Hash one ZIP member without loading it into memory."""
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive, archive.open(member) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@beartype
def run_prediction(args: argparse.Namespace) -> None:
    """Infer neural and MAP surfaces from the archived, filtered covariance."""
    inputs = (args.prepared_npz, args.prepared_json, args.checkpoint)
    input_hashes = {path: sha256(path) for path in inputs}
    metadata = json.loads(args.prepared_json.read_text())
    with np.load(args.prepared_npz, allow_pickle=False) as data:
        # Validate original values before converting dtypes or loading the model.
        example = DREEMSInput(
            sample_covariance=data["sample_covariance"],
            sample_coordinates=data["sample_xy"],
            node_coordinates=data["node_xy"],
            edges=data["edges"],
            effective_snps=metadata["effective_snps"],
            name=str(metadata["dataset_id"]),
            metadata=metadata,
        )

    started = time.perf_counter()
    print(
        json.dumps(
            {
                "event": "dreems_start",
                "dataset_id": metadata["dataset_id"],
                "samples": example.sample_coordinates.shape[0],
                "nodes": example.node_coordinates.shape[0],
                "edges": example.edges.shape[0],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    predictor = load_model(args.checkpoint)
    refinement = DREEMSMapRefinementConfig()
    result = predictor.predict(
        example,
        refinement_config=refinement,
    )
    elapsed = float(time.perf_counter() - started)
    refined = result.centered_refined_edge_log_mean
    if refined is None:
        raise RuntimeError("MAP refinement did not return an edge estimate")

    output_metadata = {
        "dataset_id": metadata["dataset_id"],
        "method": (
            f"DREEMS {predictor.checkpoint.model['architecture']['kind']} with MAP refinement"
        ),
        **checkpoint_metadata(args.checkpoint),
        "refinement": dataclasses.asdict(refinement),
        "surface_weight": float(result.surface_weight),
        "prediction_seconds": elapsed,
        "diagnostics": result.refinement_diagnostics or {},
        "prepared_npz_sha256": input_hashes[args.prepared_npz],
        "prepared_json_sha256": input_hashes[args.prepared_json],
    }
    if any(sha256(path) != digest for path, digest in input_hashes.items()):
        raise ValueError("prediction input changed during inference")
    write_npz(
        args.output_npz,
        prior_edge_log=result.centered_edge_log_mean,
        edge_log=refined,
        edge_log_sd=np.asarray(result.edge_log_sd, dtype=np.float64),
        raw_edge_log_sd=np.asarray(result.raw_edge_log_sd, dtype=np.float64),
        surface_weight=np.asarray(result.surface_weight, dtype=np.float64),
    )
    write_json_finite(args.output_json, output_metadata)
    print(json.dumps(json_finite({"event": "dreems_done", **output_metadata}), sort_keys=True))


@beartype
def main() -> None:
    """Dispatch the same operations used by the real-data Snakefile."""
    args = parse_args()
    if args.command == "predict":
        run_prediction(args)
    else:
        write_json_finite(args.output, checkpoint_metadata(args.checkpoint))


if __name__ == "__main__":
    main()
