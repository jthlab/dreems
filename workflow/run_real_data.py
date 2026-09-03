#!/usr/bin/env python3
"""Run the packaged DREEMS checkpoint on one archived real-data input."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from dreems import DREEMSInput, DREEMSMapRefinementConfig, load_model


def parse_args() -> argparse.Namespace:
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": sha256(path),
        "params_sha256": sha256_member(path, "params.eqx"),
        "model": metadata["model"],
        "edge_sd_calibration": metadata.get("edge_sd_calibration"),
        "training_provenance": metadata.get("provenance"),
    }


def sha256_member(path: Path, member: str) -> str:
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive, archive.open(member) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_prediction(args: argparse.Namespace) -> None:
    metadata = json.loads(args.prepared_json.read_text())
    with np.load(args.prepared_npz, allow_pickle=False) as data:
        sample_covariance = np.asarray(data["sample_covariance"], dtype=np.float64)
        sample_xy = np.asarray(data["sample_xy"], dtype=np.float64)
        node_xy = np.asarray(data["node_xy"], dtype=np.float64)
        edges = np.asarray(data["edges"], dtype=np.int32)

    started = time.perf_counter()
    print(
        json.dumps(
            {
                "event": "dreems_start",
                "dataset_id": metadata["dataset_id"],
                "samples": int(sample_xy.shape[0]),
                "nodes": int(node_xy.shape[0]),
                "edges": int(edges.shape[0]),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    predictor = load_model(args.checkpoint)
    refinement = DREEMSMapRefinementConfig()
    result = predictor.predict(
        DREEMSInput(
            sample_covariance=sample_covariance,
            sample_coordinates=sample_xy,
            node_coordinates=node_xy,
            edges=edges,
            effective_snps=int(metadata["effective_snps"]),
            name=str(metadata["dataset_id"]),
            metadata=metadata,
        ),
        refinement_config=refinement,
    )
    elapsed = float(time.perf_counter() - started)
    refined = result.centered_refined_edge_log_mean
    if refined is None:
        raise RuntimeError("MAP refinement did not return an edge estimate")

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    temporary_npz = args.output_npz.with_suffix(args.output_npz.suffix + ".tmp")
    with temporary_npz.open("wb") as handle:
        np.savez_compressed(
            handle,
            prior_edge_log=result.centered_edge_log_mean,
            edge_log=refined,
            edge_log_sd=np.asarray(result.edge_log_sd, dtype=np.float64),
            raw_edge_log_sd=np.asarray(result.raw_edge_log_sd, dtype=np.float64),
            surface_weight=np.asarray(result.surface_weight, dtype=np.float64),
        )
    os.replace(temporary_npz, args.output_npz)

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
    }
    write_json_atomic(args.output_json, output_metadata)
    print(json.dumps(json_ready({"event": "dreems_done", **output_metadata}), sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.command == "predict":
        run_prediction(args)
    else:
        write_json_atomic(args.output, checkpoint_metadata(args.checkpoint))


if __name__ == "__main__":
    main()
