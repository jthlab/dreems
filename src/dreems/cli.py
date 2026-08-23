"""Command line interface for DREEMS inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from dreems import features, feems, refinement
from dreems.inference import (
    DREEMSInput,
    DREEMSResult,
    genotype_covariance,
    load_model,
)


def load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if path.suffix == ".npz":
        payload = np.load(path, allow_pickle=False)
        keys = list(payload.keys())
        if len(keys) != 1:
            raise ValueError(f"{path} contains multiple arrays; use .npy or a single-array .npz")
        return np.asarray(payload[keys[0]])
    return np.asarray(np.loadtxt(path, delimiter=","))


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
        return
    fieldnames = list(records[0]) if records else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def write_diagnostics(path: Path, result: DREEMSResult) -> None:
    payload = {
        "metadata": dict(result.metadata),
        "refinement": result.refinement_diagnostics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def covariance_from_genotypes(genotypes: np.ndarray) -> tuple[np.ndarray, int]:
    return genotype_covariance(genotypes)


def build_dreems_input(args: argparse.Namespace) -> DREEMSInput:
    if args.covariance is not None:
        covariance = load_array(args.covariance)
        effective_snps = args.effective_snps
    elif args.genotypes is not None:
        covariance, inferred_snps = covariance_from_genotypes(load_array(args.genotypes))
        effective_snps = args.effective_snps or inferred_snps
    else:
        raise ValueError("DREEMS inference requires --covariance or --genotypes")
    return DREEMSInput(
        sample_covariance=covariance,
        sample_coordinates=load_array(args.sample_coordinates),
        node_coordinates=load_array(args.node_coordinates),
        edges=load_array(args.edges),
        effective_snps=effective_snps,
    )


def refinement_config_from_args(args: argparse.Namespace) -> refinement.DREEMSMapRefinementConfig:
    return refinement.DREEMSMapRefinementConfig(
        steps=int(args.refinement_steps),
        learning_rate=float(args.refinement_learning_rate),
        smoothness_weight=float(args.refinement_smoothness_weight),
        sd_floor=float(args.refinement_sd_floor),
        trust_region_radius=float(args.trust_region_radius),
    )


def run_dreems(args: argparse.Namespace) -> None:
    predictor = load_model(args.checkpoint)
    result = predictor.predict(
        build_dreems_input(args),
        refinement_config=refinement_config_from_args(args) if args.refine else None,
    )
    write_records(args.output, result.to_edge_records())
    if args.diagnostics_output is not None:
        write_diagnostics(args.diagnostics_output, result)


def run_feems(args: argparse.Namespace) -> None:
    if args.genotypes is None:
        raise ValueError("exact FEEMS fitting requires --genotypes")
    node_coordinates = load_array(args.node_coordinates)
    edges = load_array(args.edges).astype(np.int32)
    sample_coordinates = load_array(args.sample_coordinates)
    sample_nodes = features.nearest_node_assignments(sample_coordinates, node_coordinates)
    graph = {"active_full": np.arange(node_coordinates.shape[0], dtype=np.int32), "edges": edges}
    fit = feems.fit_feems(
        genotypes=load_array(args.genotypes),
        sample_nodes=sample_nodes,
        graph=graph,
        active_positions=node_coordinates,
        config=feems.FEEMSConfig(mode=args.feems_mode),
    )
    result = DREEMSResult(
        edge_log_mean=fit.edge_log_rates,
        edge_log_sd=np.full(edges.shape[0], np.nan, dtype=np.float64),
        edges=edges,
        positions=node_coordinates,
        metadata=fit.info_dict(),
    )
    records = result.to_edge_records()
    info = fit.info_dict()
    for record in records:
        record.update({key: value for key, value in info.items() if key != "mean_cv_error"})
    write_records(args.output, records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DREEMS or FEEMS surface inference.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "DREEMS checkpoint for neural inference. Defaults to the packaged "
            "diffusion-band checkpoint."
        ),
    )
    parser.add_argument("--covariance", type=Path, help="Sample covariance matrix, .npy or CSV.")
    parser.add_argument("--genotypes", type=Path, help="Genotype dosage matrix, .npy or CSV.")
    parser.add_argument("--sample-coordinates", type=Path, required=True)
    parser.add_argument("--node-coordinates", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--effective-snps", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        help="Optional JSON path for refinement and trust-region diagnostics.",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Apply DREEMS-prior MAP covariance refinement after neural prediction.",
    )
    parser.add_argument(
        "--trust-region-radius",
        type=float,
        default=0.5,
        help="Uncertainty-scaled trust-region radius for --refine.",
    )
    parser.add_argument("--refinement-steps", type=int, default=50)
    parser.add_argument("--refinement-learning-rate", type=float, default=0.15)
    parser.add_argument("--refinement-smoothness-weight", type=float, default=1.0e-4)
    parser.add_argument("--refinement-sd-floor", type=float, default=0.25)
    parser.add_argument("--feems", action="store_true", help="Run FEEMS instead of DREEMS.")
    parser.add_argument(
        "--feems-mode",
        choices=["free_q", "fixed_q", "scalar_q"],
        default="free_q",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.feems:
        run_feems(args)
    else:
        run_dreems(args)


if __name__ == "__main__":
    main()
