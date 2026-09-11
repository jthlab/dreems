"""Command line interface for DREEMS inference."""

# ruff: noqa: F722, F821, UP037

from __future__ import annotations

import argparse
import csv
import json
from io import TextIOWrapper
from pathlib import Path
from typing import Any

import numpy as np
from beartype import beartype
from jaxtyping import Shaped, jaxtyped

from dreems import feems
from dreems.evaluation import metrics
from dreems.inference import refinement
from dreems.inference.api import (
    DREEMSInput,
    DREEMSResult,
    _edge_records,
    genotype_covariance,
    load_model,
)
from dreems.neural import features
from dreems.utils.validation import _integer_array, _real_array, validate_graph_arrays
from dreems.utils.workflow_io import _atomic_output, json_ready


@jaxtyped(typechecker=beartype)
def load_array(path: Path) -> Shaped[np.ndarray, "*stored_axes"]:
    """Load one array without pickle; CSV rows stay two-dimensional."""

    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if len(payload.files) != 1:
                raise ValueError(
                    f"{path} must contain exactly one array; use .npy or a single-array .npz"
                )
            return payload[payload.files[0]]
    return np.loadtxt(path, delimiter=",", ndmin=2)


@beartype
def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    """Write an edge table as JSON or CSV without exposing a partial file."""
    if path.suffix == ".json":
        with _atomic_output(path) as handle:
            handle.write((json.dumps(records, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        return
    fieldnames = list(records[0]) if records else []
    with _atomic_output(path) as raw, TextIOWrapper(raw, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


@beartype
def json_default(value: object) -> object:
    """JSON callback for NumPy values and paths, with errors for unsupported types."""
    converted = json_ready(value)
    if converted is not value:
        return converted
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


@beartype
def write_diagnostics(path: Path, result: DREEMSResult) -> None:
    """Write inference metadata and MAP diagnostics separately from the edge table."""
    payload = {
        "metadata": dict(result.metadata),
        "refinement": result.refinement_diagnostics,
        "surface_weight": result.surface_weight,
    }
    contents = json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n"
    with _atomic_output(path) as handle:
        handle.write(contents.encode("utf-8"))


covariance_from_genotypes = genotype_covariance


@beartype
def build_dreems_input(args: argparse.Namespace) -> DREEMSInput:
    """Translate parsed CLI arguments into the same input used by the Python API."""

    if args.covariance is not None:
        covariance = load_array(args.covariance)
        effective_snps = args.effective_snps
    elif args.genotypes is not None:
        covariance, inferred_snps = covariance_from_genotypes(load_array(args.genotypes))
        effective_snps = inferred_snps if args.effective_snps is None else args.effective_snps
    else:
        raise ValueError("DREEMS inference requires --covariance or --genotypes")
    return DREEMSInput(
        sample_covariance=covariance,
        sample_coordinates=load_array(args.sample_coordinates),
        node_coordinates=load_array(args.node_coordinates),
        edges=load_array(args.edges),
        effective_snps=effective_snps,
    )


@beartype
def refinement_config_from_args(args: argparse.Namespace) -> refinement.DREEMSMapRefinementConfig:
    """Use argparse's typed values directly; the config validates their ranges."""

    return refinement.DREEMSMapRefinementConfig(
        steps=args.refinement_steps,
        learning_rate=args.refinement_learning_rate,
        smoothness_weight=args.refinement_smoothness_weight,
        sd_floor=args.refinement_sd_floor,
        trust_region_radius=args.trust_region_radius,
    )


@beartype
def run_dreems(args: argparse.Namespace) -> None:
    """Call the Python predictor and export the requested CLI outputs."""
    predictor = load_model(args.checkpoint)
    result = predictor.predict(
        build_dreems_input(args),
        refinement_config=refinement_config_from_args(args) if args.refine else None,
    )
    write_records(args.output, result.to_edge_records())
    if args.diagnostics_output is not None:
        write_diagnostics(args.diagnostics_output, result)


@beartype
def run_feems(args: argparse.Namespace) -> None:
    """Fit official FEEMS and export rates without inventing neural uncertainty."""
    if args.genotypes is None:
        raise ValueError("exact FEEMS fitting requires --genotypes")
    node_coordinates = _real_array(load_array(args.node_coordinates), name="node_coordinates")
    edges = _integer_array(load_array(args.edges), name="edges", dtype=np.int32)
    validate_graph_arrays(positions=node_coordinates, edges=edges)
    sample_coordinates = _real_array(load_array(args.sample_coordinates), name="sample_coordinates")
    sample_nodes = features.nearest_node_assignments(sample_coordinates, node_coordinates)
    graph = {"active_full": np.arange(node_coordinates.shape[0], dtype=np.int32), "edges": edges}
    fit = feems.fit_feems(
        genotypes=load_array(args.genotypes),
        sample_nodes=sample_nodes,
        graph=graph,
        active_positions=node_coordinates,
        config=feems.FEEMSConfig(mode=args.feems_mode),
    )
    records = _edge_records(
        edge_log_mean=metrics.centered(fit.edge_log_rates),
        edge_log_sd=None,
        edges=edges,
        positions=node_coordinates,
    )
    info = fit.info_dict()
    for record in records:
        record.update({key: value for key, value in info.items() if key != "mean_cv_error"})
    write_records(args.output, records)


@beartype
def build_parser() -> argparse.ArgumentParser:
    """Define the public inference arguments; statistics are validated by the API."""
    parser = argparse.ArgumentParser(description="Run DREEMS or FEEMS surface inference.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "DREEMS checkpoint for neural inference. Defaults to the packaged "
            "diffusion-band model with learned homogeneous-surface shrinkage."
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


@beartype
def main(argv: list[str] | None = None) -> None:
    """Run DREEMS by default or official FEEMS when ``--feems`` is supplied."""
    args = build_parser().parse_args(argv)
    if args.feems:
        run_feems(args)
    else:
        run_dreems(args)


if __name__ == "__main__":
    main()
