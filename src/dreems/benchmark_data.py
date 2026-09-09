"""Read simulated benchmark cases and report their shared fields.

Both methods use the same graph, sample order, truth, and score definitions.
This module does not fit FEEMS or load a neural model.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from beartype import beartype

from dreems import metrics
from dreems.validation import _integer_array, _real_array, validate_genotype_arrays
from dreems.workflow_io import read_pickle

CASE_METADATA_KEYS = (
    "simulator",
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
    "scenario_truth",
    "scenario_mask",
    "scenario_replicate",
    "scenario_seed",
    "scenario_source_notebook_sha256",
    "scenario_raw_snps",
    "scenario_biallelic_snps",
    "scenario_maf_eligible_snps",
    "scenario_requested_retained_snps",
    "scenario_random_sample_fraction",
)


@beartype
def _load_example(path: Path) -> dict[str, Any]:
    """Read one trusted simulated case, including the historical single-item pack."""
    payload = read_pickle(path)
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError(f"expected one example in {path}, found {len(payload)}")
        payload = payload[0]
    if not isinstance(payload, dict):
        raise TypeError(f"expected an example dictionary in {path}")
    return payload


@beartype
def _example_arrays(
    payload: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate arrays before narrowing indices or calculating benchmark scores."""
    graph = payload.get("graph")
    if not isinstance(graph, Mapping):
        raise TypeError("H2H example must contain a graph mapping")
    positions = _real_array(payload["active_positions"], name="active_positions")
    edges = _integer_array(np.asarray(graph["edges"]), name="edges", dtype=np.int32)
    sample_nodes = _integer_array(
        np.asarray(payload["sample_nodes"]), name="sample_nodes", dtype=np.int32
    )
    genotypes = _real_array(payload["genotypes"], name="genotypes")
    truth = _real_array(payload.get("edge_log_rates", payload.get("truth")), name="truth")
    validate_genotype_arrays(
        positions=positions, edges=edges, sample_nodes=sample_nodes, genotypes=genotypes
    )
    if truth.shape != (edges.shape[0],):
        raise ValueError("truth must contain one log migration value per edge")
    return positions, edges, sample_nodes, genotypes, truth


@beartype
def _case_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Retain simulation settings used to stratify the benchmark reports."""
    return {key: payload[key] for key in CASE_METADATA_KEYS if key in payload}


@beartype
def _base_row(
    *,
    index: int,
    payload: Mapping[str, Any],
    positions: np.ndarray,
    edges: np.ndarray,
    sample_nodes: np.ndarray,
) -> dict[str, Any]:
    """Report graph size, sampling density, and simulation settings for either method."""
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


@beartype
def _metric_fields(prefix: str, truth: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    """Apply the common centered log-rate scoring convention."""
    return metrics.edge_log_metrics(truth, estimate).to_prefixed_dict(prefix)


@beartype
def _centered_rms(values: np.ndarray) -> float:
    """Describe surface amplitude after removing the arbitrary log-rate offset."""
    centered = metrics.centered(np.asarray(values, dtype=np.float64))
    finite = centered[np.isfinite(centered)]
    return float(np.sqrt(np.mean(finite * finite))) if finite.size else float("nan")
