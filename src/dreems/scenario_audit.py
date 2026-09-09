"""Generate the crossed FEEMS-style scenarios used to audit sampling bias."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from beartype import beartype

from dreems import corpus, island_sim, workflow_io
from dreems.geography import edge_log_rates_from_surface, feems_triangular_lattice_graph
from dreems.validation import _integer_array, _real_array

SURFACE_KINDS = {
    "barrier": "feems_heterogeneous_barrier",
    "anisotropy": "feems_anisotropic",
    "homogeneous": "feems_homogeneous",
}


@dataclass(frozen=True)
class ScenarioSpec:
    """One truth, sampling mask, and simulation replicate."""

    example_id: int
    replicate: int
    truth: str
    mask: str
    simulation_seed: int


@beartype
def _settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate scenario settings before counts or seeds can be silently truncated."""
    settings = config.get("scenario_audit")
    if not isinstance(settings, Mapping):
        raise ValueError("config must contain a scenario_audit mapping")
    settings = dict(settings)
    for key, minimum in (
        ("seed", 0),
        ("replicates", 1),
        ("rows", 2),
        ("columns", 2),
        ("diploids_per_sampled_node", 1),
        ("num_snps", 1),
        ("ploidy", 1),
    ):
        if key not in settings:
            continue
        value = _integer_array(np.asarray(settings[key]), name=key, dtype=np.int64)
        if value.ndim != 0 or value.item() < minimum:
            raise ValueError(f"scenario_audit.{key} must be a scalar integer >= {minimum}")
        settings[key] = value.item()
    for key in (
        "barrier_start",
        "barrier_end",
        "barrier_rate",
        "corridor_rate",
        "anisotropy_scaler",
        "random_sample_fraction",
    ):
        if key in settings:
            value = _real_array(settings[key], name=key)
            if value.ndim != 0:
                raise ValueError(f"scenario_audit.{key} must be a scalar")
            settings[key] = value.item()
    if settings.get("barrier_start", 2.5) > settings.get("barrier_end", 8.5):
        raise ValueError("barrier_start must not exceed barrier_end")
    for key in ("barrier_rate", "corridor_rate", "anisotropy_scaler"):
        if settings.get(key, 1.0) <= 0:
            raise ValueError(f"scenario_audit.{key} must be positive")
    if not 0 < settings.get("random_sample_fraction", 0.10) <= 1:
        raise ValueError("random_sample_fraction must be in (0, 1]")
    return settings


@beartype
def _names(settings: Mapping[str, Any], key: str) -> tuple[str, ...]:
    """Keep the configured factor order, which determines every example ID."""
    values = settings.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"scenario_audit.{key} must be a nonempty list")
    if any(not isinstance(value, str) for value in values):
        raise ValueError(f"scenario_audit.{key} must contain names, not numeric IDs")
    names = tuple(values)
    if len(set(names)) != len(names):
        raise ValueError(f"scenario_audit.{key} must not contain duplicates")
    return names


@beartype
def scenario_count(config: Mapping[str, Any]) -> int:
    """Return the number of examples in the crossed design."""

    settings = _settings(config)
    replicates = settings["replicates"]
    return replicates * len(_names(settings, "truths")) * len(_names(settings, "masks"))


@beartype
def scenario_spec(config: Mapping[str, Any], example_id: int) -> ScenarioSpec:
    """Decode a stable example ID into its crossed-design factors."""

    settings = _settings(config)
    truths = _names(settings, "truths")
    masks = _names(settings, "masks")
    total = scenario_count(config)
    if isinstance(example_id, bool):
        raise ValueError("example_id must be an integer, not a boolean")
    index = example_id
    if index < 0 or index >= total:
        raise ValueError(f"scenario example_id must be in [0, {total}), got {index}")

    cases_per_replicate = len(truths) * len(masks)
    replicate, within_replicate = divmod(index, cases_per_replicate)
    truth_index, mask_index = divmod(within_replicate, len(masks))
    base_seed = settings.get("seed", 20260830)
    simulation_seed = base_seed + 1_000_003 * replicate + 10_007 * truth_index + 101 * mask_index
    return ScenarioSpec(
        example_id=index,
        replicate=replicate,
        truth=truths[truth_index],
        mask=masks[mask_index],
        simulation_seed=simulation_seed,
    )


@beartype
def _surface(settings: Mapping[str, Any], truth: str) -> dict[str, Any]:
    """Translate a named test surface to the geographic generator's parameters."""
    if truth not in SURFACE_KINDS:
        raise ValueError(f"unknown scenario truth: {truth!r}")
    return {
        "kind": SURFACE_KINDS[truth],
        "barrier_startpt": settings.get("barrier_start", 2.5),
        "barrier_endpt": settings.get("barrier_end", 8.5),
        "barrier_w": settings.get("barrier_rate", 0.1),
        "corridor_w": settings.get("corridor_rate", 0.5),
        "anisotropy_scaler": settings.get("anisotropy_scaler", 5.0),
    }


@beartype
def _sampled_nodes(
    settings: Mapping[str, Any],
    positions: np.ndarray,
    *,
    mask: str,
    replicate: int,
) -> np.ndarray:
    """Cross identical location masks with every truth, using a replicate-specific seed."""
    num_nodes = int(positions.shape[0])
    start = settings.get("barrier_start", 2.5)
    end = settings.get("barrier_end", 8.5)
    if mask == "full":
        sampled = np.arange(num_nodes, dtype=np.int32)
    elif mask == "random_10pct":
        fraction = settings.get("random_sample_fraction", 0.10)
        count = max(2, int(np.floor(fraction * num_nodes)))
        mask_seed = int(settings.get("seed", 20260830)) + 7_919 * int(replicate)
        sampled = np.sort(
            np.random.default_rng(mask_seed).choice(num_nodes, size=count, replace=False)
        ).astype(np.int32)
    elif mask == "outer_slabs":
        sampled = np.flatnonzero((positions[:, 0] <= start) | (positions[:, 0] > end)).astype(
            np.int32
        )
    elif mask == "left_missing":
        sampled = np.flatnonzero(positions[:, 0] > start).astype(np.int32)
    else:
        raise ValueError(f"unknown scenario sampling mask: {mask!r}")
    if sampled.size < 2:
        raise ValueError(f"sampling mask {mask!r} selected fewer than two nodes")
    return sampled


@beartype
def scenario_components(
    config: Mapping[str, Any],
    example_id: int,
) -> tuple[ScenarioSpec, dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Construct the graph, truth, and sample nodes without simulating SNPs."""

    settings = _settings(config)
    spec = scenario_spec(config, example_id)
    graph, positions = feems_triangular_lattice_graph(
        settings.get("rows", 8),
        settings.get("columns", 12),
    )
    surface = _surface(settings, spec.truth)
    truth = edge_log_rates_from_surface(surface, positions, graph["edges"])
    sampled = _sampled_nodes(
        settings,
        positions,
        mask=spec.mask,
        replicate=spec.replicate,
    )
    diploids_per_node = settings.get("diploids_per_sampled_node", 10)
    sample_nodes = np.repeat(sampled, diploids_per_node).astype(np.int32)
    return spec, graph, positions, truth, sample_nodes


@beartype
def observed_node_count(config: Mapping[str, Any], example_id: int) -> int:
    """Return the number of FEEMS LOO folds required by one scenario."""

    _spec, _graph, _positions, _truth, sample_nodes = scenario_components(config, example_id)
    return int(np.unique(sample_nodes).size)


@beartype
def generate_example(
    *,
    config_path: Path,
    example_id: int,
    output: Path,
    metadata: Path,
) -> None:
    """Simulate one production-ascertained genotype matrix for the audit."""

    config = workflow_io.load_config(config_path)
    settings = _settings(config)
    spec, graph, positions, truth, sample_nodes = scenario_components(config, example_id)
    num_snps = settings.get("num_snps", 1000)
    min_maf = float(settings.get("min_allele_frequency", 0.05))
    max_maf = float(settings.get("max_allele_frequency", 0.95))
    marker_model = str(settings.get("marker_model", "conditioned_random_marker"))
    mutation_rate = float(settings.get("mutation_rate", 0.01))
    ploidy = settings.get("ploidy", 2)
    if ploidy != 2:
        raise ValueError("the released DREEMS scenario audit requires diploidy")

    island_sim.require_rust_backend()
    result = island_sim.simulate_island_model(
        sample_nodes=sample_nodes,
        num_demes=positions.shape[0],
        q=np.ones(positions.shape[0], dtype=np.float64),
        migration_edges=np.asarray(graph["edges"], dtype=np.int32),
        migration_rates=np.exp(np.asarray(truth, dtype=np.float64)),
        num_snps=num_snps,
        ploidy=ploidy,
        mutation_rate=mutation_rate,
        symmetric=True,
        seed=spec.simulation_seed,
        max_attempts=None,
        use_rust=True,
        min_allele_frequency=min_maf,
        max_allele_frequency=max_maf,
        marker_model=marker_model,
    )
    example: dict[str, Any] = {
        "graph": graph,
        "active_positions": positions,
        "edge_log_rates": truth,
        "node_log_q": np.zeros(positions.shape[0], dtype=np.float32),
        "sample_nodes": sample_nodes,
        "genotypes": result.genotypes.astype(np.float64),
        "num_snps": int(result.genotypes.shape[1]),
        "effective_snps": int(result.genotypes.shape[1]),
        "simulator": "island_coalescent",
        "surface_kind": SURFACE_KINDS[spec.truth],
        "sample_design": spec.mask,
        "sample_regime": spec.mask,
        "mask_kind": "feems_triangular_lattice",
        "graph_source": "feems_triangular_lattice",
        "height": settings.get("rows", 8),
        "width": settings.get("columns", 12),
        "feems_lattice_training": True,
        "feems_lattice_rows": settings.get("rows", 8),
        "feems_lattice_columns": settings.get("columns", 12),
        "feems_lattice_sample_prob": float(np.unique(sample_nodes).size / positions.shape[0]),
        "scenario_truth": spec.truth,
        "scenario_mask": spec.mask,
        "scenario_replicate": spec.replicate,
        "scenario_seed": spec.simulation_seed,
        "island_attempts": int(result.attempts),
        "island_used_rust": bool(result.used_rust),
        "island_min_allele_frequency": min_maf,
        "island_max_allele_frequency": max_maf,
        "island_marker_model": marker_model,
        "local_log_rate_std": float(np.std(truth)),
        "local_q_log_std": 0.0,
        "local_diploids_per_node": settings.get("diploids_per_sampled_node", 10),
    }
    row = corpus.example_metadata(example, spec.example_id)
    row.update(
        {
            "scenario_truth": spec.truth,
            "scenario_mask": spec.mask,
            "scenario_replicate": spec.replicate,
            "scenario_seed": spec.simulation_seed,
        }
    )
    workflow_io.write_pickle(output, [example])
    workflow_io.write_json(
        metadata,
        {
            "stage": "generate_scenario_audit_example",
            "config": str(config_path),
            "row": row,
        },
    )
    print(
        json.dumps(
            {
                "stage": "generate_scenario_audit_example",
                "example_id": spec.example_id,
                "truth": spec.truth,
                "mask": spec.mask,
                "replicate": spec.replicate,
                "observed_nodes": int(np.unique(sample_nodes).size),
                "samples": int(sample_nodes.size),
                "snps": int(result.genotypes.shape[1]),
                "attempts": int(result.attempts),
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


@beartype
def build_parser() -> argparse.ArgumentParser:
    """Expose one deterministic simulation per Snakemake job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--example-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser


@beartype
def main(argv: Sequence[str] | None = None) -> None:
    """Generate and record the selected scenario."""
    args = build_parser().parse_args(argv)
    generate_example(
        config_path=args.config,
        example_id=args.example_id,
        output=args.output,
        metadata=args.metadata,
    )


if __name__ == "__main__":
    main()
