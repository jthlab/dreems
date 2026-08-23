"""Generate one reproducible synthetic corpus example per worker process."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from dreems import simulation, simulation_config, workflow_io


def example_metadata(example: Mapping[str, Any], index: int) -> dict[str, Any]:
    graph = example["graph"]
    edges = np.asarray(graph["edges"], dtype=np.int32)
    positions = np.asarray(example["active_positions"], dtype=np.float64)
    sample_nodes = np.asarray(example["sample_nodes"], dtype=np.int32)
    genotypes = np.asarray(example["genotypes"]) if "genotypes" in example else None
    sample_covariance = (
        np.asarray(example["sample_covariance"]) if "sample_covariance" in example else None
    )
    sample_count = np.bincount(sample_nodes, minlength=positions.shape[0])
    sampled_nodes = np.unique(sample_nodes)
    num_snps = int(
        example.get(
            "effective_snps",
            example.get(
                "num_snps",
                genotypes.shape[1] if genotypes is not None else max(sample_nodes.shape[0], 2),
            ),
        )
    )
    return {
        "index": int(index),
        "simulator": str(example.get("simulator", "")),
        "graph_source": str(example.get("graph_source", "")),
        "sample_regime": str(example.get("sample_regime", "")),
        "sample_design": str(example.get("sample_design", "")),
        "sampling_base_design": str(example.get("sampling_base_design", "")),
        "sampling_association_mode": str(example.get("sampling_association_mode", "")),
        "sampling_counterfactual_group": int(example.get("sampling_counterfactual_group", -1)),
        "sampling_association_strength": float(
            example.get("sampling_association_strength", float("nan"))
        ),
        "sampling_inclusion_migration_correlation": float(
            example.get("sampling_inclusion_migration_correlation", float("nan"))
        ),
        "sampling_count_migration_correlation": float(
            example.get("sampling_count_migration_correlation", float("nan"))
        ),
        "surface_kind": str(example.get("surface_kind", "")),
        "mask_kind": str(example.get("mask_kind", "")),
        "wolflike_graph_training": bool(example.get("wolflike_graph_training", False)),
        "wolflike_graph_source": str(example.get("wolflike_graph_source", "")),
        "feems_lattice_training": bool(example.get("feems_lattice_training", False)),
        "feems_lattice_rows": int(example.get("feems_lattice_rows", 0)),
        "feems_lattice_columns": int(example.get("feems_lattice_columns", 0)),
        "feems_lattice_sample_prob": float(example.get("feems_lattice_sample_prob", float("nan"))),
        "island_attempts": int(example.get("island_attempts", 0)),
        "island_used_rust": bool(example.get("island_used_rust", False)),
        "island_min_allele_frequency": float(
            example.get("island_min_allele_frequency", float("nan"))
        ),
        "island_max_allele_frequency": float(
            example.get("island_max_allele_frequency", float("nan"))
        ),
        "island_marker_model": str(example.get("island_marker_model", "")),
        "has_genotypes": genotypes is not None,
        "has_sample_covariance": sample_covariance is not None,
        "effective_snps": int(example.get("effective_snps", num_snps)),
        "num_nodes": int(positions.shape[0]),
        "num_edges": int(edges.shape[0]),
        "num_samples": int(sample_nodes.shape[0]),
        "num_sampled_nodes": int(sampled_nodes.shape[0]),
        "sampled_node_fraction": float(sampled_nodes.shape[0] / max(positions.shape[0], 1)),
        "max_diploids_on_node": int(sample_count.max(initial=0)),
        "mean_diploids_per_sampled_node": float(
            sample_nodes.shape[0] / max(sampled_nodes.shape[0], 1)
        ),
        "num_snps": num_snps,
        "local_log_rate_std": float(example.get("local_log_rate_std", float("nan"))),
        "local_q_log_std": float(example.get("local_q_log_std", float("nan"))),
        "local_migration_gp_length_scale": float(
            example.get("local_migration_gp_length_scale", float("nan"))
        ),
        "local_q_gp_length_scale": float(example.get("local_q_gp_length_scale", float("nan"))),
        "local_diploids_per_node": int(example.get("local_diploids_per_node", 0)),
        "local_min_samples": int(example.get("local_min_samples", 0)),
        "local_max_samples": int(example.get("local_max_samples", 0)),
        "local_min_sample_fraction": example.get("local_min_sample_fraction"),
        "local_max_sample_fraction": example.get("local_max_sample_fraction"),
    }


def generate_example(
    *,
    config_path: Path,
    split: str,
    example_id: int,
    output: Path,
) -> None:
    config = workflow_io.load_config(config_path)
    sim = simulation_config.simulation_config(config)
    base_seed = int(sim.get("seed", config.get("seed", 12001)))
    seed_shard, sampling_mode = simulation.counterfactual_sampling_assignment(
        sim,
        example_id,
        split=split,
    )
    seed = simulation.raw_example_seed(
        base_seed=base_seed,
        split=split,
        shard=seed_shard,
        index=0,
    )
    print(
        json.dumps(
            {
                "stage": "generate_raw_example",
                "event": "start",
                "split": split,
                "example_id": example_id,
                "seed": seed,
                "sampling_counterfactual_group": seed_shard,
                "sampling_association_mode": sampling_mode,
            }
        ),
        flush=True,
    )
    example_sim = dict(sim)
    if sampling_mode is not None:
        example_sim["sampling_association_mode_override"] = sampling_mode
    example = simulation.simulate_raw_example_from_seed((seed, example_sim, split, example_id, 0))
    if sampling_mode is not None:
        example["sampling_counterfactual_group"] = int(seed_shard)
    workflow_io.write_pickle(output, [example])
    print(json.dumps({"stage": "generate_raw_example", "output": str(output)}), flush=True)


def pack_examples(
    *,
    inputs: Sequence[Path],
    output: Path,
    metadata: Path,
    config_path: Path,
    split: str,
    pack_index: int,
    start_index: int,
) -> None:
    """Combine ordered one-example worker outputs into a reusable raw pack."""

    examples: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for offset, path in enumerate(inputs):
        payload = workflow_io.read_pickle(path)
        if isinstance(payload, dict):
            example = payload
        elif isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
            example = payload[0]
        else:
            raise ValueError(f"expected exactly one raw example in {path}")
        example_index = start_index + offset
        examples.append(example)
        rows.append(example_metadata(example, example_index))

    if not examples:
        raise ValueError("cannot create an empty raw pack")
    workflow_io.write_pickle(output, examples)
    config = workflow_io.load_config(config_path)
    sim = simulation_config.simulation_config(config)
    workflow_io.write_json(
        metadata,
        {
            "stage": "pack_raw_examples",
            "split": split,
            "pack_index": pack_index,
            "start_index": start_index,
            "stop_index": start_index + len(examples),
            "examples": len(examples),
            "base_seed": int(sim.get("seed", config.get("seed", 12001))),
            "simulation_config": str(config_path),
            "simulation_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "rows": rows,
        },
    )
    print(
        json.dumps(
            {
                "stage": "pack_raw_examples",
                "split": split,
                "pack_index": pack_index,
                "examples": len(examples),
                "output": str(output),
            }
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--split", choices=["train", "test"], required=True)
    generate.add_argument("--example-id", type=int, required=True)
    generate.add_argument("--output", type=Path, required=True)

    pack = subparsers.add_parser("pack")
    pack.add_argument("--inputs", type=Path, nargs="+", required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--metadata", type=Path, required=True)
    pack.add_argument("--config", type=Path, required=True)
    pack.add_argument("--split", choices=["train", "test"], required=True)
    pack.add_argument("--pack-index", type=int, required=True)
    pack.add_argument("--start-index", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        generate_example(
            config_path=args.config,
            split=args.split,
            example_id=args.example_id,
            output=args.output,
        )
        return
    if args.command == "pack":
        pack_examples(
            inputs=args.inputs,
            output=args.output,
            metadata=args.metadata,
            config_path=args.config,
            split=args.split,
            pack_index=args.pack_index,
            start_index=args.start_index,
        )
        return
    raise AssertionError(f"unknown corpus command: {args.command}")


if __name__ == "__main__":
    main()
