"""Artifact-oriented CLI for reproducible DREEMS training and evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from dreems import checkpoints, corpus, counterfactual, features, metrics, simulation, train
from dreems import model as dreems_model
from dreems.workflow_io import (
    json_ready,
    load_config,
    optional_input_paths,
    read_pickle,
    required_input_paths,
    write_json,
    write_pickle,
)


def nested_mapping(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    # Config files are nested by topic. Missing sub-blocks intentionally fall
    # through to the dataclass defaults in train.py.
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping for config key {key!r}")
    return dict(value)


def dreems_config(config: Mapping[str, Any]) -> train.DREEMSTrainConfig:
    dreems = nested_mapping(config, "dreems")
    return train.DREEMSTrainConfig(
        architecture=train.DREEMSArchitectureConfig(**nested_mapping(dreems, "architecture")),
        features=train.DREEMSFeatureConfig(**nested_mapping(dreems, "features")),
        edge_conditioner=train.LinearizedResistanceConfig(
            **nested_mapping(dreems, "edge_conditioner")
        ),
        loss=train.DREEMSLossConfig(**nested_mapping(dreems, "loss")),
        optimizer=train.DREEMSOptimizerConfig(**nested_mapping(dreems, "optimizer")),
    )


def load_examples(path: Path) -> list[dict[str, Any]]:
    # Raw shards may contain one example or a list of examples. Downstream stages
    # only deal in lists.
    payload = read_pickle(path)
    if isinstance(payload, dict):
        return [payload]
    if not isinstance(payload, list):
        raise ValueError(f"expected {path} to contain a raw example dict or list of dicts")
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"expected all examples in {path} to be dictionaries")
    return payload


def load_prepared_shards(
    paths: Sequence[Path], max_examples: int | None = None
) -> list[dict[str, Any]]:
    # Prepared shards are already feature dictionaries. This concatenates shards
    # and optionally stops early for small validation subsets.
    out: list[dict[str, Any]] = []
    for path in paths:
        shard = read_pickle(path)
        if not isinstance(shard, list):
            raise ValueError(f"expected prepared shard {path} to contain a list")
        out.extend(shard)
        if max_examples is not None and len(out) >= max_examples:
            return out[:max_examples]
    return out


def prepared_dimensions(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    if not items:
        raise ValueError("no prepared examples")
    return {
        "examples": len(items),
        "max_nodes": max(np.asarray(item["node_x"]).shape[0] for item in items),
        "max_edges": max(np.asarray(item["edges"]).shape[0] for item in items),
        "node_dim": int(np.asarray(items[0]["node_x"]).shape[1]),
        "edge_dim": int(np.asarray(items[0]["edge_x"]).shape[1]),
    }


def prepared_dimensions_from_metadata(
    path: Path,
) -> dict[str, int] | None:
    metadata_candidates = [path.with_name(path.name + ".metadata.json")]
    if path.suffix:
        metadata_candidates.append(path.with_suffix(".metadata.json"))
    metadata_path = next(
        (candidate for candidate in metadata_candidates if candidate.exists()),
        None,
    )
    if metadata_path is None:
        return None
    payload = json.loads(metadata_path.read_text())
    dimensions = payload.get("prepared_dimensions")
    if not isinstance(dimensions, Mapping):
        return None
    required = (
        "examples",
        "max_nodes",
        "max_edges",
        "node_dim",
        "edge_dim",
    )
    missing = [key for key in required if key not in dimensions]
    if missing:
        return None
    return {key: int(dimensions[key]) for key in required}


def scan_prepared_dimensions(
    paths: Sequence[Path],
) -> dict[str, int]:
    # Streaming training cannot inspect all examples at once, but padding still
    # needs fixed dimensions. Scan shard metadata cheaply before batching.
    merged: dict[str, int] | None = None
    for path in paths:
        dims = prepared_dimensions_from_metadata(path)
        if dims is None:
            dims = prepared_dimensions(read_pickle(path))
        if merged is None:
            merged = dict(dims)
            continue
        merged["examples"] += dims["examples"]
        for key in ("max_nodes", "max_edges"):
            merged[key] = max(merged[key], dims[key])
        for key in ("node_dim", "edge_dim"):
            if merged[key] != dims[key]:
                raise ValueError(f"inconsistent {key}: {merged[key]} vs {dims[key]} in {path}")
    if merged is None:
        raise ValueError("no prepared shards")
    return merged


def pad_with_dimensions(
    items: list[dict[str, np.ndarray]], dims: Mapping[str, int]
) -> dict[str, np.ndarray]:
    return features.pad_examples(
        items,
        max_nodes=int(dims["max_nodes"]),
        max_edges=int(dims["max_edges"]),
    )


def normalization_accumulator_from_prepared(paths: Sequence[Path]) -> dict[str, object]:
    # Compute additive normalization sufficient statistics from prepared shards.
    # These shard-level stats are cheap to merge and avoid one global scan job.
    node_count = 0
    edge_count = 0
    y_count = 0
    q_count = 0
    node_sum = node_sumsq = edge_sum = edge_sumsq = None
    y_sum = y_sumsq = q_sum = q_sumsq = 0.0
    for path in paths:
        for item in read_pickle(path):
            node_x = np.asarray(item["node_x"], dtype=np.float64)
            edge_x = np.asarray(item["edge_x"], dtype=np.float64)
            y = np.asarray(item["y"], dtype=np.float64)
            q = np.asarray(item["q"], dtype=np.float64)
            if node_sum is None:
                node_sum = np.zeros(node_x.shape[1], dtype=np.float64)
                node_sumsq = np.zeros(node_x.shape[1], dtype=np.float64)
                edge_sum = np.zeros(edge_x.shape[1], dtype=np.float64)
                edge_sumsq = np.zeros(edge_x.shape[1], dtype=np.float64)
            node_sum += node_x.sum(axis=0)
            node_sumsq += np.square(node_x).sum(axis=0)
            edge_sum += edge_x.sum(axis=0)
            edge_sumsq += np.square(edge_x).sum(axis=0)
            node_count += int(node_x.shape[0])
            edge_count += int(edge_x.shape[0])
            y_sum += float(y.sum())
            y_sumsq += float(np.square(y).sum())
            y_count += int(y.size)
            q_sum += float(q.sum())
            q_sumsq += float(np.square(q).sum())
            q_count += int(q.size)
    if node_sum is None or node_sumsq is None or edge_sum is None or edge_sumsq is None:
        raise ValueError("no prepared training examples")
    return {
        "stage": "normalization_shard_stats",
        "inputs": [str(path) for path in paths],
        "node_sum": node_sum.astype(float).tolist(),
        "node_sumsq": node_sumsq.astype(float).tolist(),
        "edge_sum": edge_sum.astype(float).tolist(),
        "edge_sumsq": edge_sumsq.astype(float).tolist(),
        "y_sum": y_sum,
        "y_sumsq": y_sumsq,
        "q_sum": q_sum,
        "q_sumsq": q_sumsq,
        "node_count": node_count,
        "edge_count": edge_count,
        "y_count": y_count,
        "q_count": q_count,
    }


def merge_normalization_accumulators(
    accumulators: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    node_sum = node_sumsq = edge_sum = edge_sumsq = None
    y_sum = y_sumsq = q_sum = q_sumsq = 0.0
    node_count = edge_count = y_count = q_count = 0
    inputs: list[str] = []
    for accumulator in accumulators:
        local_node_sum = np.asarray(accumulator["node_sum"], dtype=np.float64)
        local_node_sumsq = np.asarray(accumulator["node_sumsq"], dtype=np.float64)
        local_edge_sum = np.asarray(accumulator["edge_sum"], dtype=np.float64)
        local_edge_sumsq = np.asarray(accumulator["edge_sumsq"], dtype=np.float64)
        if node_sum is None:
            node_sum = np.zeros_like(local_node_sum)
            node_sumsq = np.zeros_like(local_node_sumsq)
            edge_sum = np.zeros_like(local_edge_sum)
            edge_sumsq = np.zeros_like(local_edge_sumsq)
        if node_sum.shape != local_node_sum.shape:
            raise ValueError(
                f"inconsistent node feature dimension: {node_sum.shape} vs {local_node_sum.shape}"
            )
        if edge_sum.shape != local_edge_sum.shape:
            raise ValueError(
                f"inconsistent edge feature dimension: {edge_sum.shape} vs {local_edge_sum.shape}"
            )
        node_sum += local_node_sum
        node_sumsq += local_node_sumsq
        edge_sum += local_edge_sum
        edge_sumsq += local_edge_sumsq
        y_sum += float(accumulator["y_sum"])
        y_sumsq += float(accumulator["y_sumsq"])
        q_sum += float(accumulator["q_sum"])
        q_sumsq += float(accumulator["q_sumsq"])
        node_count += int(accumulator["node_count"])
        edge_count += int(accumulator["edge_count"])
        y_count += int(accumulator["y_count"])
        q_count += int(accumulator["q_count"])
        inputs.extend(str(path) for path in accumulator.get("inputs", []))
    if node_sum is None or node_sumsq is None or edge_sum is None or edge_sumsq is None:
        raise ValueError("no normalization shard stats")
    return {
        "stage": "normalization_merged_stats",
        "inputs": inputs,
        "node_sum": node_sum.astype(float).tolist(),
        "node_sumsq": node_sumsq.astype(float).tolist(),
        "edge_sum": edge_sum.astype(float).tolist(),
        "edge_sumsq": edge_sumsq.astype(float).tolist(),
        "y_sum": y_sum,
        "y_sumsq": y_sumsq,
        "q_sum": q_sum,
        "q_sumsq": q_sumsq,
        "node_count": node_count,
        "edge_count": edge_count,
        "y_count": y_count,
        "q_count": q_count,
    }


def finalize_normalization_stats(
    accumulator: Mapping[str, Any],
    *,
    std_floor: float,
) -> dict[str, object]:
    node_sum = np.asarray(accumulator["node_sum"], dtype=np.float64)
    node_sumsq = np.asarray(accumulator["node_sumsq"], dtype=np.float64)
    edge_sum = np.asarray(accumulator["edge_sum"], dtype=np.float64)
    edge_sumsq = np.asarray(accumulator["edge_sumsq"], dtype=np.float64)
    node_count = int(accumulator["node_count"])
    edge_count = int(accumulator["edge_count"])
    y_count = int(accumulator["y_count"])
    q_count = int(accumulator["q_count"])
    y_sum = float(accumulator["y_sum"])
    y_sumsq = float(accumulator["y_sumsq"])
    q_sum = float(accumulator["q_sum"])
    q_sumsq = float(accumulator["q_sumsq"])
    if node_count <= 0 or edge_count <= 0 or y_count <= 0 or q_count <= 0:
        raise ValueError("normalization counts must be positive")
    node_mean = node_sum / max(node_count, 1)
    edge_mean = edge_sum / max(edge_count, 1)
    node_var = np.maximum(node_sumsq / max(node_count, 1) - node_mean * node_mean, 0.0)
    edge_var = np.maximum(edge_sumsq / max(edge_count, 1) - edge_mean * edge_mean, 0.0)
    y_mean = y_sum / max(y_count, 1)
    q_mean = q_sum / max(q_count, 1)
    y_scale = float(np.sqrt(max(y_sumsq / max(y_count, 1) - y_mean * y_mean, 0.0)) + 1e-6)
    q_y_scale = float(np.sqrt(max(q_sumsq / max(q_count, 1) - q_mean * q_mean, 0.0)) + 1e-6)
    floor = max(float(std_floor), 1e-12)
    return {
        "node_mean": node_mean.astype(float).tolist(),
        "node_std": np.maximum(np.sqrt(node_var), floor).astype(float).tolist(),
        "edge_mean": edge_mean.astype(float).tolist(),
        "edge_std": np.maximum(np.sqrt(edge_var), floor).astype(float).tolist(),
        "normalization_std_floor": floor,
        "y_scale": y_scale,
        "q_y_scale": q_y_scale,
        "node_count": node_count,
        "edge_count": edge_count,
    }


def normalized_shard(path: Path, stats: Mapping[str, Any]) -> list[dict[str, np.ndarray]]:
    return features.apply_normalization(read_pickle(path), stats)


def iter_training_batches(
    paths: Sequence[Path],
    stats: Mapping[str, Any],
    *,
    dims: Mapping[str, int],
    batch_size: int,
    steps: int,
    seed: int,
) -> Iterable[dict[str, np.ndarray]]:
    # Keep counterfactual groups intact so their consistency term is observable
    # inside a batch. Ordinary examples remain independent singleton units.
    rng = np.random.default_rng(seed)
    produced = 0
    grouped_pending: list[list[dict[str, np.ndarray]]] = []
    singleton_pending: list[dict[str, np.ndarray]] = []

    def units(
        shard: list[dict[str, np.ndarray]],
    ) -> tuple[list[list[dict[str, np.ndarray]]], list[dict[str, np.ndarray]]]:
        grouped: dict[int, list[dict[str, np.ndarray]]] = {}
        singletons: list[dict[str, np.ndarray]] = []
        for item in shard:
            group = int(np.asarray(item.get("counterfactual_group", -1)))
            if group < 0:
                singletons.append(item)
            else:
                grouped.setdefault(group, []).append(item)
        groups = list(grouped.values())
        for group_items in groups:
            anchors = sum(
                float(np.asarray(item.get("counterfactual_anchor", 0.0))) > 0.0
                for item in group_items
            )
            if anchors != 1:
                raise ValueError("each counterfactual training group must contain one anchor")
            if batch_size % len(group_items) != 0:
                raise ValueError(
                    f"counterfactual group size {len(group_items)} must divide batch size "
                    f"{batch_size}"
                )
        return groups, singletons

    while produced < steps:
        for path_index in rng.permutation(len(paths)):
            shard = normalized_shard(paths[int(path_index)], stats)
            if not shard:
                continue
            groups, singletons = units(shard)
            grouped_pending.extend(groups[int(index)] for index in rng.permutation(len(groups)))
            singleton_pending.extend(
                singletons[int(index)] for index in rng.permutation(len(singletons))
            )
            while grouped_pending:
                group_size = len(grouped_pending[0])
                groups_per_batch = batch_size // group_size
                if len(grouped_pending) < groups_per_batch:
                    break
                batch_groups = grouped_pending[:groups_per_batch]
                del grouped_pending[:groups_per_batch]
                batch = [item for group_items in batch_groups for item in group_items]
                yield pad_with_dimensions(batch, dims)
                produced += 1
                if produced >= steps:
                    return
            while len(singleton_pending) >= batch_size:
                batch = singleton_pending[:batch_size]
                del singleton_pending[:batch_size]
                yield pad_with_dimensions(batch, dims)
                produced += 1
                if produced >= steps:
                    return


def stage_input_paths(paths: Sequence[Path], cache_dir: Path, *, label: str) -> list[Path]:
    destination_dir = cache_dir / label
    destination_dir.mkdir(parents=True)
    unique_sources = list(dict.fromkeys(path.resolve() for path in paths))
    total_bytes = sum(path.stat().st_size for path in unique_sources)
    reserve_bytes = 2 * 1024**3
    free_bytes = shutil.disk_usage(destination_dir).free
    if free_bytes < total_bytes + reserve_bytes:
        raise OSError(
            f"insufficient local space to stage {label}: need {total_bytes + reserve_bytes} "
            f"bytes including reserve, have {free_bytes} bytes in {destination_dir}"
        )
    started = time.time()
    print(
        json.dumps(
            {
                "stage": "stage_training_inputs",
                "event": "start",
                "label": label,
                "files": len(unique_sources),
                "references": len(paths),
                "bytes": total_bytes,
            }
        ),
        flush=True,
    )
    staged_by_source: dict[Path, Path] = {}
    for index, source in enumerate(unique_sources):
        destination = destination_dir / f"{index:05d}_{source.name}"
        shutil.copyfile(source, destination)
        staged_by_source[source] = destination
        completed = index + 1
        if completed == 1 or completed == len(unique_sources) or completed % 16 == 0:
            print(
                json.dumps(
                    {
                        "stage": "stage_training_inputs",
                        "event": "progress",
                        "label": label,
                        "completed_files": completed,
                        "files": len(unique_sources),
                        "elapsed_seconds": round(time.time() - started, 3),
                    }
                ),
                flush=True,
            )
    print(
        json.dumps(
            {
                "stage": "stage_training_inputs",
                "event": "done",
                "label": label,
                "files": len(unique_sources),
                "references": len(paths),
                "bytes": total_bytes,
                "elapsed_seconds": round(time.time() - started, 3),
            }
        ),
        flush=True,
    )
    return [staged_by_source[path.resolve()] for path in paths]


def edge_coverage(truth: np.ndarray, mean: np.ndarray, sd: np.ndarray, z: float) -> float:
    finite = np.isfinite(truth) & np.isfinite(mean) & np.isfinite(sd) & (sd > 0.0)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(np.abs(truth[finite] - mean[finite]) <= z * sd[finite]))


def prepare_raw_examples(
    raw_examples: list[dict[str, Any]],
    run_config: train.DREEMSTrainConfig,
    *,
    source: Path,
    index_offset: int = 0,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Prepare one raw shard while retaining its audit metadata."""

    kept: list[tuple[int, dict[str, Any]]] = []
    skipped: list[dict[str, Any]] = []
    for index, example in enumerate(raw_examples):
        if simulation.pairwise_conditionable_example(example):
            kept.append((index, example))
        else:
            skipped.append(corpus.example_metadata(example, index_offset + index))
    if not kept:
        raise ValueError(f"all {len(raw_examples)} raw examples are unconditionable: {source}")
    examples = [example for _index, example in kept]
    prepared = features.prepare_examples(
        examples,
        covariance_components=run_config.features.covariance_components,
        spatial_encoding=run_config.features.spatial_encoding,
    )
    prepared = train.append_training_edge_conditioner_features(prepared, examples, run_config)
    rows = [
        {
            **corpus.example_metadata(example, index_offset + original_index),
            "source": str(source),
        }
        for original_index, example in kept
    ]
    return prepared, skipped, rows


def raw_example_window(
    paths: Sequence[Path],
    *,
    start_index: int,
    max_examples: int | None,
) -> list[tuple[Path, int, list[dict[str, Any]]]]:
    """Read one contiguous example window from one or more ordered raw packs."""

    if start_index < 0:
        raise ValueError("start-index must be non-negative")
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max-examples must be positive")

    selected: list[tuple[Path, int, list[dict[str, Any]]]] = []
    skipped = 0
    remaining = max_examples
    for path in paths:
        examples = load_examples(path)
        if skipped + len(examples) <= start_index:
            skipped += len(examples)
            continue
        local_start = max(start_index - skipped, 0)
        local_stop = len(examples)
        if remaining is not None:
            local_stop = min(local_stop, local_start + remaining)
        local_examples = examples[local_start:local_stop]
        if local_examples:
            selected.append((path, local_start, local_examples))
            if remaining is not None:
                remaining -= len(local_examples)
                if remaining == 0:
                    break
        skipped += len(examples)

    selected_count = sum(len(examples) for _path, _offset, examples in selected)
    if selected_count == 0:
        raise ValueError(f"no raw examples selected at index {start_index}")
    if max_examples is not None and selected_count != max_examples:
        raise ValueError(
            f"requested {max_examples} raw examples at index {start_index}, found {selected_count}"
        )
    return selected


def command_prepare_shard(args: argparse.Namespace) -> None:
    # Raw simulation shard -> prepared feature shard. The sidecar metadata is
    # for quick/sanity checks without loading the pickle.
    run_config = dreems_config(load_config(args.config))
    raw_examples = load_examples(args.input)
    index_offset = 0
    if args.example_index is not None:
        index_offset = int(args.example_index)
        if not 0 <= index_offset < len(raw_examples):
            raise IndexError(
                f"example index {index_offset} is outside {args.input} "
                f"with {len(raw_examples)} examples"
            )
        raw_examples = [raw_examples[index_offset]]
    prepared, skipped, rows = prepare_raw_examples(
        raw_examples,
        run_config,
        source=args.input,
        index_offset=index_offset,
    )
    write_pickle(args.output, prepared)
    dims = prepared_dimensions(prepared)
    write_json(
        args.metadata,
        {
            "stage": "prepare_shard",
            "input": str(args.input),
            "output": str(args.output),
            "examples": len(prepared),
            "raw_examples": len(raw_examples),
            "skipped_unconditionable_examples": len(skipped),
            "skipped_rows": skipped,
            "prepared_dimensions": dims,
            "rows": rows,
        },
    )
    print(
        json.dumps(
            {
                "stage": "prepare_shard",
                "output": str(args.output),
                "raw_examples": len(raw_examples),
                "examples": len(prepared),
                "skipped_unconditionable_examples": len(skipped),
            }
        ),
        flush=True,
    )


def command_prepare_batch(args: argparse.Namespace) -> None:
    """Prepare several independent raw shards into one ordered training pack."""

    run_config = dreems_config(load_config(args.config))
    prepared: list[dict[str, np.ndarray]] = []
    skipped: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    raw_example_count = 0
    windows = raw_example_window(
        args.inputs,
        start_index=int(args.start_index),
        max_examples=args.max_examples,
    )
    for path, index_offset, raw_examples in windows:
        raw_example_count += len(raw_examples)
        shard_prepared, shard_skipped, shard_rows = prepare_raw_examples(
            raw_examples,
            run_config,
            source=path,
            index_offset=index_offset,
        )
        prepared.extend(shard_prepared)
        skipped.extend(shard_skipped)
        rows.extend(shard_rows)

    write_pickle(args.output, prepared)
    dims = prepared_dimensions(prepared)
    write_json(
        args.metadata,
        {
            "stage": "prepare_batch",
            "inputs": [str(path) for path in args.inputs],
            "output": str(args.output),
            "source_shards": len(windows),
            "raw_examples": raw_example_count,
            "examples": len(prepared),
            "skipped_unconditionable_examples": len(skipped),
            "skipped_rows": skipped,
            "prepared_dimensions": dims,
            "rows": rows,
        },
    )
    print(
        json.dumps(
            {
                "stage": "prepare_batch",
                "output": str(args.output),
                "source_shards": len(windows),
                "raw_examples": raw_example_count,
                "examples": len(prepared),
                "skipped_unconditionable_examples": len(skipped),
            }
        ),
        flush=True,
    )


def command_prepare_counterfactual(args: argparse.Namespace) -> None:
    """Prepare four paired masks from one reusable raw covariance example."""

    run_config = dreems_config(load_config(args.config))
    selected = counterfactual.read_selection(args.selection)
    slot = int(args.slot)
    if slot < 0 or slot >= len(selected):
        raise IndexError(f"counterfactual slot {slot} is outside [0, {len(selected)})")
    base = selected[slot]
    source = Path(base.raw_path)
    raw_examples = load_examples(source)
    if base.raw_index < 0 or base.raw_index >= len(raw_examples):
        raise IndexError(
            f"raw index {base.raw_index} is outside {source} with {len(raw_examples)} examples"
        )
    variants = counterfactual.counterfactual_variants(
        raw_examples[base.raw_index],
        group=base.slot,
        seed=int(args.seed) + 1_000_003 * base.slot,
    )
    prepared, skipped, rows = prepare_raw_examples(
        variants,
        run_config,
        source=source,
        index_offset=base.original_index,
    )
    if skipped or len(prepared) != len(counterfactual.COUNTERFACTUAL_MASKS):
        raise ValueError(
            f"counterfactual slot {slot} prepared {len(prepared)} masks and skipped {len(skipped)}"
        )
    for item, variant in zip(prepared, variants, strict=True):
        item["counterfactual_group"] = np.asarray(
            variant["sampling_counterfactual_group"],
            dtype=np.int32,
        )
        item["counterfactual_anchor"] = np.asarray(
            variant["sampling_counterfactual_anchor"],
            dtype=np.float32,
        )
        item["counterfactual_mask"] = np.asarray(
            counterfactual.COUNTERFACTUAL_MASK_INDEX[str(variant["sampling_counterfactual_mask"])],
            dtype=np.int32,
        )
        item["sampled_node_fraction"] = np.asarray(
            variant["sampled_node_fraction"],
            dtype=np.float32,
        )
    write_pickle(args.output, prepared)
    dimensions = prepared_dimensions(prepared)
    write_json(
        args.metadata,
        {
            "stage": "prepare_counterfactual",
            "slot": slot,
            "source": str(source),
            "source_index": base.raw_index,
            "base": asdict(base),
            "masks": list(counterfactual.COUNTERFACTUAL_MASKS),
            "prepared_dimensions": dimensions,
            "rows": rows,
        },
    )
    print(
        json.dumps(
            {
                "stage": "prepare_counterfactual",
                "slot": slot,
                "source": str(source),
                "examples": len(prepared),
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def command_pack_prepared(args: argparse.Namespace) -> None:
    inputs = required_input_paths(
        args.inputs,
        args.input_manifest,
        argument_name="--inputs",
    )
    start = int(args.start_index)
    if start < 0:
        raise ValueError("start-index must be non-negative")
    if args.max_inputs is None:
        stop = len(inputs)
    else:
        if int(args.max_inputs) <= 0:
            raise ValueError("max-inputs must be positive")
        stop = min(len(inputs), start + int(args.max_inputs))
    selected = inputs[start:stop]
    if not selected:
        raise ValueError(f"no prepared inputs selected from {len(inputs)} paths at index {start}")

    prepared = load_prepared_shards(selected)
    dimensions = prepared_dimensions(prepared)
    write_pickle(args.output, prepared)
    write_json(
        args.metadata,
        {
            "stage": "pack_prepared",
            "output": str(args.output),
            "source_shards": len(selected),
            "source_start_index": start,
            "source_stop_index": stop,
            "examples": len(prepared),
            "prepared_dimensions": dimensions,
        },
    )
    print(
        json.dumps(
            {
                "stage": "pack_prepared",
                "output": str(args.output),
                "source_shards": len(selected),
                "examples": len(prepared),
            }
        ),
        flush=True,
    )


def command_normalization_shard_stats(args: argparse.Namespace) -> None:
    stats = normalization_accumulator_from_prepared(args.inputs)
    write_json(args.output, stats)
    print(
        json.dumps(
            {
                "stage": "normalization_shard_stats",
                "output": str(args.output),
                "node_count": stats["node_count"],
                "edge_count": stats["edge_count"],
            }
        ),
        flush=True,
    )


def command_merge_normalization(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    run_config = dreems_config(cfg)
    inputs = required_input_paths(
        args.inputs,
        args.input_manifest,
        argument_name="--inputs",
    )
    accumulator = merge_normalization_accumulators(
        [json.loads(path.read_text()) for path in inputs]
    )
    stats = finalize_normalization_stats(
        accumulator,
        std_floor=run_config.features.normalization_std_floor,
    )
    write_json(args.output, stats)
    print(
        json.dumps(
            {
                "stage": "merge_normalization",
                "output": str(args.output),
                "node_count": stats["node_count"],
                "edge_count": stats["edge_count"],
                "y_scale": stats["y_scale"],
                "q_y_scale": stats["q_y_scale"],
            }
        ),
        flush=True,
    )


def _command_train(args: argparse.Namespace, cache_dir: Path | None) -> None:
    # Train from prepared shards and write one self-contained checkpoint plus a
    # small JSON summary for Snakemake and later inspection.
    cfg = load_config(args.config)
    run_config = dreems_config(cfg)
    inputs = required_input_paths(
        args.inputs,
        args.input_manifest,
        argument_name="--inputs",
    )
    validation_inputs = optional_input_paths(
        args.validation_inputs,
        args.validation_input_manifest,
    )
    dims = scan_prepared_dimensions(inputs)
    if cache_dir is not None:
        inputs = stage_input_paths(inputs, cache_dir, label="train")
        validation_inputs = stage_input_paths(
            validation_inputs,
            cache_dir,
            label="validation",
        )
    init_checkpoint = (
        checkpoints.load_checkpoint(args.init_checkpoint)
        if args.init_checkpoint is not None
        else None
    )
    inherited_edge_sd_calibration: Mapping[str, Any] | None = None
    if init_checkpoint is None:
        stats = json.loads(args.normalization.read_text())
        init_params = None
    else:
        model_metadata = init_checkpoint.model
        expected_architecture = train.architecture_metadata(run_config.architecture)
        checkpoint_architecture = model_metadata.get("architecture")
        if not isinstance(checkpoint_architecture, dict):
            raise ValueError("init checkpoint is missing architecture metadata")
        checkpoint_architecture = train.canonical_architecture_metadata(checkpoint_architecture)
        expected_architecture = train.canonical_architecture_metadata(expected_architecture)
        extending_with_gates = (
            train.core_architecture_metadata(checkpoint_architecture)
            == train.core_architecture_metadata(expected_architecture)
            and (
                not bool(checkpoint_architecture["initializer_gate"])
                or bool(expected_architecture["initializer_gate"])
            )
            and (
                not bool(checkpoint_architecture["surface_gate"])
                or bool(expected_architecture["surface_gate"])
            )
        )
        if checkpoint_architecture != expected_architecture and not extending_with_gates:
            raise ValueError(
                "init checkpoint architecture does not match the training config: "
                f"{checkpoint_architecture} != {expected_architecture}"
            )
        expected_features = train.feature_metadata(run_config.features)
        checkpoint_features = model_metadata.get("features")
        if not isinstance(checkpoint_features, dict):
            raise ValueError("init checkpoint is missing feature metadata")
        checkpoint_features = train.canonical_feature_metadata(checkpoint_features)
        if checkpoint_features != expected_features:
            raise ValueError(
                "init checkpoint feature metadata does not match the training config: "
                f"{checkpoint_features} != {expected_features}"
            )
        if int(model_metadata["node_dim"]) != int(dims["node_dim"]):
            raise ValueError(
                f"init checkpoint node_dim={model_metadata['node_dim']} does not match "
                f"prepared node_dim={dims['node_dim']}"
            )
        if int(model_metadata["edge_dim"]) != int(dims["edge_dim"]):
            raise ValueError(
                f"init checkpoint edge_dim={model_metadata['edge_dim']} does not match "
                f"prepared edge_dim={dims['edge_dim']}"
            )
        stats = init_checkpoint.normalization
        init_params = init_checkpoint.params
        if run_config.optimizer.trainable_scope in {"initializer_gate", "surface_gate"}:
            calibration_payload = init_checkpoint.payload.get("edge_sd_calibration")
            if calibration_payload is not None:
                if not isinstance(calibration_payload, Mapping):
                    raise ValueError("init checkpoint has invalid edge SD calibration metadata")
                inherited_edge_sd_calibration = dict(calibration_payload)
    metadata_arrays_stub = {
        "node_x": np.zeros((1, 1, int(dims["node_dim"])), dtype=np.float32),
        "edge_x": np.zeros((1, 1, int(dims["edge_dim"])), dtype=np.float32),
    }
    validation_arrays = None
    validation_limit = int(cfg.get("train_validation_examples", 0) or 0)
    if validation_inputs and validation_limit > 0:
        validation_prepared = load_prepared_shards(validation_inputs, max_examples=validation_limit)
        validation_prepared = features.apply_normalization(validation_prepared, stats)
        validation_arrays = features.pad_examples(validation_prepared)
    history_path = args.summary.with_name("training_history.jsonl")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.unlink(missing_ok=True)

    def best_checkpoint_callback(
        step: int,
        params: dreems_model.DREEMS,
        losses: list[float],
        row: Mapping[str, Any],
        metadata_arrays: Mapping[str, np.ndarray],
    ) -> None:
        if row.get("validation_best") is not True:
            return
        best_path = args.output.with_name("checkpoint.best.dreems.zip")
        best_summary_path = args.output.with_name("checkpoint.best.summary.json")
        early_stopping_metric = run_config.optimizer.early_stopping_metric
        best_validation_loss = row.get(early_stopping_metric)
        best_validation_metrics = dict(row)
        train.save_checkpoint(
            best_path,
            params=params,
            normalization=stats,
            config=run_config,
            train_losses=losses,
            train_arrays=dict(metadata_arrays),
            training_run={
                "final_step": step,
                "best_step": step,
                "best_validation_loss": best_validation_loss,
                "early_stopping_metric": early_stopping_metric,
                "best_validation_metric": best_validation_loss,
                "best_validation_metrics": best_validation_metrics,
                "stopped_early": False,
                "stop_reason": None,
                "restored_best_validation": False,
                "history": [dict(row)],
            },
            edge_sd_calibration=inherited_edge_sd_calibration,
        )
        write_json(
            best_summary_path,
            {
                "stage": "train_best_checkpoint",
                "checkpoint": str(best_path),
                "step": step,
                "best_validation_loss": best_validation_loss,
                "early_stopping_metric": early_stopping_metric,
                "best_validation_metric": best_validation_loss,
                "best_validation_edge_nll": row.get("validation_edge_nll"),
                "best_validation_metrics": best_validation_metrics,
                "row": dict(row),
            },
        )

    def training_progress_callback(
        step: int,
        params: dreems_model.DREEMS,
        losses: list[float],
        row: Mapping[str, Any],
        metadata_arrays: Mapping[str, np.ndarray],
    ) -> None:
        with history_path.open("a") as handle:
            handle.write(json.dumps(json_ready(row), sort_keys=True) + "\n")
        best_checkpoint_callback(step, params, losses, row, metadata_arrays)

    if args.stream:
        # Streaming keeps memory bounded by loading one prepared shard at a time.
        metadata_arrays = metadata_arrays_stub
        train_result = train.train_from_batches(
            config=run_config,
            node_dim=int(dims["node_dim"]),
            edge_dim=int(dims["edge_dim"]),
            batches=iter_training_batches(
                inputs,
                stats,
                dims=dims,
                batch_size=run_config.optimizer.batch_size,
                steps=run_config.optimizer.steps,
                seed=run_config.optimizer.seed + 10,
            ),
            init_params=init_params,
            validation_arrays=validation_arrays,
            checkpoint_callback=lambda step, params, losses, row: training_progress_callback(
                step,
                params,
                losses,
                row,
                metadata_arrays,
            ),
        )
    else:
        # The eager path is useful for smoke tests and small local runs.
        prepared = load_prepared_shards(inputs)
        prepared = features.apply_normalization(prepared, stats)
        train_arrays = features.pad_examples(prepared)
        metadata_arrays = train_arrays
        train_result = train.train_from_arrays(
            train_arrays,
            config=run_config,
            init_params=init_params,
            validation_arrays=validation_arrays,
            checkpoint_callback=lambda step, params, losses, row: training_progress_callback(
                step,
                params,
                losses,
                row,
                metadata_arrays,
            ),
        )

    train.save_checkpoint(
        args.output,
        params=train_result.params,
        normalization=stats,
        config=run_config,
        train_losses=train_result.losses,
        train_arrays=metadata_arrays,
        training_run=train_result.summary(),
        edge_sd_calibration=inherited_edge_sd_calibration,
    )
    training_run = train_result.summary()
    best_validation_metrics = train_result.best_validation_metrics or {}
    write_json(
        args.summary,
        {
            "stage": "train",
            "checkpoint": str(args.output),
            "train_shards": [str(path) for path in inputs],
            "prepared_dimensions": dims,
            "losses": train_result.losses,
            "final_loss": train_result.losses[-1] if train_result.losses else None,
            "stream": bool(args.stream),
            "init_checkpoint": (
                str(args.init_checkpoint) if args.init_checkpoint is not None else None
            ),
            "normalization_source": (
                str(args.init_checkpoint)
                if args.init_checkpoint is not None
                else str(args.normalization)
            ),
            "training_run": training_run,
            "best_step": train_result.best_step,
            "best_validation_loss": train_result.best_validation_loss,
            "early_stopping_metric": train_result.early_stopping_metric,
            "best_validation_metric": train_result.best_validation_metric,
            "best_validation_edge_nll": best_validation_metrics.get("validation_edge_nll"),
            "best_validation_metrics": train_result.best_validation_metrics,
            "stopped_early": train_result.stopped_early,
            "stop_reason": train_result.stop_reason,
            "restored_best_validation": train_result.restored_best_validation,
        },
    )
    print(json.dumps({"stage": "train", "output": str(args.output)}), flush=True)


def command_train(args: argparse.Namespace) -> None:
    if args.cache_dir is None:
        _command_train(args, None)
        return

    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=False)
    try:
        _command_train(args, cache_dir)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def command_evaluate(args: argparse.Namespace) -> None:
    # Evaluate prepared shards with the saved checkpoint. Examples are sorted by
    # node count inside each batch to reduce padding waste.
    started = time.time()
    inputs = required_input_paths(
        args.inputs,
        args.input_manifest,
        argument_name="--inputs",
    )
    print(
        json.dumps(
            {
                "stage": "evaluate",
                "event": "start",
                "checkpoint": str(args.checkpoint),
                "inputs": [str(path) for path in inputs],
                "batch_size": int(args.batch_size),
                "max_examples": args.max_examples,
            }
        ),
        flush=True,
    )
    loaded = checkpoints.load_checkpoint(args.checkpoint)
    print(
        json.dumps(
            {
                "stage": "evaluate",
                "event": "checkpoint_loaded",
                "elapsed_seconds": round(time.time() - started, 3),
            }
        ),
        flush=True,
    )
    stats = loaded.normalization
    rows: list[dict[str, Any]] = []
    prediction_replicates: list[int] = []
    edge_offsets = [0]
    edge_truth_parts: list[np.ndarray] = []
    edge_mean_parts: list[np.ndarray] = []
    edge_sd_parts: list[np.ndarray] = []
    q_offsets = [0]
    q_truth_parts: list[np.ndarray] = []
    q_mean_parts: list[np.ndarray] = []
    q_sd_parts: list[np.ndarray] = []
    replicate = int(args.replicate_offset)
    max_examples = None if args.max_examples is None else int(args.max_examples)
    for shard_index, path in enumerate(inputs):
        if max_examples is not None and len(rows) >= max_examples:
            break
        shard_started = time.time()
        print(
            json.dumps(
                {
                    "stage": "evaluate",
                    "event": "shard_start",
                    "shard_index": shard_index,
                    "input": str(path),
                }
            ),
            flush=True,
        )
        prepared = normalized_shard(path, stats)
        indexed = list(enumerate(prepared))
        if max_examples is not None:
            indexed = indexed[: max(0, max_examples - len(rows))]
        indexed = sorted(indexed, key=lambda pair: pair[1]["node_x"].shape[0])
        print(
            json.dumps(
                {
                    "stage": "evaluate",
                    "event": "shard_loaded",
                    "shard_index": shard_index,
                    "examples": len(indexed),
                    "elapsed_seconds": round(time.time() - shard_started, 3),
                }
            ),
            flush=True,
        )
        total_batches = math.ceil(len(indexed) / int(args.batch_size))
        for batch_index, start in enumerate(range(0, len(indexed), int(args.batch_size)), start=1):
            chunk = indexed[start : start + int(args.batch_size)]
            local_indices = [index for index, _item in chunk]
            arrays = features.pad_examples([item for _index, item in chunk])
            edge_mean, edge_logvar, q_mean, q_logvar = dreems_model.predict(
                loaded.params,
                arrays,
                y_scale=float(stats["y_scale"]),
                q_y_scale=float(stats["q_y_scale"]),
                model=loaded.model,
            )
            edge_sd = np.sqrt(np.exp(np.clip(edge_logvar, -30.0, 30.0)))
            q_sd = np.sqrt(np.exp(np.clip(q_logvar, -30.0, 30.0)))
            for local, original in enumerate(local_indices):
                # Stored targets are normalized; convert them back before
                # computing publication-facing metrics.
                edge_mask = arrays["edge_mask"][local] > 0.0
                node_mask = arrays["node_mask"][local] > 0.0
                truth = np.asarray(arrays["y"][local][edge_mask], dtype=np.float64) * float(
                    stats["y_scale"]
                )
                pred = np.asarray(edge_mean[local][edge_mask], dtype=np.float64)
                raw_sd = np.asarray(edge_sd[local][edge_mask], dtype=np.float64)
                sampled_fraction = float(arrays["sampled_node_fraction"][local])
                sd = loaded.edge_sd_calibration.apply(
                    raw_sd,
                    sampled_node_fraction=(
                        sampled_fraction if np.isfinite(sampled_fraction) else None
                    ),
                )
                q_truth = np.asarray(arrays["q"][local][node_mask], dtype=np.float64) * float(
                    stats["q_y_scale"]
                )
                q_pred = np.asarray(q_mean[local][node_mask], dtype=np.float64)
                q_sd_local = np.asarray(q_sd[local][node_mask], dtype=np.float64)
                edge_metrics = metrics.edge_log_metrics(truth, pred).to_prefixed_dict("edge_log")
                q_metrics = metrics.edge_log_metrics(q_truth, q_pred).to_prefixed_dict("q_log")
                replicate_id = replicate + int(original)
                if args.predictions_output is not None:
                    prediction_replicates.append(replicate_id)
                    edge_truth_parts.append(truth.astype(np.float32))
                    edge_mean_parts.append(pred.astype(np.float32))
                    edge_sd_parts.append(sd.astype(np.float32))
                    edge_offsets.append(edge_offsets[-1] + int(truth.size))
                    q_truth_parts.append(q_truth.astype(np.float32))
                    q_mean_parts.append(q_pred.astype(np.float32))
                    q_sd_parts.append(q_sd_local.astype(np.float32))
                    q_offsets.append(q_offsets[-1] + int(q_truth.size))
                rows.append(
                    {
                        "replicate": replicate_id,
                        "replicate_within_shard": int(original),
                        **edge_metrics,
                        **q_metrics,
                        "edge_sd_mean": float(np.mean(sd)) if sd.size else float("nan"),
                        "edge_50_coverage": edge_coverage(truth, pred, sd, 0.67448975),
                        "edge_80_coverage": edge_coverage(truth, pred, sd, 1.28155157),
                        "edge_95_coverage": edge_coverage(truth, pred, sd, 1.95996398),
                        "q_sd_mean": float(np.mean(q_sd_local))
                        if q_sd_local.size
                        else float("nan"),
                        "q_80_coverage": edge_coverage(q_truth, q_pred, q_sd_local, 1.28155157),
                        "counterfactual_group": int(arrays["counterfactual_group"][local]),
                        "counterfactual_mask": int(arrays["counterfactual_mask"][local]),
                        "sampled_node_fraction": sampled_fraction,
                        "source_shard": str(path),
                    }
                )
            if batch_index == 1 or batch_index == total_batches or batch_index % 10 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "evaluate",
                            "event": "batch_done",
                            "shard_index": shard_index,
                            "batch": batch_index,
                            "batches": total_batches,
                            "rows_so_far": len(rows),
                            "elapsed_seconds": round(time.time() - shard_started, 3),
                        }
                    ),
                    flush=True,
                )
        print(
            json.dumps(
                {
                    "stage": "evaluate",
                    "event": "shard_done",
                    "shard_index": shard_index,
                    "rows_so_far": len(rows),
                    "elapsed_seconds": round(time.time() - shard_started, 3),
                }
            ),
            flush=True,
        )
    if not rows:
        raise ValueError("no evaluation rows")
    rows.sort(key=lambda row: int(row["replicate"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(json_ready(rows))
    means = summarize_numeric_rows(rows)
    write_json(
        args.output_dir / "summary.json",
        {
            "stage": "evaluate",
            "checkpoint": str(args.checkpoint),
            "examples": len(rows),
            "batch_size": int(args.batch_size),
            "max_examples": max_examples,
            "means": means,
            "rows": rows,
        },
    )
    if args.predictions_output is not None:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.predictions_output,
            replicate=np.asarray(prediction_replicates, dtype=np.int64),
            edge_offsets=np.asarray(edge_offsets, dtype=np.int64),
            edge_truth=np.concatenate(edge_truth_parts).astype(np.float32, copy=False),
            edge_mean=np.concatenate(edge_mean_parts).astype(np.float32, copy=False),
            edge_sd=np.concatenate(edge_sd_parts).astype(np.float32, copy=False),
            q_offsets=np.asarray(q_offsets, dtype=np.int64),
            q_truth=np.concatenate(q_truth_parts).astype(np.float32, copy=False),
            q_mean=np.concatenate(q_mean_parts).astype(np.float32, copy=False),
            q_sd=np.concatenate(q_sd_parts).astype(np.float32, copy=False),
        )
    print(
        json.dumps(
            {
                "stage": "evaluate",
                "event": "done",
                "examples": len(rows),
                "max_examples": max_examples,
                "output": str(csv_path),
                "elapsed_seconds": round(time.time() - started, 3),
            }
        ),
        flush=True,
    )


def summarize_numeric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    # Summaries are intentionally generic: any numeric row field gets mean/sd.
    out: dict[str, float] = {}
    keys = sorted({key for row in rows for key in row})
    for key in keys:
        if key in {"source_shard"}:
            continue
        try:
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            continue
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            out[f"{key}_mean"] = float("nan")
            out[f"{key}_sd"] = float("nan")
        else:
            out[f"{key}_mean"] = float(np.mean(finite))
            out[f"{key}_sd"] = float(np.std(finite))
    return out


def command_aggregate_evaluations(args: argparse.Namespace) -> None:
    # Merge per-shard evaluation JSON files into one CSV/JSON report.
    rows: list[dict[str, Any]] = []
    for path in args.inputs:
        payload = json.loads(path.read_text())
        rows.extend(payload.get("rows", []))
    if not rows:
        raise ValueError("no rows to aggregate")
    for index, row in enumerate(rows):
        row.setdefault("replicate_global", index)
    rows.sort(key=lambda row: int(row["replicate_global"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(json_ready(rows))
    write_json(
        args.output_dir / "summary.json",
        {
            "stage": "aggregate_evaluations",
            "examples": len(rows),
            "means": summarize_numeric_rows(rows),
            "rows": rows,
            "inputs": [str(path) for path in args.inputs],
        },
    )
    print(json.dumps({"stage": "aggregate_evaluations", "examples": len(rows)}), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-shard")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--metadata", type=Path, required=True)
    prepare.add_argument("--example-index", type=int)
    prepare.set_defaults(func=command_prepare_shard)

    prepare_batch = subparsers.add_parser("prepare-batch")
    prepare_batch.add_argument("--config", type=Path, required=True)
    prepare_batch.add_argument("--inputs", type=Path, nargs="+", required=True)
    prepare_batch.add_argument("--output", type=Path, required=True)
    prepare_batch.add_argument("--metadata", type=Path, required=True)
    prepare_batch.add_argument("--start-index", type=int, default=0)
    prepare_batch.add_argument("--max-examples", type=int)
    prepare_batch.set_defaults(func=command_prepare_batch)

    prepare_counterfactual = subparsers.add_parser("prepare-counterfactual")
    prepare_counterfactual.add_argument("--config", type=Path, required=True)
    prepare_counterfactual.add_argument("--selection", type=Path, required=True)
    prepare_counterfactual.add_argument("--slot", type=int, required=True)
    prepare_counterfactual.add_argument("--seed", type=int, required=True)
    prepare_counterfactual.add_argument("--output", type=Path, required=True)
    prepare_counterfactual.add_argument("--metadata", type=Path, required=True)
    prepare_counterfactual.set_defaults(func=command_prepare_counterfactual)

    pack = subparsers.add_parser("pack-prepared")
    pack.add_argument("--inputs", type=Path, nargs="+", default=None)
    pack.add_argument("--input-manifest", type=Path, default=None)
    pack.add_argument("--start-index", type=int, default=0)
    pack.add_argument("--max-inputs", type=int, default=None)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--metadata", type=Path, required=True)
    pack.set_defaults(func=command_pack_prepared)

    norm_shard = subparsers.add_parser("normalization-shard-stats")
    norm_shard.add_argument("--inputs", type=Path, nargs="+", required=True)
    norm_shard.add_argument("--output", type=Path, required=True)
    norm_shard.set_defaults(func=command_normalization_shard_stats)

    merge_norm = subparsers.add_parser("merge-normalization")
    merge_norm.add_argument("--config", type=Path, required=True)
    merge_norm.add_argument("--inputs", type=Path, nargs="+", default=None)
    merge_norm.add_argument("--input-manifest", type=Path, default=None)
    merge_norm.add_argument("--output", type=Path, required=True)
    merge_norm.set_defaults(func=command_merge_normalization)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--normalization", type=Path, required=True)
    train_parser.add_argument("--inputs", type=Path, nargs="+", default=None)
    train_parser.add_argument("--input-manifest", type=Path, default=None)
    train_parser.add_argument("--validation-inputs", type=Path, nargs="*", default=[])
    train_parser.add_argument("--validation-input-manifest", type=Path, default=None)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--summary", type=Path, required=True)
    train_parser.add_argument("--stream", action="store_true")
    train_parser.add_argument("--cache-dir", type=Path, default=None)
    train_parser.add_argument("--init-checkpoint", type=Path, default=None)
    train_parser.set_defaults(func=command_train)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--inputs", type=Path, nargs="+", default=None)
    evaluate.add_argument("--input-manifest", type=Path, default=None)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--replicate-offset", type=int, default=0)
    evaluate.add_argument("--max-examples", type=int, default=None)
    evaluate.add_argument("--predictions-output", type=Path, default=None)
    evaluate.set_defaults(func=command_evaluate)

    aggregate = subparsers.add_parser("aggregate-evaluations")
    aggregate.add_argument("--inputs", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.set_defaults(func=command_aggregate_evaluations)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
