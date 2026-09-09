"""Snakemake workers for feature preparation, training, and evaluation.

Raw and prepared packs are trusted pickle files. Preparation keeps graph arrays
unpadded; normalization statistics use training packs only. Training pads each
batch to fixed dimensions and delegates model updates to ``dreems.train``.
"""

# ruff: noqa: F722, F821, UP037

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from beartype import beartype
from jaxtyping import Float, Shaped, jaxtyped

from dreems import checkpoints, corpus, counterfactual, features, metrics, simulation, train
from dreems import model as dreems_model
from dreems.validation import _integer_array
from dreems.workflow_io import (
    json_ready,
    load_config,
    optional_input_paths,
    read_pickle,
    required_input_paths,
    write_csv,
    write_json,
    write_npz,
    write_pickle,
)


@beartype
def _report(stage: str, **fields: Any) -> None:
    """Flush a compact progress record to the Snakemake worker log."""
    print(json.dumps(json_ready({"stage": stage, **fields}), sort_keys=True), flush=True)


@beartype
def nested_mapping(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Read a config block; absent or null blocks use the dataclass defaults."""
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping for config key {key!r}")
    return dict(value)


@beartype
def dreems_config(config: Mapping[str, Any]) -> train.DREEMSTrainConfig:
    """Read architecture, features, loss, and optimizer settings from config."""
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


@beartype
def load_examples(path: Path) -> list[dict[str, Any]]:
    """Read a raw example or ordered raw pack, returning a list in either case."""
    payload = read_pickle(path)
    if isinstance(payload, dict):
        return [payload]
    if not isinstance(payload, list):
        raise ValueError(f"expected {path} to contain a raw example dict or list of dicts")
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"expected all examples in {path} to be dictionaries")
    return payload


@beartype
def _read_prepared_shard(path: Path) -> list[features.PreparedExample]:
    """Read the list of prepared examples stored in a pack."""
    items = read_pickle(path)
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ValueError(f"expected prepared shard {path} to contain a list of dictionaries")
    return items


@beartype
def _count(value: Any, *, name: str, minimum: int = 0) -> int:
    """Check a scalar count before conversion, including values read from JSON."""
    array = _integer_array(np.asarray(value), name=name, dtype=np.int64)
    if array.ndim != 0 or array.item() < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return array.item()


@beartype
def load_prepared_shards(
    paths: Sequence[Path], max_examples: int | None = None
) -> list[features.PreparedExample]:
    """Concatenate packs in input order, optionally taking a validation prefix."""
    if max_examples is not None:
        max_examples = _count(max_examples, name="max_examples", minimum=1)
    out: list[features.PreparedExample] = []
    for path in paths:
        shard = _read_prepared_shard(path)
        out.extend(shard)
        if max_examples is not None and len(out) >= max_examples:
            return out[:max_examples]
    return out


@beartype
def prepared_dimensions(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Find padding sizes and require consistent feature widths within a pack."""
    if not items:
        raise ValueError("no prepared examples")
    sizes = []
    widths = None
    for item in items:
        node_x, edge_x, edges = (np.asarray(item[key]) for key in ("node_x", "edge_x", "edges"))
        if (
            node_x.ndim != 2
            or edge_x.ndim != 2
            or edges.ndim != 2
            or edges.shape != (edge_x.shape[0], 2)
            or min(*node_x.shape, *edge_x.shape) <= 0
        ):
            raise ValueError("prepared features must be nonempty matrices aligned with edges")
        local_widths = (node_x.shape[1], edge_x.shape[1])
        if widths is not None and local_widths != widths:
            raise ValueError("inconsistent feature widths within prepared shard")
        widths = local_widths
        sizes.append((node_x.shape[0], edge_x.shape[0]))
    return {
        "examples": len(items),
        "max_nodes": max(nodes for nodes, _ in sizes),
        "max_edges": max(edges for _, edges in sizes),
        "node_dim": widths[0],
        "edge_dim": widths[1],
    }


@beartype
def prepared_dimensions_from_metadata(
    path: Path,
) -> dict[str, int] | None:
    """Read either historical sidecar name without opening a potentially large pack.

    Missing dimension metadata falls back to scanning the pack. Present but
    invalid counts fail instead of silently truncating them.
    """
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
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected metadata mapping in {metadata_path}")
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
    return {key: _count(dimensions[key], name=key, minimum=1) for key in required}


@beartype
def scan_prepared_dimensions(
    paths: Sequence[Path],
) -> dict[str, int]:
    """Merge pack dimensions, preferring sidecars to a full data scan."""
    merged: dict[str, int] | None = None
    for path in paths:
        dims = prepared_dimensions_from_metadata(path)
        if dims is None:
            dims = prepared_dimensions(_read_prepared_shard(path))
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


@beartype
def pad_with_dimensions(
    items: list[features.PreparedExample], dims: Mapping[str, int]
) -> dreems_model.DREEMSTrainArrays:
    """Pad a batch to the sizes obtained by scanning the training packs."""
    return features.pad_examples(
        items,
        max_nodes=dims["max_nodes"],
        max_edges=dims["max_edges"],
    )


@jaxtyped(typechecker=beartype)
def _normalization_moments(
    accumulator: Mapping[str, Any],
    prefix: str,
) -> tuple[Float[np.ndarray, "*shape"], Float[np.ndarray, "*shape"], int]:
    """Read one field's moments with matching shape and a positive count."""
    total = np.asarray(accumulator[f"{prefix}_sum"])
    squares = np.asarray(accumulator[f"{prefix}_sumsq"])
    expected_ndim = 1 if prefix in {"node", "edge"} else 0
    for values in (total, squares):
        if (
            values.dtype.kind not in "iuf"
            or values.ndim != expected_ndim
            or values.size == 0
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(f"invalid {prefix} normalization moments")
    if total.shape != squares.shape or np.any(squares < 0):
        raise ValueError(f"invalid {prefix} sum-of-squares shape or values")
    count = _count(accumulator[f"{prefix}_count"], name=f"{prefix}_count", minimum=1)
    return total.astype(np.float64), squares.astype(np.float64), count


@beartype
def normalization_accumulator_from_prepared(paths: Sequence[Path]) -> dict[str, object]:
    """Accumulate sums and squared sums over all training nodes and edges.

    The target moments use the stored, uncentered targets. Per-graph centering
    happens later in apply_normalization; this preserves the released convention.
    """
    totals: dict[str, Any] = {}
    for path in paths:
        items = _read_prepared_shard(path)
        if items:
            prepared_dimensions(items)
        for item in items:
            n, e = len(item["node_x"]), len(item["edge_x"])
            for prefix, field in (("node", "node_x"), ("edge", "edge_x"), ("y", "y"), ("q", "q")):
                values = np.asarray(item[field])
                if values.dtype.kind not in "biuf" or not np.all(np.isfinite(values)):
                    raise ValueError(f"{path}: {field} must contain finite real values")
                if field in {"y", "q"} and values.shape != ((e,) if field == "y" else (n,)):
                    raise ValueError(f"{path}: {field} must match the graph size")
                values = values.astype(np.float64, copy=False)
                for suffix, local in (
                    ("sum", values.sum(axis=0)),
                    ("sumsq", np.square(values).sum(axis=0)),
                ):
                    key = f"{prefix}_{suffix}"
                    if key not in totals:
                        totals[key] = np.zeros_like(local)
                    if totals[key].shape != local.shape:
                        raise ValueError(f"inconsistent {prefix} feature dimension")
                    totals[key] += local
                key = f"{prefix}_count"
                totals[key] = totals.get(key, 0) + values.shape[0]
    if not totals:
        raise ValueError("no prepared training examples")
    for prefix in ("node", "edge", "y", "q"):
        _normalization_moments(totals, prefix)
    return {
        "stage": "normalization_shard_stats",
        "inputs": [str(path) for path in paths],
        **json_ready(totals),
    }


@beartype
def merge_normalization_accumulators(
    accumulators: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Add pack statistics in input order without rereading training arrays."""
    if not accumulators:
        raise ValueError("no normalization shard stats")
    totals: dict[str, Any] = {}
    inputs = []
    for accumulator in accumulators:
        for prefix in ("node", "edge", "y", "q"):
            total, squares, count = _normalization_moments(accumulator, prefix)
            for suffix, local in (("sum", total), ("sumsq", squares)):
                key = f"{prefix}_{suffix}"
                if key not in totals:
                    totals[key] = np.zeros_like(local)
                if totals[key].shape != local.shape:
                    raise ValueError(f"inconsistent {prefix} feature dimension")
                totals[key] += local
            key = f"{prefix}_count"
            totals[key] = totals.get(key, 0) + count
        inputs.extend(str(path) for path in accumulator.get("inputs", []))
    return {"stage": "normalization_merged_stats", "inputs": inputs, **json_ready(totals)}


@beartype
def finalize_normalization_stats(
    accumulator: Mapping[str, Any],
    *,
    std_floor: float,
) -> dict[str, object]:
    """Convert moments to saved means/scales using the existing variance floors."""
    if not np.isfinite(std_floor) or std_floor < 0:
        raise ValueError("std_floor must be finite and nonnegative")
    floor = max(std_floor, 1e-12)
    stats: dict[str, Any] = {"normalization_std_floor": floor}
    for prefix in ("node", "edge", "y", "q"):
        total, squares, count = _normalization_moments(accumulator, prefix)
        mean = total / count
        variance = np.maximum(squares / count - mean * mean, 0.0)
        if prefix in {"node", "edge"}:
            stats[f"{prefix}_mean"] = mean.tolist()
            stats[f"{prefix}_std"] = np.maximum(np.sqrt(variance), floor).tolist()
            stats[f"{prefix}_count"] = count
        else:
            stats["y_scale" if prefix == "y" else "q_y_scale"] = (np.sqrt(variance) + 1e-6).item()
    return stats


@beartype
def normalized_shard(path: Path, stats: Mapping[str, Any]) -> list[features.PreparedExample]:
    """Apply saved training statistics to one ordered prepared pack."""
    return features.apply_normalization(_read_prepared_shard(path), stats)


@beartype
def iter_training_batches(
    paths: Sequence[Path],
    stats: Mapping[str, Any],
    *,
    dims: Mapping[str, int],
    batch_size: int,
    steps: int,
    seed: int,
) -> Iterable[dreems_model.DREEMSTrainArrays]:
    """Shuffle packs and examples deterministically, keeping sampling views together.

    Pending groups and ordinary examples carry across pack/epoch boundaries.
    Groups have one anchor and one common size that must divide the batch size.
    """
    batch_size = _count(batch_size, name="batch_size", minimum=1)
    steps = _count(steps, name="steps")
    if steps == 0:
        return
    if not paths:
        raise ValueError("no training shards")
    rng = np.random.default_rng(seed)
    produced = 0
    grouped_pending: list[list[features.PreparedExample]] = []
    singleton_pending: list[features.PreparedExample] = []
    group_size = None

    def units(
        shard: list[features.PreparedExample],
    ) -> tuple[list[list[features.PreparedExample]], list[features.PreparedExample]]:
        """Collect paired views before shuffling so anchors stay in the batch."""
        nonlocal group_size
        grouped: dict[int, list[features.PreparedExample]] = {}
        singletons: list[features.PreparedExample] = []
        for item in shard:
            group = _count(
                item.get("counterfactual_group", -1), name="counterfactual_group", minimum=-1
            )
            if group < 0:
                singletons.append(item)
            else:
                grouped.setdefault(group, []).append(item)
        groups = list(grouped.values())
        for group_items in groups:
            if group_size is not None and len(group_items) != group_size:
                raise ValueError("counterfactual training groups must all have the same size")
            group_size = len(group_items)
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
        has_examples = False
        for path_index in rng.permutation(len(paths)):
            shard = normalized_shard(paths[int(path_index)], stats)
            if not shard:
                continue
            has_examples = True
            groups, singletons = units(shard)
            grouped_pending.extend(groups[int(index)] for index in rng.permutation(len(groups)))
            singleton_pending.extend(
                singletons[int(index)] for index in rng.permutation(len(singletons))
            )
            while grouped_pending:
                groups_per_batch = batch_size // len(grouped_pending[0])
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
        if not has_examples:
            raise ValueError("all training shards are empty")


@beartype
def stage_input_paths(paths: Sequence[Path], cache_dir: Path, *, label: str) -> list[Path]:
    """Copy each distinct source once to local disk, preserving repeated references."""
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
    _report(
        "stage_training_inputs",
        event="start",
        label=label,
        files=len(unique_sources),
        references=len(paths),
        bytes=total_bytes,
    )
    staged_by_source: dict[Path, Path] = {}
    for index, source in enumerate(unique_sources):
        destination = destination_dir / f"{index:05d}_{source.name}"
        shutil.copyfile(source, destination)
        staged_by_source[source] = destination
        completed = index + 1
        if completed == 1 or completed == len(unique_sources) or completed % 16 == 0:
            _report(
                "stage_training_inputs",
                event="progress",
                label=label,
                completed_files=completed,
                files=len(unique_sources),
                elapsed_seconds=round(time.time() - started, 3),
            )
    _report(
        "stage_training_inputs",
        event="done",
        label=label,
        files=len(unique_sources),
        references=len(paths),
        bytes=total_bytes,
        elapsed_seconds=round(time.time() - started, 3),
    )
    return [staged_by_source[path.resolve()] for path in paths]


@jaxtyped(typechecker=beartype)
def edge_coverage(
    truth: Shaped[np.ndarray, " edges"],  # noqa: F722
    mean: Shaped[np.ndarray, " edges"],  # noqa: F722
    sd: Shaped[np.ndarray, " edges"],  # noqa: F722
    z: float,
) -> float:
    """Fraction inside a pointwise interval, excluding nonfinite or zero-SD edges."""
    finite = np.isfinite(truth) & np.isfinite(mean) & np.isfinite(sd) & (sd > 0.0)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(np.abs(truth[finite] - mean[finite]) <= z * sd[finite]))


@beartype
def prepare_raw_examples(
    raw_examples: list[dict[str, Any]],
    run_config: train.DREEMSTrainConfig,
    *,
    source: Path,
    index_offset: int = 0,
) -> tuple[list[features.PreparedExample], list[dict[str, Any]], list[dict[str, Any]]]:
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


@beartype
def raw_example_window(
    paths: Sequence[Path],
    *,
    start_index: int,
    max_examples: int | None,
) -> list[tuple[Path, int, list[dict[str, Any]]]]:
    """Read one contiguous example window from one or more ordered raw packs."""

    start_index = _count(start_index, name="start-index")
    if max_examples is not None:
        max_examples = _count(max_examples, name="max-examples", minimum=1)

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


@beartype
def command_prepare_shard(args: argparse.Namespace) -> None:
    """Prepare a raw pack (or selected example) and its dimension/audit sidecar."""
    run_config = dreems_config(load_config(args.config))
    raw_examples = load_examples(args.input)
    index_offset = 0
    if args.example_index is not None:
        index_offset = _count(args.example_index, name="example-index")
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
    _report(
        "prepare_shard",
        output=str(args.output),
        raw_examples=len(raw_examples),
        examples=len(prepared),
        skipped_unconditionable_examples=len(skipped),
    )


@beartype
def command_prepare_batch(args: argparse.Namespace) -> None:
    """Prepare several independent raw shards into one ordered training pack."""

    run_config = dreems_config(load_config(args.config))
    prepared: list[features.PreparedExample] = []
    skipped: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    raw_example_count = 0
    windows = raw_example_window(
        args.inputs,
        start_index=args.start_index,
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
    _report(
        "prepare_batch",
        output=str(args.output),
        source_shards=len(windows),
        raw_examples=raw_example_count,
        examples=len(prepared),
        skipped_unconditionable_examples=len(skipped),
    )


@beartype
def command_prepare_counterfactual(args: argparse.Namespace) -> None:
    """Prepare four paired masks from one reusable raw covariance example."""

    run_config = dreems_config(load_config(args.config))
    selected = counterfactual.read_selection(args.selection)
    slot = _count(args.slot, name="slot")
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
    _report(
        "prepare_counterfactual",
        slot=slot,
        source=str(source),
        examples=len(prepared),
        output=str(args.output),
    )


@beartype
def command_pack_prepared(args: argparse.Namespace) -> None:
    """Repack an ordered window of prepared files without recomputing features."""
    inputs = required_input_paths(
        args.inputs,
        args.input_manifest,
        argument_name="--inputs",
    )
    start = _count(args.start_index, name="start-index")
    if args.max_inputs is None:
        stop = len(inputs)
    else:
        stop = min(len(inputs), start + _count(args.max_inputs, name="max-inputs", minimum=1))
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
    _report(
        "pack_prepared",
        output=str(args.output),
        source_shards=len(selected),
        examples=len(prepared),
    )


@beartype
def command_normalization_shard_stats(args: argparse.Namespace) -> None:
    """Write additive training moments for one group of prepared packs."""
    stats = normalization_accumulator_from_prepared(args.inputs)
    write_json(args.output, stats)
    _report(
        "normalization_shard_stats",
        output=str(args.output),
        node_count=stats["node_count"],
        edge_count=stats["edge_count"],
    )


@beartype
def command_merge_normalization(args: argparse.Namespace) -> None:
    """Merge training-only moments and write the normalization used by the model."""
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
    _report(
        "merge_normalization",
        output=str(args.output),
        node_count=stats["node_count"],
        edge_count=stats["edge_count"],
        y_scale=stats["y_scale"],
        q_y_scale=stats["q_y_scale"],
    )


@beartype
def _command_train(args: argparse.Namespace, cache_dir: Path | None) -> None:
    """Train from prepared packs, with explicit initialization and validation data.

    A resumed model keeps its normalization. Gate-only training may also retain
    its SD calibration. Core and feature metadata must match before training.
    """
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
        checkpoint_features = checkpoints.canonical_feature_metadata(checkpoint_features)
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
    validation_arrays = None
    validation_limit = int(cfg.get("train_validation_examples", 0) or 0)
    if validation_inputs and validation_limit > 0:
        validation_prepared = load_prepared_shards(validation_inputs, max_examples=validation_limit)
        validation_prepared = features.apply_normalization(validation_prepared, stats)
        validation_arrays = features.pad_examples(validation_prepared)
    history_path = args.summary.with_name("training_history.jsonl")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.unlink(missing_ok=True)

    def training_progress_callback(
        step: int,
        params: dreems_model.DREEMS,
        losses: list[float],
        row: Mapping[str, Any],
    ) -> None:
        """Log each completed step and save improving validation checkpoints."""
        with history_path.open("a") as handle:
            handle.write(json.dumps(json_ready(row), sort_keys=True) + "\n")
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

    if args.stream:
        # Streaming keeps memory bounded by loading one prepared shard at a time.
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
            checkpoint_callback=training_progress_callback,
        )
    else:
        # The eager path is useful for smoke tests and small local runs.
        prepared = load_prepared_shards(inputs)
        prepared = features.apply_normalization(prepared, stats)
        train_arrays = features.pad_examples(prepared)
        train_result = train.train_from_arrays(
            train_arrays,
            config=run_config,
            init_params=init_params,
            validation_arrays=validation_arrays,
            checkpoint_callback=training_progress_callback,
        )

    train.save_checkpoint(
        args.output,
        params=train_result.params,
        normalization=stats,
        config=run_config,
        train_losses=train_result.losses,
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
    _report("train", output=str(args.output))


@beartype
def command_train(args: argparse.Namespace) -> None:
    """Train, cleaning up only the new local staging directory owned by this run."""
    if args.cache_dir is None:
        _command_train(args, None)
        return

    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=False)
    try:
        _command_train(args, cache_dir)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


@beartype
def command_evaluate(args: argparse.Namespace) -> None:
    """Evaluate in graph-size order while retaining each input example's identity.

    Replicate IDs follow input order across packs.
    Means and SDs are exported in log-rate units, with stored SD calibration.
    """
    batch_size = _count(args.batch_size, name="batch_size", minimum=1)
    replicate = _count(args.replicate_offset, name="replicate_offset")
    max_examples = (
        None
        if args.max_examples is None
        else _count(args.max_examples, name="max_examples", minimum=1)
    )
    started = time.time()
    inputs = required_input_paths(
        args.inputs,
        args.input_manifest,
        argument_name="--inputs",
    )
    _report(
        "evaluate",
        event="start",
        checkpoint=str(args.checkpoint),
        inputs=[str(path) for path in inputs],
        batch_size=int(args.batch_size),
        max_examples=args.max_examples,
    )
    loaded = checkpoints.load_checkpoint(args.checkpoint)
    _report("evaluate", event="checkpoint_loaded", elapsed_seconds=round(time.time() - started, 3))
    stats = loaded.normalization
    rows: list[dict[str, Any]] = []
    prediction_replicates: list[int] = []
    edge_offsets = [0]
    edge_truth_parts: list[Float[np.ndarray, "_edges"]] = []
    edge_mean_parts: list[Float[np.ndarray, "_edges"]] = []
    edge_sd_parts: list[Float[np.ndarray, "_edges"]] = []
    q_offsets = [0]
    q_truth_parts: list[Float[np.ndarray, "_nodes"]] = []
    q_mean_parts: list[Float[np.ndarray, "_nodes"]] = []
    q_sd_parts: list[Float[np.ndarray, "_nodes"]] = []
    for shard_index, path in enumerate(inputs):
        if max_examples is not None and len(rows) >= max_examples:
            break
        shard_started = time.time()
        _report("evaluate", event="shard_start", shard_index=shard_index, input=str(path))
        prepared = normalized_shard(path, stats)
        indexed = list(enumerate(prepared))
        if max_examples is not None:
            indexed = indexed[: max(0, max_examples - len(rows))]
        indexed = sorted(indexed, key=lambda pair: pair[1]["node_x"].shape[0])
        _report(
            "evaluate",
            event="shard_loaded",
            shard_index=shard_index,
            examples=len(indexed),
            elapsed_seconds=round(time.time() - shard_started, 3),
        )
        total_batches = math.ceil(len(indexed) / batch_size)
        for batch_index, start in enumerate(range(0, len(indexed), batch_size), start=1):
            chunk = indexed[start : start + batch_size]
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
                _report(
                    "evaluate",
                    event="batch_done",
                    shard_index=shard_index,
                    batch=batch_index,
                    batches=total_batches,
                    rows_so_far=len(rows),
                    elapsed_seconds=round(time.time() - shard_started, 3),
                )
        _report(
            "evaluate",
            event="shard_done",
            shard_index=shard_index,
            rows_so_far=len(rows),
            elapsed_seconds=round(time.time() - shard_started, 3),
        )
        replicate += len(prepared)
    if not rows:
        raise ValueError("no evaluation rows")
    rows.sort(key=lambda row: int(row["replicate"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "summary.csv"
    write_csv(csv_path, json_ready(rows))
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
        write_npz(
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
    _report(
        "evaluate",
        event="done",
        examples=len(rows),
        max_examples=max_examples,
        output=str(csv_path),
        elapsed_seconds=round(time.time() - started, 3),
    )


@beartype
def summarize_numeric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Summarize each numeric field over its finite rows using population SD."""
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


@beartype
def command_aggregate_evaluations(args: argparse.Namespace) -> None:
    """Join evaluation reports and assign global IDs in the declared input order."""
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
    write_csv(args.output_dir / "summary.csv", json_ready(rows))
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
    _report("aggregate_evaluations", examples=len(rows))


@beartype
def build_parser() -> argparse.ArgumentParser:
    """Expose separate workers so Snakemake can schedule and reuse each stage."""
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


@beartype
def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch one pipeline worker; Snakemake owns scheduling and retries."""
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
