"""Build reproducible visual reports for DREEMS benchmark outputs."""

# ruff: noqa: F722,F821,UP037

from __future__ import annotations

import argparse
import csv
import html
import itertools
import json
import os
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from beartype import beartype
from jaxtyping import Float, Int, Real, Shaped, jaxtyped
from scipy import stats

from dreems.utils.validation import _integer_array, _real_array, validate_graph_arrays
from dreems.utils.workflow_io import (
    _atomic_output,
    json_ready,
    parse_nonnegative_integer,
    read_npz,
    read_pickle,
    write_csv,
    write_json,
)


class ScenarioArrays(TypedDict):
    """Geometry, counts, and edge estimates for one matched simulation."""

    positions: Float[np.ndarray, "nodes 2"]
    edges: Int[np.ndarray, "edges 2"]
    truth: Float[np.ndarray, "edges"]
    sample_count: Float[np.ndarray, "nodes"]
    edge_sd: Float[np.ndarray, "edges"]
    conditioner: Float[np.ndarray, "edges"]
    dreems: Float[np.ndarray, "edges"]
    dreems_map: Float[np.ndarray, "edges"]
    feems: Float[np.ndarray, "edges"]


SNP_BINS = ((50, 128), (129, 512), (513, 2000), (2001, 10000))
SELECTION_SEED = 20260827


@beartype
def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read ordered text columns; metric and ID fields are parsed separately."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@beartype
def numeric(row: Mapping[str, Any], key: str) -> float:
    """Return NaN for absent or unavailable metrics."""
    value = row.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


@beartype
def integer(row: Mapping[str, Any], key: str) -> int:
    """Require an exact nonnegative ID or count, including in CSV text."""
    return parse_nonnegative_integer(row[key], name=key)


@beartype
def _rows_by_id(rows: Sequence[Mapping[str, Any]], key: str) -> dict[int, dict[str, Any]]:
    """Index rows without silently replacing duplicates."""
    indexed = {}
    for row in rows:
        index = integer(row, key)
        if index in indexed:
            raise ValueError(f"duplicate {key} {index}")
        indexed[index] = dict(row)
    return indexed


@beartype
def finite_mean(values: Iterable[float]) -> float:
    """Average available metrics and return NaN for an empty group."""
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


@beartype
def snp_bin_label(num_snps: int) -> str:
    """Label one of the four fixed SNP-count evaluation strata."""
    for low, high in SNP_BINS:
        if low <= num_snps <= high:
            return f"{low:,}-{high:,}"
    raise ValueError(f"SNP count {num_snps} is outside the configured bins")


@beartype
def paired_stats(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, Any]:
    """Give each case equal weight, pairing MSE and correlation independently.

    Intervals are t intervals over row differences. They assume independent
    cases; several sampling views of one simulation are not independent trials.
    """
    old_corr = np.asarray([numeric(row, "old_edge_log_corr") for row in rows])
    new_corr = np.asarray([numeric(row, "new_edge_log_corr") for row in rows])
    old_mse = np.asarray([numeric(row, "old_edge_log_mse") for row in rows])
    new_mse = np.asarray([numeric(row, "new_edge_log_mse") for row in rows])
    corr_valid = np.isfinite(old_corr) & np.isfinite(new_corr)
    mse_valid = np.isfinite(old_mse) & np.isfinite(new_mse)
    old_corr = old_corr[corr_valid]
    new_corr = new_corr[corr_valid]
    old_mse = old_mse[mse_valid]
    new_mse = new_mse[mse_valid]
    corr_delta = new_corr - old_corr
    mse_advantage = old_mse - new_mse

    @jaxtyped(typechecker=beartype)
    def delta_summary(values: Float[np.ndarray, "cases"]) -> tuple[float, float, float, float]:
        """Summarize a paired difference using the existing two-sided t test."""
        if not values.size:
            return (float("nan"),) * 4
        mean = float(values.mean())
        if values.size < 2:
            return mean, float("nan"), float("nan"), float("nan")
        se = float(values.std(ddof=1) / np.sqrt(values.size))
        critical = float(stats.t.ppf(0.975, values.size - 1))
        p_value = float(stats.ttest_1samp(values, 0.0).pvalue)
        return mean, mean - critical * se, mean + critical * se, p_value

    corr_mean, corr_low, corr_high, corr_p = delta_summary(corr_delta)
    mse_mean, mse_low, mse_high, mse_p = delta_summary(mse_advantage)
    return {
        "label": label,
        "n": len(rows),
        "n_corr": int(corr_valid.sum()),
        "n_mse": int(mse_valid.sum()),
        "old_corr": finite_mean(old_corr),
        "new_corr": finite_mean(new_corr),
        "corr_delta": corr_mean,
        "corr_delta_ci95_low": corr_low,
        "corr_delta_ci95_high": corr_high,
        "corr_delta_p": corr_p,
        "corr_win_fraction": float(np.mean(corr_delta > 0.0)) if corr_delta.size else float("nan"),
        "old_mse": finite_mean(old_mse),
        "new_mse": finite_mean(new_mse),
        "mse_advantage": mse_mean,
        "mse_advantage_ci95_low": mse_low,
        "mse_advantage_ci95_high": mse_high,
        "mse_advantage_p": mse_p,
        "mse_win_fraction": float(np.mean(mse_advantage > 0.0))
        if mse_advantage.size
        else float("nan"),
    }


@beartype
def h2h_stats(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, Any]:
    """Apply the same paired statistics with FEEMS as the reference method."""
    proxy = [
        {
            "old_edge_log_corr": numeric(row, "feems_edge_log_corr"),
            "new_edge_log_corr": numeric(row, "dreems_map_edge_log_corr"),
            "old_edge_log_mse": numeric(row, "feems_edge_log_mse"),
            "new_edge_log_mse": numeric(row, "dreems_map_edge_log_mse"),
        }
        for row in rows
    ]
    result = paired_stats(proxy, label=label)
    names = {
        "old_corr": "feems_corr",
        "new_corr": "dreems_corr",
        "old_mse": "feems_mse",
        "new_mse": "dreems_mse",
    }
    return {names.get(key, key): value for key, value in result.items()}


@beartype
def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Replace a CSV atomically, preserving column order and NumPy scalar values."""
    write_csv(path, json_ready(list(rows)))


@beartype
def load_metadata_rows(paths: Sequence[Path]) -> dict[int, dict[str, Any]]:
    """Collect metadata packs, requiring globally unique example indices."""
    rows = []
    for path in paths:
        payload = json.loads(path.read_text())
        rows.extend(payload["rows"])
    return _rows_by_id(rows, "index")


@beartype
def merge_checkpoint_rows(
    old_summary: Path,
    new_summary: Path,
    metadata_paths: Sequence[Path],
) -> list[dict[str, Any]]:
    """Match old and new predictions to the same independent raw examples."""
    old_payload = json.loads(old_summary.read_text())
    new_payload = json.loads(new_summary.read_text())
    old = _rows_by_id(old_payload["rows"], "replicate_global")
    new = _rows_by_id(new_payload["rows"], "replicate_global")
    metadata = load_metadata_rows(metadata_paths)
    if old.keys() != new.keys() or old.keys() != metadata.keys():
        raise ValueError("checkpoint rows and test metadata do not contain identical replicates")

    rows = []
    metric_names = (
        "edge_log_corr",
        "edge_log_mse",
        "edge_log_rmse",
        "edge_log_slope",
        "edge_log_scale_ratio",
        "edge_sd_mean",
        "edge_50_coverage",
        "edge_80_coverage",
        "edge_95_coverage",
        "q_log_corr",
        "q_log_mse",
        "q_80_coverage",
    )
    for replicate in sorted(old):
        meta = dict(metadata[replicate])
        row: dict[str, Any] = {
            "replicate": replicate,
            "snp_bin": snp_bin_label(integer(meta, "num_snps")),
            **meta,
        }
        for name in metric_names:
            row[f"old_{name}"] = old[replicate].get(name)
            row[f"new_{name}"] = new[replicate].get(name)
        row["corr_delta"] = numeric(row, "new_edge_log_corr") - numeric(row, "old_edge_log_corr")
        row["mse_advantage"] = numeric(row, "old_edge_log_mse") - numeric(row, "new_edge_log_mse")
        rows.append(row)
    return rows


@beartype
def deterministic_sample(
    rows: Sequence[Mapping[str, Any]],
    count: int,
    seed: int,
    *,
    key: str = "replicate",
) -> list[int]:
    """Choose case IDs without replacement, using a fixed gallery seed."""
    if isinstance(count, bool) or count < 0:
        raise ValueError("count must be nonnegative")
    ids = np.asarray([integer(row, key) for row in rows], dtype=np.int64)
    if np.unique(ids).size != ids.size:
        raise ValueError(f"duplicate {key} in gallery selection")
    if ids.size <= count:
        return ids.tolist()
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(ids, size=count, replace=False).tolist())


@beartype
def checkpoint_selections(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    """Choose fixed-seed examples and rank gains and losses within each SNP bin."""
    sections: dict[str, list[int]] = {}
    for bin_index, (low, high) in enumerate(SNP_BINS):
        subset = [row for row in rows if low <= integer(row, "num_snps") <= high]
        scored = [
            row
            for row in subset
            if np.isfinite(numeric(row, "corr_delta"))
            and np.isfinite(numeric(row, "mse_advantage"))
        ]
        prefix = f"{low:,}-{high:,} SNPs"
        sections[f"{prefix}: random"] = deterministic_sample(scored, 4, SELECTION_SEED + bin_index)
        sections[f"{prefix}: largest gains"] = [
            integer(row, "replicate")
            for row in sorted(
                scored,
                key=lambda row: numeric(row, "corr_delta") + numeric(row, "mse_advantage"),
                reverse=True,
            )[:4]
        ]
        sections[f"{prefix}: largest losses"] = [
            integer(row, "replicate")
            for row in sorted(
                scored,
                key=lambda row: numeric(row, "corr_delta") + numeric(row, "mse_advantage"),
            )[:4]
        ]
    return sections


@beartype
def command_compare_checkpoints(args: argparse.Namespace) -> None:
    """Write matched checkpoint rows, paired statistics, and gallery selections."""
    rows = merge_checkpoint_rows(args.old_summary, args.new_summary, args.metadata)
    grouped = [paired_stats(rows, label="all test examples")]
    for low, high in SNP_BINS:
        subset = [row for row in rows if low <= integer(row, "num_snps") <= high]
        grouped.append(paired_stats(subset, label=f"{low:,}-{high:,} SNPs"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "rows.csv", rows)
    write_rows(args.output_dir / "aggregate_stats.csv", grouped)
    selections = checkpoint_selections(rows)
    write_json(args.output_dir / "selected_examples.json", selections)
    write_json(
        args.output_dir / "summary.json",
        {
            "comparison": "broad SNP-count checkpoint versus released checkpoint",
            "examples": len(rows),
            "old_checkpoint": str(args.old_checkpoint),
            "new_checkpoint": str(args.new_checkpoint),
            "snp_bins": [list(bounds) for bounds in SNP_BINS],
            "aggregate_stats": grouped,
            "old_evaluation": str(args.old_summary),
            "new_evaluation": str(args.new_summary),
        },
    )


@beartype
def _plot_imports() -> tuple[Any, Any, Any, Any, Any]:
    """Load the noninteractive plotting backend only when rendering is requested."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.collections import LineCollection
    from matplotlib.colors import TwoSlopeNorm

    return plt, mtri, LineCollection, TwoSlopeNorm, matplotlib


@jaxtyped(typechecker=beartype)
def centered(values: Shaped[np.ndarray, "*input_axes"]) -> Float[np.ndarray, "edges"]:
    """Center a finite edge field without changing its relative amplitude."""
    array = _real_array(values, name="edge values")
    if array.ndim != 1 or not array.size:
        raise ValueError("edge values must be a nonempty vector")
    return array - float(np.mean(array))


@jaxtyped(typechecker=beartype)
def edge_to_node(
    values: Real[np.ndarray, "edges"], edges: Int[np.ndarray, "edges 2"], num_nodes: int
) -> Float[np.ndarray, "nodes"]:
    """Average incident edges for display only; inference remains edge-valued."""
    if num_nodes < 1 or np.any(edges < 0) or np.any(edges >= num_nodes):
        raise ValueError("edge endpoints must index the supplied nodes")
    totals = np.zeros(num_nodes, dtype=np.float64)
    counts = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(totals, edges[:, 0], values)
    np.add.at(totals, edges[:, 1], values)
    np.add.at(counts, edges[:, 0], 1.0)
    np.add.at(counts, edges[:, 1], 1.0)
    return np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0.0)


@jaxtyped(typechecker=beartype)
def plot_surface(
    ax: Any,
    positions: Real[np.ndarray, "nodes 2"],
    edges: Int[np.ndarray, "edges 2"],
    values: Real[np.ndarray, "edges"],
    sample_count: Real[np.ndarray, "nodes"],
    *,
    limit: float,
    title: str,
) -> None:
    """Interpolate for display, falling back to raw edges on degenerate meshes."""
    validate_graph_arrays(positions=positions, edges=edges)
    sample_count = _real_array(sample_count, name="sample counts")
    if np.any(sample_count < 0):
        raise ValueError("sample counts must be nonnegative")
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("color limit must be finite and positive")
    _plt, mtri, LineCollection, TwoSlopeNorm, _matplotlib = _plot_imports()
    positions = np.asarray(positions, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int32)
    values = centered(values)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    node_values = edge_to_node(values, edges, positions.shape[0])
    rendered = False
    if positions.shape[0] >= 3:
        try:
            triangulation = mtri.Triangulation(positions[:, 0], positions[:, 1])
            triangles = triangulation.triangles
            graph_lengths = np.linalg.norm(positions[edges[:, 1]] - positions[edges[:, 0]], axis=1)
            threshold = 1.8 * float(np.quantile(graph_lengths, 0.9))
            triangle_positions = positions[triangles]
            side_lengths = np.stack(
                [
                    np.linalg.norm(triangle_positions[:, 1] - triangle_positions[:, 0], axis=1),
                    np.linalg.norm(triangle_positions[:, 2] - triangle_positions[:, 1], axis=1),
                    np.linalg.norm(triangle_positions[:, 0] - triangle_positions[:, 2], axis=1),
                ],
                axis=1,
            )
            triangulation.set_mask(np.max(side_lengths, axis=1) > threshold)
            has_triangles = triangulation.mask is None or np.any(~triangulation.mask)
            if has_triangles:
                ax.tricontourf(
                    triangulation,
                    node_values,
                    levels=np.linspace(-limit, limit, 31),
                    cmap="RdBu",
                    norm=norm,
                    extend="both",
                )
                ax.tricontour(
                    triangulation,
                    node_values,
                    levels=np.linspace(-limit, limit, 9),
                    colors="#222222",
                    linewidths=0.28,
                    alpha=0.35,
                )
                rendered = True
        except (RuntimeError, ValueError):
            rendered = False
    segments = positions[edges]
    if not rendered:
        collection = LineCollection(segments, cmap="RdBu", norm=norm, linewidths=1.2)
        collection.set_array(values)
        ax.add_collection(collection)
    else:
        ax.add_collection(LineCollection(segments, colors="#202020", linewidths=0.18, alpha=0.16))
    sampled = np.flatnonzero(sample_count > 0)
    if sampled.size:
        relative = sample_count[sampled] / max(float(sample_count[sampled].max()), 1.0)
        ax.scatter(
            positions[sampled, 0],
            positions[sampled, 1],
            s=7.0 + 22.0 * np.sqrt(relative),
            facecolor="white",
            edgecolor="#151515",
            linewidth=0.35,
            zorder=5,
        )
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale()
    ax.set_xticks([])
    ax.set_yticks([])


@jaxtyped(typechecker=beartype)
def prediction_slice(
    path: Path, replicate: int
) -> tuple[Float[np.ndarray, "edges"], Float[np.ndarray, "edges"], Float[np.ndarray, "edges"]]:
    """Read one case from a packed edge archive, validating its ragged offsets."""
    if isinstance(replicate, bool) or replicate < 0:
        raise ValueError("replicate must be a nonnegative integer")
    with np.load(path, allow_pickle=False) as archive:
        replicates = _integer_array(archive["replicate"], name="replicate", dtype=np.int64)
        if (
            replicates.ndim != 1
            or np.any(replicates < 0)
            or np.unique(replicates).size != replicates.size
        ):
            raise ValueError("replicate IDs must be a unique nonnegative vector")
        matches = np.flatnonzero(replicates == replicate)
        if matches.size != 1:
            raise ValueError(f"expected one prediction for replicate {replicate} in {path}")
        index = int(matches[0])
        offsets = _integer_array(archive["edge_offsets"], name="edge_offsets", dtype=np.int64)
        values = tuple(
            _real_array(archive[key], name=key) for key in ("edge_truth", "edge_mean", "edge_sd")
        )
        if (
            offsets.shape != (replicates.size + 1,)
            or offsets[0] != 0
            or np.any(np.diff(offsets) <= 0)
            or any(value.shape != (offsets[-1],) for value in values)
        ):
            raise ValueError("edge_offsets must partition every packed edge vector exactly")
        if np.any(values[2] < 0):
            raise ValueError("edge_sd must be nonnegative")
        start, stop = int(offsets[index]), int(offsets[index + 1])
        return tuple(value[start:stop] for value in values)


@beartype
def raw_example(pattern: str, pack_size: int, replicate: int) -> Mapping[str, Any]:
    """Locate a global case ID in an ordered raw pack."""
    if isinstance(pack_size, bool) or pack_size < 1 or isinstance(replicate, bool) or replicate < 0:
        raise ValueError("pack size must be positive and replicate must be nonnegative")
    pack = replicate // pack_size
    local = replicate % pack_size
    examples = read_pickle(Path(pattern.format(pack=pack)))
    return examples[local]


@jaxtyped(typechecker=beartype)
def example_geometry(
    example: Mapping[str, Any],
) -> tuple[Float[np.ndarray, "nodes 2"], Int[np.ndarray, "edges 2"], Float[np.ndarray, "nodes"]]:
    """Extract the graph and sample counts without changing edge order."""
    positions = _real_array(example["active_positions"], name="positions")
    edges = _integer_array(example["graph"]["edges"], name="edges", dtype=np.int32)
    sample_nodes = _integer_array(example["sample_nodes"], name="sample_nodes", dtype=np.int32)
    validate_graph_arrays(positions=positions, edges=edges)
    if sample_nodes.ndim != 1 or np.any(sample_nodes < 0) or np.any(sample_nodes >= len(positions)):
        raise ValueError("sample_nodes must index graph nodes")
    sample_count = np.bincount(sample_nodes, minlength=positions.shape[0]).astype(np.float64)
    return positions, edges, sample_count


@beartype
def save_figure(fig: Any, path: Path) -> None:
    """Write the display image and vector PDF atomically, one file at a time."""
    with _atomic_output(path) as stream:
        fig.savefig(
            stream, format=path.suffix.lstrip("."), dpi=190, bbox_inches="tight", facecolor="white"
        )
    with _atomic_output(path.with_suffix(".pdf")) as stream:
        fig.savefig(stream, format="pdf", bbox_inches="tight", facecolor="white")


@jaxtyped(typechecker=beartype)
def _comparison_scatter(
    left_corr: Real[np.ndarray, "cases"],
    right_corr: Real[np.ndarray, "cases"],
    left_mse: Real[np.ndarray, "cases"],
    right_mse: Real[np.ndarray, "cases"],
    colors: Real[np.ndarray, "cases"],
    *,
    labels: tuple[str, str],
    color_label: str,
    output: Path,
    figsize: tuple[float, float],
    cmap: str,
    size: float,
    alpha: float,
) -> None:
    """Draw matched correlation/MSE scatters with the same identity-line rules."""
    plt, *_ = _plot_imports()
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    scatter = axes[0].scatter(left_corr, right_corr, c=colors, s=size, cmap=cmap, alpha=alpha)
    correlations = np.concatenate((left_corr, right_corr))
    finite_corr = correlations[np.isfinite(correlations)]
    low = float(finite_corr.min()) if finite_corr.size else -1.0
    axes[0].plot([low, 1.0], [low, 1.0], color="#222222", linewidth=1)
    axes[0].set(xlabel=f"{labels[0]} correlation", ylabel=f"{labels[1]} correlation")
    axes[0].set_aspect("equal", adjustable="box")
    axes[1].scatter(left_mse, right_mse, c=colors, s=size, cmap=cmap, alpha=alpha)
    errors = np.concatenate((left_mse, right_mse))
    finite_mse = errors[np.isfinite(errors)]
    if np.any(finite_mse < 0):
        plt.close(fig)
        raise ValueError("MSE must be nonnegative")
    lower, upper = (
        (float(finite_mse.min()), float(finite_mse.max())) if finite_mse.size else (0.0, 1.0)
    )
    if upper == 0.0:
        upper = 1.0
    axes[1].plot([lower, upper], [lower, upper], color="#222222", linewidth=1)
    scale = "log" if lower > 0.0 else "linear"
    axes[1].set(xscale=scale, yscale=scale)
    axes[1].set(xlabel=f"{labels[0]} MSE", ylabel=f"{labels[1]} MSE")
    axes[1].set_aspect("equal", adjustable="box")
    colorbar = fig.colorbar(scatter, ax=axes, fraction=0.035, pad=0.03)
    colorbar.set_label(color_label)
    save_figure(fig, output)
    plt.close(fig)


@beartype
def checkpoint_aggregate_figures(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    """Compare checkpoints overall and by the four SNP-count strata."""
    plt, _mtri, _LineCollection, _TwoSlopeNorm, _matplotlib = _plot_imports()
    old_corr = np.asarray([numeric(row, "old_edge_log_corr") for row in rows])
    new_corr = np.asarray([numeric(row, "new_edge_log_corr") for row in rows])
    old_mse = np.asarray([numeric(row, "old_edge_log_mse") for row in rows])
    new_mse = np.asarray([numeric(row, "new_edge_log_mse") for row in rows])
    snps = np.asarray([numeric(row, "num_snps") for row in rows])
    _comparison_scatter(
        old_corr,
        new_corr,
        old_mse,
        new_mse,
        np.log10(snps),
        labels=("Previous checkpoint", "Broad-SNP"),
        color_label="log10(SNP count)",
        output=output_dir / "checkpoint_scatter.png",
        figsize=(10.5, 4.4),
        cmap="viridis",
        size=9.0,
        alpha=0.62,
    )

    grouped = []
    for low_snp, high_snp in SNP_BINS:
        subset = [row for row in rows if low_snp <= integer(row, "num_snps") <= high_snp]
        grouped.append(paired_stats(subset, label=f"{low_snp:,}-{high_snp:,}"))
    labels = [row["label"] for row in grouped]
    x = np.arange(len(grouped))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for ax, key, low_key, high_key, ylabel in (
        (
            axes[0],
            "corr_delta",
            "corr_delta_ci95_low",
            "corr_delta_ci95_high",
            "Broad-SNP minus previous correlation",
        ),
        (
            axes[1],
            "mse_advantage",
            "mse_advantage_ci95_low",
            "mse_advantage_ci95_high",
            "Previous minus broad-SNP MSE",
        ),
    ):
        means = np.asarray([row[key] for row in grouped])
        errors = np.vstack(
            [
                means - np.asarray([row[low_key] for row in grouped]),
                np.asarray([row[high_key] for row in grouped]) - means,
            ]
        )
        ax.bar(x, means, color=["#c43c39" if value < 0 else "#2474a6" for value in means])
        ax.errorbar(x, means, yerr=errors, fmt="none", ecolor="#151515", capsize=3)
        ax.axhline(0.0, color="#222222", linewidth=0.8)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
    save_figure(fig, output_dir / "checkpoint_snp_bin_deltas.png")
    plt.close(fig)


@beartype
def _checkpoint_examples(
    rows_by_id: Mapping[int, Mapping[str, Any]],
    replicate_ids: Sequence[int],
    *,
    raw_pattern: str,
    prediction_pattern: str,
    pack_size: int,
    output: Path,
    router: bool,
) -> None:
    """Draw matched predictions once, optionally repeating the router's selection."""
    if not replicate_ids:
        raise ValueError("select at least one example to plot")
    plt, _mtri, _LineCollection, _TwoSlopeNorm, _matplotlib = _plot_imports()
    fig, axes = plt.subplots(
        len(replicate_ids),
        4 if router else 3,
        figsize=(16.5 if router else 12.5, max(3.0 * len(replicate_ids), 3.2)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, replicate in enumerate(replicate_ids):
        row = rows_by_id[replicate]
        example = raw_example(raw_pattern, pack_size, replicate)
        positions, edges, sample_count = example_geometry(example)
        pack = replicate // pack_size
        old_truth, old_mean, _old_sd = prediction_slice(
            Path(prediction_pattern.format(model="old", pack=pack)), replicate
        )
        new_truth, new_mean, _new_sd = prediction_slice(
            Path(prediction_pattern.format(model="new", pack=pack)), replicate
        )
        np.testing.assert_allclose(old_truth, new_truth, atol=2e-5)
        truth = centered(old_truth)
        np.testing.assert_allclose(truth, centered(example["edge_log_rates"]), atol=2e-5)
        limit = max(float(np.quantile(np.abs(truth), 0.98)), 0.15)
        subtitle = (
            f"rep {replicate} | {integer(row, 'num_snps'):,} SNPs | "
            f"{integer(row, 'num_samples'):,} samples | "
            f"{100.0 * numeric(row, 'sampled_node_fraction'):.1f}% nodes"
        )
        old_label, new_label = (
            ("Released\n", "Broad-SNP\n") if router else ("Previous | ", "Broad SNP | ")
        )
        panels = [
            (truth, f"Truth\n{subtitle}"),
            (
                old_mean,
                f"{old_label}r={numeric(row, 'old_edge_log_corr'):.3f}, MSE={numeric(row, 'old_edge_log_mse'):.3f}",
            ),
            (
                new_mean,
                f"{new_label}r={numeric(row, 'new_edge_log_corr'):.3f}, MSE={numeric(row, 'new_edge_log_mse'):.3f}",
            ),
        ]
        if router:
            selected = row["router_checkpoint"]
            if selected not in ("broad", "released"):
                raise ValueError(f"unknown router checkpoint {selected!r}")
            selected_label = "Broad-SNP" if selected == "broad" else "Released"
            panels.append(
                (
                    new_mean if selected == "broad" else old_mean,
                    f"Router: {selected_label}\nr={numeric(row, 'router_edge_log_corr'):.3f}, MSE={numeric(row, 'router_edge_log_mse'):.3f}",
                )
            )
        for column, (values, title) in enumerate(panels):
            plot_surface(
                axes[row_index, column],
                positions,
                edges,
                values,
                sample_count,
                limit=limit,
                title=title,
            )
        if router:
            axes[row_index, 3].spines[:].set_color("#2474a6")
            axes[row_index, 3].spines[:].set_linewidth(1.4)
    save_figure(fig, output)
    plt.close(fig)


@beartype
def checkpoint_example_figure(
    rows_by_id: Mapping[int, Mapping[str, Any]],
    replicate_ids: Sequence[int],
    *,
    raw_pattern: str,
    prediction_pattern: str,
    pack_size: int,
    output: Path,
) -> None:
    """Compare both checkpoints against the same truth and raw graph."""
    _checkpoint_examples(
        rows_by_id,
        replicate_ids,
        raw_pattern=raw_pattern,
        prediction_pattern=prediction_pattern,
        pack_size=pack_size,
        output=output,
        router=False,
    )


@beartype
def html_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    """Escape report cells and keep wide tables scrollable."""
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.4g}"
            values.append(f"<td>{html.escape(str(value))}</td>")
        body.append("<tr>" + "".join(values) + "</tr>")
    return f"<div class='table-wrap'><table><tr>{head}</tr>{''.join(body)}</table></div>"


@beartype
def gallery_html(title: str, intro: str, body: str) -> str:
    """Wrap trusted report fragments in the common gallery layout."""
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #171717; }}
main {{ max-width: 1500px; margin: auto; }}
h1, h2, h3 {{ letter-spacing: 0; }}
p {{ max-width: 1050px; line-height: 1.45; color: #3d3d3d; }}
img {{ display: block; max-width: 100%; border: 1px solid #d6d6d6; margin: 12px 0 30px; }}
.table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; font-size: 13px; margin: 16px 0 28px; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 6px 9px; text-align: right; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
code {{ background: #f3f3f3; padding: 1px 4px; }}
</style></head><body><main>
<h1>{html.escape(title)}</h1><p>{intro}</p>{body}
</main></body></html>"""


@beartype
def command_checkpoint_gallery(args: argparse.Namespace) -> None:
    """Render saved checkpoint comparisons without running either model."""
    rows = read_csv_rows(args.rows)
    rows_by_id = _rows_by_id(rows, "replicate")
    selections = json.loads(args.selections.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_aggregate_figures(rows, args.output_dir)
    section_html = []
    labels = [
        f"{low:,}-{high:,} SNPs: {section}"
        for low, high in SNP_BINS
        for section in ("random", "largest gains", "largest losses")
    ]
    for section_index, label in enumerate(labels):
        replicate_ids = selections[label]
        if not replicate_ids:
            continue
        filename = f"examples_{section_index:02d}.png"
        checkpoint_example_figure(
            rows_by_id,
            [parse_nonnegative_integer(value, name="replicate") for value in replicate_ids],
            raw_pattern=args.raw_pattern,
            prediction_pattern=args.prediction_pattern,
            pack_size=args.pack_size,
            output=args.output_dir / filename,
        )
        section_html.append(f"<h3>{html.escape(label)}</h3><img src='{html.escape(filename)}'>")
    summary = json.loads(args.summary.read_text())
    stats_rows = summary["aggregate_stats"]
    table = html_table(
        stats_rows,
        (
            "label",
            "n",
            "n_corr",
            "n_mse",
            "old_corr",
            "new_corr",
            "corr_delta",
            "corr_win_fraction",
            "old_mse",
            "new_mse",
            "mse_advantage",
            "mse_win_fraction",
        ),
    )
    body = (
        "<h2>Aggregate comparison</h2>"
        "<img src='checkpoint_scatter.png'><img src='checkpoint_snp_bin_deltas.png'>"
        "<p><a href='../summary.json'>summary.json</a> | "
        "<a href='../rows.csv'>rows.csv</a> | "
        "<a href='../aggregate_stats.csv'>aggregate_stats.csv</a></p>"
        + table
        + "<h2>Example surfaces</h2>"
        + "".join(section_html)
    )
    intro = (
        "Both checkpoints were evaluated on the same independent covariance simulations. "
        "The previous checkpoint was trained on 2,000-10,000 SNPs; the broad-SNP checkpoint "
        "was trained with equal representation in the four displayed SNP ranges. Positive "
        "correlation and MSE deltas favor the broad-SNP checkpoint. Example colors show "
        "centered edge-log relative migration: red is lower, white is the graph mean, and "
        "blue is higher. Each row uses the truth scale for all three panels."
    )
    with _atomic_output(args.output_dir / "index.html") as stream:
        stream.write(gallery_html("SNP-count checkpoint comparison", intro, body).encode("utf-8"))


ROUTER_METRICS = (
    "edge_log_corr",
    "edge_log_mse",
    "edge_log_rmse",
    "edge_log_slope",
    "edge_log_scale_ratio",
    "edge_sd_mean",
    "edge_50_coverage",
    "edge_80_coverage",
    "edge_95_coverage",
)


@beartype
def route_checkpoint(row: Mapping[str, Any], threshold: int) -> str:
    """Choose a checkpoint from the SNP count alone, never from truth or error."""
    if isinstance(threshold, bool) or threshold < 0:
        raise ValueError("router threshold must be nonnegative")
    return "broad" if integer(row, "num_snps") <= threshold else "released"


@beartype
def build_router_rows(rows: Sequence[Mapping[str, Any]], threshold: int) -> list[dict[str, Any]]:
    """Copy the selected checkpoint's metrics and retain the unselected comparison."""
    _rows_by_id(rows, "replicate")
    routed = []
    for source in rows:
        row = dict(source)
        selected = route_checkpoint(row, threshold)
        prefix = "new" if selected == "broad" else "old"
        alternative_prefix = "old" if prefix == "new" else "new"
        row["router_checkpoint"] = selected
        for metric in ROUTER_METRICS:
            row[f"router_{metric}"] = numeric(row, f"{prefix}_{metric}")
        router_mse = numeric(row, "router_edge_log_mse")
        alternative_mse = numeric(row, f"{alternative_prefix}_edge_log_mse")
        router_corr = numeric(row, "router_edge_log_corr")
        alternative_corr = numeric(row, f"{alternative_prefix}_edge_log_corr")
        row["router_mse_advantage_over_alternative"] = alternative_mse - router_mse
        row["router_corr_advantage_over_alternative"] = router_corr - alternative_corr
        row["router_mse_regret"] = router_mse - min(
            numeric(row, "old_edge_log_mse"), numeric(row, "new_edge_log_mse")
        )
        row["router_corr_regret"] = (
            max(numeric(row, "old_edge_log_corr"), numeric(row, "new_edge_log_corr")) - router_corr
        )
        routed.append(row)
    return routed


@beartype
def router_stats(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, Any]:
    """Summarize the existing router decision without optimizing its threshold."""
    result: dict[str, Any] = {"label": label, "n": len(rows)}
    for model in ("old", "new", "router"):
        corr = np.asarray([numeric(row, f"{model}_edge_log_corr") for row in rows])
        mse = np.asarray([numeric(row, f"{model}_edge_log_mse") for row in rows])
        result[f"{model}_corr"] = finite_mean(corr)
        result[f"{model}_mse"] = finite_mean(mse)
        if model == "router":
            result["n_corr"] = int(np.isfinite(corr).sum())
            result["n_mse"] = int(np.isfinite(mse).sum())
    mse_advantage = np.asarray(
        [numeric(row, "router_mse_advantage_over_alternative") for row in rows]
    )
    corr_advantage = np.asarray(
        [numeric(row, "router_corr_advantage_over_alternative") for row in rows]
    )
    mse_advantage = mse_advantage[np.isfinite(mse_advantage)]
    corr_advantage = corr_advantage[np.isfinite(corr_advantage)]
    result["router_mse_win_fraction_vs_alternative"] = (
        float(np.mean(mse_advantage > 0.0)) if mse_advantage.size else float("nan")
    )
    result["router_corr_win_fraction_vs_alternative"] = (
        float(np.mean(corr_advantage > 0.0)) if corr_advantage.size else float("nan")
    )
    return result


@beartype
def router_selections(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    """Select representative and extreme router cases using fixed rules."""
    selections: dict[str, list[int]] = {}
    for bin_index, (low, high) in enumerate(SNP_BINS):
        subset = [row for row in rows if low <= integer(row, "num_snps") <= high]
        subset = [
            row
            for row in subset
            if np.isfinite(numeric(row, "router_edge_log_corr"))
            and np.isfinite(numeric(row, "router_edge_log_mse"))
        ]
        selections[f"{low:,}-{high:,} SNPs: random"] = deterministic_sample(
            subset, 4, SELECTION_SEED + 100 + bin_index
        )
    finite_corr = [
        row
        for row in rows
        if np.isfinite(numeric(row, "router_edge_log_corr"))
        and np.isfinite(numeric(row, "router_edge_log_mse"))
    ]
    finite_advantage = [
        row for row in rows if np.isfinite(numeric(row, "router_mse_advantage_over_alternative"))
    ]
    selections["Router helps most"] = [
        integer(row, "replicate")
        for row in sorted(
            finite_advantage,
            key=lambda row: numeric(row, "router_mse_advantage_over_alternative"),
            reverse=True,
        )[:4]
    ]
    selections["Router hurts most"] = [
        integer(row, "replicate")
        for row in sorted(
            finite_advantage,
            key=lambda row: numeric(row, "router_mse_advantage_over_alternative"),
        )[:4]
    ]
    selections["Best router recoveries"] = [
        integer(row, "replicate")
        for row in sorted(
            finite_corr,
            key=lambda row: (
                numeric(row, "router_edge_log_corr"),
                -numeric(row, "router_edge_log_mse"),
            ),
            reverse=True,
        )[:4]
    ]
    selections["Worst router recoveries"] = [
        integer(row, "replicate")
        for row in sorted(
            finite_corr,
            key=lambda row: (
                numeric(row, "router_edge_log_corr"),
                -numeric(row, "router_edge_log_mse"),
            ),
        )[:4]
    ]
    return selections


@beartype
def command_router_report(args: argparse.Namespace) -> None:
    """Record routing decisions and their performance on saved predictions."""
    source_rows = read_csv_rows(args.rows)
    rows = build_router_rows(source_rows, args.threshold)
    grouped = [router_stats(rows, label="all test examples")]
    for low, high in SNP_BINS:
        subset = [row for row in rows if low <= integer(row, "num_snps") <= high]
        grouped.append(router_stats(subset, label=f"{low:,}-{high:,} SNPs"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "rows.csv", rows)
    write_rows(args.output_dir / "aggregate_stats.csv", grouped)
    selections = router_selections(rows)
    write_json(args.output_dir / "selected_examples.json", selections)
    write_json(
        args.output_dir / "summary.json",
        {
            "comparison": "SNP-count checkpoint router",
            "examples": len(rows),
            "threshold": args.threshold,
            "low_snp_checkpoint": "broad",
            "high_snp_checkpoint": "released",
            "aggregate_stats": grouped,
            "source_rows": str(args.rows),
        },
    )


@beartype
def router_aggregate_figures(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    """Show SNP-bin performance and compare routing with the unselected checkpoint."""
    plt, _mtri, _LineCollection, _TwoSlopeNorm, _matplotlib = _plot_imports()
    grouped = []
    for low, high in SNP_BINS:
        subset = [row for row in rows if low <= integer(row, "num_snps") <= high]
        grouped.append(router_stats(subset, label=f"{low:,}-{high:,}"))
    labels = [row["label"] for row in grouped]
    x = np.arange(len(grouped))
    width = 0.25
    colors = {"old": "#555555", "new": "#d07a27", "router": "#2474a6"}
    names = {"old": "Released", "new": "Broad-SNP", "router": "Router"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
    for model_index, model in enumerate(("old", "new", "router")):
        offset = (model_index - 1) * width
        axes[0].bar(
            x + offset,
            [row[f"{model}_corr"] for row in grouped],
            width,
            color=colors[model],
            label=names[model],
        )
        axes[1].bar(
            x + offset,
            [row[f"{model}_mse"] for row in grouped],
            width,
            color=colors[model],
            label=names[model],
        )
    for ax, ylabel in zip(axes, ("Mean edge correlation", "Mean edge MSE"), strict=True):
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
    save_figure(fig, output_dir / "router_snp_bin_metrics.png")
    plt.close(fig)

    snps = np.asarray([numeric(row, "num_snps") for row in rows])
    router_corr = np.asarray([numeric(row, "router_edge_log_corr") for row in rows])
    router_mse = np.asarray([numeric(row, "router_edge_log_mse") for row in rows])
    alternative_corr = np.asarray(
        [
            numeric(
                row,
                "old_edge_log_corr" if row["router_checkpoint"] == "broad" else "new_edge_log_corr",
            )
            for row in rows
        ]
    )
    alternative_mse = np.asarray(
        [
            numeric(
                row,
                "old_edge_log_mse" if row["router_checkpoint"] == "broad" else "new_edge_log_mse",
            )
            for row in rows
        ]
    )
    _comparison_scatter(
        alternative_corr,
        router_corr,
        alternative_mse,
        router_mse,
        np.log10(snps),
        labels=("Unselected checkpoint", "Router"),
        color_label="log10(SNP count)",
        output=output_dir / "router_vs_alternative.png",
        figsize=(10.8, 4.4),
        cmap="viridis",
        size=9.0,
        alpha=0.62,
    )


@beartype
def router_example_figure(
    rows_by_id: Mapping[int, Mapping[str, Any]],
    replicate_ids: Sequence[int],
    *,
    raw_pattern: str,
    prediction_pattern: str,
    pack_size: int,
    output: Path,
) -> None:
    """Add the selected checkpoint to the shared truth/old/new comparison."""
    _checkpoint_examples(
        rows_by_id,
        replicate_ids,
        raw_pattern=raw_pattern,
        prediction_pattern=prediction_pattern,
        pack_size=pack_size,
        output=output,
        router=True,
    )


@beartype
def command_router_gallery(args: argparse.Namespace) -> None:
    """Render the saved router decisions with both constituent checkpoints."""
    rows = read_csv_rows(args.rows)
    rows_by_id = _rows_by_id(rows, "replicate")
    selections = json.loads(args.selections.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    router_aggregate_figures(rows, args.output_dir)
    section_html = []
    labels = [
        *(f"{low:,}-{high:,} SNPs: random" for low, high in SNP_BINS),
        "Router helps most",
        "Router hurts most",
        "Best router recoveries",
        "Worst router recoveries",
    ]
    for section_index, label in enumerate(labels):
        if not selections[label]:
            continue
        filename = f"examples_{section_index:02d}.png"
        router_example_figure(
            rows_by_id,
            [parse_nonnegative_integer(value, name="replicate") for value in selections[label]],
            raw_pattern=args.raw_pattern,
            prediction_pattern=args.prediction_pattern,
            pack_size=args.pack_size,
            output=args.output_dir / filename,
        )
        section_html.append(f"<h3>{html.escape(label)}</h3><img src='{filename}'>")
    summary = json.loads(args.summary.read_text())
    table = html_table(
        summary["aggregate_stats"],
        (
            "label",
            "n",
            "n_corr",
            "old_corr",
            "new_corr",
            "router_corr",
            "old_mse",
            "new_mse",
            "router_mse",
            "router_corr_win_fraction_vs_alternative",
            "router_mse_win_fraction_vs_alternative",
        ),
    )
    body = (
        "<h2>Aggregate comparison</h2>"
        "<img src='router_snp_bin_metrics.png'><img src='router_vs_alternative.png'>"
        "<p><a href='../summary.json'>summary.json</a> | "
        "<a href='../rows.csv'>rows.csv</a> | "
        "<a href='../aggregate_stats.csv'>aggregate_stats.csv</a></p>"
        + table
        + "<h2>Example surfaces</h2>"
        + "".join(section_html)
    )
    intro = (
        f"The router uses the broad-SNP checkpoint at or below {args.threshold:,} effective "
        "SNPs and the released checkpoint above that threshold. It performs no additional "
        "fitting. Colors show centered edge-log relative migration: red is lower, white is "
        "the graph mean, and blue is higher. Each row uses the truth scale for all panels. "
        "The router panel repeats its selected checkpoint and is outlined in blue."
    )
    with _atomic_output(args.output_dir / "index.html") as stream:
        stream.write(gallery_html("SNP-aware checkpoint router", intro, body).encode("utf-8"))


@beartype
def command_router_h2h_gallery(args: argparse.Namespace) -> None:
    """Reuse a released-checkpoint gallery only when every case selects that model."""
    rows = read_csv_rows(args.reference_rows)
    _rows_by_id(rows, "index")
    if not rows:
        raise ValueError("reference gallery must contain cases")
    source = args.reference_gallery.resolve()
    destination = args.output_dir.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("source and destination galleries must not overlap")
    page = (source / "index.html").read_text()
    marker = "</h1>"
    if marker not in page:
        raise ValueError("reference gallery does not contain the expected heading")
    selections = [
        {
            "index": integer(row, "index"),
            "num_snps": integer(row, "num_snps"),
            "router_checkpoint": route_checkpoint(row, args.threshold),
        }
        for row in rows
    ]
    broad_count = sum(row["router_checkpoint"] == "broad" for row in selections)
    released_count = len(selections) - broad_count
    if broad_count:
        raise ValueError(
            "the reference gallery contains Released-checkpoint surfaces, but the router "
            f"would select Broad-SNP for {broad_count} examples; rerun those DREEMS cases"
        )

    source_provenance_path = args.reference_gallery / "provenance.json"
    source_provenance = (
        json.loads(source_provenance_path.read_text()) if source_provenance_path.is_file() else None
    )
    shutil.copytree(
        args.reference_gallery,
        args.output_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".snakemake_timestamp"),
    )
    write_rows(args.output_dir / "router_selections.csv", selections)
    write_json(
        args.output_dir / "provenance.json",
        {
            "comparison": "DREEMS SNP-count router versus FEEMS recommended LOO",
            "router_threshold": args.threshold,
            "router_selection_counts": {
                "broad": broad_count,
                "released": released_count,
            },
            "source_gallery": str(args.reference_gallery.resolve()),
            "source_rows": str(args.reference_rows.resolve()),
            "source_provenance": source_provenance,
        },
    )

    index_path = args.output_dir / "index.html"
    page = page.replace(
        "DREEMS + fixed-radius MAP vs FEEMS recommended LOO",
        "DREEMS router + fixed-radius MAP vs FEEMS recommended LOO",
    ).replace("DREEMS + MAP", "DREEMS router + MAP")
    router_note = (
        "<p class='note'><strong>Router audit:</strong> The router uses Broad-SNP at or "
        f"below {args.threshold:,} SNPs and Released above it. All {released_count:,} "
        f"examples in this FEEMS LOO benchmark exceed the {args.threshold:,}-SNP threshold, so the router "
        "selected Released for every case. The surfaces and numerical results are therefore "
        "identical to the Released-checkpoint H2H. "
        "<a href='router_selections.csv'>router selections</a></p>"
    )
    with _atomic_output(index_path) as stream:
        stream.write(page.replace(marker, marker + router_note, 1).encode("utf-8"))


@beartype
def h2h_selections(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    """Select random, extreme, sparse, and dense cases from paired results."""
    random_ids = deterministic_sample(rows, 8, SELECTION_SEED, key="index")
    corr_sorted = sorted(
        [
            row
            for row in rows
            if np.isfinite(numeric(row, "dreems_map_edge_log_corr"))
            and np.isfinite(numeric(row, "feems_edge_log_corr"))
        ],
        key=lambda row: (
            numeric(row, "dreems_map_edge_log_corr") - numeric(row, "feems_edge_log_corr")
        ),
    )
    mse_sorted = sorted(
        [
            row
            for row in rows
            if np.isfinite(numeric(row, "dreems_map_edge_log_mse"))
            and np.isfinite(numeric(row, "feems_edge_log_mse"))
        ],
        key=lambda row: (
            numeric(row, "feems_edge_log_mse") - numeric(row, "dreems_map_edge_log_mse")
        ),
    )
    sparse = [row for row in rows if numeric(row, "sampled_node_fraction") <= 0.10]
    dense = [row for row in rows if numeric(row, "sampled_node_fraction") > 0.25]
    return {
        "Random examples": random_ids,
        "Largest DREEMS correlation gains": [integer(row, "index") for row in corr_sorted[-8:]],
        "Largest DREEMS correlation losses": [integer(row, "index") for row in corr_sorted[:8]],
        "Largest DREEMS MSE gains": [integer(row, "index") for row in mse_sorted[-8:]],
        "Largest DREEMS MSE losses": [integer(row, "index") for row in mse_sorted[:8]],
        "Sparse sampling: random": deterministic_sample(sparse, 8, SELECTION_SEED + 1, key="index"),
        "Dense sampling: random": deterministic_sample(dense, 8, SELECTION_SEED + 2, key="index"),
    }


@beartype
def h2h_example_figure(
    rows_by_id: Mapping[int, Mapping[str, Any]],
    replicate_ids: Sequence[int],
    *,
    dreems_pattern: str,
    feems_pattern: str,
    output: Path,
) -> None:
    """Draw truth, DREEMS+MAP, and FEEMS on an explicitly matched graph."""
    plt, _mtri, _LineCollection, _TwoSlopeNorm, _matplotlib = _plot_imports()
    fig, axes = plt.subplots(
        len(replicate_ids),
        3,
        figsize=(12.5, max(3.0 * len(replicate_ids), 3.2)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, replicate in enumerate(replicate_ids):
        row = rows_by_id[replicate]
        arrays = _matched_predictions(
            replicate, dreems_pattern=dreems_pattern, feems_pattern=feems_pattern
        )
        positions, edges = arrays["positions"], arrays["edges"]
        truth = centered(arrays["truth"])
        dreems_map, feems_mean = arrays["dreems_map"], arrays["feems"]
        sample_count = arrays["sample_count"]
        limit = max(float(np.quantile(np.abs(truth), 0.98)), 0.15)
        subtitle = (
            f"rep {replicate} | {integer(row, 'num_snps'):,} SNPs | "
            f"{integer(row, 'num_samples'):,} samples | "
            f"{100.0 * numeric(row, 'sampled_node_fraction'):.1f}% nodes"
        )
        plot_surface(
            axes[row_index, 0],
            positions,
            edges,
            truth,
            sample_count,
            limit=limit,
            title=f"Truth\n{subtitle}",
        )
        plot_surface(
            axes[row_index, 1],
            positions,
            edges,
            dreems_map,
            sample_count,
            limit=limit,
            title=(
                f"DREEMS+MAP | r={numeric(row, 'dreems_map_edge_log_corr'):.3f}, "
                f"MSE={numeric(row, 'dreems_map_edge_log_mse'):.3f}"
            ),
        )
        plot_surface(
            axes[row_index, 2],
            positions,
            edges,
            feems_mean,
            sample_count,
            limit=limit,
            title=(
                f"FEEMS LOO | r={numeric(row, 'feems_edge_log_corr'):.3f}, "
                f"MSE={numeric(row, 'feems_edge_log_mse'):.3f}"
            ),
        )
    save_figure(fig, output)
    plt.close(fig)


@beartype
def h2h_aggregate_figures(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    """Compare FEEMS and DREEMS+MAP, coloring points by sampling density."""
    feems_corr = np.asarray([numeric(row, "feems_edge_log_corr") for row in rows])
    dreems_corr = np.asarray([numeric(row, "dreems_map_edge_log_corr") for row in rows])
    feems_mse = np.asarray([numeric(row, "feems_edge_log_mse") for row in rows])
    dreems_mse = np.asarray([numeric(row, "dreems_map_edge_log_mse") for row in rows])
    density = np.asarray([numeric(row, "sampled_node_fraction") for row in rows])
    _comparison_scatter(
        feems_corr,
        dreems_corr,
        feems_mse,
        dreems_mse,
        density,
        labels=("FEEMS LOO", "DREEMS+MAP"),
        color_label="Fraction of graph nodes sampled",
        output=output_dir / "h2h_scatter_corr_mse.png",
        figsize=(10.5, 4.4),
        cmap="plasma",
        size=12.0,
        alpha=0.7,
    )


@beartype
def command_h2h_gallery(args: argparse.Namespace) -> None:
    """Render paired cached fits and descriptive sampling-density summaries."""
    rows = read_csv_rows(args.rows)
    rows_by_id = _rows_by_id(rows, "index")
    selections = h2h_selections(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    h2h_aggregate_figures(rows, args.output_dir)
    grouped = [h2h_stats(rows, label="all paired examples")]
    for label, predicate in (
        ("sparse: <=10% nodes", lambda value: value <= 0.10),
        ("medium: 10-25% nodes", lambda value: 0.10 < value <= 0.25),
        ("dense: >25% nodes", lambda value: value > 0.25),
    ):
        subset = [row for row in rows if predicate(numeric(row, "sampled_node_fraction"))]
        if subset:
            grouped.append(h2h_stats(subset, label=label))
    write_rows(args.output_dir / "aggregate_stats.csv", grouped)
    write_json(args.output_dir / "aggregate_stats.json", {row["label"]: row for row in grouped})
    write_json(args.output_dir / "selected_examples.json", selections)
    section_html = []
    for section_index, (label, ids) in enumerate(selections.items()):
        if not ids:
            continue
        filename = f"examples_{section_index:02d}.png"
        h2h_example_figure(
            rows_by_id,
            ids,
            dreems_pattern=args.dreems_pattern,
            feems_pattern=args.feems_pattern,
            output=args.output_dir / filename,
        )
        section_html.append(f"<h3>{html.escape(label)}</h3><img src='{filename}'>")
    table = html_table(
        grouped,
        (
            "label",
            "n",
            "n_corr",
            "n_mse",
            "dreems_corr",
            "feems_corr",
            "corr_delta",
            "corr_win_fraction",
            "dreems_mse",
            "feems_mse",
            "mse_advantage",
            "mse_win_fraction",
        ),
    )
    body = (
        "<h2>Aggregate comparison</h2><img src='h2h_scatter_corr_mse.png'>"
        "<p><a href='../summary.json'>summary.json</a> | "
        "<a href='../rows.csv'>rows.csv</a> | "
        "<a href='aggregate_stats.csv'>aggregate_stats.csv</a></p>"
        + table
        + "<h2>Example surfaces</h2>"
        + "".join(section_html)
    )
    intro = (
        "DREEMS uses the checkpoint and MAP settings recorded in the accompanying results. FEEMS uses "
        "the cached recommended leave-one-observed-deme-out fit. Colors show centered "
        "edge-log relative migration: red is lower, white is the graph mean, and blue is "
        "higher. Each row uses the truth scale for all three panels. White points are "
        "sampled graph nodes, sized by sample count."
    )
    with _atomic_output(args.output_dir / "index.html") as stream:
        stream.write(gallery_html("DREEMS versus FEEMS", intro, body).encode("utf-8"))


SCENARIO_TRUTHS = ("barrier", "anisotropy", "homogeneous")
SCENARIO_MASKS = ("full", "random_10pct", "outer_slabs", "left_missing")
SCENARIO_METHODS = (
    ("conditioner", "Linearized initializer", "conditioner_edge_log"),
    ("dreems", "Raw DREEMS", "dreems_edge_log"),
    ("dreems_map", "DREEMS + MAP", "dreems_map_edge_log"),
    ("feems", "FEEMS LOO", "feems_edge_log"),
)


@beartype
def _scenario_mask_label(mask: str) -> str:
    """Use short multiline labels for the fixed notebook sampling designs."""
    if mask == "random_10pct":
        return "random\nsparse"
    return mask.replace("_", "\n")


@jaxtyped(typechecker=beartype)
def _matched_predictions(
    example: int,
    *,
    dreems_pattern: str,
    feems_pattern: str,
) -> dict[str, Shaped[np.ndarray, "*_archive_axes"]]:
    """Check common graph metadata before putting cached methods side by side."""
    dreems = read_npz(Path(dreems_pattern.format(example=example)))
    feems = read_npz(Path(feems_pattern.format(example=example)))
    for archive in (dreems, feems):
        archive["edges"] = _integer_array(archive["edges"], name="edges", dtype=np.int32)
        for key in ("positions", "truth", "sample_count"):
            archive[key] = _real_array(archive[key], name=key)
    for key in ("positions", "edges", "truth", "sample_count"):
        if not np.array_equal(dreems[key], feems[key]):
            raise ValueError(f"scenario {example} disagrees between DREEMS and FEEMS on {key}")
    arrays = dict(dreems)
    arrays["feems"] = feems["feems"]
    validate_graph_arrays(positions=arrays["positions"], edges=arrays["edges"])
    for key in ("truth", "dreems", "dreems_map", "edge_sd", "conditioner", "feems"):
        if key in arrays:
            arrays[key] = _real_array(arrays[key], name=key)
            if arrays[key].shape != (len(arrays["edges"]),):
                raise ValueError(f"{key} must align with graph edges")
    if arrays["sample_count"].shape != (len(arrays["positions"]),) or np.any(
        arrays["sample_count"] < 0
    ):
        raise ValueError("sample counts must be nonnegative and align with graph nodes")
    if "edge_sd" in arrays and np.any(arrays["edge_sd"] < 0):
        raise ValueError("edge_sd must be nonnegative")
    return arrays


@beartype
def _scenario_arrays(
    example: int,
    *,
    dreems_pattern: str,
    feems_pattern: str,
) -> ScenarioArrays:
    """Select the full scenario diagnostics from a validated method pair."""
    arrays = _matched_predictions(
        example, dreems_pattern=dreems_pattern, feems_pattern=feems_pattern
    )
    return {
        key: arrays[key]
        for key in (
            "positions",
            "edges",
            "truth",
            "sample_count",
            "edge_sd",
            "conditioner",
            "dreems",
            "dreems_map",
            "feems",
        )
    }


@beartype
def _scenario_row_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str, str], int]:
    """Require one case per truth, mask, and replicate in the crossed design."""
    _rows_by_id(rows, "index")
    lookup: dict[tuple[int, str, str], int] = {}
    for row in rows:
        key = (
            integer(row, "scenario_replicate"),
            str(row["scenario_truth"]),
            str(row["scenario_mask"]),
        )
        if key in lookup:
            raise ValueError(f"duplicate crossed scenario row: {key}")
        lookup[key] = integer(row, "index")
    expected = {
        (replicate, truth, mask)
        for replicate in sorted({integer(row, "scenario_replicate") for row in rows})
        for truth in SCENARIO_TRUTHS
        for mask in SCENARIO_MASKS
    }
    if lookup.keys() != expected:
        missing = sorted(expected - lookup.keys())
        extra = sorted(lookup.keys() - expected)
        raise ValueError(f"incomplete crossed scenario design; missing={missing}, extra={extra}")
    return lookup


@beartype
def _scenario_metric_title(
    row: Mapping[str, Any],
    *,
    method_prefix: str,
    truth: str,
) -> str:
    """Replace undefined correlation with output RMS for homogeneous truth."""
    mse = numeric(row, f"{method_prefix}_mse")
    if truth == "homogeneous":
        rms = numeric(row, f"{method_prefix}_rms")
        return f"RMS={rms:.3f} | MSE={mse:.3f}"
    corr = numeric(row, f"{method_prefix}_corr")
    return f"r={corr:.3f} | MSE={mse:.3f}"


@jaxtyped(typechecker=beartype)
def _plot_scenario_uncertainty(
    ax: Any,
    positions: Real[np.ndarray, "nodes 2"],
    edges: Int[np.ndarray, "edges 2"],
    edge_sd: Real[np.ndarray, "edges"],
    sample_count: Real[np.ndarray, "nodes"],
    *,
    title: str,
) -> None:
    """Display posterior SD separately from migration amplitude."""
    validate_graph_arrays(positions=positions, edges=edges)
    if np.any(_real_array(edge_sd, name="edge_sd") < 0) or np.any(
        _real_array(sample_count, name="sample_count") < 0
    ):
        raise ValueError("SD and sample counts must be nonnegative")
    _plt, _mtri, LineCollection, _TwoSlopeNorm, _matplotlib = _plot_imports()
    from matplotlib.colors import Normalize

    finite = edge_sd[np.isfinite(edge_sd)]
    upper = max(float(np.quantile(finite, 0.98)) if finite.size else 0.0, 0.1)
    collection = LineCollection(
        positions[edges],
        cmap="magma",
        norm=Normalize(vmin=0.0, vmax=upper),
        linewidths=1.8,
    )
    collection.set_array(edge_sd)
    ax.add_collection(collection)
    sampled = np.flatnonzero(sample_count > 0)
    if sampled.size:
        relative = sample_count[sampled] / max(float(sample_count[sampled].max()), 1.0)
        ax.scatter(
            positions[sampled, 0],
            positions[sampled, 1],
            s=7.0 + 22.0 * np.sqrt(relative),
            facecolor="white",
            edgecolor="#151515",
            linewidth=0.35,
            zorder=5,
        )
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale()
    ax.set_xticks([])
    ax.set_yticks([])


@beartype
def scenario_replicate_figure(
    rows_by_id: Mapping[int, Mapping[str, Any]],
    lookup: Mapping[tuple[int, str, str], int],
    *,
    replicate: int,
    dreems_pattern: str,
    feems_pattern: str,
    output: Path,
) -> None:
    """Draw all methods and uncertainty for one complete truth/mask crossing."""
    plt, _mtri, _LineCollection, _TwoSlopeNorm, _matplotlib = _plot_imports()
    methods = (("truth", "Truth", None),) + SCENARIO_METHODS
    column_count = len(methods) + 1
    row_count = len(SCENARIO_TRUTHS) * len(SCENARIO_MASKS)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(19.5, 2.55 * row_count),
        squeeze=False,
        constrained_layout=True,
    )

    limits: dict[str, float] = {}
    for truth in SCENARIO_TRUTHS:
        values = []
        for mask in SCENARIO_MASKS:
            example = lookup[(replicate, truth, mask)]
            arrays = _scenario_arrays(
                example,
                dreems_pattern=dreems_pattern,
                feems_pattern=feems_pattern,
            )
            values.extend(arrays[key] for key, _label, _prefix in methods)
        flattened = np.concatenate([np.abs(centered(value)) for value in values])
        finite = flattened[np.isfinite(flattened)]
        limits[truth] = max(float(np.quantile(finite, 0.98)) if finite.size else 0.0, 0.15)

    row_index = 0
    for truth in SCENARIO_TRUTHS:
        for mask in SCENARIO_MASKS:
            example = lookup[(replicate, truth, mask)]
            row = rows_by_id[example]
            arrays = _scenario_arrays(
                example,
                dreems_pattern=dreems_pattern,
                feems_pattern=feems_pattern,
            )
            observed = int(np.sum(arrays["sample_count"] > 0))
            design = str(row.get("sample_design") or mask)
            context = (
                f"{truth} | {design} | {observed}/96 nodes | {integer(row, 'num_snps'):,} SNPs"
            )
            for column_index, (key, label, prefix) in enumerate(methods):
                metric = (
                    "known generating surface"
                    if prefix is None
                    else _scenario_metric_title(
                        row,
                        method_prefix=prefix,
                        truth=truth,
                    )
                )
                plot_surface(
                    axes[row_index, column_index],
                    arrays["positions"],
                    arrays["edges"],
                    arrays[key],
                    arrays["sample_count"],
                    limit=limits[truth],
                    title=f"{label}\n{context}\n{metric}",
                )
            _plot_scenario_uncertainty(
                axes[row_index, -1],
                arrays["positions"],
                arrays["edges"],
                arrays["edge_sd"],
                arrays["sample_count"],
                title=(
                    "DREEMS posterior SD\n"
                    f"mean={numeric(row, 'posterior_sd_mean'):.3f} | "
                    f"95% coverage={numeric(row, 'neural_95_coverage'):.3f}"
                ),
            )
            row_index += 1
    fig.suptitle(f"Crossed sampling audit, replicate {replicate}", fontsize=15)
    save_figure(fig, output)
    plt.close(fig)


@beartype
def scenario_side_by_side_figure(
    rows_by_id: Mapping[int, Mapping[str, Any]],
    lookup: Mapping[tuple[int, str, str], int],
    *,
    replicate: int,
    dreems_pattern: str,
    feems_pattern: str,
    output: Path,
) -> None:
    """Render the direct truth, DREEMS, and FEEMS comparison first."""

    plt, _mtri, _LineCollection, _TwoSlopeNorm, _matplotlib = _plot_imports()
    methods = (
        ("truth", "Truth", None),
        ("dreems_map", "DREEMS + MAP", "dreems_map_edge_log"),
        ("feems", "FEEMS LOO", "feems_edge_log"),
    )
    row_count = len(SCENARIO_TRUTHS) * len(SCENARIO_MASKS)
    fig, axes = plt.subplots(
        row_count,
        len(methods),
        figsize=(12.0, 2.55 * row_count),
        squeeze=False,
        constrained_layout=True,
    )

    limits: dict[str, float] = {}
    for truth in SCENARIO_TRUTHS:
        values = []
        for mask in SCENARIO_MASKS:
            arrays = _scenario_arrays(
                lookup[(replicate, truth, mask)],
                dreems_pattern=dreems_pattern,
                feems_pattern=feems_pattern,
            )
            values.extend(arrays[key] for key, _label, _prefix in methods)
        finite = np.concatenate([np.abs(centered(value)) for value in values])
        finite = finite[np.isfinite(finite)]
        limits[truth] = max(float(np.quantile(finite, 0.98)) if finite.size else 0.0, 0.15)

    row_index = 0
    for truth in SCENARIO_TRUTHS:
        for mask in SCENARIO_MASKS:
            example = lookup[(replicate, truth, mask)]
            row = rows_by_id[example]
            arrays = _scenario_arrays(
                example,
                dreems_pattern=dreems_pattern,
                feems_pattern=feems_pattern,
            )
            observed = int(np.sum(arrays["sample_count"] > 0))
            design = str(row.get("sample_design") or mask)
            context = (
                f"{truth} | {design} | {observed}/96 nodes | {integer(row, 'num_snps'):,} SNPs"
            )
            for column_index, (key, label, prefix) in enumerate(methods):
                metric = (
                    "known generating surface"
                    if prefix is None
                    else _scenario_metric_title(row, method_prefix=prefix, truth=truth)
                )
                plot_surface(
                    axes[row_index, column_index],
                    arrays["positions"],
                    arrays["edges"],
                    arrays[key],
                    arrays["sample_count"],
                    limit=limits[truth],
                    title=f"{label}\n{context}\n{metric}",
                )
            row_index += 1
    fig.suptitle(
        f"DREEMS versus FEEMS on source-notebook cases, replicate {replicate}", fontsize=15
    )
    save_figure(fig, output)
    plt.close(fig)


@beartype
def _scenario_group_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Average each method's metrics within a truth and sampling mask."""
    output = []
    for truth in SCENARIO_TRUTHS:
        for mask in SCENARIO_MASKS:
            subset = [
                row
                for row in rows
                if row.get("scenario_truth") == truth and row.get("scenario_mask") == mask
            ]
            summary: dict[str, Any] = {"truth": truth, "mask": mask, "n": len(subset)}
            for _key, _label, prefix in SCENARIO_METHODS:
                for metric in ("corr", "mse", "rms"):
                    summary[f"{prefix}_{metric}"] = finite_mean(
                        numeric(row, f"{prefix}_{metric}") for row in subset
                    )
            output.append(summary)
    return output


@beartype
def _scenario_aggregate_figure(grouped: Sequence[Mapping[str, Any]], output: Path) -> None:
    """Show pattern recovery for structured truths and amplitude for homogeneous truth."""
    plt, _mtri, _LineCollection, _TwoSlopeNorm, _matplotlib = _plot_imports()
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), constrained_layout=True)
    colors = ("#777777", "#2a6fbb", "#00876c", "#b33b2e")
    markers = ("o", "s", "^", "D")
    x = np.arange(len(SCENARIO_MASKS))
    for column, truth in enumerate(SCENARIO_TRUTHS):
        local = {str(row["mask"]): row for row in grouped if row["truth"] == truth}
        for method_index, (_key, label, prefix) in enumerate(SCENARIO_METHODS):
            top_metric = "rms" if truth == "homogeneous" else "corr"
            axes[0, column].plot(
                x,
                [numeric(local[mask], f"{prefix}_{top_metric}") for mask in SCENARIO_MASKS],
                color=colors[method_index],
                marker=markers[method_index],
                linewidth=1.5,
                label=label,
            )
            axes[1, column].plot(
                x,
                [numeric(local[mask], f"{prefix}_mse") for mask in SCENARIO_MASKS],
                color=colors[method_index],
                marker=markers[method_index],
                linewidth=1.5,
            )
        axes[0, column].set_title(truth.capitalize())
        axes[0, column].set_ylabel("Output RMS" if truth == "homogeneous" else "Edge correlation")
        axes[1, column].set_ylabel("Edge MSE")
        for axis in axes[:, column]:
            axis.set_xticks(x, [_scenario_mask_label(mask) for mask in SCENARIO_MASKS])
            axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        if truth != "homogeneous":
            axes[0, column].set_ylim(-1.0, 1.02)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Recovery as sampling geometry changes", fontsize=14)
    save_figure(fig, output)
    plt.close(fig)


@jaxtyped(typechecker=beartype)
def _pairwise_corr(left: Real[np.ndarray, "edges"], right: Real[np.ndarray, "edges"]) -> float:
    """Compare centered patterns; a constant field has undefined correlation."""
    left = centered(left)
    right = centered(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else float("nan")


@beartype
def _scenario_similarity_rows(
    lookup: Mapping[tuple[int, str, str], int],
    *,
    dreems_pattern: str,
    feems_pattern: str,
) -> list[dict[str, Any]]:
    """Compare predicted patterns across truths, keeping the sampling mask fixed."""
    rows = []
    replicates = sorted({key[0] for key in lookup})
    truth_pairs = tuple(itertools.combinations(SCENARIO_TRUTHS, 2))
    for replicate in replicates:
        for mask in SCENARIO_MASKS:
            arrays = {
                truth: _scenario_arrays(
                    lookup[(replicate, truth, mask)],
                    dreems_pattern=dreems_pattern,
                    feems_pattern=feems_pattern,
                )
                for truth in SCENARIO_TRUTHS
            }
            for method, label, _prefix in SCENARIO_METHODS:
                for left_truth, right_truth in truth_pairs:
                    rows.append(
                        {
                            "replicate": replicate,
                            "mask": mask,
                            "method": label,
                            "truth_pair": f"{left_truth}_vs_{right_truth}",
                            "output_correlation": _pairwise_corr(
                                arrays[left_truth][method], arrays[right_truth][method]
                            ),
                        }
                    )
    return rows


@beartype
def _scenario_similarity_figure(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    """Plot output similarity across truths that share the same sampling mask."""
    plt, _mtri, _LineCollection, _TwoSlopeNorm, _matplotlib = _plot_imports()
    fig, ax = plt.subplots(figsize=(9.5, 4.6), constrained_layout=True)
    x = np.arange(len(SCENARIO_MASKS))
    colors = ("#777777", "#2a6fbb", "#00876c", "#b33b2e")
    markers = ("o", "s", "^", "D")
    for method_index, (_key, label, _prefix) in enumerate(SCENARIO_METHODS):
        means = []
        errors = []
        for mask in SCENARIO_MASKS:
            values = np.asarray(
                [
                    numeric(row, "output_correlation")
                    for row in rows
                    if row["mask"] == mask and row["method"] == label
                ],
                dtype=np.float64,
            )
            values = values[np.isfinite(values)]
            means.append(float(values.mean()) if values.size else float("nan"))
            errors.append(
                float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
            )
        ax.errorbar(
            x,
            means,
            yerr=errors,
            color=colors[method_index],
            marker=markers[method_index],
            linewidth=1.5,
            capsize=3,
            label=label,
        )
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xticks(x, [_scenario_mask_label(mask) for mask in SCENARIO_MASKS])
    ax.set_ylabel("Correlation between outputs for different truths")
    ax.set_title("Same-mask output similarity")
    ax.set_ylim(-1.0, 1.02)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, output)
    plt.close(fig)


@beartype
def _scenario_observability_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    dreems_pattern: str,
    feems_pattern: str,
) -> list[dict[str, Any]]:
    """Split edge error by the number of directly sampled endpoints.

    This is not a measure of statistical identifiability: distant samples can
    constrain an edge, and two sampled endpoints do not identify it uniquely.
    """
    output = []
    for row in rows:
        example = integer(row, "index")
        arrays = _scenario_arrays(
            example,
            dreems_pattern=dreems_pattern,
            feems_pattern=feems_pattern,
        )
        sampled = arrays["sample_count"] > 0
        edge_class = sampled[arrays["edges"]].sum(axis=1)
        truth = centered(arrays["truth"])
        for method, label, _prefix in SCENARIO_METHODS:
            estimate = centered(arrays[method])
            for sampled_endpoints in (0, 1, 2):
                keep = edge_class == sampled_endpoints
                if not np.any(keep):
                    continue
                residual = truth[keep] - estimate[keep]
                output.append(
                    {
                        "example": example,
                        "replicate": integer(row, "scenario_replicate"),
                        "truth": row["scenario_truth"],
                        "mask": row["scenario_mask"],
                        "method": label,
                        "sampled_endpoints": sampled_endpoints,
                        "edges": int(np.sum(keep)),
                        "mse": float(np.mean(residual * residual)),
                    }
                )
    return output


@beartype
def _scenario_observability_figure(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    """Group error by sampled edge endpoints, a direct-coverage diagnostic."""
    plt, _mtri, _LineCollection, _TwoSlopeNorm, _matplotlib = _plot_imports()
    fig, axes = plt.subplots(1, len(SCENARIO_MASKS), figsize=(15.5, 4.2), constrained_layout=True)
    colors = ("#777777", "#2a6fbb", "#00876c", "#b33b2e")
    markers = ("o", "s", "^", "D")
    for mask_index, mask in enumerate(SCENARIO_MASKS):
        ax = axes[mask_index]
        for method_index, (_key, label, _prefix) in enumerate(SCENARIO_METHODS):
            x_values = []
            means = []
            for sampled_endpoints in (0, 1, 2):
                values = [
                    numeric(row, "mse")
                    for row in rows
                    if row["mask"] == mask
                    and row["method"] == label
                    and integer(row, "sampled_endpoints") == sampled_endpoints
                ]
                finite = np.asarray(values, dtype=np.float64)
                finite = finite[np.isfinite(finite)]
                if finite.size:
                    x_values.append(sampled_endpoints)
                    means.append(float(finite.mean()))
            ax.plot(
                x_values,
                means,
                color=colors[method_index],
                marker=markers[method_index],
                linewidth=1.5,
                label=label,
            )
        ax.set_title(_scenario_mask_label(mask).replace("\n", " "))
        ax.set_xticks((0, 1, 2))
        ax.set_xlabel("Sampled edge endpoints")
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    axes[0].set_ylabel("Edge MSE")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Error by direct edge observability", fontsize=14)
    save_figure(fig, output)
    plt.close(fig)


@beartype
def command_scenario_gallery(args: argparse.Namespace) -> None:
    """Render the full crossed-design diagnosis using saved method outputs."""
    rows = read_csv_rows(args.rows)
    rows_by_id = _rows_by_id(rows, "index")
    lookup = _scenario_row_lookup(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grouped = _scenario_group_rows(rows)
    similarity = _scenario_similarity_rows(
        lookup,
        dreems_pattern=args.dreems_pattern,
        feems_pattern=args.feems_pattern,
    )
    observability = _scenario_observability_rows(
        rows,
        dreems_pattern=args.dreems_pattern,
        feems_pattern=args.feems_pattern,
    )
    write_rows(args.output_dir / "truth_mask_summary.csv", grouped)
    write_rows(args.output_dir / "same_mask_similarity.csv", similarity)
    write_rows(args.output_dir / "edge_observability.csv", observability)
    write_json(
        args.output_dir / "scenario_summary.json",
        {
            "truth_mask": grouped,
            "same_mask_similarity": similarity,
            "edge_observability": observability,
        },
    )
    _scenario_aggregate_figure(grouped, args.output_dir / "aggregate_by_truth_mask.png")
    _scenario_similarity_figure(similarity, args.output_dir / "same_mask_similarity.png")
    _scenario_observability_figure(observability, args.output_dir / "edge_observability_mse.png")

    replicate_sections = []
    for replicate in sorted({key[0] for key in lookup}):
        comparison_filename = f"dreems_vs_feems_replicate_{replicate:02d}.png"
        scenario_side_by_side_figure(
            rows_by_id,
            lookup,
            replicate=replicate,
            dreems_pattern=args.dreems_pattern,
            feems_pattern=args.feems_pattern,
            output=args.output_dir / comparison_filename,
        )
        filename = f"replicate_{replicate:02d}.png"
        scenario_replicate_figure(
            rows_by_id,
            lookup,
            replicate=replicate,
            dreems_pattern=args.dreems_pattern,
            feems_pattern=args.feems_pattern,
            output=args.output_dir / filename,
        )
        replicate_sections.append(
            f"<h3>Replicate {replicate}: direct comparison</h3>"
            f"<img src='{comparison_filename}'>"
            f"<h3>Replicate {replicate}: full diagnostic</h3><img src='{filename}'>"
        )

    table_columns = ["truth", "mask", "n"]
    for _key, _label, prefix in SCENARIO_METHODS:
        table_columns.extend([f"{prefix}_corr", f"{prefix}_mse", f"{prefix}_rms"])
    body = (
        "<h2>Aggregate diagnosis</h2>"
        "<img src='aggregate_by_truth_mask.png'>"
        "<img src='same_mask_similarity.png'>"
        "<img src='edge_observability_mse.png'>"
        "<p><a href='truth_mask_summary.csv'>truth/mask CSV</a> | "
        "<a href='same_mask_similarity.csv'>same-mask CSV</a> | "
        "<a href='edge_observability.csv'>edge-observability CSV</a> | "
        "<a href='../rows.csv'>all H2H rows</a> | "
        "<a href='../protocol.json'>protocol</a></p>"
        + html_table(grouped, table_columns)
        + "<h2>Paired surfaces</h2>"
        + "".join(replicate_sections)
    )
    simulators = {str(row.get("simulator", "")) for row in rows}
    if simulators == {"msprime_exact_notebook"}:
        intro = (
            "These are the exact msprime simulations and sample subsets from "
            "feems_scenarios.ipynb. Each truth is simulated once and reused across its "
            "four masks; MAF filtering is applied after subsetting, as in the notebook. "
            "The random anisotropy cell samples 20% of nodes, while the other random cells "
            "sample 10%. DREEMS uses the checkpoint recorded in the protocol and FEEMS uses "
            "its documented leave-one-observed-deme-out fit. Colors are centered edge-log "
            "relative migration and share one scale within each truth."
        )
    else:
        intro = (
            "This audit crosses three generating surfaces with four sampling masks over "
            "independent simulation replicates. SNP counts and sample sizes are shown "
            "for each case. Panels separate the linearized initializer, raw neural estimate, "
            "fixed-radius MAP refinement, and FEEMS' documented leave-one-deme-out fit. "
            "Colors are centered edge-log relative migration and use one common scale within "
            "each truth and replicate."
        )
    with _atomic_output(args.output_dir / "index.html") as stream:
        stream.write(gallery_html("Sampling-mask scenario audit", intro, body).encode("utf-8"))


SNP_COMPARISON_METHODS = (
    ("dreems", "Raw DREEMS"),
    ("dreems_map", "DREEMS + MAP"),
    ("feems", "FEEMS LOO"),
)


@beartype
def merge_scenario_snp_rows(
    low_rows_path: Path,
    high_rows_path: Path,
) -> list[dict[str, Any]]:
    """Pair scenario results that differ only in their retained SNP count."""

    low_by_id = _rows_by_id(read_csv_rows(low_rows_path), "index")
    high_by_id = _rows_by_id(read_csv_rows(high_rows_path), "index")
    if low_by_id.keys() != high_by_id.keys():
        raise ValueError("low- and high-SNP reports contain different example IDs")

    paired = []
    invariant_fields = (
        "scenario_truth",
        "scenario_mask",
        "scenario_replicate",
        "scenario_seed",
        "num_nodes",
        "num_edges",
        "num_samples",
        "num_sampled_nodes",
    )
    for example in sorted(low_by_id):
        low = low_by_id[example]
        high = high_by_id[example]
        disagreements = [key for key in invariant_fields if low.get(key) != high.get(key)]
        if disagreements:
            raise ValueError(
                f"scenario {example} differs on paired fields: {', '.join(disagreements)}"
            )
        row: dict[str, Any] = {
            "index": example,
            "truth": low["scenario_truth"],
            "mask": low["scenario_mask"],
            "sampled_nodes": integer(low, "num_sampled_nodes"),
            "total_nodes": integer(low, "num_nodes"),
            "low_snps": integer(low, "num_snps"),
            "high_snps": integer(high, "num_snps"),
        }
        if row["high_snps"] <= row["low_snps"]:
            raise ValueError(f"scenario {example} does not increase the retained SNP count")
        for prefix, _label in SNP_COMPARISON_METHODS:
            for metric in ("corr", "mse", "rms"):
                low_value = numeric(low, f"{prefix}_edge_log_{metric}")
                high_value = numeric(high, f"{prefix}_edge_log_{metric}")
                row[f"low_{prefix}_{metric}"] = low_value
                row[f"high_{prefix}_{metric}"] = high_value
                if metric == "corr":
                    row[f"{prefix}_{metric}_gain"] = high_value - low_value
                else:
                    row[f"{prefix}_{metric}_reduction"] = low_value - high_value
        paired.append(row)
    return paired


@beartype
def _scenario_snp_change_summary(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize SNP-count gains using only finite paired metrics."""
    summaries = []
    scopes = (
        ("all cases", rows, "mse", "reduction"),
        (
            "structured truths",
            [row for row in rows if row["truth"] != "homogeneous"],
            "corr",
            "gain",
        ),
        (
            "homogeneous truth",
            [row for row in rows if row["truth"] == "homogeneous"],
            "rms",
            "reduction",
        ),
    )
    for prefix, label in SNP_COMPARISON_METHODS:
        for scope, subset, metric, direction in scopes:
            low = np.asarray([numeric(row, f"low_{prefix}_{metric}") for row in subset])
            high = np.asarray([numeric(row, f"high_{prefix}_{metric}") for row in subset])
            keep = np.isfinite(low) & np.isfinite(high)
            low = low[keep]
            high = high[keep]
            change = high - low if metric == "corr" else low - high
            summaries.append(
                {
                    "method": label,
                    "scope": scope,
                    "metric": metric,
                    "n": int(change.size),
                    "low_mean": finite_mean(low),
                    "high_mean": finite_mean(high),
                    f"mean_{direction}": finite_mean(change),
                    "high_better_fraction": (
                        float(np.mean(change > 0.0)) if change.size else float("nan")
                    ),
                }
            )
    return summaries


@beartype
def _scenario_snp_delta_figure(
    rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    """Plot within-case changes in correlation and MSE as SNP count increases."""
    plt, _mtri, _LineCollection, _TwoSlopeNorm, _matplotlib = _plot_imports()
    counts = {integer(row, "high_snps") for row in rows}
    high_label = f"{next(iter(counts)):,} SNPs" if len(counts) == 1 else "higher SNP counts"
    structured = [row for row in rows if row["truth"] != "homogeneous"]
    truth_colors = {"barrier": "#b33b2e", "anisotropy": "#2a6fbb", "homogeneous": "#777777"}
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 7.8), constrained_layout=True)
    for column, (prefix, label) in enumerate(SNP_COMPARISON_METHODS):
        corr_values = [numeric(row, f"{prefix}_corr_gain") for row in structured]
        axes[0, column].bar(
            np.arange(len(structured)),
            corr_values,
            color=[truth_colors[str(row["truth"])] for row in structured],
        )
        axes[0, column].set_xticks(
            np.arange(len(structured)),
            [
                f"{row['truth']}\n{_scenario_mask_label(str(row['mask'])).replace(chr(10), ' ')}"
                for row in structured
            ],
            rotation=35,
            ha="right",
            fontsize=7,
        )
        mse_values = [numeric(row, f"{prefix}_mse_reduction") for row in rows]
        axes[1, column].bar(
            np.arange(len(rows)),
            mse_values,
            color=[truth_colors[str(row["truth"])] for row in rows],
        )
        axes[1, column].set_xticks(
            np.arange(len(rows)),
            [
                f"{row['truth']}\n{_scenario_mask_label(str(row['mask'])).replace(chr(10), ' ')}"
                for row in rows
            ],
            rotation=35,
            ha="right",
            fontsize=7,
        )
        axes[0, column].set_title(label)
        for axis in axes[:, column]:
            axis.axhline(0.0, color="#202020", linewidth=0.8)
            axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    axes[0, 0].set_ylabel(f"Correlation gain at {high_label}")
    axes[1, 0].set_ylabel(f"MSE reduction at {high_label}")
    fig.suptitle("Effect of increasing the retained SNP count", fontsize=14)
    save_figure(fig, output)
    plt.close(fig)


@beartype
def _scenario_snp_surface_figure(
    low_rows: Mapping[int, Mapping[str, Any]],
    high_rows: Mapping[int, Mapping[str, Any]],
    lookup: Mapping[tuple[int, str, str], int],
    *,
    low_dreems_pattern: str,
    low_feems_pattern: str,
    high_dreems_pattern: str,
    high_feems_pattern: str,
    output: Path,
) -> None:
    """Show matched low/high-SNP surfaces at a common scale for each truth."""
    plt, _mtri, _LineCollection, _TwoSlopeNorm, _matplotlib = _plot_imports()
    fig, axes = plt.subplots(
        len(SCENARIO_TRUTHS) * len(SCENARIO_MASKS),
        5,
        figsize=(20.0, 31.0),
        squeeze=False,
        constrained_layout=True,
    )
    limits: dict[str, float] = {}
    for truth in SCENARIO_TRUTHS:
        values = []
        for mask in SCENARIO_MASKS:
            example = lookup[(0, truth, mask)]
            low = _scenario_arrays(
                example,
                dreems_pattern=low_dreems_pattern,
                feems_pattern=low_feems_pattern,
            )
            high = _scenario_arrays(
                example,
                dreems_pattern=high_dreems_pattern,
                feems_pattern=high_feems_pattern,
            )
            for key in ("positions", "edges", "truth", "sample_count"):
                if not np.array_equal(low[key], high[key]):
                    raise ValueError(f"scenario {example} is not paired on {key}")
            values.extend(
                (high["truth"], low["dreems_map"], high["dreems_map"], low["feems"], high["feems"])
            )
        finite = np.concatenate([np.abs(centered(value)) for value in values])
        finite = finite[np.isfinite(finite)]
        limits[truth] = max(float(np.quantile(finite, 0.98)) if finite.size else 0.0, 0.15)

    row_index = 0
    for truth in SCENARIO_TRUTHS:
        for mask in SCENARIO_MASKS:
            example = lookup[(0, truth, mask)]
            low_arrays = _scenario_arrays(
                example,
                dreems_pattern=low_dreems_pattern,
                feems_pattern=low_feems_pattern,
            )
            high_arrays = _scenario_arrays(
                example,
                dreems_pattern=high_dreems_pattern,
                feems_pattern=high_feems_pattern,
            )
            low_row = low_rows[example]
            high_row = high_rows[example]
            panels = (
                (high_arrays["truth"], "Truth", "known generating surface"),
                (
                    low_arrays["dreems_map"],
                    f"DREEMS + MAP | {integer(low_row, 'num_snps')} SNPs",
                    _scenario_metric_title(
                        low_row, method_prefix="dreems_map_edge_log", truth=truth
                    ),
                ),
                (
                    high_arrays["dreems_map"],
                    f"DREEMS + MAP | {integer(high_row, 'num_snps'):,} SNPs",
                    _scenario_metric_title(
                        high_row, method_prefix="dreems_map_edge_log", truth=truth
                    ),
                ),
                (
                    low_arrays["feems"],
                    f"FEEMS LOO | {integer(low_row, 'num_snps')} SNPs",
                    _scenario_metric_title(low_row, method_prefix="feems_edge_log", truth=truth),
                ),
                (
                    high_arrays["feems"],
                    f"FEEMS LOO | {integer(high_row, 'num_snps'):,} SNPs",
                    _scenario_metric_title(high_row, method_prefix="feems_edge_log", truth=truth),
                ),
            )
            context = (
                f"{truth} | {mask} | {integer(high_row, 'num_sampled_nodes')}/"
                f"{integer(high_row, 'num_nodes')} nodes"
            )
            for column, (values, label, metric) in enumerate(panels):
                plot_surface(
                    axes[row_index, column],
                    high_arrays["positions"],
                    high_arrays["edges"],
                    values,
                    high_arrays["sample_count"],
                    limit=limits[truth],
                    title=f"{label}\n{context}\n{metric}",
                )
            row_index += 1
    counts = {integer(row, "num_snps") for row in high_rows.values()}
    high_label = f"{next(iter(counts)):,}-SNP" if len(counts) == 1 else "higher-SNP"
    fig.suptitle(f"Paired low-SNP and {high_label} recovery", fontsize=15)
    save_figure(fig, output)
    plt.close(fig)


@beartype
def command_scenario_snp_comparison(args: argparse.Namespace) -> None:
    """Compare paired saved scenarios with different retained SNP counts."""
    paired = merge_scenario_snp_rows(args.low_rows, args.high_rows)
    low_rows = _rows_by_id(read_csv_rows(args.low_rows), "index")
    high_rows = _rows_by_id(read_csv_rows(args.high_rows), "index")
    low_lookup = _scenario_row_lookup(list(low_rows.values()))
    high_lookup = _scenario_row_lookup(list(high_rows.values()))
    if low_lookup != high_lookup or set(key[0] for key in high_lookup) != {0}:
        raise ValueError("SNP comparison currently requires one exactly paired replicate")
    counts = {integer(row, "high_snps") for row in paired}
    high_label = f"{next(iter(counts)):,}-SNP" if len(counts) == 1 else "higher-SNP"
    low_gallery = os.path.relpath(args.low_rows.parent / "gallery/index.html", args.output_dir)
    high_gallery = os.path.relpath(args.high_rows.parent / "gallery/index.html", args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = _scenario_snp_change_summary(paired)
    write_rows(args.output_dir / "paired_cases.csv", paired)
    write_rows(args.output_dir / "aggregate_changes.csv", summaries)
    write_json(
        args.output_dir / "summary.json",
        {
            "paired_cases": paired,
            "aggregate_changes": summaries,
            "protocol": {
                "low_rows": str(args.low_rows),
                "high_rows": str(args.high_rows),
                "low_dreems_pattern": args.low_dreems_pattern,
                "low_feems_pattern": args.low_feems_pattern,
                "high_dreems_pattern": args.high_dreems_pattern,
                "high_feems_pattern": args.high_feems_pattern,
                "pairing": "matched graph, truth, sampling design, and seed; genotype prefix not verified by this report",
            },
        },
    )
    _scenario_snp_delta_figure(paired, args.output_dir / "metric_changes.png")
    _scenario_snp_surface_figure(
        low_rows,
        high_rows,
        high_lookup,
        low_dreems_pattern=args.low_dreems_pattern,
        low_feems_pattern=args.low_feems_pattern,
        high_dreems_pattern=args.high_dreems_pattern,
        high_feems_pattern=args.high_feems_pattern,
        output=args.output_dir / "paired_surfaces.png",
    )

    summary_columns = (
        "method",
        "scope",
        "metric",
        "n",
        "low_mean",
        "high_mean",
        "mean_gain",
        "mean_reduction",
        "high_better_fraction",
    )
    case_columns = (
        "truth",
        "mask",
        "sampled_nodes",
        "low_snps",
        "high_snps",
        "low_dreems_map_corr",
        "high_dreems_map_corr",
        "low_dreems_map_mse",
        "high_dreems_map_mse",
        "low_feems_corr",
        "high_feems_corr",
        "low_feems_mse",
        "high_feems_mse",
    )
    body = (
        "<h2>Aggregate changes</h2>"
        "<img src='metric_changes.png'>"
        "<p><a href='aggregate_changes.csv'>aggregate CSV</a> | "
        "<a href='paired_cases.csv'>paired cases CSV</a> | "
        "<a href='summary.json'>JSON</a> | "
        f"<a href='{html.escape(high_gallery, quote=True)}'>higher-SNP gallery</a> | "
        f"<a href='{html.escape(low_gallery, quote=True)}'>low-SNP gallery</a></p>"
        + html_table(summaries, summary_columns)
        + "<h2>Paired surfaces</h2><img src='paired_surfaces.png'>"
        + "<h2>Per-case metrics</h2>"
        + html_table(paired, case_columns)
    )
    intro = (
        "This is a paired SNP-count comparison. For every case, the graph, generating "
        "surface, sampled nodes, sample counts, and simulation seed are fixed. The original "
        "and higher retained SNP counts are listed for each case. This report does not "
        "verify whether the genotype matrices share an exact prefix. FEEMS "
        "uses the same documented leave-one-observed-deme-out procedure at both SNP counts. "
        "In the change plot, values above zero favor the higher SNP count."
    )
    with _atomic_output(args.output_dir / "index.html") as stream:
        stream.write(
            gallery_html(f"Low-SNP versus {high_label} hard-edge audit", intro, body).encode(
                "utf-8"
            )
        )


@beartype
def build_parser() -> argparse.ArgumentParser:
    """Expose separate report and rendering commands for Snakemake."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser("compare-checkpoints")
    compare.add_argument("--old-summary", type=Path, required=True)
    compare.add_argument("--new-summary", type=Path, required=True)
    compare.add_argument("--metadata", type=Path, nargs="+", required=True)
    compare.add_argument("--old-checkpoint", type=Path, required=True)
    compare.add_argument("--new-checkpoint", type=Path, required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.set_defaults(func=command_compare_checkpoints)

    checkpoint_gallery = subparsers.add_parser("checkpoint-gallery")
    checkpoint_gallery.add_argument("--rows", type=Path, required=True)
    checkpoint_gallery.add_argument("--summary", type=Path, required=True)
    checkpoint_gallery.add_argument("--selections", type=Path, required=True)
    checkpoint_gallery.add_argument("--raw-pattern", required=True)
    checkpoint_gallery.add_argument("--prediction-pattern", required=True)
    checkpoint_gallery.add_argument("--pack-size", type=int, required=True)
    checkpoint_gallery.add_argument("--output-dir", type=Path, required=True)
    checkpoint_gallery.set_defaults(func=command_checkpoint_gallery)

    router_report = subparsers.add_parser("router-report")
    router_report.add_argument("--rows", type=Path, required=True)
    router_report.add_argument("--threshold", type=int, default=512)
    router_report.add_argument("--output-dir", type=Path, required=True)
    router_report.set_defaults(func=command_router_report)

    router_gallery = subparsers.add_parser("router-gallery")
    router_gallery.add_argument("--rows", type=Path, required=True)
    router_gallery.add_argument("--summary", type=Path, required=True)
    router_gallery.add_argument("--selections", type=Path, required=True)
    router_gallery.add_argument("--raw-pattern", required=True)
    router_gallery.add_argument("--prediction-pattern", required=True)
    router_gallery.add_argument("--pack-size", type=int, required=True)
    router_gallery.add_argument("--threshold", type=int, default=512)
    router_gallery.add_argument("--output-dir", type=Path, required=True)
    router_gallery.set_defaults(func=command_router_gallery)

    router_h2h = subparsers.add_parser("router-h2h-gallery")
    router_h2h.add_argument("--reference-gallery", type=Path, required=True)
    router_h2h.add_argument("--reference-rows", type=Path, required=True)
    router_h2h.add_argument("--threshold", type=int, default=512)
    router_h2h.add_argument("--output-dir", type=Path, required=True)
    router_h2h.set_defaults(func=command_router_h2h_gallery)

    h2h_gallery = subparsers.add_parser("h2h-gallery")
    h2h_gallery.add_argument("--rows", type=Path, required=True)
    h2h_gallery.add_argument("--dreems-pattern", required=True)
    h2h_gallery.add_argument("--feems-pattern", required=True)
    h2h_gallery.add_argument("--output-dir", type=Path, required=True)
    h2h_gallery.set_defaults(func=command_h2h_gallery)

    scenario_gallery = subparsers.add_parser("scenario-gallery")
    scenario_gallery.add_argument("--rows", type=Path, required=True)
    scenario_gallery.add_argument("--dreems-pattern", required=True)
    scenario_gallery.add_argument("--feems-pattern", required=True)
    scenario_gallery.add_argument("--output-dir", type=Path, required=True)
    scenario_gallery.set_defaults(func=command_scenario_gallery)

    scenario_snp = subparsers.add_parser("scenario-snp-comparison")
    scenario_snp.add_argument("--low-rows", type=Path, required=True)
    scenario_snp.add_argument("--high-rows", type=Path, required=True)
    scenario_snp.add_argument("--low-dreems-pattern", required=True)
    scenario_snp.add_argument("--low-feems-pattern", required=True)
    scenario_snp.add_argument("--high-dreems-pattern", required=True)
    scenario_snp.add_argument("--high-feems-pattern", required=True)
    scenario_snp.add_argument("--output-dir", type=Path, required=True)
    scenario_snp.set_defaults(func=command_scenario_snp_comparison)
    return parser


@beartype
def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch one report command without loading a neural checkpoint."""
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
