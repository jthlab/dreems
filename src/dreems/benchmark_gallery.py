"""Build reproducible visual reports for DREEMS benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import html
import itertools
import json
import pickle
import shutil
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from .workflow_io import json_ready, write_json

SNP_BINS = ((50, 128), (129, 512), (513, 2000), (2001, 10000))
SELECTION_SEED = 20260827


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def integer(row: Mapping[str, Any], key: str) -> int:
    return int(round(numeric(row, key)))


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def snp_bin_label(num_snps: int) -> str:
    for low, high in SNP_BINS:
        if low <= num_snps <= high:
            return f"{low:,}-{high:,}"
    raise ValueError(f"SNP count {num_snps} is outside the configured bins")


def paired_stats(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, Any]:
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

    def delta_summary(values: np.ndarray) -> tuple[float, float, float, float]:
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
        "corr_win_fraction": float(np.mean(corr_delta > 0.0)),
        "old_mse": finite_mean(old_mse),
        "new_mse": finite_mean(new_mse),
        "mse_advantage": mse_mean,
        "mse_advantage_ci95_low": mse_low,
        "mse_advantage_ci95_high": mse_high,
        "mse_advantage_p": mse_p,
        "mse_win_fraction": float(np.mean(mse_advantage > 0.0)),
    }


def h2h_stats(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, Any]:
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
    return {
        "label": result["label"],
        "n": result["n"],
        "n_corr": result["n_corr"],
        "n_mse": result["n_mse"],
        "feems_corr": result["old_corr"],
        "dreems_corr": result["new_corr"],
        "corr_delta": result["corr_delta"],
        "corr_delta_ci95_low": result["corr_delta_ci95_low"],
        "corr_delta_ci95_high": result["corr_delta_ci95_high"],
        "corr_delta_p": result["corr_delta_p"],
        "corr_win_fraction": result["corr_win_fraction"],
        "feems_mse": result["old_mse"],
        "dreems_mse": result["new_mse"],
        "mse_advantage": result["mse_advantage"],
        "mse_advantage_ci95_low": result["mse_advantage_ci95_low"],
        "mse_advantage_ci95_high": result["mse_advantage_ci95_high"],
        "mse_advantage_p": result["mse_advantage_p"],
        "mse_win_fraction": result["mse_win_fraction"],
    }


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(json_ready(list(rows)))


def load_metadata_rows(paths: Sequence[Path]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        for row in payload["rows"]:
            index = int(row["index"])
            if index in rows:
                raise ValueError(f"duplicate metadata index {index}")
            rows[index] = dict(row)
    return rows


def merge_checkpoint_rows(
    old_summary: Path,
    new_summary: Path,
    metadata_paths: Sequence[Path],
) -> list[dict[str, Any]]:
    old_payload = json.loads(old_summary.read_text())
    new_payload = json.loads(new_summary.read_text())
    old = {int(row["replicate_global"]): row for row in old_payload["rows"]}
    new = {int(row["replicate_global"]): row for row in new_payload["rows"]}
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
            "snp_bin": snp_bin_label(int(meta["num_snps"])),
            **meta,
        }
        for name in metric_names:
            row[f"old_{name}"] = old[replicate].get(name)
            row[f"new_{name}"] = new[replicate].get(name)
        row["corr_delta"] = numeric(row, "new_edge_log_corr") - numeric(row, "old_edge_log_corr")
        row["mse_advantage"] = numeric(row, "old_edge_log_mse") - numeric(row, "new_edge_log_mse")
        rows.append(row)
    return rows


def deterministic_sample(
    rows: Sequence[Mapping[str, Any]],
    count: int,
    seed: int,
    *,
    key: str = "replicate",
) -> list[int]:
    ids = np.asarray([integer(row, key) for row in rows], dtype=np.int64)
    if ids.size <= count:
        return ids.tolist()
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(ids, size=count, replace=False).tolist())


def checkpoint_selections(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
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


def command_compare_checkpoints(args: argparse.Namespace) -> None:
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


def _plot_imports() -> tuple[Any, Any, Any, Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.collections import LineCollection
    from matplotlib.colors import TwoSlopeNorm

    return plt, mtri, LineCollection, TwoSlopeNorm, matplotlib


def centered(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array - float(np.mean(array))


def edge_to_node(values: np.ndarray, edges: np.ndarray, num_nodes: int) -> np.ndarray:
    totals = np.zeros(num_nodes, dtype=np.float64)
    counts = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(totals, edges[:, 0], values)
    np.add.at(totals, edges[:, 1], values)
    np.add.at(counts, edges[:, 0], 1.0)
    np.add.at(counts, edges[:, 1], 1.0)
    return np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0.0)


def plot_surface(
    ax: Any,
    positions: np.ndarray,
    edges: np.ndarray,
    values: np.ndarray,
    sample_count: np.ndarray,
    *,
    limit: float,
    title: str,
) -> None:
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


def prediction_slice(path: Path, replicate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        replicates = np.asarray(archive["replicate"], dtype=np.int64)
        matches = np.flatnonzero(replicates == int(replicate))
        if matches.size != 1:
            raise ValueError(f"expected one prediction for replicate {replicate} in {path}")
        index = int(matches[0])
        offsets = np.asarray(archive["edge_offsets"], dtype=np.int64)
        start, stop = int(offsets[index]), int(offsets[index + 1])
        return (
            np.asarray(archive["edge_truth"][start:stop], dtype=np.float64),
            np.asarray(archive["edge_mean"][start:stop], dtype=np.float64),
            np.asarray(archive["edge_sd"][start:stop], dtype=np.float64),
        )


@lru_cache(maxsize=8)
def load_pickle_pack(path: str) -> list[dict[str, Any]]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def raw_example(pattern: str, pack_size: int, replicate: int) -> Mapping[str, Any]:
    pack = replicate // pack_size
    local = replicate % pack_size
    examples = load_pickle_pack(pattern.format(pack=pack))
    return examples[local]


def example_geometry(example: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(example["active_positions"], dtype=np.float64)
    edges = np.asarray(example["graph"]["edges"], dtype=np.int32)
    sample_nodes = np.asarray(example["sample_nodes"], dtype=np.int32)
    sample_count = np.bincount(sample_nodes, minlength=positions.shape[0]).astype(np.float64)
    return positions, edges, sample_count


def save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def checkpoint_aggregate_figures(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    plt, _mtri, _LineCollection, _TwoSlopeNorm, _matplotlib = _plot_imports()
    old_corr = np.asarray([numeric(row, "old_edge_log_corr") for row in rows])
    new_corr = np.asarray([numeric(row, "new_edge_log_corr") for row in rows])
    old_mse = np.asarray([numeric(row, "old_edge_log_mse") for row in rows])
    new_mse = np.asarray([numeric(row, "new_edge_log_mse") for row in rows])
    snps = np.asarray([numeric(row, "num_snps") for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    colors = np.log10(snps)
    scatter = axes[0].scatter(old_corr, new_corr, c=colors, s=9, cmap="viridis", alpha=0.62)
    finite_corr = np.concatenate([old_corr[np.isfinite(old_corr)], new_corr[np.isfinite(new_corr)]])
    low = float(finite_corr.min())
    axes[0].plot([low, 1.0], [low, 1.0], color="#222222", linewidth=1)
    axes[0].set(xlabel="Previous checkpoint correlation", ylabel="Broad-SNP correlation")
    axes[0].set_aspect("equal", adjustable="box")
    axes[1].scatter(old_mse, new_mse, c=colors, s=9, cmap="viridis", alpha=0.62)
    positive = np.concatenate([old_mse[old_mse > 0], new_mse[new_mse > 0]])
    lower, upper = float(positive.min()), float(positive.max())
    axes[1].plot([lower, upper], [lower, upper], color="#222222", linewidth=1)
    axes[1].set(xscale="log", yscale="log")
    axes[1].set(xlabel="Previous checkpoint MSE", ylabel="Broad-SNP MSE")
    axes[1].set_aspect("equal", adjustable="box")
    colorbar = fig.colorbar(scatter, ax=axes, fraction=0.035, pad=0.03)
    colorbar.set_label("log10(SNP count)")
    save_figure(fig, output_dir / "checkpoint_scatter.png")
    plt.close(fig)

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


def checkpoint_example_figure(
    rows_by_id: Mapping[int, Mapping[str, Any]],
    replicate_ids: Sequence[int],
    *,
    raw_pattern: str,
    prediction_pattern: str,
    pack_size: int,
    output: Path,
) -> None:
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
            old_mean,
            sample_count,
            limit=limit,
            title=(
                f"Previous | r={numeric(row, 'old_edge_log_corr'):.3f}, "
                f"MSE={numeric(row, 'old_edge_log_mse'):.3f}"
            ),
        )
        plot_surface(
            axes[row_index, 2],
            positions,
            edges,
            new_mean,
            sample_count,
            limit=limit,
            title=(
                f"Broad SNP | r={numeric(row, 'new_edge_log_corr'):.3f}, "
                f"MSE={numeric(row, 'new_edge_log_mse'):.3f}"
            ),
        )
    save_figure(fig, output)
    plt.close(fig)


def html_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
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


def gallery_html(title: str, intro: str, body: str) -> str:
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


def command_checkpoint_gallery(args: argparse.Namespace) -> None:
    rows = read_csv_rows(args.rows)
    rows_by_id = {integer(row, "replicate"): row for row in rows}
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
        filename = f"examples_{section_index:02d}.png"
        checkpoint_example_figure(
            rows_by_id,
            [int(value) for value in replicate_ids],
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
    (args.output_dir / "index.html").write_text(
        gallery_html("SNP-count checkpoint comparison", intro, body)
    )


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


def route_checkpoint(row: Mapping[str, Any], threshold: int) -> str:
    return "broad" if integer(row, "num_snps") <= threshold else "released"


def build_router_rows(rows: Sequence[Mapping[str, Any]], threshold: int) -> list[dict[str, Any]]:
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


def router_stats(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, Any]:
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
    result["router_mse_win_fraction_vs_alternative"] = float(np.mean(mse_advantage > 0.0))
    result["router_corr_win_fraction_vs_alternative"] = float(np.mean(corr_advantage > 0.0))
    return result


def router_selections(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
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
    finite_corr = [row for row in rows if np.isfinite(numeric(row, "router_edge_log_corr"))]
    selections["Router helps most"] = [
        integer(row, "replicate")
        for row in sorted(
            rows,
            key=lambda row: numeric(row, "router_mse_advantage_over_alternative"),
            reverse=True,
        )[:4]
    ]
    selections["Router hurts most"] = [
        integer(row, "replicate")
        for row in sorted(
            rows,
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


def command_router_report(args: argparse.Namespace) -> None:
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


def router_aggregate_figures(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
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
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    color = np.log10(snps)
    scatter = axes[0].scatter(
        alternative_corr, router_corr, c=color, s=9, cmap="viridis", alpha=0.62
    )
    finite_corr = np.concatenate(
        [
            alternative_corr[np.isfinite(alternative_corr)],
            router_corr[np.isfinite(router_corr)],
        ]
    )
    low = float(finite_corr.min())
    axes[0].plot([low, 1.0], [low, 1.0], color="#222222", linewidth=1)
    axes[0].set(xlabel="Unselected checkpoint correlation", ylabel="Router correlation")
    axes[0].set_aspect("equal", adjustable="box")
    axes[1].scatter(alternative_mse, router_mse, c=color, s=9, cmap="viridis", alpha=0.62)
    positive = np.concatenate(
        [alternative_mse[alternative_mse > 0.0], router_mse[router_mse > 0.0]]
    )
    lower, upper = float(positive.min()), float(positive.max())
    axes[1].plot([lower, upper], [lower, upper], color="#222222", linewidth=1)
    axes[1].set(xscale="log", yscale="log")
    axes[1].set(xlabel="Unselected checkpoint MSE", ylabel="Router MSE")
    axes[1].set_aspect("equal", adjustable="box")
    colorbar = fig.colorbar(scatter, ax=axes, fraction=0.035, pad=0.03)
    colorbar.set_label("log10(SNP count)")
    save_figure(fig, output_dir / "router_vs_alternative.png")
    plt.close(fig)


def router_example_figure(
    rows_by_id: Mapping[int, Mapping[str, Any]],
    replicate_ids: Sequence[int],
    *,
    raw_pattern: str,
    prediction_pattern: str,
    pack_size: int,
    output: Path,
) -> None:
    plt, _mtri, _LineCollection, _TwoSlopeNorm, _matplotlib = _plot_imports()
    fig, axes = plt.subplots(
        len(replicate_ids),
        4,
        figsize=(16.5, max(3.0 * len(replicate_ids), 3.2)),
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
        selected = str(row["router_checkpoint"])
        selected_label = "Broad-SNP" if selected == "broad" else "Released"
        router_mean = new_mean if selected == "broad" else old_mean
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
            old_mean,
            sample_count,
            limit=limit,
            title=(
                f"Released\nr={numeric(row, 'old_edge_log_corr'):.3f}, "
                f"MSE={numeric(row, 'old_edge_log_mse'):.3f}"
            ),
        )
        plot_surface(
            axes[row_index, 2],
            positions,
            edges,
            new_mean,
            sample_count,
            limit=limit,
            title=(
                f"Broad-SNP\nr={numeric(row, 'new_edge_log_corr'):.3f}, "
                f"MSE={numeric(row, 'new_edge_log_mse'):.3f}"
            ),
        )
        plot_surface(
            axes[row_index, 3],
            positions,
            edges,
            router_mean,
            sample_count,
            limit=limit,
            title=(
                f"Router: {selected_label}\nr={numeric(row, 'router_edge_log_corr'):.3f}, "
                f"MSE={numeric(row, 'router_edge_log_mse'):.3f}"
            ),
        )
        axes[row_index, 3].spines[:].set_color("#2474a6")
        axes[row_index, 3].spines[:].set_linewidth(1.4)
    save_figure(fig, output)
    plt.close(fig)


def command_router_gallery(args: argparse.Namespace) -> None:
    rows = read_csv_rows(args.rows)
    rows_by_id = {integer(row, "replicate"): row for row in rows}
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
        filename = f"examples_{section_index:02d}.png"
        router_example_figure(
            rows_by_id,
            [int(value) for value in selections[label]],
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
    (args.output_dir / "index.html").write_text(
        gallery_html("SNP-aware checkpoint router", intro, body)
    )


def command_router_h2h_gallery(args: argparse.Namespace) -> None:
    rows = read_csv_rows(args.reference_rows)
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

    shutil.copytree(
        args.reference_gallery,
        args.output_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".snakemake_timestamp"),
    )
    write_rows(args.output_dir / "router_selections.csv", selections)
    source_provenance_path = args.reference_gallery / "provenance.json"
    source_provenance = (
        json.loads(source_provenance_path.read_text()) if source_provenance_path.is_file() else None
    )
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
    page = index_path.read_text()
    page = page.replace(
        "DREEMS + fixed-radius MAP vs FEEMS recommended LOO",
        "DREEMS router + fixed-radius MAP vs FEEMS recommended LOO",
    ).replace("DREEMS + MAP", "DREEMS router + MAP")
    marker = "</h1>"
    if marker not in page:
        raise ValueError("reference gallery does not contain the expected heading")
    router_note = (
        "<p class='note'><strong>Router audit:</strong> The router uses Broad-SNP at or "
        f"below {args.threshold:,} SNPs and Released above it. All {released_count:,} "
        "examples in this FEEMS LOO benchmark have more than 2,000 SNPs, so the router "
        "selected Released for every case. The surfaces and numerical results are therefore "
        "identical to the Released-checkpoint H2H. "
        "<a href='router_selections.csv'>router selections</a></p>"
    )
    index_path.write_text(page.replace(marker, marker + router_note, 1))


def h2h_selections(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    random_ids = deterministic_sample(rows, 8, SELECTION_SEED, key="index")
    corr_sorted = sorted(
        rows,
        key=lambda row: (
            numeric(row, "dreems_map_edge_log_corr") - numeric(row, "feems_edge_log_corr")
        ),
    )
    mse_sorted = sorted(
        rows,
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


def h2h_example_figure(
    rows_by_id: Mapping[int, Mapping[str, Any]],
    replicate_ids: Sequence[int],
    *,
    dreems_pattern: str,
    feems_pattern: str,
    output: Path,
) -> None:
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
        with (
            np.load(dreems_pattern.format(example=replicate)) as dreems,
            np.load(feems_pattern.format(example=replicate)) as feems,
        ):
            positions = np.asarray(dreems["positions"], dtype=np.float64)
            edges = np.asarray(dreems["edges"], dtype=np.int32)
            truth = centered(np.asarray(dreems["truth"], dtype=np.float64))
            dreems_map = np.asarray(dreems["dreems_map"], dtype=np.float64)
            feems_mean = np.asarray(feems["feems"], dtype=np.float64)
            sample_count = np.asarray(dreems["sample_count"], dtype=np.float64)
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


def h2h_aggregate_figures(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    plt, _mtri, _LineCollection, _TwoSlopeNorm, _matplotlib = _plot_imports()
    feems_corr = np.asarray([numeric(row, "feems_edge_log_corr") for row in rows])
    dreems_corr = np.asarray([numeric(row, "dreems_map_edge_log_corr") for row in rows])
    feems_mse = np.asarray([numeric(row, "feems_edge_log_mse") for row in rows])
    dreems_mse = np.asarray([numeric(row, "dreems_map_edge_log_mse") for row in rows])
    density = np.asarray([numeric(row, "sampled_node_fraction") for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    scatter = axes[0].scatter(feems_corr, dreems_corr, c=density, s=12, cmap="plasma", alpha=0.7)
    low = float(np.nanmin([feems_corr.min(), dreems_corr.min()]))
    axes[0].plot([low, 1.0], [low, 1.0], color="#222222", linewidth=1)
    axes[0].set(xlabel="FEEMS LOO correlation", ylabel="DREEMS+MAP correlation")
    axes[0].set_aspect("equal", adjustable="box")
    axes[1].scatter(feems_mse, dreems_mse, c=density, s=12, cmap="plasma", alpha=0.7)
    positive = np.concatenate([feems_mse[feems_mse > 0], dreems_mse[dreems_mse > 0]])
    lower, upper = float(positive.min()), float(positive.max())
    axes[1].plot([lower, upper], [lower, upper], color="#222222", linewidth=1)
    axes[1].set(xscale="log", yscale="log")
    axes[1].set(xlabel="FEEMS LOO MSE", ylabel="DREEMS+MAP MSE")
    axes[1].set_aspect("equal", adjustable="box")
    colorbar = fig.colorbar(scatter, ax=axes, fraction=0.035, pad=0.03)
    colorbar.set_label("Fraction of graph nodes sampled")
    save_figure(fig, output_dir / "h2h_scatter_corr_mse.png")
    plt.close(fig)


def command_h2h_gallery(args: argparse.Namespace) -> None:
    rows = read_csv_rows(args.rows)
    rows_by_id = {integer(row, "index"): row for row in rows}
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
        "DREEMS uses the broad-SNP checkpoint and fixed-radius MAP refinement. FEEMS uses "
        "the cached recommended leave-one-observed-deme-out fit. Colors show centered "
        "edge-log relative migration: red is lower, white is the graph mean, and blue is "
        "higher. Each row uses the truth scale for all three panels. White points are "
        "sampled graph nodes, sized by sample count."
    )
    (args.output_dir / "index.html").write_text(
        gallery_html("DREEMS versus FEEMS: corrected broad-SNP checkpoint", intro, body)
    )


SCENARIO_TRUTHS = ("barrier", "anisotropy", "homogeneous")
SCENARIO_MASKS = ("full", "random_10pct", "outer_slabs", "left_missing")
SCENARIO_METHODS = (
    ("conditioner", "Linearized initializer", "conditioner_edge_log"),
    ("dreems", "Raw DREEMS", "dreems_edge_log"),
    ("dreems_map", "DREEMS + MAP", "dreems_map_edge_log"),
    ("feems", "FEEMS LOO", "feems_edge_log"),
)


def _scenario_mask_label(mask: str) -> str:
    if mask == "random_10pct":
        return "random\nsparse"
    return mask.replace("_", "\n")


@lru_cache(maxsize=256)
def _scenario_archive(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _scenario_arrays(
    example: int,
    *,
    dreems_pattern: str,
    feems_pattern: str,
) -> dict[str, np.ndarray]:
    dreems = _scenario_archive(dreems_pattern.format(example=example))
    feems = _scenario_archive(feems_pattern.format(example=example))
    for key in ("positions", "edges", "truth", "sample_count"):
        if not np.array_equal(dreems[key], feems[key]):
            raise ValueError(f"scenario {example} disagrees between DREEMS and FEEMS on {key}")
    return {
        "positions": np.asarray(dreems["positions"], dtype=np.float64),
        "edges": np.asarray(dreems["edges"], dtype=np.int32),
        "truth": np.asarray(dreems["truth"], dtype=np.float64),
        "sample_count": np.asarray(dreems["sample_count"], dtype=np.float64),
        "edge_sd": np.asarray(dreems["edge_sd"], dtype=np.float64),
        "conditioner": np.asarray(dreems["conditioner"], dtype=np.float64),
        "dreems": np.asarray(dreems["dreems"], dtype=np.float64),
        "dreems_map": np.asarray(dreems["dreems_map"], dtype=np.float64),
        "feems": np.asarray(feems["feems"], dtype=np.float64),
    }


def _scenario_row_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str, str], int]:
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


def _scenario_metric_title(
    row: Mapping[str, Any],
    *,
    method_prefix: str,
    truth: str,
) -> str:
    mse = numeric(row, f"{method_prefix}_mse")
    if truth == "homogeneous":
        rms = numeric(row, f"{method_prefix}_rms")
        return f"RMS={rms:.3f} | MSE={mse:.3f}"
    corr = numeric(row, f"{method_prefix}_corr")
    return f"r={corr:.3f} | MSE={mse:.3f}"


def _plot_scenario_uncertainty(
    ax: Any,
    positions: np.ndarray,
    edges: np.ndarray,
    edge_sd: np.ndarray,
    sample_count: np.ndarray,
    *,
    title: str,
) -> None:
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


def scenario_replicate_figure(
    rows_by_id: Mapping[int, Mapping[str, Any]],
    lookup: Mapping[tuple[int, str, str], int],
    *,
    replicate: int,
    dreems_pattern: str,
    feems_pattern: str,
    output: Path,
) -> None:
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


def _scenario_group_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
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


def _scenario_aggregate_figure(grouped: Sequence[Mapping[str, Any]], output: Path) -> None:
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


def _pairwise_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = centered(left)
    right = centered(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else float("nan")


def _scenario_similarity_rows(
    lookup: Mapping[tuple[int, str, str], int],
    *,
    dreems_pattern: str,
    feems_pattern: str,
) -> list[dict[str, Any]]:
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


def _scenario_similarity_figure(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
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


def _scenario_observability_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    dreems_pattern: str,
    feems_pattern: str,
) -> list[dict[str, Any]]:
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


def _scenario_observability_figure(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
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


def command_scenario_gallery(args: argparse.Namespace) -> None:
    rows = read_csv_rows(args.rows)
    rows_by_id = {integer(row, "index"): row for row in rows}
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
            "independent simulation replicates. Each simulation uses constant q and 1,000 "
            "diploid SNPs. Panels separate the linearized initializer, raw neural estimate, "
            "fixed-radius MAP refinement, and FEEMS' documented leave-one-deme-out fit. "
            "Colors are centered edge-log relative migration and use one common scale within "
            "each truth and replicate."
        )
    (args.output_dir / "index.html").write_text(
        gallery_html("Sampling-mask scenario audit", intro, body)
    )


def build_parser() -> argparse.ArgumentParser:
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
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
