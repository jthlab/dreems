"""Compare the learned null-surface gate with core DREEMS and FEEMS.

The gate changes only the amplitude of a centered migration surface. This
module builds matched reports where every method sees the same simulated data,
graph, and sampling mask. It deliberately consumes cached FEEMS fits because
the gate does not change FEEMS inputs or tuning.
"""

# ruff: noqa: F722,F821,UP037

from __future__ import annotations

import argparse
import csv
import html
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from beartype import beartype
from jaxtyping import Int, Real, jaxtyped

from .validation import _integer_array, _real_array, validate_graph_arrays
from .workflow_io import _atomic_output, json_ready, parse_nonnegative_integer, write_csv
from .workflow_io import sha256 as sha256
from .workflow_io import write_json as write_json
from .workflow_io import write_npz as _atomic_npz

SCENARIO_TRUTHS = ("barrier", "anisotropy", "homogeneous")
SCENARIO_MASKS = ("full", "random_10pct", "outer_slabs", "left_missing")
METHODS = ("core", "gated", "feems")
METHOD_LABELS = {
    "core": "Core DREEMS + MAP",
    "gated": "Gated DREEMS + MAP",
    "feems": "FEEMS LOO",
}


@dataclass(frozen=True)
class EdgeMetrics:
    """Metrics for one centered edge field."""

    correlation: float
    mse: float
    rms: float


@beartype
def centered(values: np.ndarray) -> np.ndarray:
    """Return a finite one-dimensional array with its mean removed."""

    array = _real_array(values, name="edge values")
    if array.ndim != 1 or array.size == 0:
        raise ValueError("edge values must be a nonempty finite vector")
    return array - float(array.mean())


@beartype
def edge_metrics(truth: np.ndarray, estimate: np.ndarray) -> EdgeMetrics:
    """Measure centered pattern correlation and squared error."""

    target = centered(truth)
    prediction = centered(estimate)
    if target.shape != prediction.shape:
        raise ValueError("truth and estimate must have the same shape")
    residual = prediction - target
    mse = float(np.mean(residual * residual))
    denominator = float(np.linalg.norm(target) * np.linalg.norm(prediction))
    correlation = (
        float(np.dot(target, prediction) / denominator)
        if denominator > np.finfo(np.float64).eps
        else float("nan")
    )
    return EdgeMetrics(
        correlation=correlation,
        mse=mse,
        rms=float(np.sqrt(np.mean(prediction * prediction))),
    )


@beartype
def projected_alpha(core: np.ndarray, gated: np.ndarray) -> float:
    """Project a gated surface onto its core; MAP may also change the pattern."""

    anchor = centered(core)
    candidate = centered(gated)
    if anchor.shape != candidate.shape:
        raise ValueError("core and gated surfaces must have the same shape")
    denominator = float(np.dot(anchor, anchor))
    if denominator <= np.finfo(np.float64).eps:
        return float("nan")
    return float(np.dot(anchor, candidate) / denominator)


@beartype
def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load an NPZ archive without permitting pickled arrays."""

    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


@beartype
def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows in their existing order, converting NumPy scalars for CSV."""
    write_csv(path, json_ready(rows))


@beartype
def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read CSV values as text; callers parse numeric columns explicitly."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


@beartype
def _scalar_text(array: np.ndarray) -> str:
    """Read one archive metadata value without accepting an entire vector."""
    value = np.asarray(array)
    if value.size != 1:
        raise ValueError("expected one scalar string")
    return str(value.reshape(()).item())


@beartype
def _validate_equal(
    expected: np.ndarray,
    observed: np.ndarray,
    *,
    name: str,
    example: int,
) -> None:
    """Require exact graph and sampling alignment between cached methods."""
    if expected.shape != observed.shape or not np.array_equal(expected, observed):
        raise ValueError(f"scenario {example} has mismatched {name}")


@beartype
def build_scenario_bundle(
    *,
    core_bundle_path: Path,
    gated_pattern: str,
    gated_checkpoint: Path,
    output: Path,
    rows_output: Path,
    report_output: Path,
) -> None:
    """Merge the 12 exact notebook cases into one matched comparison."""

    core_hash = sha256(core_bundle_path)
    gate_checkpoint_hash = sha256(gated_checkpoint)
    core = load_npz(core_bundle_path)
    indices = _integer_array(core["index"], name="index", dtype=np.int64)
    if not np.array_equal(indices, np.arange(12, dtype=np.int64)):
        raise ValueError("the core scenario bundle must contain examples 0 through 11")

    # Check the fixed notebook design before assigning figures to named sections.
    if not np.array_equal(core["truth_name"], np.repeat(SCENARIO_TRUTHS, 4)) or not np.array_equal(
        core["mask_name"], np.tile(SCENARIO_MASKS, 3)
    ):
        raise ValueError("scenario names must follow the notebook's truth and mask order")
    core["edges"] = _integer_array(core["edges"], name="edges", dtype=np.int32)
    core["num_snps"] = _integer_array(core["num_snps"], name="num_snps", dtype=np.int64)
    if core["num_snps"].shape != (12,) or np.any(core["num_snps"] < 1):
        raise ValueError("num_snps must contain a positive count for each scenario")
    for name in ("positions", "truth", "dreems", "dreems_map", "edge_sd", "sample_count", "feems"):
        core[name] = _real_array(core[name], name=name)
        if core[name].ndim == 0 or core[name].shape[0] != 12:
            raise ValueError(f"{name} must contain all 12 scenarios")
    if np.any(core["edge_sd"] < 0) or np.any(core["sample_count"] < 0):
        raise ValueError("posterior SDs and sample counts must be nonnegative")

    gated_neural = []
    gated_map = []
    gated_sd = []
    for example in indices.tolist():
        validate_graph_arrays(positions=core["positions"][example], edges=core["edges"][example])
        edge_shape = (core["edges"].shape[1],)
        if core["sample_count"][example].shape != (core["positions"].shape[1],):
            raise ValueError("sample counts must align with graph nodes")
        for name in ("truth", "dreems", "dreems_map", "edge_sd", "feems"):
            if core[name][example].shape != edge_shape:
                raise ValueError(f"{name} must align with graph edges")
        gated = load_npz(Path(gated_pattern.format(example=example)))
        if (
            "checkpoint_sha256" in gated
            and _scalar_text(gated["checkpoint_sha256"]) != gate_checkpoint_hash
        ):
            raise ValueError(f"scenario {example} was predicted by a different gated checkpoint")
        for name in ("positions", "edges", "truth", "sample_count"):
            _validate_equal(
                np.asarray(core[name][example]),
                np.asarray(gated[name]),
                name=name,
                example=example,
            )
        for name in ("dreems", "dreems_map", "edge_sd"):
            gated[name] = _real_array(gated[name], name=name)
            if gated[name].shape != edge_shape:
                raise ValueError(f"gated {name} must align with graph edges")
        if np.any(gated["edge_sd"] < 0):
            raise ValueError("gated posterior SD must be nonnegative")
        gated_neural.append(gated["dreems"])
        gated_map.append(gated["dreems_map"])
        gated_sd.append(gated["edge_sd"])

    core_neural = np.asarray(core["dreems"], dtype=np.float64)
    core_map = np.asarray(core["dreems_map"], dtype=np.float64)
    truth = np.asarray(core["truth"], dtype=np.float64)
    feems = np.asarray(core["feems"], dtype=np.float64)
    gated_neural_array = np.stack(gated_neural)
    gated_map_array = np.stack(gated_map)
    gated_sd_array = np.stack(gated_sd)

    metric_arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    metrics_by_method: dict[str, list[EdgeMetrics]] = {method: [] for method in METHODS}
    neural_alphas = []
    map_alphas = []
    for example in indices.tolist():
        estimates = {
            "core": core_map[example],
            "gated": gated_map_array[example],
            "feems": feems[example],
        }
        metrics = {
            method: edge_metrics(truth[example], estimate) for method, estimate in estimates.items()
        }
        for method in METHODS:
            metrics_by_method[method].append(metrics[method])
        neural_alpha = projected_alpha(core_neural[example], gated_neural_array[example])
        map_alpha = projected_alpha(core_map[example], gated_map_array[example])
        neural_alphas.append(neural_alpha)
        map_alphas.append(map_alpha)
        row: dict[str, Any] = {
            "index": example,
            "truth": str(core["truth_name"][example]),
            "mask": str(core["mask_name"][example]),
            "sample_design": str(core["sample_design"][example]),
            "observed_nodes": int(np.count_nonzero(core["sample_count"][example])),
            "num_snps": int(core["num_snps"][example]),
            "gate_alpha_neural": neural_alpha,
            "gate_alpha_map": map_alpha,
        }
        for method, values in metrics.items():
            row[f"{method}_corr"] = values.correlation
            row[f"{method}_mse"] = values.mse
            row[f"{method}_rms"] = values.rms
        rows.append(row)

    for method in METHODS:
        metric_arrays[f"{method}_corr"] = np.asarray(
            [value.correlation for value in metrics_by_method[method]], dtype=np.float64
        )
        metric_arrays[f"{method}_mse"] = np.asarray(
            [value.mse for value in metrics_by_method[method]], dtype=np.float64
        )
        metric_arrays[f"{method}_rms"] = np.asarray(
            [value.rms for value in metrics_by_method[method]], dtype=np.float64
        )

    core_checkpoint_hash = _scalar_text(core["checkpoint_sha256"])
    bundle_arrays = {
        "index": indices,
        "truth_name": np.asarray(core["truth_name"]),
        "mask_name": np.asarray(core["mask_name"]),
        "sample_design": np.asarray(core["sample_design"]),
        "num_snps": np.asarray(core["num_snps"], dtype=np.int64),
        "positions": np.asarray(core["positions"], dtype=np.float64),
        "edges": np.asarray(core["edges"], dtype=np.int32),
        "truth": truth,
        "core_neural": core_neural,
        "core_map": core_map,
        "core_edge_sd": np.asarray(core["edge_sd"], dtype=np.float64),
        "gated_neural": gated_neural_array,
        "gated_map": gated_map_array,
        "gated_edge_sd": gated_sd_array,
        "feems": feems,
        "sample_count": np.asarray(core["sample_count"], dtype=np.float64),
        "gate_alpha_neural": np.asarray(neural_alphas, dtype=np.float64),
        "gate_alpha_map": np.asarray(map_alphas, dtype=np.float64),
        "source_notebook_sha256": np.asarray(_scalar_text(core["source_notebook_sha256"])),
        "core_checkpoint_sha256": np.asarray(core_checkpoint_hash),
        "gated_checkpoint_sha256": np.asarray(gate_checkpoint_hash),
        "feems_protocol": np.asarray(_scalar_text(core["feems_protocol"])),
        **metric_arrays,
    }
    if sha256(core_bundle_path) != core_hash or sha256(gated_checkpoint) != gate_checkpoint_hash:
        raise ValueError("comparison input changed while building the scenario bundle")
    _atomic_npz(output, **bundle_arrays)
    write_rows(rows_output, rows)
    report = scenario_report(rows)
    report["provenance"] = {
        "core_bundle": str(core_bundle_path.resolve()),
        "core_bundle_sha256": core_hash,
        "core_checkpoint_sha256": core_checkpoint_hash,
        "gated_checkpoint": str(gated_checkpoint.resolve()),
        "gated_checkpoint_sha256": gate_checkpoint_hash,
        "source_notebook_sha256": _scalar_text(core["source_notebook_sha256"]),
        "feems_protocol": _scalar_text(core["feems_protocol"]),
    }
    write_json(report_output, report)


@beartype
def _finite_mean(values: Sequence[float]) -> float:
    """Exclude undefined correlations without turning homogeneous cases into zeroes."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


@beartype
def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Give each case equal weight and report paired MSE differences."""
    result: dict[str, Any] = {"examples": len(rows)}
    for method in METHODS:
        result[method] = {
            "mean_corr": _finite_mean([float(row[f"{method}_corr"]) for row in rows]),
            "mean_mse": _finite_mean([float(row[f"{method}_mse"]) for row in rows]),
            "mean_rms": _finite_mean([float(row[f"{method}_rms"]) for row in rows]),
        }
    core_mse = np.asarray([float(row["core_mse"]) for row in rows])
    gated_mse = np.asarray([float(row["gated_mse"]) for row in rows])
    feems_mse = np.asarray([float(row["feems_mse"]) for row in rows])
    result["gated_vs_core"] = {
        "mean_mse_advantage": float(np.mean(core_mse - gated_mse)),
        "mse_win_fraction": float(np.mean(gated_mse < core_mse)),
    }
    result["gated_vs_feems"] = {
        "mean_mse_advantage": float(np.mean(feems_mse - gated_mse)),
        "mse_win_fraction": float(np.mean(gated_mse < feems_mse)),
    }
    return result


@beartype
def scenario_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize the exact notebook comparison by generating surface."""

    report: dict[str, Any] = {
        "stage": "surface_gate_scenario_comparison",
        "overall": _group_summary(rows),
        "by_truth": {},
        "interpretation": (
            "All methods use the same 12 notebook cases; each surface simulation is reused across four sampling masks. "
            "Positive MSE advantage means that gated DREEMS has lower edge MSE."
        ),
    }
    for truth in SCENARIO_TRUTHS:
        selected = [row for row in rows if str(row["truth"]) == truth]
        report["by_truth"][truth] = _group_summary(selected)
    return report


@beartype
def _plot_imports() -> tuple[Any, Any, Any, Any]:
    """Keep optional plotting imports out of workers that only assemble arrays."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import TwoSlopeNorm

    return plt, LineCollection, TwoSlopeNorm, matplotlib


@jaxtyped(typechecker=beartype)
def _draw_edges(
    axis: Any,
    positions: Real[np.ndarray, "nodes 2"],
    edges: Int[np.ndarray, "edges 2"],
    values: Real[np.ndarray, "edges"],
    sample_count: Real[np.ndarray, "nodes"],
    *,
    limit: float,
    title: str,
) -> Any:
    """Draw the supplied graph without interpolating or changing its edge order."""
    validate_graph_arrays(positions=positions, edges=edges)
    if np.any(_real_array(sample_count, name="sample_count") < 0):
        raise ValueError("sample counts must be nonnegative")
    _plt, LineCollection, TwoSlopeNorm, _matplotlib = _plot_imports()
    local = centered(values)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    linewidth = float(np.clip(1200.0 / max(edges.shape[0], 1), 0.35, 2.8))
    lines = LineCollection(
        positions[edges],
        cmap="RdBu",
        norm=norm,
        linewidths=linewidth,
        capstyle="round",
    )
    lines.set_array(local)
    axis.add_collection(lines)
    sampled = np.flatnonzero(sample_count > 0)
    if sampled.size:
        relative = sample_count[sampled] / max(float(sample_count[sampled].max()), 1.0)
        axis.scatter(
            positions[sampled, 0],
            positions[sampled, 1],
            s=7.0 + 24.0 * np.sqrt(relative),
            facecolor="white",
            edgecolor="#202020",
            linewidth=0.4,
            zorder=4,
        )
    axis.set_title(title, fontsize=8.5)
    axis.set_aspect("equal", adjustable="datalim")
    axis.autoscale()
    axis.margins(0.035)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(0.5)
    return lines


@beartype
def _save_figure(figure: Any, path: Path) -> None:
    """Replace the display figure and its PDF only after each file is complete."""
    with _atomic_output(path) as stream:
        figure.savefig(
            stream, format=path.suffix.lstrip("."), dpi=210, bbox_inches="tight", facecolor="white"
        )
    with _atomic_output(path.with_suffix(".pdf")) as stream:
        figure.savefig(stream, format="pdf", bbox_inches="tight", facecolor="white")


@beartype
def _metric_text(bundle: Mapping[str, np.ndarray], method: str, index: int) -> str:
    """Show output amplitude when a homogeneous truth has no defined correlation."""
    mse = float(bundle[f"{method}_mse"][index])
    correlation = float(bundle[f"{method}_corr"][index])
    if np.isfinite(correlation):
        return f"r={correlation:.3f} | MSE={mse:.3f}"
    return f"RMS={float(bundle[f'{method}_rms'][index]):.3f} | MSE={mse:.3f}"


@beartype
def scenario_family_figure(bundle: Mapping[str, np.ndarray], truth_name: str) -> Any:
    """Draw raw edges for core, gated, and FEEMS estimates."""

    if truth_name not in SCENARIO_TRUTHS:
        raise ValueError(f"unknown scenario truth {truth_name!r}")
    plt, _LineCollection, _TwoSlopeNorm, _matplotlib = _plot_imports()
    indices = np.flatnonzero(np.asarray(bundle["truth_name"]).astype(str) == truth_name)
    figure, axes = plt.subplots(
        len(indices),
        4,
        figsize=(17.0, 2.75 * len(indices)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, index in enumerate(indices.tolist()):
        values = (
            np.asarray(bundle["truth"][index]),
            np.asarray(bundle["core_map"][index]),
            np.asarray(bundle["gated_map"][index]),
            np.asarray(bundle["feems"][index]),
        )
        absolute = np.concatenate([np.abs(centered(value)) for value in values])
        limit = max(0.1, float(np.quantile(absolute, 0.995)))
        labels = (
            "Truth\nknown generating surface",
            f"{METHOD_LABELS['core']}\n{_metric_text(bundle, 'core', index)}",
            (
                f"{METHOD_LABELS['gated']}\n{_metric_text(bundle, 'gated', index)} | "
                f"alpha={float(bundle['gate_alpha_map'][index]):.3f}"
            ),
            f"{METHOD_LABELS['feems']}\n{_metric_text(bundle, 'feems', index)}",
        )
        lines = []
        for column, (value, label) in enumerate(zip(values, labels, strict=True)):
            lines.append(
                _draw_edges(
                    axes[row_index, column],
                    np.asarray(bundle["positions"][index], dtype=np.float64),
                    np.asarray(bundle["edges"][index], dtype=np.int32),
                    np.asarray(value, dtype=np.float64),
                    np.asarray(bundle["sample_count"][index], dtype=np.float64),
                    limit=limit,
                    title=label,
                )
            )
        design = str(bundle["sample_design"][index]).replace("_", " ")
        observed = int(np.count_nonzero(bundle["sample_count"][index]))
        axes[row_index, 0].set_ylabel(
            f"{design}\n{observed}/96 nodes | {int(bundle['num_snps'][index])} SNPs",
            rotation=0,
            ha="right",
            va="center",
            fontsize=8.5,
            labelpad=9,
        )
        colorbar = figure.colorbar(
            lines[-1],
            ax=axes[row_index, :],
            fraction=0.015,
            pad=0.008,
            aspect=24,
        )
        colorbar.ax.tick_params(labelsize=7)
    figure.suptitle(f"{truth_name.title()} surface", fontsize=14)
    return figure


@beartype
def _scenario_summary_rows(bundle: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    """Summarize each named truth without weighting by graph size."""
    rows = []
    truth_names = np.asarray(bundle["truth_name"]).astype(str)
    for truth in SCENARIO_TRUTHS:
        indices = np.flatnonzero(truth_names == truth)
        row: dict[str, Any] = {"truth": truth, "n": int(indices.size)}
        for method in METHODS:
            row[f"{method}_corr"] = _finite_mean(bundle[f"{method}_corr"][indices].tolist())
            row[f"{method}_mse"] = _finite_mean(bundle[f"{method}_mse"][indices].tolist())
            row[f"{method}_rms"] = _finite_mean(bundle[f"{method}_rms"][indices].tolist())
        row["gate_alpha"] = _finite_mean(bundle["gate_alpha_map"][indices].tolist())
        rows.append(row)
    return rows


@beartype
def scenario_aggregate_figure(bundle: Mapping[str, np.ndarray]) -> Any:
    """Plot method performance across sampling masks for each truth."""

    plt, _LineCollection, _TwoSlopeNorm, _matplotlib = _plot_imports()
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.0), constrained_layout=True)
    colors = {"core": "#555555", "gated": "#278a67", "feems": "#b33b2e"}
    markers = {"core": "o", "gated": "s", "feems": "D"}
    truth_names = np.asarray(bundle["truth_name"]).astype(str)
    mask_names = np.asarray(bundle["mask_name"]).astype(str)
    x = np.arange(len(SCENARIO_MASKS))
    for column, truth in enumerate(SCENARIO_TRUTHS):
        indices = [
            int(np.flatnonzero((truth_names == truth) & (mask_names == mask))[0])
            for mask in SCENARIO_MASKS
        ]
        for method in METHODS:
            top_metric = "rms" if truth == "homogeneous" else "corr"
            axes[0, column].plot(
                x,
                bundle[f"{method}_{top_metric}"][indices],
                color=colors[method],
                marker=markers[method],
                linewidth=1.5,
                label=METHOD_LABELS[method],
            )
            axes[1, column].plot(
                x,
                bundle[f"{method}_mse"][indices],
                color=colors[method],
                marker=markers[method],
                linewidth=1.5,
            )
        axes[0, column].set_title(truth.title())
        axes[0, column].set_ylabel("Output RMS" if truth == "homogeneous" else "Edge correlation")
        axes[1, column].set_ylabel("Edge MSE")
        for axis in axes[:, column]:
            axis.set_xticks(x, [mask.replace("_", "\n") for mask in SCENARIO_MASKS])
            axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Core DREEMS, gated DREEMS, and FEEMS on matched scenarios", fontsize=14)
    return figure


@beartype
def _html_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    """Escape cell text and display undefined numeric summaries as NA."""
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                rendered = "NA" if not np.isfinite(value) else f"{value:.4f}"
            else:
                rendered = str(value)
            cells.append(f"<td>{html.escape(rendered)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


@beartype
def _gallery_html(title: str, intro: str, body: str) -> str:
    """Wrap generated figures and tables in the comparison's static page layout."""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ max-width: 1500px; margin: 2rem auto; padding: 0 1rem; font-family: system-ui, sans-serif; color: #202020; }}
img {{ display: block; width: 100%; height: auto; margin: 1rem 0 2rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; }}
th, td {{ border-bottom: 1px solid #d8d8d8; padding: 0.45rem; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
code {{ background: #f1f1f1; padding: 0.1rem 0.25rem; }}
</style></head><body><h1>{html.escape(title)}</h1><p>{html.escape(intro)}</p>{body}</body></html>"""


@beartype
def render_scenario_gallery(*, bundle_path: Path, report_path: Path, output_dir: Path) -> None:
    """Render the 12-case raw-edge comparison and aggregate table."""

    bundle = load_npz(bundle_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = scenario_aggregate_figure(bundle)
    _save_figure(aggregate, output_dir / "aggregate.png")
    plt, _LineCollection, _TwoSlopeNorm, _matplotlib = _plot_imports()
    plt.close(aggregate)
    sections = ["<h2>Aggregate comparison</h2><img src='aggregate.png'>"]
    for truth in SCENARIO_TRUTHS:
        figure = scenario_family_figure(bundle, truth)
        filename = f"{truth}_raw_edges.png"
        _save_figure(figure, output_dir / filename)
        plt.close(figure)
        sections.append(f"<h2>{html.escape(truth.title())}</h2><img src='{filename}'>")
    rows = _scenario_summary_rows(bundle)
    columns = [
        "truth",
        "n",
        "core_corr",
        "gated_corr",
        "feems_corr",
        "core_mse",
        "gated_mse",
        "feems_mse",
        "gate_alpha",
    ]
    links = (
        f"<p><a href='{html.escape(str(bundle_path))}'>comparison NPZ</a> | "
        f"<a href='{html.escape(str(report_path))}'>report JSON</a></p>"
    )
    body = links + _html_table(rows, columns) + "".join(sections)
    intro = (
        "All panels use identical msprime simulations, MAF filtering, graph geometry, and "
        "sampling masks. FEEMS is the cached documented leave-one-observed-deme-out fit. "
        "Colors show centered edge-log relative migration on raw graph edges, with one "
        "shared scale per row."
    )
    with _atomic_output(output_dir / "index.html") as stream:
        stream.write(
            _gallery_html("Surface-gate comparison on FEEMS scenarios", intro, body).encode("utf-8")
        )


@beartype
def _as_float(row: Mapping[str, Any], key: str) -> float:
    """Parse a CSV metric; NaN remains meaningful for undefined correlation."""
    return float(row[key])


@beartype
def _as_int(row: Mapping[str, Any], key: str = "example") -> int:
    """Accept integral CSV values without truncation or float rounding of IDs."""
    return parse_nonnegative_integer(row[key], name=key)


@beartype
def select_broad_examples(
    rows: Sequence[Mapping[str, Any]], count: int = 6
) -> dict[str, list[int]]:
    """Select deterministic random, strongest, best, and worst gate cases."""

    if isinstance(count, bool) or count < 1:
        raise ValueError("count must be positive")
    if len(rows) < count:
        raise ValueError("not enough broad rows for gallery selection")
    rng = np.random.default_rng(20260831)
    identifiers = np.asarray([_as_int(row) for row in rows], dtype=np.int64)
    if len(set(identifiers.tolist())) != len(rows):
        raise ValueError("broad rows contain duplicate example IDs")
    random_ids = rng.choice(identifiers, size=count, replace=False).tolist()
    strongest = sorted(rows, key=lambda row: _as_float(row, "alpha"))[:count]
    helps = sorted(
        rows,
        key=lambda row: _as_float(row, "baseline_map_mse") - _as_float(row, "gated_map_mse"),
        reverse=True,
    )[:count]
    hurts = sorted(
        rows,
        key=lambda row: _as_float(row, "gated_map_mse") - _as_float(row, "baseline_map_mse"),
        reverse=True,
    )[:count]
    return {
        "Random cases": [int(value) for value in random_ids],
        "Strongest gate shrinkage": [_as_int(row) for row in strongest],
        "Largest gate improvements": [_as_int(row) for row in helps],
        "Largest gate regressions": [_as_int(row) for row in hurts],
    }


@beartype
def _broad_case_arrays(
    example: int,
    *,
    core_pattern: str,
    gated_pattern: str,
    feems_pattern: str,
) -> dict[str, np.ndarray]:
    """Load matching cached graphs; never use one method's geometry for another."""
    core = load_npz(Path(core_pattern.format(example=example)))
    gated = load_npz(Path(gated_pattern.format(example=example)))
    feems = load_npz(Path(feems_pattern.format(example=example)))
    for name in ("positions", "edges", "truth", "sample_count"):
        _validate_equal(core[name], gated[name], name=name, example=example)
        _validate_equal(core[name], feems[name], name=name, example=example)
    return {
        "positions": _real_array(core["positions"], name="positions"),
        "edges": _integer_array(core["edges"], name="edges", dtype=np.int32),
        "truth": _real_array(core["truth"], name="truth"),
        "core": _real_array(core["dreems_map"], name="core estimate"),
        "gated": _real_array(gated["dreems_map"], name="gated estimate"),
        "feems": _real_array(feems["feems"], name="FEEMS estimate"),
        "sample_count": _real_array(core["sample_count"], name="sample_count"),
    }


@beartype
def _broad_examples_figure(
    rows_by_id: Mapping[int, Mapping[str, Any]],
    examples: Sequence[int],
    *,
    core_pattern: str,
    gated_pattern: str,
    feems_pattern: str,
) -> Any:
    """Render selected cases with one shared color scale per row."""
    plt, _LineCollection, _TwoSlopeNorm, _matplotlib = _plot_imports()
    figure, axes = plt.subplots(
        len(examples),
        4,
        figsize=(17.0, 2.8 * len(examples)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, example in enumerate(examples):
        row = rows_by_id[example]
        arrays = _broad_case_arrays(
            example,
            core_pattern=core_pattern,
            gated_pattern=gated_pattern,
            feems_pattern=feems_pattern,
        )
        values = (arrays["truth"], arrays["core"], arrays["gated"], arrays["feems"])
        absolute = np.concatenate([np.abs(centered(value)) for value in values])
        limit = max(0.1, float(np.quantile(absolute, 0.995)))
        titles = (
            f"Truth\nexample {example} | known surface",
            (
                f"{METHOD_LABELS['core']}\n"
                f"r={_as_float(row, 'baseline_map_corr'):.3f} | "
                f"MSE={_as_float(row, 'baseline_map_mse'):.3f}"
            ),
            (
                f"{METHOD_LABELS['gated']}\n"
                f"r={_as_float(row, 'gated_map_corr'):.3f} | "
                f"MSE={_as_float(row, 'gated_map_mse'):.3f} | alpha={_as_float(row, 'alpha'):.3f}"
            ),
            (
                f"{METHOD_LABELS['feems']}\n"
                f"r={_as_float(row, 'feems_corr'):.3f} | "
                f"MSE={_as_float(row, 'feems_mse'):.3f}"
            ),
        )
        for column, (value, title) in enumerate(zip(values, titles, strict=True)):
            _draw_edges(
                axes[row_index, column],
                arrays["positions"],
                arrays["edges"],
                value,
                arrays["sample_count"],
                limit=limit,
                title=title,
            )
        axes[row_index, 0].set_ylabel(
            f"{_as_int(row, 'num_sampled_nodes')}/{_as_int(row, 'num_nodes')} nodes\n{_as_int(row, 'num_snps'):,} SNPs",
            rotation=0,
            ha="right",
            va="center",
            fontsize=8.5,
            labelpad=9,
        )
    return figure


@beartype
def broad_aggregate_figure(rows: Sequence[Mapping[str, Any]]) -> Any:
    """Plot gate activity and paired broad H2H performance."""

    if not rows:
        raise ValueError("broad comparison requires at least one row")
    plt, _LineCollection, _TwoSlopeNorm, _matplotlib = _plot_imports()
    alpha = np.asarray([_as_float(row, "alpha") for row in rows])
    core_mse = np.asarray([_as_float(row, "baseline_map_mse") for row in rows])
    gated_mse = np.asarray([_as_float(row, "gated_map_mse") for row in rows])
    feems_mse = np.asarray([_as_float(row, "feems_mse") for row in rows])
    core_corr = np.asarray([_as_float(row, "baseline_map_corr") for row in rows])
    gated_corr = np.asarray([_as_float(row, "gated_map_corr") for row in rows])
    feems_corr = np.asarray([_as_float(row, "feems_corr") for row in rows])
    density = np.asarray([_as_float(row, "sampled_node_fraction") for row in rows])

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 9.5), constrained_layout=True)
    shrinkage = np.maximum(1.0 - alpha, 1e-8)
    axes[0, 0].hist(np.log10(shrinkage), bins=35, color="#278a67", alpha=0.85)
    axes[0, 0].set_xlabel("log10(1 - gate alpha)")
    axes[0, 0].set_ylabel("Cases")
    axes[0, 0].set_title(f"Gate activity on {len(rows)} structured cases")

    scatter = axes[0, 1].scatter(core_mse, gated_mse, c=alpha, cmap="viridis", s=14, alpha=0.7)
    positive = np.concatenate([core_mse[core_mse > 0], gated_mse[gated_mse > 0]])
    lower, upper = (float(positive.min()), float(positive.max())) if positive.size else (0.0, 1.0)
    axes[0, 1].plot([lower, upper], [lower, upper], color="#222222", linewidth=1)
    scale = "log" if positive.size else "linear"
    axes[0, 1].set(xscale=scale, yscale=scale, xlabel="Core DREEMS MSE", ylabel="Gated DREEMS MSE")
    axes[0, 1].set_aspect("equal", adjustable="box")
    figure.colorbar(scatter, ax=axes[0, 1], label="Gate alpha")

    axes[1, 0].scatter(
        feems_corr, core_corr, c=density, cmap="plasma", s=13, alpha=0.35, label="Core"
    )
    axes[1, 0].scatter(
        feems_corr,
        gated_corr,
        c=density,
        cmap="plasma",
        s=10,
        alpha=0.75,
        marker="x",
        label="Gated",
    )
    correlations = np.concatenate((feems_corr, core_corr, gated_corr))
    finite_corr = correlations[np.isfinite(correlations)]
    low = float(finite_corr.min()) if finite_corr.size else -1.0
    axes[1, 0].plot([low, 1.0], [low, 1.0], color="#222222", linewidth=1)
    axes[1, 0].set(xlabel="FEEMS LOO correlation", ylabel="DREEMS correlation")
    axes[1, 0].set_aspect("equal", adjustable="box")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].scatter(
        feems_mse, core_mse, c=density, cmap="plasma", s=13, alpha=0.35, label="Core"
    )
    axes[1, 1].scatter(
        feems_mse, gated_mse, c=density, cmap="plasma", s=10, alpha=0.75, marker="x", label="Gated"
    )
    positive = np.concatenate(
        [feems_mse[feems_mse > 0], core_mse[core_mse > 0], gated_mse[gated_mse > 0]]
    )
    lower, upper = (float(positive.min()), float(positive.max())) if positive.size else (0.0, 1.0)
    axes[1, 1].plot([lower, upper], [lower, upper], color="#222222", linewidth=1)
    scale = "log" if positive.size else "linear"
    axes[1, 1].set(xscale=scale, yscale=scale, xlabel="FEEMS LOO MSE", ylabel="DREEMS MSE")
    axes[1, 1].set_aspect("equal", adjustable="box")
    axes[1, 1].legend(frameon=False)
    return figure


@beartype
def render_broad_gallery(
    *,
    rows_path: Path,
    report_path: Path,
    core_pattern: str,
    gated_pattern: str,
    feems_pattern: str,
    core_checkpoint: Path,
    gated_checkpoint: Path,
    output_dir: Path,
) -> None:
    """Render a matched 512-case core, gate, and FEEMS gallery."""

    rows = read_rows(rows_path)
    if len(rows) != 512:
        raise ValueError(f"expected 512 broad rows, found {len(rows)}")
    rows_by_id = {_as_int(row): row for row in rows}
    selections = select_broad_examples(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = broad_aggregate_figure(rows)
    _save_figure(aggregate, output_dir / "aggregate.png")
    plt, _LineCollection, _TwoSlopeNorm, _matplotlib = _plot_imports()
    plt.close(aggregate)

    sections = ["<h2>Aggregate comparison</h2><img src='aggregate.png'>"]
    selection_payload: dict[str, list[int]] = {}
    for section_index, (label, examples) in enumerate(selections.items()):
        figure = _broad_examples_figure(
            rows_by_id,
            examples,
            core_pattern=core_pattern,
            gated_pattern=gated_pattern,
            feems_pattern=feems_pattern,
        )
        filename = f"examples_{section_index:02d}.png"
        _save_figure(figure, output_dir / filename)
        plt.close(figure)
        sections.append(f"<h2>{html.escape(label)}</h2><img src='{filename}'>")
        selection_payload[label] = list(examples)
    write_json(output_dir / "selected_examples.json", selection_payload)

    report = json.loads(report_path.read_text())
    overall = report["overall"]
    table_rows = [
        {
            "comparison": "Gated vs core",
            "candidate_corr": overall["gated_map_vs_baseline"]["corr"]["candidate_mean"],
            "baseline_corr": overall["gated_map_vs_baseline"]["corr"]["baseline_mean"],
            "corr_advantage": overall["gated_map_vs_baseline"]["corr"]["mean_advantage"],
            "candidate_mse": overall["gated_map_vs_baseline"]["mse"]["candidate_mean"],
            "baseline_mse": overall["gated_map_vs_baseline"]["mse"]["baseline_mean"],
            "mse_advantage": overall["gated_map_vs_baseline"]["mse"]["mean_advantage"],
        },
        {
            "comparison": "Gated vs FEEMS",
            "candidate_corr": overall["gated_map_vs_feems"]["corr"]["candidate_mean"],
            "baseline_corr": overall["gated_map_vs_feems"]["corr"]["baseline_mean"],
            "corr_advantage": overall["gated_map_vs_feems"]["corr"]["mean_advantage"],
            "candidate_mse": overall["gated_map_vs_feems"]["mse"]["candidate_mean"],
            "baseline_mse": overall["gated_map_vs_feems"]["mse"]["baseline_mean"],
            "mse_advantage": overall["gated_map_vs_feems"]["mse"]["mean_advantage"],
        },
    ]
    columns = (
        "comparison",
        "candidate_corr",
        "baseline_corr",
        "corr_advantage",
        "candidate_mse",
        "baseline_mse",
        "mse_advantage",
    )
    body = (
        f"<p>Core checkpoint: <code>{html.escape(sha256(core_checkpoint))}</code><br>"
        f"Gated checkpoint: <code>{html.escape(sha256(gated_checkpoint))}</code></p>"
        + _html_table(table_rows, columns)
        + "".join(sections)
    )
    intro = (
        "All 512 cases are the immutable structured global-landscape benchmark. Core and "
        "gated DREEMS use the same fixed MAP refinement; FEEMS uses the cached documented "
        "leave-one-observed-deme-out fit. Raw edge panels share one color scale per row."
    )
    with _atomic_output(output_dir / "index.html") as stream:
        stream.write(
            _gallery_html("Surface-gate comparison on the 512-case H2H", intro, body).encode(
                "utf-8"
            )
        )


@beartype
def build_parser() -> argparse.ArgumentParser:
    """Expose bundle assembly and the two cached-result galleries as separate tasks."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scenario_bundle = subparsers.add_parser("scenario-bundle")
    scenario_bundle.add_argument("--core-bundle", type=Path, required=True)
    scenario_bundle.add_argument("--gated-pattern", required=True)
    scenario_bundle.add_argument("--gated-checkpoint", type=Path, required=True)
    scenario_bundle.add_argument("--output", type=Path, required=True)
    scenario_bundle.add_argument("--rows-output", type=Path, required=True)
    scenario_bundle.add_argument("--report-output", type=Path, required=True)

    scenario_gallery = subparsers.add_parser("scenario-gallery")
    scenario_gallery.add_argument("--bundle", type=Path, required=True)
    scenario_gallery.add_argument("--report", type=Path, required=True)
    scenario_gallery.add_argument("--output-dir", type=Path, required=True)

    broad_gallery = subparsers.add_parser("broad-gallery")
    broad_gallery.add_argument("--rows", type=Path, required=True)
    broad_gallery.add_argument("--report", type=Path, required=True)
    broad_gallery.add_argument("--core-pattern", required=True)
    broad_gallery.add_argument("--gated-pattern", required=True)
    broad_gallery.add_argument("--feems-pattern", required=True)
    broad_gallery.add_argument("--core-checkpoint", type=Path, required=True)
    broad_gallery.add_argument("--gated-checkpoint", type=Path, required=True)
    broad_gallery.add_argument("--output-dir", type=Path, required=True)
    return parser


@beartype
def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch comparison tasks without running simulation or inference."""
    args = build_parser().parse_args(argv)
    if args.command == "scenario-bundle":
        build_scenario_bundle(
            core_bundle_path=args.core_bundle,
            gated_pattern=args.gated_pattern,
            gated_checkpoint=args.gated_checkpoint,
            output=args.output,
            rows_output=args.rows_output,
            report_output=args.report_output,
        )
    elif args.command == "scenario-gallery":
        render_scenario_gallery(
            bundle_path=args.bundle,
            report_path=args.report,
            output_dir=args.output_dir,
        )
    else:
        render_broad_gallery(
            rows_path=args.rows,
            report_path=args.report,
            core_pattern=args.core_pattern,
            gated_pattern=args.gated_pattern,
            feems_pattern=args.feems_pattern,
            core_checkpoint=args.core_checkpoint,
            gated_checkpoint=args.gated_checkpoint,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
