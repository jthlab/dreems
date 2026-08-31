from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image
from scipy import stats
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"


def source_path(variable: str, default: str) -> Path:
    return Path(os.environ.get(variable, default)).expanduser().resolve()


CANONICAL_REPO = source_path(
    "DREEMS_CANONICAL_REPO",
    str(REPO_ROOT),
)
CHECKPOINT_SOURCE = source_path(
    "DREEMS_CHECKPOINT",
    str(CANONICAL_REPO / "src/dreems/data/dreems_diffusion_bands.dreems.zip"),
)
TRAIN_ROOT = source_path(
    "DREEMS_TRAIN_ROOT",
    "/scratch/jonth_root/jonth1/jonth/dreems/graph_balanced_retrain_20260821",
)
RAW_TRAIN_METADATA = source_path(
    "DREEMS_RAW_TRAIN_METADATA",
    "/scratch/jonth_root/jonth1/jonth/dreems/clean_conditioned_broad_retrain_20260718_r3/raw/train",
)
H2H_PUBLISH = source_path(
    "DREEMS_H2H_PUBLISH",
    "/home/jonth/turbo/dreems/released_graph_balanced_loo_h2h_20260822",
)
H2H_WORK = source_path(
    "DREEMS_H2H_WORK",
    "/scratch/jonth_root/jonth1/jonth/dreems/released_graph_balanced_loo_h2h_20260822/work",
)
CALIBRATION_SOURCE = source_path(
    "DREEMS_CALIBRATION_REPORT",
    "/home/jonth/turbo/dreems/graph_balanced_calibration_20260821/verification/report.json",
)
REAL_GALLERY = source_path(
    "DREEMS_REAL_GALLERY",
    "/home/jonth/turbo/dreems/real_data_published_comparison_best_checkpoint_20260822/gallery",
)
WOLF_FEEMS_SUMMARY_SOURCE = source_path(
    "DREEMS_WOLF_FEEMS_SUMMARY",
    "/nfs/turbo/lsa-jonth/jonth/dreems/dreems_genetics_manuscript_20260725/data/wolf_feems_summary.json",
)
WOLF_FEEMS_ARRAY_SOURCE = source_path(
    "DREEMS_WOLF_FEEMS_ARRAY",
    "/nfs/turbo/lsa-jonth/jonth/dreems/dreems_genetics_manuscript_20260725/empirical/wolf_feems_example_parallel/feems_wolf_prediction.npz",
)
WOLF_FEEMS_METRICS_SOURCE = source_path(
    "DREEMS_WOLF_FEEMS_METRICS",
    "/nfs/turbo/lsa-jonth/jonth/dreems/dreems_genetics_manuscript_20260725/data/wolf_snp_cv_metrics.csv",
)
WOLF_DREEMS_JSON_SOURCE = source_path(
    "DREEMS_WOLF_RESULT_JSON",
    "/scratch/jonth_root/jonth1/jonth/dreems/real_data_best_checkpoint_gallery_20260822/real/fits/north_american_gray_wolf.json",
)
WOLF_DREEMS_ARRAY_SOURCE = WOLF_DREEMS_JSON_SOURCE.with_suffix(".npz")

EXPECTED_CHECKPOINT_SHA256 = "f0b677d2cc028252002d15fac843ca7347b1b352eb2286a18bd202b724aa92b4"

H2H_ROWS_SNAPSHOT = DATA / "h2h_summary.csv"
H2H_PROTOCOL_SNAPSHOT = DATA / "h2h_protocol.json"
H2H_RELEASE_SUMMARY_SNAPSHOT = DATA / "h2h_release_summary.json"
TRAINING_SUMMARY_SNAPSHOT = DATA / "training_summary.json"
TRAINING_HISTORY_SNAPSHOT = DATA / "training_history.jsonl"
TRAINING_BENCHMARK_SNAPSHOT = DATA / "training_benchmark.tsv"
CALIBRATION_SNAPSHOT = DATA / "calibration_verification.json"
REAL_MANIFEST_SNAPSHOT = DATA / "real_world_manifest.json"
WOLF_DREEMS_SNAPSHOT = DATA / "wolf_dreems_current.json"
WOLF_FEEMS_SNAPSHOT = DATA / "wolf_feems_summary.json"
CORPUS_SNAPSHOT = DATA / "training_corpus_summary.json"
H2H_GALLERY_SELECTION = DATA / "h2h_gallery_selection.json"

COLOR_DREEMS = "#0072B2"
COLOR_NEURAL = "#56B4E9"
COLOR_FEEMS = "#D55E00"
COLOR_REFINED = "#009E73"
COLOR_GRAY = "#6F6F6F"

EXPECTED_TABLES = (
    "model_setup.tex",
    "training_corpus.tex",
    "h2h_overall.tex",
    "h2h_paired.tex",
    "h2h_regimes.tex",
    "h2h_graph_sizes.tex",
    "runtime_overall.tex",
    "runtime_atomized.tex",
    "calibration.tex",
    "uncertainty_utility.tex",
    "empirical_datasets.tex",
    "wolf_empirical.tex",
)
EXPECTED_FIGURES = (
    "method_overview.pdf",
    "training_curve.pdf",
    "h2h_scatter_corr_mse.pdf",
    "h2h_regime_delta_bars.pdf",
    "h2h_examples_random.pdf",
    "h2h_examples_worst_mse.pdf",
    "runtime_scaling.pdf",
    "calibration_utility.pdf",
    "empirical_dreems_gallery.pdf",
    "empirical_published_comparison.pdf",
    "wolf_comparison.pdf",
)
STALE_GENERATED_OUTPUTS = (
    "figures/h2h_delta_vs_sampling_and_size.pdf",
    "figures/h2h_delta_vs_sampling_and_size.png",
    "figures/wolf_snp_cv_topo_comparison.png",
    "figures/wolf_snp_cv_trust_selection.png",
    "tables/resource_accounting.tex",
    "tables/runtime_by_size.tex",
    "tables/wolf_runtime.tex",
)

SIGNED_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "dreems_signed",
    ["#B2182B", "#FFFFFF", "#2166AC"],
    N=257,
)


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def copy_snapshot(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        destination.write_text(source.read_text())
        destination.chmod(0o644)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return text


def tabular(headers: list[str], rows: list[list[object]], *, align: str) -> str:
    lines = [f"\\begin{{tabular}}{{{align}}}", r"\toprule"]
    lines.append(" & ".join(latex_escape(header) for header in headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def fmt(value: float | int, digits: int = 3) -> str:
    if not np.isfinite(float(value)):
        return "--"
    if digits == 0:
        return f"{float(value):,.0f}"
    return f"{float(value):.{digits}f}"


def fmt_p(value: float) -> str:
    if value < 1.0e-99:
        return "<1e-99"
    if value < 1.0e-3:
        return f"{value:.1e}"
    return f"{value:.3f}"


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf")
    plt.close(fig)


def build_method_overview() -> None:
    fig, axis = plt.subplots(figsize=(7.2, 2.45))
    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    axis.axis("off")

    stages = (
        (
            "A",
            "Inputs",
            "Covariance\nSample locations\nGraph",
            "#E8F1F7",
            "#3977A3",
        ),
        (
            "B",
            "Graph\nevidence",
            "Deme covariance\nSample support\nResistance estimate",
            "#F2F2F2",
            "#666666",
        ),
        (
            "C",
            "Multiple\nscales",
            "Diffusion bands\nFine to broad\nsignals",
            "#F8EBD8",
            "#B46A1F",
        ),
        (
            "D",
            "Neural\nestimate",
            "Edge-aware GNN\nGlobal context\nPredictive SD",
            "#E5F2EA",
            "#32805A",
        ),
        (
            "E",
            "Refined\nsurface",
            "50 fixed updates\nCovariance fit\nTrust radius 0.5",
            "#F3E8EC",
            "#A04660",
        ),
    )
    lefts = np.linspace(0.015, 0.815, len(stages))
    width = 0.17
    bottom = 0.29
    height = 0.57
    for index, (left, stage) in enumerate(zip(lefts, stages, strict=True)):
        letter, title, body, fill, edge = stage
        axis.add_patch(
            Rectangle(
                (left, bottom),
                width,
                height,
                facecolor=fill,
                edgecolor=edge,
                linewidth=1.1,
            )
        )
        axis.text(
            left + 0.012,
            bottom + height - 0.055,
            letter,
            fontsize=8,
            color=edge,
            fontweight="bold",
            ha="left",
            va="center",
        )
        axis.text(
            left + width / 2,
            bottom + height - 0.17,
            title,
            fontsize=8.3,
            fontweight="bold",
            linespacing=1.05,
            ha="center",
            va="center",
        )
        axis.text(
            left + width / 2,
            bottom + 0.19,
            body,
            fontsize=7.2,
            linespacing=1.35,
            ha="center",
            va="center",
        )
        if index < len(stages) - 1:
            axis.add_patch(
                FancyArrowPatch(
                    (left + width + 0.006, bottom + height / 2),
                    (lefts[index + 1] - 0.006, bottom + height / 2),
                    arrowstyle="-|>",
                    mutation_scale=10,
                    color="#4A4A4A",
                    linewidth=0.9,
                )
            )

    axis.text(
        0.015,
        0.13,
        "No environmental covariates or FEEMS input",
        fontsize=7.0,
        color="#4A4A4A",
        ha="left",
    )
    axis.text(
        0.985,
        0.13,
        "One checkpoint and one refinement setting for every dataset",
        fontsize=7.0,
        color="#4A4A4A",
        ha="right",
    )
    save_figure(fig, "method_overview")


def load_checkpoint_metadata() -> dict[str, Any]:
    if sha256(CHECKPOINT_SOURCE) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("the source checkpoint does not match the frozen manuscript checkpoint")
    with zipfile.ZipFile(CHECKPOINT_SOURCE) as archive:
        return json.loads(archive.read("metadata.json"))


def snapshot_inputs() -> None:
    copy_snapshot(H2H_PUBLISH / "rows.csv", H2H_ROWS_SNAPSHOT)
    copy_snapshot(H2H_PUBLISH / "protocol.json", H2H_PROTOCOL_SNAPSHOT)
    copy_snapshot(H2H_PUBLISH / "summary.json", H2H_RELEASE_SUMMARY_SNAPSHOT)
    copy_snapshot(TRAIN_ROOT / "model/training_summary.json", TRAINING_SUMMARY_SNAPSHOT)
    copy_snapshot(TRAIN_ROOT / "model/training_history.jsonl", TRAINING_HISTORY_SNAPSHOT)
    copy_snapshot(TRAIN_ROOT / "benchmarks/train.tsv", TRAINING_BENCHMARK_SNAPSHOT)
    copy_snapshot(CALIBRATION_SOURCE, CALIBRATION_SNAPSHOT)
    copy_snapshot(REAL_GALLERY / "manifest.json", REAL_MANIFEST_SNAPSHOT)
    copy_snapshot(WOLF_DREEMS_JSON_SOURCE, WOLF_DREEMS_SNAPSHOT)
    copy_snapshot(WOLF_FEEMS_SUMMARY_SOURCE, WOLF_FEEMS_SNAPSHOT)


def _metadata_files() -> list[Path]:
    files = sorted(RAW_TRAIN_METADATA.glob("*.metadata.json"))
    if not files:
        files = sorted(RAW_TRAIN_METADATA.glob("*.json"))
    return files


def _numeric_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def refresh_corpus_summary() -> dict[str, Any]:
    files = _metadata_files()
    if len(files) != 61_440:
        raise ValueError(f"expected 61,440 training metadata files, found {len(files)}")
    categorical_keys = (
        "graph_source",
        "island_marker_model",
        "mask_kind",
        "sample_design",
        "sample_regime",
        "simulator",
        "surface_kind",
    )
    numeric_keys = (
        "effective_snps",
        "island_attempts",
        "local_log_rate_std",
        "local_q_log_std",
        "mean_diploids_per_sampled_node",
        "num_edges",
        "num_nodes",
        "num_sampled_nodes",
        "num_samples",
        "num_snps",
        "sampled_node_fraction",
    )
    categories = {key: Counter() for key in categorical_keys}
    numbers = {key: [] for key in numeric_keys}
    sample_covariance = 0
    rust = 0
    for path in files:
        row = json.loads(path.read_text())
        for key in categorical_keys:
            if row.get(key) is not None:
                categories[key][str(row[key])] += 1
        for key in numeric_keys:
            if row.get(key) is not None:
                numbers[key].append(float(row[key]))
        sample_covariance += int(bool(row.get("sample_covariance", True)))
        rust += int(str(row.get("simulator", "")) == "island_coalescent")
    summary = {
        "examples": len(files),
        "source_directory": str(RAW_TRAIN_METADATA),
        "categorical_counts": {
            key: dict(sorted(value.items())) for key, value in categories.items()
        },
        "numeric_distributions": {key: _numeric_summary(value) for key, value in numbers.items()},
        "sample_covariance_fraction": sample_covariance / len(files),
        "rust_simulator_fraction": rust / len(files),
        "genotype_matrix_fraction": 0.0,
    }
    write_json(CORPUS_SNAPSHOT, summary)
    return summary


def load_corpus_summary(*, refresh: bool) -> dict[str, Any]:
    if refresh:
        return refresh_corpus_summary()
    if not CORPUS_SNAPSHOT.is_file():
        raise FileNotFoundError(f"{CORPUS_SNAPSHOT} is missing; rerun with --refresh-corpus")
    return json.loads(CORPUS_SNAPSHOT.read_text())


def paired_effect(
    frame: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    lower_is_better: bool,
) -> dict[str, float | int]:
    difference = (
        frame[baseline] - frame[candidate]
        if lower_is_better
        else frame[candidate] - frame[baseline]
    ).to_numpy(dtype=np.float64)
    difference = difference[np.isfinite(difference)]
    sem = float(stats.sem(difference))
    low, high = stats.t.interval(
        0.95,
        len(difference) - 1,
        loc=float(np.mean(difference)),
        scale=sem,
    )
    return {
        "n": int(len(difference)),
        "mean_advantage": float(np.mean(difference)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "p": float(stats.ttest_1samp(difference, 0.0).pvalue),
        "wins": int(np.sum(difference > 0.0)),
        "losses": int(np.sum(difference < 0.0)),
        "win_fraction": float(np.mean(difference > 0.0)),
    }


def summarize_group(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "n": int(len(frame)),
        "dreems_corr": float(frame["dreems_map_edge_log_corr"].mean()),
        "feems_corr": float(frame["feems_edge_log_corr"].mean()),
        "dreems_mse": float(frame["dreems_map_edge_log_mse"].mean()),
        "feems_mse": float(frame["feems_edge_log_mse"].mean()),
        "corr_effect": paired_effect(
            frame,
            candidate="dreems_map_edge_log_corr",
            baseline="feems_edge_log_corr",
            lower_is_better=False,
        ),
        "mse_effect": paired_effect(
            frame,
            candidate="dreems_map_edge_log_mse",
            baseline="feems_edge_log_mse",
            lower_is_better=True,
        ),
    }


def summarize_h2h(frame: pd.DataFrame) -> dict[str, Any]:
    if len(frame) != 512:
        raise ValueError(f"expected 512 H2H rows, found {len(frame)}")
    if not np.all(frame["cv_mode"] == "leave_one_out"):
        raise ValueError("the release H2H is not uniformly FEEMS leave-one-out CV")
    methods = {
        "neural": {
            "corr": float(frame["dreems_edge_log_corr"].mean()),
            "mse": float(frame["dreems_edge_log_mse"].mean()),
            "cov_corr": float(frame["covariance_initial_corr"].mean()),
            "cov_r2": float(frame["covariance_initial_r2"].mean()),
            "cov_mse": float(frame["covariance_initial_mse"].mean()),
        },
        "refined": {
            "corr": float(frame["dreems_map_edge_log_corr"].mean()),
            "mse": float(frame["dreems_map_edge_log_mse"].mean()),
            "cov_corr": float(frame["covariance_refined_corr"].mean()),
            "cov_r2": float(frame["covariance_refined_r2"].mean()),
            "cov_mse": float(frame["covariance_refined_mse"].mean()),
        },
        "feems": {
            "corr": float(frame["feems_edge_log_corr"].mean()),
            "mse": float(frame["feems_edge_log_mse"].mean()),
        },
    }
    comparisons = {
        "refined_vs_feems_corr": paired_effect(
            frame,
            candidate="dreems_map_edge_log_corr",
            baseline="feems_edge_log_corr",
            lower_is_better=False,
        ),
        "refined_vs_feems_mse": paired_effect(
            frame,
            candidate="dreems_map_edge_log_mse",
            baseline="feems_edge_log_mse",
            lower_is_better=True,
        ),
        "neural_vs_feems_corr": paired_effect(
            frame,
            candidate="dreems_edge_log_corr",
            baseline="feems_edge_log_corr",
            lower_is_better=False,
        ),
        "neural_vs_feems_mse": paired_effect(
            frame,
            candidate="dreems_edge_log_mse",
            baseline="feems_edge_log_mse",
            lower_is_better=True,
        ),
        "refined_vs_neural_corr": paired_effect(
            frame,
            candidate="dreems_map_edge_log_corr",
            baseline="dreems_edge_log_corr",
            lower_is_better=False,
        ),
        "refined_vs_neural_mse": paired_effect(
            frame,
            candidate="dreems_map_edge_log_mse",
            baseline="dreems_edge_log_mse",
            lower_is_better=True,
        ),
    }
    sampling_groups = [
        summarize_group(frame[frame["sampled_node_fraction"] <= 0.10], "Sparse <=10%"),
        summarize_group(
            frame[
                (frame["sampled_node_fraction"] > 0.10) & (frame["sampled_node_fraction"] <= 0.25)
            ],
            "Intermediate 10-25%",
        ),
        summarize_group(frame[frame["sampled_node_fraction"] > 0.25], "Dense >25%"),
    ]
    size_groups = [
        summarize_group(frame[frame["num_nodes"] <= 450], "219-450"),
        summarize_group(
            frame[(frame["num_nodes"] > 450) & (frame["num_nodes"] <= 900)],
            "451-900",
        ),
        summarize_group(frame[frame["num_nodes"] > 900], "901-1,406"),
    ]
    ranges = {}
    for key in (
        "num_nodes",
        "num_edges",
        "num_sampled_nodes",
        "sampled_node_fraction",
        "num_samples",
        "num_snps",
    ):
        values = frame[key]
        ranges[key] = {
            "min": float(values.min()),
            "median": float(values.median()),
            "max": float(values.max()),
            "mean": float(values.mean()),
        }
    return {
        "examples": len(frame),
        "methods": methods,
        "comparisons": comparisons,
        "sampling_groups": sampling_groups,
        "size_groups": size_groups,
        "ranges": ranges,
    }


def summarize_uncertainty(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    pooled: dict[str, list[np.ndarray]] = {
        "truth": [],
        "neural": [],
        "refined": [],
        "sd": [],
    }
    per_graph: list[dict[str, float | int]] = []
    z_values = {"50": 0.67448975, "80": 1.28155157, "95": 1.95996398}
    for index in frame["index"].astype(int):
        path = H2H_WORK / f"dreems/example_{index}.npz"
        with np.load(path) as arrays:
            values = {
                "truth": np.asarray(arrays["truth"], dtype=np.float64),
                "neural": np.asarray(arrays["dreems"], dtype=np.float64),
                "refined": np.asarray(arrays["dreems_map"], dtype=np.float64),
                "sd": np.asarray(arrays["edge_sd"], dtype=np.float64),
            }
        for key, array in values.items():
            pooled[key].append(array)
        refined_error = np.abs(values["refined"] - values["truth"])
        record: dict[str, float | int] = {
            "index": int(index),
            "mean_sd": float(np.mean(values["sd"])),
            "refined_mse": float(np.mean(refined_error**2)),
            "within_spearman": float(stats.spearmanr(values["sd"], refined_error).statistic),
        }
        for label, z_value in z_values.items():
            record[f"neural_{label}"] = float(
                np.mean(np.abs(values["neural"] - values["truth"]) <= z_value * values["sd"])
            )
            record[f"refined_{label}"] = float(np.mean(refined_error <= z_value * values["sd"]))
        per_graph.append(record)

    joined = {key: np.concatenate(value) for key, value in pooled.items()}
    graph_frame = pd.DataFrame(per_graph)
    squared_error = (joined["refined"] - joined["truth"]) ** 2
    absolute_error = np.sqrt(squared_error)
    sd = joined["sd"]
    quantiles = np.quantile(sd, np.linspace(0.0, 1.0, 6))
    quintile_mse = []
    for index in range(5):
        upper = sd <= quantiles[index + 1] if index == 4 else sd < quantiles[index + 1]
        mask = (sd >= quantiles[index]) & upper
        quintile_mse.append(float(np.mean(squared_error[mask])))
    order = np.argsort(sd)
    selective_mse = []
    for fraction in (0.2, 0.4, 0.6, 0.8, 1.0):
        stop = math.ceil(fraction * len(order))
        selective_mse.append(
            {"fraction": fraction, "mse": float(np.mean(squared_error[order[:stop]]))}
        )
    summary = {
        "examples": int(len(graph_frame)),
        "edges": int(len(sd)),
        "graph_balanced_coverage": {
            estimate: {
                label: float(graph_frame[f"{estimate}_{label}"].mean()) for label in z_values
            }
            for estimate in ("neural", "refined")
        },
        "pooled_coverage": {
            estimate: {
                label: float(
                    np.mean(np.abs(joined[estimate] - joined["truth"]) <= z_value * joined["sd"])
                )
                for label, z_value in z_values.items()
            }
            for estimate in ("neural", "refined")
        },
        "pooled_sd_absolute_error_spearman": float(stats.spearmanr(sd, absolute_error).statistic),
        "mean_within_graph_spearman": float(graph_frame["within_spearman"].mean()),
        "positive_within_graph_fraction": float(np.mean(graph_frame["within_spearman"] > 0.0)),
        "mean_sd_graph_mse_spearman": float(
            stats.spearmanr(graph_frame["mean_sd"], graph_frame["refined_mse"]).statistic
        ),
        "mse_by_sd_quintile": quintile_mse,
        "selective_mse": selective_mse,
    }
    write_json(DATA / "calibration_summary.json", summary)
    graph_frame.to_csv(DATA / "calibration_examples.csv", index=False)
    return summary, graph_frame


def summarize_runtime(frame: pd.DataFrame) -> dict[str, Any]:
    process_pool = frame[frame["feems_cv_jobs"] > 1].copy()
    if len(process_pool) != 511:
        raise ValueError(f"expected 511 process-pool FEEMS rows, found {len(process_pool)}")
    ratio = process_pool["feems_elapsed_seconds"] / process_pool["elapsed_seconds"]

    atomized_dir = H2H_WORK / "benchmarks/feems_cv/example_295"
    fold_files = sorted(atomized_dir.glob("fold_*.tsv"))
    if len(fold_files) != 384:
        raise ValueError(f"expected 384 atomized FEEMS folds, found {len(fold_files)}")
    folds = pd.concat([pd.read_csv(path, sep="\t") for path in fold_files], ignore_index=True)
    final = pd.read_csv(H2H_WORK / "benchmarks/feems/example_295.tsv", sep="\t").iloc[-1]
    atomized = {
        "example": 295,
        "folds": len(fold_files),
        "fold_wall_sum_seconds": float(folds["s"].sum()),
        "fold_wall_max_seconds": float(folds["s"].max()),
        "fold_cpu_sum_seconds": float(folds["cpu_time"].sum()),
        "final_wall_seconds": float(final["s"]),
        "final_cpu_seconds": float(final["cpu_time"]),
        "critical_path_compute_seconds": float(folds["s"].max() + final["s"]),
        "aggregate_cpu_seconds": float(folds["cpu_time"].sum() + final["cpu_time"]),
        "peak_rss_mb": float(max(folds["max_rss"].max(), final["max_rss"])),
    }
    training = pd.read_csv(TRAINING_BENCHMARK_SNAPSHOT, sep="\t").iloc[-1]
    summary = {
        "process_pool_cases": len(process_pool),
        "dreems_workers": 2,
        "feems_workers": 16,
        "dreems_elapsed_median_seconds": float(process_pool["elapsed_seconds"].median()),
        "dreems_elapsed_p95_seconds": float(process_pool["elapsed_seconds"].quantile(0.95)),
        "dreems_prediction_median_seconds": float(process_pool["prediction_seconds"].median()),
        "feems_elapsed_median_seconds": float(process_pool["feems_elapsed_seconds"].median()),
        "feems_elapsed_p95_seconds": float(process_pool["feems_elapsed_seconds"].quantile(0.95)),
        "paired_speedup_median": float(ratio.median()),
        "paired_speedup_geometric_mean": float(stats.gmean(ratio)),
        "atomized_example": atomized,
        "training": {
            "wall_seconds": float(training["s"]),
            "cpu_seconds": float(training["cpu_time"]),
            "max_rss_mb": float(training["max_rss"]),
        },
        "note": (
            "The 511 FEEMS elapsed times include LOO CV in a process pool and the final fit. "
            "Example 295 used one Snakemake job per fold and is reported separately."
        ),
    }
    write_json(DATA / "resource_accounting.json", summary)
    process_pool[
        [
            "index",
            "num_nodes",
            "num_edges",
            "elapsed_seconds",
            "prediction_seconds",
            "feems_elapsed_seconds",
            "feems_cv_jobs",
        ]
    ].to_csv(DATA / "runtime_examples.csv", index=False)
    return summary


def centered(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values - float(np.mean(values))


def summarize_wolf() -> dict[str, Any]:
    current = json.loads(WOLF_DREEMS_SNAPSHOT.read_text())
    feems_summary = json.loads(WOLF_FEEMS_SNAPSHOT.read_text())
    with np.load(WOLF_DREEMS_ARRAY_SOURCE) as arrays:
        neural = centered(arrays["prior_edge_log"])
        refined = centered(arrays["edge_log"])
    with np.load(WOLF_FEEMS_ARRAY_SOURCE) as arrays:
        positions = np.asarray(arrays["node_coordinates"], dtype=np.float64)
        edges = np.asarray(arrays["edges"], dtype=np.int64)
        feems = centered(arrays["edge_log_rates"])
    midpoints = 0.5 * (positions[edges[:, 0]] + positions[edges[:, 1]])
    hudson = (
        (midpoints[:, 0] >= -96.0)
        & (midpoints[:, 0] <= -74.0)
        & (midpoints[:, 1] >= 50.0)
        & (midpoints[:, 1] <= 66.0)
    )
    legacy = pd.read_csv(WOLF_FEEMS_METRICS_SOURCE)
    feems_metrics = legacy.loc[legacy["method"] == "feems"].iloc[0]
    initial_fit = current["diagnostics"]["initial_fit"]
    refined_fit = current["diagnostics"]["refined_fit"]
    methods = {
        "neural": {
            "cov_corr": float(initial_fit["corr"]),
            "cov_r2": float(initial_fit["r2"]),
            "cov_mse": float(initial_fit["mse"]),
            "edge_corr_vs_feems": float(stats.pearsonr(neural, feems).statistic),
            "edge_sd": float(np.std(neural)),
            "hudson_mean": float(np.mean(neural[hudson])),
        },
        "refined": {
            "cov_corr": float(refined_fit["corr"]),
            "cov_r2": float(refined_fit["r2"]),
            "cov_mse": float(refined_fit["mse"]),
            "edge_corr_vs_feems": float(stats.pearsonr(refined, feems).statistic),
            "edge_sd": float(np.std(refined)),
            "hudson_mean": float(np.mean(refined[hudson])),
        },
        "feems": {
            "cov_corr": float(feems_metrics["covariance_corr"]),
            "cov_r2": float(feems_metrics["covariance_r2"]),
            "cov_mse": float(feems_metrics["covariance_mse"]),
            "edge_corr_vs_feems": 1.0,
            "edge_sd": float(np.std(feems)),
            "hudson_mean": float(np.mean(feems[hudson])),
        },
    }
    summary = {
        "samples": 111,
        "snps": int(feems_summary["n_snps_after_filter"]),
        "nodes": int(feems_summary["nodes"]),
        "edges": int(feems_summary["edges"]),
        "observed_nodes": int(feems_summary["n_observed_nodes"]),
        "hudson_edges": int(np.sum(hudson)),
        "dreems_seconds": float(current["prediction_seconds"]),
        "feems_seconds": float(feems_summary["seconds"]),
        "feems_cv_folds": int(feems_summary["cv_folds"]),
        "methods": methods,
    }
    write_json(DATA / "wolf_current_summary.json", summary)
    return summary


def write_model_tables(checkpoint: dict[str, Any], corpus: dict[str, Any]) -> None:
    model = checkpoint["model"]
    architecture = model["architecture"]
    conditioner = model["edge_conditioner"]
    training = checkpoint["training_run"]
    rows = [
        ["Node / edge dimensions", f"{model['node_dim']} / {model['edge_dim']}"],
        ["Hidden channels", architecture["hidden_channels"]],
        ["Local / context graph layers", "10 / 3"],
        ["Graph-wide attention", "yes"],
        ["Covariance score maps", model["features"]["covariance_components"]],
        ["Diffusion times", "0.25, 0.5, 1, 2, 4"],
        ["Spatial representation", "topology and normalized edge length"],
        ["Coordinate-frame invariant", "translation, rotation, reflection, scale"],
        ["Linearized initializer", conditioner["method"]],
        ["Initializer pairs", "all observed-deme pairs"],
        ["Predictive family", model["posterior"]["kind"]],
        ["Training examples", f"{corpus['examples']:,}"],
        ["Validation / held-out examples", "1,024 / 4,096"],
        ["Best / final step", f"{training['best_step']:,} / {training['final_step']:,}"],
        ["Best validation loss", fmt(training["best_validation_loss"], 4)],
        ["Optimizer / learning rate / batch", "Adam / 5e-5 / 8"],
    ]
    write_text(TABLES / "model_setup.tex", tabular(["Component", "Setting"], rows, align="ll"))

    counts = corpus["categorical_counts"]["graph_source"]
    distributions = corpus["numeric_distributions"]

    def range_text(key: str, digits: int = 0) -> str:
        values = distributions[key]
        return (
            f"{fmt(values['min'], digits)}-{fmt(values['max'], digits)} "
            f"(median {fmt(values['median'], digits)})"
        )

    rows = [
        ["Total", f"{corpus['examples']:,}"],
        ["Global landmass graphs", f"{counts['global_landscape']:,}"],
        ["FEEMS triangular lattices", f"{counts['feems_triangular_lattice']:,}"],
        ["Wolf-like graphs", f"{counts['wolflike']:,}"],
        ["Masked grids and earth-like graphs", f"{counts['grid'] + counts['earth_like']:,}"],
        ["Nodes", range_text("num_nodes")],
        ["Edges", range_text("num_edges")],
        ["Observed-node fraction", range_text("sampled_node_fraction", 3)],
        ["Diploid samples", range_text("num_samples")],
        ["Accepted SNPs", range_text("num_snps")],
        ["Edge log-rate SD", range_text("local_log_rate_std", 3)],
        ["Node log-q SD", range_text("local_q_log_std", 3)],
        ["Simulator", "Rust structured coalescent"],
        ["Ascertainment", "random marker conditional on MAF >=0.05"],
    ]
    write_text(
        TABLES / "training_corpus.tex", tabular(["Corpus property", "Value"], rows, align="ll")
    )


def write_h2h_tables(summary: dict[str, Any]) -> None:
    methods = summary["methods"]
    rows = [
        [
            "DREEMS neural",
            fmt(methods["neural"]["corr"]),
            fmt(methods["neural"]["mse"]),
            fmt(methods["neural"]["cov_corr"]),
            fmt(methods["neural"]["cov_r2"]),
        ],
        [
            "DREEMS refined",
            fmt(methods["refined"]["corr"]),
            fmt(methods["refined"]["mse"]),
            fmt(methods["refined"]["cov_corr"]),
            fmt(methods["refined"]["cov_r2"]),
        ],
        [
            "FEEMS documented LOO-CV",
            fmt(methods["feems"]["corr"]),
            fmt(methods["feems"]["mse"]),
            "--",
            "--",
        ],
    ]
    write_text(
        TABLES / "h2h_overall.tex",
        tabular(["Method", "Edge corr.", "Edge MSE", "Cov. corr.", "Cov. R2"], rows, align="lrrrr"),
    )

    labels = {
        "refined_vs_feems_corr": "Refined DREEMS vs FEEMS, correlation",
        "refined_vs_feems_mse": "Refined DREEMS vs FEEMS, MSE",
        "neural_vs_feems_corr": "Neural DREEMS vs FEEMS, correlation",
        "neural_vs_feems_mse": "Neural DREEMS vs FEEMS, MSE",
        "refined_vs_neural_corr": "Refinement vs neural, correlation",
        "refined_vs_neural_mse": "Refinement vs neural, MSE",
    }
    rows = []
    for key, label in labels.items():
        effect = summary["comparisons"][key]
        rows.append(
            [
                label,
                effect["n"],
                fmt(effect["mean_advantage"], 4),
                f"[{fmt(effect['ci95_low'], 4)}, {fmt(effect['ci95_high'], 4)}]",
                fmt_p(effect["p"]),
                fmt(effect["win_fraction"]),
            ]
        )
    write_text(
        TABLES / "h2h_paired.tex",
        tabular(
            ["Paired contrast", "N", "Mean advantage", "95% CI", "P", "Win frac."],
            rows,
            align="lrrrrr",
        ),
    )

    rows = []
    for group in summary["sampling_groups"]:
        rows.append(
            [
                group["label"],
                group["n"],
                fmt(group["dreems_corr"]),
                fmt(group["feems_corr"]),
                fmt(group["corr_effect"]["mean_advantage"], 4),
                fmt(group["dreems_mse"]),
                fmt(group["feems_mse"]),
                fmt(group["mse_effect"]["mean_advantage"], 4),
            ]
        )
    write_text(
        TABLES / "h2h_regimes.tex",
        tabular(
            ["Sampling", "N", "D corr.", "F corr.", "Corr. adv.", "D MSE", "F MSE", "MSE adv."],
            rows,
            align="lrrrrrrr",
        ),
    )

    rows = []
    for group in summary["size_groups"]:
        rows.append(
            [
                group["label"],
                group["n"],
                fmt(group["dreems_corr"]),
                fmt(group["feems_corr"]),
                fmt(group["corr_effect"]["mean_advantage"], 4),
                fmt(group["dreems_mse"]),
                fmt(group["feems_mse"]),
                fmt(group["mse_effect"]["mean_advantage"], 4),
            ]
        )
    write_text(
        TABLES / "h2h_graph_sizes.tex",
        tabular(
            ["Nodes", "N", "D corr.", "F corr.", "Corr. adv.", "D MSE", "F MSE", "MSE adv."],
            rows,
            align="lrrrrrrr",
        ),
    )


def write_runtime_tables(summary: dict[str, Any]) -> None:
    rows = [
        [
            "DREEMS, complete refined estimate",
            511,
            "2 threads",
            fmt(summary["dreems_elapsed_median_seconds"], 1),
            fmt(summary["dreems_elapsed_p95_seconds"], 1),
        ],
        [
            "FEEMS LOO-CV and final fit",
            511,
            "up to 16 workers",
            fmt(summary["feems_elapsed_median_seconds"], 1),
            fmt(summary["feems_elapsed_p95_seconds"], 1),
        ],
    ]
    write_text(
        TABLES / "runtime_overall.tex",
        tabular(["Execution", "N", "Parallelism", "Median s", "P95 s"], rows, align="lrlrr"),
    )
    atomized = summary["atomized_example"]
    rows = [
        [
            "FEEMS LOO folds",
            atomized["folds"],
            fmt(atomized["fold_wall_max_seconds"], 1),
            fmt(atomized["fold_wall_sum_seconds"], 1),
            fmt(atomized["fold_cpu_sum_seconds"], 1),
        ],
        [
            "FEEMS selected final fit",
            1,
            fmt(atomized["final_wall_seconds"], 1),
            fmt(atomized["final_wall_seconds"], 1),
            fmt(atomized["final_cpu_seconds"], 1),
        ],
        [
            "Combined",
            "--",
            fmt(atomized["critical_path_compute_seconds"], 1),
            fmt(atomized["fold_wall_sum_seconds"] + atomized["final_wall_seconds"], 1),
            fmt(atomized["aggregate_cpu_seconds"], 1),
        ],
    ]
    write_text(
        TABLES / "runtime_atomized.tex",
        tabular(["Stage", "Jobs", "Critical path s", "Worker-s", "CPU-s"], rows, align="lrrrr"),
    )


def write_calibration_tables(calibration: dict[str, Any], uncertainty: dict[str, Any]) -> None:
    selected = {row["sampling"]: row for row in calibration["rows"] if row["method"] == "selected"}
    rows = []
    for label in ("all", "sparse <=10%", "medium 10-25%", "dense >25%"):
        row = selected[label]
        rows.append(
            [
                f"Independent verification: {label}",
                row["graphs"],
                fmt(row["graph_balanced_coverage_50"]),
                fmt(row["graph_balanced_coverage_80"]),
                fmt(row["graph_balanced_coverage_95"]),
            ]
        )
    for estimate, label in (
        ("neural", "Landscape H2H: neural"),
        ("refined", "Landscape H2H: refined mean, neural SD"),
    ):
        coverage = uncertainty["graph_balanced_coverage"][estimate]
        rows.append([label, 512, fmt(coverage["50"]), fmt(coverage["80"]), fmt(coverage["95"])])
    write_text(
        TABLES / "calibration.tex",
        tabular(["Evaluation", "Graphs", "50%", "80%", "95%"], rows, align="lrrrr"),
    )
    rows = [
        [
            "Pooled SD vs absolute refined error, Spearman",
            fmt(uncertainty["pooled_sd_absolute_error_spearman"]),
        ],
        ["Mean within-graph Spearman", fmt(uncertainty["mean_within_graph_spearman"])],
        [
            "Graphs with positive within-graph association",
            fmt(uncertainty["positive_within_graph_fraction"]),
        ],
        ["Mean SD vs graph MSE, Spearman", fmt(uncertainty["mean_sd_graph_mse_spearman"])],
        [
            "Lowest-SD / highest-SD quintile MSE",
            "{} / {}".format(
                fmt(uncertainty["mse_by_sd_quintile"][0]),
                fmt(uncertainty["mse_by_sd_quintile"][-1]),
            ),
        ],
    ]
    write_text(
        TABLES / "uncertainty_utility.tex", tabular(["Diagnostic", "Value"], rows, align="lr")
    )


def short_dataset_name(record: dict[str, Any]) -> str:
    return str(record.get("short_name") or record["dataset_id"]).replace("_", " ")


def write_empirical_table(records: list[dict[str, Any]]) -> None:
    rows = []
    for record in records:
        rows.append(
            [
                short_dataset_name(record),
                record["samples"],
                record["effective_snps"],
                record["nodes"],
                record["observed_nodes"],
                record["edges"],
                fmt(record["dreems_seconds"], 1),
            ]
        )
    write_text(
        TABLES / "empirical_datasets.tex",
        tabular(
            ["Dataset", "Samples", "SNPs", "Nodes", "Obs.", "Edges", "Seconds"],
            rows,
            align="lrrrrrr",
        ),
    )


def write_wolf_table(summary: dict[str, Any]) -> None:
    rows = []
    labels = {
        "neural": "DREEMS neural",
        "refined": "DREEMS refined",
        "feems": "FEEMS 94-fold LOO-CV",
    }
    for key in ("neural", "refined", "feems"):
        row = summary["methods"][key]
        rows.append(
            [
                labels[key],
                fmt(row["cov_corr"]),
                fmt(row["cov_r2"]),
                fmt(row["cov_mse"], 6),
                fmt(row["edge_corr_vs_feems"]),
                fmt(row["edge_sd"]),
                fmt(row["hudson_mean"]),
            ]
        )
    write_text(
        TABLES / "wolf_empirical.tex",
        tabular(
            [
                "Method",
                "Cov. corr.",
                "Cov. R2",
                "Cov. MSE",
                "Edge corr. vs F",
                "Edge SD",
                "Hudson mean",
            ],
            rows,
            align="lrrrrrr",
        ),
    )


def build_training_curve(training: dict[str, Any]) -> None:
    history = [
        json.loads(line)
        for line in TRAINING_HISTORY_SNAPSHOT.read_text().splitlines()
        if line.strip()
    ]
    frame = pd.DataFrame(history)
    validation = frame.dropna(subset=["validation_loss"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].plot(
        validation["step"], validation["validation_loss"], color=COLOR_DREEMS, linewidth=1.3
    )
    axes[0].scatter(
        [training["best_step"]],
        [training["best_validation_loss"]],
        color=COLOR_REFINED,
        s=25,
        zorder=3,
        label="restored checkpoint",
    )
    axes[0].axvline(
        int(validation["step"].max()),
        color=COLOR_GRAY,
        linestyle="--",
        linewidth=1,
        label="early stop",
    )
    axes[0].set(
        xlabel="Optimization step", ylabel="Graph-balanced validation loss", title="Validation"
    )
    axes[0].legend(frameon=False)
    axes[1].plot(frame["step"], frame["loss"], color=COLOR_NEURAL, linewidth=1.0)
    axes[1].set(
        xlabel="Optimization step", ylabel="Training minibatch loss", title="Training batches"
    )
    axes[1].axvline(training["best_step"], color=COLOR_REFINED, linestyle=":", linewidth=1)
    fig.tight_layout()
    save_figure(fig, "training_curve")


def build_runtime_figure(frame: pd.DataFrame) -> None:
    frame = frame[frame["feems_cv_jobs"] > 1].copy()
    frame["speedup"] = frame["feems_elapsed_seconds"] / frame["elapsed_seconds"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].scatter(
        frame["num_nodes"],
        frame["elapsed_seconds"],
        s=9,
        alpha=0.35,
        color=COLOR_DREEMS,
        label="DREEMS, 2 threads",
    )
    axes[0].scatter(
        frame["num_nodes"],
        frame["feems_elapsed_seconds"],
        s=9,
        alpha=0.25,
        color=COLOR_FEEMS,
        label="FEEMS, <=16 workers",
    )
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Graph nodes", ylabel="Elapsed seconds", title="End-to-end inference")
    axes[0].legend(frameon=False)
    axes[1].scatter(frame["num_nodes"], frame["speedup"], s=9, alpha=0.3, color=COLOR_REFINED)
    axes[1].axhline(1.0, color=COLOR_GRAY, linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set(
        xlabel="Graph nodes",
        ylabel="FEEMS / DREEMS elapsed time",
        title="Paired elapsed-time ratio",
    )
    fig.tight_layout()
    save_figure(fig, "runtime_scaling")


def build_calibration_figure(
    calibration: dict[str, Any],
    uncertainty: dict[str, Any],
) -> None:
    selected = {row["sampling"]: row for row in calibration["rows"] if row["method"] == "selected"}
    nominal = np.asarray([0.5, 0.8, 0.95])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    for label, color in (
        ("all", COLOR_DREEMS),
        ("sparse <=10%", COLOR_FEEMS),
        ("dense >25%", COLOR_REFINED),
    ):
        row = selected[label]
        values = [row[f"graph_balanced_coverage_{level}"] for level in (50, 80, 95)]
        axes[0, 0].plot(nominal, values, marker="o", label=label, color=color)
    axes[0, 0].plot([0.45, 1.0], [0.45, 1.0], linestyle="--", color=COLOR_GRAY, linewidth=1)
    axes[0, 0].set(
        xlabel="Nominal coverage",
        ylabel="Observed coverage",
        title="Independent calibration verification",
    )
    axes[0, 0].legend(frameon=False)

    for estimate, label, color in (
        ("neural", "neural mean", COLOR_NEURAL),
        ("refined", "refined mean", COLOR_REFINED),
    ):
        values = [
            uncertainty["graph_balanced_coverage"][estimate][str(level)] for level in (50, 80, 95)
        ]
        axes[0, 1].plot(nominal, values, marker="o", label=label, color=color)
    axes[0, 1].plot([0.45, 1.0], [0.45, 1.0], linestyle="--", color=COLOR_GRAY, linewidth=1)
    axes[0, 1].set(
        xlabel="Nominal coverage", ylabel="Observed coverage", title="Global-landscape benchmark"
    )
    axes[0, 1].legend(frameon=False)

    axes[1, 0].bar(np.arange(1, 6), uncertainty["mse_by_sd_quintile"], color=COLOR_DREEMS)
    axes[1, 0].set(
        xlabel="Predictive-SD quintile",
        ylabel="Pooled refined MSE",
        title="Error rises with predictive SD",
    )
    selective = uncertainty["selective_mse"]
    axes[1, 1].plot(
        [100 * row["fraction"] for row in selective],
        [row["mse"] for row in selective],
        marker="o",
        color=COLOR_REFINED,
    )
    axes[1, 1].set(
        xlabel="Lowest-SD edges retained (%)",
        ylabel="Pooled refined MSE",
        title="Selective prediction",
    )
    fig.tight_layout()
    save_figure(fig, "calibration_utility")


def _crop_dreems_contour(path: Path) -> Image.Image:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    # The source gallery is a fixed four-panel diagnostic sheet. Keep only the
    # third map, excluding its legacy title and the adjacent color bars.
    panel = rgb[
        int(height * 0.335) : int(height * 0.67),
        int(width * 0.509) : int(width * 0.745),
    ]
    return Image.fromarray(panel)


def _crop_published_wolf(path: Path) -> Image.Image:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    return Image.fromarray(
        rgb[
            int(height * 0.335) : int(height * 0.67),
            int(width * 0.515) : int(width * 0.995),
        ]
    )


def build_empirical_figures(records: list[dict[str, Any]]) -> None:
    by_id = {record["dataset_id"]: record for record in records}
    selected_ids = [
        "north_american_gray_wolf",
        "blacklegged_ticks_midwest",
        "allohistium_darters",
        "sulphur_crested_cockatoo",
        "queensland_fruit_fly",
        "eastern_red_backed_salamander",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.8))
    for axis, dataset_id in zip(axes.flat, selected_ids, strict=True):
        record = by_id[dataset_id]
        axis.imshow(_crop_dreems_contour(REAL_GALLERY / record["figure_png"]))
        axis.set_title(
            f"{short_dataset_name(record)}: {record['samples']} samples, "
            f"{record['observed_nodes']} demes",
            fontsize=8.5,
        )
        axis.axis("off")
    fig.tight_layout(pad=0.7)
    save_figure(fig, "empirical_dreems_gallery")

    fig, axes = plt.subplots(len(selected_ids), 2, figsize=(7.2, 11.5))
    for row_index, dataset_id in enumerate(selected_ids):
        record = by_id[dataset_id]
        axes[row_index, 0].imshow(_crop_dreems_contour(REAL_GALLERY / record["figure_png"]))
        axes[row_index, 0].set_title(f"{short_dataset_name(record)}: DREEMS refined", fontsize=8.5)
        axes[row_index, 0].axis("off")
        published = record["published_figures"][0]
        with Image.open(REAL_GALLERY / published["file"]) as image:
            axes[row_index, 1].imshow(image.convert("RGB"))
        axes[row_index, 1].set_title("Published FEEMS/EEMS or study figure", fontsize=8.5)
        axes[row_index, 1].axis("off")
    fig.tight_layout(pad=0.6)
    save_figure(fig, "empirical_published_comparison")

    wolf = by_id["north_american_gray_wolf"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    axes[0].imshow(_crop_dreems_contour(REAL_GALLERY / wolf["figure_png"]))
    axes[0].set_title("DREEMS refined contours")
    axes[0].axis("off")
    axes[1].imshow(_crop_published_wolf(REAL_GALLERY / wolf["published_figures"][0]["file"]))
    axes[1].set_title("FEEMS documented LOO-CV contours")
    axes[1].axis("off")
    fig.tight_layout(pad=0.4)
    save_figure(fig, "wolf_comparison")


def build_h2h_scatter(frame: pd.DataFrame) -> None:
    density = frame["sampled_node_fraction"].to_numpy(dtype=np.float64)
    norm = mcolors.LogNorm(vmin=max(float(np.min(density)), 0.003), vmax=0.5)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)
    scatter = axes[0].scatter(
        frame["feems_edge_log_corr"],
        frame["dreems_map_edge_log_corr"],
        c=density,
        norm=norm,
        cmap="viridis",
        s=17,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.25,
    )
    corr_low = float(
        min(frame["feems_edge_log_corr"].min(), frame["dreems_map_edge_log_corr"].min())
    )
    corr_high = float(
        max(frame["feems_edge_log_corr"].max(), frame["dreems_map_edge_log_corr"].max())
    )
    axes[0].plot([corr_low, corr_high], [corr_low, corr_high], "--", color=COLOR_GRAY, lw=0.8)
    axes[0].set(
        xlabel="FEEMS edge correlation",
        ylabel="Refined DREEMS edge correlation",
        title="Correlation",
        xlim=(corr_low - 0.02, corr_high + 0.02),
        ylim=(corr_low - 0.02, corr_high + 0.02),
    )

    axes[1].scatter(
        frame["feems_edge_log_mse"],
        frame["dreems_map_edge_log_mse"],
        c=density,
        norm=norm,
        cmap="viridis",
        s=17,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.25,
    )
    mse_high = float(max(frame["feems_edge_log_mse"].max(), frame["dreems_map_edge_log_mse"].max()))
    axes[1].plot([0.0, mse_high], [0.0, mse_high], "--", color=COLOR_GRAY, lw=0.8)
    axes[1].set(
        xlabel="FEEMS edge MSE",
        ylabel="Refined DREEMS edge MSE",
        title="Mean squared error",
        xlim=(-0.03, mse_high * 1.02),
        ylim=(-0.03, mse_high * 1.02),
    )
    colorbar = fig.colorbar(scatter, ax=axes, pad=0.02, fraction=0.045)
    colorbar.set_label("Observed-node fraction")
    save_figure(fig, "h2h_scatter_corr_mse")


def _effect_error(effect: dict[str, Any]) -> tuple[float, float]:
    mean = float(effect["mean_advantage"])
    return mean - float(effect["ci95_low"]), float(effect["ci95_high"]) - mean


def build_h2h_regime_bars(summary: dict[str, Any]) -> None:
    groups = (
        ("Sampling density", summary["sampling_groups"]),
        ("Graph nodes", summary["size_groups"]),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    for row, (group_title, records) in enumerate(groups):
        labels = [f"{record['label']} (n={record['n']})" for record in records]
        y = np.arange(len(records))
        for column, (key, title) in enumerate(
            (("corr_effect", "Correlation advantage"), ("mse_effect", "MSE reduction"))
        ):
            means = np.asarray(
                [float(record[key]["mean_advantage"]) for record in records], dtype=np.float64
            )
            errors = np.asarray([_effect_error(record[key]) for record in records]).T
            axis = axes[row, column]
            axis.barh(y, means, color=COLOR_REFINED, alpha=0.88, xerr=errors, capsize=2)
            axis.axvline(0.0, color=COLOR_GRAY, linestyle="--", linewidth=0.8)
            axis.set_yticks(y, labels if column == 0 else ["" for _ in labels])
            axis.invert_yaxis()
            axis.set_xlabel("Mean paired advantage")
            axis.set_title(f"{group_title}: {title}")
    save_figure(fig, "h2h_regime_delta_bars")


def _edge_midpoints(positions: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return 0.5 * (positions[edges[:, 0]] + positions[edges[:, 1]])


def _median_edge_length(positions: np.ndarray, edges: np.ndarray) -> float:
    lengths = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    lengths = lengths[np.isfinite(lengths) & (lengths > 0.0)]
    return float(np.median(lengths)) if lengths.size else 1.0


def _interpolate_edge_field(
    positions: np.ndarray,
    edges: np.ndarray,
    values: np.ndarray,
    *,
    grid_side: int = 150,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    xy = _edge_midpoints(positions, edges)[finite]
    z = values[finite]
    xmin, ymin = np.min(positions, axis=0)
    xmax, ymax = np.max(positions, axis=0)
    pad = 0.035 * max(float(xmax - xmin), float(ymax - ymin), 1e-6)
    extent = (float(xmin - pad), float(xmax + pad), float(ymin - pad), float(ymax + pad))
    if not z.size:
        return np.full((grid_side, grid_side), np.nan), extent
    xx, yy = np.meshgrid(
        np.linspace(extent[0], extent[1], grid_side),
        np.linspace(extent[2], extent[3], grid_side),
    )
    tree = cKDTree(xy)
    neighbors = min(32, xy.shape[0])
    distances, indices = tree.query(np.column_stack([xx.ravel(), yy.ravel()]), k=neighbors)
    if neighbors == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    edge_length = _median_edge_length(positions, edges)
    bandwidth = max(1.8 * edge_length, 1e-12)
    maximum = 4.0 * edge_length
    weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
    weights[distances > maximum] = 0.0
    denominator = np.sum(weights, axis=1)
    image = np.full(denominator.shape, np.nan, dtype=np.float64)
    valid = denominator > 1e-12
    image[valid] = np.sum(weights[valid] * z[indices[valid]], axis=1) / denominator[valid]
    image[distances[:, 0] > maximum] = np.nan
    return image.reshape(xx.shape), extent


def _plot_h2h_field(
    axis: plt.Axes,
    *,
    positions: np.ndarray,
    edges: np.ndarray,
    sample_count: np.ndarray,
    values: np.ndarray,
    limit: float,
    title: str,
) -> None:
    interpolated, extent = _interpolate_edge_field(positions, edges, values)
    axis.imshow(
        np.ma.masked_invalid(interpolated),
        origin="lower",
        extent=extent,
        cmap=SIGNED_CMAP,
        vmin=-limit,
        vmax=limit,
        interpolation="bilinear",
    )
    axis.add_collection(
        LineCollection(
            positions[edges],
            colors="#202020",
            linewidths=0.12,
            alpha=0.07,
            zorder=2,
        )
    )
    sampled = np.asarray(sample_count) > 0
    if np.any(sampled):
        maximum = max(float(np.max(sample_count[sampled])), 1.0)
        sizes = 5.0 + 12.0 * np.sqrt(sample_count[sampled] / maximum)
        axis.scatter(
            positions[sampled, 0],
            positions[sampled, 1],
            s=sizes,
            facecolor="white",
            edgecolor="#202020",
            linewidth=0.3,
            zorder=3,
        )
    axis.set(xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]))
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=7.4)


def _load_h2h_arrays(index: int) -> dict[str, np.ndarray]:
    with np.load(H2H_WORK / "dreems" / f"example_{index}.npz", allow_pickle=False) as dreems:
        arrays = {name: np.asarray(dreems[name]) for name in dreems.files}
    with np.load(H2H_WORK / "feems" / f"example_{index}.npz", allow_pickle=False) as feems:
        if not np.array_equal(arrays["edges"], feems["edges"]):
            raise ValueError(f"DREEMS and FEEMS edges differ for example {index}")
        arrays["feems"] = np.asarray(feems["feems"])
    return arrays


def _build_h2h_example_gallery(rows: pd.DataFrame, stem: str) -> None:
    fig, axes = plt.subplots(len(rows), 3, figsize=(7.2, 2.25 * len(rows)), squeeze=False)
    for local_row, (_, row) in enumerate(rows.iterrows()):
        index = int(row["index"])
        arrays = _load_h2h_arrays(index)
        fields = [centered(arrays[name]) for name in ("truth", "dreems_map", "feems")]
        values = np.concatenate(fields)
        limit = max(float(np.quantile(np.abs(values[np.isfinite(values)]), 0.98)), 1e-6)
        header = (
            f"Example {index}: {int(row['num_nodes'])} nodes, "
            f"{int(row['num_sampled_nodes'])} observed, {int(row['num_samples'])} samples, "
            f"{int(row['num_snps'])} SNPs"
        )
        titles = (
            f"{header}\nTruth (scale +/-{limit:.2f})",
            f"Refined DREEMS: r={row['dreems_map_edge_log_corr']:.2f}, "
            f"MSE={row['dreems_map_edge_log_mse']:.2f}",
            f"FEEMS LOO-CV: r={row['feems_edge_log_corr']:.2f}, "
            f"MSE={row['feems_edge_log_mse']:.2f}",
        )
        for column, (field, title) in enumerate(zip(fields, titles, strict=True)):
            _plot_h2h_field(
                axes[local_row, column],
                positions=arrays["positions"],
                edges=arrays["edges"].astype(np.int64),
                sample_count=arrays["sample_count"],
                values=field,
                limit=limit,
                title=title,
            )
    fig.tight_layout(pad=0.55)
    save_figure(fig, stem)


def build_h2h_figures(frame: pd.DataFrame, summary: dict[str, Any]) -> None:
    build_h2h_scatter(frame)
    build_h2h_regime_bars(summary)
    random_rows = frame.sample(n=4, random_state=20_260_822).sort_values("index")
    worst_rows = (
        frame.assign(
            dreems_mse_disadvantage=(frame["dreems_map_edge_log_mse"] - frame["feems_edge_log_mse"])
        )
        .nlargest(4, "dreems_mse_disadvantage")
        .sort_values("dreems_mse_disadvantage", ascending=False)
    )
    write_json(
        H2H_GALLERY_SELECTION,
        {
            "random_seed": 20_260_822,
            "random_examples": [int(value) for value in random_rows["index"]],
            "worst_mse_definition": "dreems_refined_mse_minus_feems_mse",
            "worst_mse_examples": [int(value) for value in worst_rows["index"]],
        },
    )
    _build_h2h_example_gallery(random_rows, "h2h_examples_random")
    _build_h2h_example_gallery(worst_rows, "h2h_examples_worst_mse")


def release_manifest() -> None:
    artifacts = [
        ROOT / "README.md",
        ROOT / "Makefile",
        ROOT / "main.tex",
        ROOT / "methods.tex",
        ROOT / "references.bib",
        ROOT / "build_assets.py",
        ROOT / "check_manuscript.py",
        H2H_ROWS_SNAPSHOT,
        H2H_PROTOCOL_SNAPSHOT,
        H2H_RELEASE_SUMMARY_SNAPSHOT,
        H2H_GALLERY_SELECTION,
        TRAINING_SUMMARY_SNAPSHOT,
        TRAINING_HISTORY_SNAPSHOT,
        TRAINING_BENCHMARK_SNAPSHOT,
        CORPUS_SNAPSHOT,
        CALIBRATION_SNAPSHOT,
        REAL_MANIFEST_SNAPSHOT,
        WOLF_DREEMS_SNAPSHOT,
        WOLF_FEEMS_SNAPSHOT,
        DATA / "runtime_examples.csv",
        DATA / "resource_accounting.json",
        DATA / "calibration_examples.csv",
        DATA / "calibration_summary.json",
        DATA / "wolf_current_summary.json",
        ROOT / "analysis_summary.json",
        ROOT / "asset_summary.json",
    ]
    artifacts.extend(TABLES / name for name in EXPECTED_TABLES)
    artifacts.extend(FIGURES / name for name in EXPECTED_FIGURES)
    if (ROOT / "main.pdf").is_file():
        artifacts.append(ROOT / "main.pdf")
    records = []
    for path in artifacts:
        if not path.is_file():
            continue
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(REPO_ROOT)
        except ValueError as error:
            raise ValueError(f"manifest artifact is outside the repository: {resolved}") from error
        records.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256(resolved),
                "bytes": resolved.stat().st_size,
            }
        )
    write_json(
        ROOT / "release_manifest.json",
        {
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "h2h_examples": 512,
            "feems_protocol": "v2_joint_lambda_lambda_q_leave_one_observed_deme_out",
            "empirical_datasets": 15,
            "records": records,
        },
    )


def build_assets(*, refresh_corpus: bool) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    for path in FIGURES.glob("*.png"):
        path.unlink()
    for relative in STALE_GENERATED_OUTPUTS:
        (ROOT / relative).unlink(missing_ok=True)
    checkpoint = load_checkpoint_metadata()
    snapshot_inputs()
    protocol = json.loads(H2H_PROTOCOL_SNAPSHOT.read_text())
    if protocol["dreems"]["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("H2H checkpoint does not match the manuscript checkpoint")
    frame = pd.read_csv(H2H_ROWS_SNAPSHOT)
    corpus = load_corpus_summary(refresh=refresh_corpus)
    training = json.loads(TRAINING_SUMMARY_SNAPSHOT.read_text())
    calibration = json.loads(CALIBRATION_SNAPSHOT.read_text())
    real_document = json.loads(REAL_MANIFEST_SNAPSHOT.read_text())
    real_records = list(real_document["datasets"])
    if len(real_records) != 15:
        raise ValueError(f"expected 15 real-data records, found {len(real_records)}")

    h2h = summarize_h2h(frame)
    uncertainty, _calibration_examples = summarize_uncertainty(frame)
    runtime = summarize_runtime(frame)
    wolf = summarize_wolf()

    write_model_tables(checkpoint, corpus)
    write_h2h_tables(h2h)
    write_runtime_tables(runtime)
    write_calibration_tables(calibration, uncertainty)
    write_empirical_table(real_records)
    write_wolf_table(wolf)
    build_method_overview()
    build_training_curve(training)
    build_runtime_figure(frame)
    build_calibration_figure(calibration, uncertainty)
    build_empirical_figures(real_records)
    build_h2h_figures(frame, h2h)

    analysis = {
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "training": {
            "examples": corpus["examples"],
            "best_step": training["best_step"],
            "final_step": 41_000,
            "best_validation_loss": training["best_validation_loss"],
        },
        "h2h": h2h,
        "calibration": {
            "verification": calibration,
            "global_landscape": uncertainty,
        },
        "runtime": runtime,
        "wolf": wolf,
        "empirical_dataset_count": len(real_records),
    }
    write_json(ROOT / "analysis_summary.json", analysis)
    write_json(
        ROOT / "asset_summary.json",
        {
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "h2h": {"examples": 512, "feems_protocol": protocol["feems"]["protocol"]},
            "calibration": {"verification_examples": calibration["verification_examples"]},
            "empirical_dataset_count": len(real_records),
            "source_commit": protocol["dreems_source_commit"],
        },
    )
    release_manifest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the DREEMS manuscript assets.")
    parser.add_argument(
        "--refresh-corpus",
        action="store_true",
        help="Re-read all 61,440 raw training metadata records.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only refresh release_manifest.json after LaTeX builds main.pdf.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_plots()
    if args.manifest_only:
        release_manifest()
        return
    build_assets(refresh_corpus=bool(args.refresh_corpus))


if __name__ == "__main__":
    main()
