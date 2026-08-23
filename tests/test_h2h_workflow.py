from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest
import yaml

from dreems import feems, h2h

REPO_ROOT = Path(__file__).resolve().parents[1]
H2H_CONFIG = REPO_ROOT / "workflow" / "config.h2h.yaml"
H2H_SNAKEFILE = REPO_ROOT / "workflow" / "h2h.Snakefile"


def _write_case(path: Path, row: dict[str, Any]) -> None:
    path.write_text(json.dumps(row) + "\n")


def test_h2h_config_exposes_resources_but_not_feems_statistics() -> None:
    config = yaml.safe_load(H2H_CONFIG.read_text())
    snakefile = H2H_SNAKEFILE.read_text()

    assert config["example_count"] == 512
    assert config["feems_fold_mem_mb"] == 4000
    assert config["archive_mem_mb"] == 1000
    assert "lambdas" not in config
    assert "lambdas_q" not in config
    assert "cv_mode" not in config
    assert "replicate_holdout" not in snakefile
    assert 'slurm_account="jonth0"' in snakefile
    assert "benchmark:" in snakefile
    assert "feems_official_parity" in snakefile
    assert "rule feems_cv_fold:" in snakefile
    assert "rule archive_feems:" in snakefile
    assert "feems_results.tar.gz.sha256" in snakefile
    assert "snakemake_one_job_per_observed_deme" in snakefile
    assert "--cv-checkpoint-dir" in snakefile


def test_h2h_aggregate_requires_and_records_documented_loo(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.dreems.zip"
    checkpoint.write_bytes(b"checkpoint")
    feems_paths: list[Path] = []
    dreems_paths: list[Path] = []
    for index, values in enumerate(((0.4, 0.8, 0.6, 0.5), (0.7, 0.6, 0.8, 0.4))):
        feems_corr, feems_mse, dreems_corr, dreems_mse = values
        common = {
            "index": index,
            "num_nodes": 10,
            "num_edges": 20,
            "num_samples": 12,
            "num_sampled_nodes": 6,
        }
        feems_path = tmp_path / f"feems_{index}.json"
        _write_case(
            feems_path,
            {
                **common,
                "method": "feems",
                "protocol": feems.FEEMS_RECOMMENDED_LOO_PROTOCOL,
                "elapsed_seconds": 3.0,
                "cv_jobs": 4,
                "cv_folds": 6,
                "cv_mode": "leave_one_out",
                "best_lambda": 1.0,
                "best_lambda_q": 0.1,
                "n_snps_after_filter": 100,
                "train_loss": 2.0,
                "feems_edge_log_corr": feems_corr,
                "feems_edge_log_mse": feems_mse,
            },
        )
        dreems_path = tmp_path / f"dreems_{index}.json"
        _write_case(
            dreems_path,
            {
                **common,
                "method": "dreems",
                "checkpoint": str(checkpoint),
                "elapsed_seconds": 1.0,
                "dreems_edge_log_corr": dreems_corr,
                "dreems_edge_log_mse": dreems_mse,
                "dreems_map_edge_log_corr": dreems_corr + 0.05,
                "dreems_map_edge_log_mse": dreems_mse - 0.05,
            },
        )
        feems_paths.append(feems_path)
        dreems_paths.append(dreems_path)

    output_json = tmp_path / "summary.json"
    output_csv = tmp_path / "rows.csv"
    output_protocol = tmp_path / "protocol.json"
    h2h.aggregate(
        Namespace(
            feems_json=feems_paths,
            dreems_json=dreems_paths,
            checkpoint=checkpoint,
            cv_execution="snakemake_one_job_per_observed_deme",
            source_commit="0123456789abcdef",
            input_pattern="inputs/example_{example}.pkl",
            metadata_pattern="inputs/example_{example}.metadata.json",
            bootstrap_seed=7,
            bootstrap_replicates=100,
            output_json=output_json,
            output_csv=output_csv,
            output_protocol=output_protocol,
        )
    )

    summary = json.loads(output_json.read_text())
    protocol = json.loads(output_protocol.read_text())
    assert summary["examples"] == 2
    assert protocol["dreems_source_commit"] == "0123456789abcdef"
    assert protocol["feems"]["protocol"] == feems.FEEMS_RECOMMENDED_LOO_PROTOCOL
    assert protocol["feems"]["cv_folds"] == "one per observed deme"
    assert protocol["feems"]["lambda_grid"] == list(feems.FEEMS_RECOMMENDED_LAMBDAS)
    assert protocol["feems"]["cv_factr"] == 1e10
    assert protocol["feems"]["fold_execution"] == "snakemake_one_job_per_observed_deme"
    assert protocol["feems"]["final_refit_factr"] == 1e7
    assert protocol["feems"]["final_refit_log_weight_bounds"] == "unbounded"
    assert summary["paired_comparisons"]["dreems_neural_corr_vs_feems"][
        "mean_advantage"
    ] == pytest.approx(0.15)
    assert output_csv.read_text().count("\n") == 3
