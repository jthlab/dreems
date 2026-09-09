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


def test_aggregate_rejects_predictions_from_another_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.zip"
    checkpoint.write_bytes(b"current checkpoint")
    baseline, neural = tmp_path / "feems.json", tmp_path / "dreems.json"
    _write_case(baseline, {"index": 0, "method": "feems"})
    _write_case(neural, {"index": 0, "method": "dreems", "checkpoint_sha256": "0" * 64})
    with pytest.raises(ValueError, match="requested checkpoint"):
        h2h.aggregate(Namespace(feems_json=[baseline], dreems_json=[neural], checkpoint=checkpoint))


@pytest.mark.parametrize("index", [None, True, False, -1, 0.5, "0"])
def test_read_rows_rejects_invalid_case_ids(tmp_path: Path, index: Any) -> None:
    path = tmp_path / "case.json"
    _write_case(path, {"method": "dreems", "index": index})
    with pytest.raises(ValueError, match="integer case ID"):
        h2h._read_rows([path], "dreems")


@pytest.mark.parametrize("digest", ["", "a" * 63, "z" * 64, "A" * 64, 123, []])
def test_read_rows_rejects_malformed_input_hashes(tmp_path: Path, digest: Any) -> None:
    path = tmp_path / "case.json"
    _write_case(path, {"method": "dreems", "index": 0, "input_sha256": digest})
    with pytest.raises(ValueError, match="SHA-256"):
        h2h._read_rows([path], "dreems")


@pytest.mark.parametrize("row", [[], None, {"method": "feems", "index": 0}])
def test_read_rows_rejects_wrong_report_type(tmp_path: Path, row: Any) -> None:
    path = tmp_path / "case.json"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="not a dreems result"):
        h2h._read_rows([path], "dreems")


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
    assert "feems_archive_checksum" in snakefile
    assert "dreems.benchmark_archive restore" in snakefile
    assert "rule extract_feems_archive:" in snakefile
    assert "localrules: all" in snakefile
    assert "rm -rf" not in snakefile


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
    args = Namespace(
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
    h2h.aggregate(args)

    summary = json.loads(output_json.read_text())
    protocol = json.loads(output_protocol.read_text())
    assert summary["examples"] == 2
    assert protocol["dreems_source_commit"] == "0123456789abcdef"
    assert protocol["feems"]["protocol"] == feems.FEEMS_RECOMMENDED_LOO_PROTOCOL
    assert protocol["feems"]["cv_folds"] == "one per observed deme"
    assert protocol["feems"]["lambda_grid"] == list(feems.FEEMS_RECOMMENDED_LAMBDAS)
    assert protocol["feems"]["cv_factr"] == 1e10
    assert protocol["feems"]["fold_execution"] == "snakemake_one_job_per_observed_deme"
    assert protocol["feems"]["result_source"] == "computed"
    assert protocol["feems"]["final_refit_factr"] == 1e7
    assert protocol["feems"]["final_refit_log_weight_bounds"] == "unbounded"
    assert summary["paired_comparisons"]["dreems_neural_corr_vs_feems"][
        "mean_advantage"
    ] == pytest.approx(0.15)
    assert output_csv.read_text().count("\n") == 3
    assert protocol["paired_input_identity"] == "not_recorded_for_all_cases"

    for path in feems_paths + dreems_paths:
        row = json.loads(path.read_text())
        row["input_sha256"] = "a" * 64
        _write_case(path, row)
    h2h.aggregate(args)
    assert json.loads(output_protocol.read_text())["paired_input_identity"] == "sha256"
    row["input_sha256"] = "b" * 64
    _write_case(dreems_paths[-1], row)
    with pytest.raises(ValueError, match="different input files"):
        h2h.aggregate(args)
    row["input_sha256"] = "a" * 64
    _write_case(dreems_paths[-1], row)
    args.feems_result_source = "verified_archive"
    with pytest.raises(ValueError, match="requires its verification report"):
        h2h.aggregate(args)
    args.feems_archive_verification = tmp_path / "verification.json"
    verification = {
        "examples": 2,
        "archive_sha256": "c" * 64,
        "input_identity": "legacy_unverified",
        "cases": [{"index": index, "current_input_sha256": "a" * 64} for index in range(2)],
    }
    args.feems_archive_verification.write_text(json.dumps(verification))
    with pytest.raises(ValueError, match="cannot be reported as verified"):
        h2h.aggregate(args)
    args.feems_result_source = "legacy_archive"
    h2h.aggregate(args)
    assert (
        json.loads(output_protocol.read_text())["feems"]["archive_verification"]["input_identity"]
        == "legacy_unverified"
    )
