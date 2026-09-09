"""Run the training DAG on a tiny corpus, including the frozen-core stage."""

from __future__ import annotations

import copy
import csv
import io
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dreems import benchmark_archive, benchmark_data, checkpoints, feems, workflow_io

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_workflow(
    directory: Path,
    config: dict[str, Any],
    *options: str,
    snakefile: str = "Snakefile",
    target: str = "all",
) -> str:
    """Use an isolated local executor, even when pytest itself runs under Slurm."""
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    environment = dict(os.environ)
    environment.pop("SLURM_JOB_ID", None)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    environment["TMPDIR"] = str(directory / "temporary")
    Path(environment["TMPDIR"]).mkdir(exist_ok=True)
    result = subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "--no-sync",
            "--project",
            str(REPO_ROOT),
            "snakemake",
            target,
            "--snakefile",
            str(REPO_ROOT / "workflow" / snakefile),
            "--configfile",
            str(config_path),
            "--profile",
            str(REPO_ROOT / "workflow/profiles/local"),
            "--cores",
            "2",
            "--local-cores",
            "2",
            "--resources",
            "mem_mb=8000",
            "gpu=0",
            *options,
        ],
        cwd=directory,
        env=environment,
        text=True,
        capture_output=True,
        timeout=1200,
    )
    log = result.stdout + result.stderr
    (directory / f"run_{len(list(directory.glob('run_*.log')))}.log").write_text(log)
    assert result.returncode == 0, log[-15000:]
    assert not list(Path(environment["TMPDIR"]).glob("dreems-train.*"))
    return result.stdout if "--summary" in options else log


def test_local_training_and_staged_gate_reuse(tmp_path: Path) -> None:
    """Simulation and preparation stay reusable when only training changes."""
    config = yaml.safe_load((REPO_ROOT / "workflow/config.example.yaml").read_text())
    run_dir = tmp_path / "joint"
    output_dir = run_dir / "outputs"
    # Spaces and colons are valid path characters, including in published files.
    publish_dir = run_dir / "published: model"
    config.update(
        outdir=str(output_dir),
        publish_dir=str(publish_dir),
        train_examples=4,
        test_examples=2,
        train_validation_examples=2,
        raw_pack_size=2,
        prepare_batch_size=2,
        train_pack_size=2,
        validation_pack_size=2,
        normalization_group_size=2,
        train_threads=2,
        eval_threads=1,
        eval_batch_size=2,
    )
    config["simulation"].update(
        case_weights={"feems_lattice": 1.0},
        global_landscape_cartopy_data_dir=str(tmp_path / "cartopy"),
        num_snps=16,
        min_num_snps=16,
        max_num_snps=16,
        feems_lattice_rows=4,
        feems_lattice_columns=4,
        feems_lattice_surface_kinds=["feems_heterogeneous_barrier"],
        feems_lattice_surface_weights={"feems_heterogeneous_barrier": 1.0},
        feems_lattice_barrier_startpt=1.0,
        feems_lattice_barrier_endpt=2.0,
        feems_lattice_sample_prob_choices=[1.0],
        feems_lattice_sample_prob_weights={"1.0": 1.0},
    )
    # Exercise seed inheritance through the actual worker config, not just the
    # Python config reader. This used to disappear when Snakemake made a snapshot.
    config["seed"] = config["simulation"].pop("seed")
    config["dreems"]["architecture"].update(
        hidden_channels=8,
        layers=1,
        context_layers=1,
        initializer_gate=True,
        surface_gate=True,
    )
    config["dreems"]["features"].update(
        covariance_components=2,
        spatial_encoding="topology_node_covariance",
    )
    config["dreems"]["optimizer"].update(
        steps=2,
        batch_size=2,
        log_every=1,
        early_stopping_warmup_steps=0,
    )
    run_workflow(run_dir, config)
    [simulation_snapshot] = list((output_dir / "config").glob("simulation.*.json"))
    recorded_simulation = json.loads(simulation_snapshot.read_text())
    assert recorded_simulation["simulation"]["seed"] == config["seed"]
    for metadata in (output_dir / "raw").rglob("*.metadata.json"):
        assert json.loads(metadata.read_text())["base_seed"] == config["seed"]
    assert (output_dir / "eval/summary.json").is_file()
    for name in (
        "checkpoint.best.dreems.zip",
        "checkpoint.best.summary.json",
        "training_summary.json",
        "training_history.jsonl",
    ):
        assert (publish_dir / name).read_bytes() == (output_dir / "model" / name).read_bytes()
    assert "Nothing to be done" in run_workflow(run_dir, config, "--dry-run")

    explicit_seed = copy.deepcopy(config)
    explicit_seed["simulation"]["seed"] = config["seed"]
    explicit_seed["seed"] = config["seed"] + 1
    # Equivalent effective seeds must retain the same config hash and raw packs.
    assert "Nothing to be done" in run_workflow(run_dir, explicit_seed, "--dry-run")

    changed = copy.deepcopy(config)
    changed["dreems"]["optimizer"]["learning_rate"] = 1e-5
    summary = run_workflow(run_dir, changed, "--dry-run", "--summary")
    # Snakemake prints a progress line before the TSV header on stdout.
    table = summary[summary.index("output_file\t") :]
    planned = [
        row
        for row in csv.DictReader(io.StringIO(table), delimiter="\t")
        if row["plan"] == "update pending"
    ]
    assert any(row["rule"] == "train" for row in planned), summary
    assert not any(
        row["rule"]
        in {
            "generate_raw_example",
            "pack_raw_examples",
            "prepare_batch",
            "merge_normalization",
            "normalization_group_stats",
        }
        for row in planned
    ), summary
    assert all(
        row["status"] == "removed temp file" for row in planned if row["rule"] == "prepare_chunk"
    ), summary

    sampling = yaml.safe_load((REPO_ROOT / "workflow/config.sampling_robustness.yaml").read_text())
    sampling_dir = tmp_path / "sampling"
    sampling_output = sampling_dir / "outputs"
    sampling.update(
        outdir=str(sampling_output),
        source_corpus=str(output_dir),
        base_checkpoint=str(tmp_path / "does-not-exist.dreems.zip"),
        dreems=copy.deepcopy(config["dreems"]),
    )
    sampling["counterfactual_augmentation"].update(
        train_base_examples=1,
        validation_base_examples=1,
        train_replay_packs=1,
        validation_replay_packs=1,
        groups_per_pack=1,
    )
    log = run_workflow(
        sampling_dir,
        sampling,
        snakefile="sampling_robustness.Snakefile",
        target="prepared",
    )
    assert "rule train_gate:" not in log
    assert "rule train_finetune:" not in log
    assert not (sampling_output / "model").exists()
    sampling_features = json.loads((sampling_output / "config/features.json").read_text())
    assert set(sampling_features["dreems"]) == {"features", "edge_conditioner"}
    for kind in ("counterfactual", "replay"):
        for split in ("train", "validation"):
            assert (sampling_output / "prepared" / kind / split / "pack_0.pkl").is_file()

    staged_dir = tmp_path / "staged"
    staged_output = staged_dir / "outputs"
    staged = copy.deepcopy(config)
    staged["dreems"]["loss"]["surface_gate_weight"] = 1.0
    staged.update(
        outdir=str(staged_output),
        publish_dir=str(staged_dir / "published: model"),
        prepared_pack_source_dir=str(output_dir / "prepared_batches"),
        raw_source_dir=str(output_dir / "raw"),
        normalization_source=str(output_dir / "normalization.json"),
    )
    staged["surface_gate_stage"] = {
        "enabled": True,
        "validation_examples": 2,
        "prepared_packs": {
            "train": {
                "pattern": str(output_dir / "prepared_batches/train/pack_{pack}.pkl"),
                "count": 2,
            },
            "validation": {
                "pattern": str(output_dir / "prepared_batches/validation/pack_{pack}.pkl"),
                "count": 1,
            },
        },
        "optimizer": {"steps": 2, "log_every": 1, "learning_rate": 5e-4},
    }
    log = run_workflow(staged_dir, staged)
    assert "rule train_core:" in log
    assert "rule train_surface_gate:" in log
    core_path = staged_output / "model/core/checkpoint.best.dreems.zip"
    final_path = staged_output / "model/checkpoint.best.dreems.zip"
    core = checkpoints.load_checkpoint(core_path)
    final = checkpoints.load_checkpoint(final_path)
    assert core.params.surface_gate_hidden is None
    assert final.params.surface_gate_hidden is not None
    # Gate training adds a head but must leave all existing parameter leaves alone.
    import jax
    import numpy as np

    for name in core.params.__dataclass_fields__:
        if name.startswith("surface_gate"):
            continue
        before = jax.tree_util.tree_leaves(getattr(core.params, name))
        after = jax.tree_util.tree_leaves(getattr(final.params, name))
        assert len(before) == len(after), name
        for left, right in zip(before, after, strict=True):
            np.testing.assert_array_equal(left, right, err_msg=name)
    for stage in ("", "core"):
        published = Path(staged["publish_dir"]) / stage
        for name in ("checkpoint.best.dreems.zip", "training_history.jsonl"):
            assert (published / name).read_bytes() == (
                staged_output / "model" / stage / name
            ).read_bytes()
        history = [
            json.loads(line)
            for line in (published / "training_history.jsonl").read_text().splitlines()
        ]
        assert len(history) >= 2
    assert "Nothing to be done" in run_workflow(staged_dir, staged, "--dry-run")


def test_fresh_release_dag_builds_views_core_and_gate(tmp_path: Path) -> None:
    """One invocation builds the complete training graph without an older checkpoint."""
    config = yaml.safe_load((REPO_ROOT / "workflow/config.example.yaml").read_text())
    output = tmp_path / "output"
    config.update(
        outdir=str(output),
        publish_dir=str(tmp_path / "published"),
        train_examples=4,
        test_examples=2,
        base_validation_examples=2,
        train_validation_examples=2,
        raw_pack_size=2,
        prepare_batch_size=2,
        train_pack_size=2,
        validation_pack_size=2,
        normalization_group_size=2,
        train_threads=2,
        eval_threads=1,
        eval_batch_size=2,
        core_validation_mask_packs=1,
    )
    config["simulation"].update(
        case_weights={
            "grid": 0.0,
            "earth_like": 0.0,
            "wolflike": 0.0,
            "global_landscape": 0.0,
            "feems_lattice": 1.0,
        },
        global_landscape_cartopy_data_dir=str(tmp_path / "cartopy"),
        num_snps_log_bins=None,
        num_snps=16,
        min_num_snps=16,
        max_num_snps=16,
        feems_lattice_rows=4,
        feems_lattice_columns=4,
        feems_lattice_surface_kinds=["feems_heterogeneous_barrier"],
        feems_lattice_barrier_startpt=1.0,
        feems_lattice_barrier_endpt=2.0,
        feems_lattice_sample_prob_choices=[1.0],
        feems_lattice_n_samples_per_node=2,
    )
    config["dreems"]["architecture"].update(
        hidden_channels=8,
        layers=1,
        context_layers=1,
        initializer_gate=True,
        surface_gate=True,
    )
    config["dreems"]["features"].update(
        covariance_components=2,
        spatial_encoding="topology_node_covariance",
    )
    config["dreems"]["loss"].update(surface_gate_weight=1.0)
    config["dreems"]["optimizer"].update(
        steps=2,
        batch_size=4,
        log_every=1,
        early_stopping_warmup_steps=0,
    )
    config["surface_gate_stage"] = {
        "optimizer": {
            "steps": 2,
            "batch_size": 4,
            "log_every": 1,
            "early_stopping_warmup_steps": 0,
        },
    }
    config["sampling_views"] = {
        "counterfactual_augmentation": {
            "train_base_examples": 1,
            "validation_base_examples": 1,
            "train_replay_packs": 1,
            "validation_replay_packs": 1,
            "groups_per_pack": 1,
        },
    }
    # Cached FEEMS numbers below are test doubles, not accuracy measurements.
    # Neural inference, calibration, rendering, and promotion run for real.
    config["release_validation"] = {}
    for label, indices in (("broad", range(6)), ("hard", range(6, 9))):
        inputs = tmp_path / "benchmark inputs" / label
        cached = inputs / "saved_feems"
        inputs.mkdir(parents=True)
        source_paths = {}
        for index in indices:
            positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
            edges = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int32)
            truth = np.array([-0.5, 0.0, 0.5]) * (1 + index / 10)
            samples = np.arange(3, dtype=np.int32)
            genotypes = np.array(
                [[0, 1, 2, 0, 2, 1], [1, 2, 0, 2, 1, 0], [2, 0, 1, 1, 0, 2]], dtype=float
            )
            payload = {
                "active_positions": positions,
                "graph": {"edges": edges},
                "sample_nodes": samples,
                "genotypes": genotypes,
                "edge_log_rates": truth,
                "surface_kind": "barrier_fixture",
                "num_snps": genotypes.shape[1],
            }
            source_paths[index] = inputs / f"example_{index}.pkl"
            workflow_io.write_pickle(source_paths[index], payload)
            row = benchmark_data._base_row(
                index=index, payload=payload, positions=positions, edges=edges, sample_nodes=samples
            )
            workflow_io.write_json(inputs / f"example_{index}.metadata.json", row)
            workflow_io.write_json(
                cached / f"example_{index}.json",
                {
                    **row,
                    "method": "feems",
                    "input_sha256": workflow_io.sha256(source_paths[index]),
                    "protocol": feems.FEEMS_RECOMMENDED_LOO_PROTOCOL,
                    "cv_mode": "leave_one_out",
                    "cv_folds": 3,
                    "cv_jobs": 1,
                    "n_snps_after_filter": genotypes.shape[1],
                    "elapsed_seconds": 1.0,
                    **benchmark_data._metric_fields("feems_edge_log", truth, truth * 0.8),
                },
            )
            workflow_io.write_npz(
                cached / f"example_{index}.npz",
                positions=positions,
                edges=edges,
                truth=truth,
                feems=truth * 0.8,
                sample_count=np.ones(3, dtype=np.int32),
            )
        archive, checksum = inputs / "feems.tar.gz", inputs / "feems.sha256"
        benchmark_archive.pack_results(cached, source_paths, archive, checksum)
        config["release_validation"][label] = {
            "input_pattern": str(inputs / "example_{example}.pkl"),
            "metadata_pattern": str(inputs / "example_{example}.metadata.json"),
            "example_ids": list(indices),
            "feems_archive": str(archive),
            "feems_archive_checksum": str(checksum),
        }
    preview = run_workflow(
        tmp_path, config, "--dry-run", snakefile="release.Snakefile", target="validated_release"
    )
    assert "generate_raw_example" in preview
    assert "release_promote_surface_gate" in preview
    assert "broad_extract_feems_archive" in preview
    assert "broad_feems_cv_fold" not in preview
    assert not (output / "raw").exists()
    log = run_workflow(tmp_path, config, snakefile="release.Snakefile", target="validated_release")
    assert (output / ".snakemake/metadata").is_dir()
    # Snakemake opens its controller log before reading the workdir directive.
    # The database and per-job state must still live under the run directory.
    assert {path.name for path in (tmp_path / ".snakemake").iterdir()} == {"log"}
    assert "rule train_core:" in log
    assert "rule train_surface_gate:" in log
    assert "checkpoint views_write_selection:" in log
    for role in ("train", "validation"):
        selection = json.loads((output / "sampling/selection" / f"{role}.json").read_text())
        assert len(selection["examples"]) == 1
        paired = output / "sampling/prepared/counterfactual" / role / "pack_0.metadata.json"
        assert json.loads(paired.read_text())["examples"] == 4
    core = json.loads((output / "model/core/training_summary.json").read_text())
    gate = json.loads((output / "model/training_summary.json").read_text())
    assert core["init_checkpoint"] is None
    assert core["prepared_dimensions"]["examples"] == 8
    assert gate["prepared_dimensions"]["examples"] == 6
    assert gate["init_checkpoint"].endswith("model/core/checkpoint.best.dreems.zip")
    for stage in ("core", ""):
        assert (tmp_path / "published" / stage / "checkpoint.best.dreems.zip").read_bytes() == (
            output / "model" / stage / "checkpoint.best.dreems.zip"
        ).read_bytes()
    candidate = checkpoints.load_checkpoint(output / "release/dreems_candidate.dreems.zip")
    gate_hash = workflow_io.sha256(output / "model/checkpoint.best.dreems.zip")
    assert candidate.payload["provenance"]["surface_gate_checkpoint_sha256"] == gate_hash
    assert candidate.payload["provenance"]["frozen_core_verified"] is True
    assert candidate.payload["edge_sd_calibration"]["provenance"] == {
        "fit_examples": 2,
        "selection_examples": 2,
        "verification_examples": 2,
    }
    for stage in ("fit", "selection", "verification"):
        report = json.loads(
            (output / "release/report" / f"sd_calibration_{stage}.json").read_text()
        )
        assert report["checkpoint_sha256"] == gate_hash
    for label in ("broad", "hard"):
        report_dir = output / "benchmarks" / label / "report"
        assert (report_dir / "gallery/index.html").is_file()
        protocol = json.loads((report_dir / "protocol.json").read_text())
        assert protocol["dreems"]["checkpoint_sha256"] == gate_hash
        assert protocol["feems"]["result_source"] == "verified_archive"
        assert "rule " + label + "_feems_case:" not in log
    assert "Nothing to be done" in run_workflow(
        tmp_path,
        config,
        "--dry-run",
        snakefile="release.Snakefile",
        target="validated_release",
    )

    changed = copy.deepcopy(config)
    changed["dreems"]["optimizer"]["learning_rate"] = 1e-5
    planned = run_workflow(
        tmp_path, changed, "--dry-run", snakefile="release.Snakefile", target="validated_release"
    )
    assert "rule train_core:" in planned
    assert "rule broad_dreems_case:" in planned
    assert "rule release_promote_surface_gate:" in planned
    assert "rule broad_extract_feems_archive:" not in planned
    assert "rule hard_extract_feems_archive:" not in planned
    for name in (
        "generate_raw_example",
        "pack_raw_examples",
        "views_prepare_counterfactual_group",
        "views_prepare_replay_pack",
    ):
        assert f"rule {name}:" not in planned

    changed = copy.deepcopy(config)
    changed["sampling_views"]["counterfactual_augmentation"]["mask_seed"] = 17
    planned = run_workflow(tmp_path, changed, "--dry-run", snakefile="release.Snakefile")
    assert "rule views_prepare_counterfactual_group:" in planned
    assert "rule generate_raw_example:" not in planned
    assert "rule views_prepare_replay_pack:" not in planned
