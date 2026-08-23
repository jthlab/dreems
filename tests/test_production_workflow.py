from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from dreems import pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = REPO_ROOT / "workflow" / "config.production.yaml"
EXAMPLE_CONFIG = REPO_ROOT / "workflow" / "config.example.yaml"
CALIBRATION_CONFIG = REPO_ROOT / "workflow" / "config.calibration.yaml"
RELEASED_CHECKPOINT = REPO_ROOT / "src" / "dreems" / "data" / "dreems_diffusion_bands.dreems.zip"
RELEASED_CHECKPOINT_SHA256 = "f0b677d2cc028252002d15fac843ca7347b1b352eb2286a18bd202b724aa92b4"
RELEASED_PARAMS_SHA256 = "d7899a6bb419e4d017e093af958ab9750c2a947ccc72087ccf7bfcd18a3c2984"
SOURCE_CHECKPOINT_SHA256 = "f237554026432e2874a293b5f3c5dfbe680c4506d2075d4e0a7f728d480e4ff4"
TRAINED_CHECKPOINT_SHA256 = "038632b1e11a8a4dc9ac858967c1f7cfa56c45920a1ec589a41823757c4a2bfe"
SOURCE_CONFIG_SHA256 = "47a44d278b50cda725d1ffeae1d642f0a7d9c70c1c23ac5e68f6c9abc8ba5c8b"
RAW_CORPUS_CONFIG_SHA256 = "0c750edcf7553f451e8cf24dbddd7347ac85e9b5a432458c5b00181a4e7c5ea5"


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    assert isinstance(payload, dict)
    return payload


def checkpoint_metadata() -> dict[str, Any]:
    with zipfile.ZipFile(RELEASED_CHECKPOINT) as archive:
        payload = json.loads(archive.read("metadata.json"))
    assert isinstance(payload, dict)
    return payload


def test_production_config_matches_released_training_config() -> None:
    config = load_yaml(PRODUCTION_CONFIG)
    checkpoint = checkpoint_metadata()
    expected = checkpoint["training_config"]

    actual_config = pipeline.dreems_config(config)
    actual_config.validate()

    assert asdict(actual_config) == expected
    assert hashlib.sha256(RELEASED_CHECKPOINT.read_bytes()).hexdigest() == (
        RELEASED_CHECKPOINT_SHA256
    )
    assert config["provenance"]["released_checkpoint_sha256"] == (RELEASED_CHECKPOINT_SHA256)
    with zipfile.ZipFile(RELEASED_CHECKPOINT) as archive:
        params = archive.read("params.eqx")
    assert hashlib.sha256(params).hexdigest() == RELEASED_PARAMS_SHA256
    assert config["provenance"]["released_params_sha256"] == RELEASED_PARAMS_SHA256
    assert config["provenance"]["source_checkpoint_sha256"] == (SOURCE_CHECKPOINT_SHA256)
    assert config["provenance"]["trained_checkpoint_sha256"] == TRAINED_CHECKPOINT_SHA256
    assert config["provenance"]["training_config_sha256"] == SOURCE_CONFIG_SHA256
    assert config["provenance"]["raw_corpus_config_sha256"] == RAW_CORPUS_CONFIG_SHA256

    provenance = checkpoint["provenance"]
    assert provenance["source_checkpoint_sha256"] == SOURCE_CHECKPOINT_SHA256
    assert provenance["source_params_sha256"] == RELEASED_PARAMS_SHA256
    assert provenance["training_config_sha256"] == SOURCE_CONFIG_SHA256
    assert provenance["raw_corpus_config_sha256"] == RAW_CORPUS_CONFIG_SHA256
    assert checkpoint["model"]["features"]["kind"] == "topology_diffusion_bands_v2"
    assert checkpoint["feature_migration"]["training_input"] == "node_covariance"
    assert checkpoint["feature_migration"]["parameter_bytes_changed"] is False
    assert checkpoint["edge_sd_calibration"]["method"] == "density"
    assert checkpoint["edge_sd_calibration"]["density_scales"] == [
        1.0752856083032754,
        0.9951167513722861,
        1.9091496173280793,
    ]
    assert (
        checkpoint["feature_migration"]["provenance"]["source_checkpoint_sha256"]
        == TRAINED_CHECKPOINT_SHA256
    )


def test_release_training_run_contract() -> None:
    checkpoint = checkpoint_metadata()

    assert checkpoint["training_run"]["best_step"] == 29_000
    assert checkpoint["training_run"]["final_step"] == 41_000
    assert checkpoint["training_run"]["stopped_early"] is True
    assert checkpoint["training_run"]["restored_best_validation"] is True
    assert len(checkpoint["training_run"]["history"]) == 42


def test_production_corpus_contract() -> None:
    config = load_yaml(PRODUCTION_CONFIG)
    simulation = config["simulation"]

    assert config["train_examples"] == 61_440
    assert config["test_examples"] == 4_096
    assert config["train_validation_examples"] == 1_024
    assert config["raw_source_layout"] == "packs"
    assert config["raw_pack_size"] == 64
    assert config["train_pack_size"] == 64
    assert config["validation_pack_size"] == 64
    assert config["train_accelerator"] == "cuda"
    assert config["train_gpus"] == 1

    assert simulation["seed"] == 971_801
    assert simulation["simulator"] == "island_coalescent"
    assert simulation["island_require_rust"] is True
    assert simulation["island_emit_covariance"] is True
    assert simulation["island_ploidy"] == 2
    assert simulation["island_symmetric_migration"] is True
    assert simulation["island_marker_model"] == "conditioned_random_marker"
    assert simulation["island_min_allele_frequency"] == 0.05
    assert simulation["island_max_allele_frequency"] == 0.95
    assert simulation["min_num_snps"] == 2_000
    assert simulation["max_num_snps"] == 10_000
    assert simulation["case_weights"] == {
        "feems_lattice": 0.2,
        "global_landscape": 0.56,
        "wolflike": 0.084,
        "earth_like": 0.0312,
        "grid": 0.1248,
    }
    assert simulation["global_landscape_target_node_choices"] == [220, 450, 900, 1400]


def test_example_config_preserves_production_model_and_conditioning() -> None:
    production = load_yaml(PRODUCTION_CONFIG)
    example = load_yaml(EXAMPLE_CONFIG)

    for section in ("architecture", "features", "edge_conditioner", "loss"):
        assert example["dreems"][section] == production["dreems"][section]

    for key in (
        "simulator",
        "island_emit_covariance",
        "island_ploidy",
        "island_symmetric_migration",
        "island_marker_model",
        "island_min_allele_frequency",
        "island_max_allele_frequency",
    ):
        assert example["simulation"][key] == production["simulation"][key]

    assert example["train_accelerator"] == "cpu"
    assert example["train_gpus"] == 0


def test_workflows_use_compacted_artifacts_and_sqlite_provenance() -> None:
    production = load_yaml(PRODUCTION_CONFIG)
    calibration = load_yaml(CALIBRATION_CONFIG)

    assert production["raw_source_layout"] == "packs"
    assert production["raw_pack_size"] == production["train_pack_size"]
    assert calibration["raw_source_layout"] == "packs"
    assert calibration["raw_pack_size"] == production["raw_pack_size"]

    for profile_name in ("local", "slurm"):
        profile = load_yaml(REPO_ROOT / "workflow" / "profiles" / profile_name / "profile.yaml")
        assert profile["persistence-backend"] == "db"
        assert profile["persistence-backend-db-url"] == "sqlite:///.snakemake/metadata.db"
