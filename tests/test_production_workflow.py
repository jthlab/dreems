from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import yaml

from dreems import pipeline, train

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = REPO_ROOT / "workflow" / "config.production.yaml"
EXAMPLE_CONFIG = REPO_ROOT / "workflow" / "config.example.yaml"
SNP_RANGE_CONFIG = REPO_ROOT / "workflow" / "config.snp_range_experiment.yaml"
FULL_RETRAIN_CONFIG = REPO_ROOT / "workflow" / "config.full_retrain.yaml"
CALIBRATION_CONFIG = REPO_ROOT / "workflow" / "config.calibration.yaml"
SURFACE_GATE_CONFIG = REPO_ROOT / "workflow" / "config.null_surface_gate.yaml"
RELEASE_CONFIG = REPO_ROOT / "workflow" / "config.surface_gate_release.yaml"
RELEASED_CHECKPOINT = REPO_ROOT / "src" / "dreems" / "data" / "dreems_diffusion_bands.dreems.zip"
RELEASED_CHECKPOINT_SHA256 = "10d21a6326e848f48a7eccdd1a508d86f54effc55f79774486a626037a9e3530"
RELEASED_PARAMS_SHA256 = "d46045e3a5bbdc870427e2ca93b07daaac5a1d94ff7bbaa25e8f82972e3bec76"
SAMPLING_ROBUSTNESS_CHECKPOINT_SHA256 = (
    "0e3e56c2c3098c3f0b04d0dfdad067ddb6bb1a928c79ca9048dec505763c61d3"
)
SURFACE_GATE_CHECKPOINT_SHA256 = "df21d92755edeb9f37ac1dcc4251f4f1523f0096ce3ea5c6f914dec41075bdfc"


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    assert isinstance(payload, dict)
    return payload


def test_great_lakes_launcher_uses_slurm_submit_directory() -> None:
    launcher = (REPO_ROOT / "workflow" / "run_great_lakes.sh").read_text()

    assert "SLURM_SUBMIT_DIR" in launcher
    assert 'PROJECT_ROOT="${DREEMS_REPO:-${SLURM_SUBMIT_DIR:-$SCRIPT_ROOT}}"' in launcher
    assert "#SBATCH --account=jonth0" in launcher


def checkpoint_metadata() -> dict[str, Any]:
    with zipfile.ZipFile(RELEASED_CHECKPOINT) as archive:
        payload = json.loads(archive.read("metadata.json"))
    assert isinstance(payload, dict)
    return payload


def test_surface_gate_is_the_released_checkpoint() -> None:
    config = load_yaml(SURFACE_GATE_CONFIG)
    release = load_yaml(RELEASE_CONFIG)
    checkpoint = checkpoint_metadata()
    expected = checkpoint["training_config"]

    actual_config = pipeline.dreems_config(config)
    actual_config.validate()

    assert train.training_config_metadata(actual_config) == expected
    assert hashlib.sha256(RELEASED_CHECKPOINT.read_bytes()).hexdigest() == (
        RELEASED_CHECKPOINT_SHA256
    )
    with zipfile.ZipFile(RELEASED_CHECKPOINT) as archive:
        params = archive.read("params.eqx")
    assert hashlib.sha256(params).hexdigest() == RELEASED_PARAMS_SHA256

    provenance = checkpoint["provenance"]
    assert provenance["sampling_robustness_checkpoint_sha256"] == (
        SAMPLING_ROBUSTNESS_CHECKPOINT_SHA256
    )
    assert provenance["surface_gate_checkpoint_sha256"] == SURFACE_GATE_CHECKPOINT_SHA256
    assert provenance["frozen_pre_gate_model_verified"] is True
    assert provenance["posterior_sd_calibration_reused"] is False
    assert release["release_checkpoint"] == ("src/dreems/data/dreems_diffusion_bands.dreems.zip")
    assert checkpoint["model"]["features"]["kind"] == "topology_diffusion_bands_v2"
    assert checkpoint["model"]["architecture"]["kind"] == "diffusion_band_surface_gate_v3"
    assert checkpoint["model"]["architecture"]["surface_gate"] is True
    assert checkpoint["edge_sd_calibration"]["method"] == "density"
    assert checkpoint["edge_sd_calibration"]["density_scales"] == [
        0.99976411153997,
        1.1560023414999268,
        1.082870834658688,
    ]


def test_release_training_run_contract() -> None:
    checkpoint = checkpoint_metadata()

    assert checkpoint["training_run"]["best_step"] == 9_250
    assert checkpoint["training_run"]["final_step"] == 10_000
    assert checkpoint["training_run"]["stopped_early"] is False
    assert checkpoint["training_run"]["restored_best_validation"] is True
    assert len(checkpoint["training_run"]["history"]) == 41


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


def test_snp_range_experiment_matches_production_except_snp_distribution() -> None:
    production = load_yaml(PRODUCTION_CONFIG)
    experiment = load_yaml(SNP_RANGE_CONFIG)

    assert experiment["train_examples"] == production["train_examples"]
    assert experiment["test_examples"] == production["test_examples"]
    assert experiment["dreems"] == production["dreems"]
    expected_simulation = dict(production["simulation"])
    expected_simulation.update(
        {
            "min_num_snps": None,
            "max_num_snps": None,
            "num_snps_log_bins": [
                [50, 128],
                [129, 512],
                [513, 2_000],
                [2_001, 10_000],
            ],
        }
    )
    assert experiment["simulation"] == expected_simulation


def test_full_retrain_stages_surface_gate_and_keeps_counterfactuals_a_minority() -> None:
    experiment = load_yaml(SNP_RANGE_CONFIG)
    retrain = load_yaml(FULL_RETRAIN_CONFIG)

    train_supplement = retrain["supplemental_prepared_packs"]["train"]
    validation_supplement = retrain["supplemental_prepared_packs"]["validation"]
    counterfactual_examples = train_supplement["count"] * experiment["train_pack_size"]
    total_examples = experiment["train_examples"] + counterfactual_examples

    assert retrain["outdir"].startswith("/scratch/jonth_root/jonth0/")
    assert retrain["raw_source_dir"].endswith("/dreems_snp_range/raw")
    assert counterfactual_examples == 8_192
    assert counterfactual_examples / total_examples < 0.12
    assert validation_supplement["count"] == 2
    assert retrain["base_validation_examples"] == 1_024
    assert retrain["train_validation_examples"] == 1_152

    architecture = retrain["dreems"]["architecture"]
    assert architecture["initializer_gate"] is True
    assert architecture["surface_gate"] is True
    assert retrain["dreems"]["features"]["spatial_encoding"] == ("topology_node_covariance")
    assert retrain["dreems"]["optimizer"]["trainable_scope"] == "all"

    surface_gate = retrain["surface_gate_stage"]
    assert surface_gate["enabled"] is True
    assert surface_gate["validation_examples"] == 1_024
    assert surface_gate["optimizer"]["trainable_scope"] == "surface_gate"
    assert surface_gate["optimizer"]["learning_rate"] == 5e-4
    assert [item["count"] for item in surface_gate["prepared_packs"]["train"]] == [
        128,
        128,
    ]
    assert [
        item["count"] for item in surface_gate["prepared_packs"]["validation"]
    ] == [8, 8]


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
