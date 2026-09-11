"""Train the release model and optionally evaluate, calibrate, and verify a candidate."""

import copy
import math
from pathlib import Path
from numbers import Integral

from snakemake.utils import update_config

from dreems.utils.workflow_io import load_config


HERE = Path(workflow.current_basedir).resolve()
if "outdir" not in config:
    raise ValueError("pass --configfile workflow/config.release.yaml and set outdir for this machine")
ROOT = Path(config["outdir"]).expanduser().resolve()
release_validation = copy.deepcopy(config.get("release_validation"))
if release_validation is not None:
    if not isinstance(release_validation, dict) or not {"broad", "hard"} <= release_validation.keys():
        raise ValueError("release_validation must specify broad and hard H2H configurations")
    # Resolve user config files before the workflow changes its working directory.
    for label in ("broad", "hard"):
        if not isinstance(release_validation[label], dict):
            raise ValueError(f"release_validation.{label} must be a mapping")
        supplied = release_validation[label]
        benchmark = load_config(HERE.parent / supplied["configfile"]) if "configfile" in supplied else {}
        update_config(benchmark, {key: value for key, value in supplied.items() if key != "configfile"})
        for key in ("input_pattern", "metadata_pattern", "feems_archive", "feems_archive_checksum", "scenario_config", "notebook_scenario_config"):
            if benchmark.get(key):
                benchmark[key] = str(Path(benchmark[key]).expanduser().resolve())
        for required in ("input_pattern", "metadata_pattern"):
            if required not in benchmark:
                raise ValueError(f"release_validation.{label} is missing {required}")
        release_validation[label] = benchmark

# Retain the recorded scientific settings; replace only historical storage paths.
recipe = load_config(HERE / "config.snp_range_experiment.yaml")
update_config(recipe, load_config(HERE / "config.full_retrain.yaml"))
recipe.update(
    outdir=str(ROOT), publish_dir=None, raw_source_dir=None,
    prepared_pack_source_dir=None, normalization_source=None,
)
recipe["simulation"]["global_landscape_cartopy_data_dir"] = str(ROOT / "cartopy")
update_config(recipe, config)
recipe["outdir"] = str(ROOT)
for name in ("raw_source_dir", "prepared_pack_source_dir", "normalization_source", "publish_dir"):
    if recipe.get(name):
        recipe[name] = str(Path(recipe[name]).expanduser().resolve())
recipe["simulation"]["global_landscape_cartopy_data_dir"] = str(
    Path(recipe["simulation"]["global_landscape_cartopy_data_dir"]).expanduser().resolve()
)

# Keep the scheduler database and default worker logs beside this run's artifacts.
workdir: str(ROOT)

sampling = load_config(HERE / "config.sampling_robustness.yaml")
update_config(sampling, config.get("sampling_views", {}))
sampling.update(
    outdir=str(ROOT / "sampling"),
    source_corpus=str(Path(recipe["raw_source_dir"]).expanduser().resolve().parent)
    if recipe["raw_source_dir"] else str(ROOT),
    uv_cache_dir=recipe["uv_cache_dir"],
    uv_run_args=recipe["uv_run_args"],
    dreems=copy.deepcopy(recipe["dreems"]),
    source_pack_counts={
        split: math.ceil(recipe[f"{split}_examples"] / recipe["raw_pack_size"])
        for split in ("train", "test")
    },
)
if recipe.get("raw_source_layout") != "packs":
    raise ValueError("release training expects packed raw inputs")
augmentation = sampling["counterfactual_augmentation"]
groups_per_pack = augmentation["groups_per_pack"]
if groups_per_pack <= 0:
    raise ValueError("groups_per_pack must be positive")
paired_counts = {
    role: math.ceil(augmentation[f"{role}_base_examples"] / groups_per_pack)
    for role in ("train", "validation")
}
core_validation_packs = int(config.get("core_validation_mask_packs", 2))
if not 0 < core_validation_packs <= paired_counts["validation"]:
    raise ValueError("core_validation_mask_packs must fit within the prepared validation packs")


def pack_spec(kind, role, count):
    return {
        "pattern": str(ROOT / "sampling/prepared" / kind / role / "pack_{pack}.pkl"),
        "count": count,
    }


recipe["supplemental_prepared_packs"] = {
    "train": pack_spec("counterfactual", "train", paired_counts["train"]),
    "validation": pack_spec("counterfactual", "validation", core_validation_packs),
}
recipe["train_validation_examples"] = recipe["base_validation_examples"] + 4 * min(
    augmentation["validation_base_examples"], core_validation_packs * groups_per_pack,
)
gate = recipe["surface_gate_stage"]
gate["prepared_packs"] = {
    role: [
        pack_spec("counterfactual", role, paired_counts[role]),
        pack_spec("replay", role, augmentation[f"{role}_replay_packs"]),
    ]
    for role in ("train", "validation")
}
# This is an upper bound when the final raw pack is partial. The trainer records
# the actual loaded validation count and includes every prepared validation view.
gate["validation_examples"] = (
    4 * augmentation["validation_base_examples"]
    + recipe["raw_pack_size"] * augmentation["validation_replay_packs"]
)

# The main workflow owns the common runner and resource settings. Include it in
# this namespace; sampling preparation has separate paths and its own module.
config.clear()
config.update(recipe)
include: str(HERE / "Snakefile")

module views:
    snakefile: str(HERE / "sampling_robustness.Snakefile")
    config: sampling

use rule * from views as views_*


if release_validation is not None:
    # The candidate uses fresh neural predictions. The H2H modules retain their
    # own immutable FEEMS results, independent of which checkpoint is evaluated.
    candidate_gate = str(ROOT / "model/checkpoint.best.dreems.zip")
    for label in ("broad", "hard"):
        release_validation[label].update({
            "outdir": str(ROOT / "benchmarks" / label / "work"),
            "publish_dir": str(ROOT / "benchmarks" / label / "report"),
            "checkpoint": candidate_gate,
        })

    module broad_eval:
        snakefile: str(HERE / "h2h.Snakefile")
        config: release_validation["broad"]

    use rule * from broad_eval as broad_*

    module hard_eval:
        snakefile: str(HERE / "h2h.Snakefile")
        config: release_validation["hard"]

    use rule * from hard_eval as hard_*

    def benchmark_ids(label):
        """Keep calibration IDs identical to the selected H2H examples."""
        settings = release_validation[label]
        ids = settings.get("example_ids")
        if ids is None:
            first, count = settings.get("first_example", 0), settings["example_count"]
            if any(isinstance(value, bool) or not isinstance(value, Integral) for value in (first, count)):
                raise ValueError(f"{label} first_example and example_count must be integers")
            ids = list(range(first, first + count))
        return ids

    candidate = ROOT / "release/dreems_candidate.dreems.zip"
    promotion = dict(
        outdir=str(ROOT / "release/work"), publish_dir=str(ROOT / "release/report"),
        release_checkpoint=str(candidate), core_checkpoint=str(ROOT / "model/core/checkpoint.best.dreems.zip"),
        core_training_summary=str(ROOT / "model/core/training_summary.json"),
        core_training_history=str(ROOT / "model/core/training_history.jsonl"),
        surface_gate_checkpoint=candidate_gate,
        surface_gate_training_summary=str(ROOT / "model/training_summary.json"),
        surface_gate_training_history=str(ROOT / "model/training_history.jsonl"),
        training_configs=[TRAINING_CONFIG, SURFACE_GATE_TRAINING_CONFIG],
        prediction_pattern=str(ROOT / "benchmarks/broad/work/dreems/example_{example}.npz"),
        prediction_ids=benchmark_ids("broad"),
        independent_rows=str(ROOT / "benchmarks/broad/report/rows.csv"),
        broad_protocol=str(ROOT / "benchmarks/broad/report/protocol.json"),
        broad_report=str(ROOT / "benchmarks/broad/report/summary.json"),
        hard_protocol=str(ROOT / "benchmarks/hard/report/protocol.json"),
        hard_report=str(ROOT / "benchmarks/hard/report/summary.json"),
        expected_broad_examples=len(benchmark_ids("broad")),
        expected_hard_examples=len(benchmark_ids("hard")),
        calibration_seed=release_validation.get("calibration_seed", 20261001),
        require_checkpoint_hash=True, threads=1, mem_mb=4000, runtime="30m", partition="standard",
    )

    module verified_release:
        snakefile: str(HERE / "surface_gate_release.Snakefile")
        config: promotion

    use rule * from verified_release as release_*

    # Keep training-only `all` available. `validated_release` also produces the
    # benchmark reports and galleries, and never overwrites the packaged weights.
    localrules: validated_release

    rule validated_release:
        input:
            rules.all.input,
            rules.broad_all.input,
            rules.hard_all.input,
            rules.release_all.input,
