import copy
import json
import math
import os
import shlex
from pathlib import Path

from dreems import counterfactual


PROJECT_ROOT = Path(workflow.basedir).parent.resolve()
OUTDIR = Path(str(config["outdir"])).expanduser()
SOURCE_ROOT = Path(str(config["source_corpus"])).expanduser()
BASE_CHECKPOINT = Path(str(config["base_checkpoint"])).expanduser()
UV_CACHE_DIR = str(config.get("uv_cache_dir") or os.environ.get("UV_CACHE_DIR", "")).strip()
UV_CACHE_PREFIX = f"UV_CACHE_DIR={shlex.quote(UV_CACHE_DIR)} " if UV_CACHE_DIR else ""
UV_RUN_ARGS = str(config.get("uv_run_args", "--frozen --no-sync"))
UV_RUN = f"{UV_CACHE_PREFIX}uv --project {shlex.quote(str(PROJECT_ROOT))} run {UV_RUN_ARGS}"


def source_paths(*names):
    return [str(PROJECT_ROOT / "src" / "dreems" / name) for name in names]


PREPARE_SOURCES = source_paths(
    "counterfactual.py",
    "corpus.py",
    "features.py",
    "linearized.py",
    "pipeline.py",
    "simulation.py",
    "train.py",
    "workflow_io.py",
)
TRAIN_SOURCES = source_paths(
    "checkpoints.py",
    "features.py",
    "linearized.py",
    "losses.py",
    "model.py",
    "pipeline.py",
    "train.py",
    "workflow_io.py",
)
EVAL_SOURCES = source_paths(
    "checkpoints.py",
    "features.py",
    "metrics.py",
    "model.py",
    "pipeline.py",
    "workflow_io.py",
)


def pack_number(path):
    return int(Path(path).name.removeprefix("pack_").split(".", 1)[0])


def source_metadata(split):
    return sorted(
        (SOURCE_ROOT / "raw" / split).glob("pack_*.metadata.json"),
        key=pack_number,
    )


def source_raw_packs(split):
    return sorted((SOURCE_ROOT / "raw" / split).glob("pack_*.pkl"), key=pack_number)


AUGMENTATION = config["counterfactual_augmentation"]
TRAIN_BASE_COUNT = int(AUGMENTATION["train_base_examples"])
VALIDATION_BASE_COUNT = int(AUGMENTATION["validation_base_examples"])
GROUPS_PER_PACK = int(AUGMENTATION.get("groups_per_pack", 16))
if GROUPS_PER_PACK <= 0:
    raise ValueError("groups_per_pack must be positive")
MIN_SAMPLED_FRACTION = float(AUGMENTATION.get("min_sampled_fraction", 0.99))
MIN_SAMPLED_NODES = int(AUGMENTATION.get("min_sampled_nodes", 12))
SELECTION_SEED = int(AUGMENTATION.get("selection_seed", 20260831))
MASK_SEED = int(AUGMENTATION.get("mask_seed", 20260901))

SELECTIONS = {
    "train": counterfactual.select_base_examples(
        source_metadata("train"),
        count=TRAIN_BASE_COUNT,
        seed=SELECTION_SEED,
        min_sampled_fraction=MIN_SAMPLED_FRACTION,
        min_sampled_nodes=MIN_SAMPLED_NODES,
    ),
    "validation": counterfactual.select_base_examples(
        source_metadata("test"),
        count=VALIDATION_BASE_COUNT,
        seed=SELECTION_SEED + 1,
        min_sampled_fraction=MIN_SAMPLED_FRACTION,
        min_sampled_nodes=MIN_SAMPLED_NODES,
    ),
}
BASE_COUNTS = {"train": TRAIN_BASE_COUNT, "validation": VALIDATION_BASE_COUNT}


def selection_path(split):
    return OUTDIR / "selection" / f"{split}.json"


def group_path(split, slot):
    return OUTDIR / "staging" / "counterfactual" / split / f"group_{slot}.pkl"


def group_metadata_path(split, slot):
    return OUTDIR / "staging" / "counterfactual" / split / f"group_{slot}.metadata.json"


def counterfactual_pack_path(split, pack):
    return OUTDIR / "prepared" / "counterfactual" / split / f"pack_{pack}.pkl"


def counterfactual_pack_metadata_path(split, pack):
    return OUTDIR / "prepared" / "counterfactual" / split / f"pack_{pack}.metadata.json"


def counterfactual_pack_indices(split):
    return range(math.ceil(BASE_COUNTS[split] / GROUPS_PER_PACK))


def counterfactual_group_inputs(wildcards):
    split = str(wildcards.split)
    pack = int(wildcards.pack)
    start = pack * GROUPS_PER_PACK
    stop = min(start + GROUPS_PER_PACK, BASE_COUNTS[split])
    return [group_path(split, slot) for slot in range(start, stop)]


def selected_source_path(wildcards):
    return SELECTIONS[str(wildcards.split)][int(wildcards.slot)].raw_path


def replay_pack_path(split, pack):
    return OUTDIR / "prepared" / "replay" / split / f"pack_{pack}.pkl"


def replay_pack_metadata_path(split, pack):
    return OUTDIR / "prepared" / "replay" / split / f"pack_{pack}.metadata.json"


def choose_replay_packs(split, count, seed):
    available = source_raw_packs("train" if split == "train" else "test")
    if count > len(available):
        raise ValueError(f"requested {count} replay packs from {len(available)} available")
    rng = __import__("numpy").random.default_rng(seed)
    chosen = rng.choice(len(available), size=count, replace=False)
    return [available[int(index)] for index in sorted(chosen)]


REPLAY_PACKS = {
    "train": choose_replay_packs(
        "train",
        int(AUGMENTATION["train_replay_packs"]),
        SELECTION_SEED + 2,
    ),
    "validation": choose_replay_packs(
        "validation",
        int(AUGMENTATION["validation_replay_packs"]),
        SELECTION_SEED + 3,
    ),
}


def replay_source_path(wildcards):
    return REPLAY_PACKS[str(wildcards.split)][int(wildcards.pack)]


def stage_config(stage):
    payload = {
        "train_validation_examples": (
            4 * VALIDATION_BASE_COUNT + 64 * len(REPLAY_PACKS["validation"])
        ),
        "dreems": copy.deepcopy(config["dreems"]),
    }
    payload["dreems"]["optimizer"] = copy.deepcopy(config[f"{stage}_optimizer"])
    return payload


STAGE_CONFIGS = {stage: stage_config(stage) for stage in ("gate", "finetune")}


def stage_config_path(stage):
    return OUTDIR / "config" / f"{stage}.json"


def manifest_path(role):
    return OUTDIR / "manifests" / f"{role}.txt"


def training_pack_paths(role):
    paths = [
        counterfactual_pack_path(role, pack)
        for pack in counterfactual_pack_indices(role)
    ]
    paths.extend(replay_pack_path(role, pack) for pack in range(len(REPLAY_PACKS[role])))
    return paths


def model_path(stage, name):
    return OUTDIR / "model" / stage / name


def checkpoint_for_model(model_name):
    if model_name == "base":
        return BASE_CHECKPOINT
    return model_path(model_name, "checkpoint.best.dreems.zip")


MODELS = ("base", "gate", "finetune")
EVAL_DATASETS = ("replay", "counterfactual")


def eval_pack_indices(dataset):
    if dataset == "replay":
        return range(len(REPLAY_PACKS["validation"]))
    return counterfactual_pack_indices("validation")


def eval_input_path(wildcards):
    if str(wildcards.dataset) == "replay":
        return replay_pack_path("validation", int(wildcards.pack))
    return counterfactual_pack_path("validation", int(wildcards.pack))


def eval_summary_path(model_name, dataset, pack):
    return OUTDIR / "evaluation" / model_name / dataset / f"pack_{pack}" / "summary.json"


def eval_csv_path(model_name, dataset, pack):
    return OUTDIR / "evaluation" / model_name / dataset / f"pack_{pack}" / "summary.csv"


def aggregate_eval_inputs(wildcards):
    model_name = str(wildcards.model_name)
    dataset = str(wildcards.dataset)
    return [
        eval_summary_path(model_name, dataset, pack)
        for pack in eval_pack_indices(dataset)
    ]


localrules: all, write_selection, write_stage_config, write_manifest, extract_normalization, aggregate_evaluation


rule all:
    input:
        model_path("gate", "training_summary.json"),
        model_path("finetune", "training_summary.json"),
        expand(
            str(OUTDIR / "evaluation" / "{model_name}" / "{dataset}" / "summary.json"),
            model_name=MODELS,
            dataset=EVAL_DATASETS,
        ),


rule write_selection:
    input:
        lambda wildcards: source_metadata("train" if wildcards.split == "train" else "test"),
    output:
        str(selection_path("{split}")),
    wildcard_constraints:
        split="train|validation",
    run:
        split = str(wildcards.split)
        payload = counterfactual.selection_payload(
            SELECTIONS[split],
            seed=SELECTION_SEED + (split == "validation"),
            min_sampled_fraction=MIN_SAMPLED_FRACTION,
            min_sampled_nodes=MIN_SAMPLED_NODES,
        )
        path = Path(output[0])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


rule write_stage_config:
    output:
        str(stage_config_path("{stage}")),
    wildcard_constraints:
        stage="gate|finetune",
    run:
        path = Path(output[0])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(STAGE_CONFIGS[str(wildcards.stage)], indent=2, sort_keys=True) + "\n")


rule prepare_counterfactual_group:
    input:
        raw=selected_source_path,
        selection=lambda wildcards: selection_path(str(wildcards.split)),
        config=stage_config_path("gate"),
        code=PREPARE_SOURCES,
    output:
        prepared=temp(str(group_path("{split}", "{slot}"))),
        metadata=temp(str(group_metadata_path("{split}", "{slot}"))),
    wildcard_constraints:
        split="train|validation",
        slot=r"\d+",
    threads: int(config.get("prepare_threads", 1))
    resources:
        mem_mb=int(config.get("prepare_mem_mb", 6000)),
        runtime=str(config.get("prepare_runtime", "60m")),
        slurm_partition=str(config.get("prepare_partition", "standard")),
    shell:
        """
        export JAX_PLATFORMS=cpu
        export CUDA_VISIBLE_DEVICES=""
        export OMP_NUM_THREADS={threads}
        export OPENBLAS_NUM_THREADS={threads}
        export MKL_NUM_THREADS={threads}
        {UV_RUN} python -m dreems.pipeline prepare-counterfactual \
            --config {input.config:q} \
            --selection {input.selection:q} \
            --slot {wildcards.slot} \
            --seed {MASK_SEED} \
            --output {output.prepared:q} \
            --metadata {output.metadata:q}
        """


rule pack_counterfactual:
    input:
        prepared=counterfactual_group_inputs,
        code=source_paths("pipeline.py", "workflow_io.py"),
    output:
        packed=str(counterfactual_pack_path("{split}", "{pack}")),
        metadata=str(counterfactual_pack_metadata_path("{split}", "{pack}")),
    benchmark:
        str(OUTDIR / "benchmarks" / "pack_counterfactual" / "{split}" / "{pack}.tsv")
    wildcard_constraints:
        split="train|validation",
        pack=r"\d+",
    threads: 1
    resources:
        mem_mb=4000,
        runtime="30m",
        slurm_partition="standard",
    shell:
        """
        export JAX_PLATFORMS=cpu
        export CUDA_VISIBLE_DEVICES=""
        {UV_RUN} python -m dreems.pipeline pack-prepared \
            --inputs {input.prepared:q} \
            --output {output.packed:q} \
            --metadata {output.metadata:q}
        """


rule prepare_replay_pack:
    input:
        raw=replay_source_path,
        config=stage_config_path("gate"),
        code=PREPARE_SOURCES,
    output:
        packed=str(replay_pack_path("{split}", "{pack}")),
        metadata=str(replay_pack_metadata_path("{split}", "{pack}")),
    benchmark:
        str(OUTDIR / "benchmarks" / "prepare_replay" / "{split}" / "{pack}.tsv")
    wildcard_constraints:
        split="train|validation",
        pack=r"\d+",
    threads: int(config.get("prepare_threads", 1))
    resources:
        mem_mb=int(config.get("prepare_mem_mb", 6000)),
        runtime=str(config.get("prepare_runtime", "60m")),
        slurm_partition=str(config.get("prepare_partition", "standard")),
    shell:
        """
        export JAX_PLATFORMS=cpu
        export CUDA_VISIBLE_DEVICES=""
        export OMP_NUM_THREADS={threads}
        export OPENBLAS_NUM_THREADS={threads}
        export MKL_NUM_THREADS={threads}
        {UV_RUN} python -m dreems.pipeline prepare-batch \
            --config {input.config:q} \
            --inputs {input.raw:q} \
            --output {output.packed:q} \
            --metadata {output.metadata:q} \
            --start-index 0 \
            --max-examples 64
        """


rule write_manifest:
    input:
        lambda wildcards: training_pack_paths(str(wildcards.role)),
    output:
        str(manifest_path("{role}")),
    wildcard_constraints:
        role="train|validation",
    run:
        path = Path(output[0])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(str(Path(item).resolve()) for item in input) + "\n")


rule extract_normalization:
    input:
        checkpoint=str(BASE_CHECKPOINT),
    output:
        str(OUTDIR / "base_normalization.json"),
    run:
        import zipfile

        with zipfile.ZipFile(input.checkpoint) as archive:
            payload = json.loads(archive.read("metadata.json"))
        path = Path(output[0])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload["normalization"], indent=2, sort_keys=True) + "\n")


rule train_gate:
    input:
        normalization=str(OUTDIR / "base_normalization.json"),
        train_manifest=str(manifest_path("train")),
        validation_manifest=str(manifest_path("validation")),
        config=str(stage_config_path("gate")),
        init=str(BASE_CHECKPOINT),
        code=TRAIN_SOURCES,
    output:
        checkpoint=str(model_path("gate", "checkpoint.dreems.zip")),
        best=str(model_path("gate", "checkpoint.best.dreems.zip")),
        best_summary=str(model_path("gate", "checkpoint.best.summary.json")),
        summary=str(model_path("gate", "training_summary.json")),
    benchmark:
        str(OUTDIR / "benchmarks" / "train_gate.tsv")
    threads: int(config.get("train_threads", 4))
    resources:
        mem_mb=int(config.get("train_mem_mb", 24000)),
        runtime=str(config.get("gate_train_runtime", "12h")),
        gres="gpu:1",
        slurm_partition=str(config.get("train_partition", "gpu_mig40,spgpu,gpu")),
    shell:
        """
        export JAX_PLATFORMS=cuda
        export XLA_PYTHON_CLIENT_MEM_FRACTION={config[xla_mem_fraction]}
        CACHE_DIR="${{TMPDIR:-/tmp}}/dreems-gate-${{SLURM_JOB_ID}}"
        {UV_RUN} python -m dreems.pipeline train \
            --config {input.config:q} \
            --normalization {input.normalization:q} \
            --input-manifest {input.train_manifest:q} \
            --validation-input-manifest {input.validation_manifest:q} \
            --init-checkpoint {input.init:q} \
            --cache-dir "$CACHE_DIR" \
            --output {output.checkpoint:q} \
            --summary {output.summary:q} \
            --stream
        """


rule train_finetune:
    input:
        normalization=str(OUTDIR / "base_normalization.json"),
        train_manifest=str(manifest_path("train")),
        validation_manifest=str(manifest_path("validation")),
        config=str(stage_config_path("finetune")),
        init=str(model_path("gate", "checkpoint.best.dreems.zip")),
        code=TRAIN_SOURCES,
    output:
        checkpoint=str(model_path("finetune", "checkpoint.dreems.zip")),
        best=str(model_path("finetune", "checkpoint.best.dreems.zip")),
        best_summary=str(model_path("finetune", "checkpoint.best.summary.json")),
        summary=str(model_path("finetune", "training_summary.json")),
    benchmark:
        str(OUTDIR / "benchmarks" / "train_finetune.tsv")
    threads: int(config.get("train_threads", 4))
    resources:
        mem_mb=int(config.get("train_mem_mb", 24000)),
        runtime=str(config.get("finetune_train_runtime", "24h")),
        gres="gpu:1",
        slurm_partition=str(config.get("train_partition", "gpu_mig40,spgpu,gpu")),
    shell:
        """
        export JAX_PLATFORMS=cuda
        export XLA_PYTHON_CLIENT_MEM_FRACTION={config[xla_mem_fraction]}
        CACHE_DIR="${{TMPDIR:-/tmp}}/dreems-finetune-${{SLURM_JOB_ID}}"
        {UV_RUN} python -m dreems.pipeline train \
            --config {input.config:q} \
            --normalization {input.normalization:q} \
            --input-manifest {input.train_manifest:q} \
            --validation-input-manifest {input.validation_manifest:q} \
            --init-checkpoint {input.init:q} \
            --cache-dir "$CACHE_DIR" \
            --output {output.checkpoint:q} \
            --summary {output.summary:q} \
            --stream
        """


rule evaluate_pack:
    input:
        checkpoint=lambda wildcards: checkpoint_for_model(str(wildcards.model_name)),
        prepared=eval_input_path,
        code=EVAL_SOURCES,
    output:
        summary=str(
            OUTDIR
            / "evaluation"
            / "{model_name}"
            / "{dataset}"
            / "pack_{pack}"
            / "summary.json"
        ),
        csv=str(
            OUTDIR
            / "evaluation"
            / "{model_name}"
            / "{dataset}"
            / "pack_{pack}"
            / "summary.csv"
        ),
    wildcard_constraints:
        model_name="base|gate|finetune",
        dataset="replay|counterfactual",
        pack=r"\d+",
    threads: 1
    resources:
        mem_mb=8000,
        runtime="60m",
        slurm_partition="standard",
    params:
        output_dir=lambda wildcards: str(
            OUTDIR
            / "evaluation"
            / str(wildcards.model_name)
            / str(wildcards.dataset)
            / f"pack_{wildcards.pack}"
        ),
    shell:
        """
        export JAX_PLATFORMS=cpu
        export CUDA_VISIBLE_DEVICES=""
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        {UV_RUN} python -m dreems.pipeline evaluate \
            --checkpoint {input.checkpoint:q} \
            --inputs {input.prepared:q} \
            --output-dir {params.output_dir:q} \
            --batch-size 4
        """


rule aggregate_evaluation:
    input:
        summaries=aggregate_eval_inputs,
        code=EVAL_SOURCES,
    output:
        summary=str(OUTDIR / "evaluation" / "{model_name}" / "{dataset}" / "summary.json"),
        csv=str(OUTDIR / "evaluation" / "{model_name}" / "{dataset}" / "summary.csv"),
    wildcard_constraints:
        model_name="base|gate|finetune",
        dataset="replay|counterfactual",
    shell:
        """
        {UV_RUN} python -m dreems.pipeline aggregate-evaluations \
            --inputs {input.summaries:q} \
            --output-dir $(dirname {output.summary:q})
        """
