"""Prepare paired sampling views and ordinary replay packs for release training."""

import math
import os
import shlex
from pathlib import Path

import numpy as np

from dreems.training import counterfactual
from dreems.utils import workflow_io


PROJECT_ROOT = Path(workflow.basedir).parent.resolve()
OUTDIR = Path(config["outdir"]).expanduser()
SOURCE_ROOT = Path(config["source_corpus"]).expanduser()
UV_CACHE_DIR = str(config.get("uv_cache_dir") or os.environ.get("UV_CACHE_DIR", "")).strip()
UV_CACHE_PREFIX = f"UV_CACHE_DIR={shlex.quote(UV_CACHE_DIR)} " if UV_CACHE_DIR else ""
UV_RUN = (
    f"{UV_CACHE_PREFIX}uv --project {shlex.quote(str(PROJECT_ROOT))} run "
    f"{config.get('uv_run_args', '--frozen --no-sync')}"
)
PREPARE_SOURCES = [
    str(PROJECT_ROOT / "src" / "dreems" / name)
    for name in (
        "training/counterfactual.py", "data/corpus.py", "neural/features.py", "neural/linearized.py",
        "training/pipeline.py", "data/simulation.py", "training/train.py", "utils/validation.py", "utils/workflow_io.py",
    )
]
AUGMENTATION = config["counterfactual_augmentation"]
BASE_COUNTS = {
    "train": int(AUGMENTATION["train_base_examples"]),
    "validation": int(AUGMENTATION["validation_base_examples"]),
}
REPLAY_COUNTS = {
    "train": int(AUGMENTATION["train_replay_packs"]),
    "validation": int(AUGMENTATION["validation_replay_packs"]),
}
GROUPS_PER_PACK = int(AUGMENTATION.get("groups_per_pack", 16))
if GROUPS_PER_PACK <= 0 or min(BASE_COUNTS.values()) <= 0 or min(REPLAY_COUNTS.values()) < 0:
    raise ValueError("base counts and groups_per_pack must be positive; replay counts cannot be negative")
SELECTION_SEED = int(AUGMENTATION.get("selection_seed", 20260831))
MASK_SEED = int(AUGMENTATION.get("mask_seed", 20260901))
SELECTION = str(OUTDIR / "selection" / "{split}.json")
GROUP = str(OUTDIR / "staging/counterfactual/{split}/group_{slot}.pkl")
GROUP_METADATA = str(OUTDIR / "staging/counterfactual/{split}/group_{slot}.metadata.json")
PAIRED_PACK = str(OUTDIR / "prepared/counterfactual/{split}/pack_{pack}.pkl")
PAIRED_METADATA = str(OUTDIR / "prepared/counterfactual/{split}/pack_{pack}.metadata.json")
REPLAY_PACK = str(OUTDIR / "prepared/replay/{split}/pack_{pack}.pkl")
REPLAY_METADATA = str(OUTDIR / "prepared/replay/{split}/pack_{pack}.metadata.json")
FEATURE_CONFIG = str(OUTDIR / "config/features.json")
FEATURE_PAYLOAD = {
    "dreems": {key: config["dreems"][key] for key in ("features", "edge_conditioner")}
}

# A fresh DAG knows its source pack counts before those files exist. Standalone
# preparation can instead discover an already-completed corpus.
SOURCE_PACKS = {}
REPLAY_SOURCES = {}
for role, raw_split, seed_offset in (("train", "train", 2), ("validation", "test", 3)):
    counts = config.get("source_pack_counts")
    if counts is None:
        available = sorted(
            (SOURCE_ROOT / "raw" / raw_split).glob("pack_*.pkl"),
            key=lambda path: int(path.stem.removeprefix("pack_")),
        )
    else:
        available = [
            SOURCE_ROOT / "raw" / raw_split / f"pack_{pack}.pkl"
            for pack in range(int(counts[raw_split]))
        ]
    if not available:
        raise ValueError(f"no {raw_split} packs; supply source_pack_counts for a fresh corpus")
    if REPLAY_COUNTS[role] > len(available):
        raise ValueError(f"requested {REPLAY_COUNTS[role]} replay packs from {len(available)} available")
    SOURCE_PACKS[role] = available
    rng = np.random.default_rng(SELECTION_SEED + seed_offset)
    chosen = rng.choice(len(available), size=REPLAY_COUNTS[role], replace=False)
    REPLAY_SOURCES[role] = [available[int(index)] for index in sorted(chosen)]


def prepared_paths():
    return [
        *[
            PAIRED_PACK.format(split=role, pack=pack)
            for role in BASE_COUNTS
            for pack in range(math.ceil(BASE_COUNTS[role] / GROUPS_PER_PACK))
        ],
        *[
            REPLAY_PACK.format(split=role, pack=pack)
            for role in REPLAY_COUNTS
            for pack in range(REPLAY_COUNTS[role])
        ],
    ]


def selected_source(wildcards):
    # Snakemake reevaluates this input only after selection has finished.
    selected = checkpoints.write_selection.get(split=wildcards.split).output[0]
    return counterfactual.read_selection(Path(selected))[int(wildcards.slot)].raw_path


def group_inputs(wildcards):
    start = int(wildcards.pack) * GROUPS_PER_PACK
    stop = min(start + GROUPS_PER_PACK, BASE_COUNTS[wildcards.split])
    return [GROUP.format(split=wildcards.split, slot=slot) for slot in range(start, stop)]


rule all:
    input:
        prepared_paths()


rule prepared:
    input:
        prepared_paths()


checkpoint write_selection:
    localrule: True
    input:
        metadata=lambda wc: [path.with_suffix(".metadata.json") for path in SOURCE_PACKS[wc.split]],
        raw=lambda wc: SOURCE_PACKS[wc.split],
        code=[str(PROJECT_ROOT / "src/dreems" / name) for name in ("training/counterfactual.py", "utils/workflow_io.py")],
    output:
        SELECTION,
    wildcard_constraints:
        split="train|validation",
    params:
        n_examples=lambda wc: BASE_COUNTS[wc.split],
        seed=lambda wc: SELECTION_SEED + (wc.split == "validation"),
        fraction=float(AUGMENTATION.get("min_sampled_fraction", 0.99)),
        nodes=int(AUGMENTATION.get("min_sampled_nodes", 12)),
    run:
        selected = counterfactual.select_base_examples(
            [Path(path) for path in input.metadata], count=params.n_examples,
            seed=params.seed, min_sampled_fraction=params.fraction,
            min_sampled_nodes=params.nodes,
        )
        workflow_io.write_json(
            Path(output[0]),
            counterfactual.selection_payload(
                selected, seed=params.seed, min_sampled_fraction=params.fraction,
                min_sampled_nodes=params.nodes,
            ),
        )


rule write_feature_config:
    localrule: True
    output:
        FEATURE_CONFIG,
    params:
        payload=FEATURE_PAYLOAD,
    run:
        workflow_io.write_json(Path(output[0]), params.payload)


rule prepare_counterfactual_group:
    input:
        raw=selected_source,
        selection=lambda wc: SELECTION.format(split=wc.split),
        config=FEATURE_CONFIG,
        code=PREPARE_SOURCES,
    output:
        prepared=temp(GROUP),
        metadata=temp(GROUP_METADATA),
    wildcard_constraints:
        split="train|validation",
        slot=r"\d+",
    benchmark:
        str(OUTDIR / "benchmarks/prepare_counterfactual/{split}/group_{slot}.tsv"),
    threads: int(config.get("prepare_threads", 1))
    resources:
        mem_mb=int(config.get("prepare_mem_mb", 6000)),
        runtime=str(config.get("prepare_runtime", "60m")),
        slurm_partition=str(config.get("prepare_partition", "standard")),
    params:
        seed=MASK_SEED,
        runner=UV_RUN,
    shell:
        """
        export JAX_PLATFORMS=cpu
        export CUDA_VISIBLE_DEVICES=""
        export OMP_NUM_THREADS={threads}
        export OPENBLAS_NUM_THREADS={threads}
        export MKL_NUM_THREADS={threads}
        {params.runner} python -m dreems.training.pipeline prepare-counterfactual \
            --config {input.config:q} --selection {input.selection:q} \
            --slot {wildcards.slot} --seed {params.seed} \
            --output {output.prepared:q} --metadata {output.metadata:q}
        """


rule pack_counterfactual:
    input:
        prepared=group_inputs,
        code=[str(PROJECT_ROOT / "src/dreems" / name) for name in ("training/pipeline.py", "utils/workflow_io.py")],
    output:
        packed=PAIRED_PACK,
        metadata=PAIRED_METADATA,
    benchmark:
        str(OUTDIR / "benchmarks/pack_counterfactual/{split}/{pack}.tsv"),
    wildcard_constraints:
        split="train|validation",
        pack=r"\d+",
    threads: 1
    resources:
        mem_mb=4000,
        runtime="30m",
        slurm_partition="standard",
    params:
        runner=UV_RUN,
    shell:
        """
        export JAX_PLATFORMS=cpu
        export CUDA_VISIBLE_DEVICES=""
        {params.runner} python -m dreems.training.pipeline pack-prepared \
            --inputs {input.prepared:q} --output {output.packed:q} \
            --metadata {output.metadata:q}
        """


rule prepare_replay_pack:
    input:
        raw=lambda wc: REPLAY_SOURCES[wc.split][int(wc.pack)],
        config=FEATURE_CONFIG,
        code=PREPARE_SOURCES,
    output:
        packed=REPLAY_PACK,
        metadata=REPLAY_METADATA,
    benchmark:
        str(OUTDIR / "benchmarks/prepare_replay/{split}/{pack}.tsv"),
    wildcard_constraints:
        split="train|validation",
        pack=r"\d+",
    threads: int(config.get("prepare_threads", 1))
    resources:
        mem_mb=int(config.get("prepare_mem_mb", 6000)),
        runtime=str(config.get("prepare_runtime", "60m")),
        slurm_partition=str(config.get("prepare_partition", "standard")),
    params:
        runner=UV_RUN,
    shell:
        """
        export JAX_PLATFORMS=cpu
        export CUDA_VISIBLE_DEVICES=""
        export OMP_NUM_THREADS={threads}
        export OPENBLAS_NUM_THREADS={threads}
        export MKL_NUM_THREADS={threads}
        {params.runner} python -m dreems.training.pipeline prepare-batch \
            --config {input.config:q} --inputs {input.raw:q} \
            --output {output.packed:q} --metadata {output.metadata:q}
        """
