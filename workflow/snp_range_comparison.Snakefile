"""Compare two checkpoints on the independent broad SNP-count test corpus."""

import math
import os
import shlex
from pathlib import Path


PROJECT_ROOT = Path(workflow.basedir).parent.resolve()
SOURCE_ROOT = Path(config["source_root"]).expanduser().resolve()
OUTDIR = Path(config["outdir"]).expanduser().resolve()
PUBLISH_DIR = Path(config["publish_dir"]).expanduser().resolve()
ROUTER_PUBLISH_DIR = Path(
    config.get("router_publish_dir", f"{PUBLISH_DIR}_router")
).expanduser().resolve()
ROUTER_H2H_PUBLISH_DIR = Path(
    config.get("router_h2h_publish_dir", f"{PUBLISH_DIR}_router_h2h")
).expanduser().resolve()
ROUTER_H2H_REFERENCE_GALLERY = Path(
    config["router_h2h_reference_gallery"]
).expanduser().resolve()
ROUTER_H2H_REFERENCE_ROWS = Path(config["router_h2h_reference_rows"]).expanduser().resolve()
FEATURE_CONFIG = Path(config["feature_config"]).expanduser().resolve()
OLD_CHECKPOINT = Path(config["old_checkpoint"]).expanduser().resolve()
NEW_CHECKPOINT = Path(config["new_checkpoint"]).expanduser().resolve()
RAW_PACK_PATTERN = str(config["raw_pack_pattern"])
RAW_METADATA_PATTERN = str(config["raw_metadata_pattern"])
REUSED_PREPARED_DIR = (
    Path(config["reused_prepared_dir"]).expanduser().resolve()
    if config.get("reused_prepared_dir")
    else None
)
REUSED_PREPARED_PACKS = int(config.get("reused_prepared_packs", 0))
TEST_EXAMPLES = int(config["test_examples"])
PACK_SIZE = int(config["pack_size"])
PACKS = list(range(math.ceil(TEST_EXAMPLES / PACK_SIZE)))
MODELS = ("old", "new")
ROUTER_THRESHOLD = int(config.get("router_threshold", 512))

UV_CACHE_DIR = str(config.get("uv_cache_dir") or os.environ.get("UV_CACHE_DIR", "")).strip()
UV_CACHE_PREFIX = f"UV_CACHE_DIR={shlex.quote(UV_CACHE_DIR)} " if UV_CACHE_DIR else ""
UV_RUN_ARGS = str(config.get("uv_run_args", "--frozen --no-sync"))
UV_RUN = f"{UV_CACHE_PREFIX}uv --project {shlex.quote(str(PROJECT_ROOT))} run {UV_RUN_ARGS}"


def raw_pack(pack):
    return RAW_PACK_PATTERN.format(pack=int(pack))


def raw_metadata(pack):
    return RAW_METADATA_PATTERN.format(pack=int(pack))


def generated_prepared_pack(pack):
    return str(OUTDIR / "prepared" / f"pack_{int(pack)}.pkl")


def generated_prepared_metadata(pack):
    return str(OUTDIR / "prepared" / f"pack_{int(pack)}.metadata.json")


def prepared_pack(pack):
    pack = int(pack)
    if REUSED_PREPARED_DIR is not None and pack < REUSED_PREPARED_PACKS:
        return str(REUSED_PREPARED_DIR / f"pack_{pack}.pkl")
    return generated_prepared_pack(pack)


def checkpoint(model):
    if str(model) == "old":
        return str(OLD_CHECKPOINT)
    if str(model) == "new":
        return str(NEW_CHECKPOINT)
    raise ValueError(f"unknown model {model!r}")


def eval_dir(model, pack):
    return OUTDIR / "evaluation" / str(model) / f"pack_{int(pack)}"


def eval_summary(model, pack):
    return str(eval_dir(model, pack) / "summary.json")


def eval_csv(model, pack):
    return str(eval_dir(model, pack) / "summary.csv")


def prediction_archive(model, pack):
    return str(eval_dir(model, pack) / "predictions.npz")


def aggregate_summary(model):
    return str(PUBLISH_DIR / "evaluations" / str(model) / "summary.json")


def aggregate_csv(model):
    return str(PUBLISH_DIR / "evaluations" / str(model) / "summary.csv")


def pack_count(pack):
    start = int(pack) * PACK_SIZE
    return min(PACK_SIZE, TEST_EXAMPLES - start)


localrules: all, aggregate_evaluations, compare_checkpoints, router_report


rule all:
    input:
        str(PUBLISH_DIR / "summary.json"),
        str(PUBLISH_DIR / "rows.csv"),
        str(PUBLISH_DIR / "gallery"),
        str(ROUTER_PUBLISH_DIR / "gallery"),
        str(ROUTER_H2H_PUBLISH_DIR / "gallery"),


rule prepare_pack:
    input:
        raw=lambda wildcards: raw_pack(wildcards.pack),
        feature_config=str(FEATURE_CONFIG),
        code=[
            "src/dreems/pipeline.py",
            "src/dreems/features.py",
            "src/dreems/linearized.py",
            "src/dreems/workflow_io.py",
        ],
    output:
        prepared=str(OUTDIR / "prepared" / "pack_{pack}.pkl"),
        metadata=str(OUTDIR / "prepared" / "pack_{pack}.metadata.json"),
    benchmark:
        str(OUTDIR / "benchmarks" / "prepare" / "{pack}.tsv")
    wildcard_constraints:
        pack=r"\d+",
    threads: int(config.get("prepare_threads", 1))
    resources:
        mem_mb=int(config.get("prepare_mem_mb", 2000)),
        runtime=str(config.get("prepare_runtime", "30m")),
        slurm_partition=str(config.get("partition", "standard")),
        slurm_account="jonth0",
    params:
        max_examples=lambda wildcards: pack_count(wildcards.pack),
    shell:
        """
        set -euo pipefail
        export JAX_PLATFORMS=cpu
        export CUDA_VISIBLE_DEVICES=""
        export OMP_NUM_THREADS={threads}
        export OPENBLAS_NUM_THREADS={threads}
        export MKL_NUM_THREADS={threads}
        {UV_RUN} python -m dreems.pipeline prepare-batch \
            --config {input.feature_config:q} \
            --inputs {input.raw:q} \
            --output {output.prepared:q} \
            --metadata {output.metadata:q} \
            --start-index 0 \
            --max-examples {params.max_examples}
        """


rule evaluate_pack:
    input:
        prepared=lambda wildcards: prepared_pack(wildcards.pack),
        checkpoint=lambda wildcards: checkpoint(wildcards.model),
        code=[
            "src/dreems/pipeline.py",
            "src/dreems/checkpoints.py",
            "src/dreems/features.py",
            "src/dreems/metrics.py",
            "src/dreems/model.py",
        ],
    output:
        summary=str(OUTDIR / "evaluation" / "{model}" / "pack_{pack}" / "summary.json"),
        csv=str(OUTDIR / "evaluation" / "{model}" / "pack_{pack}" / "summary.csv"),
        predictions=str(
            OUTDIR / "evaluation" / "{model}" / "pack_{pack}" / "predictions.npz"
        ),
    benchmark:
        str(OUTDIR / "benchmarks" / "evaluate" / "{model}" / "{pack}.tsv")
    wildcard_constraints:
        model="old|new",
        pack=r"\d+",
    threads: int(config.get("eval_threads", 1))
    resources:
        mem_mb=int(config.get("eval_mem_mb", 6000)),
        runtime=str(config.get("eval_runtime", "60m")),
        slurm_partition=str(config.get("partition", "standard")),
        slurm_account="jonth0",
    params:
        output_dir=lambda wildcards: str(eval_dir(wildcards.model, wildcards.pack)),
        offset=lambda wildcards: int(wildcards.pack) * PACK_SIZE,
    shell:
        """
        set -euo pipefail
        export JAX_LOGGING_LEVEL=CRITICAL
        export JAX_PLATFORMS=cpu
        export JAX_PLATFORM_NAME=cpu
        export CUDA_VISIBLE_DEVICES=""
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        export OMP_NUM_THREADS={threads}
        export OPENBLAS_NUM_THREADS={threads}
        export MKL_NUM_THREADS={threads}
        {UV_RUN} python -m dreems.pipeline evaluate \
            --checkpoint {input.checkpoint:q} \
            --inputs {input.prepared:q} \
            --output-dir {params.output_dir:q} \
            --batch-size {config[eval_batch_size]} \
            --replicate-offset {params.offset} \
            --predictions-output {output.predictions:q}
        """


rule aggregate_evaluations:
    input:
        summaries=lambda wildcards: [
            eval_summary(wildcards.model, pack) for pack in PACKS
        ],
        code="src/dreems/pipeline.py",
    output:
        summary=str(PUBLISH_DIR / "evaluations" / "{model}" / "summary.json"),
        csv=str(PUBLISH_DIR / "evaluations" / "{model}" / "summary.csv"),
    wildcard_constraints:
        model="old|new",
    threads: 1
    resources:
        mem_mb=4000,
        runtime="20m",
    params:
        output_dir=lambda wildcards: str(PUBLISH_DIR / "evaluations" / wildcards.model),
    shell:
        """
        set -euo pipefail
        {UV_RUN} python -m dreems.pipeline aggregate-evaluations \
            --inputs {input.summaries:q} \
            --output-dir {params.output_dir:q}
        """


rule compare_checkpoints:
    input:
        old=aggregate_summary("old"),
        new=aggregate_summary("new"),
        metadata=[raw_metadata(pack) for pack in PACKS],
        old_checkpoint=str(OLD_CHECKPOINT),
        new_checkpoint=str(NEW_CHECKPOINT),
        code="src/dreems/benchmark_gallery.py",
    output:
        summary=str(PUBLISH_DIR / "summary.json"),
        rows=str(PUBLISH_DIR / "rows.csv"),
        stats=str(PUBLISH_DIR / "aggregate_stats.csv"),
        selections=str(PUBLISH_DIR / "selected_examples.json"),
    threads: 1
    resources:
        mem_mb=4000,
        runtime="20m",
    shell:
        """
        set -euo pipefail
        {UV_RUN} python -m dreems.benchmark_gallery compare-checkpoints \
            --old-summary {input.old:q} \
            --new-summary {input.new:q} \
            --metadata {input.metadata:q} \
            --old-checkpoint {input.old_checkpoint:q} \
            --new-checkpoint {input.new_checkpoint:q} \
            --output-dir {PUBLISH_DIR:q}
        """


rule router_report:
    input:
        rows=str(PUBLISH_DIR / "rows.csv"),
        code="src/dreems/benchmark_gallery.py",
    output:
        summary=str(ROUTER_PUBLISH_DIR / "summary.json"),
        rows=str(ROUTER_PUBLISH_DIR / "rows.csv"),
        stats=str(ROUTER_PUBLISH_DIR / "aggregate_stats.csv"),
        selections=str(ROUTER_PUBLISH_DIR / "selected_examples.json"),
    threads: 1
    resources:
        mem_mb=2000,
        runtime="20m",
    shell:
        """
        set -euo pipefail
        {UV_RUN} python -m dreems.benchmark_gallery router-report \
            --rows {input.rows:q} \
            --threshold {ROUTER_THRESHOLD} \
            --output-dir {ROUTER_PUBLISH_DIR:q}
        """


rule checkpoint_gallery:
    input:
        rows=str(PUBLISH_DIR / "rows.csv"),
        summary=str(PUBLISH_DIR / "summary.json"),
        selections=str(PUBLISH_DIR / "selected_examples.json"),
        predictions=[
            prediction_archive(model, pack) for model in MODELS for pack in PACKS
        ],
        raw=[raw_pack(pack) for pack in PACKS],
        code="src/dreems/benchmark_gallery.py",
    output:
        gallery=directory(str(PUBLISH_DIR / "gallery")),
    threads: int(config.get("gallery_threads", 2))
    resources:
        mem_mb=int(config.get("gallery_mem_mb", 12000)),
        runtime=str(config.get("gallery_runtime", "120m")),
        slurm_partition=str(config.get("partition", "standard")),
        slurm_account="jonth0",
    params:
        prediction_pattern=lambda _wildcards: str(
            OUTDIR / "evaluation" / "{model}" / "pack_{pack}" / "predictions.npz"
        ),
    shell:
        """
        set -euo pipefail
        export MPLCONFIGDIR="${{TMPDIR:-/tmp}}/matplotlib-${{SLURM_JOB_ID:-local}}"
        export OMP_NUM_THREADS={threads}
        export OPENBLAS_NUM_THREADS={threads}
        export MKL_NUM_THREADS={threads}
        {UV_RUN} python -m dreems.benchmark_gallery checkpoint-gallery \
            --rows {input.rows:q} \
            --summary {input.summary:q} \
            --selections {input.selections:q} \
            --raw-pattern {RAW_PACK_PATTERN:q} \
            --prediction-pattern {params.prediction_pattern:q} \
            --pack-size {PACK_SIZE} \
            --output-dir {output.gallery:q}
        """


rule router_gallery:
    input:
        rows=str(ROUTER_PUBLISH_DIR / "rows.csv"),
        summary=str(ROUTER_PUBLISH_DIR / "summary.json"),
        selections=str(ROUTER_PUBLISH_DIR / "selected_examples.json"),
        predictions=[
            prediction_archive(model, pack) for model in MODELS for pack in PACKS
        ],
        raw=[raw_pack(pack) for pack in PACKS],
        code="src/dreems/benchmark_gallery.py",
    output:
        gallery=directory(str(ROUTER_PUBLISH_DIR / "gallery")),
    threads: int(config.get("router_gallery_threads", 2))
    resources:
        mem_mb=int(config.get("router_gallery_mem_mb", 4000)),
        runtime=str(config.get("router_gallery_runtime", "30m")),
        slurm_partition=str(config.get("partition", "standard")),
        slurm_account="jonth0",
    params:
        prediction_pattern=lambda _wildcards: str(
            OUTDIR / "evaluation" / "{model}" / "pack_{pack}" / "predictions.npz"
        ),
    shell:
        """
        set -euo pipefail
        export MPLCONFIGDIR="${{TMPDIR:-/tmp}}/matplotlib-${{SLURM_JOB_ID:-local}}"
        export OMP_NUM_THREADS={threads}
        export OPENBLAS_NUM_THREADS={threads}
        export MKL_NUM_THREADS={threads}
        {UV_RUN} python -m dreems.benchmark_gallery router-gallery \
            --rows {input.rows:q} \
            --summary {input.summary:q} \
            --selections {input.selections:q} \
            --raw-pattern {RAW_PACK_PATTERN:q} \
            --prediction-pattern {params.prediction_pattern:q} \
            --pack-size {PACK_SIZE} \
            --threshold {ROUTER_THRESHOLD} \
            --output-dir {output.gallery:q}
        """


rule router_h2h_gallery:
    input:
        reference_gallery=str(ROUTER_H2H_REFERENCE_GALLERY),
        reference_rows=str(ROUTER_H2H_REFERENCE_ROWS),
        code="src/dreems/benchmark_gallery.py",
    output:
        gallery=directory(str(ROUTER_H2H_PUBLISH_DIR / "gallery")),
    threads: 1
    resources:
        mem_mb=int(config.get("router_h2h_gallery_mem_mb", 2000)),
        runtime=str(config.get("router_h2h_gallery_runtime", "20m")),
        slurm_partition=str(config.get("partition", "standard")),
        slurm_account="jonth0",
    shell:
        """
        set -euo pipefail
        {UV_RUN} python -m dreems.benchmark_gallery router-h2h-gallery \
            --reference-gallery {input.reference_gallery:q} \
            --reference-rows {input.reference_rows:q} \
            --threshold {ROUTER_THRESHOLD} \
            --output-dir {output.gallery:q}
        """
