import os
import shlex
from pathlib import Path


PROJECT_ROOT = Path(workflow.basedir).parent.resolve()
OUTDIR = Path(str(config["outdir"])).expanduser()
PUBLISH_DIR = Path(str(config["publish_dir"])).expanduser()
SOURCE_RUN = Path(str(config["source_run"])).expanduser()
BASE_CHECKPOINT = Path(str(config["base_checkpoint"])).expanduser()
CONFIG_PATH = PROJECT_ROOT / "workflow" / "config.null_surface_gate.yaml"
UV_CACHE_DIR = str(config.get("uv_cache_dir") or os.environ.get("UV_CACHE_DIR", "")).strip()
UV_CACHE_PREFIX = f"UV_CACHE_DIR={shlex.quote(UV_CACHE_DIR)} " if UV_CACHE_DIR else ""
UV_RUN_ARGS = str(config.get("uv_run_args", "--frozen --no-sync"))
UV_RUN = f"{UV_CACHE_PREFIX}uv --project {shlex.quote(str(PROJECT_ROOT))} run {UV_RUN_ARGS}"


def source_paths(*names):
    return [str(PROJECT_ROOT / "src" / "dreems" / name) for name in names]


TRAIN_CODE = source_paths(
    "checkpoints.py",
    "features.py",
    "linearized.py",
    "losses.py",
    "model.py",
    "pipeline.py",
    "train.py",
    "workflow_io.py",
)
EVAL_CODE = source_paths(
    "calibration.py",
    "checkpoints.py",
    "features.py",
    "h2h.py",
    "inference.py",
    "metrics.py",
    "model.py",
    "pipeline.py",
    "refinement.py",
    "workflow_io.py",
)
ANALYSIS_CODE = str(PROJECT_ROOT / "workflow" / "analyze_surface_gate.py")
TRAIN_MANIFEST = SOURCE_RUN / "manifests" / "train.txt"
VALIDATION_MANIFEST = SOURCE_RUN / "manifests" / "validation.txt"
NORMALIZATION = SOURCE_RUN / "base_normalization.json"


def paths_from_manifest(path):
    return [Path(line.strip()) for line in path.read_text().splitlines() if line.strip()]


TRAIN_PACKS = paths_from_manifest(TRAIN_MANIFEST)
VALIDATION_PACKS = paths_from_manifest(VALIDATION_MANIFEST)
VALIDATION_COUNTERFACTUAL_PACKS = [
    SOURCE_RUN / "prepared" / "counterfactual" / "validation" / f"pack_{pack}.pkl"
    for pack in range(8)
]
EXACT_EXAMPLES = range(12)
MODEL_DIR = OUTDIR / "model"
CHECKPOINT = MODEL_DIR / "checkpoint.dreems.zip"
BEST_CHECKPOINT = MODEL_DIR / "checkpoint.best.dreems.zip"
TRAINING_SUMMARY = MODEL_DIR / "training_summary.json"
PUBLISHED_CHECKPOINT = PUBLISH_DIR / "dreems_null_surface_gate.dreems.zip"


def exact_path(pattern, example):
    return Path(str(pattern).format(example=example))


def validation_output(pack, name):
    return OUTDIR / "evaluation" / "validation" / f"pack_{pack}" / name


def exact_output(example, suffix):
    return OUTDIR / "evaluation" / "exact" / f"example_{example}.{suffix}"


localrules: all, publish_checkpoint


rule all:
    input:
        str(PUBLISHED_CHECKPOINT),
        str(PUBLISH_DIR / "report.json"),
        str(PUBLISH_DIR / "rows.csv"),
        str(PUBLISH_DIR / "null_surface_gate_audit.pdf"),
        str(PUBLISH_DIR / "null_surface_gate_audit.png"),


rule train_surface_gate:
    input:
        normalization=str(NORMALIZATION),
        train_manifest=str(TRAIN_MANIFEST),
        validation_manifest=str(VALIDATION_MANIFEST),
        train_packs=[str(path) for path in TRAIN_PACKS],
        validation_packs=[str(path) for path in VALIDATION_PACKS],
        config=str(CONFIG_PATH),
        init=str(BASE_CHECKPOINT),
        code=TRAIN_CODE,
    output:
        checkpoint=str(CHECKPOINT),
        best=str(BEST_CHECKPOINT),
        best_summary=str(MODEL_DIR / "checkpoint.best.summary.json"),
        summary=str(TRAINING_SUMMARY),
    benchmark:
        str(OUTDIR / "benchmarks" / "train_surface_gate.tsv")
    threads: int(config.get("train_threads", 4))
    resources:
        mem_mb=int(config.get("train_mem_mb", 32000)),
        runtime=str(config.get("train_runtime", "12h")),
        gres="gpu:1",
        slurm_partition=str(config.get("train_partition", "gpu_mig40,spgpu,gpu")),
    shell:
        """
        export JAX_PLATFORMS=cuda
        export XLA_PYTHON_CLIENT_MEM_FRACTION={config[xla_mem_fraction]}
        CACHE_DIR="${{TMPDIR:-/tmp}}/dreems-null-gate-${{SLURM_JOB_ID}}"
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

rule publish_checkpoint:
    input:
        str(BEST_CHECKPOINT),
    output:
        str(PUBLISHED_CHECKPOINT),
    shell:
        """
        mkdir -p "$(dirname {output:q})"
        cp {input:q} {output:q}.tmp
        mv {output:q}.tmp {output:q}
        """


rule evaluate_validation_pack:
    input:
        checkpoint=str(BEST_CHECKPOINT),
        prepared=lambda wildcards: str(VALIDATION_COUNTERFACTUAL_PACKS[int(wildcards.pack)]),
        code=EVAL_CODE,
    output:
        summary=str(OUTDIR / "evaluation" / "validation" / "pack_{pack}" / "summary.json"),
        csv=str(OUTDIR / "evaluation" / "validation" / "pack_{pack}" / "summary.csv"),
        predictions=str(
            OUTDIR / "evaluation" / "validation" / "pack_{pack}" / "predictions.npz"
        ),
    benchmark:
        str(OUTDIR / "benchmarks" / "evaluate_validation" / "pack_{pack}.tsv")
    wildcard_constraints:
        pack="[0-7]",
    threads: int(config.get("eval_threads", 1))
    resources:
        mem_mb=int(config.get("eval_mem_mb", 8000)),
        runtime=str(config.get("eval_runtime", "60m")),
        slurm_partition=str(config.get("eval_partition", "standard")),
    params:
        output_dir=lambda wildcards: str(
            OUTDIR / "evaluation" / "validation" / f"pack_{wildcards.pack}"
        ),
        replicate_offset=lambda wildcards: 64 * int(wildcards.pack),
    shell:
        """
        export JAX_PLATFORMS=cpu
        export CUDA_VISIBLE_DEVICES=""
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        {UV_RUN} python -m dreems.pipeline evaluate \
            --checkpoint {input.checkpoint:q} \
            --inputs {input.prepared:q} \
            --output-dir {params.output_dir:q} \
            --batch-size 4 \
            --replicate-offset {params.replicate_offset} \
            --predictions-output {output.predictions:q}
        """


rule evaluate_exact_case:
    input:
        checkpoint=str(BEST_CHECKPOINT),
        example=lambda wildcards: str(
            exact_path(config["exact_input_pattern"], wildcards.example)
        ),
        metadata=lambda wildcards: str(
            exact_path(config["exact_metadata_pattern"], wildcards.example)
        ),
        code=EVAL_CODE,
    output:
        json=str(OUTDIR / "evaluation" / "exact" / "example_{example}.json"),
        npz=str(OUTDIR / "evaluation" / "exact" / "example_{example}.npz"),
    benchmark:
        str(OUTDIR / "benchmarks" / "evaluate_exact" / "example_{example}.tsv")
    wildcard_constraints:
        example=r"\d+",
    threads: int(config.get("eval_threads", 1))
    resources:
        mem_mb=int(config.get("eval_mem_mb", 8000)),
        runtime=str(config.get("eval_runtime", "60m")),
        slurm_partition=str(config.get("eval_partition", "standard")),
    shell:
        """
        export JAX_PLATFORMS=cpu
        export CUDA_VISIBLE_DEVICES=""
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        {UV_RUN} python -m dreems.h2h dreems-case \
            --input {input.example:q} \
            --index {wildcards.example} \
            --checkpoint {input.checkpoint:q} \
            --output-json {output.json:q} \
            --output-npz {output.npz:q}
        """


rule analyze_surface_gate:
    input:
        training_summary=str(TRAINING_SUMMARY),
        selection=str(SOURCE_RUN / "selection" / "validation.json"),
        baseline_validation=[
            str(config["validation_baseline_pattern"]).format(pack=pack) for pack in range(8)
        ],
        gated_validation=[
            str(validation_output(pack, "predictions.npz")) for pack in range(8)
        ],
        baseline_exact=[
            str(config["exact_baseline_pattern"]).format(example=example)
            for example in EXACT_EXAMPLES
        ],
        gated_exact=[str(exact_output(example, "npz")) for example in EXACT_EXAMPLES],
        gated_exact_json=[str(exact_output(example, "json")) for example in EXACT_EXAMPLES],
        code=ANALYSIS_CODE,
    output:
        report=str(PUBLISH_DIR / "report.json"),
        rows=str(PUBLISH_DIR / "rows.csv"),
        figure_pdf=str(PUBLISH_DIR / "null_surface_gate_audit.pdf"),
        figure_png=str(PUBLISH_DIR / "null_surface_gate_audit.png"),
    benchmark:
        str(OUTDIR / "benchmarks" / "analyze.tsv")
    threads: int(config.get("analysis_threads", 1))
    resources:
        mem_mb=int(config.get("analysis_mem_mb", 8000)),
        runtime=str(config.get("analysis_runtime", "30m")),
        slurm_partition=str(config.get("analysis_partition", "standard")),
    shell:
        """
        export JAX_PLATFORMS=cpu
        export CUDA_VISIBLE_DEVICES=""
        {UV_RUN} python {input.code:q} \
            --training-summary {input.training_summary:q} \
            --selection {input.selection:q} \
            --baseline-validation {input.baseline_validation:q} \
            --gated-validation {input.gated_validation:q} \
            --baseline-exact {input.baseline_exact:q} \
            --gated-exact {input.gated_exact:q} \
            --gated-exact-json {input.gated_exact_json:q} \
            --output {output.report:q} \
            --rows-output {output.rows:q} \
            --figure {output.figure_pdf:q}
        """
