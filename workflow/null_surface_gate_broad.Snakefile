"""Broad DREEMS-only verification of the learned null-surface gate."""

import os
import shlex
from pathlib import Path


PROJECT_ROOT = Path(workflow.basedir).parent.resolve()
OUTDIR = Path(str(config["outdir"])).expanduser()
PUBLISH_DIR = Path(str(config["publish_dir"])).expanduser()
CHECKPOINT = Path(str(config["checkpoint"])).expanduser()
EXAMPLES = range(int(config["example_count"]))
UV_CACHE_DIR = str(config.get("uv_cache_dir") or os.environ.get("UV_CACHE_DIR", "")).strip()
UV_CACHE_PREFIX = f"UV_CACHE_DIR={shlex.quote(UV_CACHE_DIR)} " if UV_CACHE_DIR else ""
UV_RUN_ARGS = str(config.get("uv_run_args", "--frozen --no-sync"))
UV_RUN = f"{UV_CACHE_PREFIX}uv --project {shlex.quote(str(PROJECT_ROOT))} run {UV_RUN_ARGS}"


def source_paths(*names):
    return [str(PROJECT_ROOT / "src" / "dreems" / name) for name in names]


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
ANALYSIS_CODE = str(PROJECT_ROOT / "workflow" / "analyze_surface_gate_broad.py")


def configured_path(pattern, example):
    return str(pattern).format(example=example)


def gated_output(example, suffix):
    return OUTDIR / "dreems" / f"example_{example}.{suffix}"


localrules: all


rule all:
    input:
        str(PUBLISH_DIR / "broad_report.json"),
        str(PUBLISH_DIR / "broad_rows.csv"),
        str(PUBLISH_DIR / "broad_surface_gate_audit.pdf"),
        str(PUBLISH_DIR / "broad_surface_gate_audit.png"),


rule evaluate_broad_case:
    input:
        checkpoint=str(CHECKPOINT),
        example=lambda wildcards: configured_path(config["input_pattern"], wildcards.example),
        metadata=lambda wildcards: configured_path(
            config["metadata_pattern"], wildcards.example
        ),
        code=EVAL_CODE,
    output:
        json=str(OUTDIR / "dreems" / "example_{example}.json"),
        npz=str(OUTDIR / "dreems" / "example_{example}.npz"),
    benchmark:
        str(OUTDIR / "benchmarks" / "evaluate" / "example_{example}.tsv")
    wildcard_constraints:
        example=r"\d+",
    threads: int(config.get("eval_threads", 1))
    resources:
        mem_mb=int(config.get("eval_mem_mb", 6000)),
        runtime=str(config.get("eval_runtime", "30m")),
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


rule analyze_broad_surface_gate:
    input:
        gated_json=[str(gated_output(example, "json")) for example in EXAMPLES],
        gated_npz=[str(gated_output(example, "npz")) for example in EXAMPLES],
        baseline_json=[
            configured_path(config["baseline_json_pattern"], example) for example in EXAMPLES
        ],
        baseline_npz=[
            configured_path(config["baseline_npz_pattern"], example) for example in EXAMPLES
        ],
        feems_json=[
            configured_path(config["feems_json_pattern"], example) for example in EXAMPLES
        ],
        code=ANALYSIS_CODE,
    output:
        report=str(PUBLISH_DIR / "broad_report.json"),
        rows=str(PUBLISH_DIR / "broad_rows.csv"),
        figure_pdf=str(PUBLISH_DIR / "broad_surface_gate_audit.pdf"),
        figure_png=str(PUBLISH_DIR / "broad_surface_gate_audit.png"),
    benchmark:
        str(OUTDIR / "benchmarks" / "analyze.tsv")
    threads: int(config.get("analysis_threads", 1))
    resources:
        mem_mb=int(config.get("analysis_mem_mb", 8000)),
        runtime=str(config.get("analysis_runtime", "30m")),
        slurm_partition=str(config.get("analysis_partition", "standard")),
    params:
        example_count=len(EXAMPLES),
        gated_pattern=lambda _wildcards: str(OUTDIR / "dreems" / "example_{example}"),
        baseline_json_pattern=lambda _wildcards: str(config["baseline_json_pattern"]),
        baseline_npz_pattern=lambda _wildcards: str(config["baseline_npz_pattern"]),
        feems_json_pattern=lambda _wildcards: str(config["feems_json_pattern"]),
    shell:
        """
        export JAX_PLATFORMS=cpu
        export CUDA_VISIBLE_DEVICES=""
        {UV_RUN} python {input.code:q} \
            --example-count {params.example_count} \
            --gated-pattern {params.gated_pattern:q} \
            --baseline-json-pattern {params.baseline_json_pattern:q} \
            --baseline-npz-pattern {params.baseline_npz_pattern:q} \
            --feems-json-pattern {params.feems_json_pattern:q} \
            --output {output.report:q} \
            --rows-output {output.rows:q} \
            --figure {output.figure_pdf:q}
        """
