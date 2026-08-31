"""Forward-model evidence audit for shrinkage toward uniform conductance."""

import os
import shlex
from pathlib import Path


PROJECT_ROOT = Path(workflow.basedir).parent.resolve()
OUTDIR = Path(str(config["outdir"])).expanduser()
PUBLISH_DIR = Path(str(config["publish_dir"])).expanduser()
PACKS = tuple(int(pack) for pack in config["pack_ids"])
EXACT = tuple(int(example) for example in config["exact_example_ids"])
SCRIPT = PROJECT_ROOT / "workflow" / "audit_forward_null_evidence.py"
HYBRID_SCRIPT = PROJECT_ROOT / "workflow" / "analyze_forward_null_hybrid.py"
CHECKPOINT = Path(str(config["checkpoint"])).expanduser()
UV_CACHE_DIR = str(config.get("uv_cache_dir") or os.environ.get("UV_CACHE_DIR", "")).strip()
UV_CACHE_PREFIX = f"UV_CACHE_DIR={shlex.quote(UV_CACHE_DIR)} " if UV_CACHE_DIR else ""
UV_RUN_ARGS = str(config.get("uv_run_args", "--frozen --no-sync"))
UV_RUN = f"{UV_CACHE_PREFIX}uv --project {shlex.quote(str(PROJECT_ROOT))} run {UV_RUN_ARGS}"


def configured_path(pattern, value, *, field):
    return str(pattern).format(**{field: value})


def pack_output(pack):
    return OUTDIR / "packs" / f"pack_{pack}.json"


def exact_output(example):
    return OUTDIR / "exact" / f"example_{example}.json"


localrules: all


rule all:
    input:
        str(PUBLISH_DIR / "forward_null_evidence_report.json"),
        str(PUBLISH_DIR / "forward_null_evidence_rows.csv"),
        str(PUBLISH_DIR / "forward_null_evidence_audit.pdf"),
        str(PUBLISH_DIR / "forward_null_evidence_audit.png"),
        str(PUBLISH_DIR / "guarded_null_surface_gate_report.json"),
        str(PUBLISH_DIR / "guarded_null_surface_gate_rows.csv"),
        str(PUBLISH_DIR / "guarded_null_surface_gate_audit.pdf"),
        str(PUBLISH_DIR / "guarded_null_surface_gate_audit.png"),
        str(PUBLISH_DIR / "absolute_fit_guarded_null_surface_gate_report.json"),
        str(PUBLISH_DIR / "absolute_fit_guarded_null_surface_gate_rows.csv"),
        str(PUBLISH_DIR / "absolute_fit_guarded_null_surface_gate_audit.pdf"),
        str(PUBLISH_DIR / "absolute_fit_guarded_null_surface_gate_audit.png"),


rule audit_pack:
    input:
        raw=lambda wildcards: configured_path(config["raw_pack_pattern"], wildcards.pack, field="pack"),
        predictions=lambda wildcards: configured_path(
            config["prediction_pattern"], wildcards.pack, field="pack"
        ),
        script=str(SCRIPT),
        refinement=str(PROJECT_ROOT / "src" / "dreems" / "refinement.py"),
    output:
        str(OUTDIR / "packs" / "pack_{pack}.json"),
    benchmark:
        str(OUTDIR / "benchmarks" / "packs" / "pack_{pack}.tsv")
    wildcard_constraints:
        pack="|".join(str(pack) for pack in PACKS),
    threads: 1
    resources:
        mem_mb=int(config.get("pack_mem_mb", 8000)),
        runtime=str(config.get("pack_runtime", "60m")),
        slurm_partition=str(config.get("partition", "standard")),
        slurm_account="jonth0",
    shell:
        """
        export JAX_PLATFORMS=cpu
        export CUDA_VISIBLE_DEVICES=""
        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        {UV_RUN} python {input.script:q} pack \
            --raw-pack {input.raw:q} \
            --predictions {input.predictions:q} \
            --pack-id {wildcards.pack} \
            --output {output:q}
        """


rule audit_exact:
    input:
        example=lambda wildcards: configured_path(
            config["exact_input_pattern"], wildcards.example, field="example"
        ),
        checkpoint=str(CHECKPOINT),
        script=str(SCRIPT),
        model=str(PROJECT_ROOT / "src" / "dreems" / "model.py"),
        inference=str(PROJECT_ROOT / "src" / "dreems" / "inference.py"),
        refinement=str(PROJECT_ROOT / "src" / "dreems" / "refinement.py"),
    output:
        str(OUTDIR / "exact" / "example_{example}.json"),
    benchmark:
        str(OUTDIR / "benchmarks" / "exact" / "example_{example}.tsv")
    wildcard_constraints:
        example="|".join(str(example) for example in EXACT),
    threads: 1
    resources:
        mem_mb=int(config.get("exact_mem_mb", 8000)),
        runtime=str(config.get("exact_runtime", "30m")),
        slurm_partition=str(config.get("partition", "standard")),
        slurm_account="jonth0",
    shell:
        """
        export JAX_PLATFORMS=cpu
        export CUDA_VISIBLE_DEVICES=""
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        {UV_RUN} python {input.script:q} exact \
            --input {input.example:q} \
            --checkpoint {input.checkpoint:q} \
            --example-id {wildcards.example} \
            --output {output:q}
        """


rule aggregate:
    input:
        packs=[str(pack_output(pack)) for pack in PACKS],
        exact=[str(exact_output(example)) for example in EXACT],
        selection=str(config["selection"]),
        script=str(SCRIPT),
    output:
        report=str(PUBLISH_DIR / "forward_null_evidence_report.json"),
        rows=str(PUBLISH_DIR / "forward_null_evidence_rows.csv"),
        pdf=str(PUBLISH_DIR / "forward_null_evidence_audit.pdf"),
        png=str(PUBLISH_DIR / "forward_null_evidence_audit.png"),
    benchmark:
        str(OUTDIR / "benchmarks" / "aggregate.tsv")
    threads: 1
    resources:
        mem_mb=int(config.get("aggregate_mem_mb", 8000)),
        runtime=str(config.get("aggregate_runtime", "30m")),
        slurm_partition=str(config.get("partition", "standard")),
        slurm_account="jonth0",
    shell:
        """
        export JAX_PLATFORMS=cpu
        export CUDA_VISIBLE_DEVICES=""
        {UV_RUN} python {input.script:q} aggregate \
            --packs {input.packs:q} \
            --exact {input.exact:q} \
            --selection {input.selection:q} \
            --output {output.report:q} \
            --rows-output {output.rows:q} \
            --figure {output.pdf:q}
        """


rule analyze_hybrid:
    input:
        forward_report=str(PUBLISH_DIR / "forward_null_evidence_report.json"),
        forward_rows=str(PUBLISH_DIR / "forward_null_evidence_rows.csv"),
        gate_rows=str(config["independent_gate_rows"]),
        baseline_exact=[
            configured_path(config["baseline_exact_pattern"], example, field="example")
            for example in EXACT
        ],
        gated_exact=[
            configured_path(config["gated_exact_pattern"], example, field="example")
            for example in EXACT
        ],
        script=str(HYBRID_SCRIPT),
    output:
        report=str(PUBLISH_DIR / "guarded_null_surface_gate_report.json"),
        rows=str(PUBLISH_DIR / "guarded_null_surface_gate_rows.csv"),
        pdf=str(PUBLISH_DIR / "guarded_null_surface_gate_audit.pdf"),
        png=str(PUBLISH_DIR / "guarded_null_surface_gate_audit.png"),
    benchmark:
        str(OUTDIR / "benchmarks" / "analyze_hybrid.tsv")
    threads: 1
    resources:
        mem_mb=int(config.get("aggregate_mem_mb", 8000)),
        runtime=str(config.get("aggregate_runtime", "30m")),
        slurm_partition=str(config.get("partition", "standard")),
        slurm_account="jonth0",
    shell:
        """
        export JAX_PLATFORMS=cpu
        export CUDA_VISIBLE_DEVICES=""
        {UV_RUN} python {input.script:q} \
            --forward-report {input.forward_report:q} \
            --forward-rows {input.forward_rows:q} \
            --gate-rows {input.gate_rows:q} \
            --baseline-exact {input.baseline_exact:q} \
            --gated-exact {input.gated_exact:q} \
            --output {output.report:q} \
            --rows-output {output.rows:q} \
            --figure {output.pdf:q}
        """


rule analyze_absolute_fit_guard:
    input:
        forward_report=str(PUBLISH_DIR / "forward_null_evidence_report.json"),
        forward_rows=str(PUBLISH_DIR / "forward_null_evidence_rows.csv"),
        gate_rows=str(config["independent_gate_rows"]),
        baseline_exact=[
            configured_path(config["baseline_exact_pattern"], example, field="example")
            for example in EXACT
        ],
        gated_exact=[
            configured_path(config["gated_exact_pattern"], example, field="example")
            for example in EXACT
        ],
        script=str(HYBRID_SCRIPT),
    output:
        report=str(PUBLISH_DIR / "absolute_fit_guarded_null_surface_gate_report.json"),
        rows=str(PUBLISH_DIR / "absolute_fit_guarded_null_surface_gate_rows.csv"),
        pdf=str(PUBLISH_DIR / "absolute_fit_guarded_null_surface_gate_audit.pdf"),
        png=str(PUBLISH_DIR / "absolute_fit_guarded_null_surface_gate_audit.png"),
    benchmark:
        str(OUTDIR / "benchmarks" / "analyze_absolute_fit_guard.tsv")
    params:
        null_quantile=float(config.get("absolute_fit_null_quantile", 0.05)),
    threads: 1
    resources:
        mem_mb=int(config.get("aggregate_mem_mb", 8000)),
        runtime=str(config.get("aggregate_runtime", "30m")),
        slurm_partition=str(config.get("partition", "standard")),
        slurm_account="jonth0",
    shell:
        """
        export JAX_PLATFORMS=cpu
        export CUDA_VISIBLE_DEVICES=""
        {UV_RUN} python {input.script:q} \
            --forward-report {input.forward_report:q} \
            --forward-rows {input.forward_rows:q} \
            --gate-rows {input.gate_rows:q} \
            --baseline-exact {input.baseline_exact:q} \
            --gated-exact {input.gated_exact:q} \
            --uniform-r2-null-quantile {params.null_quantile} \
            --output {output.report:q} \
            --rows-output {output.rows:q} \
            --figure {output.pdf:q}
        """
