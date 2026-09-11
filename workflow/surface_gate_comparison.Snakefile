"""Build matched core, gated, and FEEMS comparison artifacts."""

import os
import shlex
from pathlib import Path


PROJECT_ROOT = Path(workflow.current_basedir).parent.resolve()
OUTDIR = Path(str(config["outdir"])).expanduser().resolve()
PUBLISH_DIR = Path(str(config["publish_dir"])).expanduser().resolve()
CORE_CHECKPOINT = Path(str(config["core_checkpoint"])).expanduser().resolve()
GATED_CHECKPOINT = Path(str(config["gated_checkpoint"])).expanduser().resolve()
CORE_SCENARIO_BUNDLE = Path(str(config["core_scenario_bundle"])).expanduser().resolve()
EXAMPLES = range(int(config["example_count"]))
COMPARISON_CODE = PROJECT_ROOT / "src" / "dreems" / "evaluation/surface_gate_comparison.py"
WORKFLOW_IO_CODE = PROJECT_ROOT / "src" / "dreems" / "utils/workflow_io.py"
SUPPORT_CODE = [str(WORKFLOW_IO_CODE), str(PROJECT_ROOT / "src/dreems/utils/validation.py")]
ENVIRONMENT = [str(PROJECT_ROOT / name) for name in ("pyproject.toml", "uv.lock")]
NOTEBOOK_SOURCE = PROJECT_ROOT / "workflow" / "notebooks" / "feems_comparison_surface_gate.md"
UV_CACHE_DIR = str(config.get("uv_cache_dir") or os.environ.get("UV_CACHE_DIR", "")).strip()
UV_CACHE_PREFIX = f"UV_CACHE_DIR={shlex.quote(UV_CACHE_DIR)} " if UV_CACHE_DIR else ""
UV_RUN_ARGS = str(config.get("uv_run_args", "--frozen --no-sync"))
UV_RUN = f"{UV_CACHE_PREFIX}uv --project {shlex.quote(str(PROJECT_ROOT))} run {UV_RUN_ARGS}"

SCENARIO_BUNDLE = PUBLISH_DIR / "scenario_comparison.npz"
SCENARIO_ROWS = PUBLISH_DIR / "scenario_rows.csv"
SCENARIO_REPORT = PUBLISH_DIR / "scenario_report.json"
SCENARIO_GALLERY = PUBLISH_DIR / "scenario_gallery"
BROAD_GALLERY = PUBLISH_DIR / "broad_gallery"
PREPARED_NOTEBOOK = OUTDIR / "notebook" / "feems_comparison_surface_gate.prepared.ipynb"
EXECUTED_NOTEBOOK = PUBLISH_DIR / "feems_comparison_surface_gate.ipynb"
NOTEBOOK_HTML = PUBLISH_DIR / "feems_comparison_surface_gate.html"


def configured_path(pattern, example):
    """Resolve a case within a saved result set without reordering its IDs."""
    return str(pattern).format(example=example)


GATED_SCENARIOS = [
    configured_path(config["gated_scenario_pattern"], example) for example in range(12)
]
CORE_BROAD = [configured_path(config["core_broad_pattern"], example) for example in EXAMPLES]
GATED_BROAD = [configured_path(config["gated_broad_pattern"], example) for example in EXAMPLES]
FEEMS_BROAD = [configured_path(config["feems_broad_pattern"], example) for example in EXAMPLES]


localrules: all, prepare_notebook


rule all:
    input:
        str(SCENARIO_BUNDLE),
        str(SCENARIO_ROWS),
        str(SCENARIO_REPORT),
        str(SCENARIO_GALLERY),
        str(BROAD_GALLERY),
        str(EXECUTED_NOTEBOOK),
        str(NOTEBOOK_HTML),


rule build_scenario_bundle:
    input:
        core=str(CORE_SCENARIO_BUNDLE),
        gated=GATED_SCENARIOS,
        checkpoint=str(GATED_CHECKPOINT),
        code=str(COMPARISON_CODE),
        support=SUPPORT_CODE,
        environment=ENVIRONMENT,
    output:
        bundle=str(SCENARIO_BUNDLE),
        rows=str(SCENARIO_ROWS),
        report=str(SCENARIO_REPORT),
    benchmark:
        str(OUTDIR / "benchmarks" / "scenario_bundle.tsv")
    threads: int(config.get("analysis_threads", 2))
    resources:
        mem_mb=int(config.get("analysis_mem_mb", 12000)),
        runtime=str(config.get("analysis_runtime", "90m")),
        slurm_partition=str(config.get("analysis_partition", "standard")),
        slurm_account="jonth0",
    params:
        uv=UV_RUN,
        gated_pattern=lambda _wildcards: str(config["gated_scenario_pattern"]),
    shell:
        """
        set -euo pipefail
        {params.uv} python -m dreems.evaluation.surface_gate_comparison scenario-bundle \
            --core-bundle {input.core:q} \
            --gated-pattern {params.gated_pattern:q} \
            --gated-checkpoint {input.checkpoint:q} \
            --output {output.bundle:q} \
            --rows-output {output.rows:q} \
            --report-output {output.report:q}
        """


rule render_scenario_gallery:
    input:
        bundle=str(SCENARIO_BUNDLE),
        report=str(SCENARIO_REPORT),
        code=str(COMPARISON_CODE),
        support=SUPPORT_CODE,
        environment=ENVIRONMENT,
    output:
        directory(str(SCENARIO_GALLERY)),
    benchmark:
        str(OUTDIR / "benchmarks" / "scenario_gallery.tsv")
    params:
        uv=UV_RUN,
    threads: int(config.get("analysis_threads", 2))
    resources:
        mem_mb=int(config.get("analysis_mem_mb", 12000)),
        runtime=str(config.get("analysis_runtime", "90m")),
        slurm_partition=str(config.get("analysis_partition", "standard")),
        slurm_account="jonth0",
    shell:
        """
        set -euo pipefail
        {params.uv} python -m dreems.evaluation.surface_gate_comparison scenario-gallery \
            --bundle {input.bundle:q} \
            --report {input.report:q} \
            --output-dir {output:q}
        """


rule render_broad_gallery:
    input:
        rows=str(config["broad_rows"]),
        report=str(config["broad_report"]),
        core=CORE_BROAD,
        gated=GATED_BROAD,
        feems=FEEMS_BROAD,
        core_checkpoint=str(CORE_CHECKPOINT),
        gated_checkpoint=str(GATED_CHECKPOINT),
        code=str(COMPARISON_CODE),
        support=SUPPORT_CODE,
        environment=ENVIRONMENT,
    output:
        directory(str(BROAD_GALLERY)),
    benchmark:
        str(OUTDIR / "benchmarks" / "broad_gallery.tsv")
    threads: int(config.get("analysis_threads", 2))
    resources:
        mem_mb=int(config.get("analysis_mem_mb", 12000)),
        runtime=str(config.get("analysis_runtime", "90m")),
        slurm_partition=str(config.get("analysis_partition", "standard")),
        slurm_account="jonth0",
    params:
        uv=UV_RUN,
        core_pattern=lambda _wildcards: str(config["core_broad_pattern"]),
        gated_pattern=lambda _wildcards: str(config["gated_broad_pattern"]),
        feems_pattern=lambda _wildcards: str(config["feems_broad_pattern"]),
    shell:
        """
        set -euo pipefail
        {params.uv} python -m dreems.evaluation.surface_gate_comparison broad-gallery \
            --rows {input.rows:q} \
            --report {input.report:q} \
            --core-pattern {params.core_pattern:q} \
            --gated-pattern {params.gated_pattern:q} \
            --feems-pattern {params.feems_pattern:q} \
            --core-checkpoint {input.core_checkpoint:q} \
            --gated-checkpoint {input.gated_checkpoint:q} \
            --output-dir {output:q}
        """


rule prepare_notebook:
    input:
        source=str(NOTEBOOK_SOURCE),
        bundle=str(SCENARIO_BUNDLE),
        workflow_io=str(WORKFLOW_IO_CODE),
        environment=ENVIRONMENT,
    output:
        temp(str(PREPARED_NOTEBOOK)),
    benchmark:
        str(OUTDIR / "benchmarks" / "prepare_notebook.tsv")
    threads: 1
    resources:
        mem_mb=4000,
        runtime="20m",
        slurm_partition=str(config.get("notebook_partition", "standard")),
        slurm_account="jonth0",
    run:
        import jupytext
        from dreems.utils.workflow_io import _atomic_output

        notebook = jupytext.read(input.source)
        placeholder = 'Path("__SCENARIO_BUNDLE__")'
        replacements = 0
        for cell in notebook.cells:
            if cell.cell_type == "code":
                replacements += cell.source.count(placeholder)
                cell.source = cell.source.replace(placeholder, f"Path({str(input.bundle)!r})")
        if replacements != 1:
            raise ValueError("expected one scenario-bundle placeholder in the notebook")
        with _atomic_output(Path(output[0])) as stream:
            stream.write(jupytext.writes(notebook, fmt="ipynb").encode("utf-8"))


rule execute_notebook:
    input:
        notebook=str(PREPARED_NOTEBOOK),
        bundle=str(SCENARIO_BUNDLE),
        code=str(COMPARISON_CODE),
        support=SUPPORT_CODE,
        environment=ENVIRONMENT,
    output:
        str(EXECUTED_NOTEBOOK),
    benchmark:
        str(OUTDIR / "benchmarks" / "execute_notebook.tsv")
    params:
        uv=UV_RUN,
    threads: int(config.get("notebook_threads", 2))
    resources:
        mem_mb=int(config.get("notebook_mem_mb", 8000)),
        runtime=str(config.get("notebook_runtime", "30m")),
        slurm_partition=str(config.get("notebook_partition", "standard")),
        slurm_account="jonth0",
    shell:
        """
        set -euo pipefail
        export JAX_PLATFORMS=cpu
        export CUDA_VISIBLE_DEVICES=""
        mkdir -p "$(dirname {output:q})"
        {params.uv} jupyter nbconvert \
            --to notebook \
            --execute \
            --ExecutePreprocessor.timeout=-1 \
            --output {output:q} \
            {input.notebook:q}
        """


rule render_notebook_html:
    input:
        notebook=str(EXECUTED_NOTEBOOK),
        environment=ENVIRONMENT,
    output:
        str(NOTEBOOK_HTML),
    benchmark:
        str(OUTDIR / "benchmarks" / "notebook_html.tsv")
    params:
        uv=UV_RUN,
    threads: 1
    resources:
        mem_mb=4000,
        runtime="20m",
        slurm_partition=str(config.get("notebook_partition", "standard")),
        slurm_account="jonth0",
    shell:
        """
        set -euo pipefail
        {params.uv} jupyter nbconvert --to html --output {output:q} {input.notebook:q}
        """
