"""Render archived real-data inputs with one explicit DREEMS checkpoint."""

from pathlib import Path

import yaml


PROJECT = Path(workflow.current_basedir).parent.resolve()
OUTDIR = Path(str(config["outdir"])).expanduser().resolve()
PUBLISH_DIR = Path(str(config["publish_dir"])).expanduser().resolve()
PREPARED_DIR = Path(str(config["prepared_dir"])).expanduser().resolve()
CHECKPOINT = Path(str(config["checkpoint"])).expanduser().resolve()
RENDER_CONFIG = Path(str(config["render_config"])).expanduser().resolve()
LITERATURE_REPORT = Path(str(config["literature_report"])).expanduser().resolve()
TOPO_CACHE = Path(str(config["topo_cache"])).expanduser().resolve()
RUNNER = PROJECT / "workflow/run_real_data.py"
RENDERER = PROJECT / "workflow/render_real_data_gallery.py"
GALLERY = PUBLISH_DIR / "gallery"
FITS = OUTDIR / "fits"
PROVENANCE = PUBLISH_DIR / "provenance"
DATASETS = [str(dataset) for dataset in config["datasets"]]
if not DATASETS or len(DATASETS) != len(set(DATASETS)):
    raise ValueError("datasets must be nonempty and unique")

RENDER_SETTINGS = yaml.safe_load(RENDER_CONFIG.read_text())
PUBLISHED_SOURCES = [
    figure["path"]
    for dataset in DATASETS
    for figure in RENDER_SETTINGS["datasets"].get(dataset, {}).get(
        "published_figures", []
    )
]
INFERENCE_CODE = [str(PROJECT / "src/dreems" / name) for name in (
    "inference/api.py", "neural/checkpoints.py", "neural/model.py", "neural/features.py", "neural/linearized.py",
    "inference/refinement.py", "inference/calibration.py", "utils/validation.py", "evaluation/metrics.py", "utils/workflow_io.py",
)]
RENDER_CODE = [str(PROJECT / "src/dreems" / name) for name in ("utils/validation.py", "utils/workflow_io.py")]
PROVENANCE_CODE = [str(PROJECT / "src/dreems" / name) for name in ("neural/checkpoints.py", "utils/workflow_io.py")]
ENVIRONMENT = [str(PROJECT / "pyproject.toml"), str(PROJECT / "uv.lock")]


def prepared_path(dataset, suffix):
    return str(PREPARED_DIR / f"{dataset}.{suffix}")


def grid_side(wildcards):
    return RENDER_SETTINGS["datasets"].get(wildcards.dataset, {}).get(
        "grid_side", RENDER_SETTINGS["grid_side"]
    )


def topo_zoom(wildcards):
    return RENDER_SETTINGS["datasets"].get(wildcards.dataset, {}).get(
        "topo_zoom", RENDER_SETTINGS["topo_zoom"]
    )


localrules: all, checkpoint_provenance, stage_workflow_provenance


rule all:
    input:
        str(GALLERY / "index.html"),
        str(GALLERY / "manifest.csv"),
        str(GALLERY / "manifest.json"),
        str(GALLERY / "overview.png"),
        str(GALLERY / "checkpoint_provenance.json"),
        str(PROVENANCE / "real_data_gallery.Snakefile"),
        str(PROVENANCE / "run_real_data.py"),
        str(PROVENANCE / "render_real_data_gallery.py"),
        str(PROVENANCE / "render_config.yaml"),
        str(PROVENANCE / "pyproject.toml"),
        str(PROVENANCE / "uv.lock"),
        expand(str(GALLERY / "{dataset}.png"), dataset=DATASETS),
        expand(str(GALLERY / "{dataset}.pdf"), dataset=DATASETS),


rule run_dreems:
    input:
        prepared=lambda wildcards: prepared_path(wildcards.dataset, "npz"),
        metadata=lambda wildcards: prepared_path(wildcards.dataset, "json"),
        checkpoint=str(CHECKPOINT),
        runner=str(RUNNER),
        package=INFERENCE_CODE,
        environment=ENVIRONMENT,
    output:
        prediction=str(FITS / "{dataset}.npz"),
        metadata=str(FITS / "{dataset}.json"),
    log:
        str(OUTDIR / "logs/inference/{dataset}.log")
    benchmark:
        str(PUBLISH_DIR / "benchmarks/inference/{dataset}.tsv")
    threads: int(config.get("inference_threads", 4))
    params:
        project=str(PROJECT),
    resources:
        mem_mb=int(config.get("inference_mem_mb", 16000)),
        runtime=str(config.get("inference_runtime", "120m")),
        slurm_partition=str(config.get("partition", "standard")),
        slurm_account="jonth0",
    shell:
        """
        set -euo pipefail
        export JAX_PLATFORMS=cpu
        export JAX_PLATFORM_NAME=cpu
        export CUDA_VISIBLE_DEVICES=""
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        export OMP_NUM_THREADS={threads}
        export OPENBLAS_NUM_THREADS={threads}
        export MKL_NUM_THREADS={threads}
        export NUMEXPR_NUM_THREADS={threads}
        uv --project {params.project:q} run --frozen --no-sync python {input.runner:q} predict \
            --prepared-npz {input.prepared:q} \
            --prepared-json {input.metadata:q} \
            --checkpoint {input.checkpoint:q} \
            --output-npz {output.prediction:q} \
            --output-json {output.metadata:q} \
            > {log:q} 2>&1
        """


rule render_case:
    input:
        prepared=lambda wildcards: prepared_path(wildcards.dataset, "npz"),
        prepared_json=lambda wildcards: prepared_path(wildcards.dataset, "json"),
        dreems=str(FITS / "{dataset}.npz"),
        dreems_json=str(FITS / "{dataset}.json"),
        renderer=str(RENDERER),
        code=RENDER_CODE,
        environment=ENVIRONMENT,
    output:
        png=str(GALLERY / "{dataset}.png"),
        pdf=str(GALLERY / "{dataset}.pdf"),
        row=str(GALLERY / "cases/{dataset}.json"),
    log:
        str(OUTDIR / "logs/render/{dataset}.log")
    benchmark:
        str(PUBLISH_DIR / "benchmarks/render/{dataset}.tsv")
    params:
        project=str(PROJECT),
        grid_side=grid_side,
        topo_zoom=topo_zoom,
        topo_max_tiles=RENDER_SETTINGS["topo_max_tiles"],
        topo_max_pixels=RENDER_SETTINGS["topo_max_pixels"],
        dpi=RENDER_SETTINGS["dpi"],
        topo_cache=str(TOPO_CACHE),
    threads: 1
    resources:
        mem_mb=int(config.get("render_mem_mb", 12000)),
        runtime=str(config.get("render_runtime", "90m")),
        slurm_partition=str(config.get("partition", "standard")),
        slurm_account="jonth0",
    shell:
        """
        set -euo pipefail
        export MPLCONFIGDIR="${{TMPDIR:-/tmp}}/matplotlib-${{SLURM_JOB_ID:-local}}"
        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1
        uv --project {params.project:q} run --frozen --no-sync python {input.renderer:q} case \
            --prepared-npz {input.prepared:q} \
            --prepared-json {input.prepared_json:q} \
            --dreems-npz {input.dreems:q} \
            --dreems-json {input.dreems_json:q} \
            --topo-cache {params.topo_cache:q} \
            --grid-side {params.grid_side} \
            --topo-zoom {params.topo_zoom} \
            --topo-max-tiles {params.topo_max_tiles} \
            --topo-max-pixels {params.topo_max_pixels} \
            --dpi {params.dpi} \
            --output-png {output.png:q} \
            --output-pdf {output.pdf:q} \
            --output-json {output.row:q} \
            > {log:q} 2>&1
        """


rule assemble_gallery:
    input:
        rows=expand(str(GALLERY / "cases/{dataset}.json"), dataset=DATASETS),
        figures=expand(str(GALLERY / "{dataset}.png"), dataset=DATASETS),
        published=PUBLISHED_SOURCES,
        renderer=str(RENDERER),
        code=RENDER_CODE,
        render_config=str(RENDER_CONFIG),
        literature_report=str(LITERATURE_REPORT),
        environment=ENVIRONMENT,
    output:
        index=str(GALLERY / "index.html"),
        manifest_csv=str(GALLERY / "manifest.csv"),
        manifest_json=str(GALLERY / "manifest.json"),
        overview=str(GALLERY / "overview.png"),
        report=str(GALLERY / "literature_candidates.md"),
        published=directory(str(GALLERY / "published")),
    log:
        str(OUTDIR / "logs/assemble.log")
    benchmark:
        str(PUBLISH_DIR / "benchmarks/assemble.tsv")
    threads: 1
    params:
        project=str(PROJECT),
        gallery=str(GALLERY),
    resources:
        mem_mb=int(config.get("assemble_mem_mb", 8000)),
        runtime=str(config.get("assemble_runtime", "60m")),
        slurm_partition=str(config.get("partition", "standard")),
        slurm_account="jonth0",
    shell:
        """
        set -euo pipefail
        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1
        uv --project {params.project:q} run --frozen --no-sync python {input.renderer:q} index \
            --config {input.render_config:q} \
            --row-json {input.rows:q} \
            --output-dir {params.gallery:q} \
            --output-index {output.index:q} \
            --output-manifest-csv {output.manifest_csv:q} \
            --output-manifest-json {output.manifest_json:q} \
            --output-overview {output.overview:q} \
            --literature-report {input.literature_report:q} \
            --output-literature-report {output.report:q} \
            > {log:q} 2>&1
        """


rule checkpoint_provenance:
    input:
        checkpoint=str(CHECKPOINT),
        runner=str(RUNNER),
        code=PROVENANCE_CODE,
        environment=ENVIRONMENT,
    output:
        str(GALLERY / "checkpoint_provenance.json")
    params:
        project=str(PROJECT),
    shell:
        """
        uv --project {params.project:q} run --frozen --no-sync python {input.runner:q} provenance \
            --checkpoint {input.checkpoint:q} \
            --output {output:q}
        """


rule stage_workflow_provenance:
    input:
        snakefile=str(PROJECT / "workflow/real_data_gallery.Snakefile"),
        runner=str(RUNNER),
        renderer=str(RENDERER),
        render_config=str(RENDER_CONFIG),
        pyproject=str(PROJECT / "pyproject.toml"),
        lock=str(PROJECT / "uv.lock"),
    output:
        snakefile=str(PROVENANCE / "real_data_gallery.Snakefile"),
        runner=str(PROVENANCE / "run_real_data.py"),
        renderer=str(PROVENANCE / "render_real_data_gallery.py"),
        render_config=str(PROVENANCE / "render_config.yaml"),
        pyproject=str(PROVENANCE / "pyproject.toml"),
        lock=str(PROVENANCE / "uv.lock"),
    params:
        directory=str(PROVENANCE),
    shell:
        """
        mkdir -p {params.directory:q}
        cp {input.snakefile:q} {output.snakefile:q}
        cp {input.runner:q} {output.runner:q}
        cp {input.renderer:q} {output.renderer:q}
        cp {input.render_config:q} {output.render_config:q}
        cp {input.pyproject:q} {output.pyproject:q}
        cp {input.lock:q} {output.lock:q}
        """
