import os
import shlex
from pathlib import Path


PROJECT_ROOT = Path(workflow.basedir).parent.resolve()
OUTDIR = Path(str(config["outdir"])).expanduser()
PUBLISH_DIR = Path(str(config["publish_dir"])).expanduser()
CHECKPOINT = Path(str(config["checkpoint"])).expanduser()
PREPARED_DIR = Path(str(config["counterfactual_prepared_dir"])).expanduser()
SELECTION = Path(str(config["counterfactual_selection"])).expanduser()
EXACT_NOTEBOOK = Path(str(config["exact_notebook_edges"])).expanduser()
BROAD_DIR = Path(str(config["broad_h2h_dreems_dir"])).expanduser()
PACKS = tuple(int(value) for value in config["counterfactual_packs"])
BROAD_PREDICTIONS = sorted(BROAD_DIR.glob("example_*.npz"), key=lambda path: int(path.stem.split("_")[1]))
if len(BROAD_PREDICTIONS) != 512:
    raise ValueError(f"expected 512 broad H2H predictions, found {len(BROAD_PREDICTIONS)}")

UV_CACHE_DIR = str(config.get("uv_cache_dir") or os.environ.get("UV_CACHE_DIR", "")).strip()
UV_CACHE_PREFIX = f"UV_CACHE_DIR={shlex.quote(UV_CACHE_DIR)} " if UV_CACHE_DIR else ""
UV_RUN_ARGS = str(config.get("uv_run_args", "--frozen --no-sync"))
UV_RUN = f"{UV_CACHE_PREFIX}uv --project {shlex.quote(str(PROJECT_ROOT))} run {UV_RUN_ARGS}"


def source_paths(*names):
    return [str(PROJECT_ROOT / "src" / "dreems" / name) for name in names]


def evaluation_dir(pack):
    return OUTDIR / "evaluation" / f"pack_{pack}"


def evaluation_output(pack, name):
    return evaluation_dir(pack) / name


EVALUATION_CODE = source_paths(
    "calibration.py",
    "checkpoints.py",
    "features.py",
    "metrics.py",
    "model.py",
    "pipeline.py",
    "workflow_io.py",
)
ANALYSIS_CODE = [
    str(PROJECT_ROOT / "workflow" / "analyze_null_shrinkage.py"),
    *source_paths("metrics.py", "null_shrinkage.py", "workflow_io.py"),
]


rule all:
    input:
        str(PUBLISH_DIR / "report.json"),
        str(PUBLISH_DIR / "rows.csv"),
        str(PUBLISH_DIR / "null_shrinkage_audit.pdf"),
        str(PUBLISH_DIR / "null_shrinkage_audit.png"),


rule evaluate_counterfactual_pack:
    input:
        checkpoint=str(CHECKPOINT),
        prepared=lambda wildcards: str(PREPARED_DIR / f"pack_{wildcards.pack}.pkl"),
        code=EVALUATION_CODE,
    output:
        summary=str(OUTDIR / "evaluation" / "pack_{pack}" / "summary.json"),
        csv=str(OUTDIR / "evaluation" / "pack_{pack}" / "summary.csv"),
        predictions=str(OUTDIR / "evaluation" / "pack_{pack}" / "predictions.npz"),
    benchmark:
        str(OUTDIR / "benchmarks" / "evaluate" / "pack_{pack}.tsv")
    wildcard_constraints:
        pack="|".join(str(pack) for pack in PACKS),
    threads: int(config.get("eval_threads", 1))
    resources:
        mem_mb=int(config.get("eval_mem_mb", 8000)),
        runtime=str(config.get("eval_runtime", "60m")),
        slurm_partition=str(config.get("eval_partition", "standard")),
    params:
        output_dir=lambda wildcards: str(evaluation_dir(wildcards.pack)),
        replicate_offset=lambda wildcards: 64 * int(wildcards.pack),
    shell:
        """
        export JAX_PLATFORMS=cpu
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
            --replicate-offset {params.replicate_offset} \
            --predictions-output {output.predictions:q}
        """


rule analyze_null_shrinkage:
    input:
        predictions=[str(evaluation_output(pack, "predictions.npz")) for pack in PACKS],
        summaries=[str(evaluation_output(pack, "summary.json")) for pack in PACKS],
        selection=str(SELECTION),
        exact=str(EXACT_NOTEBOOK),
        broad=[str(path) for path in BROAD_PREDICTIONS],
        code=ANALYSIS_CODE,
    output:
        report=str(PUBLISH_DIR / "report.json"),
        rows=str(PUBLISH_DIR / "rows.csv"),
        figure_pdf=str(PUBLISH_DIR / "null_shrinkage_audit.pdf"),
        figure_png=str(PUBLISH_DIR / "null_shrinkage_audit.png"),
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
        export OMP_NUM_THREADS={threads}
        export OPENBLAS_NUM_THREADS={threads}
        export MKL_NUM_THREADS={threads}
        {UV_RUN} python workflow/analyze_null_shrinkage.py \
            --predictions {input.predictions:q} \
            --summaries {input.summaries:q} \
            --selection {input.selection:q} \
            --exact-notebook {input.exact:q} \
            --broad-predictions {input.broad:q} \
            --output {output.report:q} \
            --rows-output {output.rows:q} \
            --figure {output.figure_pdf:q} \
            --seed {config[analysis_seed]} \
            --max-nonnull-increase {config[max_nonnull_relative_mse_increase]}
        """
