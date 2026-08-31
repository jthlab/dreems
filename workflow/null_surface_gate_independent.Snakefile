"""Independent null/structured verification for the learned surface gate."""

import os
import shlex
from pathlib import Path


PROJECT_ROOT = Path(workflow.basedir).parent.resolve()
OUTDIR = Path(str(config["outdir"])).expanduser()
PUBLISH_DIR = Path(str(config["publish_dir"])).expanduser()
PACKS = tuple(int(pack) for pack in config["pack_ids"])
MODELS = ("baseline", "gated")
CHECKPOINTS = {
    "baseline": Path(str(config["baseline_checkpoint"])).expanduser(),
    "gated": Path(str(config["gated_checkpoint"])).expanduser(),
}
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
    "metrics.py",
    "model.py",
    "pipeline.py",
    "workflow_io.py",
)
ANALYSIS_CODE = str(PROJECT_ROOT / "workflow" / "analyze_surface_gate_independent.py")


def configured_path(pattern, pack):
    return str(pattern).format(pack=pack)


def evaluation_output(model, pack, name):
    return OUTDIR / "evaluation" / model / f"pack_{pack}" / name


localrules: all


rule all:
    input:
        str(PUBLISH_DIR / "independent_report.json"),
        str(PUBLISH_DIR / "independent_rows.csv"),
        str(PUBLISH_DIR / "independent_surface_gate_audit.pdf"),
        str(PUBLISH_DIR / "independent_surface_gate_audit.png"),


rule evaluate_pack:
    input:
        checkpoint=lambda wildcards: str(CHECKPOINTS[str(wildcards.model)]),
        prepared=lambda wildcards: configured_path(config["prepared_pattern"], wildcards.pack),
        metadata=lambda wildcards: configured_path(config["metadata_pattern"], wildcards.pack),
        code=EVAL_CODE,
    output:
        summary=str(OUTDIR / "evaluation" / "{model}" / "pack_{pack}" / "summary.json"),
        csv=str(OUTDIR / "evaluation" / "{model}" / "pack_{pack}" / "summary.csv"),
        predictions=str(
            OUTDIR / "evaluation" / "{model}" / "pack_{pack}" / "predictions.npz"
        ),
    benchmark:
        str(OUTDIR / "benchmarks" / "evaluate" / "{model}" / "pack_{pack}.tsv")
    wildcard_constraints:
        model="baseline|gated",
        pack="|".join(str(pack) for pack in PACKS),
    threads: int(config.get("eval_threads", 1))
    resources:
        mem_mb=int(config.get("eval_mem_mb", 8000)),
        runtime=str(config.get("eval_runtime", "60m")),
        slurm_partition=str(config.get("eval_partition", "standard")),
    params:
        output_dir=lambda wildcards: str(
            OUTDIR / "evaluation" / str(wildcards.model) / f"pack_{wildcards.pack}"
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
            --batch-size {config[eval_batch_size]} \
            --replicate-offset {params.replicate_offset} \
            --predictions-output {output.predictions:q}
        """


rule analyze_independent_surface_gate:
    input:
        baseline=[
            str(evaluation_output("baseline", pack, "predictions.npz")) for pack in PACKS
        ],
        gated=[str(evaluation_output("gated", pack, "predictions.npz")) for pack in PACKS],
        metadata=[configured_path(config["metadata_pattern"], pack) for pack in PACKS],
        selection=str(config["selection"]),
        code=ANALYSIS_CODE,
    output:
        report=str(PUBLISH_DIR / "independent_report.json"),
        rows=str(PUBLISH_DIR / "independent_rows.csv"),
        figure_pdf=str(PUBLISH_DIR / "independent_surface_gate_audit.pdf"),
        figure_png=str(PUBLISH_DIR / "independent_surface_gate_audit.png"),
    benchmark:
        str(OUTDIR / "benchmarks" / "analyze.tsv")
    threads: int(config.get("analysis_threads", 1))
    resources:
        mem_mb=int(config.get("analysis_mem_mb", 8000)),
        runtime=str(config.get("analysis_runtime", "30m")),
        slurm_partition=str(config.get("analysis_partition", "standard")),
    params:
        included_packs=" ".join(str(pack) for pack in PACKS),
        excluded_replay_packs=" ".join(
            str(pack) for pack in config["excluded_replay_pack_ids"]
        ),
    shell:
        """
        export JAX_PLATFORMS=cpu
        export CUDA_VISIBLE_DEVICES=""
        {UV_RUN} python {input.code:q} \
            --baseline {input.baseline:q} \
            --gated {input.gated:q} \
            --metadata {input.metadata:q} \
            --selection {input.selection:q} \
            --included-packs {params.included_packs} \
            --excluded-replay-packs {params.excluded_replay_packs} \
            --output {output.report:q} \
            --rows-output {output.rows:q} \
            --figure {output.figure_pdf:q}
        """
