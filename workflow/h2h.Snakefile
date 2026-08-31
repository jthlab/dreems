"""Paired DREEMS versus documented FEEMS leave-one-deme-out evaluation."""

import json
import subprocess
from functools import lru_cache
from pathlib import Path


OUTDIR = str(config["outdir"])
PUBLISH_DIR = str(config.get("publish_dir", f"{OUTDIR}/report"))
INPUT_PATTERN = str(config["input_pattern"])
METADATA_PATTERN = str(config["metadata_pattern"])
CHECKPOINT = str(config["checkpoint"])
SCENARIO_CONFIG = config.get("scenario_config")
NOTEBOOK_SCENARIO_CONFIG = config.get("notebook_scenario_config")
NOTEBOOK_BASES_PRECOMPUTED = bool(config.get("notebook_scenario_bases_precomputed", False))
NOTEBOOK_CASES_PRECOMPUTED = bool(config.get("notebook_scenario_cases_precomputed", False))
GALLERY_COMMAND = str(config.get("gallery_command", "h2h-gallery"))
FEEMS_ARCHIVE = str(config.get("feems_archive", "")).strip()
FEEMS_ARCHIVE_CHECKSUM = str(config.get("feems_archive_checksum", "")).strip()
if bool(FEEMS_ARCHIVE) != bool(FEEMS_ARCHIVE_CHECKSUM):
    raise ValueError("feems_archive and feems_archive_checksum must be configured together")
CV_EXECUTION = "snakemake_one_job_per_observed_deme"
FEEMS_RESULT_SOURCE = "verified_archive" if FEEMS_ARCHIVE else "computed"
SOURCE_COMMIT = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    text=True,
).strip()

configured_ids = config.get("example_ids")
if configured_ids:
    EXAMPLES = [int(value) for value in configured_ids]
else:
    first = int(config.get("first_example", 0))
    count = int(config["example_count"])
    EXAMPLES = list(range(first, first + count))
if not EXAMPLES or len(EXAMPLES) != len(set(EXAMPLES)):
    raise ValueError("H2H example IDs must be nonempty and unique")
if GALLERY_COMMAND not in {"h2h-gallery", "scenario-gallery"}:
    raise ValueError(f"unsupported gallery_command: {GALLERY_COMMAND!r}")
if SCENARIO_CONFIG and NOTEBOOK_SCENARIO_CONFIG:
    raise ValueError("scenario_config and notebook_scenario_config are mutually exclusive")
if SCENARIO_CONFIG:
    from dreems.scenario_audit import observed_node_count, scenario_count

    if EXAMPLES != list(range(scenario_count(config))):
        raise ValueError(
            "scenario audit example IDs must cover the complete zero-based crossed design"
        )
if NOTEBOOK_SCENARIO_CONFIG:
    from dreems.notebook_scenarios import (
        observed_node_count as notebook_observed_node_count,
        scenario_count as notebook_scenario_count,
        scenario_spec as notebook_scenario_spec,
    )

    if EXAMPLES != list(range(notebook_scenario_count(config))):
        raise ValueError(
            "notebook scenario IDs must cover the complete zero-based notebook design"
        )

NOTEBOOK_BASE_PATTERN = str(
    config.get("notebook_scenario_base_pattern", f"{OUTDIR}/inputs/base_{{truth}}.npz")
)


def source_path(pattern, example):
    return pattern.format(example=int(example))


@lru_cache(maxsize=None)
def example_metadata(example):
    path = Path(source_path(METADATA_PATTERN, example))
    payload = json.loads(path.read_text())
    row = payload.get("row", payload)
    return row


def feems_json(example):
    return f"{OUTDIR}/feems/example_{example}.json"


def feems_npz(example):
    return f"{OUTDIR}/feems/example_{example}.npz"


def dreems_json(example):
    return f"{OUTDIR}/dreems/example_{example}.json"


def dreems_npz(example):
    return f"{OUTDIR}/dreems/example_{example}.npz"


def feems_cv_checkpoint_dir(example):
    return f"{OUTDIR}/feems_cv/example_{example}"


def feems_cv_fold(example, fold):
    return f"{feems_cv_checkpoint_dir(example)}/fold_{int(fold):04d}.npz"


def feems_cv_folds(wildcards):
    if FEEMS_ARCHIVE:
        return []
    if SCENARIO_CONFIG:
        count = observed_node_count(config, int(wildcards.example))
    elif NOTEBOOK_SCENARIO_CONFIG:
        count = notebook_observed_node_count(config, int(wildcards.example))
    else:
        count = int(example_metadata(wildcards.example)["num_sampled_nodes"])
    return [feems_cv_fold(wildcards.example, fold) for fold in range(count)]


localrules: all


rule all:
    input:
        f"{PUBLISH_DIR}/summary.json",
        f"{PUBLISH_DIR}/rows.csv",
        f"{PUBLISH_DIR}/protocol.json",
        f"{PUBLISH_DIR}/feems_results.tar.gz",
        f"{PUBLISH_DIR}/feems_results.tar.gz.sha256",
        f"{PUBLISH_DIR}/gallery",


if SCENARIO_CONFIG:
    rule generate_scenario_case:
        input:
            config=str(SCENARIO_CONFIG),
            runner="src/dreems/scenario_audit.py",
        output:
            example=INPUT_PATTERN,
            metadata=METADATA_PATTERN,
        benchmark:
            f"{OUTDIR}/benchmarks/generate/example_{{example}}.tsv"
        wildcard_constraints:
            example="\\d+"
        threads: int(config.get("scenario_threads", 1))
        resources:
            mem_mb=int(config.get("scenario_mem_mb", 4000)),
            runtime=str(config.get("scenario_runtime", "60m")),
            slurm_partition=str(config.get("scenario_partition", "standard")),
            slurm_account="jonth0",
        shell:
            """
            set -euo pipefail
            export OMP_NUM_THREADS={threads}
            export OPENBLAS_NUM_THREADS={threads}
            export MKL_NUM_THREADS={threads}
            uv run --frozen --no-sync python -m dreems.scenario_audit \
                --config {input.config:q} \
                --example-id {wildcards.example} \
                --output {output.example:q} \
                --metadata {output.metadata:q}
            """


if NOTEBOOK_SCENARIO_CONFIG:
    def notebook_base_for_example(example):
        truth = notebook_scenario_spec(config, int(example)).truth
        return NOTEBOOK_BASE_PATTERN.format(truth=truth)


if NOTEBOOK_SCENARIO_CONFIG and not NOTEBOOK_BASES_PRECOMPUTED:
    rule generate_notebook_scenario_base:
        input:
            config=str(NOTEBOOK_SCENARIO_CONFIG),
            notebook=lambda _wildcards: str(config["notebook_scenarios"]["source_notebook"]),
            runner="src/dreems/notebook_scenarios.py",
        output:
            NOTEBOOK_BASE_PATTERN
        benchmark:
            f"{OUTDIR}/benchmarks/generate_base/{{truth}}.tsv"
        wildcard_constraints:
            truth="barrier|anisotropy|homogeneous"
        threads: int(config.get("notebook_base_threads", 1))
        resources:
            mem_mb=int(config.get("notebook_base_mem_mb", 8000)),
            runtime=str(config.get("notebook_base_runtime", "4h")),
            slurm_partition=str(config.get("notebook_base_partition", "standard")),
            slurm_account="jonth0",
        shell:
            """
            set -euo pipefail
            export OMP_NUM_THREADS={threads}
            export OPENBLAS_NUM_THREADS={threads}
            export MKL_NUM_THREADS={threads}
            uv run --frozen --no-sync python -m dreems.notebook_scenarios generate-base \
                --config {input.config:q} \
                --truth {wildcards.truth:q} \
                --output {output:q}
            """


if NOTEBOOK_SCENARIO_CONFIG and not NOTEBOOK_CASES_PRECOMPUTED:
    rule derive_notebook_scenario_case:
        input:
            base=lambda wildcards: notebook_base_for_example(wildcards.example),
            config=str(NOTEBOOK_SCENARIO_CONFIG),
            runner="src/dreems/notebook_scenarios.py",
        output:
            example=INPUT_PATTERN,
            metadata=METADATA_PATTERN,
        benchmark:
            f"{OUTDIR}/benchmarks/derive/example_{{example}}.tsv"
        wildcard_constraints:
            example="\\d+"
        threads: 1
        resources:
            mem_mb=int(config.get("notebook_derive_mem_mb", 8000)),
            runtime=str(config.get("notebook_derive_runtime", "30m")),
            slurm_partition=str(config.get("notebook_base_partition", "standard")),
            slurm_account="jonth0",
        shell:
            """
            set -euo pipefail
            export OMP_NUM_THREADS=1
            export OPENBLAS_NUM_THREADS=1
            export MKL_NUM_THREADS=1
            uv run --frozen --no-sync python -m dreems.notebook_scenarios derive-case \
                --config {input.config:q} \
                --example-id {wildcards.example} \
                --base {input.base:q} \
                --output {output.example:q} \
                --metadata {output.metadata:q}
            """


rule feems_official_parity:
    input:
        adapter="src/dreems/feems.py",
        test="tests/test_feems_official_parity.py",
    output:
        touch(f"{OUTDIR}/protocol_checks/official_cv_parity.ok")
    benchmark:
        f"{OUTDIR}/benchmarks/protocol_checks/official_cv_parity.tsv"
    threads: 2
    resources:
        mem_mb=4000,
        runtime="30m",
        slurm_partition=str(config.get("protocol_check_partition", "standard")),
        slurm_account="jonth0",
    shell:
        """
        set -euo pipefail
        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        uv run --frozen --no-sync pytest -q \
            tests/test_feems_official_parity.py::test_fold_parallel_joint_cv_matches_official_feems_serial_cv
        """


rule feems_cv_fold:
    input:
        example=lambda wildcards: source_path(INPUT_PATTERN, wildcards.example),
        metadata=lambda wildcards: source_path(METADATA_PATTERN, wildcards.example),
        runner="src/dreems/h2h.py",
        adapter="src/dreems/feems.py",
    output:
        temp(f"{OUTDIR}/feems_cv/example_{{example}}/fold_{{fold}}.npz")
    benchmark:
        f"{OUTDIR}/benchmarks/feems_cv/example_{{example}}/fold_{{fold}}.tsv"
    wildcard_constraints:
        example="\\d+",
        fold="\\d{4}",
    params:
        fold=lambda wildcards: int(wildcards.fold),
    threads: 1
    resources:
        mem_mb=int(config.get("feems_fold_mem_mb", 4000)),
        runtime=str(config.get("feems_fold_runtime", "60m")),
        slurm_partition=str(config.get("feems_partition", "standard")),
        slurm_account="jonth0",
    shell:
        """
        set -euo pipefail
        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1
        uv run --frozen --no-sync python -m dreems.h2h feems-fold \
            --input {input.example:q} \
            --fold {params.fold} \
            --output-npz {output:q}
        """


rule feems_case:
    input:
        example=lambda wildcards: source_path(INPUT_PATTERN, wildcards.example),
        metadata=lambda wildcards: source_path(METADATA_PATTERN, wildcards.example),
        folds=feems_cv_folds,
        archive=lambda _wildcards: [FEEMS_ARCHIVE] if FEEMS_ARCHIVE else [],
        archive_checksum=lambda _wildcards: (
            [FEEMS_ARCHIVE_CHECKSUM] if FEEMS_ARCHIVE_CHECKSUM else []
        ),
        runner="src/dreems/h2h.py",
        adapter="src/dreems/feems.py",
    output:
        json=f"{OUTDIR}/feems/example_{{example}}.json",
        npz=f"{OUTDIR}/feems/example_{{example}}.npz",
    benchmark:
        f"{OUTDIR}/benchmarks/feems/example_{{example}}.tsv"
    wildcard_constraints:
        example="\\d+"
    params:
        cv_checkpoint_dir=lambda wildcards: feems_cv_checkpoint_dir(wildcards.example),
        use_archive="1" if FEEMS_ARCHIVE else "0",
    threads: 1
    resources:
        mem_mb=int(config.get("feems_final_mem_mb", 8000)),
        runtime=str(config.get("feems_final_runtime", "60m")),
        slurm_partition=str(config.get("feems_partition", "standard")),
        slurm_account="jonth0",
    shell:
        """
        set -euo pipefail
        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1
        if [[ {params.use_archive:q} == "1" ]]; then
            archive={input.archive:q}
            checksum={input.archive_checksum:q}
            (
                cd "$(dirname "$archive")"
                sha256sum -c "$(basename "$checksum")"
            )
            mkdir -p "$(dirname {output.json:q})"
            json_tmp={output.json:q}.tmp
            npz_tmp={output.npz:q}.tmp
            trap 'rm -f "$json_tmp" "$npz_tmp"' EXIT
            tar -xOf "$archive" "feems/example_{wildcards.example}.json" > "$json_tmp"
            tar -xOf "$archive" "feems/example_{wildcards.example}.npz" > "$npz_tmp"
            mv "$json_tmp" {output.json:q}
            mv "$npz_tmp" {output.npz:q}
        else
            uv run --frozen --no-sync python -m dreems.h2h feems-case \
                --input {input.example:q} \
                --index {wildcards.example} \
                --cv-jobs 1 \
                --cv-checkpoint-dir {params.cv_checkpoint_dir:q} \
                --output-json {output.json:q} \
                --output-npz {output.npz:q}
        fi
        """


rule dreems_case:
    input:
        example=lambda wildcards: source_path(INPUT_PATTERN, wildcards.example),
        metadata=lambda wildcards: source_path(METADATA_PATTERN, wildcards.example),
        checkpoint=CHECKPOINT,
        runner="src/dreems/h2h.py",
    output:
        json=f"{OUTDIR}/dreems/example_{{example}}.json",
        npz=f"{OUTDIR}/dreems/example_{{example}}.npz",
    wildcard_constraints:
        example="\\d+"
    threads: int(config.get("dreems_threads", 2))
    resources:
        mem_mb=int(config.get("dreems_mem_mb", 6000)),
        runtime=str(config.get("dreems_runtime", "60m")),
        slurm_partition=str(config.get("dreems_partition", "standard")),
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
        uv run --frozen --no-sync python -m dreems.h2h dreems-case \
            --input {input.example:q} \
            --index {wildcards.example} \
            --checkpoint {input.checkpoint:q} \
            --output-json {output.json:q} \
            --output-npz {output.npz:q}
        """


rule aggregate:
    input:
        feems=[feems_json(example) for example in EXAMPLES],
        dreems=[dreems_json(example) for example in EXAMPLES],
        parity=f"{OUTDIR}/protocol_checks/official_cv_parity.ok",
        checkpoint=CHECKPOINT,
        runner="src/dreems/h2h.py",
    output:
        summary=f"{PUBLISH_DIR}/summary.json",
        rows=f"{PUBLISH_DIR}/rows.csv",
        protocol=f"{PUBLISH_DIR}/protocol.json",
    benchmark:
        f"{PUBLISH_DIR}/benchmarks/aggregate.tsv"
    threads: 1
    resources:
        mem_mb=int(config.get("aggregate_mem_mb", 4000)),
        runtime=str(config.get("aggregate_runtime", "30m")),
        slurm_partition=str(config.get("aggregate_partition", "standard")),
        slurm_account="jonth0",
    shell:
        """
        set -euo pipefail
        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        uv run --frozen --no-sync python -m dreems.h2h aggregate \
            --feems-json {input.feems:q} \
            --dreems-json {input.dreems:q} \
            --checkpoint {input.checkpoint:q} \
            --cv-execution {CV_EXECUTION:q} \
            --feems-result-source {FEEMS_RESULT_SOURCE:q} \
            --source-commit {SOURCE_COMMIT:q} \
            --input-pattern {INPUT_PATTERN:q} \
            --metadata-pattern {METADATA_PATTERN:q} \
            --output-json {output.summary:q} \
            --output-csv {output.rows:q} \
            --output-protocol {output.protocol:q}
        """


rule archive_feems:
    input:
        json=[feems_json(example) for example in EXAMPLES],
        npz=[feems_npz(example) for example in EXAMPLES],
    output:
        archive=f"{PUBLISH_DIR}/feems_results.tar.gz",
        checksum=f"{PUBLISH_DIR}/feems_results.tar.gz.sha256",
    benchmark:
        f"{PUBLISH_DIR}/benchmarks/archive_feems.tsv"
    threads: 1
    resources:
        mem_mb=int(config.get("archive_mem_mb", 1000)),
        runtime=str(config.get("archive_runtime", "15m")),
        slurm_partition=str(config.get("aggregate_partition", "standard")),
        slurm_account="jonth0",
    params:
        source_dir=OUTDIR,
    shell:
        """
        set -euo pipefail
        mkdir -p "$(dirname {output.archive:q})"
        list="$(mktemp)"
        archive_tmp="{output.archive}.tmp"
        checksum_tmp="{output.checksum}.tmp"
        trap 'rm -f "$list" "$archive_tmp" "$checksum_tmp"' EXIT
        printf '%s\n' {input.json:q} {input.npz:q} \
            | sed 's|^{params.source_dir:q}/||' \
            | LC_ALL=C sort > "$list"
        tar \
            --sort=name \
            --mtime='UTC 1970-01-01' \
            --owner=0 \
            --group=0 \
            --numeric-owner \
            -C {params.source_dir:q} \
            -czf "$archive_tmp" \
            -T "$list"
        mv "$archive_tmp" {output.archive:q}
        printf '%s  %s\n' \
            "$(sha256sum {output.archive:q} | cut -d ' ' -f 1)" \
            "$(basename {output.archive:q})" > "$checksum_tmp"
        mv "$checksum_tmp" {output.checksum:q}
        """


rule gallery:
    input:
        rows=f"{PUBLISH_DIR}/rows.csv",
        summary=f"{PUBLISH_DIR}/summary.json",
        dreems=[dreems_npz(example) for example in EXAMPLES],
        feems=[feems_npz(example) for example in EXAMPLES],
        renderer="src/dreems/benchmark_gallery.py",
    output:
        directory(f"{PUBLISH_DIR}/gallery"),
    benchmark:
        f"{PUBLISH_DIR}/benchmarks/gallery.tsv"
    threads: int(config.get("gallery_threads", 1))
    resources:
        mem_mb=int(config.get("gallery_mem_mb", 2000)),
        runtime=str(config.get("gallery_runtime", "120m")),
        slurm_partition=str(config.get("aggregate_partition", "standard")),
        slurm_account="jonth0",
    params:
        dreems_pattern=lambda _wildcards: f"{OUTDIR}/dreems/example_{{example}}.npz",
        feems_pattern=lambda _wildcards: f"{OUTDIR}/feems/example_{{example}}.npz",
        command=GALLERY_COMMAND,
    shell:
        """
        set -euo pipefail
        export MPLCONFIGDIR="${{TMPDIR:-/tmp}}/matplotlib-${{SLURM_JOB_ID:-local}}"
        export OMP_NUM_THREADS={threads}
        export OPENBLAS_NUM_THREADS={threads}
        export MKL_NUM_THREADS={threads}
        uv run --frozen --no-sync python -m dreems.benchmark_gallery {params.command:q} \
            --rows {input.rows:q} \
            --dreems-pattern {params.dreems_pattern:q} \
            --feems-pattern {params.feems_pattern:q} \
            --output-dir {output:q}
        """
