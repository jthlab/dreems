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
CV_EXECUTION = "snakemake_one_job_per_observed_deme"
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
        uv run --frozen --no-sync python -m dreems.h2h feems-case \
            --input {input.example:q} \
            --index {wildcards.example} \
            --cv-jobs 1 \
            --cv-checkpoint-dir {params.cv_checkpoint_dir:q} \
            --output-json {output.json:q} \
            --output-npz {output.npz:q}
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
