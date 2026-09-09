"""Paired DREEMS versus documented FEEMS leave-one-deme-out evaluation."""

import json
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

from dreems.validation import _integer_array


PROJECT = Path(workflow.current_basedir).parent.resolve()
INFERENCE_CODE = [str(PROJECT / "src/dreems" / name) for name in (
    "inference.py", "checkpoints.py", "model.py", "features.py", "linearized.py",
    "refinement.py", "calibration.py", "validation.py", "metrics.py",
)]
BENCHMARK_CODE = [str(PROJECT / "src/dreems" / name) for name in (
    "benchmark_data.py", "validation.py", "metrics.py", "workflow_io.py",
)]
ENVIRONMENT = [str(PROJECT / "pyproject.toml"), str(PROJECT / "uv.lock")]

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
ALLOW_LEGACY_ARCHIVE = config.get("allow_legacy_feems_archive", False)
if not isinstance(ALLOW_LEGACY_ARCHIVE, bool):
    raise ValueError("allow_legacy_feems_archive must be a boolean")
if bool(FEEMS_ARCHIVE) != bool(FEEMS_ARCHIVE_CHECKSUM):
    raise ValueError("feems_archive and feems_archive_checksum must be configured together")
CV_EXECUTION = "snakemake_one_job_per_observed_deme"
FEEMS_RESULT_SOURCE = (
    "legacy_archive" if ALLOW_LEGACY_ARCHIVE else "verified_archive"
) if FEEMS_ARCHIVE else "computed"
ARCHIVE_VERIFICATION = f"{PUBLISH_DIR}/feems_archive_verification.json"
ARCHIVE_IS_PUBLISHED = bool(FEEMS_ARCHIVE) and Path(FEEMS_ARCHIVE).resolve() == Path(
    PUBLISH_DIR, "feems_results.tar.gz"
).resolve()
if ARCHIVE_IS_PUBLISHED and Path(FEEMS_ARCHIVE_CHECKSUM).resolve() != Path(
    PUBLISH_DIR, "feems_results.tar.gz.sha256"
).resolve():
    raise ValueError("in-place archive reuse needs the published checksum path too")
try:
    SOURCE_COMMIT = subprocess.check_output(
        ["git", "-C", str(PROJECT), "rev-parse", "HEAD"],
        text=True, stderr=subprocess.DEVNULL,
    ).strip()
except subprocess.CalledProcessError:
    SOURCE_COMMIT = "unknown"

configured_ids = config.get("example_ids")
if configured_ids is not None:
    if any(isinstance(value, bool) for value in configured_ids):
        raise ValueError("example_ids must contain integers, not booleans")
    ids = _integer_array(np.asarray(configured_ids), name="example_ids", dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError("example_ids must be a list of integers")
    EXAMPLES = ids.tolist()
else:
    if isinstance(config.get("first_example", 0), bool) or isinstance(config["example_count"], bool):
        raise ValueError("first_example and example_count must be integers, not booleans")
    first, count = _integer_array(
        np.asarray([config.get("first_example", 0), config["example_count"]]),
        name="first_example and example_count", dtype=np.int64,
    ).tolist()
    EXAMPLES = list(range(first, first + count))
if not EXAMPLES or min(EXAMPLES) < 0 or len(EXAMPLES) != len(set(EXAMPLES)):
    raise ValueError("H2H example IDs must be nonempty, nonnegative, and unique")
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
        count = example_metadata(wildcards.example)["num_sampled_nodes"]
    count = _integer_array(np.asarray(count), name="num_sampled_nodes", dtype=np.int64).item()
    if count < 2:
        raise ValueError("FEEMS LOO requires at least two sampled nodes")
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
        [ARCHIVE_VERIFICATION] if FEEMS_ARCHIVE else [],


if SCENARIO_CONFIG:
    rule generate_scenario_case:
        input:
            config=str(SCENARIO_CONFIG),
            runner=str(PROJECT / "src/dreems/scenario_audit.py"),
            code=[str(PROJECT / "src/dreems" / name) for name in (
                "geography.py", "corpus.py", "island_sim.py", "validation.py", "workflow_io.py",
            )] + [str(PROJECT / "rust/island_sim" / name) for name in (
                "Cargo.toml", "Cargo.lock", "src/lib.rs",
            )],
            environment=ENVIRONMENT,
        output:
            example=INPUT_PATTERN,
            metadata=METADATA_PATTERN,
        benchmark:
            f"{OUTDIR}/benchmarks/generate/example_{{example}}.tsv"
        wildcard_constraints:
            example="\\d+"
        threads: int(config.get("scenario_threads", 1))
        params:
            project=str(PROJECT),
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
            uv --project {params.project:q} run --frozen --no-sync python -m dreems.scenario_audit \
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
            runner=str(PROJECT / "src/dreems/notebook_scenarios.py"),
            code=[str(PROJECT / "src/dreems" / name) for name in ("validation.py", "workflow_io.py")],
            environment=ENVIRONMENT,
        output:
            NOTEBOOK_BASE_PATTERN
        benchmark:
            f"{OUTDIR}/benchmarks/generate_base/{{truth}}.tsv"
        wildcard_constraints:
            truth="barrier|anisotropy|homogeneous"
        threads: int(config.get("notebook_base_threads", 1))
        params:
            project=str(PROJECT),
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
            uv --project {params.project:q} run --frozen --no-sync python -m dreems.notebook_scenarios generate-base \
                --config {input.config:q} \
                --truth {wildcards.truth:q} \
                --output {output:q}
            """


if NOTEBOOK_SCENARIO_CONFIG and not NOTEBOOK_CASES_PRECOMPUTED:
    rule derive_notebook_scenario_case:
        input:
            base=lambda wildcards: notebook_base_for_example(wildcards.example),
            config=str(NOTEBOOK_SCENARIO_CONFIG),
            runner=str(PROJECT / "src/dreems/notebook_scenarios.py"),
            code=[str(PROJECT / "src/dreems" / name) for name in (
                "corpus.py", "features.py", "genotypes.py", "datasets.py", "inference.py",
                "validation.py", "workflow_io.py",
            )],
            environment=ENVIRONMENT,
        output:
            example=INPUT_PATTERN,
            metadata=METADATA_PATTERN,
        benchmark:
            f"{OUTDIR}/benchmarks/derive/example_{{example}}.tsv"
        wildcard_constraints:
            example="\\d+"
        threads: 1
        params:
            project=str(PROJECT),
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
            uv --project {params.project:q} run --frozen --no-sync python -m dreems.notebook_scenarios derive-case \
                --config {input.config:q} \
                --example-id {wildcards.example} \
                --base {input.base:q} \
                --output {output.example:q} \
                --metadata {output.metadata:q}
            """


rule feems_official_parity:
    input:
        adapter=str(PROJECT / "src/dreems/feems.py"),
        test=str(PROJECT / "tests/test_feems_official_parity.py"),
        environment=ENVIRONMENT,
    output:
        touch(f"{OUTDIR}/protocol_checks/official_cv_parity.ok")
    benchmark:
        f"{OUTDIR}/benchmarks/protocol_checks/official_cv_parity.tsv"
    threads: 2
    params:
        project=str(PROJECT),
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
        uv --project {params.project:q} run --frozen --no-sync pytest -q \
            {input.test:q}::test_fold_parallel_joint_cv_matches_official_feems_serial_cv
        """


rule feems_cv_fold:
    input:
        example=lambda wildcards: source_path(INPUT_PATTERN, wildcards.example),
        metadata=lambda wildcards: source_path(METADATA_PATTERN, wildcards.example),
        runner=str(PROJECT / "src/dreems/feems_benchmark.py"),
        adapter=str(PROJECT / "src/dreems/feems.py"),
        code=BENCHMARK_CODE,
        environment=ENVIRONMENT,
    output:
        temp(f"{OUTDIR}/feems_cv/example_{{example}}/fold_{{fold}}.npz")
    benchmark:
        f"{OUTDIR}/benchmarks/feems_cv/example_{{example}}/fold_{{fold}}.tsv"
    wildcard_constraints:
        example="\\d+",
        fold="\\d{4}",
    params:
        project=str(PROJECT),
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
        uv --project {params.project:q} run --frozen --no-sync python -m dreems.feems_benchmark feems-fold \
            --input {input.example:q} \
            --fold {params.fold} \
            --output-npz {output:q}
        """


if FEEMS_ARCHIVE:
    rule extract_feems_archive:
        input:
            archive=FEEMS_ARCHIVE,
            checksum=FEEMS_ARCHIVE_CHECKSUM,
            examples=[source_path(INPUT_PATTERN, example) for example in EXAMPLES],
            runner=str(PROJECT / "src/dreems/benchmark_archive.py"),
            adapter=str(PROJECT / "src/dreems/feems.py"),
            code=BENCHMARK_CODE,
            environment=ENVIRONMENT,
        output:
            json=[feems_json(example) for example in EXAMPLES],
            npz=[feems_npz(example) for example in EXAMPLES],
            verification=ARCHIVE_VERIFICATION,
        benchmark:
            f"{OUTDIR}/benchmarks/extract_feems.tsv"
        params:
            target=f"{OUTDIR}/feems",
            examples=EXAMPLES,
            input_pattern=lambda _wildcards: INPUT_PATTERN,
            project=str(PROJECT),
            legacy=["--allow-legacy"] if ALLOW_LEGACY_ARCHIVE else [],
        threads: 1
        resources:
            mem_mb=int(config.get("archive_mem_mb", 1000)),
            runtime=str(config.get("archive_runtime", "15m")),
            slurm_partition=str(config.get("aggregate_partition", "standard")),
            slurm_account="jonth0",
        shell:
            """
            set -euo pipefail
            uv --project {params.project:q} run --frozen --no-sync python -m dreems.benchmark_archive restore \
                --archive {input.archive:q} --checksum {input.checksum:q} \
                --results-dir {params.target:q} --input-pattern {params.input_pattern:q} \
                --examples {params.examples:q} --verification {output.verification:q} {params.legacy:q}
            """
else:
    rule feems_case:
        input:
            example=lambda wildcards: source_path(INPUT_PATTERN, wildcards.example),
            metadata=lambda wildcards: source_path(METADATA_PATTERN, wildcards.example),
            folds=feems_cv_folds,
            runner=str(PROJECT / "src/dreems/feems_benchmark.py"),
            adapter=str(PROJECT / "src/dreems/feems.py"),
            code=BENCHMARK_CODE,
            environment=ENVIRONMENT,
        output:
            json=f"{OUTDIR}/feems/example_{{example}}.json",
            npz=f"{OUTDIR}/feems/example_{{example}}.npz",
        benchmark:
            f"{OUTDIR}/benchmarks/feems/example_{{example}}.tsv"
        wildcard_constraints:
            example="\\d+"
        params:
            project=str(PROJECT),
            cv_checkpoint_dir=lambda wildcards: feems_cv_checkpoint_dir(
                wildcards.example
            ),
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
            uv --project {params.project:q} run --frozen --no-sync python -m dreems.feems_benchmark feems-case \
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
        runner=str(PROJECT / "src/dreems/h2h.py"),
        code=INFERENCE_CODE + BENCHMARK_CODE,
        environment=ENVIRONMENT,
    output:
        json=f"{OUTDIR}/dreems/example_{{example}}.json",
        npz=f"{OUTDIR}/dreems/example_{{example}}.npz",
    benchmark:
        f"{OUTDIR}/benchmarks/dreems/example_{{example}}.tsv"
    wildcard_constraints:
        example="\\d+"
    threads: int(config.get("dreems_threads", 2))
    params:
        project=str(PROJECT),
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
        uv --project {params.project:q} run --frozen --no-sync python -m dreems.h2h dreems-case \
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
        runner=str(PROJECT / "src/dreems/h2h.py"),
        adapter=str(PROJECT / "src/dreems/feems.py"),
        code=BENCHMARK_CODE,
        environment=ENVIRONMENT,
        verification=[ARCHIVE_VERIFICATION] if FEEMS_ARCHIVE else [],
    output:
        summary=f"{PUBLISH_DIR}/summary.json",
        rows=f"{PUBLISH_DIR}/rows.csv",
        protocol=f"{PUBLISH_DIR}/protocol.json",
    benchmark:
        f"{PUBLISH_DIR}/benchmarks/aggregate.tsv"
    threads: 1
    params:
        project=str(PROJECT),
        cv_execution=CV_EXECUTION,
        result_source=FEEMS_RESULT_SOURCE,
        source_commit=SOURCE_COMMIT,
        input_pattern=lambda _wildcards: INPUT_PATTERN,
        metadata_pattern=lambda _wildcards: METADATA_PATTERN,
        verification_args=["--feems-archive-verification", ARCHIVE_VERIFICATION] if FEEMS_ARCHIVE else [],
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
        uv --project {params.project:q} run --frozen --no-sync python -m dreems.h2h aggregate \
            --feems-json {input.feems:q} \
            --dreems-json {input.dreems:q} \
            --checkpoint {input.checkpoint:q} \
            --cv-execution {params.cv_execution:q} \
            --feems-result-source {params.result_source:q} \
            --source-commit {params.source_commit:q} \
            --input-pattern {params.input_pattern:q} \
            --metadata-pattern {params.metadata_pattern:q} \
            {params.verification_args:q} \
            --output-json {output.summary:q} \
            --output-csv {output.rows:q} \
            --output-protocol {output.protocol:q}
        """


if not ARCHIVE_IS_PUBLISHED:
    rule archive_feems:
        input:
            json=[feems_json(example) for example in EXAMPLES],
            npz=[feems_npz(example) for example in EXAMPLES],
            examples=[source_path(INPUT_PATTERN, example) for example in EXAMPLES],
            runner=str(PROJECT / "src/dreems/benchmark_archive.py"),
            adapter=str(PROJECT / "src/dreems/feems.py"),
            code=BENCHMARK_CODE,
            environment=ENVIRONMENT,
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
            results_dir=f"{OUTDIR}/feems",
            examples=EXAMPLES,
            input_pattern=lambda _wildcards: INPUT_PATTERN,
            project=str(PROJECT),
            legacy=["--allow-legacy"] if ALLOW_LEGACY_ARCHIVE else [],
        shell:
            """
            set -euo pipefail
            uv --project {params.project:q} run --frozen --no-sync python -m dreems.benchmark_archive pack \
                --archive {output.archive:q} --checksum {output.checksum:q} \
                --results-dir {params.results_dir:q} --input-pattern {params.input_pattern:q} \
                --examples {params.examples:q} {params.legacy:q}
            """


rule gallery:
    input:
        rows=f"{PUBLISH_DIR}/rows.csv",
        summary=f"{PUBLISH_DIR}/summary.json",
        dreems=[dreems_npz(example) for example in EXAMPLES],
        feems=[feems_npz(example) for example in EXAMPLES],
        renderer=str(PROJECT / "src/dreems/benchmark_gallery.py"),
        environment=ENVIRONMENT,
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
        project=str(PROJECT),
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
        uv --project {params.project:q} run --frozen --no-sync python -m dreems.benchmark_gallery {params.command:q} \
            --rows {input.rows:q} \
            --dreems-pattern {params.dreems_pattern:q} \
            --feems-pattern {params.feems_pattern:q} \
            --output-dir {output:q}
        """
