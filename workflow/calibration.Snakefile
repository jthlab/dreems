"""Fit, select, and verify posterior-SD calibration on disjoint test splits."""

from pathlib import Path


ROOT = str(Path(config["outdir"]).expanduser().resolve())
CHECKPOINT = str(Path(config["checkpoint"]).expanduser().resolve())
CANONICAL_CHECKPOINT = f"{ROOT}/checkpoint.canonical.dreems.zip"
RAW_DIR = str(Path(config["raw_source_dir"]).expanduser().resolve())
RAW_SOURCE_LAYOUT = str(config.get("raw_source_layout", "examples"))
RAW_PACK_SIZE = int(config.get("raw_pack_size", 64))
if RAW_SOURCE_LAYOUT not in {"examples", "packs"}:
    raise ValueError("raw_source_layout must be 'examples' or 'packs'")
if RAW_PACK_SIZE <= 0:
    raise ValueError("raw_pack_size must be positive")
SCRIPT = "workflow/calibrate_checkpoint.py"
CODE = [SCRIPT, *sorted(str(path) for path in Path("src/dreems").glob("*.py"))]
FIT_INDICES = list(range(int(config.get("fit_start", 1024)), int(config.get("fit_stop", 2048))))
SELECTION_INDICES = list(
    range(int(config.get("selection_start", 2048)), int(config.get("selection_stop", 3072)))
)
VERIFICATION_INDICES = list(
    range(
        int(config.get("verification_start", 3072)),
        int(config.get("verification_stop", 4096)),
    )
)


localrules: all, canonicalize


def case_path(role, index):
    return f"{ROOT}/cases/{role}/{index}.npz"


def raw_case_path(wildcards):
    index = int(wildcards.index)
    if RAW_SOURCE_LAYOUT == "examples":
        return f"{RAW_DIR}/test/chunk_{index}.pkl"
    return f"{RAW_DIR}/test/pack_{index // RAW_PACK_SIZE}.pkl"


def raw_case_index(wildcards):
    if RAW_SOURCE_LAYOUT == "examples":
        return 0
    return int(wildcards.index) % RAW_PACK_SIZE


rule all:
    input:
        f"{ROOT}/checkpoint.calibrated.dreems.zip",
        f"{ROOT}/verification/report.json",
        f"{ROOT}/verification/by_sampling.csv",


rule canonicalize:
    input:
        checkpoint=CHECKPOINT,
        code=CODE,
    output:
        CANONICAL_CHECKPOINT,
    log:
        f"{ROOT}/logs/canonicalize.log",
    benchmark:
        f"{ROOT}/benchmarks/canonicalize.tsv",
    threads: 1
    shell:
        """
        set -o pipefail
        export PYTHONPATH="$PWD/src"
        uv run --frozen --no-sync python {SCRIPT:q} canonicalize \
            --checkpoint {input.checkpoint:q} \
            --output {output:q} \
            2>&1 | tee {log:q}
        """


rule case:
    input:
        checkpoint=CANONICAL_CHECKPOINT,
        raw=raw_case_path,
        code=CODE,
    output:
        temp(f"{ROOT}/cases/{{role}}/{{index}}.npz"),
    params:
        example_index=raw_case_index,
    threads: 2
    resources:
        slurm_account="jonth0",
        slurm_partition="standard",
        mem_mb=5000,
        runtime="30m",
    shell:
        """
        export JAX_PLATFORMS=cpu
        export JAX_PLATFORM_NAME=cpu
        export CUDA_VISIBLE_DEVICES=""
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        export PYTHONPATH="$PWD/src"
        export OMP_NUM_THREADS={threads}
        export OPENBLAS_NUM_THREADS={threads}
        export MKL_NUM_THREADS={threads}
        uv run --frozen --no-sync python {SCRIPT:q} case \
            --checkpoint {input.checkpoint:q} \
            --input {input.raw:q} \
            --role {wildcards.role:q} \
            --index {wildcards.index} \
            --example-index {params.example_index} \
            --output {output:q}
        """


rule fit:
    input:
        cases=[case_path("fit", index) for index in FIT_INDICES],
        script=SCRIPT,
    output:
        f"{ROOT}/fit/candidates.json",
    log:
        f"{ROOT}/logs/fit.log",
    benchmark:
        f"{ROOT}/benchmarks/fit.tsv",
    threads: 1
    resources:
        slurm_account="jonth0",
        slurm_partition="standard",
        mem_mb=4000,
        runtime="30m",
    shell:
        """
        set -o pipefail
        export PYTHONPATH="$PWD/src"
        uv run --frozen --no-sync python {input.script:q} fit \
            --inputs {input.cases:q} \
            --output {output:q} \
            2>&1 | tee {log:q}
        """


rule select:
    input:
        fit=f"{ROOT}/fit/candidates.json",
        cases=[case_path("selection", index) for index in SELECTION_INDICES],
        script=SCRIPT,
    output:
        f"{ROOT}/selection/selected.json",
    log:
        f"{ROOT}/logs/select.log",
    benchmark:
        f"{ROOT}/benchmarks/select.tsv",
    threads: 1
    resources:
        slurm_account="jonth0",
        slurm_partition="standard",
        mem_mb=4000,
        runtime="30m",
    shell:
        """
        set -o pipefail
        export PYTHONPATH="$PWD/src"
        uv run --frozen --no-sync python {input.script:q} select \
            --fit {input.fit:q} \
            --inputs {input.cases:q} \
            --output {output:q} \
            2>&1 | tee {log:q}
        """


rule attach:
    input:
        checkpoint=CANONICAL_CHECKPOINT,
        fit=f"{ROOT}/fit/candidates.json",
        selection=f"{ROOT}/selection/selected.json",
        script=SCRIPT,
    output:
        f"{ROOT}/checkpoint.calibrated.dreems.zip",
    log:
        f"{ROOT}/logs/attach.log",
    benchmark:
        f"{ROOT}/benchmarks/attach.tsv",
    threads: 1
    resources:
        slurm_account="jonth0",
        slurm_partition="standard",
        mem_mb=4000,
        runtime="10m",
    shell:
        """
        set -o pipefail
        export PYTHONPATH="$PWD/src"
        uv run --frozen --no-sync python {input.script:q} attach \
            --checkpoint {input.checkpoint:q} \
            --fit {input.fit:q} \
            --selection {input.selection:q} \
            --output {output:q} \
            2>&1 | tee {log:q}
        """


rule verify:
    input:
        checkpoint=f"{ROOT}/checkpoint.calibrated.dreems.zip",
        selection=f"{ROOT}/selection/selected.json",
        cases=[case_path("verification", index) for index in VERIFICATION_INDICES],
        script=SCRIPT,
    output:
        report=f"{ROOT}/verification/report.json",
        csv=f"{ROOT}/verification/by_sampling.csv",
    log:
        f"{ROOT}/logs/verify.log",
    benchmark:
        f"{ROOT}/benchmarks/verify.tsv",
    threads: 1
    resources:
        slurm_account="jonth0",
        slurm_partition="standard",
        mem_mb=4000,
        runtime="30m",
    shell:
        """
        set -o pipefail
        export PYTHONPATH="$PWD/src"
        uv run --frozen --no-sync python {input.script:q} verify \
            --checkpoint {input.checkpoint:q} \
            --selection {input.selection:q} \
            --inputs {input.cases:q} \
            --output-json {output.report:q} \
            --output-csv {output.csv:q} \
            2>&1 | tee {log:q}
        """
