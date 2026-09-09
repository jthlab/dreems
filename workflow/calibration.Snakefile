"""Fit, select, and verify posterior-SD calibration on disjoint test splits."""

from pathlib import Path
from itertools import combinations
from numbers import Integral


ROOT = str(Path(config["outdir"]).expanduser().resolve())
CHECKPOINT = str(Path(config["checkpoint"]).expanduser().resolve())
CANONICAL_CHECKPOINT = f"{ROOT}/checkpoint.canonical.dreems.zip"
RAW_DIR = str(Path(config["raw_source_dir"]).expanduser().resolve())
RAW_SOURCE_LAYOUT = str(config.get("raw_source_layout", "examples"))
RAW_PACK_SIZE = config.get("raw_pack_size", 64)
if RAW_SOURCE_LAYOUT not in {"examples", "packs"}:
    raise ValueError("raw_source_layout must be 'examples' or 'packs'")
if isinstance(RAW_PACK_SIZE, bool) or not isinstance(RAW_PACK_SIZE, Integral) or RAW_PACK_SIZE <= 0:
    raise ValueError("raw_pack_size must be positive")
PROJECT_ROOT = Path(workflow.current_basedir).parent.resolve()
SCRIPT = str(PROJECT_ROOT / "workflow/calibrate_checkpoint.py")
ENVIRONMENT = [str(PROJECT_ROOT / name) for name in ("pyproject.toml", "uv.lock")]
CALIBRATION_CODE = [str(PROJECT_ROOT / "src/dreems" / name) for name in ("calibration.py", "validation.py", "workflow_io.py")]
CHECKPOINT_CODE = CALIBRATION_CODE + [str(PROJECT_ROOT / "src/dreems" / name) for name in ("checkpoints.py", "model.py")]
CASE_CODE = CHECKPOINT_CODE + [str(PROJECT_ROOT / "src/dreems" / name) for name in (
    "features.py", "linearized.py", "corpus.py", "metrics.py",
)]
INTERVALS = {}
for role, start, stop in (("fit", 1024, 2048), ("selection", 2048, 3072), ("verification", 3072, 4096)):
    start, stop = config.get(f"{role}_start", start), config.get(f"{role}_stop", stop)
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in (start, stop)):
        raise ValueError(f"{role} bounds must be integers")
    if not 0 <= start < stop:
        raise ValueError(f"{role} range must be nonempty and nonnegative")
    INTERVALS[role] = range(start, stop)
for left, right in combinations(INTERVALS, 2):
    a, b = INTERVALS[left], INTERVALS[right]
    if max(a.start, b.start) < min(a.stop, b.stop):
        raise ValueError(f"calibration ranges overlap: {left} and {right}")


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
        script=SCRIPT,
        code=CHECKPOINT_CODE,
        environment=ENVIRONMENT,
    output:
        CANONICAL_CHECKPOINT,
    log:
        f"{ROOT}/logs/canonicalize.log",
    benchmark:
        f"{ROOT}/benchmarks/canonicalize.tsv",
    threads: 1
    params:
        project=str(PROJECT_ROOT),
    shell:
        """
        set -o pipefail
        uv --project {params.project:q} run --frozen --no-sync python {input.script:q} canonicalize \
            --checkpoint {input.checkpoint:q} \
            --output {output:q} \
            2>&1 | tee {log:q}
        """


rule case:
    input:
        checkpoint=CANONICAL_CHECKPOINT,
        raw=raw_case_path,
        script=SCRIPT,
        code=CASE_CODE,
        environment=ENVIRONMENT,
    output:
        temp(f"{ROOT}/cases/{{role}}/{{index}}.npz"),
    params:
        example_index=raw_case_index,
        project=str(PROJECT_ROOT),
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
        export OMP_NUM_THREADS={threads}
        export OPENBLAS_NUM_THREADS={threads}
        export MKL_NUM_THREADS={threads}
        uv --project {params.project:q} run --frozen --no-sync python {input.script:q} case \
            --checkpoint {input.checkpoint:q} \
            --input {input.raw:q} \
            --role {wildcards.role:q} \
            --index {wildcards.index} \
            --example-index {params.example_index} \
            --output {output:q}
        """


rule fit:
    input:
        cases=[case_path("fit", index) for index in INTERVALS["fit"]],
        script=SCRIPT,
        code=CALIBRATION_CODE,
        environment=ENVIRONMENT,
    output:
        f"{ROOT}/fit/candidates.json",
    log:
        f"{ROOT}/logs/fit.log",
    benchmark:
        f"{ROOT}/benchmarks/fit.tsv",
    params:
        project=str(PROJECT_ROOT),
    threads: 1
    resources:
        slurm_account="jonth0",
        slurm_partition="standard",
        mem_mb=4000,
        runtime="30m",
    shell:
        """
        set -o pipefail
        uv --project {params.project:q} run --frozen --no-sync python {input.script:q} fit \
            --inputs {input.cases:q} \
            --output {output:q} \
            2>&1 | tee {log:q}
        """


rule select:
    input:
        fit=f"{ROOT}/fit/candidates.json",
        cases=[case_path("selection", index) for index in INTERVALS["selection"]],
        script=SCRIPT,
        code=CALIBRATION_CODE,
        environment=ENVIRONMENT,
    output:
        f"{ROOT}/selection/selected.json",
    log:
        f"{ROOT}/logs/select.log",
    benchmark:
        f"{ROOT}/benchmarks/select.tsv",
    params:
        project=str(PROJECT_ROOT),
    threads: 1
    resources:
        slurm_account="jonth0",
        slurm_partition="standard",
        mem_mb=4000,
        runtime="30m",
    shell:
        """
        set -o pipefail
        uv --project {params.project:q} run --frozen --no-sync python {input.script:q} select \
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
        code=CHECKPOINT_CODE,
        environment=ENVIRONMENT,
    output:
        f"{ROOT}/checkpoint.calibrated.dreems.zip",
    log:
        f"{ROOT}/logs/attach.log",
    benchmark:
        f"{ROOT}/benchmarks/attach.tsv",
    params:
        project=str(PROJECT_ROOT),
    threads: 1
    resources:
        slurm_account="jonth0",
        slurm_partition="standard",
        mem_mb=4000,
        runtime="10m",
    shell:
        """
        set -o pipefail
        uv --project {params.project:q} run --frozen --no-sync python {input.script:q} attach \
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
        cases=[case_path("verification", index) for index in INTERVALS["verification"]],
        script=SCRIPT,
        code=CHECKPOINT_CODE,
        environment=ENVIRONMENT,
    output:
        report=f"{ROOT}/verification/report.json",
        csv=f"{ROOT}/verification/by_sampling.csv",
    log:
        f"{ROOT}/logs/verify.log",
    benchmark:
        f"{ROOT}/benchmarks/verify.tsv",
    params:
        project=str(PROJECT_ROOT),
    threads: 1
    resources:
        slurm_account="jonth0",
        slurm_partition="standard",
        mem_mb=4000,
        runtime="30m",
    shell:
        """
        set -o pipefail
        uv --project {params.project:q} run --frozen --no-sync python {input.script:q} verify \
            --checkpoint {input.checkpoint:q} \
            --selection {input.selection:q} \
            --inputs {input.cases:q} \
            --output-json {output.report:q} \
            --output-csv {output.csv:q} \
            2>&1 | tee {log:q}
        """
