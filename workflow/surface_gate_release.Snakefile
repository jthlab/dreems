"""Calibrate and promote the audited surface-gated production checkpoint."""

from pathlib import Path
from numbers import Integral


PROJECT_ROOT = Path(workflow.current_basedir).parent.resolve()
OUTDIR = Path(str(config["outdir"])).expanduser().resolve()
PUBLISH_DIR = Path(str(config["publish_dir"])).expanduser().resolve()
RELEASE_CHECKPOINT = PROJECT_ROOT / str(config["release_checkpoint"])
if "prediction_packs" in config and "prediction_ids" in config:
    raise ValueError("choose prediction_packs or prediction_ids, not both")
indices = config.get("prediction_packs", config.get("prediction_ids"))
if indices is not None:
    if (not isinstance(indices, list) or not indices
        or any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in indices)
        or len(set(indices)) != len(indices)):
        raise ValueError("prediction IDs must be nonempty, unique, nonnegative integers")
    field = "pack" if "prediction_packs" in config else "example"
    PREDICTIONS = [
        str(config["prediction_pattern"]).format(**{field: index}) for index in indices
    ]
else:
    prediction_start = config.get("prediction_start", 0)
    prediction_count = config["prediction_count"]
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in (prediction_start, prediction_count)):
        raise ValueError("prediction_start and prediction_count must be integers")
    if prediction_start < 0 or prediction_count <= 0:
        raise ValueError("prediction range must be nonempty and nonnegative")
    PREDICTIONS = [
        str(config["prediction_pattern"]).format(example=example)
        for example in range(prediction_start, prediction_start + prediction_count)
    ]
if len(set(PREDICTIONS)) != len(PREDICTIONS):
    raise ValueError("prediction_pattern must produce a distinct path for each ID")
REQUIRE_HASH = config.get("require_checkpoint_hash", False)
if not isinstance(REQUIRE_HASH, bool):
    raise ValueError("require_checkpoint_hash must be a boolean")
for name in ("expected_broad_examples", "expected_hard_examples", "calibration_seed"):
    value = config[name]
    minimum = 0 if name == "calibration_seed" else 1
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
CALIBRATED = OUTDIR / "dreems_surface_gate.calibrated.dreems.zip"
FIT = PUBLISH_DIR / "sd_calibration_fit.json"
SELECTION = PUBLISH_DIR / "sd_calibration_selection.json"
VERIFICATION = PUBLISH_DIR / "sd_calibration_verification.json"
VERIFICATION_CSV = PUBLISH_DIR / "sd_calibration_verification.csv"
CALIBRATION_SCRIPT = PROJECT_ROOT / "workflow" / "calibrate_surface_gate.py"
PROMOTION_SCRIPT = PROJECT_ROOT / "workflow" / "promote_surface_gate.py"
CALIBRATION_CODE = [str(PROJECT_ROOT / "src/dreems" / name) for name in ("inference/calibration.py", "neural/checkpoints.py", "evaluation/metrics.py", "utils/validation.py", "utils/workflow_io.py")]
PROMOTION_CODE = [str(PROJECT_ROOT / "src/dreems" / name) for name in ("inference/calibration.py", "neural/checkpoints.py", "neural/model.py", "utils/validation.py", "utils/workflow_io.py")]
ENVIRONMENT = [str(PROJECT_ROOT / name) for name in ("pyproject.toml", "uv.lock")]


rule all:
    input:
        str(RELEASE_CHECKPOINT),
        str(FIT),
        str(SELECTION),
        str(VERIFICATION),
        str(VERIFICATION_CSV),


rule calibrate_surface_gate:
    input:
        checkpoint=str(config["surface_gate_checkpoint"]),
        predictions=PREDICTIONS,
        rows=str(config["independent_rows"]),
        script=str(CALIBRATION_SCRIPT),
        code=CALIBRATION_CODE,
        environment=ENVIRONMENT,
    output:
        checkpoint=str(CALIBRATED),
        fit=str(FIT),
        selection=str(SELECTION),
        verification=str(VERIFICATION),
        verification_csv=str(VERIFICATION_CSV),
    threads: int(config["threads"])
    benchmark:
        str(OUTDIR / "benchmarks/calibrate.tsv")
    params:
        seed=config["calibration_seed"],
        project=str(PROJECT_ROOT),
        hash_option="--require-checkpoint-hash" if REQUIRE_HASH else "",
    resources:
        slurm_account="jonth0",
        slurm_partition=str(config["partition"]),
        mem_mb=int(config["mem_mb"]),
        runtime=str(config["runtime"]),
    shell:
        """
        export JAX_PLATFORMS=cpu
        export CUDA_VISIBLE_DEVICES=""
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        uv --project {params.project:q} run --frozen --no-sync python {input.script:q} \
            --checkpoint {input.checkpoint:q} \
            --predictions {input.predictions:q} \
            {params.hash_option} \
            --included-rows {input.rows:q} \
            --seed {params.seed} \
            --fit-output {output.fit:q} \
            --selection-output {output.selection:q} \
            --verification-output {output.verification:q} \
            --verification-csv {output.verification_csv:q} \
            --output-checkpoint {output.checkpoint:q}
        """


rule promote_surface_gate:
    input:
        core=str(config["core_checkpoint"]),
        gate=str(config["surface_gate_checkpoint"]),
        calibrated=str(CALIBRATED),
        training_configs=[str(path) for path in config["training_configs"]],
        gate_summary=str(config["surface_gate_training_summary"]),
        gate_history=str(config["surface_gate_training_history"]),
        core_summary=str(config["core_training_summary"]),
        core_history=str(config["core_training_history"]),
        calibration_fit=str(FIT),
        calibration_selection=str(SELECTION),
        calibration_verification=str(VERIFICATION),
        broad_protocol=str(config["broad_protocol"]),
        broad_report=str(config["broad_report"]),
        hard_protocol=str(config["hard_protocol"]),
        hard_report=str(config["hard_report"]),
        script=str(PROMOTION_SCRIPT),
        code=PROMOTION_CODE,
        environment=ENVIRONMENT,
    output:
        str(RELEASE_CHECKPOINT),
    threads: int(config["threads"])
    benchmark:
        str(OUTDIR / "benchmarks/promote.tsv")
    params:
        broad_examples=config["expected_broad_examples"],
        project=str(PROJECT_ROOT),
        hard_examples=config["expected_hard_examples"],
    resources:
        slurm_account="jonth0",
        slurm_partition=str(config["partition"]),
        mem_mb=int(config["mem_mb"]),
        runtime=str(config["runtime"]),
    shell:
        """
        export JAX_PLATFORMS=cpu
        export CUDA_VISIBLE_DEVICES=""
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        uv --project {params.project:q} run --frozen --no-sync python {input.script:q} \
            --core {input.core:q} \
            --surface-gate {input.gate:q} \
            --calibrated-surface-gate {input.calibrated:q} \
            --training-configs {input.training_configs:q} \
            --surface-gate-summary {input.gate_summary:q} \
            --surface-gate-history {input.gate_history:q} \
            --core-summary {input.core_summary:q} \
            --core-history {input.core_history:q} \
            --calibration-fit {input.calibration_fit:q} \
            --calibration-selection {input.calibration_selection:q} \
            --calibration-verification {input.calibration_verification:q} \
            --broad-protocol {input.broad_protocol:q} \
            --broad-report {input.broad_report:q} \
            --hard-protocol {input.hard_protocol:q} \
            --hard-report {input.hard_report:q} \
            --expected-broad-examples {params.broad_examples} \
            --expected-hard-examples {params.hard_examples} \
            --output {output:q}
        """
