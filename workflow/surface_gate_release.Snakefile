"""Calibrate and promote the audited surface-gated production checkpoint."""

from pathlib import Path


PROJECT_ROOT = Path(workflow.basedir).parent.resolve()
OUTDIR = Path(str(config["outdir"])).expanduser()
PUBLISH_DIR = Path(str(config["publish_dir"])).expanduser()
RELEASE_CHECKPOINT = PROJECT_ROOT / str(config["release_checkpoint"])
PACKS = tuple(int(pack) for pack in config["prediction_packs"])
PREDICTIONS = [
    str(config["prediction_pattern"]).format(pack=pack)
    for pack in PACKS
]
CALIBRATED = OUTDIR / "dreems_surface_gate.calibrated.dreems.zip"
FIT = PUBLISH_DIR / "sd_calibration_fit.json"
SELECTION = PUBLISH_DIR / "sd_calibration_selection.json"
VERIFICATION = PUBLISH_DIR / "sd_calibration_verification.json"
VERIFICATION_CSV = PUBLISH_DIR / "sd_calibration_verification.csv"
CALIBRATION_SCRIPT = PROJECT_ROOT / "workflow" / "calibrate_surface_gate.py"
PROMOTION_SCRIPT = PROJECT_ROOT / "workflow" / "promote_surface_gate.py"


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
    output:
        checkpoint=str(CALIBRATED),
        fit=str(FIT),
        selection=str(SELECTION),
        verification=str(VERIFICATION),
        verification_csv=str(VERIFICATION_CSV),
    threads: int(config["threads"])
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
        uv --project {PROJECT_ROOT:q} run --frozen --no-sync python {input.script:q} \
            --checkpoint {input.checkpoint:q} \
            --predictions {input.predictions:q} \
            --included-rows {input.rows:q} \
            --seed {config[calibration_seed]} \
            --fit-output {output.fit:q} \
            --selection-output {output.selection:q} \
            --verification-output {output.verification:q} \
            --verification-csv {output.verification_csv:q} \
            --output-checkpoint {output.checkpoint:q}
        """


rule promote_surface_gate:
    input:
        sampling=str(config["sampling_robustness_checkpoint"]),
        gate=str(config["surface_gate_checkpoint"]),
        calibrated=str(CALIBRATED),
        gate_config=str(config["surface_gate_config"]),
        gate_summary=str(config["surface_gate_training_summary"]),
        gate_history=str(config["surface_gate_training_history"]),
        sampling_summary=str(config["sampling_robustness_training_summary"]),
        sampling_history=str(config["sampling_robustness_training_history"]),
        calibration_fit=str(FIT),
        calibration_selection=str(SELECTION),
        calibration_verification=str(VERIFICATION),
        independent=str(config["independent_report"]),
        broad=str(config["broad_report"]),
        scenario=str(config["scenario_report"]),
        script=str(PROMOTION_SCRIPT),
    output:
        str(RELEASE_CHECKPOINT),
    threads: int(config["threads"])
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
        uv --project {PROJECT_ROOT:q} run --frozen --no-sync python {input.script:q} \
            --sampling-robustness {input.sampling:q} \
            --surface-gate {input.gate:q} \
            --calibrated-surface-gate {input.calibrated:q} \
            --surface-gate-config {input.gate_config:q} \
            --surface-gate-summary {input.gate_summary:q} \
            --surface-gate-history {input.gate_history:q} \
            --sampling-robustness-summary {input.sampling_summary:q} \
            --sampling-robustness-history {input.sampling_history:q} \
            --calibration-fit {input.calibration_fit:q} \
            --calibration-selection {input.calibration_selection:q} \
            --calibration-verification {input.calibration_verification:q} \
            --independent-report {input.independent:q} \
            --broad-report {input.broad:q} \
            --scenario-report {input.scenario:q} \
            --output {output:q}
        """
