#!/usr/bin/env bash
#SBATCH --job-name=dreems-snake
#SBATCH --account=jonth0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/controller-%j.out
#SBATCH --error=logs/slurm/controller-%j.err

set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${DREEMS_REPO:-${SLURM_SUBMIT_DIR:-$SCRIPT_ROOT}}"
CONFIG="${DREEMS_CONFIG:-workflow/config.production.yaml}"
CONFIG_FILES_RAW="${DREEMS_CONFIG_FILES:-}"
SNAKEFILE="${DREEMS_SNAKEFILE:-workflow/Snakefile}"

cd "$PROJECT_ROOT"
mkdir -p logs/slurm

if [[ "${DREEMS_BUILD_RUST:-0}" == "1" ]]; then
    module load rust/1.92.0 2>/dev/null || module load rust 2>/dev/null || true
    uv sync --frozen --extra cuda --group workflow --group rust --group dev
    uv run maturin develop --release --manifest-path rust/island_sim/Cargo.toml
fi

uv run --frozen --no-sync python - <<'PY'
try:
    import dreems._island_sim  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dreems._island_sim Rust extension. Build it with:\n"
        "  module load rust/1.92.0\n"
        "  uv sync --frozen --extra cuda --group workflow --group rust --group dev\n"
        "  uv run maturin develop --release --manifest-path rust/island_sim/Cargo.toml\n"
        "or submit with DREEMS_BUILD_RUST=1."
    ) from exc
PY

extra_config=()
if [[ -n "${DREEMS_RUN_ROOT:-}" ]]; then
    extra_config=(--config "outdir=${DREEMS_RUN_ROOT}")
fi

config_files=("$CONFIG")
if [[ -n "$CONFIG_FILES_RAW" ]]; then
    IFS=':' read -r -a config_files <<< "$CONFIG_FILES_RAW"
fi

exec uv run --frozen --no-sync snakemake \
    -s "$SNAKEFILE" \
    --configfile "${config_files[@]}" \
    --profile workflow/profiles/slurm \
    "${extra_config[@]}" \
    --rerun-incomplete \
    "$@"
