#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${DREEMS_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${DREEMS_CONFIG:-workflow/config.production.yaml}"
LOCAL_CORES="${DREEMS_LOCAL_CORES:-60}"
LOCAL_MEM_MB="${DREEMS_LOCAL_MEM_MB:-240000}"
LOCAL_GPUS="${DREEMS_LOCAL_GPUS:-1}"

cd "$PROJECT_ROOT"

if [[ "${DREEMS_BUILD_RUST:-0}" == "1" ]]; then
    uv sync --frozen --extra cuda --group workflow --group rust --group dev
    uv run maturin develop --release --manifest-path rust/island_sim/Cargo.toml
fi

uv run --frozen --no-sync python - <<'PY'
try:
    import dreems._island_sim  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dreems._island_sim Rust extension. Build it with:\n"
        "  uv sync --frozen --extra cuda --group workflow --group rust --group dev\n"
        "  uv run maturin develop --release --manifest-path rust/island_sim/Cargo.toml\n"
        "or run with DREEMS_BUILD_RUST=1."
    ) from exc
PY

extra_config=()
if [[ -n "${DREEMS_RUN_ROOT:-}" ]]; then
    extra_config=(--config "outdir=${DREEMS_RUN_ROOT}")
fi

exec uv run --frozen --no-sync snakemake \
    -s workflow/Snakefile \
    --configfile "$CONFIG" \
    --profile workflow/profiles/local \
    --cores "$LOCAL_CORES" \
    --resources "mem_mb=${LOCAL_MEM_MB}" "gpu=${LOCAL_GPUS}" \
    "${extra_config[@]}" \
    --rerun-incomplete \
    "$@"
