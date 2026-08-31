"""Cluster-friendly package quality checks."""

from pathlib import Path


QUALITY_INPUTS = [
    ".python-version",
    "README.md",
    "examples/dreems_quickstart.md",
    "pyproject.toml",
    "uv.lock",
    ".github/workflows/ci.yml",
    *sorted(str(path) for path in Path("workflow").rglob("*") if path.is_file()),
    *sorted(str(path) for path in Path("src").rglob("*.py")),
    *sorted(str(path) for path in Path("src/dreems/data").rglob("*.dreems.zip")),
    *sorted(str(path) for path in Path("tests").rglob("*.py")),
]


rule all:
    input:
        "artifacts/quality/ruff.ok",
        "artifacts/quality/pytest.ok",
        "artifacts/quality/workflow_dry_run.ok",
        "artifacts/quality/dreems_quickstart.executed.ipynb",
        "artifacts/quality/wheel_smoke.ok",


rule ruff:
    input:
        QUALITY_INPUTS
    output:
        touch("artifacts/quality/ruff.ok")
    benchmark:
        "artifacts/quality/ruff.benchmark.tsv"
    threads: 1
    resources:
        slurm_account="jonth0",
        slurm_partition="standard",
        mem_mb=4000,
        runtime="10m",
    shell:
        "uv run --frozen --no-sync ruff check src tests && "
        "uv run --frozen --no-sync ruff format --check src tests"


rule pytest:
    input:
        QUALITY_INPUTS
    output:
        touch("artifacts/quality/pytest.ok")
    benchmark:
        "artifacts/quality/pytest.benchmark.tsv"
    threads: 4
    resources:
        slurm_account="jonth0",
        slurm_partition="standard",
        mem_mb=8000,
        runtime="60m",
    shell:
        "PYTHONPATH=src JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 "
        "uv run --frozen --no-sync pytest -q"


rule workflow_dry_run:
    input:
        QUALITY_INPUTS
    output:
        touch("artifacts/quality/workflow_dry_run.ok")
    benchmark:
        "artifacts/quality/workflow_dry_run.benchmark.tsv"
    threads: 1
    resources:
        slurm_account="jonth0",
        slurm_partition="standard",
        mem_mb=4000,
        runtime="10m",
    shell:
        "uv run --frozen --no-sync snakemake "
        "-s workflow/Snakefile "
        "--configfile workflow/config.example.yaml "
        "--profile workflow/profiles/local --cores 1 -n && "
        "bash -n workflow/run_great_lakes.sh workflow/run_maturin.sh"


rule quickstart_notebook:
    input:
        QUALITY_INPUTS
    output:
        "artifacts/quality/dreems_quickstart.executed.ipynb"
    benchmark:
        "artifacts/quality/dreems_quickstart.benchmark.tsv"
    threads: 4
    resources:
        slurm_account="jonth0",
        slurm_partition="standard",
        mem_mb=8000,
        runtime="30m",
    shell:
        "PYTHONPATH=src JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 "
        "uv run --frozen --no-sync jupytext "
        "--to ipynb --execute --output {output:q} "
        "examples/dreems_quickstart.md"


rule wheel_smoke:
    input:
        QUALITY_INPUTS
    output:
        touch("artifacts/quality/wheel_smoke.ok")
    benchmark:
        "artifacts/quality/wheel_smoke.benchmark.tsv"
    threads: 1
    resources:
        slurm_account="jonth0",
        slurm_partition="standard",
        mem_mb=4000,
        runtime="15m",
    shell:
        r"""
        set -euo pipefail
        tmp_dir="$(mktemp -d "${{TMPDIR:-/tmp}}/dreems-wheel-smoke.XXXXXX")"
        trap 'rm -rf "$tmp_dir"' EXIT
        uv build --wheel --out-dir "$tmp_dir/dist"
        uv venv "$tmp_dir/venv" --python 3.11
        uv pip install --python "$tmp_dir/venv/bin/python" "$tmp_dir"/dist/*.whl
        cd "$tmp_dir"
        "$tmp_dir/venv/bin/dreems" --help >/dev/null
        "$tmp_dir/venv/bin/python" -c '
import dreems
from importlib.metadata import distributions
from importlib.util import find_spec

installed = {{dist.metadata["Name"].lower() for dist in distributions()}}
unexpected_cuda = sorted(
    name for name in installed if name.startswith(("jax-cuda", "nvidia-"))
)
assert not unexpected_cuda, unexpected_cuda
assert find_spec("dreems._island_sim") is None
model = dreems.load_model()
assert model.checkpoint.model["model_name"] == "dreems"
assert model.checkpoint.model["node_dim"] == 109
assert model.checkpoint.model["edge_dim"] == 253
assert model.checkpoint.model["architecture"]["kind"] == "diffusion_band_surface_gate_v3"
assert model.checkpoint.params.surface_gate_hidden is not None
print("wheel and packaged checkpoint are usable")
'
        """
