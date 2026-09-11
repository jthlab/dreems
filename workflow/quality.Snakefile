"""Cluster-friendly package quality checks."""

from pathlib import Path


QUALITY_DIR = Path(config.get("output_dir", "artifacts/quality"))
QUALITY_INPUTS = [
    ".python-version",
    "README.md",
    "examples/dreems_quickstart.md",
    "pyproject.toml",
    "uv.lock",
    ".github/workflows/ci.yml",
    *sorted(
        str(path) for path in Path("workflow").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ),
    *sorted(str(path) for path in Path("src").rglob("*.py")),
    *sorted(str(path) for path in Path("src/dreems/data").rglob("*.dreems.zip")),
    *sorted(str(path) for path in Path("tests").rglob("*.py")),
]


rule all:
    input:
        f"{QUALITY_DIR}/ruff.ok",
        f"{QUALITY_DIR}/pytest.ok",
        f"{QUALITY_DIR}/workflow_dry_run.ok",
        f"{QUALITY_DIR}/dreems_quickstart.executed.ipynb",
        f"{QUALITY_DIR}/wheel_smoke.ok",


rule ruff:
    input:
        QUALITY_INPUTS
    output:
        touch(f"{QUALITY_DIR}/ruff.ok")
    benchmark:
        f"{QUALITY_DIR}/ruff.benchmark.tsv"
    threads: 1
    resources:
        slurm_account="jonth0",
        slurm_partition="standard",
        mem_mb=4000,
        runtime="10m",
    shell:
        "uv run --frozen --no-sync ruff check src tests workflow && "
        "uv run --frozen --no-sync ruff format --check src tests workflow"


rule pytest:
    input:
        QUALITY_INPUTS
    output:
        touch(f"{QUALITY_DIR}/pytest.ok")
    benchmark:
        f"{QUALITY_DIR}/pytest.benchmark.tsv"
    params:
        temporary=f"{QUALITY_DIR}/pytest-temp",
        junit=f"{QUALITY_DIR}/pytest.xml",
    threads: 4
    resources:
        slurm_account="jonth0",
        slurm_partition="standard",
        mem_mb=8000,
        runtime="60m",
    shell:
        "PYTHONPATH=src JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 "
        "uv run --frozen --no-sync pytest -q "
        "--basetemp {params.temporary:q} "
        "--junitxml {params.junit:q}"


rule workflow_dry_run:
    input:
        QUALITY_INPUTS
    output:
        touch(f"{QUALITY_DIR}/workflow_dry_run.ok")
    benchmark:
        f"{QUALITY_DIR}/workflow_dry_run.benchmark.tsv"
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
        f"{QUALITY_DIR}/dreems_quickstart.executed.ipynb"
    benchmark:
        f"{QUALITY_DIR}/dreems_quickstart.benchmark.tsv"
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
        touch(f"{QUALITY_DIR}/wheel_smoke.ok")
    benchmark:
        f"{QUALITY_DIR}/wheel_smoke.benchmark.tsv"
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
from pathlib import Path

installed = {{dist.metadata["Name"].lower() for dist in distributions()}}
unexpected_cuda = sorted(
    name for name in installed if name.startswith(("jax-cuda", "nvidia-"))
)
assert not unexpected_cuda, unexpected_cuda
assert find_spec("dreems._island_sim") is None
package = Path(dreems.__file__).parent
assert {{path.name for path in package.glob("*.py")}} == {{"__init__.py", "cli.py", "feems.py"}}
for name in ("data", "neural", "training", "inference", "evaluation", "utils"):
    assert (package / name / "__init__.py").is_file(), name
model = dreems.load_model()
assert model.checkpoint.model["model_name"] == "dreems"
assert model.checkpoint.model["node_dim"] == 109
assert model.checkpoint.model["edge_dim"] == 253
assert model.checkpoint.model["architecture"]["kind"] == "diffusion_band_surface_gate_v3"
assert model.checkpoint.params.surface_gate_hidden is not None
print("wheel and packaged checkpoint are usable")
'
        """
