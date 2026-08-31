"""Regenerate the sampling hard-case notebook with an explicit checkpoint."""

import hashlib
from pathlib import Path

import nbformat


SOURCE = Path(str(config["source_notebook"])).expanduser().resolve()
CHECKPOINT = Path(str(config["checkpoint"])).expanduser().resolve()
OUTDIR = Path(str(config["outdir"])).expanduser().resolve()
WORKDIR = Path(str(config.get("workdir", OUTDIR / "work"))).expanduser().resolve()
PREPARED = WORKDIR / "feems_scenarios.prepared.ipynb"
EXECUTED = OUTDIR / "feems_scenarios_sampling_robustness.ipynb"
HTML = OUTDIR / "feems_scenarios_sampling_robustness.html"
PACKAGE_SOURCES = sorted(str(path) for path in Path("src/dreems").glob("*.py"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


localrules: all, prepare_notebook


rule all:
    input:
        str(EXECUTED),
        str(HTML),


rule prepare_notebook:
    input:
        source=str(SOURCE),
        checkpoint=str(CHECKPOINT),
    output:
        temp(str(PREPARED)),
    run:
        notebook = nbformat.read(input.source, as_version=4)
        source_hash = sha256(Path(input.source))
        checkpoint_hash = sha256(Path(input.checkpoint))

        # The source notebook contains one adjacent duplicate anisotropy cell.
        # Remove only consecutive, byte-identical code cells.
        cells = []
        for cell in notebook.cells:
            if (
                cells
                and cell.cell_type == "code"
                and cells[-1].cell_type == "code"
                and cell.source == cells[-1].source
            ):
                continue
            cells.append(cell)
        notebook.cells = cells

        setup = notebook.cells[0]
        if setup.cell_type != "code" or "def create_lattice" not in setup.source:
            raise ValueError("the source notebook no longer has the expected setup cell")
        setup.source = setup.source.replace(
            "import networkx as nx\n",
            "import functools\nfrom pathlib import Path\n\nimport networkx as nx\n",
            1,
        )
        setup.source = setup.source.replace(
            "def create_lattice(",
            "@functools.lru_cache(maxsize=None)\ndef create_lattice(",
            1,
        )
        setup.source += (
            "\n\nCHECKPOINT = Path(" + repr(str(CHECKPOINT)) + ")\n"
            "PREDICTOR = dreems.load_model(CHECKPOINT)\n"
            "print(f'Checkpoint: {CHECKPOINT}')\n"
        )

        replacements = 0
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            old = "surface = dreems.infer(data)"
            replacements += cell.source.count(old)
            cell.source = cell.source.replace(
                old,
                "surface = dreems.infer(data, checkpoint=PREDICTOR)",
            )
            cell.outputs = []
            cell.execution_count = None
        if replacements != 12:
            raise ValueError(f"expected 12 inference cells after deduplication, found {replacements}")

        notebook.cells.insert(
            0,
            nbformat.v4.new_markdown_cell(
                "# Sampling hard-case audit\n\n"
                "This is the original FEEMS-scenario notebook rerun with the "
                "sampling-robustness DREEMS candidate. The msprime parameters, "
                "random seeds, MAF filtering, and sampling masks are unchanged. "
                "Repeated masks reuse the same simulated SNP data.\n\n"
                f"- Source notebook SHA-256: `{source_hash}`\n"
                f"- Checkpoint SHA-256: `{checkpoint_hash}`"
            ),
        )
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, output[0])


rule execute_notebook:
    input:
        notebook=str(PREPARED),
        checkpoint=str(CHECKPOINT),
        package=PACKAGE_SOURCES,
    output:
        str(EXECUTED),
    benchmark:
        str(OUTDIR / "benchmarks" / "execute.tsv")
    threads: int(config.get("threads", 4))
    resources:
        mem_mb=int(config.get("mem_mb", 32000)),
        runtime=str(config.get("runtime", "12h")),
        slurm_partition=str(config.get("partition", "standard")),
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
        mkdir -p "$(dirname {output:q})"
        uv run --frozen --no-sync jupyter nbconvert \
            --to notebook \
            --execute \
            --ExecutePreprocessor.timeout=-1 \
            --output {output:q} \
            {input.notebook:q}
        """


rule render_html:
    input:
        notebook=str(EXECUTED),
    output:
        str(HTML),
    benchmark:
        str(OUTDIR / "benchmarks" / "html.tsv")
    threads: 1
    resources:
        mem_mb=4000,
        runtime="30m",
        slurm_partition=str(config.get("partition", "standard")),
        slurm_account="jonth0",
    shell:
        """
        set -euo pipefail
        uv run --frozen --no-sync jupyter nbconvert \
            --to html \
            --output {output:q} \
            {input.notebook:q}
        """
