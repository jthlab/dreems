"""Test rule invalidation using Snakemake's touch executor, without scientific fits."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Keep mutations and Snakemake state in an isolated copy on scratch."""
    project = tmp_path / "project"
    for folder in ("workflow", "src/dreems", "tests"):
        (project / folder).mkdir(parents=True, exist_ok=True)
    for source in (REPO / "src/dreems").rglob("*.py"):
        destination = project / source.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for name in (
        "h2h.Snakefile",
        "real_data_gallery.Snakefile",
        "run_real_data.py",
        "render_real_data_gallery.py",
        "calibration.Snakefile",
        "calibrate_checkpoint.py",
        "surface_gate_release.Snakefile",
        "calibrate_surface_gate.py",
        "promote_surface_gate.py",
        "surface_gate_comparison.Snakefile",
        "hard_case_notebook.Snakefile",
        "snp_range_comparison.Snakefile",
    ):
        shutil.copy2(REPO / "workflow" / name, project / "workflow" / name)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(REPO / name, project / name)
    shutil.copy2(REPO / "tests/test_feems_official_parity.py", project / "tests")
    (project / "checkpoint.zip").write_bytes(b"checkpoint fixture")
    shutil.copytree(REPO / "workflow/notebooks", project / "workflow/notebooks")
    return project


def run_workflow(
    project: Path, config: dict, *, dry_run: bool = False, execute: bool = False
) -> str:
    path = project / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    command = [
        "uv",
        "--project",
        str(REPO),
        "run",
        "--frozen",
        "--no-sync",
        "snakemake",
        "-s",
        str(project / "audit.Snakefile"),
        "--configfile",
        str(path),
        "--executor",
        "local",
        "--cores",
        "1",
        "--rerun-incomplete",
        "--notemp",
        "--printshellcmds",
        "all",
    ]
    if dry_run:
        command.append("--dry-run")
    elif not execute:
        command.append("--touch")
        if not (project / "run_0.log").exists():
            command.append("--forceall")
    result = subprocess.run(
        command,
        cwd=project,
        text=True,
        capture_output=True,
        timeout=120,
        env={**os.environ, "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1"},
    )
    logs = len(list(project.glob("run_*.log")))
    (project / f"run_{logs}.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout + result.stderr


def stub_rules(
    project: Path,
    snakefile: str,
    rules: list[str],
    targets: list[Path],
    outputs: list[Path],
) -> None:
    """Import unchanged rules and seed their outputs for the touch executor."""
    lines = [
        f"module production:\n    snakefile: {str(project / 'workflow' / snakefile)!r}\n    config: config\n"
    ]
    for name in rules:
        lines.append(f"use rule {name} from production as {name}\n")
    lines.append(f"rule all:\n    input: {[str(path) for path in targets]!r}\n")
    (project / "audit.Snakefile").write_text("\n".join(lines))
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"dependency fixture, not a scientific result")


def test_h2h_invalidates_only_the_affected_method(workspace: Path) -> None:
    out = workspace / "output"
    inputs = workspace / "inputs"
    inputs.mkdir()
    (inputs / "example_0.pkl").write_bytes(b"immutable raw example")
    (inputs / "example_0.json").write_text(json.dumps({"num_sampled_nodes": 2}))
    config = dict(
        outdir=str(out),
        checkpoint=str(workspace / "checkpoint.zip"),
        example_count=1,
        input_pattern=str(inputs / "example_{example}.pkl"),
        metadata_pattern=str(inputs / "example_{example}.json"),
    )
    stub_rules(
        workspace,
        "h2h.Snakefile",
        ["feems_cv_fold", "feems_case", "dreems_case"],
        [out / "feems/example_0.json", out / "dreems/example_0.json"],
        [
            out / method / f"example_0.{ext}"
            for method in ("feems", "dreems")
            for ext in ("json", "npz")
        ]
        + [out / f"feems_cv/example_0/fold_{fold:04d}.npz" for fold in range(2)],
    )
    run_workflow(workspace, config)
    assert "Nothing to be done" in run_workflow(workspace, config, dry_run=True)
    feems = out / "feems/example_0.json"
    original_feems_mtime = feems.stat().st_mtime_ns
    for name in (
        "checkpoint.zip",
        "src/dreems/neural/model.py",
        "src/dreems/neural/features.py",
        "src/dreems/inference/api.py",
        "src/dreems/inference/calibration.py",
        "src/dreems/evaluation/h2h.py",
    ):
        path = workspace / name
        path.write_bytes(path.read_bytes() + b"\n# dependency change\n")
        planned = run_workflow(workspace, config, dry_run=True)
        assert "rule dreems_case:" in planned, name
        assert "rule feems_case:" not in planned, name
        assert "rule feems_cv_fold:" not in planned, name
        run_workflow(workspace, config)
        assert feems.stat().st_mtime_ns == original_feems_mtime
    plotter = workspace / "src/dreems/evaluation/benchmark_gallery.py"
    plotter.write_text(plotter.read_text() + "\n# render change\n")
    assert "Nothing to be done" in run_workflow(workspace, config, dry_run=True)
    for name in ("feems.py", "evaluation/feems_benchmark.py"):
        path = workspace / "src/dreems" / name
        path.write_text(path.read_text() + "\n# FEEMS change\n")
        planned = run_workflow(workspace, config, dry_run=True)
        assert "rule feems_case:" in planned
        assert "rule feems_cv_fold:" in planned
        assert "rule dreems_case:" not in planned
        run_workflow(workspace, config)


def test_calibration_module_tracks_only_relevant_source(workspace: Path) -> None:
    out, raw = workspace / "calibration", workspace / "raw"
    (raw / "test").mkdir(parents=True)
    for index in range(3):
        (raw / "test" / f"chunk_{index}.pkl").write_bytes(b"raw fixture")
    config = {
        "outdir": str(out),
        "checkpoint": str(workspace / "checkpoint.zip"),
        "raw_source_dir": str(raw),
        "fit_start": 0,
        "fit_stop": 1,
        "selection_start": 1,
        "selection_stop": 2,
        "verification_start": 2,
        "verification_stop": 3,
    }
    outputs = [
        out / "checkpoint.canonical.dreems.zip",
        out / "checkpoint.calibrated.dreems.zip",
        out / "fit/candidates.json",
        out / "selection/selected.json",
        out / "verification/report.json",
        out / "verification/by_sampling.csv",
        *[
            out / "cases" / role / f"{index}.npz"
            for index, role in enumerate(("fit", "selection", "verification"))
        ],
    ]
    stub_rules(
        workspace,
        "calibration.Snakefile",
        ["canonicalize", "case", "fit", "select", "attach", "verify"],
        [out / "verification/report.json", out / "verification/by_sampling.csv"],
        outputs,
    )
    run_workflow(workspace, config)
    assert "Nothing to be done" in run_workflow(workspace, config, dry_run=True)
    for name in ("feems.py", "evaluation/benchmark_gallery.py", "data/simulation.py"):
        source = workspace / "src/dreems" / name
        source.write_text(source.read_text() + "\n# unrelated edit\n")
        assert "Nothing to be done" in run_workflow(workspace, config, dry_run=True), name
    source = workspace / "src/dreems/neural/features.py"
    source.write_text(source.read_text() + "\n# feature edit\n")
    planned = run_workflow(workspace, config, dry_run=True)
    assert "rule case:" in planned
    assert "rule canonicalize:" not in planned
    run_workflow(workspace, config)
    source = workspace / "uv.lock"
    source.write_text(source.read_text() + "\n# environment edit\n")
    planned = run_workflow(workspace, config, dry_run=True)
    assert "rule canonicalize:" in planned
    assert "rule verify:" in planned


def test_promotion_module_does_not_refit_calibration_after_model_edit(workspace: Path) -> None:
    out = workspace / "release"
    config = {
        "outdir": str(out),
        "publish_dir": str(out / "report"),
        "release_checkpoint": str(out / "candidate.zip"),
        "core_checkpoint": str(workspace / "checkpoint.zip"),
        "surface_gate_checkpoint": str(workspace / "checkpoint.zip"),
        "prediction_pattern": str(workspace / "example_{example}.npz"),
        "prediction_count": 3,
        "calibration_seed": 7,
        "expected_broad_examples": 3,
        "expected_hard_examples": 1,
        "threads": 1,
        "mem_mb": 1000,
        "runtime": "10m",
        "partition": "standard",
    }
    for name in (
        "independent_rows",
        "core_training_summary",
        "core_training_history",
        "surface_gate_training_summary",
        "surface_gate_training_history",
        "broad_protocol",
        "broad_report",
        "hard_protocol",
        "hard_report",
    ):
        path = workspace / f"{name}.json"
        path.write_text("{}")
        config[name] = str(path)
    config["training_configs"] = [config["core_training_summary"]]
    for index in range(3):
        (workspace / f"example_{index}.npz").write_bytes(b"prediction fixture")
    outputs = [out / "candidate.zip", out / "dreems_surface_gate.calibrated.dreems.zip"]
    outputs += [
        out / "report" / f"sd_calibration_{name}.{ext}"
        for name, ext in (
            ("fit", "json"),
            ("selection", "json"),
            ("verification", "json"),
            ("verification", "csv"),
        )
    ]
    stub_rules(
        workspace,
        "surface_gate_release.Snakefile",
        ["calibrate_surface_gate", "promote_surface_gate"],
        [out / "candidate.zip"],
        outputs,
    )
    run_workflow(workspace, config)
    assert "Nothing to be done" in run_workflow(workspace, config, dry_run=True)
    source = workspace / "src/dreems/neural/model.py"
    source.write_text(source.read_text() + "\n# model edit\n")
    planned = run_workflow(workspace, config, dry_run=True)
    assert "rule promote_surface_gate:" in planned
    assert "rule calibrate_surface_gate:" not in planned


def test_real_data_render_parameters_do_not_refit(workspace: Path) -> None:
    out, prepared, published = (workspace / name for name in ("output", "prepared", "published"))
    prepared.mkdir()
    for suffix in ("npz", "json"):
        (prepared / f"study.{suffix}").write_bytes(b"prepared input")
    settings = dict(
        grid_side=20,
        topo_zoom=3,
        topo_max_tiles=4,
        topo_max_pixels=80,
        dpi=100,
        datasets={"study": {}},
    )
    render = workspace / "render.yaml"
    render.write_text(yaml.safe_dump(settings))
    config = dict(
        outdir=str(out),
        publish_dir=str(published),
        prepared_dir=str(prepared),
        checkpoint=str(workspace / "checkpoint.zip"),
        render_config=str(render),
        literature_report=str(workspace / "literature.md"),
        topo_cache=str(workspace / "tiles"),
        datasets=["study"],
    )
    stub_rules(
        workspace,
        "real_data_gallery.Snakefile",
        ["run_dreems", "render_case"],
        [published / "gallery/study.png"],
        [out / f"fits/study.{ext}" for ext in ("npz", "json")]
        + [published / f"gallery/study.{ext}" for ext in ("png", "pdf")]
        + [published / "gallery/cases/study.json"],
    )
    run_workflow(workspace, config)
    fit_time = (out / "fits/study.npz").stat().st_mtime_ns
    assert "Nothing to be done" in run_workflow(workspace, config, dry_run=True)
    for name in ("grid_side", "topo_zoom", "topo_max_tiles", "topo_max_pixels", "dpi"):
        settings[name] += 1
        render.write_text(yaml.safe_dump(settings))
        planned = run_workflow(workspace, config, dry_run=True)
        assert "rule render_case:" in planned, name
        assert "rule run_dreems:" not in planned, name
        run_workflow(workspace, config)
        assert (out / "fits/study.npz").stat().st_mtime_ns == fit_time
    for name in ("data/simulation.py", "evaluation/benchmark_gallery.py", "feems.py"):
        path = workspace / "src/dreems" / name
        path.write_text(path.read_text() + "\n# unrelated change\n")
    assert "Nothing to be done" in run_workflow(workspace, config, dry_run=True)
    path = workspace / "src/dreems/inference/api.py"
    path.write_text(path.read_text() + "\n# inference change\n")
    planned = run_workflow(workspace, config, dry_run=True)
    assert "rule run_dreems:" in planned
    assert "rule render_case:" in planned


def test_scenario_simulator_changes_invalidate_generation(workspace: Path) -> None:
    shutil.copytree(
        REPO / "rust/island_sim",
        workspace / "rust/island_sim",
        ignore=shutil.ignore_patterns("target"),
    )
    config = yaml.safe_load((REPO / "workflow/config.feems_scenarios.yaml").read_text())
    config["scenario_audit"].update(replicates=1, truths=["barrier"], masks=["full"])
    out = workspace / "scenarios"
    config.update(
        outdir=str(out),
        checkpoint=str(workspace / "checkpoint.zip"),
        example_count=1,
        scenario_config=str(workspace / "config.yaml"),
        input_pattern=str(out / "example_{example}.pkl"),
        metadata_pattern=str(out / "example_{example}.json"),
    )
    outputs = [out / "example_0.pkl", out / "example_0.json"]
    stub_rules(workspace, "h2h.Snakefile", ["generate_scenario_case"], outputs, outputs)
    run_workflow(workspace, config)
    assert "Nothing to be done" in run_workflow(workspace, config, dry_run=True)
    path = workspace / "src/dreems/neural/model.py"
    path.write_text(path.read_text() + "\n# not a simulator dependency\n")
    assert "Nothing to be done" in run_workflow(workspace, config, dry_run=True)
    for name in (
        "src/dreems/data/island_sim.py",
        "src/dreems/utils/validation.py",
        "rust/island_sim/src/lib.rs",
    ):
        path = workspace / name
        comment = "//" if path.suffix == ".rs" else "#"
        path.write_text(path.read_text() + f"\n{comment} dependency changed\n")
        assert "rule generate_scenario_case:" in run_workflow(workspace, config, dry_run=True)
        run_workflow(workspace, config)


def snp_comparison_config(workspace: Path) -> dict:
    """Use one partial reused pack to exercise imported SNP-comparison rules."""
    for name in ("prepared", "raw", "reference"):
        (workspace / name).mkdir()
    (workspace / "prepared/pack_0.pkl").write_bytes(b"prepared fixture")
    (workspace / "raw/pack_0.pkl").write_bytes(b"raw fixture")
    (workspace / "raw/pack_0.json").write_text('{"rows": []}')
    (workspace / "features.json").write_text("{}")
    (workspace / "reference/index.html").write_text("<h1>Reference</h1>")
    (workspace / "reference/panel.png").write_bytes(b"saved image")
    (workspace / "reference/rows.csv").write_text("index,num_snps\n0,1000\n")
    return dict(
        outdir=str(workspace / "work"),
        publish_dir=str(workspace / "published"),
        router_publish_dir=str(workspace / "router"),
        router_h2h_publish_dir=str(workspace / "router_h2h"),
        router_h2h_reference_gallery=str(workspace / "reference"),
        router_h2h_reference_rows=str(workspace / "reference/rows.csv"),
        feature_config=str(workspace / "features.json"),
        old_checkpoint=str(workspace / "checkpoint.zip"),
        new_checkpoint=str(workspace / "checkpoint.zip"),
        raw_pack_pattern=str(workspace / "raw/pack_{pack}.pkl"),
        raw_metadata_pattern=str(workspace / "raw/pack_{pack}.json"),
        reused_prepared_dir=str(workspace / "prepared"),
        reused_prepared_packs=1,
        test_examples=3,
        pack_size=4,
        eval_batch_size=1,
    )


def test_snp_module_tracks_evaluation_and_reference_files(workspace: Path) -> None:
    config = snp_comparison_config(workspace)
    evaluation = workspace / "work/evaluation/new/pack_0"
    outputs = [evaluation / name for name in ("summary.json", "summary.csv", "predictions.npz")]
    target = workspace / "router_h2h/gallery"
    target.mkdir(parents=True)
    stub_rules(
        workspace,
        "snp_range_comparison.Snakefile",
        ["evaluate_pack", "router_h2h_gallery"],
        [outputs[0], target],
        outputs,
    )
    planned = run_workflow(workspace, config)
    assert "--max-examples 3" in planned
    assert "--replicate-offset 0" in planned
    assert "Nothing to be done" in run_workflow(workspace, config, dry_run=True)
    path = workspace / "src/dreems/inference/calibration.py"
    path.write_text(path.read_text() + "\n# calibration changed\n")
    planned = run_workflow(workspace, config, dry_run=True)
    assert "rule evaluate_pack:" in planned
    assert "rule router_h2h_gallery:" not in planned
    run_workflow(workspace, config)
    (workspace / "reference/panel.png").write_bytes(b"updated existing image")
    planned = run_workflow(workspace, config, dry_run=True)
    assert "rule router_h2h_gallery:" in planned
    assert "rule evaluate_pack:" not in planned


def test_snp_reports_execute_as_imported_rules(workspace: Path) -> None:
    config = snp_comparison_config(workspace)
    (workspace / ".venv").symlink_to(REPO / ".venv", target_is_directory=True)
    published = workspace / "published"
    for model, offset in (("old", 0.0), ("new", 0.1)):
        path = published / "evaluations" / model / "summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "rows": [
                        dict(
                            replicate_global=index,
                            edge_log_corr=0.5 + index * 0.01 + offset,
                            edge_log_mse=0.7 - index * 0.01 - offset,
                        )
                        for index in range(3)
                    ]
                }
            )
        )
    (workspace / "raw/pack_0.json").write_text(
        json.dumps(
            {
                "rows": [
                    dict(index=index, num_snps=count) for index, count in enumerate((50, 129, 513))
                ]
            }
        )
    )
    stub_rules(
        workspace,
        "snp_range_comparison.Snakefile",
        ["compare_checkpoints", "router_report"],
        [workspace / "router/summary.json"],
        [],
    )
    run_workflow(workspace, config, execute=True)
    report = json.loads((workspace / "router/summary.json").read_text())
    assert report["examples"] == 3
    assert report["threshold"] == 512
    assert "Nothing to be done" in run_workflow(workspace, config, dry_run=True)


def comparison_config(workspace: Path) -> dict:
    """Point an imported comparison workflow at local placeholder artifacts."""
    config = yaml.safe_load((REPO / "workflow/config.surface_gate_comparison.yaml").read_text())
    config.update(
        outdir=str(workspace / "work"),
        publish_dir=str(workspace / "published & 'quoted' | path"),
        core_checkpoint=str(workspace / "checkpoint.zip"),
        gated_checkpoint=str(workspace / "checkpoint.zip"),
        core_scenario_bundle=str(workspace / "core.npz"),
        gated_scenario_pattern=str(workspace / "gate_{example}.npz"),
        core_broad_pattern=str(workspace / "core_{example}.npz"),
        gated_broad_pattern=str(workspace / "gate_{example}.npz"),
        feems_broad_pattern=str(workspace / "feems_{example}.npz"),
    )
    (workspace / "core.npz").write_bytes(b"dependency fixture")
    for index in range(12):
        (workspace / f"gate_{index}.npz").write_bytes(b"dependency fixture")
    return config


def test_comparison_module_tracks_report_dependencies(workspace: Path) -> None:
    config = comparison_config(workspace)
    published = Path(config["publish_dir"])
    outputs = [
        published / name
        for name in ("scenario_comparison.npz", "scenario_rows.csv", "scenario_report.json")
    ]
    stub_rules(
        workspace, "surface_gate_comparison.Snakefile", ["build_scenario_bundle"], outputs, outputs
    )
    run_workflow(workspace, config)
    assert "Nothing to be done" in run_workflow(workspace, config, dry_run=True)
    (workspace / "src/dreems/neural/model.py").write_text("# unrelated model edit\n")
    assert "Nothing to be done" in run_workflow(workspace, config, dry_run=True)
    for name in ("src/dreems/utils/workflow_io.py", "src/dreems/utils/validation.py", "uv.lock"):
        path = workspace / name
        path.write_text(path.read_text() + "\n# dependency edit\n")
        assert "rule build_scenario_bundle:" in run_workflow(workspace, config, dry_run=True)
        run_workflow(workspace, config)


def test_notebook_preparation_quotes_paths_and_preserves_source(workspace: Path) -> None:
    import ast

    import nbformat

    config = comparison_config(workspace)
    bundle = Path(config["publish_dir"]) / "scenario_comparison.npz"
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"read by the notebook, not during preparation")
    output = workspace / "work/notebook/feems_comparison_surface_gate.prepared.ipynb"
    stub_rules(workspace, "surface_gate_comparison.Snakefile", ["prepare_notebook"], [output], [])
    source = workspace / "workflow/notebooks/feems_comparison_surface_gate.md"
    before = source.read_bytes()
    run_workflow(workspace, config, execute=True)
    notebook = nbformat.read(output, as_version=4)
    assignments = [
        node
        for cell in notebook.cells
        if cell.cell_type == "code"
        for node in ast.walk(ast.parse(cell.source))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "BUNDLE_PATH" for target in node.targets
        )
    ]
    assert len(assignments) == 1
    assert ast.literal_eval(assignments[0].value.args[0]) == str(bundle)
    assert source.read_bytes() == before
    assert not output.with_suffix(".md").exists()


def test_hard_case_preparation_preserves_simulation_cells(workspace: Path) -> None:
    import ast
    import hashlib

    import nbformat

    source = workspace / "student.ipynb"
    setup = nbformat.v4.new_code_cell(
        "import networkx as nx\nimport dreems\n"
        "def create_lattice(rows=8, columns=12):\n"
        "    return nx.grid_2d_graph(rows, columns)\n"
    )
    scenarios = [
        nbformat.v4.new_code_cell(
            f"data = simulate(seed={index}, num_snps=1000)\nsurface = dreems.infer(data)",
            execution_count=index + 1,
            outputs=[nbformat.v4.new_output("stream", text="old output\n")],
        )
        for index in range(12)
    ]
    duplicate = nbformat.v4.new_code_cell(scenarios[0].source)
    notebook = nbformat.v4.new_notebook(cells=[setup, duplicate, *scenarios])
    nbformat.write(notebook, source)
    before = source.read_bytes()
    config = {
        "source_notebook": str(source),
        "checkpoint": str(workspace / "checkpoint.zip"),
        "outdir": str(workspace / "published"),
        "workdir": str(workspace / "work & 'quoted'"),
    }
    output = Path(config["workdir"]) / "feems_scenarios.prepared.ipynb"
    stub_rules(workspace, "hard_case_notebook.Snakefile", ["prepare_notebook"], [output], [])
    run_workflow(workspace, config, execute=True)
    prepared = nbformat.read(output, as_version=4)
    assert len(prepared.cells) == 14
    assert hashlib.sha256(before).hexdigest() in prepared.cells[0].source
    assert hashlib.sha256((workspace / "checkpoint.zip").read_bytes()).hexdigest() in (
        prepared.cells[0].source
    )
    assert "specified DREEMS checkpoint" in prepared.cells[0].source
    for old, new in zip(scenarios, prepared.cells[2:], strict=True):
        assert new.source == old.source.replace(
            "surface = dreems.infer(data)", "surface = dreems.infer(data, checkpoint=PREDICTOR)"
        )
        assert new.outputs == []
        assert new.execution_count is None
    assignments = [
        node
        for node in ast.walk(ast.parse(prepared.cells[1].source))
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "CHECKPOINT" for t in node.targets)
    ]
    assert len(assignments) == 1
    assert ast.literal_eval(assignments[0].value.args[0]) == config["checkpoint"]
    assert source.read_bytes() == before
    assert "Nothing to be done" in run_workflow(workspace, config, dry_run=True)


def test_notebook_derivation_does_not_invalidate_base_simulation(workspace: Path) -> None:
    config = yaml.safe_load((REPO / "workflow/config.feems_scenarios_exact_h2h.yaml").read_text())
    out = workspace / "notebook"
    config.update(
        outdir=str(out),
        publish_dir=str(out / "report"),
        checkpoint=str(workspace / "checkpoint.zip"),
        notebook_scenario_config=str(workspace / "config.yaml"),
        notebook_scenario_base_pattern=str(out / "base_{truth}.npz"),
        notebook_scenario_bases_precomputed=False,
        notebook_scenario_cases_precomputed=False,
        input_pattern=str(out / "example_{example}.pkl"),
        metadata_pattern=str(out / "example_{example}.json"),
    )
    outputs = [out / "base_barrier.npz", out / "example_0.pkl", out / "example_0.json"]
    stub_rules(
        workspace,
        "h2h.Snakefile",
        ["generate_notebook_scenario_base", "derive_notebook_scenario_case"],
        outputs[1:],
        outputs,
    )
    run_workflow(workspace, config)
    assert "Nothing to be done" in run_workflow(workspace, config, dry_run=True)
    for name in ("neural/features.py", "data/genotypes.py"):
        path = workspace / "src/dreems" / name
        path.write_text(path.read_text() + "\n# derivation changed\n")
        planned = run_workflow(workspace, config, dry_run=True)
        assert "rule derive_notebook_scenario_case:" in planned
        assert "rule generate_notebook_scenario_base:" not in planned
        run_workflow(workspace, config)
