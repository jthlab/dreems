"""Archive reuse must preserve fitted bytes and reject mismatched inputs."""

from __future__ import annotations

import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from dreems import feems
from dreems.evaluation import benchmark_data as archives
from dreems.utils import workflow_io


@pytest.fixture
def cases(tmp_path: Path) -> tuple[Path, dict[int, Path]]:
    results = tmp_path / "results with spaces"
    results.mkdir()
    inputs = {}
    for index in (0, 1):
        positions = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        edges = np.asarray([[0, 1], [1, 2]], dtype=np.int32)
        truth = np.asarray([-0.5, 0.5])
        payload = {
            "graph": {"edges": edges},
            "active_positions": positions,
            "sample_nodes": np.arange(3, dtype=np.int32),
            "edge_log_rates": truth,
            "genotypes": np.asarray([[0, 1], [1, 2], [2, 0]], dtype=np.float64),
        }
        inputs[index] = tmp_path / f"input_{index}.pkl"
        workflow_io.write_pickle(inputs[index], payload)
        workflow_io.write_json(
            results / f"example_{index}.json",
            {
                "index": index,
                "method": "feems",
                "input_sha256": workflow_io.sha256(inputs[index]),
                "protocol": feems.FEEMS_RECOMMENDED_LOO_PROTOCOL,
                "cv_mode": "leave_one_out",
                "cv_folds": 3,
                "num_nodes": 3,
                "num_edges": 2,
                "num_samples": 3,
                "num_sampled_nodes": 3,
                "n_snps_after_filter": 2,
            },
        )
        workflow_io.write_npz(
            results / f"example_{index}.npz",
            positions=positions,
            edges=edges,
            truth=truth,
            sample_count=np.ones(3, dtype=np.int32),
            feems=truth * 0.8,
        )
    return results, inputs


def pack(tmp_path: Path, cases: tuple[Path, dict[int, Path]]) -> tuple[Path, Path]:
    archive, checksum = tmp_path / "original.tar.gz", tmp_path / "checksum elsewhere.txt"
    archives.pack_results(*cases, archive, checksum)
    return archive, checksum


def test_round_trip_rename_subset_and_existing_results(tmp_path: Path, cases) -> None:
    archive, checksum = pack(tmp_path, cases)
    renamed = tmp_path / "renamed archive.tar.gz"
    archive.rename(renamed)
    assert archives.verify_checksum(renamed, checksum) == workflow_io.sha256(renamed)
    results, inputs = cases
    output = tmp_path / "restored"
    output.mkdir()
    unrelated = output / "example_900.json"
    unrelated.write_text("keep this fit")
    report = tmp_path / "verification.json"
    archives.restore_results(renamed, checksum, output, {1: inputs[1]}, report)
    for suffix in ("json", "npz"):
        assert (output / f"example_1.{suffix}").read_bytes() == (
            results / f"example_1.{suffix}"
        ).read_bytes()
    assert not (output / "example_0.json").exists()
    assert unrelated.read_text() == "keep this fit"
    summary = json.loads(report.read_text())
    assert summary["examples"] == 1
    assert summary["input_identity"] == "sha256"
    assert summary["cases"][0]["index"] == 1


def test_packing_is_reproducible_across_paths_order_and_file_modes(tmp_path: Path, cases) -> None:
    first, _ = pack(tmp_path, cases)
    results, inputs = cases
    for path in results.iterdir():
        path.chmod(0o600)
        path.touch()
    second, checksum = tmp_path / "second.tar.gz", tmp_path / "second.sha256"
    archives.pack_results(results, dict(reversed(list(inputs.items()))), second, checksum)
    assert first.read_bytes() == second.read_bytes()
    assert checksum.read_text().endswith("  second.tar.gz\n")


def test_late_input_mismatch_does_not_publish_any_case(tmp_path: Path, cases) -> None:
    archive, checksum = pack(tmp_path, cases)
    _, inputs = cases
    payload = workflow_io.read_pickle(inputs[1])
    payload["genotypes"] = payload["genotypes"][::-1].copy()
    workflow_io.write_pickle(inputs[1], payload)
    output = tmp_path / "restored"
    output.mkdir()
    existing = output / "example_0.json"
    existing.write_text("existing result")
    with pytest.raises(ValueError, match="input hash mismatch"):
        archives.restore_results(archive, checksum, output, inputs, tmp_path / "report.json")
    assert existing.read_text() == "existing result"
    assert not (output / "example_0.npz").exists()
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize(
    "kind", ["traversal", "absolute", "symlink", "hardlink", "duplicate", "extra"]
)
def test_bad_members_never_reach_the_filesystem(tmp_path: Path, cases, kind: str) -> None:
    results, inputs = cases
    archive, checksum = tmp_path / "bad.tar.gz", tmp_path / "bad.sha256"
    with tarfile.open(archive, "w:gz") as target:
        for path in sorted(results.iterdir()):
            target.add(path, arcname=f"feems/{path.name}")
        name = {
            "traversal": "../escaped",
            "absolute": str(tmp_path / "escaped"),
            "extra": "unexpected.txt",
        }.get(kind, "feems/example_0.json")
        entry = tarfile.TarInfo(name)
        if kind in {"symlink", "hardlink"}:
            entry.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
            entry.linkname = str(tmp_path / "escaped")
        else:
            entry.size = 1
        target.addfile(entry, io.BytesIO(b"x"))
    checksum.write_text(workflow_io.sha256(archive))
    with pytest.raises(ValueError, match="unsafe|unexpected|duplicate"):
        archives.restore_results(
            archive, checksum, tmp_path / "restored", inputs, tmp_path / "report.json"
        )
    assert not (tmp_path / "escaped").exists()
    assert not (tmp_path / "restored").exists()


def test_checksum_and_missing_cases_fail_closed(tmp_path: Path, cases) -> None:
    archive, checksum = pack(tmp_path, cases)
    with pytest.raises(ValueError, match="lacks"):
        archives.restore_results(
            archive, checksum, tmp_path / "restored", {99: cases[1][0]}, tmp_path / "report.json"
        )
    checksum.write_text("0" * 64)
    with pytest.raises(ValueError, match="checksum mismatch"):
        archives.restore_results(
            archive, checksum, tmp_path / "restored", cases[1], tmp_path / "report.json"
        )
    assert not (tmp_path / "restored").exists()


def test_legacy_reuse_is_explicit_and_does_not_invent_a_hash(tmp_path: Path, cases) -> None:
    results, inputs = cases
    for index in inputs:
        path = results / f"example_{index}.json"
        row = json.loads(path.read_text())
        del row["input_sha256"]
        workflow_io.write_json(path, row)
    archive, checksum = tmp_path / "legacy.tar.gz", tmp_path / "legacy.sha256"
    with pytest.raises(ValueError, match="no recorded input hash"):
        archives.pack_results(results, inputs, archive, checksum)
    archives.pack_results(results, inputs, archive, checksum, allow_legacy=True)
    report = tmp_path / "verification.json"
    with pytest.raises(ValueError, match="no recorded input hash"):
        archives.restore_results(archive, checksum, tmp_path / "restored", inputs, report)
    archives.restore_results(
        archive, checksum, tmp_path / "restored", inputs, report, allow_legacy=True
    )
    summary = json.loads(report.read_text())
    assert summary["input_identity"] == "legacy_unverified"
    assert all(case["recorded_input_sha256"] is None for case in summary["cases"])
    assert (tmp_path / "restored/example_0.json").read_bytes() == (
        results / "example_0.json"
    ).read_bytes()


@pytest.mark.parametrize("field", ["positions", "edges", "truth", "sample_count", "feems"])
def test_result_arrays_are_checked_against_the_inputs(tmp_path: Path, cases, field: str) -> None:
    results, inputs = cases
    path = results / "example_0.npz"
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    arrays[field] = arrays[field].astype(float)
    arrays[field].flat[0] = np.nan if field == "feems" else arrays[field].flat[0] + 1
    workflow_io.write_npz(path, **arrays)
    with pytest.raises(ValueError, match=field.upper() if field == "feems" else field):
        archives.pack_results(results, inputs, tmp_path / "bad.tar.gz", tmp_path / "bad.sha256")


@pytest.mark.parametrize(
    "field,value",
    [
        ("index", 9),
        ("method", "dreems"),
        ("cv_mode", "holdout"),
        ("cv_folds", 2),
        ("n_snps_after_filter", 100),
    ],
)
def test_report_metadata_is_checked(cases, field, value) -> None:
    results, inputs = cases
    path = results / "example_0.json"
    row = json.loads(path.read_text())
    row[field] = value
    workflow_io.write_json(path, row)
    with pytest.raises(ValueError):
        archives.validate_case(results, 0, inputs[0])


@pytest.mark.parametrize("in_place", [False, True])
def test_snakemake_restores_and_reuses_without_fitting(
    tmp_path: Path, cases, in_place: bool
) -> None:
    """Execute the actual archive rules, not a mocked copy of their shell commands."""
    repo = Path(__file__).resolve().parents[1]
    archive, checksum = pack(tmp_path, cases)
    publish = tmp_path / "published report"
    publish.mkdir()
    if in_place:
        archive = archive.rename(publish / "feems_results.tar.gz")
        checksum = checksum.rename(publish / "feems_results.tar.gz.sha256")
    work = tmp_path / "work output"
    config = {
        "outdir": str(work),
        "publish_dir": str(publish),
        "checkpoint": "unused-checkpoint.zip",
        "input_pattern": str(tmp_path / "input_{example}.pkl"),
        "metadata_pattern": "unused_{example}.json",
        "example_count": 2,
        "feems_archive": str(archive),
        "feems_archive_checksum": str(checksum),
    }
    configuration = tmp_path / "config.yaml"
    configuration.write_text(yaml.safe_dump(config))
    selected = "extract_feems_archive" if in_place else "extract_feems_archive, archive_feems"
    snakefile = tmp_path / "audit.Snakefile"
    snakefile.write_text(
        f"module production:\n    snakefile: {str(repo / 'workflow/h2h.Snakefile')!r}\n    config: config\n"
        f"use rule {selected} from production as *\n"
        f"rule all:\n    input: {str(publish / 'feems_archive_verification.json')!r}, {str(publish / 'feems_results.tar.gz')!r}\n"
    )
    command = [
        "uv",
        "--project",
        str(repo),
        "run",
        "--frozen",
        "--no-sync",
        "snakemake",
        "all",
        "-s",
        str(snakefile),
        "--configfile",
        str(configuration),
        "--executor",
        "local",
        "--cores",
        "1",
    ]
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    first = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )
    assert first.returncode == 0, first.stdout + first.stderr
    assert "rule feems_case:" not in first.stdout + first.stderr
    for index in (0, 1):
        for suffix in ("json", "npz"):
            name = f"example_{index}.{suffix}"
            assert (work / "feems" / name).read_bytes() == (cases[0] / name).read_bytes()
    archive_path = publish / "feems_results.tar.gz"
    old_time = archive_path.stat().st_mtime_ns
    second = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )
    assert second.returncode == 0, second.stdout + second.stderr
    assert "Nothing to be done" in second.stdout + second.stderr
    assert archive_path.stat().st_mtime_ns == old_time
