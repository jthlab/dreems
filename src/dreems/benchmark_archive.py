"""Pack and restore FEEMS benchmark results without fitting either method.

Input pickles are trusted pipeline files. Archive members are never extracted
by pathname: only the requested regular JSON/NPZ files are copied and checked.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import tarfile
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from beartype import beartype

from dreems import feems, metrics
from dreems.benchmark_data import _example_arrays, _load_example
from dreems.workflow_io import _atomic_output, sha256, write_json


@beartype
def verify_checksum(archive: Path, checksum: Path) -> str:
    """Check the supplied archive, not the historical filename in its sidecar."""
    lines = checksum.read_text().strip().splitlines()
    if len(lines) != 1 or not re.fullmatch(r"[0-9a-fA-F]{64}(?:[ \t]+[^\r\n]+)?", lines[0]):
        raise ValueError("expected one SHA-256 digest in the checksum file")
    expected = lines[0].split(maxsplit=1)[0].lower()
    actual = sha256(archive)
    if actual != expected:
        raise ValueError(f"archive checksum mismatch: {archive}")
    return actual


@beartype
def _case_ids(inputs: Mapping[int, Path]) -> list[int]:
    """Require unique, nonnegative case IDs before constructing any paths."""
    if not inputs or any(
        not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in inputs
    ):
        raise ValueError("case IDs must be nonempty, nonnegative integers")
    return sorted(inputs)


@beartype
def _members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    """Reject links, traversal, duplicates, and unexpected files throughout a tar."""
    members = {}
    for member in archive:
        if member.isdir() and member.name.rstrip("/") == "feems":
            continue
        if not member.isfile() or not re.fullmatch(
            r"feems/example_(0|[1-9][0-9]*)\.(json|npz)", member.name
        ):
            raise ValueError(f"unsafe or unexpected archive member: {member.name!r}")
        if member.name in members:
            raise ValueError(f"duplicate archive member: {member.name!r}")
        members[member.name] = member
    return members


@beartype
def validate_case(
    directory: Path, index: int, input_path: Path, *, allow_legacy: bool = False
) -> dict[str, Any]:
    """Check fit identity, edge order, truth, sampling, and the declared LOO protocol.

    A legacy fit has no recorded genotype-file hash. Geometry and sample counts
    can still be checked, but they do not establish genotype identity.
    """
    row = json.loads((directory / f"example_{index}.json").read_text())
    if (
        not isinstance(row, dict)
        or type(row.get("index")) is not int
        or row["index"] != index
        or row.get("method") != "feems"
    ):
        raise ValueError(f"invalid FEEMS report for example {index}")
    digest = sha256(input_path)
    recorded = row.get("input_sha256")
    if recorded is None:
        if not allow_legacy:
            raise ValueError(
                f"example {index} has no recorded input hash; legacy reuse requires explicit consent"
            )
    elif recorded != digest:
        raise ValueError(f"input hash mismatch for example {index}")
    if (
        row.get("protocol") != feems.FEEMS_RECOMMENDED_LOO_PROTOCOL
        or row.get("cv_mode") != "leave_one_out"
    ):
        raise ValueError(f"example {index} did not use documented FEEMS LOO")
    positions, edges, samples, genotypes, truth = _example_arrays(_load_example(input_path))
    expected_counts = {
        "num_nodes": positions.shape[0],
        "num_edges": edges.shape[0],
        "num_samples": samples.size,
        "num_sampled_nodes": np.unique(samples).size,
        "cv_folds": np.unique(samples).size,
        "n_snps_after_filter": np.count_nonzero(feems.polymorphic_snp_mask(genotypes)),
    }
    for key, expected in expected_counts.items():
        if row.get(key) != expected:
            raise ValueError(f"example {index} disagrees on {key}")
    with np.load(directory / f"example_{index}.npz", allow_pickle=False) as data:
        for key, expected in {
            "positions": positions,
            "edges": edges,
            "truth": metrics.centered(truth),
            "sample_count": np.bincount(samples, minlength=positions.shape[0]),
        }.items():
            if key not in data or not np.array_equal(data[key], expected):
                raise ValueError(f"example {index} disagrees on {key}")
        if (
            "feems" not in data
            or data["feems"].shape != truth.shape
            or not np.all(np.isfinite(data["feems"]))
        ):
            raise ValueError(f"invalid FEEMS surface for example {index}")
    return {"index": index, "current_input_sha256": digest, "recorded_input_sha256": recorded}


@beartype
def restore_results(
    archive: Path,
    checksum: Path,
    results_dir: Path,
    inputs: Mapping[int, Path],
    verification: Path,
    *,
    allow_legacy: bool = False,
) -> None:
    """Validate every requested case before replacing individual output files.

    Unrequested files in results_dir are left alone. Snakemake handles recovery
    if a filesystem error interrupts publication of multiple validated files.
    """
    indices = _case_ids(inputs)
    digest = verify_checksum(archive, checksum)
    results_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".feems-restore-", dir=results_dir.parent) as temporary:
        staging = Path(temporary)
        with tarfile.open(archive, "r:gz") as source:
            members = _members(source)
            for index in indices:
                for suffix in ("json", "npz"):
                    name = f"example_{index}.{suffix}"
                    member = members.get(f"feems/{name}")
                    if member is None or member.size == 0:
                        raise ValueError(f"archive lacks a nonempty feems/{name}")
                    with (
                        source.extractfile(member) as incoming,
                        (staging / name).open("wb") as outgoing,
                    ):
                        shutil.copyfileobj(incoming, outgoing)
        cases = [
            validate_case(staging, index, inputs[index], allow_legacy=allow_legacy)
            for index in indices
        ]
        for index in indices:
            for suffix in ("json", "npz"):
                name = f"example_{index}.{suffix}"
                with (
                    (staging / name).open("rb") as incoming,
                    _atomic_output(results_dir / name) as outgoing,
                ):
                    shutil.copyfileobj(incoming, outgoing)
    write_json(
        verification,
        {
            "archive_sha256": digest,
            "examples": len(cases),
            "input_identity": "sha256"
            if all(case["recorded_input_sha256"] is not None for case in cases)
            else "legacy_unverified",
            "cases": cases,
        },
    )


@beartype
def pack_results(
    results_dir: Path,
    inputs: Mapping[int, Path],
    archive: Path,
    checksum: Path,
    *,
    allow_legacy: bool = False,
) -> None:
    """Write a reproducible tar/gzip and a checksum naming the published file."""
    indices = _case_ids(inputs)
    for index in indices:
        validate_case(results_dir, index, inputs[index], allow_legacy=allow_legacy)
    names = sorted(f"example_{index}.{suffix}" for index in indices for suffix in ("json", "npz"))
    with (
        _atomic_output(archive) as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as target,
    ):
        for name in names:
            path = results_dir / name
            info = tarfile.TarInfo(f"feems/{name}")
            info.size = path.stat().st_size
            info.mode = 0o644
            with path.open("rb") as incoming:
                target.addfile(info, incoming)
    with _atomic_output(checksum) as handle:
        handle.write(f"{sha256(archive)}  {archive.name}\n".encode())


@beartype
def main() -> None:
    """Run archive I/O as a Snakemake worker, independently of fitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pack", "restore"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checksum", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--input-pattern", required=True)
    parser.add_argument("--examples", type=int, nargs="+", required=True)
    parser.add_argument("--allow-legacy", action="store_true")
    parser.add_argument("--verification", type=Path)
    args = parser.parse_args()
    if len(set(args.examples)) != len(args.examples):
        parser.error("example IDs must be unique")
    inputs = {index: Path(args.input_pattern.format(example=index)) for index in args.examples}
    if args.command == "pack":
        pack_results(
            args.results_dir, inputs, args.archive, args.checksum, allow_legacy=args.allow_legacy
        )
    else:
        if args.verification is None:
            parser.error("restore requires --verification")
        restore_results(
            args.archive,
            args.checksum,
            args.results_dir,
            inputs,
            args.verification,
            allow_legacy=args.allow_legacy,
        )


if __name__ == "__main__":
    main()
