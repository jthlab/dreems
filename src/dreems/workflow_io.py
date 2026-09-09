"""Read worker configs and manifests, and replace complete outputs atomically.

Pickle is for trusted pipeline artifacts, never for data from an untrusted source.
Atomic replacement protects an existing file if serialization or writing fails;
Snakemake still owns retries and the dependencies between multiple outputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import stat
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from io import TextIOWrapper
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import IO, Any

import numpy as np
import yaml
from beartype import beartype


@contextmanager
@beartype
def _atomic_output(path: Path) -> Iterator[IO[bytes]]:
    """Close the new file before replacing its destination on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix=f".{path.name}.") as directory:
        temporary = Path(directory) / path.name
        with temporary.open("wb") as handle:
            yield handle
        if path.exists():
            temporary.chmod(stat.S_IMODE(path.stat().st_mode))
        temporary.replace(path)


@beartype
def parse_nonnegative_integer(value: Any, *, name: str) -> int:
    """Parse an integer CSV field without rounding or losing large IDs."""
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"{name} must be a nonnegative integer") from error
    if not number.is_finite() or number < 0 or number != number.to_integral_value():
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(number)


@beartype
def read_pickle(path: Path) -> Any:
    """Load a trusted raw or prepared pack, retaining its stored NumPy dtypes."""
    with path.open("rb") as handle:
        return pickle.load(handle)


@beartype
def write_pickle(path: Path, payload: Any) -> None:
    """Write a complete pack before replacing any existing output."""
    with _atomic_output(path) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


@beartype
def write_npz(path: Path, **arrays: np.ndarray) -> None:
    """Replace a compressed NumPy archive only after every array is written."""
    with _atomic_output(path) as handle:
        np.savez_compressed(handle, **arrays)


@beartype
def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Replace a complete CSV, preserving first-seen column order and values.

    Callers choose their missing-value/NumPy conversion policy before writing.
    """
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with _atomic_output(path) as raw, TextIOWrapper(raw, encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@beartype
def sha256(path: Path) -> str:
    """Hash an artifact without loading the entire file into memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@beartype
def json_ready(value: Any) -> Any:
    """Convert nested NumPy values to the JSON representation used by reports."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


@beartype
def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable, sorted report JSON without exposing a partial file."""
    contents = json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n"
    with _atomic_output(path) as handle:
        handle.write(contents.encode("utf-8"))


@beartype
def json_finite(value: Any) -> Any:
    """Use JSON null for missing numeric values, including those inside arrays.

    Browser-facing reports need standard JSON. Scientific reports that retain
    NaN should continue to use ``json_ready`` instead.
    """
    value = json_ready(value)
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_finite(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@beartype
def write_json_finite(path: Path, payload: Any) -> None:
    """Atomically write a browser-facing report with null for missing values."""
    contents = json.dumps(json_finite(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with _atomic_output(path) as handle:
        handle.write(contents.encode("utf-8"))


@beartype
def load_config(path: Path) -> dict[str, Any]:
    """Read YAML or JSON; only an empty document or null means an empty config."""
    text = path.read_text()
    payload = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected a config mapping in {path}")
    if not all(isinstance(key, str) for key in payload):
        raise ValueError(f"expected string config keys in {path}")
    return payload


@beartype
def read_path_manifest(path: Path) -> list[Path]:
    """Read paths in order, resolving relative entries beside the manifest."""
    paths: list[Path] = []
    base = path.parent
    for line in path.read_text().splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        item = Path(value)
        paths.append(item if item.is_absolute() else base / item)
    return paths


@beartype
def required_input_paths(
    inputs: Sequence[Path] | None,
    manifest: Path | None,
    *,
    argument_name: str,
) -> list[Path]:
    """Collect inputs and reject an empty result with the CLI argument name."""
    paths = optional_input_paths(inputs, manifest)
    if not paths:
        raise ValueError(f"provide {argument_name} or an input manifest")
    return paths


@beartype
def optional_input_paths(
    inputs: Sequence[Path] | None,
    manifest: Path | None,
) -> list[Path]:
    """Put manifest entries first, followed by explicit paths, without sorting."""
    paths: list[Path] = []
    if manifest is not None:
        paths.extend(read_path_manifest(manifest))
    if inputs is not None:
        paths.extend(inputs)
    return paths
