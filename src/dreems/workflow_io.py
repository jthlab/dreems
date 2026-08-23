"""Small file and manifest primitives shared by Snakemake worker commands."""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text()
    payload = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    payload = payload or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected a config mapping in {path}")
    return payload


def read_path_manifest(path: Path) -> list[Path]:
    paths: list[Path] = []
    base = path.parent
    for line in path.read_text().splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        item = Path(value)
        paths.append(item if item.is_absolute() else base / item)
    return paths


def required_input_paths(
    inputs: Sequence[Path] | None,
    manifest: Path | None,
    *,
    argument_name: str,
) -> list[Path]:
    paths = optional_input_paths(inputs, manifest)
    if not paths:
        raise ValueError(f"provide {argument_name} or an input manifest")
    return paths


def optional_input_paths(
    inputs: Sequence[Path] | None,
    manifest: Path | None,
) -> list[Path]:
    paths: list[Path] = []
    if manifest is not None:
        paths.extend(read_path_manifest(manifest))
    if inputs is not None:
        paths.extend(inputs)
    return paths
