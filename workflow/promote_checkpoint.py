"""Package a training checkpoint as the reproducible DREEMS release artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

METADATA_NAME = "metadata.json"
PARAMS_NAME = "params.eqx"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""

    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def read_history(path: Path) -> list[dict[str, Any]]:
    """Read the JSON-lines validation history."""

    return [json.loads(line) for line in path.read_text().splitlines() if line]


def zip_member(name: str, data: bytes) -> tuple[zipfile.ZipInfo, bytes]:
    """Build a ZIP member with stable metadata."""

    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    return info, data


def promote(
    *,
    source: Path,
    summary_path: Path,
    history_path: Path,
    training_config: Path,
    output: Path,
    source_run: str,
    raw_corpus_run: str,
    raw_corpus_config_sha256: str,
) -> None:
    """Promote best parameters while recording the complete training run."""

    summary = read_json(summary_path)
    training_run = summary["training_run"]
    history = read_history(history_path)
    if training_run["history"] != history:
        raise ValueError("training summary and JSON-lines history disagree")

    with zipfile.ZipFile(source) as archive:
        metadata = json.loads(archive.read(METADATA_NAME))
        params = archive.read(PARAMS_NAME)

    source_run_metadata = metadata["training_run"]
    if source_run_metadata["best_step"] != training_run["best_step"]:
        raise ValueError("source checkpoint is not the best step in the summary")
    if source_run_metadata["best_validation_loss"] != training_run["best_validation_loss"]:
        raise ValueError("source checkpoint and summary validation losses disagree")

    params_sha256 = hashlib.sha256(params).hexdigest()
    metadata["train_losses"] = summary["losses"]
    metadata["training_run"] = training_run
    metadata["provenance"] = {
        "raw_corpus_config_sha256": raw_corpus_config_sha256,
        "raw_corpus_run": raw_corpus_run,
        "source_checkpoint_sha256": sha256(source),
        "source_params_sha256": params_sha256,
        "source_run": source_run,
        "training_config_sha256": sha256(training_config),
        "training_history_sha256": sha256(history_path),
        "training_summary_sha256": sha256(summary_path),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_bytes = json.dumps(metadata, sort_keys=True).encode()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(*zip_member(METADATA_NAME, metadata_bytes))
        archive.writestr(*zip_member(PARAMS_NAME, params))


def main() -> None:
    """Parse command-line arguments and write the release archive."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--raw-corpus-run", required=True)
    parser.add_argument("--raw-corpus-config-sha256", required=True)
    args = parser.parse_args()
    promote(
        source=args.source,
        summary_path=args.summary,
        history_path=args.history,
        training_config=args.training_config,
        output=args.output,
        source_run=args.source_run,
        raw_corpus_run=args.raw_corpus_run,
        raw_corpus_config_sha256=args.raw_corpus_config_sha256,
    )


if __name__ == "__main__":
    main()
