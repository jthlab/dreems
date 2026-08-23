from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import numpy as np

from dreems import calibration, features
from dreems import model as dreems_model

CHECKPOINT_METADATA = "metadata.json"
CHECKPOINT_PARAMS = "params.eqx"
DEFAULT_CHECKPOINT_RESOURCE = "data/dreems_diffusion_bands.dreems.zip"


@dataclass(frozen=True)
class DREEMSCheckpoint:
    """Loaded DREEMS checkpoint with normalization and model metadata."""

    path: Path
    params: dreems_model.DREEMS
    normalization: dict[str, Any]
    payload: dict[str, Any]

    @property
    def model(self) -> dict[str, Any]:
        model = self.payload.get("model")
        if not isinstance(model, dict):
            raise ValueError(f"checkpoint {self.path} has no model metadata")
        return model

    @property
    def edge_sd_calibration(self) -> calibration.SDCalibration:
        payload = self.payload.get("edge_sd_calibration")
        if payload is None:
            return calibration.SDCalibration()
        if not isinstance(payload, Mapping):
            raise ValueError(f"checkpoint {self.path} has invalid edge SD calibration")
        return calibration.SDCalibration.from_dict(payload)


@dataclass(frozen=True)
class DREEMSPrediction:
    edge_mean: list[np.ndarray]
    edge_sd: list[np.ndarray]
    raw_edge_sd: list[np.ndarray]
    node_log_q_mean: list[np.ndarray]
    node_log_q_sd: list[np.ndarray]
    checkpoint: DREEMSCheckpoint


def canonical_feature_metadata(model_features: dict[str, Any]) -> dict[str, object]:
    return {
        "kind": str(model_features["kind"]),
        "covariance_components": int(model_features["covariance_components"]),
    }


def feature_spatial_encoding(model_features: dict[str, object]) -> features.SpatialEncoding:
    kind = str(model_features["kind"])
    if kind == "diffusion_bands_v1":
        return "embedded"
    if kind == "topology_diffusion_bands_v1":
        return "topology"
    if kind == "topology_diffusion_bands_v2":
        return "topology_node_covariance"
    raise ValueError(f"unsupported checkpoint feature kind: {kind!r}")


def load_checkpoint(path: Path | str | None = None) -> DREEMSCheckpoint:
    """Load a production DREEMS checkpoint."""

    if path is None:
        checkpoint_resource = resources.files("dreems").joinpath(DEFAULT_CHECKPOINT_RESOURCE)
        with resources.as_file(checkpoint_resource) as checkpoint_path:
            return load_checkpoint(checkpoint_path)

    checkpoint_path = Path(path)
    with zipfile.ZipFile(checkpoint_path, "r") as archive:
        payload = json.loads(archive.read(CHECKPOINT_METADATA).decode("utf-8"))
        params_data = io.BytesIO(archive.read(CHECKPOINT_PARAMS))
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {checkpoint_path} does not contain a payload dict")
    stats = payload.get("normalization")
    model = payload.get("model")
    if not isinstance(model, dict) or not isinstance(stats, dict):
        raise ValueError(f"checkpoint {checkpoint_path} is missing params or normalization")
    dreems_model.validate_production_model_metadata(model)
    params = eqx.tree_deserialise_leaves(params_data, model_from_metadata(model))
    return DREEMSCheckpoint(
        path=checkpoint_path,
        params=params,
        normalization=stats,
        payload=payload,
    )


def model_from_metadata(model: dict[str, Any]) -> dreems_model.DREEMS:
    """Construct the model structure used by Equinox deserialization."""

    architecture = model.get("architecture", model)
    if not isinstance(architecture, dict):
        raise ValueError("checkpoint model metadata has invalid architecture")
    context_layers = architecture.get("context_layers")
    if context_layers is None:
        raise ValueError("checkpoint model metadata is missing context_layers")
    return dreems_model.DREEMS(
        jax.random.PRNGKey(0),
        node_dim=int(model["node_dim"]),
        edge_dim=int(model["edge_dim"]),
        hidden_channels=int(architecture["hidden_channels"]),
        layers=int(architecture["layers"]),
        context_layers=int(context_layers),
    )


def predict_examples(
    checkpoint: DREEMSCheckpoint | Path | str,
    examples: list[dict[str, object]],
) -> DREEMSPrediction:
    """Predict edge log-rate means and posterior SDs for prepared examples.

    This is the production inference path: diffusion-band features,
    optional linearized-resistance edge conditioning, and the DREEMS GNN
    forward pass.
    """

    loaded = (
        load_checkpoint(checkpoint) if not isinstance(checkpoint, DREEMSCheckpoint) else checkpoint
    )
    model = loaded.model
    dreems_model.validate_production_model_metadata(model)
    feature_metadata = model.get("features", {})
    if not isinstance(feature_metadata, dict):
        raise ValueError("checkpoint model metadata is missing features")
    feature_metadata = canonical_feature_metadata(feature_metadata)
    prepared = features.prepare_examples(
        examples,
        covariance_components=int(feature_metadata["covariance_components"]),
        spatial_encoding=feature_spatial_encoding(feature_metadata),
    )
    prepared = features.append_edge_conditioner_features_for_checkpoint(
        prepared,
        examples,
        model,
        loaded.normalization,
    )
    prepared = features.apply_normalization(prepared, loaded.normalization)
    arrays = features.pad_examples(prepared)
    edge_mean, edge_logvar, q_mean, q_logvar = dreems_model.predict(
        loaded.params,
        arrays,
        y_scale=float(loaded.normalization["y_scale"]),
        q_y_scale=float(loaded.normalization["q_y_scale"]),
        model=model,
    )
    means: list[np.ndarray] = []
    sds: list[np.ndarray] = []
    raw_sds: list[np.ndarray] = []
    q_means: list[np.ndarray] = []
    q_sds: list[np.ndarray] = []
    edge_sd_calibration = loaded.edge_sd_calibration
    for index in range(len(examples)):
        edge_mask = arrays["edge_mask"][index] > 0.0
        node_mask = arrays["node_mask"][index] > 0.0
        means.append(np.asarray(edge_mean[index][edge_mask], dtype=np.float64))
        logvar = np.asarray(edge_logvar[index][edge_mask], dtype=np.float64)
        raw_sd = np.sqrt(np.exp(np.clip(logvar, -30.0, 30.0)))
        raw_sds.append(raw_sd)
        sds.append(
            edge_sd_calibration.apply(
                raw_sd,
                sampled_node_fraction=_sampled_node_fraction(examples[index]),
            )
        )
        q_means.append(np.asarray(q_mean[index][node_mask], dtype=np.float64))
        q_logvar_local = np.asarray(q_logvar[index][node_mask], dtype=np.float64)
        q_sds.append(np.sqrt(np.exp(np.clip(q_logvar_local, -30.0, 30.0))))
    return DREEMSPrediction(
        edge_mean=means,
        edge_sd=sds,
        raw_edge_sd=raw_sds,
        node_log_q_mean=q_means,
        node_log_q_sd=q_sds,
        checkpoint=loaded,
    )


def _sampled_node_fraction(example: Mapping[str, object]) -> float:
    if "active_positions" in example:
        positions = np.asarray(example["active_positions"], dtype=np.float64)
    elif "positions" in example:
        positions = np.asarray(example["positions"], dtype=np.float64)
    else:
        raise ValueError("example is missing graph node positions")
    if positions.ndim != 2 or positions.shape[0] == 0:
        raise ValueError("graph node positions must contain at least one node")

    if "node_sample_count" in example:
        counts = np.asarray(example["node_sample_count"], dtype=np.float64)
        if counts.shape != (positions.shape[0],):
            raise ValueError("node_sample_count does not match graph nodes")
        sampled_nodes = int(np.count_nonzero(counts > 0.0))
    elif "sample_nodes" in example:
        nodes = np.asarray(example["sample_nodes"], dtype=np.int64)
        sampled_nodes = int(np.unique(nodes).size)
    elif "sample_coordinates" in example:
        nodes = features.nearest_node_assignments(
            np.asarray(example["sample_coordinates"], dtype=np.float64),
            positions,
        )
        sampled_nodes = int(np.unique(nodes).size)
    else:
        raise ValueError("example is missing sample locations")
    return float(sampled_nodes / positions.shape[0])


def _read_checkpoint_archive(path: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    with zipfile.ZipFile(path, "r") as archive:
        members = {item.filename: archive.read(item.filename) for item in archive.infolist()}
    if CHECKPOINT_METADATA not in members or CHECKPOINT_PARAMS not in members:
        raise ValueError(f"checkpoint {path} is missing metadata or parameters")
    payload = json.loads(members[CHECKPOINT_METADATA].decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {path} does not contain a payload dict")
    return members, payload


def _write_checkpoint_archive(
    path: Path,
    members: Mapping[str, bytes],
    payload: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    updated = dict(members)
    updated[CHECKPOINT_METADATA] = json.dumps(payload, sort_keys=True).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, contents in updated.items():
                archive.writestr(name, contents)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def migrate_node_covariance_checkpoint(
    source: Path | str,
    destination: Path | str,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    """Relabel a topology checkpoint trained from node covariance.

    The old and canonical feature paths are identical when the training input
    already contains node covariance. This migration changes only metadata so
    genotype, sample-covariance, and node-covariance inference all use that
    same representation. Equinox parameter bytes remain unchanged.
    """

    source_path = Path(source)
    destination_path = Path(destination)
    members, payload = _read_checkpoint_archive(source_path)
    model = payload.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"checkpoint {source_path} has no model metadata")
    model_features = model.get("features")
    if not isinstance(model_features, dict):
        raise ValueError(f"checkpoint {source_path} has no feature metadata")
    source_kind = model_features.get("kind")
    if source_kind != "topology_diffusion_bands_v1":
        raise ValueError(
            f"node-covariance migration requires topology_diffusion_bands_v1, found {source_kind!r}"
        )
    model_features["kind"] = "topology_diffusion_bands_v2"
    dreems_model.validate_production_model_metadata(model)
    migration: dict[str, Any] = {
        "parameter_bytes_changed": False,
        "source_kind": source_kind,
        "target_kind": "topology_diffusion_bands_v2",
        "training_input": "node_covariance",
    }
    if provenance is not None:
        migration["provenance"] = dict(provenance)
    payload["feature_migration"] = migration
    _write_checkpoint_archive(destination_path, members, payload)


def attach_edge_sd_calibration(
    source: Path | str,
    destination: Path | str,
    fitted: calibration.SDCalibration,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    """Copy a checkpoint and attach fitted edge-SD calibration metadata.

    Equinox parameter bytes are copied unchanged. The destination is replaced
    atomically after the complete archive has been written.
    """

    source_path = Path(source)
    destination_path = Path(destination)
    members, payload = _read_checkpoint_archive(source_path)
    calibration_payload = fitted.to_dict()
    if provenance is not None:
        calibration_payload["provenance"] = dict(provenance)
    payload["edge_sd_calibration"] = calibration_payload
    _write_checkpoint_archive(destination_path, members, payload)
