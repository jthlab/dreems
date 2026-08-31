"""Training loop for the production DREEMS model."""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from dreems import checkpoints, features, losses
from dreems import model as dreems_model


@dataclass(frozen=True)
class DREEMSArchitectureConfig:
    hidden_channels: int = 64
    layers: int = 10
    context_layers: int = 3
    initializer_gate: bool = False
    initializer_gate_initial_logit: float = 6.0
    surface_gate: bool = False
    surface_gate_initial_logit: float = 6.0

    def validate(self) -> None:
        if self.hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if self.layers < 0:
            raise ValueError("layers must be nonnegative")
        if self.context_layers < 0:
            raise ValueError("context_layers must be nonnegative")
        if not np.isfinite(self.initializer_gate_initial_logit):
            raise ValueError("initializer_gate_initial_logit must be finite")
        if not np.isfinite(self.surface_gate_initial_logit):
            raise ValueError("surface_gate_initial_logit must be finite")


@dataclass(frozen=True)
class DREEMSFeatureConfig:
    covariance_components: int = 20
    normalization_std_floor: float = 0.2
    spatial_encoding: str = "embedded"

    def validate(self) -> None:
        if self.covariance_components < 0:
            raise ValueError("covariance_components must be nonnegative")
        if self.normalization_std_floor <= 0.0:
            raise ValueError("normalization_std_floor must be positive")
        if self.spatial_encoding not in features.SPATIAL_ENCODINGS:
            raise ValueError(f"unsupported spatial_encoding: {self.spatial_encoding!r}")


@dataclass(frozen=True)
class LinearizedResistanceConfig:
    smoothness: float = 0.01
    ridge_scale: float = 0.001
    pair_weight_power: float = 0.0
    length_weighted_baseline: bool = True
    output_sd: float | None = 1.0
    batch_size: int = 256

    def validate(self) -> None:
        if self.smoothness < 0.0:
            raise ValueError("edge conditioner smoothness must be nonnegative")
        if self.ridge_scale < 0.0:
            raise ValueError("edge conditioner ridge_scale must be nonnegative")
        if self.batch_size <= 0:
            raise ValueError("edge conditioner batch_size must be positive")


@dataclass(frozen=True)
class DREEMSLossConfig:
    min_logvar: float = -7.0
    max_logvar: float = 5.0
    context_weight: float = 0.5
    q_weight: float = 0.5
    slope_weight: float = 0.25
    slope_target: float = 1.0
    slope_target_floor: float = 0.05
    slope_huber_delta: float = 1.0
    correlation_weight: float = 0.2
    correlation_target_floor: float = 0.05
    orientation_weight: float = 0.35
    orientation_target_floor: float = 0.05
    orientation_margin_fraction: float = 0.5
    initializer_regret_weight: float = 0.0
    initializer_regret_margin: float = 0.0
    counterfactual_consistency_weight: float = 0.0
    surface_gate_weight: float = 0.0
    surface_gate_target_floor: float = 0.05

    def validate(self) -> None:
        if self.min_logvar > self.max_logvar:
            raise ValueError("min_logvar cannot exceed max_logvar")
        for name in (
            "context_weight",
            "q_weight",
            "slope_weight",
            "correlation_weight",
            "orientation_weight",
            "initializer_regret_weight",
            "counterfactual_consistency_weight",
            "surface_gate_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.initializer_regret_margin < 0.0:
            raise ValueError("initializer_regret_margin must be nonnegative")
        if self.surface_gate_target_floor < 0.0:
            raise ValueError("surface_gate_target_floor must be nonnegative")


@dataclass(frozen=True)
class DREEMSOptimizerConfig:
    seed: int = 12001
    steps: int = 90_000
    batch_size: int = 16
    learning_rate: float = 5e-5
    gradient_clip_norm: float = 1.0
    log_every: int | None = None
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    early_stopping_warmup_steps: int = 0
    early_stopping_metric: str = "validation_loss"
    restore_best_validation: bool = True
    trainable_scope: str = "all"

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.log_every is not None and self.log_every <= 0:
            raise ValueError("log_every must be positive when set")
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive when set")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be nonnegative")
        if self.early_stopping_warmup_steps < 0:
            raise ValueError("early_stopping_warmup_steps must be nonnegative")
        if not self.early_stopping_metric:
            raise ValueError("early_stopping_metric must be non-empty")
        if self.trainable_scope not in {"all", "initializer_gate", "surface_gate"}:
            raise ValueError("trainable_scope must be 'all', 'initializer_gate', or 'surface_gate'")


@dataclass(frozen=True)
class DREEMSTrainConfig:
    architecture: DREEMSArchitectureConfig = field(default_factory=DREEMSArchitectureConfig)
    features: DREEMSFeatureConfig = field(default_factory=DREEMSFeatureConfig)
    edge_conditioner: LinearizedResistanceConfig = field(default_factory=LinearizedResistanceConfig)
    loss: DREEMSLossConfig = field(default_factory=DREEMSLossConfig)
    optimizer: DREEMSOptimizerConfig = field(default_factory=DREEMSOptimizerConfig)

    def validate(self) -> None:
        self.architecture.validate()
        self.features.validate()
        self.edge_conditioner.validate()
        self.loss.validate()
        self.optimizer.validate()
        if (
            self.optimizer.trainable_scope == "initializer_gate"
            and not self.architecture.initializer_gate
        ):
            raise ValueError("initializer_gate training scope requires initializer_gate=true")
        if self.optimizer.trainable_scope == "surface_gate" and not self.architecture.surface_gate:
            raise ValueError("surface_gate training scope requires surface_gate=true")


@dataclass(frozen=True)
class TrainingRunResult:
    """Final model parameters plus training/validation bookkeeping."""

    params: dreems_model.DREEMS
    losses: list[float]
    history: list[dict[str, object]]
    final_step: int
    best_step: int | None
    best_validation_loss: float | None
    early_stopping_metric: str
    best_validation_metric: float | None
    best_validation_metrics: dict[str, object] | None
    stopped_early: bool
    stop_reason: str | None
    restored_best_validation: bool

    def summary(self) -> dict[str, object]:
        return {
            "final_step": self.final_step,
            "best_step": self.best_step,
            "best_validation_loss": self.best_validation_loss,
            "early_stopping_metric": self.early_stopping_metric,
            "best_validation_metric": self.best_validation_metric,
            "best_validation_metrics": self.best_validation_metrics,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "restored_best_validation": self.restored_best_validation,
            "history": self.history,
        }


def architecture_metadata(config: DREEMSArchitectureConfig) -> dict[str, object]:
    if config.surface_gate:
        kind = "diffusion_band_surface_gate_v3"
    elif config.initializer_gate:
        kind = "diffusion_band_initializer_gate_v2"
    else:
        kind = "diffusion_band_gnn_v1"
    metadata: dict[str, object] = {
        "kind": kind,
        "hidden_channels": config.hidden_channels,
        "layers": config.layers,
        "context_layers": config.context_layers,
    }
    if config.initializer_gate:
        metadata["initializer_gate"] = True
    if config.surface_gate:
        metadata["surface_gate"] = True
    return metadata


def feature_metadata(config: DREEMSFeatureConfig) -> dict[str, object]:
    feature_kind = {
        "embedded": "diffusion_bands_v1",
        "topology": "topology_diffusion_bands_v1",
        "topology_node_covariance": "topology_diffusion_bands_v2",
    }[config.spatial_encoding]
    return {
        "kind": feature_kind,
        "covariance_components": config.covariance_components,
    }


def canonical_feature_metadata(model_features: Mapping[str, Any]) -> dict[str, object]:
    return {
        "kind": str(model_features["kind"]),
        "covariance_components": int(model_features["covariance_components"]),
    }


def edge_conditioner_metadata(config: LinearizedResistanceConfig) -> dict[str, object]:
    return {
        "kind": "linearized_resistance",
        "method": "all_pair_diagonal_adjoint",
        "smoothness": config.smoothness,
        "ridge_scale": config.ridge_scale,
        "pair_weight_power": config.pair_weight_power,
        "length_weighted_baseline": config.length_weighted_baseline,
        "output_sd": config.output_sd,
        "batch_size": config.batch_size,
        "feature_names": list(features.EDGE_CONDITIONER_FEATURE_NAMES),
    }


def posterior_metadata(config: DREEMSLossConfig) -> dict[str, object]:
    return {
        "kind": "factorized_normal",
        "min_logvar": config.min_logvar,
        "max_logvar": config.max_logvar,
    }


def training_config_metadata(config: DREEMSTrainConfig) -> dict[str, Any]:
    """Serialize configs without changing the released v1 metadata contract."""

    payload = asdict(config)
    if (
        not config.architecture.initializer_gate
        and config.architecture.initializer_gate_initial_logit
        == DREEMSArchitectureConfig.initializer_gate_initial_logit
    ):
        payload["architecture"].pop("initializer_gate")
        payload["architecture"].pop("initializer_gate_initial_logit")
    if (
        not config.architecture.surface_gate
        and config.architecture.surface_gate_initial_logit
        == DREEMSArchitectureConfig.surface_gate_initial_logit
    ):
        payload["architecture"].pop("surface_gate")
        payload["architecture"].pop("surface_gate_initial_logit")
    if (
        config.loss.initializer_regret_weight == 0.0
        and config.loss.initializer_regret_margin == 0.0
        and config.loss.counterfactual_consistency_weight == 0.0
        and config.loss.surface_gate_weight == 0.0
        and config.loss.surface_gate_target_floor == DREEMSLossConfig.surface_gate_target_floor
    ):
        payload["loss"].pop("initializer_regret_weight")
        payload["loss"].pop("initializer_regret_margin")
        payload["loss"].pop("counterfactual_consistency_weight")
        payload["loss"].pop("surface_gate_weight")
        payload["loss"].pop("surface_gate_target_floor")
    if config.optimizer.trainable_scope == "all":
        payload["optimizer"].pop("trainable_scope")
    return payload


def model_metadata(config: DREEMSTrainConfig, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    config.validate()
    return {
        "schema_version": 2,
        "model_name": "dreems",
        "node_dim": arrays["node_x"].shape[-1],
        "edge_dim": arrays["edge_x"].shape[-1],
        "architecture": architecture_metadata(config.architecture),
        "features": feature_metadata(config.features),
        "edge_conditioner": edge_conditioner_metadata(config.edge_conditioner),
        "posterior": posterior_metadata(config.loss),
    }


def canonical_architecture_metadata(model: Mapping[str, Any]) -> dict[str, object]:
    return {
        "kind": str(model["kind"]),
        "hidden_channels": int(model["hidden_channels"]),
        "layers": int(model["layers"]),
        "context_layers": int(model["context_layers"]),
        "initializer_gate": bool(
            model.get(
                "initializer_gate",
                model.get("kind") == "diffusion_band_initializer_gate_v2",
            )
        ),
        "surface_gate": bool(
            model.get(
                "surface_gate",
                model.get("kind") == "diffusion_band_surface_gate_v3",
            )
        ),
    }


def core_architecture_metadata(model: Mapping[str, Any]) -> dict[str, object]:
    """Return architecture fields that must match when adding a new gate."""

    canonical = canonical_architecture_metadata(model)
    return {key: canonical[key] for key in ("hidden_channels", "layers", "context_layers")}


def append_training_edge_conditioner_features(
    prepared: list[dict[str, np.ndarray]],
    examples: list[dict[str, object]],
    config: DREEMSTrainConfig,
) -> list[dict[str, np.ndarray]]:
    """Append the linearized-resistance edge conditioner to prepared examples."""

    if not prepared:
        return prepared
    current_edge_dims = {item["edge_x"].shape[1] for item in prepared}
    if len(current_edge_dims) != 1:
        raise ValueError(
            f"prepared examples have inconsistent edge_x dimensions: {current_edge_dims}"
        )
    current_edge_dim = next(iter(current_edge_dims))
    model_config = {
        "edge_conditioner": edge_conditioner_metadata(config.edge_conditioner),
        "edge_dim": current_edge_dim + len(features.EDGE_CONDITIONER_FEATURE_NAMES),
    }
    return features.append_edge_conditioner_features_for_checkpoint(
        prepared,
        examples,
        model=model_config,
        stats={},
    )


def save_checkpoint(
    path: Path | str,
    *,
    params: dreems_model.DREEMS,
    normalization: dict[str, Any],
    config: DREEMSTrainConfig,
    train_losses: list[float],
    train_arrays: dict[str, np.ndarray],
    training_run: Mapping[str, Any] | None = None,
    edge_sd_calibration: Mapping[str, Any] | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "dreems-equinox-v1",
        "normalization": normalization,
        "train_losses": train_losses,
        "model": model_metadata(config, train_arrays),
        "training_config": training_config_metadata(config),
    }
    if training_run is not None:
        payload["training_run"] = dict(training_run)
    if edge_sd_calibration is not None:
        payload["edge_sd_calibration"] = dict(edge_sd_calibration)
    params_data = io.BytesIO()
    eqx.tree_serialise_leaves(params_data, params)
    with zipfile.ZipFile(checkpoint_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            checkpoints.CHECKPOINT_METADATA,
            json.dumps(payload, sort_keys=True).encode("utf-8"),
        )
        archive.writestr(checkpoints.CHECKPOINT_PARAMS, params_data.getvalue())


def batch_stream(
    arrays: dict[str, np.ndarray],
    *,
    batch_size: int,
    steps: int,
    seed: int,
) -> Iterable[dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    size = arrays["node_x"].shape[0]
    for _step in range(steps):
        indices = rng.integers(0, size, size=batch_size)
        yield {name: value[indices] for name, value in arrays.items()}


def loss_terms(
    params: dreems_model.DREEMS,
    batch: dict[str, jax.Array],
    config: DREEMSTrainConfig,
) -> dict[str, jax.Array]:
    pred, edge_logvar, context_pred, q_pred, q_logvar, neural_weight, surface_weight = (
        dreems_model.apply_batch_with_diagnostics(params, batch)
    )
    loss_config = config.loss
    edge_nll = losses.masked_gaussian_nll(
        pred,
        batch["y"],
        edge_logvar,
        batch["edge_mask"],
        min_logvar=loss_config.min_logvar,
        max_logvar=loss_config.max_logvar,
    )
    q_nll = losses.masked_gaussian_nll(
        q_pred,
        batch["q"],
        q_logvar,
        batch["node_mask"],
        min_logvar=loss_config.min_logvar,
        max_logvar=loss_config.max_logvar,
    )
    context_mse = losses.masked_mean_squared_error(
        context_pred,
        batch["y"],
        batch["edge_mask"],
    )
    slope = losses.masked_regression_slope_loss(
        pred,
        batch["y"],
        batch["edge_mask"],
        target_slope=loss_config.slope_target,
        target_floor=loss_config.slope_target_floor,
        huber_delta=loss_config.slope_huber_delta,
    )
    correlation = losses.masked_centered_correlation_loss(
        pred,
        batch["y"],
        batch["edge_mask"],
        target_floor=loss_config.correlation_target_floor,
    )
    orientation = losses.masked_edge_orientation_loss(
        pred,
        batch["y"],
        batch["edge_mask"],
        target_floor=loss_config.orientation_target_floor,
        margin_fraction=loss_config.orientation_margin_fraction,
    )
    initializer_regret = losses.masked_initializer_regret_loss(
        pred,
        batch["y"],
        batch.get("edge_initializer", jnp.zeros_like(pred)),
        batch.get("edge_initializer_valid", jnp.zeros_like(pred)),
        batch["edge_mask"],
        margin=loss_config.initializer_regret_margin,
    )
    counterfactual_consistency = losses.counterfactual_anchor_consistency_loss(
        pred,
        batch["edge_mask"],
        batch.get(
            "counterfactual_group",
            jnp.full((pred.shape[0],), -1, dtype=jnp.int32),
        ),
        batch.get(
            "counterfactual_anchor",
            jnp.zeros((pred.shape[0],), dtype=pred.dtype),
        ),
    )
    surface_gate = losses.surface_gate_binary_cross_entropy(
        surface_weight,
        batch["y"],
        batch["edge_mask"],
        target_floor=loss_config.surface_gate_target_floor,
    )
    valid_initializer = batch.get("edge_initializer_valid", jnp.zeros_like(pred))
    gate_count = jnp.sum(valid_initializer * batch["edge_mask"])
    neural_weight_mean = jnp.sum(neural_weight * valid_initializer * batch["edge_mask"])
    neural_weight_mean /= jnp.maximum(gate_count, 1.0)
    total = (
        edge_nll
        + loss_config.context_weight * context_mse
        + loss_config.q_weight * q_nll
        + loss_config.slope_weight * slope
        + loss_config.correlation_weight * correlation
        + loss_config.orientation_weight * orientation
        + loss_config.initializer_regret_weight * initializer_regret
        + loss_config.counterfactual_consistency_weight * counterfactual_consistency
        + loss_config.surface_gate_weight * surface_gate
    )
    return {
        "loss": total,
        "edge_nll": edge_nll,
        "q_nll": q_nll,
        "context_mse": context_mse,
        "slope": slope,
        "correlation": correlation,
        "orientation": orientation,
        "initializer_regret": initializer_regret,
        "counterfactual_consistency": counterfactual_consistency,
        "surface_gate_bce": surface_gate,
        "surface_weight_mean": jnp.mean(surface_weight),
        "neural_weight_mean": neural_weight_mean,
    }


def scalar_terms(terms: dict[str, jax.Array]) -> dict[str, float]:
    return {key: float(jax.device_get(value)) for key, value in terms.items()}


def validation_metrics(
    params: dreems_model.DREEMS,
    arrays: dict[str, np.ndarray],
    config: DREEMSTrainConfig,
    *,
    batch_size: int | None = None,
    terms_fn: Callable[[dreems_model.DREEMS, dict[str, jax.Array]], dict[str, jax.Array]]
    | None = None,
) -> dict[str, float]:
    num_examples = int(arrays["node_x"].shape[0])
    if num_examples <= 0:
        raise ValueError("validation arrays must contain at least one example")
    if batch_size is None:
        batch_size = num_examples
    if batch_size <= 0:
        raise ValueError("validation batch_size must be positive")

    compute_terms = (
        (lambda candidate, batch: loss_terms(candidate, batch, config))
        if terms_fn is None
        else terms_fn
    )
    totals: dict[str, float] = {}
    total_weight = 0
    for start in range(0, num_examples, batch_size):
        stop = min(start + batch_size, num_examples)
        batch = {
            name: jax.device_put(jnp.asarray(value[start:stop])) for name, value in arrays.items()
        }
        terms = scalar_terms(compute_terms(params, batch))
        weight = stop - start
        total_weight += weight
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.0) + value * weight

    return {f"validation_{key}": value / total_weight for key, value in totals.items()}


def train_from_batches(
    *,
    config: DREEMSTrainConfig,
    node_dim: int,
    edge_dim: int,
    batches: Iterable[dict[str, np.ndarray]],
    init_params: dreems_model.DREEMS | None = None,
    validation_arrays: dict[str, np.ndarray] | None = None,
    checkpoint_callback=None,
) -> TrainingRunResult:
    config.validate()
    architecture = config.architecture
    optimizer_config = config.optimizer
    key = jax.random.PRNGKey(optimizer_config.seed)
    if init_params is None:
        params = dreems_model.DREEMS(
            key,
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_channels=architecture.hidden_channels,
            layers=architecture.layers,
            context_layers=architecture.context_layers,
            initializer_gate=architecture.initializer_gate,
            initializer_gate_initial_logit=architecture.initializer_gate_initial_logit,
            surface_gate=architecture.surface_gate,
            surface_gate_initial_logit=architecture.surface_gate_initial_logit,
        )
    else:
        params = dreems_model.tree_to_jax(init_params)
        if architecture.initializer_gate and params.initializer_gate_hidden is None:
            params = dreems_model.add_initializer_gate(
                params,
                key=key,
                node_dim=node_dim,
                edge_dim=edge_dim,
                hidden_channels=architecture.hidden_channels,
                layers=architecture.layers,
                context_layers=architecture.context_layers,
                initial_logit=architecture.initializer_gate_initial_logit,
            )
        elif not architecture.initializer_gate and params.initializer_gate_hidden is not None:
            raise ValueError("cannot initialize a v1 model from a gated checkpoint")
        if architecture.surface_gate and params.surface_gate_hidden is None:
            params = dreems_model.add_surface_gate(
                params,
                key=jax.random.fold_in(key, 1),
                node_dim=node_dim,
                edge_dim=edge_dim,
                hidden_channels=architecture.hidden_channels,
                layers=architecture.layers,
                context_layers=architecture.context_layers,
                initial_logit=architecture.surface_gate_initial_logit,
            )
        elif not architecture.surface_gate and params.surface_gate_hidden is not None:
            raise ValueError("cannot initialize a model without the checkpoint's surface gate")
    optimizer = optax.chain(
        optax.clip_by_global_norm(optimizer_config.gradient_clip_norm),
        optax.adam(optimizer_config.learning_rate),
    )
    if optimizer_config.trainable_scope in {"initializer_gate", "surface_gate"}:
        if optimizer_config.trainable_scope == "initializer_gate":
            if params.initializer_gate_hidden is None or params.initializer_gate_out is None:
                raise ValueError("initializer_gate scope requires an enabled initializer gate")

            def gate_selector(model: dreems_model.DREEMS):
                return model.initializer_gate_hidden, model.initializer_gate_out

        else:
            if params.surface_gate_hidden is None or params.surface_gate_out is None:
                raise ValueError("surface_gate scope requires an enabled surface gate")

            def gate_selector(model: dreems_model.DREEMS):
                return model.surface_gate_hidden, model.surface_gate_out

        gate_params = gate_selector(params)
        opt_state = optimizer.init(gate_params)

        @jax.jit
        def train_step(
            current_params: dreems_model.DREEMS,
            current_state: optax.OptState,
            batch: dict[str, jax.Array],
        ) -> tuple[dreems_model.DREEMS, optax.OptState, jax.Array]:
            current_gate = gate_selector(current_params)

            def objective(candidate_gate: tuple[eqx.nn.Linear, eqx.nn.Linear]) -> jax.Array:
                candidate = eqx.tree_at(gate_selector, current_params, candidate_gate)
                return loss_terms(candidate, batch, config)["loss"]

            loss, grads = jax.value_and_grad(objective)(current_gate)
            updates, next_state = optimizer.update(grads, current_state, current_gate)
            next_gate = optax.apply_updates(current_gate, updates)
            next_params = eqx.tree_at(gate_selector, current_params, next_gate)
            return next_params, next_state, loss

    else:
        opt_state = optimizer.init(params)

        @jax.jit
        def train_step(
            current_params: dreems_model.DREEMS,
            current_state: optax.OptState,
            batch: dict[str, jax.Array],
        ) -> tuple[dreems_model.DREEMS, optax.OptState, jax.Array]:
            def objective(candidate_params: dreems_model.DREEMS) -> jax.Array:
                return loss_terms(candidate_params, batch, config)["loss"]

            loss, grads = jax.value_and_grad(objective)(current_params)
            updates, next_state = optimizer.update(grads, current_state, current_params)
            next_params = optax.apply_updates(current_params, updates)
            return next_params, next_state, loss

    @jax.jit
    def validation_step(
        current_params: dreems_model.DREEMS,
        batch: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        return loss_terms(current_params, batch, config)

    losses_out: list[float] = []
    history: list[dict[str, object]] = []
    best_params: dreems_model.DREEMS | None = None
    best_step: int | None = None
    best_validation_loss: float | None = None
    best_validation_metrics: dict[str, object] | None = None
    checks_since_best = 0
    final_step = 0
    stopped_early = False
    stop_reason: str | None = None
    log_every = optimizer_config.log_every or max(1, optimizer_config.steps // 20)
    for step, batch_arrays in enumerate(batches, start=1):
        if step > optimizer_config.steps:
            break
        final_step = step
        batch = {name: jax.device_put(jnp.asarray(value)) for name, value in batch_arrays.items()}
        params, opt_state, loss = train_step(params, opt_state, batch)
        loss_value = float(jax.device_get(loss))
        if step == 1 or step == optimizer_config.steps or step % log_every == 0:
            losses_out.append(loss_value)
            row: dict[str, object] = {"stage": "train", "step": step, "loss": loss_value}
            if validation_arrays is not None:
                row.update(
                    validation_metrics(
                        params,
                        validation_arrays,
                        config,
                        batch_size=optimizer_config.batch_size,
                        terms_fn=validation_step,
                    )
                )
                metric = row.get(optimizer_config.early_stopping_metric)
                if metric is None:
                    raise ValueError(
                        f"early stopping metric {optimizer_config.early_stopping_metric!r} "
                        "was not computed"
                    )
                metric_value = float(metric)
                if np.isfinite(metric_value):
                    improved = (
                        best_validation_loss is None
                        or metric_value
                        < best_validation_loss - optimizer_config.early_stopping_min_delta
                    )
                    if improved:
                        best_validation_loss = metric_value
                        best_validation_metrics = dict(row)
                        best_step = step
                        checks_since_best = 0
                        if optimizer_config.restore_best_validation:
                            best_params = params
                    else:
                        checks_since_best += 1
                    row["validation_best"] = improved
                    row["validation_checks_since_best"] = checks_since_best
                    if (
                        optimizer_config.early_stopping_patience is not None
                        and step >= optimizer_config.early_stopping_warmup_steps
                        and checks_since_best >= optimizer_config.early_stopping_patience
                    ):
                        stopped_early = True
                        stop_reason = (
                            f"{optimizer_config.early_stopping_metric} did not improve by "
                            f"{optimizer_config.early_stopping_min_delta} for "
                            f"{checks_since_best} validation checks"
                        )
                        row["stopped_early"] = True
            print(json.dumps(row), flush=True)
            history.append(row)
            if checkpoint_callback is not None:
                checkpoint_callback(step, params, list(losses_out), dict(row))
            if stopped_early:
                break
    restored = (
        validation_arrays is not None
        and optimizer_config.restore_best_validation
        and best_params is not None
    )
    return TrainingRunResult(
        params=best_params if restored else params,
        losses=losses_out,
        history=history,
        final_step=final_step,
        best_step=best_step,
        best_validation_loss=best_validation_loss,
        early_stopping_metric=optimizer_config.early_stopping_metric,
        best_validation_metric=best_validation_loss,
        best_validation_metrics=best_validation_metrics,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        restored_best_validation=restored,
    )


def train_from_arrays(
    arrays: dict[str, np.ndarray],
    *,
    config: DREEMSTrainConfig,
    init_params: dreems_model.DREEMS | None = None,
    validation_arrays: dict[str, np.ndarray] | None = None,
    checkpoint_callback=None,
) -> TrainingRunResult:
    return train_from_batches(
        config=config,
        node_dim=arrays["node_x"].shape[-1],
        edge_dim=arrays["edge_x"].shape[-1],
        batches=batch_stream(
            arrays,
            batch_size=config.optimizer.batch_size,
            steps=config.optimizer.steps,
            seed=config.optimizer.seed + 10,
        ),
        init_params=init_params,
        validation_arrays=validation_arrays,
        checkpoint_callback=checkpoint_callback,
    )


def prepare_training_arrays(
    train_examples: list[dict[str, object]],
    validation_examples: list[dict[str, object]] | None,
    *,
    config: DREEMSTrainConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None, dict[str, object]]:
    train_prepared = features.prepare_examples(
        train_examples,
        covariance_components=config.features.covariance_components,
        spatial_encoding=config.features.spatial_encoding,
    )
    train_prepared = append_training_edge_conditioner_features(
        train_prepared,
        train_examples,
        config,
    )
    validation_prepared = None
    if validation_examples is not None:
        validation_prepared = features.prepare_examples(
            validation_examples,
            covariance_components=config.features.covariance_components,
            spatial_encoding=config.features.spatial_encoding,
        )
        validation_prepared = append_training_edge_conditioner_features(
            validation_prepared,
            validation_examples,
            config,
        )
    train_norm, validation_norm, stats = features.normalize_prepared(
        train_prepared,
        validation_prepared,
        std_floor=config.features.normalization_std_floor,
    )
    train_arrays = features.pad_examples(train_norm)
    validation_arrays = (
        features.pad_examples(validation_norm) if validation_norm is not None else None
    )
    return train_arrays, validation_arrays, stats
