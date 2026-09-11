"""The released DREEMS graph neural network."""
# ruff: noqa: F722, F821

from operator import itemgetter
from typing import Any, NotRequired, TypeAlias, TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped

NodeX: TypeAlias = Float[Array, "nodes node_dim"]
EdgeX: TypeAlias = Float[Array, "edges edge_dim"]
Edges: TypeAlias = Int[Array, "edges 2"]
NodeMask: TypeAlias = Float[Array, "nodes"]
EdgeMask: TypeAlias = Float[Array, "edges"]
OneOutput: TypeAlias = tuple[
    Float[Array, "edges"],
    Float[Array, "edges"],
    Float[Array, "edges"],
    Float[Array, "nodes"],
    Float[Array, "nodes"],
]
OneOutputWithGate: TypeAlias = tuple[
    Float[Array, "edges"],
    Float[Array, "edges"],
    Float[Array, "edges"],
    Float[Array, "nodes"],
    Float[Array, "nodes"],
    Float[Array, "edges"],
]
OneOutputWithDiagnostics: TypeAlias = tuple[
    Float[Array, "edges"],
    Float[Array, "edges"],
    Float[Array, "edges"],
    Float[Array, "nodes"],
    Float[Array, "nodes"],
    Float[Array, "edges"],
    Float[Array, ""],
]


class DREEMSBatch(TypedDict):
    """Padded graphs on the JAX device; masks exclude padding from inference."""

    node_x: Float[Array, "batch nodes node_dim"]
    edge_x: Float[Array, "batch edges edge_dim"]
    edges: Int[Array, "batch edges 2"]
    node_mask: Float[Array, "batch nodes"]
    edge_mask: Float[Array, "batch edges"]
    edge_initializer: NotRequired[Float[Array, "batch edges"]]
    edge_initializer_valid: NotRequired[Float[Array, "batch edges"]]


class DREEMSArrays(TypedDict):
    """Host arrays with the same layout as :class:`DREEMSBatch`."""

    node_x: Float[np.ndarray, "batch nodes node_dim"]
    edge_x: Float[np.ndarray, "batch edges edge_dim"]
    edges: Int[np.ndarray, "batch edges 2"]
    node_mask: Float[np.ndarray, "batch nodes"]
    edge_mask: Float[np.ndarray, "batch edges"]
    edge_initializer: NotRequired[Float[np.ndarray, "batch edges"]]
    edge_initializer_valid: NotRequired[Float[np.ndarray, "batch edges"]]


class DREEMSTrainBatch(DREEMSBatch):
    """Model inputs and normalized targets on the device."""

    y: Float[Array, "batch edges"]
    q: Float[Array, "batch nodes"]
    counterfactual_group: NotRequired[Int[Array, "batch"]]
    counterfactual_anchor: NotRequired[Float[Array, "batch"]]
    counterfactual_mask: NotRequired[Int[Array, "batch"]]
    sampled_node_fraction: NotRequired[Float[Array, "batch"]]


class DREEMSTrainArrays(DREEMSArrays):
    """Padded features and normalized targets before transfer to the device."""

    y: Float[np.ndarray, "batch edges"]
    q: Float[np.ndarray, "batch nodes"]
    counterfactual_group: NotRequired[Int[np.ndarray, "batch"]]
    counterfactual_anchor: NotRequired[Float[np.ndarray, "batch"]]
    counterfactual_mask: NotRequired[Int[np.ndarray, "batch"]]
    sampled_node_fraction: NotRequired[Float[np.ndarray, "batch"]]


BatchOutput: TypeAlias = tuple[
    Float[Array, "batch edges"],
    Float[Array, "batch edges"],
    Float[Array, "batch edges"],
    Float[Array, "batch nodes"],
    Float[Array, "batch nodes"],
]
BatchOutputWithGate: TypeAlias = tuple[
    Float[Array, "batch edges"],
    Float[Array, "batch edges"],
    Float[Array, "batch edges"],
    Float[Array, "batch nodes"],
    Float[Array, "batch nodes"],
    Float[Array, "batch edges"],
]
BatchOutputWithDiagnostics: TypeAlias = tuple[
    Float[Array, "batch edges"],
    Float[Array, "batch edges"],
    Float[Array, "batch edges"],
    Float[Array, "batch nodes"],
    Float[Array, "batch nodes"],
    Float[Array, "batch edges"],
    Float[Array, "batch"],
]
PredictionOutput: TypeAlias = tuple[
    Float[np.ndarray, "batch edges"],
    Float[np.ndarray, "batch edges"],
    Float[np.ndarray, "batch nodes"],
    Float[np.ndarray, "batch nodes"],
]
PredictionOutputWithDiagnostics: TypeAlias = tuple[
    Float[np.ndarray, "batch edges"],
    Float[np.ndarray, "batch edges"],
    Float[np.ndarray, "batch nodes"],
    Float[np.ndarray, "batch nodes"],
    Float[np.ndarray, "batch"],
]

DREEMS_BATCH_FIELDS = itemgetter("node_x", "edge_x", "edges", "node_mask", "edge_mask")


@jaxtyped(typechecker=beartype)
def tree_to_jax(tree: Any) -> Any:
    """Move NumPy leaves to JAX without changing the model's PyTree structure."""

    return jax.tree.map(
        lambda value: jnp.asarray(value) if isinstance(value, np.ndarray) else value,
        tree,
    )


class DREEMS(eqx.Module):
    """Fixed diffusion-band architecture used by the released checkpoint.

    A local GNN uses edge features in each message. A second branch supplies
    context on the same graph at full resolution. Each edge then
    attends to all nodes before predicting a mean and log variance. The input
    features already contain the diffusion bands, computed in ``features.py``.

    Optional gates blend the neural estimate with the linear initializer and
    shrink the whole surface toward zero. All means are in normalized log-rate
    units here; ``predict`` restores the training scale.
    """

    # Field order preserves the Equinox leaf order of the released checkpoint.
    input: eqx.nn.Linear
    self_layers: tuple[eqx.nn.Linear, ...]
    neigh_layers: tuple[eqx.nn.Linear, ...]
    context_self_layers: tuple[eqx.nn.Linear, ...]
    context_neigh_layers: tuple[eqx.nn.Linear, ...]
    fusion: eqx.nn.Linear
    context_edge_hidden: eqx.nn.Linear
    context_edge_out: eqx.nn.Linear
    global_query: eqx.nn.Linear
    global_key: eqx.nn.Linear
    global_value: eqx.nn.Linear
    edge_hidden: eqx.nn.Linear
    edge_out: eqx.nn.Linear
    edge_logvar_out: eqx.nn.Linear
    node_q_hidden: eqx.nn.Linear
    node_q_out: eqx.nn.Linear
    node_q_logvar_out: eqx.nn.Linear
    edge_feature_hidden: eqx.nn.Linear
    edge_feature_gate: eqx.nn.Linear
    edge_message_layers: tuple[eqx.nn.Linear, ...]
    edge_gate_layers: tuple[eqx.nn.Linear, ...]
    initializer_gate_hidden: eqx.nn.Linear | None
    initializer_gate_out: eqx.nn.Linear | None
    surface_gate_hidden: eqx.nn.Linear | None
    surface_gate_out: eqx.nn.Linear | None

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        node_dim: int,
        edge_dim: int,
        hidden_channels: int,
        layers: int,
        context_layers: int,
        initializer_gate: bool = False,
        initializer_gate_initial_logit: float = 6.0,
        surface_gate: bool = False,
        surface_gate_initial_logit: float = 6.0,
    ) -> None:
        def next_key() -> PRNGKeyArray:
            nonlocal key
            key, result = jax.random.split(key)
            return result

        self.input = eqx.nn.Linear(node_dim, hidden_channels, key=next_key())
        self.self_layers = tuple(
            eqx.nn.Linear(hidden_channels, hidden_channels, key=next_key()) for _ in range(layers)
        )
        self.neigh_layers = tuple(
            eqx.nn.Linear(hidden_channels, hidden_channels, key=next_key()) for _ in range(layers)
        )
        self.context_self_layers = tuple(
            eqx.nn.Linear(hidden_channels, hidden_channels, key=next_key())
            for _ in range(context_layers)
        )
        self.context_neigh_layers = tuple(
            eqx.nn.Linear(hidden_channels, hidden_channels, key=next_key())
            for _ in range(context_layers)
        )
        self.fusion = eqx.nn.Linear(2 * hidden_channels, hidden_channels, key=next_key())
        self.context_edge_hidden = eqx.nn.Linear(
            2 * hidden_channels,
            hidden_channels,
            key=next_key(),
        )
        self.context_edge_out = eqx.nn.Linear(hidden_channels, 1, key=next_key())
        self.global_query = eqx.nn.Linear(
            2 * hidden_channels + edge_dim,
            hidden_channels,
            key=next_key(),
        )
        self.global_key = eqx.nn.Linear(hidden_channels, hidden_channels, key=next_key())
        self.global_value = eqx.nn.Linear(hidden_channels, hidden_channels, key=next_key())
        self.edge_hidden = eqx.nn.Linear(
            3 * hidden_channels + edge_dim,
            hidden_channels,
            key=next_key(),
        )
        self.edge_out = eqx.nn.Linear(hidden_channels, 1, key=next_key())

        edge_logvar_out = eqx.nn.Linear(hidden_channels, 1, key=next_key())
        self.edge_logvar_out = eqx.tree_at(
            lambda layer: (layer.weight, layer.bias),
            edge_logvar_out,
            (jnp.zeros_like(edge_logvar_out.weight), jnp.zeros_like(edge_logvar_out.bias)),
        )
        self.node_q_hidden = eqx.nn.Linear(hidden_channels, hidden_channels, key=next_key())
        self.node_q_out = eqx.nn.Linear(hidden_channels, 1, key=next_key())
        node_q_logvar_out = eqx.nn.Linear(hidden_channels, 1, key=next_key())
        self.node_q_logvar_out = eqx.tree_at(
            lambda layer: (layer.weight, layer.bias),
            node_q_logvar_out,
            (jnp.zeros_like(node_q_logvar_out.weight), jnp.zeros_like(node_q_logvar_out.bias)),
        )
        self.edge_feature_hidden = eqx.nn.Linear(edge_dim, hidden_channels, key=next_key())
        self.edge_feature_gate = eqx.nn.Linear(edge_dim, hidden_channels, key=next_key())
        self.edge_message_layers = tuple(
            eqx.nn.Linear(edge_dim, hidden_channels, key=next_key()) for _ in range(layers)
        )
        self.edge_gate_layers = tuple(
            eqx.nn.Linear(edge_dim, hidden_channels, key=next_key()) for _ in range(layers)
        )
        if initializer_gate:
            self.initializer_gate_hidden = eqx.nn.Linear(
                hidden_channels,
                hidden_channels,
                key=next_key(),
            )
            initializer_gate_out = eqx.nn.Linear(hidden_channels, 1, key=next_key())
            self.initializer_gate_out = eqx.tree_at(
                lambda layer: (layer.weight, layer.bias),
                initializer_gate_out,
                (
                    jnp.zeros_like(initializer_gate_out.weight),
                    jnp.full_like(initializer_gate_out.bias, initializer_gate_initial_logit),
                ),
            )
        else:
            self.initializer_gate_hidden = None
            self.initializer_gate_out = None
        if surface_gate:
            self.surface_gate_hidden = eqx.nn.Linear(
                2 * hidden_channels + 3,
                hidden_channels,
                key=next_key(),
            )
            surface_gate_out = eqx.nn.Linear(hidden_channels, 1, key=next_key())
            self.surface_gate_out = eqx.tree_at(
                lambda layer: (layer.weight, layer.bias),
                surface_gate_out,
                (
                    jnp.zeros_like(surface_gate_out.weight),
                    jnp.full_like(surface_gate_out.bias, surface_gate_initial_logit),
                ),
            )
        else:
            self.surface_gate_hidden = None
            self.surface_gate_out = None

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def _propagate(
        z: Float[Array, "nodes hidden"],
        edge_x: EdgeX,
        edges: Edges,
        node_mask: NodeMask,
        edge_mask: EdgeMask,
        self_layers: tuple[eqx.nn.Linear, ...],
        neigh_layers: tuple[eqx.nn.Linear, ...],
        edge_message_layers: tuple[eqx.nn.Linear, ...] | None = None,
        edge_gate_layers: tuple[eqx.nn.Linear, ...] | None = None,
    ) -> Float[Array, "nodes hidden"]:
        """Send messages in both directions, then add a masked residual update."""

        sender = jnp.concatenate([edges[:, 0], edges[:, 1]])
        receiver = jnp.concatenate([edges[:, 1], edges[:, 0]])
        directed_mask = jnp.concatenate([edge_mask, edge_mask])
        directed_edge_x = jnp.concatenate([edge_x, edge_x], axis=0)
        degree = jax.ops.segment_sum(directed_mask, receiver, num_segments=z.shape[0])
        for index, (self_layer, neigh_layer) in enumerate(
            zip(self_layers, neigh_layers, strict=True)
        ):
            messages = z[sender]
            if edge_message_layers is not None and edge_gate_layers is not None:
                edge_message = jax.nn.gelu(jax.vmap(edge_message_layers[index])(directed_edge_x))
                edge_gate = jax.nn.sigmoid(jax.vmap(edge_gate_layers[index])(directed_edge_x))
                messages = messages * edge_gate + edge_message
            messages *= directed_mask[:, None]
            summed = jax.ops.segment_sum(messages, receiver, num_segments=z.shape[0])
            neighbor_mean = summed / jnp.maximum(degree[:, None], 1.0)
            update = (
                jax.nn.gelu(jax.vmap(self_layer)(z) + jax.vmap(neigh_layer)(neighbor_mean))
                * node_mask[:, None]
            )
            z = (z + update) * node_mask[:, None]
        return z

    @jaxtyped(typechecker=beartype)
    def apply_one(
        self,
        node_x: NodeX,
        edge_x: EdgeX,
        edges: Edges,
        node_mask: NodeMask,
        edge_mask: EdgeMask,
        edge_initializer: EdgeMask | None = None,
        edge_initializer_valid: EdgeMask | None = None,
    ) -> OneOutput:
        """Return edge mean/log variance, context mean, and node q mean/log variance."""

        output = self.apply_one_with_diagnostics(
            node_x,
            edge_x,
            edges,
            node_mask,
            edge_mask,
            edge_initializer,
            edge_initializer_valid,
        )
        return output[:5]

    @jaxtyped(typechecker=beartype)
    def apply_one_with_gate(
        self,
        node_x: NodeX,
        edge_x: EdgeX,
        edges: Edges,
        node_mask: NodeMask,
        edge_mask: EdgeMask,
        edge_initializer: EdgeMask | None = None,
        edge_initializer_valid: EdgeMask | None = None,
    ) -> OneOutputWithGate:
        """Also return the neural fraction of each edge's initializer blend."""

        return self.apply_one_with_diagnostics(
            node_x,
            edge_x,
            edges,
            node_mask,
            edge_mask,
            edge_initializer,
            edge_initializer_valid,
        )[:6]

    @jaxtyped(typechecker=beartype)
    def apply_one_with_diagnostics(
        self,
        node_x: NodeX,
        edge_x: EdgeX,
        edges: Edges,
        node_mask: NodeMask,
        edge_mask: EdgeMask,
        edge_initializer: EdgeMask | None = None,
        edge_initializer_valid: EdgeMask | None = None,
    ) -> OneOutputWithDiagnostics:
        z = jax.nn.gelu(jax.vmap(self.input)(node_x)) * node_mask[:, None]
        z = self._propagate(
            z,
            edge_x,
            edges,
            node_mask,
            edge_mask,
            self.self_layers,
            self.neigh_layers,
            self.edge_message_layers,
            self.edge_gate_layers,
        )

        # The released model's context graph is the identity coarsening: it is a
        # second message-passing branch over the same nodes and edges.
        context_z = self._propagate(
            z,
            edge_x,
            edges,
            node_mask,
            edge_mask,
            self.context_self_layers,
            self.context_neigh_layers,
        )
        context_left = context_z[edges[:, 0]]
        context_right = context_z[edges[:, 1]]
        context_features = jnp.concatenate(
            [context_left + context_right, jnp.abs(context_left - context_right)],
            axis=1,
        )
        context_hidden = jax.nn.gelu(jax.vmap(self.context_edge_hidden)(context_features))
        context_pred = jnp.squeeze(jax.vmap(self.context_edge_out)(context_hidden), axis=1)
        context_pred -= jnp.sum(context_pred * edge_mask) / jnp.maximum(jnp.sum(edge_mask), 1.0)
        context_pred *= edge_mask

        z = jax.nn.gelu(jax.vmap(self.fusion)(jnp.concatenate([z, context_z], axis=1)))
        z *= node_mask[:, None]

        q_hidden = jax.nn.gelu(jax.vmap(self.node_q_hidden)(z))
        q_pred = jnp.squeeze(jax.vmap(self.node_q_out)(q_hidden), axis=1)
        q_logvar = jnp.squeeze(jax.vmap(self.node_q_logvar_out)(q_hidden), axis=1)
        q_pred -= jnp.sum(q_pred * node_mask) / jnp.maximum(jnp.sum(node_mask), 1.0)
        q_pred *= node_mask

        left = z[edges[:, 0]]
        right = z[edges[:, 1]]
        local_edge_features = jnp.concatenate(
            [left + right, jnp.abs(left - right), edge_x],
            axis=1,
        )
        query = jax.vmap(self.global_query)(local_edge_features)
        key = jax.vmap(self.global_key)(context_z)
        value = jax.vmap(self.global_value)(context_z)
        global_context = jax.nn.dot_product_attention(
            query[:, None, :],
            key[:, None, :],
            value[:, None, :],
            mask=node_mask[None, :] > 0.0,
        )
        global_context = jnp.squeeze(global_context, axis=1) * edge_mask[:, None]
        hidden = jax.nn.gelu(
            jax.vmap(self.edge_hidden)(
                jnp.concatenate([local_edge_features, global_context], axis=1)
            )
        )
        edge_condition = jax.nn.gelu(jax.vmap(self.edge_feature_hidden)(edge_x))
        edge_gate = jax.nn.sigmoid(jax.vmap(self.edge_feature_gate)(edge_x))
        hidden += edge_gate * edge_condition * edge_mask[:, None]

        proposal = jnp.squeeze(jax.vmap(self.edge_out)(hidden), axis=1)
        proposal -= jnp.sum(proposal * edge_mask) / jnp.maximum(jnp.sum(edge_mask), 1.0)
        proposal *= edge_mask
        if edge_initializer is None:
            edge_initializer = jnp.zeros_like(edge_mask)
        if edge_initializer_valid is None:
            edge_initializer_valid = jnp.zeros_like(edge_mask)
        initializer_valid = (
            (edge_initializer_valid > 0.0) & jnp.isfinite(edge_initializer)
        ).astype(edge_mask.dtype)
        safe_initializer = jnp.where(initializer_valid > 0.0, edge_initializer, 0.0)
        if self.initializer_gate_hidden is None or self.initializer_gate_out is None:
            pred = proposal
            neural_weight = edge_mask
        else:
            gate_hidden = jax.nn.gelu(jax.vmap(self.initializer_gate_hidden)(hidden))
            gate_logit = jnp.squeeze(jax.vmap(self.initializer_gate_out)(gate_hidden), axis=1)
            neural_weight = jax.nn.sigmoid(gate_logit)
            neural_weight = (
                initializer_valid * neural_weight + (1.0 - initializer_valid)
            ) * edge_mask
            pred = safe_initializer + neural_weight * (proposal - safe_initializer)
            pred -= jnp.sum(pred * edge_mask) / jnp.maximum(jnp.sum(edge_mask), 1.0)
            pred *= edge_mask
        edge_logvar = jnp.squeeze(jax.vmap(self.edge_logvar_out)(hidden), axis=1)
        if self.surface_gate_hidden is None or self.surface_gate_out is None:
            surface_weight = jnp.asarray(1.0, dtype=pred.dtype)
        else:
            edge_count = jnp.maximum(jnp.sum(edge_mask), 1.0)
            hidden_mean = jnp.sum(hidden * edge_mask[:, None], axis=0) / edge_count
            hidden_rms = jnp.sqrt(
                jnp.sum(hidden * hidden * edge_mask[:, None], axis=0) / edge_count + 1e-8
            )
            signal_rms = jnp.sqrt(jnp.sum(pred * pred * edge_mask) / edge_count + 1e-8)
            mean_logvar = jnp.sum(edge_logvar * edge_mask) / edge_count
            valid_fraction = jnp.sum(initializer_valid * edge_mask) / edge_count
            summary = jnp.concatenate(
                [
                    hidden_mean,
                    hidden_rms,
                    jnp.stack(
                        [
                            jnp.log(signal_rms + 1e-6),
                            jnp.log(signal_rms + 1e-6) - 0.5 * mean_logvar,
                            valid_fraction,
                        ]
                    ),
                ]
            )
            surface_hidden = jax.nn.gelu(self.surface_gate_hidden(summary))
            surface_logit = jnp.squeeze(self.surface_gate_out(surface_hidden), axis=0)
            surface_weight = jax.nn.sigmoid(surface_logit)
            pred *= surface_weight
            pred *= edge_mask
        return (
            pred,
            edge_logvar * edge_mask,
            context_pred,
            q_pred,
            q_logvar * node_mask,
            neural_weight,
            surface_weight,
        )


@jaxtyped(typechecker=beartype)
def apply_batch(model: DREEMS, batch: DREEMSBatch) -> BatchOutput:
    """Evaluate independent padded graphs with shared model parameters."""

    return apply_batch_with_gate(model, batch)[:5]


@jaxtyped(typechecker=beartype)
def apply_batch_with_gate(model: DREEMS, batch: DREEMSBatch) -> BatchOutputWithGate:
    """Return batched predictions and the per-edge neural blend weights."""

    return apply_batch_with_diagnostics(model, batch)[:6]


@jaxtyped(typechecker=beartype)
def apply_batch_with_diagnostics(
    model: DREEMS,
    batch: DREEMSBatch,
) -> BatchOutputWithDiagnostics:
    """Include both gates; a missing initializer uses the neural estimate alone."""

    edge_mask = batch["edge_mask"]
    edge_initializer = batch.get("edge_initializer", jnp.zeros_like(edge_mask))
    edge_initializer_valid = batch.get(
        "edge_initializer_valid",
        jnp.zeros_like(edge_mask),
    )
    return jax.vmap(model.apply_one_with_diagnostics)(
        *DREEMS_BATCH_FIELDS(batch),
        edge_initializer,
        edge_initializer_valid,
    )


@jaxtyped(typechecker=beartype)
def add_initializer_gate(
    source: DREEMS,
    *,
    key: PRNGKeyArray,
    node_dim: int,
    edge_dim: int,
    hidden_channels: int,
    layers: int,
    context_layers: int,
    initial_logit: float,
) -> DREEMS:
    """Add a gate while copying every parameter from an existing v1 model."""

    if source.initializer_gate_hidden is not None or source.initializer_gate_out is not None:
        return source
    target = DREEMS(
        key,
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_channels=hidden_channels,
        layers=layers,
        context_layers=context_layers,
        initializer_gate=True,
        initializer_gate_initial_logit=initial_logit,
        surface_gate=source.surface_gate_hidden is not None,
    )
    # Replace only the absent gate. Every trained leaf stays on the source model.
    return eqx.tree_at(
        lambda candidate: (candidate.initializer_gate_hidden, candidate.initializer_gate_out),
        source,
        (target.initializer_gate_hidden, target.initializer_gate_out),
        is_leaf=lambda value: value is None,
    )


@jaxtyped(typechecker=beartype)
def add_surface_gate(
    source: DREEMS,
    *,
    key: PRNGKeyArray,
    node_dim: int,
    edge_dim: int,
    hidden_channels: int,
    layers: int,
    context_layers: int,
    initial_logit: float,
) -> DREEMS:
    """Add a graph-level surface gate while preserving existing parameters."""

    if source.surface_gate_hidden is not None or source.surface_gate_out is not None:
        return source
    target = DREEMS(
        key,
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_channels=hidden_channels,
        layers=layers,
        context_layers=context_layers,
        initializer_gate=source.initializer_gate_hidden is not None,
        surface_gate=True,
        surface_gate_initial_logit=initial_logit,
    )
    return eqx.tree_at(
        lambda candidate: (candidate.surface_gate_hidden, candidate.surface_gate_out),
        source,
        (target.surface_gate_hidden, target.surface_gate_out),
        is_leaf=lambda value: value is None,
    )


@jaxtyped(typechecker=beartype)
def model_logvar_bounds(metadata: dict[str, Any]) -> tuple[float | None, float | None]:
    """Read optional bounds in normalized log-variance units from the checkpoint."""

    posterior = metadata.get("posterior", {})
    if not isinstance(posterior, dict) or posterior.get("kind") != "factorized_normal":
        return None, None
    minimum = posterior.get("min_logvar")
    maximum = posterior.get("max_logvar")
    if minimum is None or maximum is None:
        return None, None
    return float(minimum), float(maximum)


@jaxtyped(typechecker=beartype)
def validate_production_model_metadata(metadata: dict[str, Any]) -> None:
    """Reject checkpoints that do not describe the released architecture."""

    if metadata.get("schema_version") != 2:
        raise ValueError("checkpoint model schema_version must be 2")
    architecture = metadata.get("architecture")
    feature_metadata = metadata.get("features")
    conditioner = metadata.get("edge_conditioner")
    if not isinstance(architecture, dict):
        raise ValueError("checkpoint is missing architecture metadata")
    if not isinstance(feature_metadata, dict):
        raise ValueError("checkpoint is missing feature metadata")
    if not isinstance(conditioner, dict):
        raise ValueError("checkpoint is missing edge conditioner metadata")
    if architecture.get("kind") not in {
        "diffusion_band_gnn_v1",
        "diffusion_band_initializer_gate_v2",
        "diffusion_band_surface_gate_v3",
    }:
        raise ValueError("checkpoint does not use the released DREEMS architecture")
    if feature_metadata.get("kind") not in {
        "diffusion_bands_v1",
        "topology_diffusion_bands_v1",
        "topology_diffusion_bands_v2",
    }:
        raise ValueError("checkpoint does not use the released diffusion-band features")
    if conditioner.get("method") != "all_pair_diagonal_adjoint":
        raise ValueError("checkpoint does not use the released all-pair conditioner")


@jaxtyped(typechecker=beartype)
def predict(
    params: DREEMS,
    arrays: DREEMSArrays,
    *,
    y_scale: float,
    q_y_scale: float,
    model: dict[str, Any],
) -> PredictionOutput:
    """Return NumPy means and log variances in the original log-rate units."""

    pred, edge_logvar, q_pred, q_logvar, _surface_weight = predict_with_diagnostics(
        params,
        arrays,
        y_scale=y_scale,
        q_y_scale=q_y_scale,
        model=model,
    )
    return pred, edge_logvar, q_pred, q_logvar


@jaxtyped(typechecker=beartype)
def predict_with_diagnostics(
    params: DREEMS,
    arrays: DREEMSArrays,
    *,
    y_scale: float,
    q_y_scale: float,
    model: dict[str, Any],
) -> PredictionOutputWithDiagnostics:
    """Run inference and return the graph-level surface gate weight."""

    batch = {name: jax.device_put(jnp.asarray(value)) for name, value in arrays.items()}
    pred, edge_logvar, _context_pred, q_pred, q_logvar, _neural_weight, surface_weight = (
        apply_batch_with_diagnostics(
            tree_to_jax(params),
            batch,
        )
    )
    # Own the buffers: NumPy views of JAX arrays may be read-only. In-place
    # scaling below also preserves the model's floating-point dtype.
    edge_logvar = np.array(edge_logvar, copy=True)
    q_logvar = np.array(q_logvar, copy=True)
    minimum, maximum = model_logvar_bounds(model)
    if minimum is not None and maximum is not None:
        np.clip(edge_logvar, minimum, maximum, out=edge_logvar)
        np.clip(q_logvar, minimum, maximum, out=q_logvar)
    edge_logvar += 2.0 * np.log(np.maximum(y_scale, 1e-12))
    q_logvar += 2.0 * np.log(np.maximum(q_y_scale, 1e-12))
    return (
        np.asarray(pred) * y_scale,
        edge_logvar,
        np.asarray(q_pred) * q_y_scale,
        q_logvar,
        np.asarray(surface_weight),
    )
