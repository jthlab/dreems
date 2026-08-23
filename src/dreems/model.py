"""The released DREEMS graph neural network."""
# ruff: noqa: F722, F821

from operator import itemgetter
from typing import Any, TypeAlias, TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PRNGKeyArray

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


class DREEMSBatch(TypedDict):
    node_x: Float[Array, "batch nodes node_dim"]
    edge_x: Float[Array, "batch edges edge_dim"]
    edges: Int[Array, "batch edges 2"]
    node_mask: Float[Array, "batch nodes"]
    edge_mask: Float[Array, "batch edges"]


class DREEMSArrays(TypedDict):
    node_x: Float[np.ndarray, "batch nodes node_dim"]
    edge_x: Float[np.ndarray, "batch edges edge_dim"]
    edges: Int[np.ndarray, "batch edges 2"]
    node_mask: Float[np.ndarray, "batch nodes"]
    edge_mask: Float[np.ndarray, "batch edges"]


BatchOutput: TypeAlias = tuple[
    Float[Array, "batch edges"],
    Float[Array, "batch edges"],
    Float[Array, "batch edges"],
    Float[Array, "batch nodes"],
    Float[Array, "batch nodes"],
]
PredictionOutput: TypeAlias = tuple[
    Float[np.ndarray, "batch edges"],
    Float[np.ndarray, "batch edges"],
    Float[np.ndarray, "batch nodes"],
    Float[np.ndarray, "batch nodes"],
]

DREEMS_BATCH_FIELDS = itemgetter("node_x", "edge_x", "edges", "node_mask", "edge_mask")


def tree_to_jax(tree: Any) -> Any:
    return jax.tree.map(
        lambda value: jnp.asarray(value) if isinstance(value, np.ndarray) else value,
        tree,
    )


class DREEMS(eqx.Module):
    """Fixed diffusion-band architecture used by the released checkpoint.

    The network has a local edge-aware residual GNN, a second three-layer
    graph-context branch on the same graph, global edge-to-node attention, and
    edge and node Gaussian heads. Width and depth remain constructor arguments
    so small tests can use the same architecture at reduced size.
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

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        node_dim: int,
        edge_dim: int,
        hidden_channels: int,
        layers: int,
        context_layers: int,
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

    @staticmethod
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
        sender = jnp.concatenate([edges[:, 0], edges[:, 1]])
        receiver = jnp.concatenate([edges[:, 1], edges[:, 0]])
        directed_mask = jnp.concatenate([edge_mask, edge_mask])
        directed_edge_x = jnp.concatenate([edge_x, edge_x], axis=0)
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
            degree = jax.ops.segment_sum(directed_mask, receiver, num_segments=z.shape[0])
            neighbor_mean = summed / jnp.maximum(degree[:, None], 1.0)
            update = (
                jax.nn.gelu(jax.vmap(self_layer)(z) + jax.vmap(neigh_layer)(neighbor_mean))
                * node_mask[:, None]
            )
            z = (z + update) * node_mask[:, None]
        return z

    def apply_one(
        self,
        node_x: NodeX,
        edge_x: EdgeX,
        edges: Edges,
        node_mask: NodeMask,
        edge_mask: EdgeMask,
    ) -> OneOutput:
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

        pred = jnp.squeeze(jax.vmap(self.edge_out)(hidden), axis=1)
        pred -= jnp.sum(pred * edge_mask) / jnp.maximum(jnp.sum(edge_mask), 1.0)
        edge_logvar = jnp.squeeze(jax.vmap(self.edge_logvar_out)(hidden), axis=1)
        return pred * edge_mask, edge_logvar * edge_mask, context_pred, q_pred, q_logvar * node_mask


def apply_batch(model: DREEMS, batch: DREEMSBatch) -> BatchOutput:
    return jax.vmap(model.apply_one)(*DREEMS_BATCH_FIELDS(batch))


def model_logvar_bounds(metadata: dict[str, Any]) -> tuple[float | None, float | None]:
    posterior = metadata.get("posterior", {})
    if not isinstance(posterior, dict) or posterior.get("kind") != "factorized_normal":
        return None, None
    minimum = posterior.get("min_logvar")
    maximum = posterior.get("max_logvar")
    if minimum is None or maximum is None:
        return None, None
    return float(minimum), float(maximum)


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
    if architecture.get("kind") != "diffusion_band_gnn_v1":
        raise ValueError("checkpoint does not use the released DREEMS architecture")
    if feature_metadata.get("kind") not in {
        "diffusion_bands_v1",
        "topology_diffusion_bands_v1",
        "topology_diffusion_bands_v2",
    }:
        raise ValueError("checkpoint does not use the released diffusion-band features")
    if conditioner.get("method") != "all_pair_diagonal_adjoint":
        raise ValueError("checkpoint does not use the released all-pair conditioner")


def predict(
    params: DREEMS,
    arrays: DREEMSArrays,
    *,
    y_scale: float,
    q_y_scale: float,
    model: dict[str, Any],
) -> PredictionOutput:
    batch = {name: jax.device_put(jnp.asarray(value)) for name, value in arrays.items()}
    pred, edge_logvar, _context_pred, q_pred, q_logvar = apply_batch(
        tree_to_jax(params),
        batch,
    )
    edge_logvar = np.asarray(edge_logvar)
    q_logvar = np.asarray(q_logvar)
    minimum, maximum = model_logvar_bounds(model)
    if minimum is not None and maximum is not None:
        edge_logvar = np.clip(edge_logvar, minimum, maximum)
        q_logvar = np.clip(q_logvar, minimum, maximum)
    edge_logvar += 2.0 * np.log(np.maximum(y_scale, 1e-12))
    q_logvar += 2.0 * np.log(np.maximum(q_y_scale, 1e-12))
    return np.asarray(pred) * y_scale, edge_logvar, np.asarray(q_pred) * q_y_scale, q_logvar
