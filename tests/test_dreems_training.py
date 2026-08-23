from __future__ import annotations

import zipfile

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from dreems import checkpoints, model, train


def toy_training_arrays() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    batch = 3
    nodes = 4
    edges = np.asarray([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=np.int32)
    edge_count = edges.shape[0]
    return {
        "node_x": rng.normal(size=(batch, nodes, 6)).astype(np.float32),
        "edge_x": rng.normal(size=(batch, edge_count, 5)).astype(np.float32),
        "edges": np.broadcast_to(edges, (batch, edge_count, 2)).copy(),
        "y": rng.normal(size=(batch, edge_count)).astype(np.float32),
        "q": rng.normal(size=(batch, nodes)).astype(np.float32),
        "node_mask": np.ones((batch, nodes), dtype=np.float32),
        "edge_mask": np.ones((batch, edge_count), dtype=np.float32),
    }


def small_config(*, steps: int = 2, patience: int | None = None) -> train.DREEMSTrainConfig:
    return train.DREEMSTrainConfig(
        architecture=train.DREEMSArchitectureConfig(
            hidden_channels=8,
            layers=1,
            context_layers=1,
        ),
        optimizer=train.DREEMSOptimizerConfig(
            seed=11,
            steps=steps,
            batch_size=2,
            learning_rate=1e-3,
            log_every=1,
            early_stopping_patience=patience,
            early_stopping_min_delta=1e9 if patience is not None else 0.0,
        ),
    )


def test_train_save_and_load_production_checkpoint(tmp_path) -> None:
    arrays = toy_training_arrays()
    config = small_config()

    result = train.train_from_arrays(arrays, config=config, validation_arrays=arrays)

    assert len(result.losses) == 2
    assert np.all(np.isfinite(result.losses))
    assert result.best_step is not None
    assert isinstance(result.params, model.DREEMS)
    assert set(result.history[-1]) >= {
        "validation_loss",
        "validation_edge_nll",
        "validation_q_nll",
        "validation_context_mse",
        "validation_slope",
        "validation_correlation",
        "validation_orientation",
    }

    normalization = {
        "node_mean": [0.0] * arrays["node_x"].shape[-1],
        "node_std": [1.0] * arrays["node_x"].shape[-1],
        "edge_mean": [0.0] * arrays["edge_x"].shape[-1],
        "edge_std": [1.0] * arrays["edge_x"].shape[-1],
        "y_scale": 1.0,
        "q_y_scale": 1.0,
    }
    path = tmp_path / "checkpoint.dreems.zip"
    train.save_checkpoint(
        path,
        params=result.params,
        normalization=normalization,
        config=config,
        train_losses=result.losses,
        train_arrays=arrays,
        training_run=result.summary(),
    )

    assert zipfile.is_zipfile(path)
    loaded = checkpoints.load_checkpoint(path)
    assert loaded.model == train.model_metadata(config, arrays)
    assert loaded.model["schema_version"] == 2
    assert loaded.model["architecture"]["kind"] == "diffusion_band_gnn_v1"
    assert loaded.model["features"] == {
        "kind": "diffusion_bands_v1",
        "covariance_components": 20,
    }
    assert loaded.model["edge_conditioner"]["method"] == "all_pair_diagonal_adjoint"
    assert loaded.payload["training_config"]["optimizer"]["learning_rate"] == 1e-3


def test_train_from_arrays_stops_on_validation_plateau() -> None:
    arrays = toy_training_arrays()

    result = train.train_from_arrays(
        arrays,
        config=small_config(steps=5, patience=1),
        validation_arrays=arrays,
    )

    assert result.stopped_early
    assert result.final_step == 2
    assert result.best_step == 1
    assert result.restored_best_validation


def test_validation_metrics_can_batch_examples() -> None:
    arrays = toy_training_arrays()
    arrays["edge_mask"] = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    arrays["node_mask"] = np.asarray(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    config = small_config(steps=1)
    result = train.train_from_arrays(arrays, config=config)

    together = train.validation_metrics(result.params, arrays, config)
    separately = train.validation_metrics(result.params, arrays, config, batch_size=1)
    split = train.validation_metrics(result.params, arrays, config, batch_size=2)

    assert together.keys() == separately.keys()
    for key in together:
        np.testing.assert_allclose(separately[key], together[key], rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(split[key], together[key], rtol=2e-5, atol=2e-5)


def test_model_is_equivariant_to_graph_representation() -> None:
    rng = np.random.default_rng(19)
    edges = np.asarray(
        [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4], [1, 4]],
        dtype=np.int32,
    )
    node_x = rng.normal(size=(5, 7)).astype(np.float32)
    edge_x = rng.normal(size=(edges.shape[0], 9)).astype(np.float32)
    node_mask = np.ones(5, dtype=np.float32)
    edge_mask = np.ones(edges.shape[0], dtype=np.float32)
    edge_target = rng.normal(size=edges.shape[0]).astype(np.float32)
    node_target = rng.normal(size=5).astype(np.float32)
    network = model.DREEMS(
        key=jax.random.PRNGKey(23),
        node_dim=node_x.shape[1],
        edge_dim=edge_x.shape[1],
        hidden_channels=8,
        layers=2,
        context_layers=1,
    )

    reference = network.apply_one(
        jnp.asarray(node_x),
        jnp.asarray(edge_x),
        jnp.asarray(edges),
        jnp.asarray(node_mask),
        jnp.asarray(edge_mask),
    )

    old_to_new = np.asarray([3, 0, 4, 1, 2], dtype=np.int32)
    edge_permutation = np.asarray([5, 2, 0, 4, 1, 3], dtype=np.int32)
    relabeled_node_x = np.empty_like(node_x)
    relabeled_node_x[old_to_new] = node_x
    relabeled_node_target = np.empty_like(node_target)
    relabeled_node_target[old_to_new] = node_target
    represented_edges = old_to_new[edges[edge_permutation]][:, ::-1]
    represented_edge_x = edge_x[edge_permutation]
    represented = network.apply_one(
        jnp.asarray(relabeled_node_x),
        jnp.asarray(represented_edge_x),
        jnp.asarray(represented_edges),
        jnp.asarray(node_mask),
        jnp.asarray(edge_mask),
    )

    for reference_values, represented_values in zip(reference[:3], represented[:3], strict=True):
        aligned = jnp.empty_like(represented_values).at[edge_permutation].set(represented_values)
        np.testing.assert_allclose(aligned, reference_values, rtol=2e-6, atol=2e-6)
    for reference_values, represented_values in zip(reference[3:], represented[3:], strict=True):
        np.testing.assert_allclose(
            represented_values[old_to_new],
            reference_values,
            rtol=2e-6,
            atol=2e-6,
        )

    def batch(
        batch_node_x: np.ndarray,
        batch_edge_x: np.ndarray,
        batch_edges: np.ndarray,
        batch_edge_target: np.ndarray,
        batch_node_target: np.ndarray,
    ) -> dict[str, jax.Array]:
        return {
            "node_x": jnp.asarray(batch_node_x[None, ...]),
            "edge_x": jnp.asarray(batch_edge_x[None, ...]),
            "edges": jnp.asarray(batch_edges[None, ...]),
            "y": jnp.asarray(batch_edge_target[None, ...]),
            "q": jnp.asarray(batch_node_target[None, ...]),
            "node_mask": jnp.asarray(node_mask[None, ...]),
            "edge_mask": jnp.asarray(edge_mask[None, ...]),
        }

    reference_batch = batch(node_x, edge_x, edges, edge_target, node_target)
    represented_batch = batch(
        relabeled_node_x,
        represented_edge_x,
        represented_edges,
        edge_target[edge_permutation],
        relabeled_node_target,
    )
    config = small_config(steps=1)

    def objective(candidate: model.DREEMS, arrays: dict[str, jax.Array]) -> jax.Array:
        return train.loss_terms(candidate, arrays, config)["loss"]

    reference_loss, reference_gradient = eqx.filter_value_and_grad(objective)(
        network,
        reference_batch,
    )
    represented_loss, represented_gradient = eqx.filter_value_and_grad(objective)(
        network,
        represented_batch,
    )
    np.testing.assert_allclose(represented_loss, reference_loss, rtol=2e-6, atol=2e-6)
    for represented_leaf, reference_leaf in zip(
        jax.tree_util.tree_leaves(represented_gradient),
        jax.tree_util.tree_leaves(reference_gradient),
        strict=True,
    ):
        np.testing.assert_allclose(
            represented_leaf,
            reference_leaf,
            rtol=2e-5,
            atol=2e-5,
        )
