from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import TypeCheckError

from dreems.training import losses


def test_masked_gaussian_nll_checks_array_shapes() -> None:
    with pytest.raises(TypeCheckError):
        losses.masked_gaussian_nll(
            jnp.ones((2, 3)),
            jnp.ones((2, 3)),
            jnp.ones((2, 3)),
            jnp.ones((2, 4)),
            min_logvar=-7.0,
            max_logvar=5.0,
        )


def test_masked_losses_weight_graphs_equally() -> None:
    prediction = jnp.asarray(
        [
            [2.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    target = jnp.zeros_like(prediction)
    mask = jnp.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    logvar = jnp.zeros_like(prediction)

    mse = losses.masked_mean_squared_error(prediction, target, mask)
    nll = losses.masked_gaussian_nll(
        prediction,
        target,
        logvar,
        mask,
        min_logvar=-7.0,
        max_logvar=5.0,
    )

    # The graph MSEs are 4 and 1. Pooling all five active edges would give 1.6.
    assert float(mse) == pytest.approx(2.5)
    assert float(nll) == pytest.approx(1.25)


def test_active_shape_losses_reward_the_target_pattern() -> None:
    target = jnp.asarray([[-1.0, -0.2, 0.4, 0.8]], dtype=jnp.float32)
    mask = jnp.ones_like(target)
    matching = target
    reversed_pattern = -target

    matching_correlation = losses.masked_centered_correlation_loss(
        matching,
        target,
        mask,
        target_floor=0.0,
    )
    reversed_correlation = losses.masked_centered_correlation_loss(
        reversed_pattern,
        target,
        mask,
        target_floor=0.0,
    )
    matching_orientation = losses.masked_edge_orientation_loss(
        matching,
        target,
        mask,
        target_floor=0.0,
        margin_fraction=0.5,
    )
    reversed_orientation = losses.masked_edge_orientation_loss(
        reversed_pattern,
        target,
        mask,
        target_floor=0.0,
        margin_fraction=0.5,
    )

    assert float(matching_correlation) < 1e-5
    assert float(reversed_correlation) > 1.9
    assert float(matching_orientation) < 1e-5
    assert float(reversed_orientation) > float(matching_orientation)


def test_initializer_regret_only_penalizes_degradation() -> None:
    target = jnp.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=jnp.float32)
    initializer = jnp.asarray([[0.1, 0.9], [0.5, 0.5]], dtype=jnp.float32)
    prediction = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    mask = jnp.ones_like(target)

    regret = losses.masked_initializer_regret_loss(
        prediction,
        target,
        initializer,
        jnp.ones_like(target),
        mask,
    )

    # The first graph improves on the initializer and contributes zero. The
    # second changes MSE from 0.25 to 1.0, so the graph-balanced mean is 0.375.
    assert float(regret) == pytest.approx(0.375)

    invalid_regret = losses.masked_initializer_regret_loss(
        prediction,
        target,
        jnp.full_like(target, jnp.nan),
        jnp.zeros_like(target),
        mask,
    )
    assert float(invalid_regret) == 0.0


def test_counterfactual_consistency_compares_each_mask_to_its_anchor() -> None:
    prediction = jnp.asarray(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [5.0, 5.0],
        ],
        dtype=jnp.float32,
    )
    mask = jnp.ones_like(prediction)
    group = jnp.asarray([7, 7, 7, -1], dtype=jnp.int32)
    anchor = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

    value = losses.counterfactual_anchor_consistency_loss(
        prediction,
        mask,
        group,
        anchor,
    )

    # The two anchor pairs have MSE 0 and 1. The ungrouped row is ignored.
    assert float(value) == pytest.approx(0.5)

    gradient = jax.grad(losses.counterfactual_anchor_consistency_loss)(
        prediction,
        mask,
        group,
        anchor,
    )
    np.testing.assert_array_equal(gradient[0], np.zeros(2, dtype=np.float32))
    assert np.any(np.asarray(gradient[2]) != 0.0)


def test_surface_gate_classifies_constant_and_structured_targets() -> None:
    target = jnp.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [-1.0, -0.5, 0.5, 1.0],
        ],
        dtype=jnp.float32,
    )
    mask = jnp.ones_like(target)

    correct = losses.surface_gate_binary_cross_entropy(
        jnp.asarray([0.01, 0.99]),
        target,
        mask,
        target_floor=0.05,
    )
    reversed_gate = losses.surface_gate_binary_cross_entropy(
        jnp.asarray([0.99, 0.01]),
        target,
        mask,
        target_floor=0.05,
    )

    assert float(correct) < 0.02
    assert float(reversed_gate) > 4.0
