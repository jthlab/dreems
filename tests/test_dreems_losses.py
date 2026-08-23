from __future__ import annotations

import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from dreems import losses


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
