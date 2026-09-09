"""Supervised losses for simulated log migration and log coalescence rates.

Each term averages within a graph before averaging across the batch, so large
graphs do not dominate. Unless stated otherwise, inactive graphs contribute
zero but remain in the batch denominator. Masks must be finite, with zero for
padding and one for observations; padded prediction and target values must also
be finite. These are edge-recovery losses, not a covariance likelihood.
"""
# ruff: noqa: F722, F821

from typing import TypeAlias

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, Int, jaxtyped

LossArray: TypeAlias = Float[Array, "batch items"]
Scalar: TypeAlias = Float[Array, ""]


@jaxtyped(typechecker=beartype)
def straight_through_clip(
    value: Float[Array, "*dims"],
    min_value: float,
    max_value: float,
) -> Float[Array, "*dims"]:
    """Bound the forward value but pass the gradient through unchanged.

    This lets the variance head recover after it leaves the allowed interval;
    ordinary clipping would give it zero gradient outside that interval.
    """

    clipped = jnp.clip(value, min_value, max_value)
    return value + jax.lax.stop_gradient(clipped - value)


@jaxtyped(typechecker=beartype)
def masked_mean_squared_error(
    prediction: LossArray,
    target: LossArray,
    mask: LossArray,
) -> Scalar:
    """Average squared error within each graph, then across graphs."""

    count = jnp.sum(mask, axis=1)
    active = (count > 0.0).astype(prediction.dtype)
    per_example = jnp.sum(((prediction - target) ** 2) * mask, axis=1)
    per_example = per_example / jnp.maximum(count, 1.0)
    return jnp.mean(per_example * active)


@jaxtyped(typechecker=beartype)
def masked_initializer_regret_loss(
    prediction: LossArray,
    target: LossArray,
    initializer: LossArray,
    initializer_valid: LossArray,
    mask: LossArray,
    *,
    margin: float = 0.0,
) -> Scalar:
    """Penalize a graph only when it is worse than its valid initializer."""

    valid_mask = ((mask > 0.0) & (initializer_valid > 0.0) & jnp.isfinite(initializer)).astype(
        prediction.dtype
    )
    safe_initializer = jnp.where(valid_mask > 0.0, initializer, target)
    count = jnp.sum(valid_mask, axis=1)
    active = (count > 0.0).astype(prediction.dtype)
    prediction_mse = jnp.sum(((prediction - target) ** 2) * valid_mask, axis=1)
    initializer_mse = jnp.sum(((safe_initializer - target) ** 2) * valid_mask, axis=1)
    prediction_mse /= jnp.maximum(count, 1.0)
    initializer_mse /= jnp.maximum(count, 1.0)
    regret = jnp.maximum(prediction_mse - jax.lax.stop_gradient(initializer_mse) - margin, 0.0)
    return jnp.mean(regret * active)


@jaxtyped(typechecker=beartype)
def counterfactual_anchor_consistency_loss(
    prediction: LossArray,
    mask: LossArray,
    group: Int[Array, "batch"],
    anchor: Float[Array, "batch"],
) -> Scalar:
    """Match subsamples to their richer anchor from the same simulation.

    Only pairs with a shared nonnegative group ID and exactly one anchor count.
    Gradients flow to the subsample prediction, not the anchor. Unlike the other
    losses, this term averages over valid pairs rather than over all graphs.
    """

    batch_size = prediction.shape[0]
    index = jnp.arange(batch_size)
    same_group = group[:, None] == group[None, :]
    valid_group = (group[:, None] >= 0) & (group[None, :] >= 0)
    left_is_anchor = anchor[:, None] > 0.0
    right_is_anchor = anchor[None, :] > 0.0
    anchor_pair = left_is_anchor ^ right_is_anchor
    upper_triangle = index[:, None] < index[None, :]
    pair_active = same_group & valid_group & anchor_pair & upper_triangle

    shared_mask = mask[:, None, :] * mask[None, :, :]
    count = jnp.sum(shared_mask, axis=2)
    left = prediction[:, None, :]
    right = prediction[None, :, :]
    teacher = jnp.where(left_is_anchor[:, :, None], left, right)
    student = jnp.where(left_is_anchor[:, :, None], right, left)
    squared_difference = (student - jax.lax.stop_gradient(teacher)) ** 2
    pair_mse = jnp.sum(squared_difference * shared_mask, axis=2) / jnp.maximum(count, 1.0)
    pair_active_float = pair_active.astype(prediction.dtype) * (count > 0.0)
    return jnp.sum(pair_mse * pair_active_float) / jnp.maximum(
        jnp.sum(pair_active_float),
        1.0,
    )


@jaxtyped(typechecker=beartype)
def masked_gaussian_nll(
    prediction: LossArray,
    target: LossArray,
    logvar: LossArray,
    mask: LossArray,
    *,
    min_logvar: float,
    max_logvar: float,
) -> Scalar:
    """Gaussian NLL for the target log rate, omitting the constant log(2*pi)/2.

    The predicted log variance is bounded in the forward pass with a
    straight-through gradient. This loss can be negative; it is not MSE.
    """

    clipped_logvar = straight_through_clip(logvar, min_logvar, max_logvar)
    squared_error = ((prediction - target) ** 2) * mask
    terms = 0.5 * (squared_error * jnp.exp(-clipped_logvar) + clipped_logvar * mask)
    count = jnp.sum(mask, axis=1)
    active = (count > 0.0).astype(prediction.dtype)
    per_example = jnp.sum(terms, axis=1) / jnp.maximum(count, 1.0)
    return jnp.mean(per_example * active)


@jaxtyped(typechecker=beartype)
def surface_gate_binary_cross_entropy(
    surface_weight: Float[Array, "batch"],
    target: LossArray,
    mask: LossArray,
    *,
    target_floor: float,
    eps: float = 1e-6,
) -> Scalar:
    """Classify whether a graph has nonconstant relative migration."""

    count = jnp.sum(mask, axis=1)
    active = (count > 0.0).astype(target.dtype)
    target_rms = jnp.sqrt(jnp.sum(target * target * mask, axis=1) / jnp.maximum(count, 1.0))
    structured = (target_rms > target_floor).astype(target.dtype)
    probability = jnp.clip(surface_weight, eps, 1.0 - eps)
    cross_entropy = -structured * jnp.log(probability)
    cross_entropy -= (1.0 - structured) * jnp.log1p(-probability)
    return jnp.mean(cross_entropy * active)


@jaxtyped(typechecker=beartype)
def _centered_signals(
    prediction: LossArray,
    target: LossArray,
    mask: LossArray,
) -> tuple[LossArray, LossArray, Float[Array, "batch"]]:
    """Center each graph using its valid edges; leave padded entries at zero."""

    count = jnp.maximum(jnp.sum(mask, axis=1), 1.0)
    prediction_mean = jnp.sum(prediction * mask, axis=1) / count
    target_mean = jnp.sum(target * mask, axis=1) / count
    return (
        (prediction - prediction_mean[:, None]) * mask,
        (target - target_mean[:, None]) * mask,
        count,
    )


@jaxtyped(typechecker=beartype)
def masked_regression_slope_loss(
    prediction: LossArray,
    target: LossArray,
    mask: LossArray,
    *,
    target_slope: float,
    target_floor: float,
    huber_delta: float = 0.0,
    eps: float = 1e-6,
) -> Scalar:
    """Penalize the regression slope of prediction on truth, skipping flat truth.

    A positive ``huber_delta`` uses twice the usual Huber loss so the quadratic
    part agrees with squared error. Keep this option static when using JIT.
    """

    prediction_centered, target_centered, count = _centered_signals(prediction, target, mask)
    target_var = jnp.sum(target_centered * target_centered, axis=1) / count
    covariance = jnp.sum(prediction_centered * target_centered, axis=1) / count
    target_rms = jnp.sqrt(target_var + eps)
    active = (target_rms > target_floor).astype(prediction.dtype)
    slope = covariance / (jax.lax.stop_gradient(target_var) + eps)
    error = slope - target_slope
    if huber_delta > 0.0:
        delta = huber_delta
        abs_error = jnp.abs(error)
        penalty = jnp.where(
            abs_error <= delta,
            error * error,
            2.0 * delta * abs_error - delta * delta,
        )
    else:
        penalty = error * error
    return jnp.mean(penalty * active)


@jaxtyped(typechecker=beartype)
def masked_centered_correlation_loss(
    prediction: LossArray,
    target: LossArray,
    mask: LossArray,
    *,
    target_floor: float,
    eps: float = 1e-6,
) -> Scalar:
    """Average ``1 - r`` with a stabilized denominator, skipping flat truth.

    Logged as ``correlation`` (or ``validation_correlation``) by the trainer.
    It is a loss, not Pearson r: smaller is better. Inactive graphs contribute
    zero, so subtracting this batch loss from one is not a mean Pearson r.
    """

    prediction_centered, target_centered, count = _centered_signals(prediction, target, mask)
    prediction_var = jnp.sum(prediction_centered * prediction_centered, axis=1) / count
    target_var = jnp.sum(target_centered * target_centered, axis=1) / count
    covariance = jnp.sum(prediction_centered * target_centered, axis=1) / count
    target_rms = jnp.sqrt(target_var + eps)
    active = (target_rms > target_floor).astype(prediction.dtype)
    denominator = jnp.sqrt(prediction_var * jax.lax.stop_gradient(target_var) + eps)
    correlation = covariance / denominator
    return jnp.mean((1.0 - correlation) * active)


@jaxtyped(typechecker=beartype)
def masked_edge_orientation_loss(
    prediction: LossArray,
    target: LossArray,
    mask: LossArray,
    *,
    target_floor: float,
    margin_fraction: float,
    eps: float = 1e-6,
) -> Scalar:
    """Require a high-versus-low contrast with the same sign as the truth.

    Truth determines the two tails and their weights. The predicted contrast
    must reach ``margin_fraction`` of the true contrast to avoid a penalty.
    """

    prediction_centered, target_centered, count = _centered_signals(prediction, target, mask)
    target_rms = jnp.sqrt(jnp.sum(target_centered * target_centered, axis=1) / count + eps)
    active = (target_rms > target_floor).astype(prediction.dtype)

    positive_weight = jax.lax.stop_gradient(jnp.maximum(target_centered, 0.0) * mask)
    negative_weight = jax.lax.stop_gradient(jnp.maximum(-target_centered, 0.0) * mask)
    positive_sum = jnp.sum(positive_weight, axis=1)
    negative_sum = jnp.sum(negative_weight, axis=1)
    has_both_tails = ((positive_sum > eps) & (negative_sum > eps)).astype(prediction.dtype)

    target_high = jnp.sum(target_centered * positive_weight, axis=1) / (positive_sum + eps)
    target_low = jnp.sum(target_centered * negative_weight, axis=1) / (negative_sum + eps)
    prediction_high = jnp.sum(prediction_centered * positive_weight, axis=1) / (positive_sum + eps)
    prediction_low = jnp.sum(prediction_centered * negative_weight, axis=1) / (negative_sum + eps)
    target_contrast = jax.lax.stop_gradient(jnp.maximum(target_high - target_low, eps))
    prediction_contrast = prediction_high - prediction_low
    contrast_ratio = prediction_contrast / target_contrast
    penalty = jnp.square(jnp.maximum(margin_fraction - contrast_ratio, 0.0))
    active = active * has_both_tails
    return jnp.mean(penalty * active)
