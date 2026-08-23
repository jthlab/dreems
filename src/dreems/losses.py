"""Loss terms used by the production DREEMS model."""
# ruff: noqa: F722, F821

from typing import TypeAlias

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

LossArray: TypeAlias = Float[Array, "batch items"]
Scalar: TypeAlias = Float[Array, ""]


def straight_through_clip(
    value: Float[Array, "*dims"],
    min_value: float,
    max_value: float,
) -> Float[Array, "*dims"]:
    clipped = jnp.clip(value, min_value, max_value)
    return value + jax.lax.stop_gradient(clipped - value)


def _mean_over_examples(
    values: Float[Array, "batch"],
    active: Float[Array, "batch"],
) -> Scalar:
    """Average graph-level values, assigning zero loss to inactive graphs."""

    return jnp.mean(values * active)


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
    return _mean_over_examples(per_example, active)


def masked_gaussian_nll(
    prediction: LossArray,
    target: LossArray,
    logvar: LossArray,
    mask: LossArray,
    *,
    min_logvar: float,
    max_logvar: float,
) -> Scalar:
    clipped_logvar = straight_through_clip(logvar, min_logvar, max_logvar)
    squared_error = ((prediction - target) ** 2) * mask
    terms = 0.5 * (squared_error * jnp.exp(-clipped_logvar) + clipped_logvar * mask)
    count = jnp.sum(mask, axis=1)
    active = (count > 0.0).astype(prediction.dtype)
    per_example = jnp.sum(terms, axis=1) / jnp.maximum(count, 1.0)
    return _mean_over_examples(per_example, active)


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
    count = jnp.maximum(jnp.sum(mask, axis=1), 1.0)
    prediction_mean = jnp.sum(prediction * mask, axis=1) / count
    target_mean = jnp.sum(target * mask, axis=1) / count
    prediction_centered = (prediction - prediction_mean[:, None]) * mask
    target_centered = (target - target_mean[:, None]) * mask
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
    return _mean_over_examples(penalty, active)


def masked_centered_correlation_loss(
    prediction: LossArray,
    target: LossArray,
    mask: LossArray,
    *,
    target_floor: float,
    eps: float = 1e-6,
) -> Scalar:
    count = jnp.maximum(jnp.sum(mask, axis=1), 1.0)
    prediction_mean = jnp.sum(prediction * mask, axis=1) / count
    target_mean = jnp.sum(target * mask, axis=1) / count
    prediction_centered = (prediction - prediction_mean[:, None]) * mask
    target_centered = (target - target_mean[:, None]) * mask
    prediction_var = jnp.sum(prediction_centered * prediction_centered, axis=1) / count
    target_var = jnp.sum(target_centered * target_centered, axis=1) / count
    covariance = jnp.sum(prediction_centered * target_centered, axis=1) / count
    target_rms = jnp.sqrt(target_var + eps)
    active = (target_rms > target_floor).astype(prediction.dtype)
    denominator = jnp.sqrt(prediction_var * jax.lax.stop_gradient(target_var) + eps)
    correlation = covariance / denominator
    return _mean_over_examples(1.0 - correlation, active)


def masked_edge_orientation_loss(
    prediction: LossArray,
    target: LossArray,
    mask: LossArray,
    *,
    target_floor: float,
    margin_fraction: float,
    eps: float = 1e-6,
) -> Scalar:
    count = jnp.maximum(jnp.sum(mask, axis=1), 1.0)
    prediction_mean = jnp.sum(prediction * mask, axis=1) / count
    target_mean = jnp.sum(target * mask, axis=1) / count
    prediction_centered = (prediction - prediction_mean[:, None]) * mask
    target_centered = (target - target_mean[:, None]) * mask
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
    return _mean_over_examples(penalty, active)
