"""Score a pair of migration surfaces after centering their log rates.

Nonfinite pairs are excluded, and a constant truth has no defined correlation.
Every retained edge contributes equally.
"""

# ruff: noqa: F722, F821, UP037

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from beartype import beartype
from jaxtyping import Float, Shaped, jaxtyped


@dataclass(frozen=True)
class EdgeLogMetrics:
    """Correlation, error, and amplitude comparisons for one graph."""

    corr: float
    rmse: float
    mse: float
    scale_ratio: float
    slope: float

    @jaxtyped(typechecker=beartype)
    def to_prefixed_dict(self, prefix: str = "edge_log") -> dict[str, float]:
        """Prefix the saved field names without maintaining a second field list."""
        return {f"{prefix}_{name}": value for name, value in asdict(self).items()}


@jaxtyped(typechecker=beartype)
def centered(values: Shaped[np.ndarray, "*shape"]) -> Float[np.ndarray, "*shape"]:  # noqa: F722
    """Return finite entries centered by their finite mean, preserving NaNs."""

    if values.dtype.kind not in "biuf":
        raise ValueError("edge values must contain real numbers")
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if np.any(finite):
        out[finite] = values[finite] - float(np.mean(values[finite]))
    return out


@jaxtyped(typechecker=beartype)
def edge_log_metrics(
    truth: Shaped[np.ndarray, "*truth_shape"],
    estimate: Shaped[np.ndarray, "*estimate_shape"],
) -> EdgeLogMetrics:
    """Compare two edge-log surfaces after removing their arbitrary global scale.

    FEEMS/EEMS-style migration surfaces are interpreted up to an additive
    constant on the log scale, so all metrics are computed after centering.
    Nonfinite pairs are dropped. Correlation is NaN when either surface is
    constant. Scale ratio and slope are NaN only when the truth is constant;
    a constant estimate against a varying truth has ratio and slope zero.
    """

    if truth.dtype.kind not in "biuf" or estimate.dtype.kind not in "biuf":
        raise ValueError("truth and estimate must contain real numbers")
    truth = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if truth.shape != estimate.shape:
        raise ValueError(f"shape mismatch: truth {truth.shape} vs estimate {estimate.shape}")
    finite = np.isfinite(truth) & np.isfinite(estimate)
    if not np.any(finite):
        return EdgeLogMetrics(
            corr=float("nan"),
            rmse=float("nan"),
            mse=float("nan"),
            scale_ratio=float("nan"),
            slope=float("nan"),
        )

    truth_centered = truth[finite] - float(np.mean(truth[finite]))
    estimate_centered = estimate[finite] - float(np.mean(estimate[finite]))
    residual = truth_centered - estimate_centered
    mse = float(np.mean(residual * residual))
    rmse = float(math.sqrt(mse))
    truth_ss = float(np.dot(truth_centered, truth_centered))
    estimate_ss = float(np.dot(estimate_centered, estimate_centered))
    denom = math.sqrt(max(truth_ss * estimate_ss, 0.0))
    corr = float(np.dot(truth_centered, estimate_centered) / denom) if denom > 0.0 else float("nan")
    scale_ratio = math.sqrt(estimate_ss / truth_ss) if truth_ss > 0.0 else float("nan")
    slope = (
        float(np.dot(estimate_centered, truth_centered) / truth_ss)
        if truth_ss > 0.0
        else float("nan")
    )
    return EdgeLogMetrics(
        corr=corr,
        rmse=rmse,
        mse=mse,
        scale_ratio=scale_ratio,
        slope=slope,
    )
