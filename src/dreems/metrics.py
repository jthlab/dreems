from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class EdgeLogMetrics:
    """Scale-free and scale-sensitive metrics for centered edge log surfaces."""

    corr: float
    rmse: float
    mse: float
    scale_ratio: float
    slope: float

    def to_prefixed_dict(self, prefix: str = "edge_log") -> dict[str, float]:
        values = asdict(self)
        return {
            f"{prefix}_corr": values["corr"],
            f"{prefix}_rmse": values["rmse"],
            f"{prefix}_mse": values["mse"],
            f"{prefix}_scale_ratio": values["scale_ratio"],
            f"{prefix}_slope": values["slope"],
        }


def centered(values: np.ndarray) -> np.ndarray:
    """Return finite entries centered by their finite mean, preserving NaNs."""

    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if np.any(finite):
        out[finite] = values[finite] - float(np.mean(values[finite]))
    return out


def edge_log_metrics(truth: np.ndarray, estimate: np.ndarray) -> EdgeLogMetrics:
    """Compare two edge-log surfaces after removing their arbitrary global scale.

    FEEMS/EEMS-style migration surfaces are interpreted up to an additive
    constant on the log scale, so all metrics are computed after centering.
    Non-finite pairs are dropped. If there is no finite variation in either
    surface, correlation, scale ratio, and slope are reported as NaN.
    """

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
