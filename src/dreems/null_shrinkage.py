"""Graph-level shrinkage toward a constant migration surface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def centered(values: np.ndarray) -> np.ndarray:
    """Center one finite edge signal."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("edge values must be a nonempty finite vector")
    return array - float(np.mean(array))


def rms(values: np.ndarray) -> float:
    """Return root mean square after centering."""

    local = centered(values)
    return float(np.sqrt(np.mean(local * local)))


def pattern_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Correlate two centered edge patterns, returning zero for a flat pattern."""

    left_centered = centered(left)
    right_centered = centered(right)
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def mask_stability(anchor: np.ndarray, masked: list[np.ndarray]) -> float:
    """Median pattern correlation between an anchor fit and masked refits."""

    if not masked:
        raise ValueError("at least one masked estimate is required")
    anchor_array = np.asarray(anchor)
    if any(np.asarray(candidate).shape != anchor_array.shape for candidate in masked):
        raise ValueError("anchor and masked edge estimates must have the same shape")
    correlations = [pattern_correlation(anchor_array, candidate) for candidate in masked]
    return float(np.median(np.asarray(correlations, dtype=np.float64)))


def posterior_signal_ratio(
    edge_mean: np.ndarray, edge_sd: np.ndarray, *, floor: float = 1e-8
) -> float:
    """Compare predicted surface amplitude with posterior noise amplitude."""

    if floor <= 0.0 or not np.isfinite(floor):
        raise ValueError("floor must be positive and finite")
    mean = centered(edge_mean)
    sd = np.asarray(edge_sd, dtype=np.float64)
    if sd.shape != mean.shape or not np.all(np.isfinite(sd)) or np.any(sd < 0.0):
        raise ValueError("edge_sd must be finite, nonnegative, and aligned with edge_mean")
    signal = float(np.sqrt(np.mean(mean * mean)))
    noise = float(np.sqrt(np.mean(sd * sd)))
    return signal / max(noise, floor)


@dataclass(frozen=True)
class NullShrinkageRule:
    """Smoothly shrink an unstable, low-signal edge field toward zero.

    The score combines the log posterior signal-to-noise ratio with agreement
    between the full-data prediction and predictions after masking sampled
    nodes. A positive scalar preserves edge ordering and therefore cannot flip
    a recovered spatial pattern.
    """

    stability_weight: float
    threshold: float
    slope: float
    minimum_alpha: float = 0.0
    sd_scale: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.stability_weight <= 1.0:
            raise ValueError("stability_weight must lie between zero and one")
        if not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        if self.slope <= 0.0 or not np.isfinite(self.slope):
            raise ValueError("slope must be positive and finite")
        if not 0.0 <= self.minimum_alpha <= 1.0:
            raise ValueError("minimum_alpha must lie between zero and one")
        if self.sd_scale <= 0.0 or not np.isfinite(self.sd_scale):
            raise ValueError("sd_scale must be positive and finite")

    def score(
        self,
        edge_mean: np.ndarray,
        edge_sd: np.ndarray,
        *,
        stability: float | None = None,
    ) -> float:
        """Compute the scalar evidence score used by the gate."""

        if self.stability_weight > 0.0:
            if stability is None or not np.isfinite(stability):
                raise ValueError("this rule requires finite mask stability")
            local_stability = float(np.clip(stability, -1.0, 1.0))
        else:
            local_stability = 0.0
        ratio = posterior_signal_ratio(edge_mean, self.sd_scale * np.asarray(edge_sd))
        log_ratio = float(np.log(max(ratio, 1e-8)))
        return self.stability_weight * local_stability + (1.0 - self.stability_weight) * log_ratio

    def alpha(
        self,
        edge_mean: np.ndarray,
        edge_sd: np.ndarray,
        *,
        stability: float | None = None,
    ) -> float:
        """Return a shrinkage multiplier in ``[minimum_alpha, 1]``."""

        value = np.clip(
            self.slope * (self.score(edge_mean, edge_sd, stability=stability) - self.threshold),
            -60.0,
            60.0,
        )
        probability = float(1.0 / (1.0 + np.exp(-value)))
        return self.minimum_alpha + (1.0 - self.minimum_alpha) * probability

    def apply(
        self,
        edge_mean: np.ndarray,
        edge_sd: np.ndarray,
        *,
        stability: float | None = None,
    ) -> np.ndarray:
        """Center and shrink one edge field."""

        return centered(edge_mean) * self.alpha(edge_mean, edge_sd, stability=stability)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NullShrinkageRule:
        return cls(**payload)
