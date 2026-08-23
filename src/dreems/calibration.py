"""Post-hoc calibration for edge posterior standard deviations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

CalibrationMethod = Literal["raw", "global", "density"]
DEFAULT_DENSITY_BREAKS = (0.10, 0.25)


def _positive_finite(value: float, *, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


@dataclass(frozen=True)
class SDCalibration:
    """Multiplicative calibration for edge posterior standard deviations.

    Scales are fit after reducing squared standardized errors within each
    graph. This prevents large graphs from dominating calibration simply
    because they contain more edges.
    """

    method: CalibrationMethod = "raw"
    global_scale: float = 1.0
    density_breaks: tuple[float, ...] = DEFAULT_DENSITY_BREAKS
    density_scales: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.method not in ("raw", "global", "density"):
            raise ValueError(f"unknown SD calibration method: {self.method!r}")
        global_scale = _positive_finite(self.global_scale, name="global_scale")
        breaks = tuple(float(value) for value in self.density_breaks)
        if any(not np.isfinite(value) or value <= 0.0 or value >= 1.0 for value in breaks):
            raise ValueError("density_breaks must lie strictly between zero and one")
        if any(right <= left for left, right in zip(breaks, breaks[1:], strict=False)):
            raise ValueError("density_breaks must be strictly increasing")
        scales = tuple(
            _positive_finite(value, name="density scale") for value in self.density_scales
        )
        if scales and len(scales) != len(breaks) + 1:
            raise ValueError("density_scales must have one value per density interval")
        if self.method == "density" and not scales:
            raise ValueError("density calibration requires density_scales")
        object.__setattr__(self, "global_scale", global_scale)
        object.__setattr__(self, "density_breaks", breaks)
        object.__setattr__(self, "density_scales", scales)

    @classmethod
    def from_squared_errors(
        cls,
        errors: np.ndarray,
        raw_sd: np.ndarray,
        *,
        floor: float = 1e-12,
    ) -> SDCalibration:
        """Fit one global scale from a flat collection of edge errors."""

        graph_score = _mean_squared_standardized_error(errors, raw_sd, floor=floor)
        if not np.isfinite(graph_score):
            return cls()
        return cls(method="global", global_scale=max(float(np.sqrt(graph_score)), floor))

    @classmethod
    def from_graphs(
        cls,
        errors: Sequence[np.ndarray],
        raw_sd: Sequence[np.ndarray],
        sampled_node_fraction: Sequence[float],
        *,
        density_breaks: Sequence[float] = DEFAULT_DENSITY_BREAKS,
        density_shrinkage: float = 25.0,
        floor: float = 1e-12,
    ) -> SDCalibration:
        """Fit graph-balanced global and sampling-density scales.

        ``density_shrinkage`` is the number of global pseudo-graphs mixed into
        every density-bin estimate. It stabilizes small bins while allowing
        well-represented sparse and dense regimes to calibrate separately.
        """

        if len(errors) != len(raw_sd) or len(errors) != len(sampled_node_fraction):
            raise ValueError("errors, raw_sd, and sampled_node_fraction must align by graph")
        if density_shrinkage < 0.0 or not np.isfinite(density_shrinkage):
            raise ValueError("density_shrinkage must be finite and nonnegative")
        breaks = tuple(float(value) for value in density_breaks)
        graph_scores = np.asarray(
            [
                _mean_squared_standardized_error(error, sd, floor=floor)
                for error, sd in zip(errors, raw_sd, strict=True)
            ],
            dtype=np.float64,
        )
        density = np.asarray(sampled_node_fraction, dtype=np.float64)
        valid = (
            np.isfinite(graph_scores) & np.isfinite(density) & (density >= 0.0) & (density <= 1.0)
        )
        if not np.any(valid):
            return cls()
        graph_scores = graph_scores[valid]
        density = density[valid]
        global_score = float(np.mean(graph_scores))
        global_scale = max(float(np.sqrt(global_score)), floor)

        bin_index = np.searchsorted(np.asarray(breaks, dtype=np.float64), density, side="left")
        scales: list[float] = []
        for index in range(len(breaks) + 1):
            local = graph_scores[bin_index == index]
            if local.size == 0:
                scales.append(global_scale)
                continue
            shrunk_score = (float(np.sum(local)) + float(density_shrinkage) * global_score) / (
                float(local.size) + float(density_shrinkage)
            )
            scales.append(max(float(np.sqrt(shrunk_score)), floor))
        return cls(
            method="density",
            global_scale=global_scale,
            density_breaks=breaks,
            density_scales=tuple(scales),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> SDCalibration:
        """Load calibration metadata; missing metadata means raw SDs."""

        if payload is None:
            return cls()
        return cls(
            method=str(payload.get("method", "raw")),  # type: ignore[arg-type]
            global_scale=float(payload.get("global_scale", 1.0)),
            density_breaks=tuple(payload.get("density_breaks", DEFAULT_DENSITY_BREAKS)),
            density_scales=tuple(payload.get("density_scales", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method": self.method,
            "global_scale": self.global_scale,
            "density_breaks": list(self.density_breaks),
            "density_scales": list(self.density_scales),
        }

    def scale_for_density(self, sampled_node_fraction: float | None) -> float:
        if self.method == "raw":
            return 1.0
        if self.method == "global" or sampled_node_fraction is None or not self.density_scales:
            return self.global_scale
        density = float(sampled_node_fraction)
        if not np.isfinite(density) or density < 0.0 or density > 1.0:
            raise ValueError("sampled_node_fraction must lie between zero and one")
        index = int(np.searchsorted(self.density_breaks, density, side="left"))
        return self.density_scales[index]

    def apply(
        self,
        raw_sd: np.ndarray,
        *,
        sampled_node_fraction: float | None = None,
    ) -> np.ndarray:
        raw = np.asarray(raw_sd, dtype=np.float64)
        return raw * self.scale_for_density(sampled_node_fraction)


def _mean_squared_standardized_error(
    errors: np.ndarray,
    raw_sd: np.ndarray,
    *,
    floor: float,
) -> float:
    error = np.asarray(errors, dtype=np.float64)
    sd = np.asarray(raw_sd, dtype=np.float64)
    if error.shape != sd.shape:
        raise ValueError("errors and raw_sd must have matching shapes")
    valid = np.isfinite(error) & np.isfinite(sd) & (sd > floor)
    if not np.any(valid):
        return float("nan")
    standardized = error[valid] / np.maximum(sd[valid], floor)
    return float(np.mean(standardized * standardized))
