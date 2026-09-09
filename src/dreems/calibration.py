"""Fit multiplicative corrections to posterior standard deviations."""

# ruff: noqa: F722, F821, UP037

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

import numpy as np
from beartype import beartype
from jaxtyping import Float, Shaped, jaxtyped
from jaxtyping import Real as RealArray

CalibrationMethod = Literal["raw", "global", "density"]
DEFAULT_DENSITY_BREAKS = (0.10, 0.25)
INTERVAL_LEVELS = {
    "50": 0.6744897501960817,
    "80": 1.2815515655446004,
    "95": 1.959963984540054,
}


@beartype
def _positive_finite(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite")
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

    @beartype
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
    @beartype
    def from_squared_errors(
        cls,
        errors: Shaped[np.ndarray, "*shape"],
        raw_sd: Shaped[np.ndarray, "*shape"],
        *,
        floor: Real = 1e-12,
    ) -> SDCalibration:
        """Fit a global scale from residuals, which this method squares itself.

        Every residual must be finite and every SD positive. ``floor`` limits
        the denominator; small SDs do not cause observations to be discarded.
        """

        graph_score = _mean_squared_standardized_error(errors, raw_sd, floor=floor)
        return cls(method="global", global_scale=max(float(np.sqrt(graph_score)), floor))

    @classmethod
    @beartype
    def from_graphs(
        cls,
        errors: Sequence[RealArray[np.ndarray, "*_edge_shape"]],
        raw_sd: Sequence[RealArray[np.ndarray, "*_edge_shape"]],
        sampled_node_fraction: Sequence[Real] | RealArray[np.ndarray, "graphs"],
        *,
        density_breaks: Sequence[Real] | RealArray[np.ndarray, "breaks"] = DEFAULT_DENSITY_BREAKS,
        density_shrinkage: Real = 25.0,
        floor: Real = 1e-12,
    ) -> SDCalibration:
        """Fit global scales and separate scales for sampling-density bins.

        Arrays may have any shape; each graph contributes one mean squared
        standardized error over all its entries. Within
        each density bin, ``density_shrinkage`` sets the weight of the global
        score, measured in equivalent graphs. This stabilizes sparsely
        populated bins. All supplied observations must be valid.
        """

        if len(errors) != len(raw_sd) or len(errors) != len(sampled_node_fraction):
            raise ValueError("errors, raw_sd, and sampled_node_fraction must align by graph")
        if density_shrinkage < 0.0 or not np.isfinite(density_shrinkage):
            raise ValueError("density_shrinkage must be finite and nonnegative")
        floor = _positive_finite(floor, name="floor")
        breaks = cls(density_breaks=tuple(density_breaks)).density_breaks
        density = np.asarray(sampled_node_fraction, dtype=np.float64)
        if density.shape != (len(errors),) or not density.size:
            raise ValueError("calibration needs a nonempty vector of graph sampling fractions")
        if np.any(~np.isfinite(density) | (density < 0.0) | (density > 1.0)):
            raise ValueError("sampled_node_fraction must lie between zero and one")
        graph_scores = np.asarray(
            [
                _mean_squared_standardized_error(error, sd, floor=floor)
                for error, sd in zip(errors, raw_sd, strict=True)
            ],
            dtype=np.float64,
        )
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
    @beartype
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

    @beartype
    def to_dict(self) -> dict[str, Any]:
        """Return calibration settings in the checkpoint metadata format."""
        return {
            "schema_version": 1,
            "method": self.method,
            "global_scale": self.global_scale,
            "density_breaks": list(self.density_breaks),
            "density_scales": list(self.density_scales),
        }

    @beartype
    def scale_for_density(self, sampled_node_fraction: Real | None) -> float:
        """Choose a bin with a closed upper boundary; None uses the global scale."""
        if sampled_node_fraction is not None and (
            not np.isfinite(sampled_node_fraction) or not 0.0 <= sampled_node_fraction <= 1.0
        ):
            raise ValueError("sampled_node_fraction must lie between zero and one")
        if self.method == "raw":
            return 1.0
        if self.method == "global" or sampled_node_fraction is None or not self.density_scales:
            return self.global_scale
        density = float(sampled_node_fraction)
        index = int(np.searchsorted(self.density_breaks, density, side="left"))
        return self.density_scales[index]

    @jaxtyped(typechecker=beartype)
    def apply(
        self,
        raw_sd: Shaped[np.ndarray, "*shape"],
        *,
        sampled_node_fraction: Real | None = None,
    ) -> Float[np.ndarray, "*shape"] | np.float64:
        """Scale SDs, leaving means unchanged. Zero remains zero; negative SDs fail."""
        raw = np.asarray(raw_sd, dtype=np.float64)
        if np.any(~np.isfinite(raw) | (raw < 0.0)):
            raise ValueError("raw_sd must be finite and nonnegative")
        with np.errstate(over="ignore"):
            scaled = raw * self.scale_for_density(sampled_node_fraction)
        if not np.all(np.isfinite(scaled)):
            raise ValueError("calibrated SDs overflowed")
        return scaled


@jaxtyped(typechecker=beartype)
def _standardized_errors(
    errors: Shaped[np.ndarray, "*shape"],
    raw_sd: Shaped[np.ndarray, "*sd_shape"],
    *,
    floor: float,
) -> Float[np.ndarray, "*shape"] | np.float64:
    """Validate one graph, raising ValueError for mismatched input shapes."""
    error = np.asarray(errors, dtype=np.float64)
    sd = np.asarray(raw_sd, dtype=np.float64)
    if error.shape != sd.shape:
        raise ValueError("errors and raw_sd must have matching shapes")
    if not error.size or not np.all(np.isfinite(error)):
        raise ValueError("calibration errors must be nonempty and finite")
    if np.any(~np.isfinite(sd) | (sd <= 0.0)):
        raise ValueError("calibration SDs must be positive and finite")
    with np.errstate(over="ignore"):
        standardized = error / np.maximum(sd, floor)
    if not np.all(np.isfinite(standardized)):
        raise ValueError("standardized calibration errors overflowed")
    return standardized


@beartype
def _mean_squared_standardized_error(
    errors: Shaped[np.ndarray, "*shape"], raw_sd: Shaped[np.ndarray, "*shape"], *, floor: Real
) -> float:
    """Give each graph one squared-error score, regardless of its edge count."""
    standardized = _standardized_errors(errors, raw_sd, floor=_positive_finite(floor, name="floor"))
    with np.errstate(over="ignore"):
        score = float(np.mean(standardized * standardized))
    if not np.isfinite(score):
        raise ValueError("squared calibration errors overflowed")
    return score


@beartype
def candidates(
    errors: Sequence[RealArray[np.ndarray, "*_edge_shape"]],
    raw_sd: Sequence[RealArray[np.ndarray, "*_edge_shape"]],
    density: RealArray[np.ndarray, "graphs"],
) -> dict[str, SDCalibration]:
    """Fit the same raw, global, and density options for both release workflows."""
    density_fit = SDCalibration.from_graphs(errors, raw_sd, density, density_shrinkage=25.0)
    return {
        "raw": SDCalibration(),
        "global": SDCalibration(method="global", global_scale=density_fit.global_scale),
        "density": density_fit,
    }


@beartype
def summarize(
    errors: Sequence[RealArray[np.ndarray, "*_edge_shape"]],
    raw_sd: Sequence[RealArray[np.ndarray, "*_edge_shape"]],
    density: RealArray[np.ndarray, "graphs"],
    fitted: SDCalibration,
) -> dict[str, float | int]:
    """Average NLL and interval coverage within each graph, then across graphs.

    NLL omits the shared log(2*pi)/2 constant. Every supplied observation
    contributes to the result; invalid observations raise an error.
    """
    if density.shape != (len(errors),) or len(raw_sd) != len(errors) or not len(errors):
        raise ValueError("calibration arrays must align over a nonempty set of graphs")
    nll = []
    coverage = {level: [] for level in INTERVAL_LEVELS}
    edge_count = 0
    for error, sd, local_density in zip(errors, raw_sd, density, strict=True):
        calibrated = fitted.apply(sd, sampled_node_fraction=float(local_density))
        standardized = _standardized_errors(error, calibrated, floor=0.0)
        with np.errstate(over="ignore"):
            score = float(np.mean(np.log(calibrated) + 0.5 * standardized * standardized))
        if not np.isfinite(score):
            raise ValueError("calibration NLL overflowed")
        nll.append(score)
        for level, z_value in INTERVAL_LEVELS.items():
            coverage[level].append(float(np.mean(np.abs(standardized) <= z_value)))
        edge_count += standardized.size
    with np.errstate(over="ignore"):
        mean_nll = float(np.mean(nll))
    if not np.isfinite(mean_nll):
        raise ValueError("graph-balanced calibration NLL overflowed")
    result: dict[str, float | int] = {
        "graphs": len(nll),
        "edges": edge_count,
        "graph_balanced_nll": mean_nll,
    }
    result.update(
        {
            f"graph_balanced_coverage_{level}": float(np.mean(values))
            for level, values in coverage.items()
        }
    )
    return result
