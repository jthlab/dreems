from __future__ import annotations

import numpy as np
import pytest

from dreems.calibration import SDCalibration


def test_graph_balanced_fit_does_not_overweight_large_graphs() -> None:
    fitted = SDCalibration.from_graphs(
        errors=[np.asarray([3.0]), np.ones(100)],
        raw_sd=[np.ones(1), np.ones(100)],
        sampled_node_fraction=[0.05, 0.5],
        density_shrinkage=0.0,
    )

    assert fitted.method == "density"
    assert fitted.global_scale == pytest.approx(np.sqrt(5.0))
    assert fitted.density_scales == pytest.approx((3.0, np.sqrt(5.0), 1.0))


def test_density_calibration_uses_closed_intervals_at_breaks() -> None:
    fitted = SDCalibration(
        method="density",
        global_scale=2.0,
        density_breaks=(0.1, 0.25),
        density_scales=(3.0, 2.0, 1.0),
    )

    assert fitted.scale_for_density(0.1) == 3.0
    assert fitted.scale_for_density(0.25) == 2.0
    np.testing.assert_allclose(
        fitted.apply(np.asarray([1.0, 2.0]), sampled_node_fraction=0.5),
        [1.0, 2.0],
    )
    assert SDCalibration.from_dict(fitted.to_dict()) == fitted


def test_missing_calibration_is_identity() -> None:
    fitted = SDCalibration.from_dict(None)
    raw = np.asarray([0.5, 1.5])

    assert fitted.method == "raw"
    np.testing.assert_array_equal(fitted.apply(raw, sampled_node_fraction=0.01), raw)
