from __future__ import annotations

import numpy as np
import pytest

from dreems.metrics import centered, edge_log_metrics


def test_edge_metrics_reports_signed_correlation() -> None:
    truth = np.asarray([2.0, 1.0, -1.0, -2.0], dtype=np.float64)

    assert edge_log_metrics(truth, truth).corr == pytest.approx(1.0)
    assert edge_log_metrics(truth, -truth).corr == pytest.approx(-1.0)


def test_constant_estimate_has_zero_slope_but_undefined_correlation():
    measured = edge_log_metrics(np.asarray([-1.0, 1.0]), np.ones(2))
    assert np.isnan(measured.corr)
    assert measured.slope == measured.scale_ratio == 0
    assert measured.mse == measured.rmse == 1
    assert list(measured.to_prefixed_dict()) == [
        "edge_log_corr",
        "edge_log_rmse",
        "edge_log_mse",
        "edge_log_scale_ratio",
        "edge_log_slope",
    ]


def test_metrics_keep_finite_pairs_and_centering_preserves_missing_values():
    truth = np.asarray([0.0, 2.0, np.nan, np.inf])
    estimate = np.asarray([4.0, 6.0, 100.0, 200.0])
    assert edge_log_metrics(truth, estimate).mse == 0
    np.testing.assert_equal(centered(truth), [-1.0, 1.0, np.nan, np.nan])


def test_metrics_reject_complex_arrays_before_conversion():
    with pytest.raises(ValueError, match="real"):
        centered(np.asarray([1j, 0j]))
    with pytest.raises(ValueError, match="real"):
        edge_log_metrics(np.asarray([1j, 0j]), np.zeros(2))
