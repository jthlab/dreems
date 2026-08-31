from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from dreems import checkpoints
from dreems.feems import (
    FEEMS_RECOMMENDED_CV_FOLDS,
    FEEMS_RECOMMENDED_LAMBDAS,
    FEEMS_RECOMMENDED_LAMBDAS_Q,
    FEEMSConfig,
    feems_edge_log_rates,
    polymorphic_snp_mask,
)
from dreems.metrics import centered, edge_log_metrics


def test_edge_log_metrics_are_centered_and_scale_aware() -> None:
    truth = np.asarray([1.0, 2.0, 4.0, 8.0])
    estimate = 2.0 * truth + 11.0

    metrics = edge_log_metrics(truth, estimate)

    assert metrics.corr == pytest.approx(1.0)
    assert metrics.scale_ratio == pytest.approx(2.0)
    assert metrics.slope == pytest.approx(2.0)
    assert metrics.mse > 0.0
    assert metrics.to_prefixed_dict()["edge_log_corr"] == pytest.approx(1.0)


def test_edge_log_metrics_drop_nonfinite_pairs() -> None:
    truth = np.asarray([1.0, np.nan, 3.0])
    estimate = np.asarray([2.0, 4.0, np.inf])

    metrics = edge_log_metrics(truth, estimate)

    assert math.isfinite(metrics.mse)
    assert math.isnan(metrics.corr)


def test_centered_preserves_nan_and_centers_finite_values() -> None:
    values = np.asarray([1.0, np.nan, 3.0])

    out = centered(values)

    assert out[0] == pytest.approx(-1.0)
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(1.0)


def test_feems_config_uses_leave_one_out_sentinel() -> None:
    assert FEEMS_RECOMMENDED_CV_FOLDS == 0
    assert FEEMSConfig().sklearn_cv_folds is None
    assert FEEMSConfig(cv_folds=5).sklearn_cv_folds == 5
    assert FEEMSConfig(cv_mode="replicate_holdout").cv_mode == "replicate_holdout"
    assert len(FEEMS_RECOMMENDED_LAMBDAS) == 10
    assert len(FEEMS_RECOMMENDED_LAMBDAS_Q) == 5
    assert FEEMS_RECOMMENDED_LAMBDAS[0] > FEEMS_RECOMMENDED_LAMBDAS[-1]


def test_polymorphic_snp_mask_filters_invariant_sites() -> None:
    genotypes = np.asarray(
        [
            [0, 0, 2, 1],
            [0, 1, 2, 1],
            [0, 2, 2, 1],
        ],
        dtype=np.float64,
    )

    assert polymorphic_snp_mask(genotypes).tolist() == [False, True, False, True]


def test_feems_edge_log_rates_preserves_requested_edge_order() -> None:
    fake_graph = SimpleNamespace(
        nnz_idx=(np.asarray([0, 0, 1]), np.asarray([1, 2, 2])),
        w=np.asarray([2.0, 8.0, 4.0]),
    )
    requested_edges = np.asarray([[1, 2], [0, 1], [2, 0]], dtype=np.int32)

    out = feems_edge_log_rates(fake_graph, requested_edges)

    assert out.tolist() == pytest.approx([math.log(4.0), math.log(2.0), math.log(8.0)])


def test_default_checkpoint_is_diffusion_band_model() -> None:
    loaded = checkpoints.load_checkpoint()

    assert loaded.model["schema_version"] == 2
    assert loaded.model["features"] == {
        "kind": "topology_diffusion_bands_v2",
        "covariance_components": 20,
    }
    assert loaded.model["node_dim"] == 109
    assert loaded.model["edge_dim"] == 253
    assert loaded.model["architecture"]["kind"] == "diffusion_band_surface_gate_v3"
    assert loaded.model["architecture"]["surface_gate"] is True
    assert loaded.params.surface_gate_hidden is not None
