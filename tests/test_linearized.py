from __future__ import annotations

import numpy as np
import pytest

from dreems.linearized import all_pair_adjoint_conductance_initializer


def test_all_pair_adjoint_initializer_returns_edge_field() -> None:
    sample_covariance = np.asarray(
        [
            [1.0, 0.20, 0.05],
            [0.20, 1.0, 0.35],
            [0.05, 0.35, 1.0],
        ],
        dtype=np.float64,
    )
    sample_nodes = np.asarray([0, 1, 3], dtype=np.int32)
    node_coordinates = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float64,
    )
    edges = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int32)

    estimate = all_pair_adjoint_conductance_initializer(
        sample_covariance=sample_covariance,
        sample_nodes=sample_nodes,
        node_coordinates=node_coordinates,
        edges=edges,
        effective_snps=100,
        smoothness=0.1,
        output_sd=0.5,
    )

    assert estimate.edge_log_conductance.shape == (3,)
    assert estimate.edge_information.shape == (3,)
    assert np.all(np.isfinite(estimate.edge_log_conductance))
    assert np.all(estimate.edge_information >= 0.0)
    assert estimate.diagnostics["observed_nodes"] == 3
    assert estimate.diagnostics["observed_pairs"] == 3
    assert estimate.diagnostics["edge_log_sd"] == pytest.approx(0.5)


def test_all_pair_adjoint_initializer_requires_two_observed_nodes() -> None:
    sample_covariance = np.eye(1, dtype=np.float64)
    sample_nodes = np.asarray([0], dtype=np.int32)
    node_coordinates = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    edges = np.asarray([[0, 1]], dtype=np.int32)

    with pytest.raises(ValueError, match="at least two observed"):
        all_pair_adjoint_conductance_initializer(
            sample_covariance=sample_covariance,
            sample_nodes=sample_nodes,
            node_coordinates=node_coordinates,
            edges=edges,
        )
