from __future__ import annotations

import numpy as np

from dreems.feems import feems_edge_log_rates


class FakeSpatialGraph:
    def __init__(self) -> None:
        self.nnz_idx = (
            np.asarray([2, 0, 1], dtype=np.int32),
            np.asarray([3, 1, 2], dtype=np.int32),
        )
        self.w = np.asarray([8.0, 2.0, 4.0], dtype=np.float64)


def test_feems_log_rates_follow_requested_edge_order() -> None:
    graph_edges = np.asarray(
        [
            [1, 2],
            [0, 1],
            [2, 3],
        ],
        dtype=np.int32,
    )

    log_rates = feems_edge_log_rates(FakeSpatialGraph(), graph_edges)

    np.testing.assert_allclose(log_rates, np.log([4.0, 2.0, 8.0]))


def test_feems_log_rates_treat_edges_as_undirected() -> None:
    graph_edges = np.asarray(
        [
            [3, 2],
            [1, 0],
        ],
        dtype=np.int32,
    )

    log_rates = feems_edge_log_rates(FakeSpatialGraph(), graph_edges)

    np.testing.assert_allclose(log_rates, np.log([8.0, 2.0]))
