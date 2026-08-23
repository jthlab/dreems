from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import dreems.feems as feems


def _copied_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value
        for key, value in kwargs.items()
    }


class FakeFitGraph:
    def __init__(self, n_nodes: int = 3) -> None:
        self.n_nodes = n_nodes
        self.w = np.asarray([1.0, 1.0], dtype=np.float64)
        self.s2: np.ndarray | float = np.full(n_nodes, 2.0, dtype=np.float64)
        self.fit_calls: list[dict[str, Any]] = []
        self.train_loss = math.nan

    def __len__(self) -> int:
        return self.n_nodes

    def fit(self, **kwargs: Any) -> None:
        self.fit_calls.append(_copied_kwargs(kwargs))
        value = float(kwargs["lamb"]) + float(kwargs.get("lamb_q", 0.0))
        self.w = np.asarray([value, value + 1.0], dtype=np.float64)
        if kwargs.get("optimize_q") == "1-dim":
            self.s2 = value + 2.0
        elif kwargs.get("optimize_q") == "n-dim":
            self.s2 = np.full(self.n_nodes, value + 2.0, dtype=np.float64)
        self.train_loss = value


class FakeSpatialGraph(FakeFitGraph):
    instances: list[FakeSpatialGraph] = []

    def __init__(
        self,
        *,
        genotypes: np.ndarray,
        sample_pos: np.ndarray,
        node_pos: np.ndarray,
        edges: np.ndarray,
        scale_snps: bool,
    ) -> None:
        super().__init__(len(node_pos))
        self.genotypes = np.asarray(genotypes)
        self.sample_pos = np.asarray(sample_pos)
        self.node_pos = np.asarray(node_pos)
        self.edges = np.asarray(edges)
        self.scale_snps = scale_snps
        self.n_snps = self.genotypes.shape[1]
        self.n_observed_nodes = len(np.unique(self.sample_pos, axis=0))
        self.nnz_idx = (
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([1, 2], dtype=np.int32),
        )
        self.factor: object | None = object()
        self.w0 = np.asarray([10.0, 20.0], dtype=np.float64)
        self.laplacian_weights: np.ndarray | None = None
        type(self).instances.append(self)

    def fit_null_model(self, *, verbose: bool) -> None:
        assert verbose is False
        self.w0 = np.asarray([10.0, 20.0], dtype=np.float64)
        self.s2 = np.full(len(self), 2.0, dtype=np.float64)

    def comp_graph_laplacian(self, weights: np.ndarray) -> None:
        self.laplacian_weights = np.asarray(weights, dtype=np.float64).copy()
        self.w = self.laplacian_weights.copy()

    def comp_precision(self, *, s2: np.ndarray) -> None:
        self.s2 = np.asarray(s2, dtype=np.float64).copy()


class FakeObjective:
    def __init__(self, sp_graph: FakeSpatialGraph) -> None:
        self.sp_graph = sp_graph
        self.inverted = False

    def inv(self) -> None:
        self.inverted = True

    def neg_log_lik(self) -> float:
        assert self.inverted
        return float(self.sp_graph.n_snps * self.sp_graph.w[0])


class FakeFEEMSAPI:
    def __init__(self) -> None:
        self.spatial_graph = FakeSpatialGraph
        self.objective = FakeObjective
        self.cv = self
        self.joint_cv_calls: list[tuple[FakeSpatialGraph, dict[str, Any]]] = []
        self.cv_calls: list[tuple[FakeSpatialGraph, dict[str, Any]]] = []
        self.setup_calls: list[tuple[FakeSpatialGraph, int, int]] = []
        self.train_graphs: list[FakeFitGraph] = []

    def run_cv_joint(self, sp_graph: FakeSpatialGraph, **kwargs: Any) -> np.ndarray:
        self.joint_cv_calls.append((sp_graph, _copied_kwargs(kwargs)))
        return np.asarray(
            [
                [[4.0, 3.0], [1.0, 9.0]],
                [[6.0, 5.0], [3.0, 11.0]],
            ],
            dtype=np.float64,
        )

    def run_cv(self, sp_graph: FakeSpatialGraph, **kwargs: Any) -> np.ndarray:
        self.cv_calls.append((sp_graph, _copied_kwargs(kwargs)))
        return np.asarray([[[1.0], [5.0]], [[3.0], [7.0]]], dtype=np.float64)

    def setup_k_fold_cv(
        self,
        sp_graph: FakeSpatialGraph,
        n_splits: int,
        *,
        random_state: int,
    ) -> np.ndarray:
        self.setup_calls.append((sp_graph, n_splits, random_state))
        is_train = np.ones((sp_graph.sample_pos.shape[0], n_splits), dtype=bool)
        for fold in range(n_splits):
            is_train[fold % is_train.shape[0], fold] = False
        return is_train

    def train_test_split(
        self,
        sp_graph: FakeSpatialGraph,
        is_train: np.ndarray,
    ) -> tuple[FakeFitGraph, SimpleNamespace]:
        assert is_train.dtype == np.bool_
        train = FakeFitGraph(len(sp_graph))
        self.train_graphs.append(train)
        return train, SimpleNamespace()

    def predict_snps(
        self,
        sp_graph: FakeSpatialGraph,
        sp_graph_train: FakeFitGraph,
        sp_graph_test: SimpleNamespace,
    ) -> tuple[None, float]:
        del sp_graph, sp_graph_test
        call = sp_graph_train.fit_calls[-1]
        return None, float(call["lamb"] + call.get("lamb_q", 0.0))


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> FakeFEEMSAPI:
    FakeSpatialGraph.instances = []
    api = FakeFEEMSAPI()
    monkeypatch.setattr(feems, "_import_feems", lambda: api)
    return api


@pytest.fixture
def fit_inputs() -> dict[str, Any]:
    return {
        "genotypes": np.asarray(
            [[0.0, 0.0, 2.0], [1.0, 1.0, 2.0], [2.0, 2.0, 2.0]],
            dtype=np.float64,
        ),
        "sample_nodes": np.asarray([0, 1, 2], dtype=np.int32),
        "graph": {
            "edges": np.asarray([[1, 2], [0, 1]], dtype=np.int32),
        },
        "active_positions": np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
    }


def test_recommended_protocol_defaults() -> None:
    config = feems.FEEMSConfig()

    assert config.sklearn_cv_folds is None
    np.testing.assert_allclose(config.lambdas, np.geomspace(1e-3, 1e2, 10)[::-1])
    np.testing.assert_allclose(config.lambdas_q, np.geomspace(1e-2, 1e2, 5)[::-1])
    assert config.factr == 1e10
    assert config.maxiter == 15000
    assert config.random_state == 500
    assert (
        feems.FEEMS_RECOMMENDED_LOO_PROTOCOL
        == "v2_joint_lambda_lambda_q_leave_one_observed_deme_out"
    )


def test_free_q_uses_official_joint_cv_defaults_and_requested_edge_order(
    fake_api: FakeFEEMSAPI,
    fit_inputs: dict[str, Any],
) -> None:
    config = feems.FEEMSConfig(
        mode="free_q",
        lambdas=(100.0, 10.0),
        lambdas_q=(20.0, 2.0),
        lb=1e-4,
        ub=1e4,
        factr=123.0,
        maxiter=456,
        random_state=789,
    )

    result = feems.fit_feems(config=config, **fit_inputs)

    sp_graph, cv_kwargs = fake_api.joint_cv_calls.pop()
    np.testing.assert_array_equal(sp_graph.edges, fit_inputs["graph"]["edges"] + 1)
    assert cv_kwargs["n_folds"] is None
    np.testing.assert_array_equal(cv_kwargs["lamb_grid"], [100.0, 10.0])
    np.testing.assert_array_equal(cv_kwargs["lamb_q_grid"], [20.0, 2.0])
    assert cv_kwargs | {"lamb_grid": None, "lamb_q_grid": None} == {
        "lamb_grid": None,
        "lamb_q_grid": None,
        "n_folds": None,
        "lb": 1e-4,
        "ub": 1e4,
        "factr": 123.0,
        "random_state": 789,
        "outer_verbose": False,
        "inner_verbose": False,
    }
    final_fit = sp_graph.fit_calls.pop()
    assert final_fit == {
        "lamb": 100.0,
        "lamb_q": 2.0,
        "optimize_q": "n-dim",
        "lb": math.log(1e-4),
        "ub": math.log(1e4),
        "factr": 123.0,
        "maxiter": 456,
        "verbose": False,
    }
    assert result.best_lambda == 100.0
    assert result.best_lambda_q == 2.0
    assert result.cv_folds == 3
    assert result.cv_mode == "leave_one_out"
    assert result.protocol == "v2_joint_lambda_lambda_q"
    assert result.n_snps_after_filter == 2
    np.testing.assert_allclose(result.mean_cv_error, [[5.0, 4.0], [2.0, 10.0]])
    np.testing.assert_allclose(result.edge_log_rates, np.log([103.0, 102.0]))


def test_recommended_loo_uses_documented_final_fit_call(
    fake_api: FakeFEEMSAPI,
    fit_inputs: dict[str, Any],
) -> None:
    result = feems.fit_feems_recommended_loo(cv_jobs=1, **fit_inputs)

    sp_graph = FakeSpatialGraph.instances[0]
    assert sp_graph.fit_calls == [{"lamb": result.best_lambda, "lamb_q": result.best_lambda_q}]
    assert result.cv_mode == "leave_one_out"
    assert result.cv_folds == 3
    assert result.protocol == "v2_joint_lambda_lambda_q"


def test_fixed_q_uses_official_lambda_cv(
    fake_api: FakeFEEMSAPI, fit_inputs: dict[str, Any]
) -> None:
    config = feems.FEEMSConfig(
        mode="fixed_q",
        lambdas=(100.0, 10.0),
        cv_folds=2,
        lb=1e-5,
        ub=1e5,
        factr=321.0,
        maxiter=654,
        random_state=987,
    )

    result = feems.fit_feems(config=config, **fit_inputs)

    sp_graph, cv_kwargs = fake_api.cv_calls.pop()
    assert cv_kwargs["n_folds"] == 2
    np.testing.assert_array_equal(cv_kwargs["lamb_grid"], [100.0, 10.0])
    assert sp_graph.fit_calls == [
        {
            "lamb": 100.0,
            "optimize_q": None,
            "lb": math.log(1e-5),
            "ub": math.log(1e5),
            "factr": 321.0,
            "maxiter": 654,
            "verbose": False,
        }
    ]
    assert result.best_lambda == 100.0
    assert math.isnan(result.best_lambda_q)
    assert result.cv_folds == 2
    assert result.cv_mode == "k_fold"
    assert result.optimize_q == "none"
    assert result.protocol == "v1_lambda_only_fixed_q"
    np.testing.assert_allclose(result.mean_cv_error, [2.0, 6.0])


def test_scalar_q_preserves_leave_one_node_out_cv_and_lambda_warm_starts(
    fake_api: FakeFEEMSAPI,
    fit_inputs: dict[str, Any],
) -> None:
    config = feems.FEEMSConfig(
        mode="scalar_q",
        lambdas=(100.0, 10.0),
        random_state=246,
    )

    result = feems.fit_feems(config=config, **fit_inputs)

    _, n_folds, random_state = fake_api.setup_calls.pop()
    assert n_folds == 3
    assert random_state == 246
    assert len(fake_api.train_graphs) == 3
    for train_graph in fake_api.train_graphs:
        first, second = train_graph.fit_calls
        np.testing.assert_array_equal(first["w_init"], [10.0, 20.0])
        np.testing.assert_array_equal(first["s2_init"], [2.0])
        np.testing.assert_array_equal(second["w_init"], [100.0, 101.0])
        np.testing.assert_array_equal(second["s2_init"], [102.0])
        assert [first["lamb"], second["lamb"]] == [100.0, 10.0]
        assert first["optimize_q"] == second["optimize_q"] == "1-dim"
    assert result.best_lambda == 10.0
    assert result.best_lambda_q == 0.0
    assert result.cv_folds == 3
    assert result.optimize_q == "1-dim"
    assert result.protocol == "scalar_q_lambda_cv"
    assert result.scalar_s2 == 12.0
    np.testing.assert_allclose(result.mean_cv_error, [100.0, 10.0])


def test_parallel_joint_cv_preserves_feems_grid_and_warm_start_order(
    fake_api: FakeFEEMSAPI,
    fit_inputs: dict[str, Any],
) -> None:
    config = feems.FEEMSConfig(
        lambdas=(100.0, 10.0),
        lambdas_q=(20.0, 2.0),
        cv_folds=2,
        cv_jobs=1,
    )
    sp_graph = feems.make_spatial_graph(**fit_inputs)

    errors = feems.run_cv_joint_fold_parallel(
        sp_graph=sp_graph,
        lamb_grid=np.asarray(config.lambdas),
        lamb_q_grid=np.asarray(config.lambdas_q),
        config=config,
        **fit_inputs,
    )

    assert errors.shape == (2, 2, 2)
    np.testing.assert_allclose(errors[0], [[120.0, 30.0], [102.0, 12.0]])
    calls = fake_api.train_graphs[0].fit_calls
    assert [(call["lamb"], call["lamb_q"]) for call in calls] == [
        (100.0, 20.0),
        (100.0, 2.0),
        (10.0, 20.0),
        (10.0, 2.0),
    ]
    np.testing.assert_array_equal(calls[0]["w_init"], [10.0, 20.0])
    np.testing.assert_array_equal(calls[1]["w_init"], [120.0, 121.0])
    np.testing.assert_array_equal(calls[2]["w_init"], [102.0, 103.0])
    np.testing.assert_array_equal(calls[2]["s2_init"], [104.0, 104.0, 104.0])
    np.testing.assert_array_equal(calls[3]["w_init"], [30.0, 31.0])
    np.testing.assert_array_equal(calls[3]["s2_init"], [104.0, 104.0, 104.0])
    assert all("maxiter" not in call for call in calls)


def test_parallel_joint_cv_reuses_completed_fold_checkpoints(
    fake_api: FakeFEEMSAPI,
    fit_inputs: dict[str, Any],
    tmp_path: Path,
) -> None:
    config = feems.FEEMSConfig(
        lambdas=(100.0, 10.0),
        lambdas_q=(20.0, 2.0),
        cv_folds=2,
        cv_jobs=1,
    )
    sp_graph = feems.make_spatial_graph(**fit_inputs)
    checkpoint_dir = tmp_path / "folds"
    kwargs = {
        "sp_graph": sp_graph,
        "lamb_grid": np.asarray(config.lambdas),
        "lamb_q_grid": np.asarray(config.lambdas_q),
        "config": config,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_key": "input-and-protocol-hash",
        **fit_inputs,
    }

    first = feems.run_cv_joint_fold_parallel(**kwargs)

    assert sorted(path.name for path in checkpoint_dir.iterdir()) == [
        "fold_0000.npz",
        "fold_0001.npz",
    ]
    fake_api.train_graphs.clear()
    second = feems.run_cv_joint_fold_parallel(**kwargs)

    np.testing.assert_array_equal(second, first)
    assert fake_api.train_graphs == []


def test_recommended_loo_fold_runs_one_documented_split(
    fake_api: FakeFEEMSAPI,
    fit_inputs: dict[str, Any],
) -> None:
    errors = feems.recommended_loo_fold_errors(fold=1, **fit_inputs)

    assert errors.shape == (
        len(feems.FEEMS_RECOMMENDED_LAMBDAS_Q),
        len(feems.FEEMS_RECOMMENDED_LAMBDAS),
    )
    _, n_folds, random_state = fake_api.setup_calls.pop()
    assert n_folds == 3
    assert random_state == 500
    assert len(fake_api.train_graphs) == 1


def test_replicate_holdout_preserves_grid_warm_starts_and_refits_selected_model(
    fake_api: FakeFEEMSAPI,
    fit_inputs: dict[str, Any],
) -> None:
    config = feems.FEEMSConfig(
        mode="free_q",
        cv_mode="replicate_holdout",
        lambdas=(100.0, 10.0),
        lambdas_q=(20.0, 2.0),
    )
    validation_genotypes = np.asarray(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        dtype=np.float64,
    )

    result = feems.fit_feems(
        config=config,
        validation_genotypes=validation_genotypes,
        **fit_inputs,
    )

    tuning_graph = FakeSpatialGraph.instances[0]
    calls = tuning_graph.fit_calls
    assert [(call["lamb"], call["lamb_q"]) for call in calls] == [
        (100.0, 20.0),
        (100.0, 2.0),
        (10.0, 20.0),
        (10.0, 2.0),
    ]
    np.testing.assert_array_equal(calls[0]["w_init"], [10.0, 20.0])
    np.testing.assert_array_equal(calls[1]["w_init"], [120.0, 121.0])
    np.testing.assert_array_equal(calls[2]["w_init"], [120.0, 121.0])
    np.testing.assert_array_equal(calls[3]["w_init"], [30.0, 31.0])
    final_graph = FakeSpatialGraph.instances[-1]
    assert final_graph is not tuning_graph
    assert final_graph.fit_calls == [
        {
            "lamb": 10.0,
            "lamb_q": 2.0,
            "optimize_q": "n-dim",
            "lb": math.log(config.lb),
            "ub": math.log(config.ub),
            "factr": config.factr,
            "maxiter": config.maxiter,
            "verbose": False,
        }
    ]
    assert result.best_lambda == 10.0
    assert result.best_lambda_q == 2.0
    assert result.cv_folds == 1
    assert result.cv_mode == "replicate_holdout"
    assert result.validation_n_snps == 2
    assert result.protocol == "v2_replicate_holdout_lambda_lambda_q"
    np.testing.assert_allclose(result.mean_cv_error, [[120.0, 30.0], [102.0, 12.0]])


def test_replicate_holdout_rejects_unsupported_modes_before_import(
    monkeypatch: pytest.MonkeyPatch,
    fit_inputs: dict[str, Any],
) -> None:
    monkeypatch.setattr(
        feems,
        "_import_feems",
        lambda: pytest.fail("unsupported mode should be rejected before importing FEEMS"),
    )

    with pytest.raises(ValueError, match="supports mode='free_q'"):
        feems.fit_feems(
            config=feems.FEEMSConfig(mode="fixed_q", cv_mode="replicate_holdout"),
            **fit_inputs,
        )


def test_node_cv_rejects_unknown_mode_before_import(
    monkeypatch: pytest.MonkeyPatch,
    fit_inputs: dict[str, Any],
) -> None:
    monkeypatch.setattr(
        feems,
        "_import_feems",
        lambda: pytest.fail("unknown mode should be rejected before importing FEEMS"),
    )

    with pytest.raises(ValueError, match="unknown FEEMS mode: invalid"):
        feems.fit_feems(
            config=feems.FEEMSConfig(mode="invalid"),  # type: ignore[arg-type]
            **fit_inputs,
        )
