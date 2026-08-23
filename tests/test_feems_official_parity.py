from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dreems import feems


def _small_feems_case() -> dict[str, object]:
    positions = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    edges = np.asarray(
        [[0, 1], [0, 2], [0, 3], [1, 3], [2, 3]],
        dtype=np.int32,
    )
    # Two diploid samples per observed node and no monomorphic columns.
    genotypes = np.asarray(
        [
            [0, 0, 0, 1, 1, 0, 2, 1],
            [0, 1, 0, 0, 1, 1, 2, 0],
            [0, 1, 1, 1, 2, 0, 1, 1],
            [1, 1, 0, 2, 1, 1, 1, 2],
            [1, 0, 2, 1, 0, 2, 1, 1],
            [1, 1, 1, 0, 0, 1, 0, 2],
            [2, 2, 1, 1, 1, 2, 0, 1],
            [2, 1, 2, 2, 1, 1, 0, 0],
        ],
        dtype=np.float64,
    )
    return {
        "genotypes": genotypes,
        "sample_nodes": np.repeat(np.arange(4, dtype=np.int32), 2),
        "graph": {"edges": edges},
        "active_positions": positions,
    }


def test_recommended_loo_constructor_is_frozen() -> None:
    config = feems.recommended_loo_config(cv_jobs=7)

    feems.require_recommended_loo(config)
    assert config.mode == "free_q"
    assert config.cv_mode == "node_cv"
    assert config.sklearn_cv_folds is None
    assert config.cv_jobs == 7

    with pytest.raises(ValueError, match="publication H2H requires"):
        feems.require_recommended_loo(feems.FEEMSConfig(cv_folds=5, cv_jobs=7))


def test_fold_parallel_joint_cv_matches_official_feems_serial_cv(tmp_path: Path) -> None:
    pytest.importorskip("feems")
    inputs = _small_feems_case()
    config = feems.FEEMSConfig(
        lambdas=(10.0, 0.1),
        lambdas_q=(10.0, 0.1),
        cv_jobs=2,
    )
    serial_graph = feems.make_spatial_graph(**inputs)  # type: ignore[arg-type]
    parallel_graph = feems.make_spatial_graph(**inputs)  # type: ignore[arg-type]
    api = feems._import_feems()
    lamb_grid = np.asarray(config.lambdas, dtype=np.float64)
    lamb_q_grid = np.asarray(config.lambdas_q, dtype=np.float64)

    serial = api.cv.run_cv_joint(
        serial_graph,
        lamb_grid=lamb_grid,
        lamb_q_grid=lamb_q_grid,
        n_folds=None,
        lb=config.lb,
        ub=config.ub,
        factr=config.factr,
        random_state=config.random_state,
        outer_verbose=False,
        inner_verbose=False,
    )
    fold_parallel = feems.run_cv_joint_fold_parallel(
        sp_graph=parallel_graph,
        lamb_grid=lamb_grid,
        lamb_q_grid=lamb_q_grid,
        config=config,
        checkpoint_dir=tmp_path / "folds",
        checkpoint_key="parity-test",
        **inputs,  # type: ignore[arg-type]
    )

    np.testing.assert_allclose(fold_parallel, serial, rtol=1e-10, atol=1e-12, equal_nan=True)
