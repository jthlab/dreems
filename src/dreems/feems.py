"""FEEMS baseline protocols for DREEMS publication comparisons."""

from __future__ import annotations

import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np

FEEMSMode = Literal["free_q", "fixed_q", "scalar_q"]
FEEMSCVMode = Literal["node_cv", "replicate_holdout"]

FEEMS_RECOMMENDED_LAMBDAS = tuple(
    float(x) for x in np.geomspace(1e-3, 1e2, 10, endpoint=True)[::-1]
)
FEEMS_RECOMMENDED_LAMBDAS_Q = tuple(
    float(x) for x in np.geomspace(1e-2, 1e2, 5, endpoint=True)[::-1]
)
FEEMS_RECOMMENDED_CV_FOLDS = 0
FEEMS_RECOMMENDED_FACTR = 1e10
FEEMS_RECOMMENDED_MAXITER = 15000
FEEMS_RECOMMENDED_LOO_PROTOCOL = "v2_joint_lambda_lambda_q_leave_one_observed_deme_out"
FEEMS_DEFAULT_FINAL_FACTR = 1e7


def _import_feems() -> SimpleNamespace:
    try:
        from feems import cross_validation
        from feems.objective import Objective
        from feems.spatial_graph import SpatialGraph
    except ModuleNotFoundError as exc:
        missing = exc.name or "<unknown>"
        raise ModuleNotFoundError(
            "FEEMS fitting requires the optional FEEMS dependencies. "
            f"Missing module: {missing!r}. Install them with `uv sync --extra feems`."
        ) from exc
    return SimpleNamespace(
        spatial_graph=SpatialGraph,
        objective=Objective,
        cv=cross_validation,
    )


@dataclass(frozen=True)
class FEEMSConfig:
    """FEEMS fitting protocol used for paper baselines.

    `cv_folds <= 0` means FEEMS' default leave-one-observed-node-out CV.
    """

    mode: FEEMSMode = "free_q"
    cv_mode: FEEMSCVMode = "node_cv"
    lambdas: tuple[float, ...] = FEEMS_RECOMMENDED_LAMBDAS
    lambdas_q: tuple[float, ...] = FEEMS_RECOMMENDED_LAMBDAS_Q
    cv_folds: int = FEEMS_RECOMMENDED_CV_FOLDS
    factr: float = FEEMS_RECOMMENDED_FACTR
    maxiter: int = FEEMS_RECOMMENDED_MAXITER
    lb: float = 1e-6
    ub: float = 1e6
    random_state: int = 500
    cv_jobs: int = 1

    @property
    def sklearn_cv_folds(self) -> int | None:
        return None if self.cv_folds <= 0 else self.cv_folds

    @property
    def parallel_cv_jobs(self) -> int:
        return max(int(self.cv_jobs), 1)


def recommended_loo_config(*, cv_jobs: int = 1) -> FEEMSConfig:
    """Return the FEEMS v2 cross-validation recipe used in paper comparisons.

    FEEMS' cross-validation notebook jointly tunes the edge and node-variance
    penalties. Leaving ``n_folds`` unset invokes its default: one held-out
    observed deme per fold. ``cv_jobs`` only controls parallel execution of
    those independent folds.
    """

    return FEEMSConfig(cv_jobs=cv_jobs)


def require_recommended_loo(config: FEEMSConfig) -> None:
    """Reject changes to the documented FEEMS comparison protocol."""

    expected = recommended_loo_config(cv_jobs=config.cv_jobs)
    if config != expected:
        raise ValueError(
            "publication H2H requires FEEMS v2 joint lambda/lambda_q "
            "leave-one-observed-deme-out CV with the documented defaults"
        )


@dataclass(frozen=True)
class FEEMSResult:
    edge_log_rates: np.ndarray
    best_lambda: float
    best_lambda_q: float
    cv_folds: int
    cv_mode: str
    optimize_q: str
    protocol: str
    n_snps_after_filter: int
    n_observed_nodes: int
    train_loss: float
    mean_cv_error: np.ndarray
    scalar_s2: float | None = None
    validation_n_snps: int | None = None

    def info_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "feems_protocol": self.protocol,
            "best_lambda": self.best_lambda,
            "best_lambda_q": self.best_lambda_q,
            "cv_folds": self.cv_folds,
            "cv_mode": self.cv_mode,
            "optimize_q": self.optimize_q,
            "n_snps_after_filter": self.n_snps_after_filter,
            "n_observed_nodes": self.n_observed_nodes,
            "train_loss": self.train_loss,
            "mean_cv_error": self.mean_cv_error.tolist(),
        }
        if self.scalar_s2 is not None:
            out["scalar_s2"] = self.scalar_s2
        if self.validation_n_snps is not None:
            out["validation_n_snps"] = self.validation_n_snps
        return out


@dataclass(frozen=True)
class _FEEMSInputs:
    genotypes: np.ndarray
    sample_nodes: np.ndarray
    graph: dict[str, np.ndarray]
    active_positions: np.ndarray

    @property
    def edges(self) -> np.ndarray:
        return np.asarray(self.graph["edges"], dtype=np.int32)


def polymorphic_snp_mask(genotypes: np.ndarray) -> np.ndarray:
    genotypes = np.asarray(genotypes)
    allele_sum = genotypes.sum(axis=0)
    return (allele_sum > 0.0) & (allele_sum < 2.0 * genotypes.shape[0])


def make_spatial_graph(
    *,
    genotypes: np.ndarray,
    sample_nodes: np.ndarray,
    graph: dict[str, np.ndarray],
    active_positions: np.ndarray,
) -> Any:
    return _make_spatial_graph(
        _FEEMSInputs(genotypes, sample_nodes, graph, active_positions),
        _import_feems(),
    )


def _make_spatial_graph(inputs: _FEEMSInputs, api: Any) -> Any:
    genotypes = np.asarray(inputs.genotypes)
    mask = polymorphic_snp_mask(genotypes)
    if not np.any(mask):
        raise ValueError("FEEMS has no polymorphic SNPs")
    sample_nodes = np.asarray(inputs.sample_nodes, dtype=np.int32)
    active_positions = np.asarray(inputs.active_positions, dtype=np.float64)
    return api.spatial_graph(
        genotypes=genotypes[:, mask],
        sample_pos=active_positions[sample_nodes],
        node_pos=active_positions,
        edges=inputs.edges + 1,
        scale_snps=True,
    )


def _node_cv_folds(sp_graph: Any, config: FEEMSConfig) -> int:
    if config.sklearn_cv_folds is None:
        return int(sp_graph.n_observed_nodes)
    return int(config.sklearn_cv_folds)


def _fit_spatial_graph(
    sp_graph: Any,
    config: FEEMSConfig,
    *,
    lamb: float,
    optimize_q: str | None,
    lamb_q: float | None = None,
    include_maxiter: bool = True,
    **initial_values: Any,
) -> None:
    fit_kwargs: dict[str, Any] = {
        "lamb": float(lamb),
        "optimize_q": optimize_q,
        **initial_values,
        "lb": math.log(float(config.lb)),
        "ub": math.log(float(config.ub)),
        "factr": config.factr,
    }
    if lamb_q is not None:
        fit_kwargs["lamb_q"] = float(lamb_q)
    if include_maxiter:
        fit_kwargs["maxiter"] = config.maxiter
    fit_kwargs["verbose"] = False
    sp_graph.fit(**fit_kwargs)


_joint_cv_worker_context: tuple[_FEEMSInputs, np.ndarray, np.ndarray, FEEMSConfig, float] | None = (
    None
)


def _initialize_joint_cv_fold_worker(
    inputs: _FEEMSInputs,
    lamb_grid: np.ndarray,
    lamb_q_grid: np.ndarray,
    config: FEEMSConfig,
    alpha_fact: float,
) -> None:
    global _joint_cv_worker_context
    _joint_cv_worker_context = (inputs, lamb_grid, lamb_q_grid, config, alpha_fact)


def _fit_joint_cv_fold_on_graph(
    *,
    fold: int,
    is_train: np.ndarray,
    api: Any,
    sp_graph: Any,
    lamb_grid: np.ndarray,
    lamb_q_grid: np.ndarray,
    config: FEEMSConfig,
    alpha_fact: float,
) -> tuple[int, np.ndarray]:
    sp_graph.fit_null_model(verbose=False)
    w0 = sp_graph.w0
    s2 = _n_dim_s2_init(sp_graph, sp_graph.s2)
    alpha_cv = alpha_fact / float(np.mean(w0))
    alpha_q = alpha_fact / float(np.mean(s2))
    if sp_graph.factor is not None:
        sp_graph.factor = None
    sp_graph_train, sp_graph_test = api.cv.train_test_split(
        sp_graph,
        np.asarray(is_train, dtype=bool),
    )
    errors = np.empty((lamb_q_grid.shape[0], lamb_grid.shape[0]), dtype=np.float64)
    init_w_list = [w0]
    init_s2_list = [s2]
    for i, lw in enumerate(lamb_grid):
        w_init = init_w_list[-1]
        for iq, lq in enumerate(lamb_q_grid):
            s2_init = init_s2_list[-1]
            try:
                _fit_spatial_graph(
                    sp_graph_train,
                    config,
                    lamb=lw,
                    lamb_q=lq,
                    optimize_q="n-dim",
                    include_maxiter=False,
                    w_init=w_init,
                    s2_init=s2_init,
                    alpha_q=alpha_q,
                    alpha=alpha_cv,
                )
                _pred, error = api.cv.predict_snps(sp_graph, sp_graph_train, sp_graph_test)
                errors[iq, i] = float(error)
            except Exception:  # noqa: BLE001
                errors[iq, i] = np.nan
            w_init = np.asarray(sp_graph_train.w, dtype=np.float64).copy()
            s2_init = _n_dim_s2_init(sp_graph_train, sp_graph_train.s2)
            if i == 0:
                init_w_list.append(w_init)
                init_s2_list.append(s2_init)
    return int(fold), errors


def _fit_joint_cv_fold(
    *,
    fold: int,
    is_train: np.ndarray,
    inputs: _FEEMSInputs,
    lamb_grid: np.ndarray,
    lamb_q_grid: np.ndarray,
    config: FEEMSConfig,
    alpha_fact: float,
) -> tuple[int, np.ndarray]:
    api = _import_feems()
    sp_graph = _make_spatial_graph(inputs, api)
    return _fit_joint_cv_fold_on_graph(
        fold=fold,
        is_train=is_train,
        api=api,
        sp_graph=sp_graph,
        lamb_grid=lamb_grid,
        lamb_q_grid=lamb_q_grid,
        config=config,
        alpha_fact=alpha_fact,
    )


def recommended_loo_fold_errors(
    *,
    genotypes: np.ndarray,
    sample_nodes: np.ndarray,
    graph: dict[str, np.ndarray],
    active_positions: np.ndarray,
    fold: int,
) -> np.ndarray:
    """Fit one observed-deme fold from the documented FEEMS v2 LOO grid."""

    config = recommended_loo_config()
    require_recommended_loo(config)
    inputs = _FEEMSInputs(genotypes, sample_nodes, graph, active_positions)
    api = _import_feems()
    sp_graph = _make_spatial_graph(inputs, api)
    n_folds = _node_cv_folds(sp_graph, config)
    if fold < 0 or fold >= n_folds:
        raise ValueError(f"FEEMS LOO fold must be in [0, {n_folds}), got {fold}")
    is_train = api.cv.setup_k_fold_cv(sp_graph, n_folds, random_state=config.random_state)
    _, errors = _fit_joint_cv_fold_on_graph(
        fold=fold,
        is_train=is_train[:, fold],
        api=api,
        sp_graph=sp_graph,
        lamb_grid=np.asarray(config.lambdas, dtype=np.float64),
        lamb_q_grid=np.asarray(config.lambdas_q, dtype=np.float64),
        config=config,
        alpha_fact=1.0,
    )
    return errors


def _fit_joint_cv_fold_worker(
    payload: tuple[int, np.ndarray],
) -> tuple[int, np.ndarray]:
    if _joint_cv_worker_context is None:
        raise RuntimeError("FEEMS CV worker was not initialized")
    inputs, lamb_grid, lamb_q_grid, config, alpha_fact = _joint_cv_worker_context
    fold, is_train = payload
    return _fit_joint_cv_fold(
        fold=fold,
        is_train=is_train,
        inputs=inputs,
        lamb_grid=lamb_grid,
        lamb_q_grid=lamb_q_grid,
        config=config,
        alpha_fact=alpha_fact,
    )


def _cv_fold_checkpoint_path(checkpoint_dir: Path, fold: int) -> Path:
    return checkpoint_dir / f"fold_{fold:04d}.npz"


def _load_cv_fold_checkpoint(
    path: Path,
    *,
    fold: int,
    checkpoint_key: str,
    expected_shape: tuple[int, int],
) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            stored_fold = int(np.asarray(data["fold"]).item())
            stored_key = str(np.asarray(data["checkpoint_key"]).item())
            errors = np.asarray(data["errors"], dtype=np.float64)
    except (OSError, KeyError, TypeError, ValueError):
        return None
    if stored_fold != fold or stored_key != checkpoint_key:
        return None
    if errors.shape != expected_shape or np.any(np.isinf(errors)):
        return None
    return errors


def write_cv_fold_checkpoint(
    path: Path,
    *,
    fold: int,
    checkpoint_key: str,
    errors: np.ndarray,
) -> None:
    """Atomically store one FEEMS CV fold for reuse or downstream aggregation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            fold=np.asarray(fold, dtype=np.int64),
            checkpoint_key=np.asarray(checkpoint_key),
            errors=np.asarray(errors, dtype=np.float64),
        )
    os.replace(temporary, path)


def _report_cv_progress(*, completed: int, total: int, reused: int) -> None:
    print(
        f"FEEMS LOO folds complete: {completed}/{total} ({reused} reused)",
        flush=True,
    )


def run_cv_joint_fold_parallel(
    *,
    genotypes: np.ndarray,
    sample_nodes: np.ndarray,
    graph: dict[str, np.ndarray],
    active_positions: np.ndarray,
    sp_graph: Any,
    lamb_grid: np.ndarray,
    lamb_q_grid: np.ndarray,
    config: FEEMSConfig,
    alpha_fact: float = 1.0,
    checkpoint_dir: Path | None = None,
    checkpoint_key: str | None = None,
) -> np.ndarray:
    """Run FEEMS joint CV with independent folds distributed over processes.

    This mirrors ``feems.cross_validation.run_cv_joint`` except for the outer
    fold loop.  Within each fold, lambda, lambda_q, and warm starts are traversed
    in FEEMS' original order.
    """

    if checkpoint_dir is not None and checkpoint_key is None:
        raise ValueError("checkpoint_key is required when checkpoint_dir is set")

    api = _import_feems()
    inputs = _FEEMSInputs(genotypes, sample_nodes, graph, active_positions)
    n_folds = _node_cv_folds(sp_graph, config)
    is_train = api.cv.setup_k_fold_cv(sp_graph, n_folds, random_state=config.random_state)
    jobs = min(config.parallel_cv_jobs, n_folds)
    lamb_grid = np.asarray(lamb_grid, dtype=np.float64)
    lamb_q_grid = np.asarray(lamb_q_grid, dtype=np.float64)
    payloads = [(fold, is_train[:, fold]) for fold in range(n_folds)]
    fold_shape = (lamb_q_grid.shape[0], lamb_grid.shape[0])
    cv_errors = np.full((n_folds, *fold_shape), np.nan, dtype=np.float64)
    pending_payloads: list[tuple[int, np.ndarray]] = []
    reused = 0
    for payload in payloads:
        fold = int(payload[0])
        cached = None
        if checkpoint_dir is not None:
            cached = _load_cv_fold_checkpoint(
                _cv_fold_checkpoint_path(checkpoint_dir, fold),
                fold=fold,
                checkpoint_key=str(checkpoint_key),
                expected_shape=fold_shape,
            )
        if cached is None:
            pending_payloads.append(payload)
        else:
            cv_errors[fold] = cached
            reused += 1

    completed = reused
    report_every = max(1, n_folds // 20)
    _report_cv_progress(completed=completed, total=n_folds, reused=reused)

    def record(result: tuple[int, np.ndarray]) -> None:
        nonlocal completed
        fold, errors = result
        errors = np.asarray(errors, dtype=np.float64)
        if errors.shape != fold_shape or np.any(np.isinf(errors)):
            raise RuntimeError(f"FEEMS CV fold {fold} returned invalid errors")
        cv_errors[int(fold)] = errors
        if checkpoint_dir is not None:
            write_cv_fold_checkpoint(
                _cv_fold_checkpoint_path(checkpoint_dir, int(fold)),
                fold=int(fold),
                checkpoint_key=str(checkpoint_key),
                errors=errors,
            )
        completed += 1
        if completed == n_folds or completed % report_every == 0:
            _report_cv_progress(completed=completed, total=n_folds, reused=reused)

    if jobs <= 1:
        for fold, train_mask in pending_payloads:
            record(
                _fit_joint_cv_fold(
                    fold=fold,
                    is_train=train_mask,
                    inputs=inputs,
                    lamb_grid=lamb_grid,
                    lamb_q_grid=lamb_q_grid,
                    config=config,
                    alpha_fact=float(alpha_fact),
                )
            )
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=jobs,
            mp_context=context,
            initializer=_initialize_joint_cv_fold_worker,
            initargs=(inputs, lamb_grid, lamb_q_grid, config, float(alpha_fact)),
        ) as executor:
            futures = [
                executor.submit(_fit_joint_cv_fold_worker, payload) for payload in pending_payloads
            ]
            for future in as_completed(futures):
                record(future.result())
    return cv_errors


def feems_edge_log_rates(sp_graph: Any, edges: np.ndarray) -> np.ndarray:
    weights: dict[tuple[int, int], float] = {}
    for left, right, weight in zip(
        sp_graph.nnz_idx[0],
        sp_graph.nnz_idx[1],
        sp_graph.w,
        strict=True,
    ):
        weights[(int(min(left, right)), int(max(left, right)))] = float(weight)
    out = []
    for left, right in np.asarray(edges, dtype=np.int32):
        key = (int(min(left, right)), int(max(left, right)))
        out.append(math.log(weights[key]))
    return np.asarray(out, dtype=np.float64)


def _n_dim_s2_init(sp_graph: Any, s2: np.ndarray | float) -> np.ndarray:
    values = np.asarray(s2, dtype=np.float64).reshape(-1)
    if values.size == 1:
        return np.full(len(sp_graph), float(values[0]), dtype=np.float64)
    return values.astype(np.float64, copy=True)


def _result(
    *,
    config: FEEMSConfig,
    sp_graph: Any,
    edges: np.ndarray,
    best_lambda: float,
    best_lambda_q: float,
    optimize_q: str,
    protocol: str,
    mean_cv: np.ndarray,
    scalar_s2: float | None = None,
    validation_n_snps: int | None = None,
) -> FEEMSResult:
    if config.cv_mode == "replicate_holdout":
        cv_folds = 1
        cv_mode = "replicate_holdout"
    else:
        cv_folds = (
            int(sp_graph.n_observed_nodes)
            if config.sklearn_cv_folds is None
            else config.sklearn_cv_folds
        )
        cv_mode = "leave_one_out" if config.sklearn_cv_folds is None else "k_fold"
    return FEEMSResult(
        edge_log_rates=feems_edge_log_rates(sp_graph, edges),
        best_lambda=best_lambda,
        best_lambda_q=best_lambda_q,
        cv_folds=cv_folds,
        cv_mode=cv_mode,
        optimize_q=optimize_q,
        protocol=protocol,
        n_snps_after_filter=int(sp_graph.n_snps),
        n_observed_nodes=int(sp_graph.n_observed_nodes),
        train_loss=float(sp_graph.train_loss),
        mean_cv_error=np.asarray(mean_cv, dtype=np.float64),
        scalar_s2=scalar_s2,
        validation_n_snps=validation_n_snps,
    )


def _official_cv_options(config: FEEMSConfig) -> dict[str, Any]:
    return {
        "n_folds": config.sklearn_cv_folds,
        "lb": config.lb,
        "ub": config.ub,
        "factr": config.factr,
        "random_state": config.random_state,
        "outer_verbose": False,
        "inner_verbose": False,
    }


def _replicate_holdout_validation_nll(
    *,
    api: Any,
    inputs: _FEEMSInputs,
    weights: np.ndarray,
    s2: np.ndarray | float,
) -> tuple[float, int]:
    sp_graph = _make_spatial_graph(inputs, api)
    sp_graph.comp_graph_laplacian(np.asarray(weights, dtype=np.float64))
    sp_graph.comp_precision(s2=_n_dim_s2_init(sp_graph, s2))
    objective = api.objective(sp_graph)
    objective.inv()
    score = objective.neg_log_lik() / max(int(sp_graph.n_snps), 1)
    return float(score), int(sp_graph.n_snps)


def _fit_node_cv(
    inputs: _FEEMSInputs,
    config: FEEMSConfig,
    mode: FEEMSMode,
    *,
    use_feems_final_defaults: bool = False,
    cv_checkpoint_dir: Path | None = None,
    cv_checkpoint_key: str | None = None,
) -> FEEMSResult:
    api = _import_feems()
    sp_graph = _make_spatial_graph(inputs, api)
    lamb_grid = np.asarray(config.lambdas, dtype=np.float64)

    if mode == "free_q":
        lamb_q_grid = np.asarray(config.lambdas_q, dtype=np.float64)
        if config.parallel_cv_jobs > 1 or cv_checkpoint_dir is not None:
            cv_errors = run_cv_joint_fold_parallel(
                genotypes=inputs.genotypes,
                sample_nodes=inputs.sample_nodes,
                graph=inputs.graph,
                active_positions=inputs.active_positions,
                sp_graph=sp_graph,
                lamb_grid=lamb_grid,
                lamb_q_grid=lamb_q_grid,
                config=config,
                checkpoint_dir=cv_checkpoint_dir,
                checkpoint_key=cv_checkpoint_key,
            )
        else:
            cv_errors = api.cv.run_cv_joint(
                sp_graph,
                lamb_grid=lamb_grid,
                lamb_q_grid=lamb_q_grid,
                **_official_cv_options(config),
            )
        mean_cv = np.nanmean(cv_errors, axis=0)
        best_q, best_w = np.unravel_index(int(np.nanargmin(mean_cv)), mean_cv.shape)
        best_lambda_q = float(lamb_q_grid[best_q])
        optimize_q = "n-dim"
        protocol = "v2_joint_lambda_lambda_q"
    elif mode == "fixed_q":
        cv_errors = api.cv.run_cv(
            sp_graph,
            lamb_grid=lamb_grid,
            **_official_cv_options(config),
        )
        mean_cv = np.nanmean(cv_errors, axis=(0, 2))
        best_w = int(np.nanargmin(mean_cv))
        best_lambda_q = math.nan
        optimize_q = None
        protocol = "v1_lambda_only_fixed_q"
    elif mode == "scalar_q":
        sp_graph.fit_null_model(verbose=False)
        w0 = np.asarray(sp_graph.w0, dtype=np.float64)
        s2 = float(np.asarray(sp_graph.s2).reshape(-1)[0])
        n_folds = _node_cv_folds(sp_graph, config)
        is_train = api.cv.setup_k_fold_cv(sp_graph, n_folds, random_state=config.random_state)
        alpha = 1.0 / float(w0.mean())
        cv_errors = np.full((n_folds, lamb_grid.shape[0]), np.nan, dtype=np.float64)
        for fold in range(n_folds):
            if sp_graph.factor is not None:
                sp_graph.factor = None
            train_graph, test_graph = api.cv.train_test_split(sp_graph, is_train[:, fold])
            w_init = w0
            s2_init = np.asarray([s2], dtype=np.float64)
            for index, lamb in enumerate(lamb_grid):
                try:
                    _fit_spatial_graph(
                        train_graph,
                        config,
                        lamb=lamb,
                        lamb_q=0.0,
                        optimize_q="1-dim",
                        w_init=w_init,
                        s2_init=s2_init,
                        alpha=alpha,
                        alpha_q=0.0,
                    )
                    _predicted, error = api.cv.predict_snps(sp_graph, train_graph, test_graph)
                    cv_errors[fold, index] = error
                    w_init = np.asarray(train_graph.w, dtype=np.float64).copy()
                    fitted_s2 = float(np.asarray(train_graph.s2).reshape(-1)[0])
                    s2_init = np.asarray([fitted_s2])
                except Exception:  # noqa: BLE001
                    cv_errors[fold, index] = np.nan
        mean_cv = np.nanmean(cv_errors, axis=0)
        best_w = int(np.nanargmin(mean_cv))
        best_lambda_q = 0.0
        optimize_q = "1-dim"
        protocol = "scalar_q_lambda_cv"
    else:
        raise ValueError(f"unknown FEEMS mode: {mode}")

    best_lambda = float(lamb_grid[best_w])
    if use_feems_final_defaults:
        if mode != "free_q":
            raise ValueError("documented FEEMS final refit is only defined for free_q")
        # This is the exact final call in FEEMS' v2 cross-validation notebook.
        # In particular, it restores SpatialGraph.fit's stricter factr=1e7.
        sp_graph.fit(lamb=best_lambda, lamb_q=best_lambda_q)
    else:
        _fit_spatial_graph(
            sp_graph,
            config,
            lamb=best_lambda,
            lamb_q=None if optimize_q is None else best_lambda_q,
            optimize_q=optimize_q,
        )
    scalar_s2 = float(np.asarray(sp_graph.s2).reshape(-1)[0]) if mode == "scalar_q" else None
    return _result(
        config=config,
        sp_graph=sp_graph,
        edges=inputs.edges,
        best_lambda=best_lambda,
        best_lambda_q=best_lambda_q,
        optimize_q="none" if optimize_q is None else optimize_q,
        protocol=protocol,
        mean_cv=mean_cv,
        scalar_s2=scalar_s2,
    )


def _fit_replicate_holdout(
    inputs: _FEEMSInputs,
    validation_genotypes: np.ndarray | None,
    config: FEEMSConfig,
) -> FEEMSResult:
    if validation_genotypes is None:
        raise ValueError("FEEMS replicate_holdout CV requires validation_genotypes")

    api = _import_feems()
    validation_inputs = _FEEMSInputs(
        np.asarray(validation_genotypes),
        inputs.sample_nodes,
        inputs.graph,
        inputs.active_positions,
    )
    sp_graph = _make_spatial_graph(inputs, api)
    sp_graph.fit_null_model(verbose=False)
    w0 = np.asarray(sp_graph.w0, dtype=np.float64).copy()
    s20 = _n_dim_s2_init(sp_graph, sp_graph.s2)
    alpha = 1.0 / float(np.mean(w0))
    alpha_q = 1.0 / float(np.mean(s20))
    lamb_grid = np.asarray(config.lambdas, dtype=np.float64)
    lamb_q_grid = np.asarray(config.lambdas_q, dtype=np.float64)
    validation_nll = np.full(
        (lamb_q_grid.shape[0], lamb_grid.shape[0]),
        np.nan,
        dtype=np.float64,
    )
    validation_n_snps = 0
    first_error: Exception | None = None
    next_lambda_w = w0
    next_lambda_s2 = s20
    for lamb_index, lamb in enumerate(lamb_grid):
        w_init = next_lambda_w.copy()
        s2_init = next_lambda_s2.copy()
        for lamb_q_index, lamb_q in enumerate(lamb_q_grid):
            try:
                _fit_spatial_graph(
                    sp_graph,
                    config,
                    lamb=lamb,
                    lamb_q=lamb_q,
                    optimize_q="n-dim",
                    w_init=w_init,
                    s2_init=s2_init,
                    alpha=alpha,
                    alpha_q=alpha_q,
                )
                score, validation_n_snps = _replicate_holdout_validation_nll(
                    api=api,
                    inputs=validation_inputs,
                    weights=np.asarray(sp_graph.w, dtype=np.float64),
                    s2=np.asarray(sp_graph.s2, dtype=np.float64),
                )
                validation_nll[lamb_q_index, lamb_index] = score
                w_init = np.asarray(sp_graph.w, dtype=np.float64).copy()
                s2_init = _n_dim_s2_init(sp_graph, sp_graph.s2)
                if lamb_q_index == 0:
                    next_lambda_w = w_init.copy()
                    next_lambda_s2 = s2_init.copy()
            except Exception as exc:  # noqa: BLE001
                if first_error is None:
                    first_error = exc
                validation_nll[lamb_q_index, lamb_index] = np.nan
    if not np.any(np.isfinite(validation_nll)):
        raise RuntimeError("all FEEMS replicate-holdout validation fits failed") from first_error

    best_q, best_w = np.unravel_index(int(np.nanargmin(validation_nll)), validation_nll.shape)
    best_lambda = float(lamb_grid[best_w])
    best_lambda_q = float(lamb_q_grid[best_q])
    final_graph = _make_spatial_graph(inputs, api)
    _fit_spatial_graph(
        final_graph,
        config,
        lamb=best_lambda,
        lamb_q=best_lambda_q,
        optimize_q="n-dim",
    )
    return _result(
        config=config,
        sp_graph=final_graph,
        edges=inputs.edges,
        best_lambda=best_lambda,
        best_lambda_q=best_lambda_q,
        optimize_q="n-dim",
        protocol="v2_replicate_holdout_lambda_lambda_q",
        mean_cv=validation_nll,
        validation_n_snps=validation_n_snps,
    )


def fit_feems(
    *,
    genotypes: np.ndarray,
    sample_nodes: np.ndarray,
    graph: dict[str, np.ndarray],
    active_positions: np.ndarray,
    config: FEEMSConfig | None = None,
    validation_genotypes: np.ndarray | None = None,
) -> FEEMSResult:
    config = FEEMSConfig() if config is None else config
    inputs = _FEEMSInputs(genotypes, sample_nodes, graph, active_positions)
    if config.cv_mode == "replicate_holdout":
        if config.mode != "free_q":
            raise ValueError("FEEMS replicate_holdout CV currently supports mode='free_q'")
        return _fit_replicate_holdout(inputs, validation_genotypes, config)
    if config.mode not in ("free_q", "fixed_q", "scalar_q"):
        raise ValueError(f"unknown FEEMS mode: {config.mode}")
    return _fit_node_cv(inputs, config, config.mode)


def fit_feems_free_q(
    *,
    genotypes: np.ndarray,
    sample_nodes: np.ndarray,
    graph: dict[str, np.ndarray],
    active_positions: np.ndarray,
    config: FEEMSConfig | None = None,
) -> FEEMSResult:
    config = FEEMSConfig(mode="free_q") if config is None else config
    inputs = _FEEMSInputs(genotypes, sample_nodes, graph, active_positions)
    return _fit_node_cv(inputs, config, "free_q")


def fit_feems_recommended_loo(
    *,
    genotypes: np.ndarray,
    sample_nodes: np.ndarray,
    graph: dict[str, np.ndarray],
    active_positions: np.ndarray,
    cv_jobs: int = 1,
    cv_checkpoint_dir: Path | None = None,
    cv_checkpoint_key: str | None = None,
) -> FEEMSResult:
    """Run the documented FEEMS v2 LOO selection and final refit."""

    config = recommended_loo_config(cv_jobs=cv_jobs)
    require_recommended_loo(config)
    inputs = _FEEMSInputs(genotypes, sample_nodes, graph, active_positions)
    return _fit_node_cv(
        inputs,
        config,
        "free_q",
        use_feems_final_defaults=True,
        cv_checkpoint_dir=cv_checkpoint_dir,
        cv_checkpoint_key=cv_checkpoint_key,
    )


def fit_feems_free_q_replicate_holdout(
    *,
    genotypes: np.ndarray,
    validation_genotypes: np.ndarray | None,
    sample_nodes: np.ndarray,
    graph: dict[str, np.ndarray],
    active_positions: np.ndarray,
    config: FEEMSConfig | None = None,
) -> FEEMSResult:
    """Tune FEEMS hyperparameters by independent SNP-replicate validation."""

    config = FEEMSConfig(mode="free_q", cv_mode="replicate_holdout") if config is None else config
    inputs = _FEEMSInputs(genotypes, sample_nodes, graph, active_positions)
    return _fit_replicate_holdout(inputs, validation_genotypes, config)


def fit_feems_fixed_q(
    *,
    genotypes: np.ndarray,
    sample_nodes: np.ndarray,
    graph: dict[str, np.ndarray],
    active_positions: np.ndarray,
    config: FEEMSConfig | None = None,
) -> FEEMSResult:
    config = FEEMSConfig(mode="fixed_q") if config is None else config
    inputs = _FEEMSInputs(genotypes, sample_nodes, graph, active_positions)
    return _fit_node_cv(inputs, config, "fixed_q")


def fit_feems_scalar_q(
    *,
    genotypes: np.ndarray,
    sample_nodes: np.ndarray,
    graph: dict[str, np.ndarray],
    active_positions: np.ndarray,
    config: FEEMSConfig | None = None,
) -> FEEMSResult:
    config = FEEMSConfig(mode="scalar_q") if config is None else config
    inputs = _FEEMSInputs(genotypes, sample_nodes, graph, active_positions)
    return _fit_node_cv(inputs, config, "scalar_q")
