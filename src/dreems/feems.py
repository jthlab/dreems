"""Fit the official FEEMS implementation on DREEMS inputs.

Inputs use zero-based node IDs and complete diploid dosages. This adapter
converts edge IDs to FEEMS' one-based format and returns rates in input edge
order. Only LOO folds run independently; the penalty traversal and warm starts
inside each fold follow the pinned FEEMS implementation.
"""

# ruff: noqa: F722, F821, UP037

from __future__ import annotations

import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np
from beartype import beartype
from jaxtyping import Bool, Float, Int, Real, Shaped, jaxtyped

from dreems.utils.validation import (
    _integer_array,
    _real_array,
    validate_diploid_dosages,
    validate_genotype_arrays,
)
from dreems.utils.workflow_io import _atomic_output

FEEMSMode = Literal["free_q", "fixed_q", "scalar_q"]
FEEMSCVMode = Literal["node_cv", "replicate_holdout"]

JointCVErrors: TypeAlias = Float[np.ndarray, "lambda_q lambda_w"]


class FEEMSGraph(TypedDict):
    """Graph field read by the adapter; node positions are supplied separately."""

    edges: Shaped[np.ndarray, "edges 2"]


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


@jaxtyped(typechecker=beartype)
def _import_feems() -> SimpleNamespace:
    """Load the optional solver only when a FEEMS operation is requested."""
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
    Positive fold counts must be at least two. Penalty order is preserved
    because it determines warm starts. Bounds are on positive conductances;
    the adapter takes logs only for the low-level fit call. `cv_jobs` counts
    independent fold processes.
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

    @jaxtyped(typechecker=beartype)
    def __post_init__(self) -> None:
        """Reject invalid settings without changing field values or defaults."""
        if self.mode not in ("free_q", "fixed_q", "scalar_q"):
            raise ValueError(f"unknown FEEMS mode: {self.mode}")
        if self.cv_mode not in ("node_cv", "replicate_holdout"):
            raise ValueError(f"unknown FEEMS CV mode: {self.cv_mode}")
        for name in ("cv_folds", "maxiter", "random_state", "cv_jobs"):
            count = getattr(self, name)
            if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral):
                raise ValueError(f"{name} must be a scalar integer")
        if self.cv_folds == 1:
            raise ValueError("cv_folds must be at least two, or nonpositive for LOO")
        if self.maxiter < 1 or self.cv_jobs < 1:
            raise ValueError("maxiter and cv_jobs must be positive")
        if not 0 <= self.random_state <= np.iinfo(np.uint32).max:
            raise ValueError("random_state must be in [0, 2**32 - 1]")
        for name in ("factr", "lb", "ub"):
            value = _real_array(getattr(self, name), name=name)
            if value.ndim != 0 or value <= 0:
                raise ValueError(f"{name} must be a positive scalar")
        if self.lb >= self.ub:
            raise ValueError("lb must be smaller than ub")
        _penalty_grid(np.asarray(self.lambdas), name="lambdas")
        _penalty_grid(np.asarray(self.lambdas_q), name="lambdas_q")

    @property
    @jaxtyped(typechecker=beartype)
    def sklearn_cv_folds(self) -> int | None:
        """Translate the retained nonpositive LOO convention to FEEMS' None."""
        return None if self.cv_folds <= 0 else int(self.cv_folds)

    @property
    @jaxtyped(typechecker=beartype)
    def parallel_cv_jobs(self) -> int:
        """Return the validated number of fold processes."""
        return int(self.cv_jobs)


@jaxtyped(typechecker=beartype)
def recommended_loo_config(*, cv_jobs: int = 1) -> FEEMSConfig:
    """Return the FEEMS v2 cross-validation recipe used in paper comparisons.

    FEEMS' cross-validation notebook jointly tunes the edge and node-variance
    penalties. Leaving ``n_folds`` unset invokes its default: one held-out
    observed deme per fold. ``cv_jobs`` only controls parallel execution of
    those independent folds.
    """

    return FEEMSConfig(cv_jobs=cv_jobs)


@jaxtyped(typechecker=beartype)
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
    """Uncentered edge log-conductances and the recorded selection protocol."""

    edge_log_rates: Float[np.ndarray, "edges"]
    best_lambda: float
    best_lambda_q: float
    cv_folds: int
    cv_mode: str
    optimize_q: str
    protocol: str
    n_snps_after_filter: int
    n_observed_nodes: int
    train_loss: float
    mean_cv_error: JointCVErrors | Float[np.ndarray, "lambda_w"]
    scalar_s2: float | None = None
    validation_n_snps: int | None = None

    @jaxtyped(typechecker=beartype)
    def info_dict(self) -> dict[str, object]:
        """Export fit statistics and the selection protocol."""
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
    """Keep caller edge order and validate assignments before calling FEEMS."""

    genotypes: Shaped[np.ndarray, "samples snps"]
    sample_nodes: Shaped[np.ndarray, "samples"]
    graph: FEEMSGraph
    active_positions: Shaped[np.ndarray, "nodes 2"]

    @jaxtyped(typechecker=beartype)
    def __post_init__(self) -> None:
        """Validate before any lossy coordinate or node-index conversion."""
        positions = _real_array(self.active_positions, name="active_positions")
        validate_genotype_arrays(
            positions=positions,
            edges=self.edges,
            sample_nodes=np.asarray(self.sample_nodes),
            genotypes=np.asarray(self.genotypes),
        )
        # FEEMS assigns each sample to the nearest node coordinate.
        if np.unique(positions, axis=0).shape[0] != positions.shape[0]:
            raise ValueError("FEEMS node coordinates must be unique to preserve sample assignments")

    @property
    @jaxtyped(typechecker=beartype)
    def edges(self) -> Int[np.ndarray, "edges 2"]:
        """Read zero-based endpoints without truncating fractional IDs."""
        return _integer_array(np.asarray(self.graph["edges"]), name="edges", dtype=np.int32)


@jaxtyped(typechecker=beartype)
def polymorphic_snp_mask(genotypes: Shaped[np.ndarray, "samples snps"]) -> Bool[np.ndarray, "snps"]:
    """Select segregating SNPs from validated, complete diploid dosages."""
    genotypes = np.asarray(genotypes)
    validate_diploid_dosages(genotypes)
    allele_sum = genotypes.sum(axis=0)
    return (allele_sum > 0.0) & (allele_sum < 2.0 * genotypes.shape[0])


@jaxtyped(typechecker=beartype)
def make_spatial_graph(
    *,
    genotypes: Shaped[np.ndarray, "samples snps"],
    sample_nodes: Shaped[np.ndarray, "sample_assignments"],
    graph: FEEMSGraph,
    active_positions: Shaped[np.ndarray, "nodes 2"],
) -> Any:
    """Construct FEEMS' graph, dropping monomorphic SNPs and scaling by heterozygosity.

    Missing calls must be resolved upstream. This function does not impose a
    MAF threshold or perform LD pruning; those are dataset QC decisions.
    """
    return _make_spatial_graph(
        _FEEMSInputs(genotypes, sample_nodes, graph, active_positions),
        _import_feems(),
    )


@jaxtyped(typechecker=beartype)
def _make_spatial_graph(inputs: _FEEMSInputs, api: Any) -> Any:
    """Translate validated zero-based inputs to FEEMS' graph constructor."""
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


@jaxtyped(typechecker=beartype)
def _node_cv_folds(sp_graph: Any, config: FEEMSConfig) -> int:
    """Resolve LOO and reject fold counts that exceed observed demes."""
    count = sp_graph.n_observed_nodes if config.sklearn_cv_folds is None else config.cv_folds
    if not 2 <= count <= sp_graph.n_observed_nodes:
        raise ValueError("FEEMS CV needs between two and the number of observed demes folds")
    return int(count)


@jaxtyped(typechecker=beartype)
def _penalty_grid(
    values: Shaped[np.ndarray, "*input_axes"], *, name: str
) -> Float[np.ndarray, "penalties"]:
    """Keep the caller's penalty order, requiring a nonempty nonnegative grid."""
    values = _real_array(values, name=name)
    if values.ndim != 1 or values.size == 0 or np.any(values < 0):
        raise ValueError(f"{name} must be a nonempty vector of nonnegative penalties")
    return values


@jaxtyped(typechecker=beartype)
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
    """Pass common bounds and convergence settings to a low-level FEEMS fit."""
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


_joint_cv_worker_context: (
    tuple[
        _FEEMSInputs,
        Float[np.ndarray, "lambda_w"],
        Float[np.ndarray, "lambda_q"],
        FEEMSConfig,
        float,
    ]
    | None
) = None


@jaxtyped(typechecker=beartype)
def _initialize_joint_cv_fold_worker(
    inputs: _FEEMSInputs,
    lamb_grid: Shaped[np.ndarray, "lambda_w"],
    lamb_q_grid: Shaped[np.ndarray, "lambda_q"],
    config: FEEMSConfig,
    alpha_fact: float,
) -> None:
    """Send common data once to each spawned CV process."""
    global _joint_cv_worker_context
    _joint_cv_worker_context = (inputs, lamb_grid, lamb_q_grid, config, alpha_fact)


@jaxtyped(typechecker=beartype)
def _fit_joint_cv_fold_on_graph(
    *,
    fold: int,
    is_train: Bool[np.ndarray, "observed"],
    api: Any,
    sp_graph: Any,
    lamb_grid: Shaped[np.ndarray, "lambda_w"],
    lamb_q_grid: Shaped[np.ndarray, "lambda_q"],
    config: FEEMSConfig,
    alpha_fact: float,
) -> tuple[int, JointCVErrors]:
    """Evaluate one fold with FEEMS' exact nested penalty and warm-start order."""
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


@jaxtyped(typechecker=beartype)
def _fit_joint_cv_fold(
    *,
    fold: int,
    is_train: Bool[np.ndarray, "observed"],
    inputs: _FEEMSInputs,
    lamb_grid: Shaped[np.ndarray, "lambda_w"],
    lamb_q_grid: Shaped[np.ndarray, "lambda_q"],
    config: FEEMSConfig,
    alpha_fact: float,
) -> tuple[int, JointCVErrors]:
    """Construct a fresh graph so a fold cannot inherit another fold's fit."""
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


@jaxtyped(typechecker=beartype)
def recommended_loo_fold_errors(
    *,
    genotypes: Shaped[np.ndarray, "samples snps"],
    sample_nodes: Shaped[np.ndarray, "sample_assignments"],
    graph: FEEMSGraph,
    active_positions: Shaped[np.ndarray, "nodes 2"],
    fold: int,
) -> JointCVErrors:
    """Fit one observed-deme fold from the documented FEEMS v2 LOO grid."""

    config = recommended_loo_config()
    _integer_array(np.asarray(fold), name="fold", dtype=np.int64)
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


@jaxtyped(typechecker=beartype)
def _fit_joint_cv_fold_worker(
    payload: tuple[int, Bool[np.ndarray, "observed"]],
) -> tuple[int, JointCVErrors]:
    """Read process-local inputs and fit the submitted held-out mask."""
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


@jaxtyped(typechecker=beartype)
def _cv_fold_checkpoint_path(checkpoint_dir: Path, fold: int) -> Path:
    """Use the established fold filenames for resumable LOO runs."""
    return checkpoint_dir / f"fold_{fold:04d}.npz"


@jaxtyped(typechecker=beartype)
def _load_cv_fold_checkpoint(
    path: Path,
    *,
    fold: int,
    checkpoint_key: str,
    expected_shape: tuple[int, int],
) -> JointCVErrors | None:
    """Reuse matching fold errors; malformed or stale files are cache misses."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            stored_fold = _integer_array(data["fold"], name="fold", dtype=np.int64)
            stored_key = str(np.asarray(data["checkpoint_key"]).item())
            errors = data["errors"]
    except (OSError, KeyError, TypeError, ValueError):
        return None
    if stored_fold.ndim != 0 or stored_fold != fold or stored_key != checkpoint_key:
        return None
    if errors.dtype.kind not in "iuf" or errors.shape != expected_shape or np.any(np.isinf(errors)):
        return None
    return errors.astype(np.float64, copy=False)


@jaxtyped(typechecker=beartype)
def write_cv_fold_checkpoint(
    path: Path,
    *,
    fold: int,
    checkpoint_key: str,
    errors: Shaped[np.ndarray, "*input_axes"],
) -> None:
    """Atomically store one FEEMS CV fold for reuse or downstream aggregation."""

    count = _integer_array(np.asarray(fold), name="fold", dtype=np.int64)
    if count.ndim != 0 or count < 0:
        raise ValueError("fold must be a nonnegative scalar integer")
    if not checkpoint_key:
        raise ValueError("checkpoint_key must not be empty")
    if (
        errors.dtype.kind not in "iuf"
        or errors.ndim != 2
        or 0 in errors.shape
        or np.any(np.isinf(errors))
    ):
        raise ValueError("errors must be a nonempty real matrix; only NaN may mark a failed fit")
    with _atomic_output(path) as stream:
        np.savez(
            stream,
            fold=np.asarray(fold, dtype=np.int64),
            checkpoint_key=np.asarray(checkpoint_key),
            errors=np.asarray(errors, dtype=np.float64),
        )


@jaxtyped(typechecker=beartype)
def _report_cv_progress(*, completed: int, total: int, reused: int) -> None:
    """Report completed and reused folds."""
    print(
        f"FEEMS LOO folds complete: {completed}/{total} ({reused} reused)",
        flush=True,
    )


@jaxtyped(typechecker=beartype)
def run_cv_joint_fold_parallel(
    *,
    genotypes: Shaped[np.ndarray, "samples snps"],
    sample_nodes: Shaped[np.ndarray, "sample_assignments"],
    graph: FEEMSGraph,
    active_positions: Shaped[np.ndarray, "nodes 2"],
    sp_graph: Any,
    lamb_grid: Shaped[np.ndarray, "lambda_w"],
    lamb_q_grid: Shaped[np.ndarray, "lambda_q"],
    config: FEEMSConfig,
    alpha_fact: float = 1.0,
    checkpoint_dir: Path | None = None,
    checkpoint_key: str | None = None,
) -> Float[np.ndarray, "folds lambda_q lambda_w"]:
    """Run FEEMS joint CV with independent folds distributed over processes.

    This mirrors ``feems.cross_validation.run_cv_joint`` except for the outer
    fold loop.  Within each fold, lambda, lambda_q, and warm starts are traversed
    in FEEMS' original order.
    """

    if checkpoint_dir is not None and checkpoint_key is None:
        raise ValueError("checkpoint_key is required when checkpoint_dir is set")
    if not np.isfinite(alpha_fact) or alpha_fact <= 0:
        raise ValueError("alpha_fact must be finite and positive")

    api = _import_feems()
    inputs = _FEEMSInputs(genotypes, sample_nodes, graph, active_positions)
    n_folds = _node_cv_folds(sp_graph, config)
    is_train = api.cv.setup_k_fold_cv(sp_graph, n_folds, random_state=config.random_state)
    jobs = min(config.parallel_cv_jobs, n_folds)
    lamb_grid = _penalty_grid(lamb_grid, name="lamb_grid")
    lamb_q_grid = _penalty_grid(lamb_q_grid, name="lamb_q_grid")
    payloads = [(fold, is_train[:, fold]) for fold in range(n_folds)]
    fold_shape = (lamb_q_grid.shape[0], lamb_grid.shape[0])
    cv_errors = np.full((n_folds, *fold_shape), np.nan, dtype=np.float64)
    pending_payloads: list[tuple[int, Bool[np.ndarray, "observed"]]] = []
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

    def record(result: tuple[int, JointCVErrors]) -> None:
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


@jaxtyped(typechecker=beartype)
def feems_edge_log_rates(
    sp_graph: Any, edges: Shaped[np.ndarray, "edges 2"]
) -> Float[np.ndarray, "edges"]:
    """Map FEEMS' internal edge order back to the caller's undirected edges."""
    edges = _integer_array(edges, name="edges", dtype=np.int32)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (edges, 2)")
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


@jaxtyped(typechecker=beartype)
def _n_dim_s2_init(
    sp_graph: Any, s2: Real[np.ndarray, "*variance_axes"] | float
) -> Float[np.ndarray, "nodes"]:
    """Expand a null-model scalar variance to FEEMS' node-variance vector."""
    values = np.asarray(s2, dtype=np.float64).reshape(-1)
    if values.size == 1:
        return np.full(len(sp_graph), float(values[0]), dtype=np.float64)
    return values.astype(np.float64, copy=True)


@jaxtyped(typechecker=beartype)
def _result(
    *,
    config: FEEMSConfig,
    sp_graph: Any,
    edges: Shaped[np.ndarray, "edges 2"],
    best_lambda: float,
    best_lambda_q: float,
    optimize_q: str,
    protocol: str,
    mean_cv: JointCVErrors | Float[np.ndarray, "lambda_w"],
    scalar_s2: float | None = None,
    validation_n_snps: int | None = None,
) -> FEEMSResult:
    """Record the fit without centering rates or changing the report fields."""
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


@jaxtyped(typechecker=beartype)
def _official_cv_options(config: FEEMSConfig) -> dict[str, Any]:
    """Use the same keyword arguments for official fixed- and free-q CV."""
    return {
        "n_folds": config.sklearn_cv_folds,
        "lb": config.lb,
        "ub": config.ub,
        "factr": config.factr,
        "random_state": config.random_state,
        "outer_verbose": False,
        "inner_verbose": False,
    }


@jaxtyped(typechecker=beartype)
def _replicate_holdout_validation_nll(
    *,
    api: Any,
    inputs: _FEEMSInputs,
    weights: Real[np.ndarray, "edges"],
    s2: Real[np.ndarray, "*variance_axes"] | float,
) -> tuple[float, int]:
    """Score fitted parameters on an independent genotype replicate, per SNP."""
    sp_graph = _make_spatial_graph(inputs, api)
    sp_graph.comp_graph_laplacian(np.asarray(weights, dtype=np.float64))
    sp_graph.comp_precision(s2=_n_dim_s2_init(sp_graph, s2))
    objective = api.objective(sp_graph)
    objective.inv()
    score = objective.neg_log_lik() / max(int(sp_graph.n_snps), 1)
    return float(score), int(sp_graph.n_snps)


@jaxtyped(typechecker=beartype)
def _fit_node_cv(
    inputs: _FEEMSInputs,
    config: FEEMSConfig,
    mode: FEEMSMode,
    *,
    use_feems_final_defaults: bool = False,
    cv_checkpoint_dir: Path | None = None,
    cv_checkpoint_key: str | None = None,
) -> FEEMSResult:
    """Select penalties by held-out demes and refit all samples once."""
    config = replace(config, mode=mode, cv_mode="node_cv")
    api = _import_feems()
    sp_graph = _make_spatial_graph(inputs, api)
    _node_cv_folds(sp_graph, config)
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


@jaxtyped(typechecker=beartype)
def _fit_replicate_holdout(
    inputs: _FEEMSInputs,
    validation_genotypes: Shaped[np.ndarray, "samples validation_snps"] | None,
    config: FEEMSConfig,
) -> FEEMSResult:
    """Select penalties on independent SNPs with the same sample-node order."""
    config = replace(config, mode="free_q", cv_mode="replicate_holdout")
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


@jaxtyped(typechecker=beartype)
def fit_feems(
    *,
    genotypes: Shaped[np.ndarray, "samples snps"],
    sample_nodes: Shaped[np.ndarray, "sample_assignments"],
    graph: FEEMSGraph,
    active_positions: Shaped[np.ndarray, "nodes 2"],
    config: FEEMSConfig | None = None,
    validation_genotypes: Shaped[np.ndarray, "validation_samples validation_snps"] | None = None,
) -> FEEMSResult:
    """Fit the requested q model, using deme CV or an independent SNP replicate."""
    config = FEEMSConfig() if config is None else config
    inputs = _FEEMSInputs(genotypes, sample_nodes, graph, active_positions)
    if config.cv_mode == "replicate_holdout":
        if config.mode != "free_q":
            raise ValueError("FEEMS replicate_holdout CV currently supports mode='free_q'")
        return _fit_replicate_holdout(inputs, validation_genotypes, config)
    if config.mode not in ("free_q", "fixed_q", "scalar_q"):
        raise ValueError(f"unknown FEEMS mode: {config.mode}")
    return _fit_node_cv(inputs, config, config.mode)


@jaxtyped(typechecker=beartype)
def fit_feems_recommended_loo(
    *,
    genotypes: Shaped[np.ndarray, "samples snps"],
    sample_nodes: Shaped[np.ndarray, "sample_assignments"],
    graph: FEEMSGraph,
    active_positions: Shaped[np.ndarray, "nodes 2"],
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
