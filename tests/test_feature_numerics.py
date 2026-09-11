"""Small numerical references for preprocessing, independent of trained weights."""

from __future__ import annotations

import numpy as np
import pytest
from beartype import beartype
from jaxtyping import jaxtyped

from dreems.neural import features, linearized


@pytest.mark.parametrize(
    "transform", [features.centered, features.scaled_log_support, features.absolute_log_support]
)
def test_array_helpers_keep_shape_bindings_local(transform) -> None:
    @jaxtyped(typechecker=beartype)
    def transform_edges_and_nodes():
        return transform(np.arange(6.0)), transform(np.arange(4.0))

    edges, nodes = transform_edges_and_nodes()
    assert edges.shape == (6,)
    assert nodes.shape == (4,)


@pytest.fixture
def observation() -> dict:
    rng = np.random.default_rng(1207)
    positions = np.asarray([(x, y) for y in range(5) for x in range(5)], dtype=float)
    edges = np.asarray(
        [
            (i, j)
            for i in range(25)
            for j in range(i + 1, 25)
            if np.linalg.norm(positions[i] - positions[j]) == 1.0
        ],
        dtype=np.int32,
    )
    sample_nodes = np.concatenate([np.arange(25), [0, 1, 1, 7, 12]]).astype(np.int32)
    factors = rng.normal(size=(30, 35))
    factors -= factors.mean(axis=0)
    return {
        "sample_covariance": factors @ factors.T / 35,
        "sample_nodes": sample_nodes,
        "node_coordinates": positions,
        "edges": edges,
        "effective_snps": 35,
    }


def dense_line_laplacian(nodes: int, edges: np.ndarray) -> np.ndarray:
    """Each common endpoint contributes one unit of edge-to-edge coupling."""
    result = np.zeros((len(edges), len(edges)))
    for node in range(nodes):
        incident = [index for index, edge in enumerate(edges) for end in edge if end == node]
        for left in incident:
            for right in incident:
                if left != right:
                    result[left, left] += 1.0
                    result[left, right] -= 1.0
    return result


@pytest.mark.parametrize(
    "pairs",
    [[], [[0, 1]], [[0, 1], [0, 2], [0, 3]], [[0, 1], [1, 0], [1, 2]], [[0, 0], [0, 1], [1, 2]]],
)
def test_line_graph_incidence_matches_endpoint_reference(pairs: list) -> None:
    edges = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
    actual = features.line_graph_laplacian(4, edges).toarray()
    np.testing.assert_array_equal(actual, dense_line_laplacian(4, edges))


@pytest.mark.parametrize("batch_size", [1, 17, 256, 400])
@pytest.mark.parametrize("pair_weight_power", [0.0, 0.5])
def test_all_pairs_matches_dense_diagonal_normal_equations(
    observation: dict,
    batch_size: int,
    pair_weight_power: float,
) -> None:
    data = observation
    edges = data["edges"]
    nodes = len(data["node_coordinates"])
    samples = data["sample_nodes"]
    counts = np.bincount(samples, minlength=nodes)
    averaging = np.zeros((nodes, len(samples)))
    averaging[samples, np.arange(len(samples))] = 1.0 / counts[samples]
    covariance = averaging @ data["sample_covariance"] @ averaging.T
    left, right = np.triu_indices(nodes, k=1)
    observed = np.log(
        np.diag(covariance)[left] + np.diag(covariance)[right] - 2.0 * covariance[left, right]
    )
    observed -= observed.mean()

    incidence = np.zeros((len(edges), nodes))
    incidence[np.arange(len(edges)), edges[:, 0]] = 1.0
    incidence[np.arange(len(edges)), edges[:, 1]] = -1.0
    laplacian = incidence.T @ incidence
    inverse = np.zeros((nodes, nodes))
    inverse[:-1, :-1] = np.linalg.inv(laplacian[:-1, :-1])
    potentials = inverse[:, left] - inverse[:, right]
    resistance = potentials[left, np.arange(len(left))] - potentials[right, np.arange(len(left))]
    baseline = np.log(resistance)
    baseline -= baseline.mean()
    residual = observed - baseline
    sensitivity = -((incidence @ potentials) ** 2) / resistance[None, :]
    weights = (counts[left] * counts[right]).astype(float) ** pair_weight_power
    weights /= weights.mean()
    information = (sensitivity**2 * weights).sum(axis=1)
    ridge = 0.001 * np.median(information[information > 0.0])
    system = np.diag(information + ridge) + 0.01 * dense_line_laplacian(nodes, edges)
    expected = np.linalg.solve(system, sensitivity @ (weights * residual))
    expected -= expected.mean()

    actual = linearized.all_pair_adjoint_conductance_initializer(
        **data,
        batch_size=batch_size,
        pair_weight_power=pair_weight_power,
        output_sd=None,
    )
    assert actual.diagnostics["observed_pairs"] == 300  # More than the old 256-pair cap.
    np.testing.assert_allclose(actual.observed_pair_log_resistance, observed, atol=1e-12)
    np.testing.assert_allclose(actual.baseline_pair_log_resistance, baseline, atol=1e-12)
    np.testing.assert_allclose(actual.edge_information, information, atol=1e-12)
    np.testing.assert_allclose(actual.edge_log_conductance, expected, atol=1e-11)


@pytest.mark.parametrize("output_sd", [0.0, 0.5, 1.0])
def test_initializer_amplitude_setting(observation: dict, output_sd: float) -> None:
    result = linearized.all_pair_adjoint_conductance_initializer(**observation, output_sd=output_sd)
    np.testing.assert_allclose(result.edge_log_conductance.std(), output_sd, atol=1e-14)
    np.testing.assert_allclose(result.edge_log_conductance.mean(), 0.0, atol=1e-14)


@pytest.mark.parametrize(
    "override, message",
    [
        ({"smoothness": -1.0}, "smoothness"),
        ({"ridge_scale": np.nan}, "ridge_scale"),
        ({"output_sd": np.inf}, "output_sd"),
        ({"pair_weight_power": np.inf}, "pair_weight_power"),
        ({"batch_size": 0}, "batch_size"),
        ({"effective_snps": 1}, "effective_snps"),
        ({"edges": np.asarray([[0, 25]], dtype=np.int64)}, "out-of-range"),
        ({"edges": np.asarray([[0, 1]], dtype=np.int32)}, "connected"),
    ],
)
def test_initializer_rejects_invalid_inputs(
    observation: dict, override: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        linearized.all_pair_adjoint_conductance_initializer(**(observation | override))


@pytest.mark.parametrize("failed_solve", [1, 2])
def test_nonfinite_sparse_solve_is_never_a_valid_feature(
    observation: dict,
    monkeypatch,
    failed_solve: int,
) -> None:
    original = linearized.spla.spsolve
    calls = 0

    def broken_solve(matrix, rhs):
        nonlocal calls
        calls += 1
        if calls == failed_solve:
            return np.full_like(rhs, np.nan)
        return original(matrix, rhs)

    monkeypatch.setattr(linearized.spla, "spsolve", broken_solve)
    with pytest.raises(ValueError, match="non-finite"):
        linearized.all_pair_adjoint_conductance_initializer(**observation)

    example = {
        **observation,
        "active_positions": observation["node_coordinates"],
        "graph": {"edges": observation["edges"]},
    }
    prepared = [{"edges": observation["edges"], "edge_x": np.zeros((40, 1), dtype=np.float32)}]
    metadata = {
        "edge_dim": 3,
        "edge_conditioner": {
            "kind": "linearized_resistance",
            "method": "all_pair_diagonal_adjoint",
        },
    }
    calls = 0
    with pytest.raises(ValueError, match="conditioner failed.*non-finite"):
        features.append_edge_conditioner_features_for_checkpoint(prepared, [example], metadata, {})
    calls = 0
    [fallback] = features.append_edge_conditioner_features_for_checkpoint(
        prepared,
        [example],
        metadata,
        {},
        fail_on_error=False,
    )
    np.testing.assert_array_equal(fallback["linearized_resistance_valid"], np.zeros(40))
    np.testing.assert_array_equal(fallback["edge_x"][:, -2:], np.zeros((40, 2)))
    assert np.isnan(fallback["linearized_resistance_edge_log"]).all()


def test_covariance_fallback_does_not_turn_negative_modes_positive(monkeypatch) -> None:
    def failed_eigh(_matrix):
        raise np.linalg.LinAlgError("forced failure")

    monkeypatch.setattr(np.linalg, "eigh", failed_eigh)
    covariance = np.diag([-2.0, 0.0, 3.0])
    values, vectors = features.symmetric_psd_eigh(covariance)
    np.testing.assert_allclose(values, [-2.0, 0.0, 3.0])
    np.testing.assert_allclose((vectors * values) @ vectors.T, covariance)
    factors = features._covariance_factors(covariance, 10)
    np.testing.assert_allclose(factors @ factors.T / 10, np.diag([0.0, 0.0, 3.0]))


@pytest.mark.parametrize("with_edges", [False, True])
def test_diffusion_bands_reconstruct_the_standardized_signal(with_edges: bool) -> None:
    positions = np.asarray([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    edges = np.asarray([[0, 1], [1, 2], [2, 3]] if with_edges else [], dtype=np.int32).reshape(
        -1, 2
    )
    signal = np.arange(12, dtype=float).reshape(4, 3) ** 2
    actual = features.diffusion_band_summaries(positions, edges, signal)
    blocks = actual["node_bands"].reshape(4, 6, 3)
    np.testing.assert_allclose(blocks.sum(axis=1), features.standardized_signal_columns(signal))
    flipped = features.diffusion_band_summaries(positions, edges[:, ::-1], signal)
    for name in actual:
        np.testing.assert_array_equal(actual[name], flipped[name])
    if not with_edges:
        np.testing.assert_array_equal(blocks[:, :-1], np.zeros((4, 5, 3)))


@pytest.mark.parametrize("times", [(-1.0,), (np.nan,), (0.5, 0.25), (0.5, 0.5)])
def test_diffusion_rejects_invalid_times(observation: dict, times: tuple) -> None:
    with pytest.raises(ValueError, match="diffusion times"):
        features.diffusion_band_summaries(
            observation["node_coordinates"],
            observation["edges"],
            np.ones((25, 1)),
            times=times,
        )


def test_unsigned_coordinates_do_not_wrap_edge_lengths() -> None:
    positions = np.asarray([[250, 0], [0, 0], [125, 0]], dtype=np.uint8)
    edges = np.asarray([[0, 1], [1, 2]], dtype=np.int32)
    result = features.geometry_conductance(positions, edges, length_weighted=True)
    np.testing.assert_array_equal(result, [0.75, 1.5])


@pytest.mark.parametrize("field", ["edge_x", "y", "q", "edge_initializer"])
def test_padding_rejects_rows_that_numpy_would_broadcast(field: str) -> None:
    item = {
        "node_x": np.zeros((3, 4)),
        "edge_x": np.zeros((2, 3)),
        "edges": np.asarray([[0, 1], [1, 2]], dtype=np.int32),
        "y": np.zeros(2),
        "q": np.zeros(3),
        "edge_initializer": np.zeros(2),
    }
    item[field] = item[field][:1]
    with pytest.raises(ValueError, match=f"{field} must have shape"):
        features.pad_examples([item])
