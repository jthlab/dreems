from __future__ import annotations

import numpy as np
import pytest

from dreems.calibration import SDCalibration, candidates, summarize


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


@pytest.mark.parametrize("floor", [0.0, -1.0, np.nan, np.inf, True])
def test_invalid_floor_is_rejected_before_fitting(floor) -> None:
    with pytest.raises(ValueError, match="floor"):
        SDCalibration.from_squared_errors(np.ones(2), np.ones(2), floor=floor)


@pytest.mark.parametrize(
    ("errors", "sd"),
    [
        ([], []),
        ([np.nan], [1]),
        ([np.inf], [1]),
        ([1], [np.nan]),
        ([1], [0]),
        ([1], [-1]),
        ([1], [np.inf]),
        ([1, 2], [1]),
        ([1e300], [1e-300]),
    ],
)
def test_invalid_observations_are_not_dropped(errors, sd) -> None:
    error, raw = np.asarray(errors), np.asarray(sd)
    with pytest.raises(ValueError):
        SDCalibration.from_squared_errors(error, raw)
    with pytest.raises(ValueError):
        SDCalibration.from_graphs([error], [raw], [0.1])
    with pytest.raises(ValueError):
        summarize([error], [raw], np.asarray([0.1]), SDCalibration())


def test_small_sds_are_floored_not_excluded() -> None:
    fitted = SDCalibration.from_squared_errors(
        np.asarray([2.0, 1.0]), np.asarray([1e-20, 1.0]), floor=1.0
    )
    assert fitted.global_scale == pytest.approx(np.sqrt(2.5))


@pytest.mark.parametrize("density", [np.nan, np.inf, -0.1, 1.1])
def test_invalid_density_fails_for_every_method(density) -> None:
    with pytest.raises(ValueError, match="sampled_node_fraction"):
        SDCalibration.from_graphs([np.ones(2)], [np.ones(2)], [density])
    for fitted in candidates([np.ones(2)], [np.ones(2)], np.asarray([0.1])).values():
        with pytest.raises(ValueError, match="sampled_node_fraction"):
            fitted.scale_for_density(density)


@pytest.mark.parametrize("breaks", [(0.5, 0.1), (0.1, 0.1), (0.0,), (1.0,), (np.nan,)])
def test_invalid_breaks_do_not_bypass_validation_on_empty_data(breaks) -> None:
    with pytest.raises(ValueError, match="density_breaks"):
        SDCalibration.from_graphs([], [], [], density_breaks=breaks)


def test_empty_or_misaligned_graph_collections_fail() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        SDCalibration.from_graphs([], [], [])
    with pytest.raises(ValueError, match="align"):
        SDCalibration.from_graphs([np.ones(2)], [], [0.1])
    with pytest.raises(ValueError, match="nonempty"):
        summarize([], [], np.asarray([]), SDCalibration())


@pytest.mark.parametrize("sd", [np.asarray([-1.0]), np.asarray([np.nan]), np.asarray([np.inf])])
def test_invalid_sds_cannot_be_reported(sd) -> None:
    with pytest.raises(ValueError, match="raw_sd"):
        SDCalibration().apply(sd)


def test_scalar_and_numpy_numeric_inputs_still_work() -> None:
    fitted = SDCalibration.from_graphs(
        [np.asarray(2.0)],
        [np.asarray(1.0)],
        np.asarray([0.1]),
        density_breaks=np.asarray([0.1, 0.25]),
        density_shrinkage=0,
        floor=np.float32(1e-8),
    )
    assert fitted.scale_for_density(np.float32(0.05)) == 2.0
    assert fitted.apply(np.asarray(1.0), sampled_node_fraction=0) == 2.0
    assert SDCalibration().apply(np.asarray([0.0])) == 0.0


def test_metrics_weight_graphs_equally_and_preserve_keys() -> None:
    summary = summarize(
        [np.asarray([2.0]), np.zeros(100)],
        [np.ones(1), np.ones(100)],
        np.asarray([0.05, 0.5]),
        SDCalibration(),
    )
    assert summary == {
        "graphs": 2,
        "edges": 101,
        "graph_balanced_nll": 1.0,
        "graph_balanced_coverage_50": 0.5,
        "graph_balanced_coverage_80": 0.5,
        "graph_balanced_coverage_95": 0.5,
    }


def test_graph_arrays_accept_scalar_vector_and_matrix_shapes() -> None:
    errors = [np.full(shape, 2.0) for shape in ((), (3,), (2, 2))]
    raw_sd = [np.ones_like(error) for error in errors]
    density = np.asarray([0.05, 0.2, 0.5])
    fitted = candidates(errors, raw_sd, density)["global"]

    assert fitted.global_scale == 2.0


def test_overflow_is_not_a_successful_calibration() -> None:
    with pytest.raises(ValueError, match="overflow"):
        SDCalibration(method="global", global_scale=1e300).apply(np.asarray([1e300]))
