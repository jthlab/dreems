"""Validation must not alter the graph or precision before checking it."""

from __future__ import annotations

import ast
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import xarray as xr
from jaxtyping import TypeCheckError

from dreems import cli, datasets, features, feems, inference, linearized


@pytest.fixture
def node_input() -> dict[str, Any]:
    return {
        "node_coordinates": np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        "edges": np.asarray([[0, 1], [1, 2], [0, 2]], dtype=np.int32),
        "observed_nodes": np.asarray([2, 0, 1], dtype=np.int32),
        "node_sample_count": np.asarray([1, 2, 3], dtype=np.int64),
        "node_covariance": np.eye(3),
        "effective_snps": 100,
    }


@pytest.mark.parametrize(
    "module",
    [
        "inference",
        "linearized",
        "benchmark_data",
        "simulation",
        "genotypes",
        "refinement",
        "calibration",
    ],
)
def test_array_annotations_describe_dimensions(module: str) -> None:
    source = Path(__file__).parents[1] / "src/dreems" / f"{module}.py"
    tree = ast.parse(source.read_text())

    def check(annotation: ast.AST) -> None:
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            check(ast.parse(annotation.value, mode="eval").body)
            return
        if isinstance(annotation, ast.Subscript) and isinstance(annotation.slice, ast.Tuple):
            arguments = annotation.slice.elts
            if len(arguments) == 2 and ast.unparse(arguments[0]) == "np.ndarray":
                assert isinstance(arguments[1], ast.Constant), ast.unparse(annotation)
                assert isinstance(arguments[1].value, str), ast.unparse(annotation)
                return
        assert ast.unparse(annotation) != "np.ndarray", f"{module}:{annotation.lineno}"
        for child in ast.iter_child_nodes(annotation):
            check(child)

    for node in ast.walk(tree):
        if isinstance(node, (ast.arg, ast.AnnAssign)) and node.annotation is not None:
            check(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is not None:
            check(node.returns)


def test_initializer_result_requires_matching_edge_dimensions() -> None:
    with pytest.raises(TypeCheckError):
        linearized.LinearizedConductanceEstimate(
            edge_log_conductance=np.zeros(3),
            edge_information=np.zeros(4),
            observed_nodes=np.asarray([0, 1]),
            observed_pair_log_resistance=np.zeros(1),
            baseline_pair_log_resistance=np.zeros(1),
            pair_residual=np.zeros(1),
            diagnostics={},
        )


def test_covariance_input_requires_a_matrix_of_dosages() -> None:
    with pytest.raises(TypeCheckError):
        inference.genotype_covariance(np.asarray([0.0, 1.0, 2.0]))


def dataset_from_input(values: dict[str, Any]) -> xr.Dataset:
    kwargs = dict(values)
    kwargs["node_positions"] = kwargs.pop("node_coordinates")
    return datasets.create_dataset(**kwargs)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("edges", np.asarray([[0, 1.2], [1, 2], [0, 2]])),
        ("edges", np.asarray([[0, 2**32 + 1], [1, 2], [0, 2]], dtype=np.uint64)),
        ("observed_nodes", np.asarray([2.4, 0.0, 1.0])),
        ("observed_nodes", np.asarray([2**32 + 2, 0, 1], dtype=np.uint64)),
        ("node_sample_count", np.asarray([1.000001, 2.0, 3.0])),
        ("node_sample_count", np.asarray([1.0, 2.0, np.inf])),
        ("node_sample_count", np.asarray([1.0, 2.0, 2.0**63])),
    ],
)
def test_indices_and_counts_are_checked_before_conversion(node_input, field, invalid) -> None:
    values = {**node_input, field: invalid}
    with pytest.raises(ValueError, match=field):
        inference.NodeCovarianceInput(**values)
    with pytest.raises(ValueError, match=field):
        dataset_from_input(values)
    with pytest.raises(ValueError, match=field):
        inference.coerce_inference_example(values)


@pytest.mark.parametrize("value", [50.5, np.nan, np.inf, True, "100", [100], 2**63])
def test_snp_count_is_not_silently_truncated(node_input, value) -> None:
    values = {**node_input, "effective_snps": value}
    with pytest.raises(ValueError, match="effective_snps"):
        inference.NodeCovarianceInput(**values)
    with pytest.raises(ValueError, match="effective_snps"):
        dataset_from_input(values)
    with pytest.raises(ValueError, match="effective_snps"):
        inference.coerce_inference_example(values)
    with pytest.raises(ValueError, match="effective_snps"):
        inference.DREEMSInput(
            sample_covariance=np.eye(3),
            sample_coordinates=values["node_coordinates"],
            node_coordinates=values["node_coordinates"],
            edges=values["edges"],
            effective_snps=value,
        )


def test_integral_float_arrays_and_numpy_snp_scalar_remain_supported(node_input) -> None:
    values = {
        **node_input,
        "edges": node_input["edges"].astype(float),
        "observed_nodes": node_input["observed_nodes"].astype(float),
        "node_sample_count": node_input["node_sample_count"].astype(float),
        "effective_snps": np.int64(100),
    }
    converted = datasets.dataset_to_input(dataset_from_input(values))
    assert converted.edges.dtype == np.int32
    assert converted.observed_nodes.dtype == np.int32
    assert converted.node_sample_count.dtype == np.int64
    assert type(converted.effective_snps) is int
    np.testing.assert_array_equal(converted.observed_nodes, [2, 0, 1])


@pytest.mark.parametrize("field", ["edges", "sample_nodes"])
def test_genotype_input_checks_indices_before_conversion(node_input, field) -> None:
    values = {
        "positions": node_input["node_coordinates"],
        "edges": node_input["edges"].astype(float),
        "sample_nodes": np.asarray([0.0, 1.0, 2.0]),
        "genotypes": np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]]),
    }
    values[field].flat[0] += 0.1
    with pytest.raises(ValueError, match=field):
        inference.InferenceExample(**values)


@pytest.mark.parametrize("invalid", [-9, -0.1, 2.1, np.nan, np.inf])
def test_raw_dosages_require_qc_before_inference(node_input, invalid) -> None:
    genotypes = np.asarray([[0.0, 0.5], [1.0, 2.0], [2.0, 0.0]])
    genotypes[0, 0] = invalid
    with pytest.raises(ValueError):
        inference.InferenceExample(
            positions=node_input["node_coordinates"],
            edges=node_input["edges"],
            sample_nodes=np.arange(3),
            genotypes=genotypes,
        )
    with pytest.raises(ValueError):
        features.standardized_polymorphic_genotypes(genotypes)
    with pytest.raises(ValueError):
        feems.polymorphic_snp_mask(genotypes)


def test_fractional_imputed_dosages_remain_supported(node_input) -> None:
    genotypes = np.asarray([[0.1, 0.5], [1.0, 1.8], [1.9, 0.2]])
    example = inference.InferenceExample(
        positions=node_input["node_coordinates"],
        edges=node_input["edges"],
        sample_nodes=np.arange(3),
        genotypes=genotypes,
    )
    np.testing.assert_array_equal(example.genotypes, genotypes)
    np.testing.assert_array_equal(feems.polymorphic_snp_mask(genotypes), [True, True])
    assert np.all(np.isfinite(features.standardized_polymorphic_genotypes(genotypes)))


def test_cli_does_not_replace_explicit_zero_snp_count(node_input, tmp_path) -> None:
    arrays = {
        "genotypes": np.asarray([[0.0, 0.5], [1.0, 2.0], [2.0, 0.0]]),
        "sample_coordinates": node_input["node_coordinates"],
        "node_coordinates": node_input["node_coordinates"],
        "edges": node_input["edges"],
    }
    paths = {}
    for key, value in arrays.items():
        paths[key] = tmp_path / f"{key}.npy"
        np.save(paths[key], value)
    args = Namespace(**paths, covariance=None, effective_snps=0)
    with pytest.raises(ValueError, match="effective_snps"):
        cli.build_dreems_input(args)
    args.effective_snps = None
    assert cli.build_dreems_input(args).effective_snps == 2


@pytest.mark.parametrize(
    "edges",
    [np.asarray([[0, 1], [1, 0], [1, 2]]), np.asarray([[0, 1]])],
)
def test_numpy_and_xarray_reject_invalid_graphs_consistently(node_input, edges) -> None:
    values = {**node_input, "edges": edges}
    message = "duplicate|connected"
    with pytest.raises(ValueError, match=message):
        inference.NodeCovarianceInput(**values)
    with pytest.raises(ValueError, match=message):
        dataset_from_input(values)
    with pytest.raises(ValueError, match=message):
        inference.DREEMSInput(
            sample_covariance=np.eye(3),
            sample_coordinates=values["node_coordinates"],
            node_coordinates=values["node_coordinates"],
            edges=edges,
        )


def test_node_covariance_requires_psd_with_tolerance_for_roundoff(node_input) -> None:
    covariance = np.diag([1.0, 1.0, -0.1])
    with pytest.raises(ValueError, match="positive semidefinite"):
        inference.NodeCovarianceInput(**{**node_input, "node_covariance": covariance})
    covariance[-1, -1] = -1e-9
    inference.NodeCovarianceInput(**{**node_input, "node_covariance": covariance})


def test_dataset_validation_does_not_hide_fractional_fields(node_input) -> None:
    source = dataset_from_input(node_input)
    mutations = {
        "edge_node": (("edges", "edge_endpoints"), node_input["edges"].astype(float) + 0.1),
        "node_sample_count": (("nodes",), np.asarray([1.000001, 2.0, 3.0])),
        "effective_snps": ((), 100.5),
    }
    for name, replacement in mutations.items():
        changed = source.copy(deep=True)
        changed[name] = replacement
        with pytest.raises(ValueError):
            datasets.validate_dataset(changed)

    changed = source.assign_coords(
        observed_nodes=np.asarray([2.4, 0.0, 1.0]),
        observed_nodes_2=np.asarray([2.4, 0.0, 1.0]),
    )
    with pytest.raises(ValueError, match="observed_nodes"):
        datasets.validate_dataset(changed)


def test_covariance_axes_must_be_explicit_and_agree(node_input) -> None:
    source = dataset_from_input(node_input)
    with pytest.raises(ValueError, match="explicit coordinate"):
        datasets.validate_dataset(source.drop_vars("observed_nodes"))
    changed = source.assign_coords(observed_nodes_2=np.asarray([0, 1, 2]))
    with pytest.raises(ValueError, match="coordinates must match"):
        datasets.validate_dataset(changed)


def test_result_replacement_removes_old_optional_outputs(node_input) -> None:
    source = dataset_from_input(node_input)
    source.attrs["study"] = "retain this"
    fitted = inference.DREEMSResult(
        positions=node_input["node_coordinates"],
        edges=node_input["edges"],
        edge_log_mean=np.asarray([-0.2, 0.0, 0.2]),
        edge_log_sd=np.ones(3),
        raw_edge_log_sd=np.full(3, 2.0),
        refined_edge_log_mean=np.asarray([-0.1, 0.1, 0.0]),
        refinement_diagnostics={"steps": 5},
        node_log_q_mean=np.zeros(3),
        node_log_q_sd=np.ones(3),
    )
    old = datasets.result_to_dataset(source, fitted)
    neural_only = replace(
        fitted,
        raw_edge_log_sd=np.ones(3),
        refined_edge_log_mean=None,
        refinement_diagnostics=None,
        node_log_q_mean=None,
        node_log_q_sd=None,
    )
    current = datasets.result_to_dataset(old, neural_only)
    for name in (
        "edge_log_migration_refined",
        "edge_relative_migration_refined",
        "edge_log_migration_sd_raw",
        "node_log_q_mean",
        "node_log_q_sd",
    ):
        assert name in old
        assert name not in current
    assert "dreems_refinement" not in current.attrs
    assert "dreems_refinement" in old.attrs
    assert not current.attrs["dreems_refinement_applied"]
    assert current.attrs["study"] == "retain this"
    for name in source.variables:
        xr.testing.assert_identical(source[name], current[name])

    with pytest.raises(ValueError, match="same order"):
        datasets.result_to_dataset(source, replace(fitted, edges=fitted.edges[::-1]))


def test_edge_record_export_centers_each_surface_only_once(node_input, monkeypatch) -> None:
    result = inference.DREEMSResult(
        positions=node_input["node_coordinates"],
        edges=node_input["edges"],
        edge_log_mean=np.asarray([0.1, 0.2, 0.3]),
        edge_log_sd=np.ones(3),
        refined_edge_log_mean=np.asarray([0.3, 0.2, 0.1]),
    )
    calls = []
    original = inference.metrics.centered

    def counted(values):
        calls.append(values)
        return original(values)

    monkeypatch.setattr(inference.metrics, "centered", counted)
    records = result.to_edge_records()
    assert len(calls) == 2
    np.testing.assert_allclose(
        [row["edge_log_mean"] for row in records], [-0.1, 0.0, 0.1], atol=1e-15
    )


def test_cli_loads_a_single_csv_edge_as_a_matrix(tmp_path) -> None:
    path = tmp_path / "edge.csv"
    path.write_text("0,1\n")
    np.testing.assert_array_equal(cli.load_array(path), [[0.0, 1.0]])
    archive = tmp_path / "edge.npz"
    np.savez(archive, edges=np.asarray([[0, 1]]))
    np.testing.assert_array_equal(cli.load_array(archive), [[0, 1]])


@pytest.fixture
def result_input(node_input) -> dict[str, Any]:
    return {
        "positions": node_input["node_coordinates"],
        "edges": node_input["edges"],
        "edge_log_mean": np.asarray([-0.2, 0.0, 0.2]),
        "edge_log_sd": np.ones(3),
        "node_log_q_mean": np.zeros(3),
        "node_log_q_sd": np.ones(3),
    }


@pytest.mark.parametrize(
    "field",
    [
        "edge_log_mean",
        "edge_log_sd",
        "raw_edge_log_sd",
        "refined_edge_log_mean",
        "node_log_q_mean",
        "node_log_q_sd",
    ],
)
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_result_rejects_nonfinite_arrays(result_input, field, invalid) -> None:
    with pytest.raises(ValueError, match=field):
        inference.DREEMSResult(**{**result_input, field: np.asarray([1.0, invalid, 1.0])})


@pytest.mark.parametrize("field", ["edge_log_sd", "raw_edge_log_sd", "node_log_q_sd"])
def test_result_rejects_negative_uncertainty(result_input, field) -> None:
    with pytest.raises(ValueError, match=field + " must be nonnegative"):
        inference.DREEMSResult(**{**result_input, field: np.asarray([1.0, -0.1, 0.0])})


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("edges", [[0, -1], [1, 2], [0, 2]]),
        ("edges", [[0, 3], [1, 2], [0, 2]]),
        ("positions", [[0.0, 0.0], [np.nan, 0.0], [0.0, 1.0]]),
        ("edge_log_mean", [[0.0, 0.0, 0.0]]),
        ("edge_log_sd", [1.0, 1.0]),
        ("raw_edge_log_sd", [1.0]),
        ("refined_edge_log_mean", [[1.0, 1.0, 1.0]]),
        ("node_log_q_sd", [1.0, 1.0]),
    ],
)
def test_result_checks_indices_positions_and_shapes(result_input, field, invalid) -> None:
    with pytest.raises(ValueError, match=field):
        inference.DREEMSResult(**{**result_input, field: invalid})


def test_result_accepts_lists_zero_sd_and_disconnected_edge_subsets() -> None:
    result = inference.DREEMSResult(
        edge_log_mean=[-1.0, 1.0],
        edge_log_sd=[0.0, 0.0],
        edges=[[0, 1], [2, 3]],
        positions=[[0, 0], [1, 0], [2, 0], [3, 0]],
        raw_edge_log_sd=[0.0, 0.0],
        refined_edge_log_mean=[-0.5, 0.5],
    )
    assert result.edge_log_mean.dtype == np.float64
    assert result.edges.dtype == np.int32
    lower, upper = result.centered_edge_log_interval(np.float32(0.8))
    np.testing.assert_array_equal(lower, result.centered_edge_log_mean)
    np.testing.assert_array_equal(upper, lower)
    assert result.refinement_applied
    assert len(result.to_edge_records()) == 2


@pytest.mark.parametrize("level", [np.nan, np.inf, 0.0, 1.0, -0.1])
def test_result_interval_rejects_invalid_levels(result_input, level) -> None:
    with pytest.raises(ValueError, match="interval level"):
        inference.DREEMSResult(**result_input).relative_migration_interval(level)


@pytest.mark.parametrize("suffix", [".json", ".csv"])
def test_feems_cli_exports_rates_without_a_neural_sd(tmp_path, monkeypatch, suffix) -> None:
    import csv
    import json

    positions = np.asarray([[0, 0], [1, 0], [2, 0]])
    edges = np.asarray([[0, 1], [1, 2]], dtype=np.int32)
    arrays = {
        "nodes.npy": positions,
        "samples.npy": positions,
        "edges.npy": edges,
        "genotypes.npy": np.asarray([[0, 1], [1, 2], [2, 0]]),
    }
    for name, values in arrays.items():
        np.save(tmp_path / name, values)
    info = {"best_lambda": 1.0, "mean_cv_error": [0.1, 0.2]}
    calls = []

    def fake_fit(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(edge_log_rates=np.asarray([1.0, 3.0]), info_dict=lambda: info)

    monkeypatch.setattr(cli.feems, "fit_feems", fake_fit)
    output = tmp_path / ("fit" + suffix)
    cli.main(
        [
            "--feems",
            "--feems-mode",
            "fixed_q",
            "--genotypes",
            str(tmp_path / "genotypes.npy"),
            "--sample-coordinates",
            str(tmp_path / "samples.npy"),
            "--node-coordinates",
            str(tmp_path / "nodes.npy"),
            "--edges",
            str(tmp_path / "edges.npy"),
            "--output",
            str(output),
        ]
    )
    if suffix == ".json":
        records = json.loads(output.read_text())
    else:
        with output.open() as handle:
            records = list(csv.DictReader(handle))
    assert len(calls) == 1
    assert calls[0]["config"].mode == "fixed_q"
    np.testing.assert_array_equal(calls[0]["sample_nodes"], [0, 1, 2])
    np.testing.assert_array_equal([float(r["edge_log_mean"]) for r in records], [-1.0, 1.0])
    assert all(np.isnan(float(r["edge_log_sd"])) for r in records)
    assert all(float(r["best_lambda"]) == 1.0 for r in records)
    assert all("mean_cv_error" not in r for r in records)
    assert [int(r["right_node"]) for r in records] == [1, 2]


@pytest.mark.parametrize("field", ["node_covariance", "node_coordinates"])
@pytest.mark.parametrize("dtype", [complex, str, object])
def test_node_inputs_reject_non_real_numeric_arrays(node_input, field, dtype) -> None:
    values = {**node_input, field: node_input[field].astype(dtype)}
    with pytest.raises(ValueError, match="real numbers"):
        inference.NodeCovarianceInput(**values)
    with pytest.raises(ValueError, match="real numbers"):
        dataset_from_input(values)


@pytest.mark.parametrize("field", ["sample_covariance", "sample_coordinates", "node_coordinates"])
def test_sample_inputs_cannot_discard_imaginary_components(node_input, field) -> None:
    values = {
        "sample_covariance": np.eye(3),
        "sample_coordinates": node_input["node_coordinates"],
        "node_coordinates": node_input["node_coordinates"],
        "edges": node_input["edges"],
    }
    values[field] = values[field].astype(complex) + 0.01j
    with pytest.raises(ValueError, match=field):
        inference.DREEMSInput(**values)


@pytest.mark.parametrize("field", ["edge_log_mean", "edge_log_sd", "positions"])
def test_results_cannot_discard_imaginary_components(result_input, field) -> None:
    result_input[field] = result_input[field].astype(complex) + 0.01j
    with pytest.raises(ValueError, match=field):
        inference.DREEMSResult(**result_input)


def test_genotype_covariance_rejects_complex_calls() -> None:
    with pytest.raises(ValueError, match="real numbers"):
        inference.genotype_covariance(np.asarray([[0, 1], [1, 0]], dtype=complex))


@pytest.mark.parametrize("coordinate", ["sample_coordinates", "node_coordinates"])
def test_feems_cli_rejects_complex_coordinates(tmp_path, monkeypatch, coordinate) -> None:
    coordinates = np.asarray([[0, 0], [1, 0], [2, 0]], dtype=float)
    arrays = dict(
        sample_coordinates=coordinates,
        node_coordinates=coordinates,
        edges=np.asarray([[0, 1], [1, 2]], dtype=np.int32),
        genotypes=np.asarray([[0, 1], [1, 2], [2, 0]]),
    )
    arrays[coordinate] = coordinates.astype(complex) + 0.01j
    paths = {}
    for name, values in arrays.items():
        paths[name] = tmp_path / f"{name}.npy"
        np.save(paths[name], values)
    monkeypatch.setattr(feems, "fit_feems", lambda **kwargs: pytest.fail("must validate first"))
    with pytest.raises(ValueError, match=coordinate):
        cli.run_feems(Namespace(**paths, feems_mode="free_q", output=tmp_path / "unused.json"))
