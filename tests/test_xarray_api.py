from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import dreems


def toy_graph() -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [1.5, 1.0],
        ],
        dtype=np.float64,
    )
    edges = np.asarray(
        [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]],
        dtype=np.int32,
    )
    return positions, edges


def toy_dreems_dataset() -> xr.Dataset:
    positions, edges = toy_graph()
    node_sample_count = np.asarray([2, 0, 3, 1], dtype=np.int64)
    factor = np.asarray(
        [
            [1.0, 0.2],
            [0.3, 0.8],
            [-0.4, 0.5],
        ],
        dtype=np.float64,
    )
    covariance = factor @ factor.T
    return dreems.create_dataset(
        node_positions=positions,
        edges=edges,
        node_covariance=covariance,
        observed_nodes=np.asarray([0, 2, 3], dtype=np.int32),
        node_sample_count=node_sample_count,
        effective_snps=100,
        attrs={"name": "toy"},
    )


def toy_genotype_dataset() -> xr.Dataset:
    calls = np.asarray(
        [
            [[0, 0], [0, 1], [1, 1], [0, 0], [0, 1], [1, 1]],
            [[1, 1], [0, 1], [0, 0], [1, 1], [0, 1], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 1], [-1, -1], [1, 1], [0, 0], [0, 1], [0, 0]],
            [[0, 0], [0, 2], [0, 1], [0, 0], [0, 1], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1]],
        ],
        dtype=np.int8,
    )
    return xr.Dataset(
        {
            "call_genotype": (("variants", "samples", "ploidy"), calls),
            "sample_id": (
                ("samples",),
                np.asarray(["a", "b", "c", "d", "e", "f"], dtype=str),
            ),
        }
    )


def toy_tetraploid_genotype_dataset() -> xr.Dataset:
    calls = np.asarray(
        [
            [
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 0, 1, 1],
            ],
            [
                [1, 1, 1, 1],
                [0, 1, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 1, 1],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 1],
                [-1, -1, -1, -1],
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
            ],
        ],
        dtype=np.int8,
    )
    return xr.Dataset(
        {
            "call_genotype": (("variants", "samples", "ploidy"), calls),
            "sample_id": (
                ("samples",),
                np.asarray(["a", "b", "c", "d", "e", "f"], dtype=str),
            ),
        }
    )


def test_create_dataset_uses_observed_node_covariance_schema() -> None:
    dataset = toy_dreems_dataset()

    assert dataset["node_position"].dims == ("nodes", "coordinate_axes")
    assert dataset["edge_node"].dims == ("edges", "edge_endpoints")
    assert dataset["node_covariance"].dims == ("observed_nodes", "observed_nodes_2")
    np.testing.assert_array_equal(dataset.coords["observed_nodes"], [0, 2, 3])
    np.testing.assert_array_equal(dataset["node_sample_count"], [2, 0, 3, 1])
    assert int(dataset["effective_snps"]) == 100
    dreems.validate_dataset(dataset)


def test_create_dataset_rejects_counts_missing_from_covariance() -> None:
    positions, edges = toy_graph()

    with pytest.raises(ValueError, match="positive node_sample_count"):
        dreems.create_dataset(
            node_positions=positions,
            edges=edges,
            node_covariance=np.eye(2),
            observed_nodes=np.asarray([0, 2]),
            node_sample_count=np.asarray([1, 1, 1, 0]),
            effective_snps=20,
        )


def test_attach_sample_coordinates_matches_sgkit_sample_ids() -> None:
    genotype_dataset = toy_genotype_dataset()
    coordinates = {
        sample_id: (42.0 + index * 0.1, -84.0 + index * 0.1)
        for index, sample_id in enumerate(("a", "b", "c", "d", "e", "f"))
    }

    attached = dreems.attach_sample_coordinates(genotype_dataset, coordinates)

    assert "sample_latitude" not in genotype_dataset
    assert attached["sample_latitude"].dims == ("samples",)
    assert attached["sample_longitude"].dims == ("samples",)
    np.testing.assert_allclose(attached["sample_latitude"], [42.0, 42.1, 42.2, 42.3, 42.4, 42.5])
    np.testing.assert_allclose(
        attached["sample_longitude"],
        [-84.0, -83.9, -83.8, -83.7, -83.6, -83.5],
    )


def test_from_genotype_dataset_applies_qc_and_aggregates_by_node() -> None:
    genotype_dataset = toy_genotype_dataset()
    coordinates = {
        "a": (42.00, -84.00),
        "b": (42.01, -84.01),
        "c": (42.50, -83.50),
        "d": (42.51, -83.51),
        "e": (43.00, -83.00),
        "f": (43.01, -83.01),
    }
    node_latitude = np.asarray([42.0, 42.5, 43.0])
    node_longitude = np.asarray([-84.0, -83.5, -83.0])
    edges = np.asarray([[0, 1], [1, 2]], dtype=np.int32)

    output = dreems.from_vcf_zarr(
        genotype_dataset,
        node_latitude=node_latitude,
        node_longitude=node_longitude,
        edges=edges,
        sample_coordinates=coordinates,
        min_maf=0.05,
        min_variant_call_rate=0.8,
    )

    dreems.validate_dataset(output)
    np.testing.assert_array_equal(output["node_sample_count"], [2, 2, 2])
    np.testing.assert_array_equal(output.coords["observed_nodes"], [0, 1, 2])
    assert int(output["effective_snps"]) == 4
    assert int(output.attrs["input_variants"]) == 6
    assert int(output.attrs["retained_variants"]) == 4
    np.testing.assert_allclose(
        output["node_covariance"],
        output["node_covariance"].values.T,
    )
    assert "node_latitude" in output.coords
    assert "node_longitude" in output.coords


def test_from_genotype_dataset_accepts_tetraploid_dosage() -> None:
    genotype_dataset = toy_tetraploid_genotype_dataset()
    coordinates = {
        "a": (42.00, -84.00),
        "b": (42.01, -84.01),
        "c": (42.50, -83.50),
        "d": (42.51, -83.51),
        "e": (43.00, -83.00),
        "f": (43.01, -83.01),
    }

    output = dreems.from_genotype_dataset(
        genotype_dataset,
        node_latitude=np.asarray([42.0, 42.5, 43.0]),
        node_longitude=np.asarray([-84.0, -83.5, -83.0]),
        edges=np.asarray([[0, 1], [1, 2]], dtype=np.int32),
        sample_coordinates=coordinates,
        min_maf=0.05,
        min_variant_call_rate=0.8,
    )

    dreems.validate_dataset(output)
    assert int(output.attrs["ploidy"]) == 4
    assert "uniform-ploidy (4x)" in output.attrs["genotype_qc"]
    assert int(output.attrs["retained_variants"]) == 3
    assert int(output["effective_snps"]) == 3
    np.testing.assert_array_equal(output["node_sample_count"], [2, 2, 2])
    assert np.isfinite(output["node_covariance"]).all()


def test_infer_returns_xarray_surface() -> None:
    dataset = toy_dreems_dataset()

    result = dreems.infer(dataset, refine=False)

    assert result["edge_log_migration_mean"].dims == ("edges",)
    assert result["edge_log_migration_sd"].dims == ("edges",)
    assert result["edge_relative_migration_mean"].dims == ("edges",)
    assert result["node_log_q_mean"].dims == ("nodes",)
    assert np.isfinite(result["edge_log_migration_mean"]).all()
    assert np.all(result["edge_relative_migration_mean"] > 0.0)
    np.testing.assert_allclose(
        np.exp(result["edge_log_migration_mean"]).prod() ** (1.0 / result.sizes["edges"]),
        1.0,
    )
    assert not bool(result.attrs["dreems_refinement_applied"])


def test_infer_refines_directly_against_node_covariance() -> None:
    dataset = toy_dreems_dataset()

    result = dreems.infer(
        dataset,
        refine=dreems.DREEMSMapRefinementConfig(
            steps=1,
            learning_rate=0.01,
            trust_region_radius=0.5,
            edge_batch_size=3,
        ),
    )

    assert bool(result.attrs["dreems_refinement_applied"])
    assert "dreems_refinement" in result.attrs
    assert "edge_log_migration_refined" in result
    assert "edge_relative_migration_refined" in result
