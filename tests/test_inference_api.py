from __future__ import annotations

import zipfile

import jax
import numpy as np
import pytest

import dreems
from dreems import checkpoints, cli, features, model, train
from dreems.calibration import SDCalibration


def small_raw_example() -> dict[str, object]:
    positions = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ],
        dtype=np.float64,
    )
    edges = np.asarray(
        [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4], [1, 4]],
        dtype=np.int32,
    )
    genotypes = np.asarray(
        [
            [0, 1, 2, 0, 1, 2],
            [1, 1, 1, 0, 2, 0],
            [2, 0, 1, 1, 0, 1],
            [0, 2, 0, 2, 1, 1],
            [1, 0, 2, 1, 2, 0],
            [2, 1, 0, 0, 0, 2],
            [0, 2, 1, 2, 1, 0],
            [1, 0, 0, 1, 2, 1],
        ],
        dtype=np.float64,
    )
    return {
        "graph": {"active_full": np.arange(positions.shape[0]), "edges": edges},
        "active_positions": positions,
        "sample_nodes": np.asarray([0, 1, 1, 2, 3, 4, 4, 0], dtype=np.int32),
        "genotypes": genotypes,
        "edge_log_rates": np.linspace(-0.5, 0.6, edges.shape[0], dtype=np.float32),
        "node_log_q": np.linspace(-0.2, 0.2, positions.shape[0], dtype=np.float32),
    }


def write_test_checkpoint(path, *, spatial_encoding: str = "embedded") -> None:
    raw = small_raw_example()
    config = train.DREEMSTrainConfig(
        architecture=train.DREEMSArchitectureConfig(
            hidden_channels=8,
            layers=1,
            context_layers=1,
        ),
        features=train.DREEMSFeatureConfig(
            covariance_components=3,
            spatial_encoding=spatial_encoding,
        ),
        optimizer=train.DREEMSOptimizerConfig(steps=1, batch_size=1),
    )
    arrays, _validation, stats = train.prepare_training_arrays([raw], None, config=config)
    params = model.DREEMS(
        key=jax.random.PRNGKey(0),
        node_dim=arrays["node_x"].shape[-1],
        edge_dim=arrays["edge_x"].shape[-1],
        hidden_channels=config.architecture.hidden_channels,
        layers=config.architecture.layers,
        context_layers=config.architecture.context_layers,
    )
    train.save_checkpoint(
        path,
        params=params,
        normalization=stats,
        config=config,
        train_losses=[],
        train_arrays=arrays,
    )


def covariance_inputs(raw: dict[str, object]) -> tuple[np.ndarray, np.ndarray, int]:
    covariance, effective_snps = cli.covariance_from_genotypes(np.asarray(raw["genotypes"]))
    sample_coordinates = np.asarray(raw["active_positions"])[np.asarray(raw["sample_nodes"])]
    return covariance, sample_coordinates, effective_snps


def test_public_inference_api_predicts_edge_surface(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.dreems.zip"
    raw = small_raw_example()
    write_test_checkpoint(checkpoint)
    covariance, sample_coordinates, effective_snps = covariance_inputs(raw)

    example = dreems.DREEMSInput(
        sample_covariance=covariance,
        sample_coordinates=sample_coordinates,
        node_coordinates=raw["active_positions"],
        edges=raw["graph"]["edges"],
        effective_snps=effective_snps,
        name="toy",
        metadata={"species": "example"},
    )
    predictor = dreems.load_model(checkpoint)

    result = predictor.predict(example)

    assert result.name == "toy"
    assert result.metadata["species"] == "example"
    assert result.edge_log_mean.shape == (raw["graph"]["edges"].shape[0],)
    assert result.edge_log_sd.shape == result.edge_log_mean.shape
    assert result.node_log_q_mean.shape == (raw["active_positions"].shape[0],)
    assert result.node_log_q_sd.shape == result.node_log_q_mean.shape
    assert np.isfinite(result.centered_edge_log_mean).all()
    assert np.all(result.relative_migration_mean > 0.0)
    lower, upper = result.relative_migration_interval(level=0.8)
    assert lower.shape == upper.shape == result.edge_log_mean.shape
    assert np.all(lower < upper)
    records = result.to_edge_records()
    assert len(records) == raw["graph"]["edges"].shape[0]
    assert records[0]["name"] == "toy"
    assert "relative_migration_mean" in records[0]


def test_canonical_checkpoint_predicts_equally_from_all_input_forms(tmp_path) -> None:
    checkpoint = tmp_path / "canonical.dreems.zip"
    raw = small_raw_example()
    write_test_checkpoint(checkpoint, spatial_encoding="topology_node_covariance")
    covariance, sample_coordinates, effective_snps = covariance_inputs(raw)
    sample_nodes = np.asarray(raw["sample_nodes"], dtype=np.int32)
    node_covariance, observed_nodes, node_sample_count = (
        features.sample_covariance_to_node_covariance(
            covariance,
            sample_nodes,
            num_nodes=np.asarray(raw["active_positions"]).shape[0],
        )
    )
    inputs = [
        dreems.InferenceExample(
            positions=raw["active_positions"],
            edges=raw["graph"]["edges"],
            sample_nodes=sample_nodes,
            genotypes=raw["genotypes"],
        ),
        dreems.DREEMSInput(
            sample_covariance=covariance,
            sample_coordinates=sample_coordinates,
            node_coordinates=raw["active_positions"],
            edges=raw["graph"]["edges"],
            effective_snps=effective_snps,
        ),
        dreems.NodeCovarianceInput(
            node_covariance=node_covariance,
            observed_nodes=observed_nodes,
            node_sample_count=node_sample_count,
            node_coordinates=raw["active_positions"],
            edges=raw["graph"]["edges"],
            effective_snps=effective_snps,
        ),
    ]

    results = dreems.load_model(checkpoint).predict_many(inputs)
    for result in results[1:]:
        np.testing.assert_allclose(result.edge_log_mean, results[0].edge_log_mean, atol=2e-6)
        np.testing.assert_allclose(result.edge_log_sd, results[0].edge_log_sd, atol=2e-6)
        np.testing.assert_allclose(
            result.node_log_q_mean,
            results[0].node_log_q_mean,
            atol=2e-6,
        )
        np.testing.assert_allclose(result.node_log_q_sd, results[0].node_log_q_sd, atol=2e-6)


def test_node_covariance_checkpoint_migration_preserves_parameters_and_predictions(
    tmp_path,
) -> None:
    source = tmp_path / "topology.dreems.zip"
    migrated = tmp_path / "canonical.dreems.zip"
    raw = small_raw_example()
    write_test_checkpoint(source, spatial_encoding="topology")
    covariance, _sample_coordinates, effective_snps = covariance_inputs(raw)
    node_covariance, observed_nodes, node_sample_count = (
        features.sample_covariance_to_node_covariance(
            covariance,
            np.asarray(raw["sample_nodes"], dtype=np.int32),
            num_nodes=np.asarray(raw["active_positions"]).shape[0],
        )
    )
    node_example = {
        key: value for key, value in raw.items() if key not in {"genotypes", "sample_nodes"}
    }
    node_example.update(
        {
            "node_covariance": node_covariance,
            "observed_nodes": observed_nodes,
            "node_sample_count": node_sample_count,
            "effective_snps": effective_snps,
        }
    )
    before = checkpoints.predict_examples(source, [node_example])

    checkpoints.migrate_node_covariance_checkpoint(
        source,
        migrated,
        provenance={"test": True},
    )

    after = checkpoints.predict_examples(migrated, [node_example])
    with zipfile.ZipFile(source) as archive:
        source_params = archive.read(checkpoints.CHECKPOINT_PARAMS)
    with zipfile.ZipFile(migrated) as archive:
        migrated_params = archive.read(checkpoints.CHECKPOINT_PARAMS)
    assert migrated_params == source_params
    assert after.checkpoint.model["features"]["kind"] == "topology_diffusion_bands_v2"
    assert after.checkpoint.payload["feature_migration"]["provenance"] == {"test": True}
    np.testing.assert_array_equal(after.edge_mean[0], before.edge_mean[0])
    np.testing.assert_array_equal(after.edge_sd[0], before.edge_sd[0])
    np.testing.assert_array_equal(after.node_log_q_mean[0], before.node_log_q_mean[0])
    np.testing.assert_array_equal(after.node_log_q_sd[0], before.node_log_q_sd[0])
    assert after.surface_weight == before.surface_weight == [1.0]


def test_packaged_default_checkpoint_loads() -> None:
    predictor = dreems.load_model()

    assert predictor.checkpoint.model["model_name"] == "dreems"
    assert predictor.checkpoint.model["node_dim"] == 109
    assert predictor.checkpoint.model["edge_dim"] == 253
    assert checkpoints.canonical_feature_metadata(predictor.checkpoint.model["features"]) == {
        "kind": "topology_diffusion_bands_v2",
        "covariance_components": 20,
    }
    assert predictor.checkpoint.model["architecture"] == {
        "context_layers": 3,
        "hidden_channels": 128,
        "initializer_gate": True,
        "kind": "diffusion_band_surface_gate_v3",
        "layers": 10,
        "surface_gate": True,
    }
    assert predictor.checkpoint.params.surface_gate_hidden is not None
    assert predictor.checkpoint.params.surface_gate_out is not None


def test_predict_uses_packaged_default_checkpoint() -> None:
    raw = small_raw_example()
    covariance, sample_coordinates, effective_snps = covariance_inputs(raw)
    result = dreems.predict(
        dreems.DREEMSInput(
            sample_covariance=covariance,
            sample_coordinates=sample_coordinates,
            node_coordinates=raw["active_positions"],
            edges=raw["graph"]["edges"],
            effective_snps=effective_snps,
        )
    )

    assert result.edge_log_mean.shape == (raw["graph"]["edges"].shape[0],)
    assert 0.0 <= result.surface_weight <= 1.0


def test_packaged_checkpoint_is_coordinate_frame_invariant() -> None:
    raw = small_raw_example()
    edges = np.asarray(raw["graph"]["edges"])
    permutation = np.asarray([5, 2, 0, 4, 1, 3], dtype=np.int64)
    transformed = small_raw_example()
    transformed["active_positions"] = np.asarray(raw["active_positions"]) @ np.asarray(
        [[0.0, -1.0], [1.0, 0.0]]
    ).T + np.asarray([17.0, -4.0])
    transformed["graph"] = {
        **transformed["graph"],
        "edges": edges[permutation, ::-1],
    }
    transformed["edge_log_rates"] = np.asarray(raw["edge_log_rates"])[permutation]

    checkpoint = checkpoints.load_checkpoint()
    reference = checkpoints.predict_examples(checkpoint, [raw])
    changed = checkpoints.predict_examples(checkpoint, [transformed])
    aligned_mean = np.empty_like(changed.edge_mean[0])
    aligned_mean[permutation] = changed.edge_mean[0]
    aligned_sd = np.empty_like(changed.edge_sd[0])
    aligned_sd[permutation] = changed.edge_sd[0]

    np.testing.assert_allclose(aligned_mean, reference.edge_mean[0], atol=2e-6)
    np.testing.assert_allclose(aligned_sd, reference.edge_sd[0], atol=2e-6)
    np.testing.assert_allclose(changed.surface_weight, reference.surface_weight, atol=2e-6)


def test_released_feature_and_prediction_contract() -> None:
    """Guard the numerical estimator while internal modules are simplified."""

    raw = small_raw_example()
    prepared = features.prepare_examples(
        [raw],
        covariance_components=20,
        spatial_encoding="topology_node_covariance",
    )
    node_x = prepared[0]["node_x"]
    edge_x = prepared[0]["edge_x"]
    assert node_x.shape == (5, 109)
    assert edge_x.shape == (6, 251)
    assert np.all(np.isfinite(node_x))
    assert np.all(np.isfinite(edge_x))
    np.testing.assert_allclose(np.linalg.norm(node_x), 8.819232, rtol=2e-5)
    np.testing.assert_allclose(np.linalg.norm(edge_x), 11.572653, rtol=2e-5)

    prediction = checkpoints.predict_examples(checkpoints.load_checkpoint(), [raw])
    np.testing.assert_allclose(
        prediction.edge_mean[0],
        [0.01196847, -0.01295716, 0.00660560, -0.00490301, -0.00585365, 0.00513974],
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        prediction.edge_sd[0],
        [1.09670546, 1.11194005, 1.14263821, 1.19255759, 1.16527384, 1.05273488],
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        prediction.raw_edge_sd[0],
        [1.19506136, 1.21166223, 1.24511350, 1.29950980, 1.26977916, 1.14714736],
        rtol=2e-5,
        atol=2e-6,
    )
    assert prediction.surface_weight[0] == pytest.approx(0.0644742474)


def test_checkpoint_calibration_preserves_parameters_and_exposes_raw_sd(tmp_path) -> None:
    source = tmp_path / "source.dreems.zip"
    calibrated = tmp_path / "calibrated.dreems.zip"
    raw = small_raw_example()
    write_test_checkpoint(source)
    fitted = SDCalibration(
        method="density",
        global_scale=1.5,
        density_scales=(3.0, 2.5, 2.0),
    )

    checkpoints.attach_edge_sd_calibration(
        source,
        calibrated,
        fitted,
        provenance={"fit_examples": 100},
    )

    with zipfile.ZipFile(source) as archive:
        source_params = archive.read(checkpoints.CHECKPOINT_PARAMS)
    with zipfile.ZipFile(calibrated) as archive:
        calibrated_params = archive.read(checkpoints.CHECKPOINT_PARAMS)
    assert calibrated_params == source_params
    loaded = checkpoints.load_checkpoint(calibrated)
    assert loaded.edge_sd_calibration == fitted
    assert loaded.payload["edge_sd_calibration"]["provenance"] == {"fit_examples": 100}

    raw_prediction = checkpoints.predict_examples(source, [raw])
    prediction = checkpoints.predict_examples(loaded, [raw])
    np.testing.assert_allclose(prediction.raw_edge_sd[0], raw_prediction.edge_sd[0])
    np.testing.assert_allclose(prediction.edge_sd[0], 2.0 * raw_prediction.edge_sd[0])

    result = dreems.load_model(calibrated).predict(raw)
    np.testing.assert_allclose(result.edge_log_sd, prediction.edge_sd[0])
    np.testing.assert_allclose(result.raw_edge_log_sd, prediction.raw_edge_sd[0])
    assert "edge_log_sd_raw" in result.to_edge_records()[0]


def test_predict_convenience_accepts_covariance_mapping(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.dreems.zip"
    raw = small_raw_example()
    write_test_checkpoint(checkpoint)
    covariance, sample_coordinates, effective_snps = covariance_inputs(raw)

    result = dreems.predict(
        checkpoint,
        {
            "sample_covariance": covariance,
            "sample_coordinates": sample_coordinates,
            "node_coordinates": raw["active_positions"],
            "edges": raw["graph"]["edges"],
            "effective_snps": effective_snps,
        },
    )

    assert result.edges.shape == raw["graph"]["edges"].shape
    assert result.positions.shape == raw["active_positions"].shape


def test_inference_example_validates_user_arrays() -> None:
    raw = small_raw_example()
    with pytest.raises(ValueError, match="sample_nodes"):
        dreems.InferenceExample(
            positions=raw["active_positions"],
            edges=raw["graph"]["edges"],
            sample_nodes=np.asarray([0], dtype=np.int32),
            genotypes=raw["genotypes"],
        )


def test_cli_runs_dreems_from_covariance(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.dreems.zip"
    raw = small_raw_example()
    write_test_checkpoint(checkpoint)
    covariance, sample_coordinates, effective_snps = covariance_inputs(raw)
    covariance_path = tmp_path / "covariance.npy"
    sample_coordinates_path = tmp_path / "sample_coordinates.npy"
    node_coordinates_path = tmp_path / "node_coordinates.npy"
    edges_path = tmp_path / "edges.npy"
    output_path = tmp_path / "surface.csv"
    np.save(covariance_path, covariance)
    np.save(sample_coordinates_path, sample_coordinates)
    np.save(node_coordinates_path, raw["active_positions"])
    np.save(edges_path, raw["graph"]["edges"])

    cli.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--covariance",
            str(covariance_path),
            "--sample-coordinates",
            str(sample_coordinates_path),
            "--node-coordinates",
            str(node_coordinates_path),
            "--edges",
            str(edges_path),
            "--effective-snps",
            str(effective_snps),
            "--output",
            str(output_path),
        ]
    )

    text = output_path.read_text()
    assert "relative_migration_mean" in text
    assert len(text.splitlines()) == raw["graph"]["edges"].shape[0] + 1


def test_public_inference_api_predicts_with_map_refinement(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.dreems.zip"
    raw = small_raw_example()
    write_test_checkpoint(checkpoint)
    covariance, sample_coordinates, effective_snps = covariance_inputs(raw)
    predictor = dreems.load_model(checkpoint)
    example = dreems.DREEMSInput(
        sample_covariance=covariance,
        sample_coordinates=sample_coordinates,
        node_coordinates=raw["active_positions"],
        edges=raw["graph"]["edges"],
        effective_snps=effective_snps,
    )

    result = predictor.predict(
        example,
        refinement_config=dreems.DREEMSMapRefinementConfig(
            steps=2,
            learning_rate=0.01,
            trust_region_radius=0.5,
            edge_batch_size=3,
        ),
    )

    assert result.refinement_applied
    assert result.refined_edge_log_mean is not None
    assert result.refined_edge_log_mean.shape == result.edge_log_mean.shape
    assert result.refinement_diagnostics is not None
    assert "refined_fit" in result.refinement_diagnostics
    assert result.refinement_diagnostics["diagnostics"]["q_aware"]
    assert "q_scale" in result.refinement_diagnostics["refined_fit"]
    records = result.to_edge_records()
    assert "refined_edge_log_mean" in records[0]


def test_cli_runs_dreems_with_refinement_from_covariance(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.dreems.zip"
    raw = small_raw_example()
    write_test_checkpoint(checkpoint)
    covariance, sample_coordinates, effective_snps = covariance_inputs(raw)
    covariance_path = tmp_path / "covariance.npy"
    sample_coordinates_path = tmp_path / "sample_coordinates.npy"
    node_coordinates_path = tmp_path / "node_coordinates.npy"
    edges_path = tmp_path / "edges.npy"
    output_path = tmp_path / "surface.csv"
    diagnostics_path = tmp_path / "diagnostics.json"
    np.save(covariance_path, covariance)
    np.save(sample_coordinates_path, sample_coordinates)
    np.save(node_coordinates_path, raw["active_positions"])
    np.save(edges_path, raw["graph"]["edges"])

    cli.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--covariance",
            str(covariance_path),
            "--sample-coordinates",
            str(sample_coordinates_path),
            "--node-coordinates",
            str(node_coordinates_path),
            "--edges",
            str(edges_path),
            "--effective-snps",
            str(effective_snps),
            "--output",
            str(output_path),
            "--diagnostics-output",
            str(diagnostics_path),
            "--refine",
            "--refinement-steps",
            "1",
        ]
    )

    text = output_path.read_text()
    assert "edge_log_mean" in text
    assert "refined_edge_log_mean" in text
    diagnostics = diagnostics_path.read_text()
    assert "refined_fit" in diagnostics
