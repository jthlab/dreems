from __future__ import annotations

import numpy as np

from dreems.refinement import (
    DREEMSMapRefinementConfig,
    center_matrix,
    covariance_fit,
    covariance_objective_and_gradient,
    line_graph_laplacian,
    model_covariance_at_observed_nodes,
    project_edge_log_trust_region,
    q_covariance_at_observed_nodes,
    refine_edge_log_conductance,
    trust_region_diagnostics,
)


def test_covariance_fit_separates_migration_and_q_scales() -> None:
    edges = np.asarray(
        [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]],
        dtype=np.int32,
    )
    observed_nodes = np.arange(4, dtype=np.int32)
    model_covariance = model_covariance_at_observed_nodes(
        num_nodes=4,
        edges=edges,
        edge_logs=np.asarray([0.5, -0.4, 0.2, -0.3, 0.1], dtype=np.float64),
        observed_nodes=observed_nodes,
    )
    q_covariance = q_covariance_at_observed_nodes(
        node_log_q=np.asarray([-0.8, 0.4, 1.0, -0.6], dtype=np.float64),
        observed_nodes=observed_nodes,
        observed_sample_count=np.asarray([1.0, 2.0, 1.0, 3.0]),
    )
    empirical_covariance = 1.7 * model_covariance + 0.6 * q_covariance

    fit = covariance_fit(
        empirical_covariance=empirical_covariance,
        model_covariance=model_covariance,
        q_covariance=q_covariance,
    )

    np.testing.assert_allclose(fit.alpha, 1.7, rtol=1.0e-10)
    np.testing.assert_allclose(fit.q_scale, 0.6, rtol=1.0e-10)
    np.testing.assert_allclose(fit.r2, 1.0, atol=1.0e-12)


def test_q_covariance_accounts_for_deme_sample_count() -> None:
    covariance = q_covariance_at_observed_nodes(
        node_log_q=np.zeros(3, dtype=np.float64),
        observed_nodes=np.asarray([0, 2], dtype=np.int32),
        observed_sample_count=np.asarray([1.0, 4.0], dtype=np.float64),
    )

    np.testing.assert_allclose(np.diag(covariance), np.asarray([1.0, 0.25]))
    np.testing.assert_allclose(covariance - np.diag(np.diag(covariance)), 0.0)


def test_covariance_gradient_matches_finite_difference() -> None:
    node_coordinates = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        dtype=np.float64,
    )
    edges = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    observed_nodes = np.asarray([0, 1, 2], dtype=np.int32)
    empirical = model_covariance_at_observed_nodes(
        num_nodes=node_coordinates.shape[0],
        edges=edges,
        edge_logs=np.asarray([0.25, -0.75, 0.4], dtype=np.float64),
        observed_nodes=observed_nodes,
    )
    edge_logs = np.asarray([0.1, -0.2, 0.05], dtype=np.float64)
    line_laplacian = line_graph_laplacian(node_coordinates.shape[0], edges)

    def objective(values: np.ndarray) -> float:
        loss, _grad, _stats = covariance_objective_and_gradient(
            edge_logs=values,
            initial_logs=np.zeros(edges.shape[0], dtype=np.float64),
            prior_sd=np.ones(edges.shape[0], dtype=np.float64),
            num_nodes=node_coordinates.shape[0],
            edges=edges,
            observed_nodes=observed_nodes,
            empirical_centered_covariance=center_matrix(empirical),
            line_laplacian=line_laplacian,
            prior_weight=0.0,
            smoothness_weight=0.0,
            sd_floor=0.25,
            edge_batch_size=2,
        )
        return loss

    _loss, grad, _stats = covariance_objective_and_gradient(
        edge_logs=edge_logs,
        initial_logs=np.zeros(edges.shape[0], dtype=np.float64),
        prior_sd=np.ones(edges.shape[0], dtype=np.float64),
        num_nodes=node_coordinates.shape[0],
        edges=edges,
        observed_nodes=observed_nodes,
        empirical_centered_covariance=center_matrix(empirical),
        line_laplacian=line_laplacian,
        prior_weight=0.0,
        smoothness_weight=0.0,
        sd_floor=0.25,
        edge_batch_size=2,
    )
    eps = 1.0e-5
    direction = np.asarray([0.2, -0.7, 0.5], dtype=np.float64)
    direction = direction - np.mean(direction)
    finite_difference = (
        objective(edge_logs + eps * direction) - objective(edge_logs - eps * direction)
    ) / (2.0 * eps)

    np.testing.assert_allclose(
        finite_difference,
        float(np.dot(grad, direction)),
        rtol=3.0e-4,
        atol=1.0e-8,
    )


def test_q_aware_covariance_gradient_matches_finite_difference() -> None:
    edges = np.asarray(
        [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]],
        dtype=np.int32,
    )
    observed_nodes = np.arange(4, dtype=np.int32)
    truth_model = model_covariance_at_observed_nodes(
        num_nodes=4,
        edges=edges,
        edge_logs=np.asarray([0.5, -0.4, 0.2, -0.3, 0.1], dtype=np.float64),
        observed_nodes=observed_nodes,
    )
    q_covariance = q_covariance_at_observed_nodes(
        node_log_q=np.asarray([-0.8, 0.4, 1.0, -0.6], dtype=np.float64),
        observed_nodes=observed_nodes,
        observed_sample_count=np.asarray([1.0, 2.0, 1.0, 3.0]),
    )
    empirical = 1.5 * truth_model + 0.7 * q_covariance
    edge_logs = np.asarray([0.2, -0.1, 0.05, -0.2, 0.05], dtype=np.float64)
    line_laplacian = line_graph_laplacian(4, edges)

    def objective(values: np.ndarray) -> float:
        loss, _grad, _stats = covariance_objective_and_gradient(
            edge_logs=values,
            initial_logs=np.zeros(edges.shape[0], dtype=np.float64),
            prior_sd=np.ones(edges.shape[0], dtype=np.float64),
            num_nodes=4,
            edges=edges,
            observed_nodes=observed_nodes,
            empirical_centered_covariance=center_matrix(empirical),
            line_laplacian=line_laplacian,
            q_centered_covariance=center_matrix(q_covariance),
            prior_weight=0.0,
            smoothness_weight=0.0,
            sd_floor=0.25,
            edge_batch_size=2,
        )
        return loss

    _loss, grad, stats = covariance_objective_and_gradient(
        edge_logs=edge_logs,
        initial_logs=np.zeros(edges.shape[0], dtype=np.float64),
        prior_sd=np.ones(edges.shape[0], dtype=np.float64),
        num_nodes=4,
        edges=edges,
        observed_nodes=observed_nodes,
        empirical_centered_covariance=center_matrix(empirical),
        line_laplacian=line_laplacian,
        q_centered_covariance=center_matrix(q_covariance),
        prior_weight=0.0,
        smoothness_weight=0.0,
        sd_floor=0.25,
        edge_batch_size=2,
    )
    direction = np.asarray([0.2, -0.7, 0.5, 0.1, -0.1], dtype=np.float64)
    direction -= np.mean(direction)
    eps = 1.0e-5
    finite_difference = (
        objective(edge_logs + eps * direction) - objective(edge_logs - eps * direction)
    ) / (2.0 * eps)

    assert stats["q_scale"] > 0.0
    np.testing.assert_allclose(
        finite_difference,
        float(np.dot(grad, direction)),
        rtol=3.0e-4,
        atol=1.0e-8,
    )


def test_covariance_gradient_with_pair_subset_matches_finite_difference() -> None:
    node_coordinates = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    edges = np.asarray([[0, 1], [0, 2], [1, 3], [2, 3]], dtype=np.int32)
    observed_nodes = np.asarray([0, 1, 2], dtype=np.int32)
    empirical = model_covariance_at_observed_nodes(
        num_nodes=node_coordinates.shape[0],
        edges=edges,
        edge_logs=np.asarray([0.4, -0.5, 0.2, -0.1], dtype=np.float64),
        observed_nodes=observed_nodes,
    )
    edge_logs = np.asarray([0.1, -0.2, 0.05, 0.3], dtype=np.float64)
    line_laplacian = line_graph_laplacian(node_coordinates.shape[0], edges)
    pair_indices = (np.asarray([0, 1], dtype=np.int64), np.asarray([1, 2], dtype=np.int64))

    def objective(values: np.ndarray) -> float:
        loss, _grad, _stats = covariance_objective_and_gradient(
            edge_logs=values,
            initial_logs=np.zeros(edges.shape[0], dtype=np.float64),
            prior_sd=np.ones(edges.shape[0], dtype=np.float64),
            num_nodes=node_coordinates.shape[0],
            edges=edges,
            observed_nodes=observed_nodes,
            empirical_centered_covariance=center_matrix(empirical),
            line_laplacian=line_laplacian,
            pair_indices=pair_indices,
            prior_weight=0.0,
            smoothness_weight=0.0,
            sd_floor=0.25,
            edge_batch_size=2,
        )
        return loss

    _loss, grad, _stats = covariance_objective_and_gradient(
        edge_logs=edge_logs,
        initial_logs=np.zeros(edges.shape[0], dtype=np.float64),
        prior_sd=np.ones(edges.shape[0], dtype=np.float64),
        num_nodes=node_coordinates.shape[0],
        edges=edges,
        observed_nodes=observed_nodes,
        empirical_centered_covariance=center_matrix(empirical),
        line_laplacian=line_laplacian,
        pair_indices=pair_indices,
        prior_weight=0.0,
        smoothness_weight=0.0,
        sd_floor=0.25,
        edge_batch_size=2,
    )
    eps = 1.0e-5
    direction = np.asarray([0.2, -0.7, 0.5, 0.1], dtype=np.float64)
    direction = direction - np.mean(direction)
    finite_difference = (
        objective(edge_logs + eps * direction) - objective(edge_logs - eps * direction)
    ) / (2.0 * eps)

    np.testing.assert_allclose(
        finite_difference,
        float(np.dot(grad, direction)),
        rtol=3.0e-4,
        atol=1.0e-8,
    )


def test_refinement_improves_covariance_fit_on_path_graph() -> None:
    node_coordinates = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        dtype=np.float64,
    )
    edges = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    truth = np.asarray([0.6, -1.0, 0.4], dtype=np.float64)
    sample_nodes = np.asarray([0, 1, 2], dtype=np.int32)
    sample_covariance = model_covariance_at_observed_nodes(
        num_nodes=node_coordinates.shape[0],
        edges=edges,
        edge_logs=truth,
        observed_nodes=sample_nodes,
    )
    initial = np.zeros(edges.shape[0], dtype=np.float64)
    initial_model = model_covariance_at_observed_nodes(
        num_nodes=node_coordinates.shape[0],
        edges=edges,
        edge_logs=initial,
        observed_nodes=sample_nodes,
    )
    initial_fit = covariance_fit(
        empirical_covariance=sample_covariance,
        model_covariance=initial_model,
    )

    result = refine_edge_log_conductance(
        sample_covariance=sample_covariance,
        sample_nodes=sample_nodes,
        node_coordinates=node_coordinates,
        edges=edges,
        initial_edge_logs=initial,
        initial_edge_sd=np.ones(edges.shape[0], dtype=np.float64),
        effective_snps=100,
        config=DREEMSMapRefinementConfig(
            steps=20,
            learning_rate=0.05,
            smoothness_weight=0.0,
            edge_batch_size=2,
            log_every=10,
        ),
    )

    assert result.edge_log_conductance.shape == (edges.shape[0],)
    assert np.all(np.isfinite(result.edge_log_conductance))
    assert result.refined_fit.mse < initial_fit.mse
    assert result.refined_fit.r2 > result.initial_fit.r2


def test_trust_region_projection_enforces_posterior_sd_radius() -> None:
    initial = np.zeros(5, dtype=np.float64)
    prior_sd = np.asarray([0.5, 1.0, 2.0, 1.0, 0.5], dtype=np.float64)
    candidate = np.asarray([-5.0, -2.0, 0.0, 2.0, 5.0], dtype=np.float64)

    projected = project_edge_log_trust_region(
        candidate,
        initial_logs=initial,
        prior_sd=prior_sd,
        sd_floor=0.25,
        radius=0.5,
        max_edge_radius=0.75,
    )
    stats = trust_region_diagnostics(
        projected,
        initial_logs=initial,
        prior_sd=prior_sd,
        sd_floor=0.25,
    )

    assert abs(float(np.mean(projected))) < 1.0e-12
    assert stats["trust_region_rms"] <= 0.5 + 1.0e-12
    assert stats["trust_region_max_abs"] <= 0.75 + 1.0e-12


def test_refinement_respects_trust_region_radius() -> None:
    node_coordinates = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        dtype=np.float64,
    )
    edges = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    truth = np.asarray([0.6, -1.0, 0.4], dtype=np.float64)
    sample_nodes = np.asarray([0, 1, 2], dtype=np.int32)
    sample_covariance = model_covariance_at_observed_nodes(
        num_nodes=node_coordinates.shape[0],
        edges=edges,
        edge_logs=truth,
        observed_nodes=sample_nodes,
    )

    result = refine_edge_log_conductance(
        sample_covariance=sample_covariance,
        sample_nodes=sample_nodes,
        node_coordinates=node_coordinates,
        edges=edges,
        initial_edge_logs=np.zeros(edges.shape[0], dtype=np.float64),
        initial_edge_sd=np.ones(edges.shape[0], dtype=np.float64),
        effective_snps=100,
        config=DREEMSMapRefinementConfig(
            steps=20,
            learning_rate=0.5,
            smoothness_weight=0.0,
            trust_region_radius=0.1,
            edge_batch_size=2,
            log_every=10,
        ),
    )

    assert result.diagnostics["trust_region_rms"] <= 0.1 + 1.0e-12
    assert result.diagnostics["trust_region_radius"] == 0.1


def test_default_refinement_config_uses_trust_region_half() -> None:
    config = DREEMSMapRefinementConfig()

    assert config.steps == 50
    assert config.trust_region_radius == 0.5
    assert config.smoothness_weight == 1.0e-4
