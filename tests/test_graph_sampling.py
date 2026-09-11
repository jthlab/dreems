"""Graph builders and site samplers must fail clearly and retain their ordering."""

from unittest.mock import patch

import numpy as np
import pytest
from jaxtyping import TypeCheckError
from shapely.geometry import box

from dreems.data import geography, motifs


@pytest.fixture
def line() -> tuple[np.ndarray, np.ndarray]:
    positions = np.column_stack((np.arange(12, dtype=float), np.zeros(12)))
    edges = np.column_stack((np.arange(11), np.arange(1, 12))).astype(np.int32)
    return positions, edges


@pytest.mark.parametrize("spacing", [0.0, -1.0, np.nan, np.inf])
def test_hex_spacing_rejects_nonadvancing_steps(spacing: float) -> None:
    with pytest.raises(ValueError, match="spacing"):
        geography.hex_candidates((0.0, 0.0, 1.0, 1.0), spacing)
    with pytest.raises(ValueError, match="spacing"):
        geography.edges_from_hex_points(np.zeros((1, 2)), spacing)


def test_hex_spacing_rejects_roundoff_and_overflow() -> None:
    with pytest.raises(ValueError, match="advance"):
        geography.hex_candidates((1e20, 1e20, 2e20, 2e20), 1.0)
    with pytest.raises(ValueError, match="overflow"):
        geography.hex_candidates((0.0, 0.0, 1e308, 1e308), 1e308)


@pytest.mark.parametrize(
    "bounds", [(1.0, 0.0, 0.0, 1.0), (0.0, 2.0, 1.0, 1.0), (0.0, np.nan, 1.0, 1.0)]
)
def test_hex_bounds_are_checked(bounds: tuple) -> None:
    with pytest.raises(ValueError, match="bounds"):
        geography.hex_candidates(bounds, 0.2)


def test_hex_mesh_edges_are_sorted_and_connected() -> None:
    spacing, points = geography.choose_hex_spacing(box(0, 0, 8, 5), 60)
    edges = geography.edges_from_hex_points(points, spacing)
    assert abs(len(points) - 60) <= 5
    assert edges.tolist() == sorted(edges.tolist())
    selected, remapped, indices = geography.largest_graph_component(points, edges)
    np.testing.assert_array_equal(selected, points)
    np.testing.assert_array_equal(remapped, edges)
    np.testing.assert_array_equal(indices, np.arange(len(points)))


@pytest.mark.parametrize("count", [0, 1, 4])
def test_largest_edgeless_component_has_at_most_one_node(count: int) -> None:
    positions = np.column_stack((np.arange(count), np.zeros(count)))
    empty = np.empty((0, 2), dtype=np.int32)
    selected, edges, indices = geography.largest_graph_component(positions, empty)
    np.testing.assert_array_equal(selected, positions[:1])
    np.testing.assert_array_equal(edges, empty)
    np.testing.assert_array_equal(indices, np.arange(min(count, 1)))


def test_largest_component_remaps_edges_without_reordering(line) -> None:
    positions, _ = line
    edges = np.asarray([[5, 7], [0, 2], [7, 9], [2, 4]], dtype=np.int32)
    selected, remapped, indices = geography.largest_graph_component(positions, edges)
    np.testing.assert_array_equal(indices, [0, 2, 4])
    np.testing.assert_array_equal(selected, positions[indices])
    np.testing.assert_array_equal(remapped, [[0, 1], [1, 2]])


def test_path_sampler_solves_once(line) -> None:
    positions, edges = line
    with patch.object(motifs, "shortest_path", wraps=motifs.shortest_path) as solve:
        chosen = motifs.choose_path_sample_sites(np.random.default_rng(14), positions, edges, 4)
    assert solve.call_count == 1
    assert len(np.unique(chosen)) == 4
    assert np.all((chosen >= 0) & (chosen < len(positions)))


def test_path_sampler_uses_disconnected_fallback(line) -> None:
    positions, _ = line
    chosen = motifs.choose_path_sample_sites(
        np.random.default_rng(2), positions, np.empty((0, 2), dtype=np.int32), 5
    )
    assert len(np.unique(chosen)) == 5


@pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
def test_land_sampler_never_selects_water(line, probability: float) -> None:
    positions, edges = line
    land_mask = np.arange(len(positions)) % 3 != 1
    chosen = motifs.choose_global_landscape_sample_sites(
        np.random.default_rng(4), positions, edges, land_mask, 5, path_probability=probability
    )
    assert len(np.unique(chosen)) == 5
    assert land_mask[chosen].all()


def test_path_errors_are_not_replaced_with_other_sampling(line) -> None:
    positions, edges = line
    with (
        patch.object(motifs, "choose_path_sample_sites", side_effect=RuntimeError("broken solve")),
        pytest.raises(RuntimeError, match="broken solve"),
    ):
        motifs.choose_global_landscape_sample_sites(
            np.random.default_rng(3), positions, edges, np.ones(len(positions), dtype=bool), 3
        )


@pytest.mark.parametrize("probability", [-0.1, 1.1, np.nan])
def test_land_sampler_rejects_invalid_probability(line, probability: float) -> None:
    positions, edges = line
    with pytest.raises(ValueError, match="path_probability"):
        motifs.choose_global_landscape_sample_sites(
            np.random.default_rng(1),
            positions,
            edges,
            np.ones(12, dtype=bool),
            3,
            path_probability=probability,
        )


def test_land_mask_must_be_boolean_and_match_graph(line) -> None:
    positions, edges = line
    for mask in (np.ones(11, dtype=bool), np.ones(12, dtype=float)):
        with pytest.raises(TypeCheckError):
            motifs.choose_global_landscape_sample_sites(
                np.random.default_rng(1), positions, edges, mask, 3
            )
    with pytest.raises(ValueError, match="sampleable"):
        motifs.choose_global_landscape_sample_sites(
            np.random.default_rng(1), positions, edges, np.zeros(12, dtype=bool), 3
        )


def test_graph_indices_cannot_wrap_or_truncate(line) -> None:
    positions, _ = line
    for edges in (np.asarray([[0, -1]]), np.asarray([[0, 12]])):
        with pytest.raises(ValueError, match="indices"):
            motifs.choose_path_sample_sites(np.random.default_rng(1), positions, edges, 3)
        with pytest.raises(ValueError, match="indices"):
            geography.largest_graph_component(positions, edges)
    with pytest.raises(TypeCheckError):
        motifs.choose_path_sample_sites(
            np.random.default_rng(1), positions, np.asarray([[0.0, 1.5]]), 3
        )


def test_predecessor_path_and_unreachable_sentinel() -> None:
    predecessors = np.asarray([-9999, 0, 1, -9999], dtype=np.int32)
    np.testing.assert_array_equal(motifs.reconstruct_shortest_path(predecessors, 0, 2), [0, 1, 2])
    assert motifs.reconstruct_shortest_path(predecessors, 0, 3).size == 0
    assert motifs.reconstruct_shortest_path(np.asarray([-9999, 2, 1]), 0, 2).size == 0
    with pytest.raises(ValueError, match="endpoints"):
        motifs.reconstruct_shortest_path(predecessors, -1, 2)
    with pytest.raises(ValueError, match="out-of-range"):
        motifs.reconstruct_shortest_path(np.asarray([-9999, 5]), 0, 1)
