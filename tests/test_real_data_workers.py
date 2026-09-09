"""The real-data workers must preserve input identity and render the supplied graph."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from PIL import Image

from dreems import workflow_io

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "src/dreems/data/dreems_diffusion_bands.dreems.zip"


@pytest.fixture(scope="module")
def workers():
    modules = {}
    for stem in ("run_real_data", "render_real_data_gallery", "calibrate_checkpoint"):
        name = "tested_" + stem
        spec = importlib.util.spec_from_file_location(name, ROOT / "workflow" / f"{stem}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules[stem] = module
    return SimpleNamespace(**modules)


@pytest.fixture
def inputs(tmp_path):
    positions = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    arrays = {
        "sample_covariance": np.asarray([[1.0, 0.1, 0.2], [0.1, 1.0, 0.3], [0.2, 0.3, 1.0]]),
        "sample_xy": positions,
        "node_xy": positions,
        "node_lonlat": positions,
        "sample_lonlat": positions,
        "edges": np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int32),
        "sample_nodes": np.arange(3, dtype=np.int32),
    }
    args = Namespace(
        prepared_npz=tmp_path / "prepared.npz",
        prepared_json=tmp_path / "prepared.json",
        checkpoint=WEIGHTS,
        output_npz=tmp_path / "result.npz",
        output_json=tmp_path / "result.json",
    )
    metadata = {"dataset_id": "fixture", "effective_snps": 100}
    workflow_io.write_npz(args.prepared_npz, **arrays)
    workflow_io.write_json(args.prepared_json, metadata)
    return args, arrays, metadata


@pytest.mark.parametrize(
    "field", ["edges", "sample_covariance", "sample_xy", "node_xy", "effective_snps"]
)
def test_runner_rejects_conversion_loss_before_loading_model(workers, inputs, monkeypatch, field):
    args, arrays, metadata = inputs
    if field == "effective_snps":
        metadata[field] = 3.5
        workflow_io.write_json(args.prepared_json, metadata)
    else:
        arrays[field] = arrays[field].astype(float if field == "edges" else complex)
        arrays[field].flat[0] += 0.5 if field == "edges" else 1j
        workflow_io.write_npz(args.prepared_npz, **arrays)
    monkeypatch.setattr(workers.run_real_data, "load_model", lambda _: pytest.fail("model loaded"))
    with pytest.raises(ValueError, match="integer|real"):
        workers.run_real_data.run_prediction(args)
    assert not args.output_npz.exists()


def test_runner_rejects_input_changed_during_prediction(workers, inputs, monkeypatch):
    args, arrays, metadata = inputs
    args.output_npz.write_bytes(b"previous prediction")

    def predict(example, *, refinement_config):
        np.testing.assert_array_equal(example.edges, arrays["edges"])
        workflow_io.write_json(args.prepared_json, {**metadata, "changed": True})
        return SimpleNamespace(
            centered_edge_log_mean=np.zeros(3),
            centered_refined_edge_log_mean=np.zeros(3),
            edge_log_sd=np.ones(3),
            raw_edge_log_sd=np.ones(3),
            surface_weight=1.0,
            refinement_diagnostics={},
        )

    predictor = SimpleNamespace(
        predict=predict, checkpoint=SimpleNamespace(model={"architecture": {"kind": "test"}})
    )
    monkeypatch.setattr(workers.run_real_data, "load_model", lambda _: predictor)
    with pytest.raises(ValueError, match="input changed"):
        workers.run_real_data.run_prediction(args)
    assert args.output_npz.read_bytes() == b"previous prediction"


def test_browser_json_uses_null_inside_arrays_and_accepts_numpy_booleans(workers, tmp_path):
    value = {
        "array": np.asarray([np.nan, np.inf, -np.inf, 2.0]),
        "flag": np.bool_(True),
        "nested": [float("nan")],
    }
    expected = {"array": [None, None, None, 2.0], "flag": True, "nested": [None]}
    assert workers.run_real_data.json_ready(value) == expected
    assert workers.render_real_data_gallery.json_ready(value) == expected
    for write in (
        workers.run_real_data.write_json_atomic,
        workers.render_real_data_gallery.atomic_write_json,
    ):
        path = tmp_path / "report.json"
        write(path, value)
        assert (
            json.loads(path.read_text(), parse_constant=lambda _: pytest.fail("nonstandard JSON"))
            == expected
        )


def test_failed_json_write_preserves_previous_report(workers, tmp_path):
    path = tmp_path / "report.json"
    path.write_text("previous")
    with pytest.raises(TypeError):
        workers.run_real_data.write_json_atomic(path, {"bad": object()})
    assert path.read_text() == "previous"
    assert list(tmp_path.iterdir()) == [path]


def test_invalid_calibration_arrays_do_not_lose_imaginary_parts(workers, tmp_path):
    worker = workers.calibrate_checkpoint
    with pytest.raises(ValueError, match="real"):
        worker.truth_edge_logs({"truth": np.asarray([1j, 0j])})
    path = tmp_path / "case.npz"
    for field in ("error", "raw_sd", "sampled_node_fraction"):
        arrays = {
            "error": np.ones(2),
            "raw_sd": np.ones(2),
            "sampled_node_fraction": np.asarray(0.5),
        }
        arrays[field] = np.asarray(arrays[field].astype(complex) + 1j)
        workflow_io.write_npz(path, **arrays)
        with pytest.raises(ValueError, match="real"):
            worker.load_cases([path])


def test_tile_download_uses_tls_and_checks_image_before_caching(workers, tmp_path, monkeypatch):
    renderer = workers.render_real_data_gallery
    stream = io.BytesIO()
    Image.new("RGB", (256, 256), "green").save(stream, format="PNG")
    body = stream.getvalue()
    calls = []

    def get(url, **kwargs):
        assert kwargs.get("verify", True) is True
        assert url.startswith("https://")
        calls.append(url)
        return SimpleNamespace(content=body, raise_for_status=lambda: None)

    monkeypatch.setattr(renderer.requests, "get", get)
    kwargs = dict(cache_dir=tmp_path, zoom=2, tile_x=1, tile_y=1, headers={})
    path = renderer.fetch_esri_topo_tile(**kwargs)
    assert path.read_bytes() == body
    assert renderer.fetch_esri_topo_tile(**kwargs) == path
    assert len(calls) == 1
    body = b"not an image"
    with pytest.raises(OSError):
        renderer.fetch_esri_topo_tile(**{**kwargs, "tile_y": 2})
    assert not (tmp_path / "z2/1/2.jpg").exists()


def test_basemap_fallback_warns_but_does_not_hide_programming_errors(
    workers, tmp_path, monkeypatch
):
    renderer = workers.render_real_data_gallery
    kwargs = dict(
        extent=(-2.0, 2.0, -1.0, 1.0),
        cache_dir=tmp_path,
        maximum_zoom=2,
        maximum_tiles=4,
        maximum_pixels=256,
        fetch_topo=True,
    )

    def missing(*args, **kwargs):
        raise renderer.requests.ConnectionError("offline")

    monkeypatch.setattr(renderer, "fetch_esri_topo_mosaic", missing)
    with pytest.warns(UserWarning, match="stock imagery"):
        result = renderer.load_topo_basemap(**kwargs)
    assert result.image is None

    def broken(*args, **kwargs):
        raise ValueError("bad bounds")

    monkeypatch.setattr(renderer, "fetch_esri_topo_mosaic", broken)
    with pytest.raises(ValueError, match="bad bounds"):
        renderer.load_topo_basemap(**kwargs)


def test_interpolation_is_clipped_to_graph_faces_and_accepts_integer_positions(workers):
    renderer = workers.render_real_data_gallery
    positions = np.asarray([[0, 0], [1, 0], [0, 1]])
    edges = np.asarray([[0, 1], [0, 2], [1, 2]])
    np.testing.assert_array_equal(
        renderer.edge_midpoints(positions, edges), [[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]]
    )
    image, extent = renderer.interpolate_edges(positions, edges, np.ones(3), grid_side=25)
    xx, yy = np.meshgrid(np.linspace(*extent[:2], 25), np.linspace(*extent[2:], 25))
    support = (xx >= 0) & (yy >= 0) & (xx + yy <= 1)
    assert np.all(np.isnan(image[~support]))
    interior = (xx > 1e-12) & (yy > 1e-12) & (xx + yy < 1 - 1e-12)
    np.testing.assert_allclose(image[interior], 1.0)


@pytest.mark.parametrize(
    "changes",
    [{"grid_side": 0}, {"grid_side": 2.5}, {"neighbors": 0}, {"kernel_bandwidth_edges": -1.0}],
)
def test_interpolation_rejects_bad_controls(workers, changes):
    with pytest.raises((ValueError, BeartypeCallHintParamViolation)):
        workers.render_real_data_gallery.interpolate_edges(
            np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            np.asarray([[0, 1], [0, 2], [1, 2]]),
            np.ones(3),
            **{"grid_side": 10, **changes},
        )


@pytest.mark.parametrize(
    "change", ["dataset", "hash", "fractional_edges", "prior_shape", "sd_shape", "negative_sd"]
)
def test_renderer_rejects_misaligned_data_before_plotting(workers, inputs, monkeypatch, change):
    args, arrays, metadata = inputs
    args.dreems_npz = args.output_npz
    args.dreems_json = args.output_json
    prediction = {"prior_edge_log": np.zeros(3), "edge_log": np.zeros(3), "edge_log_sd": np.ones(3)}
    report = {"dataset_id": metadata["dataset_id"]}
    if change == "dataset":
        report["dataset_id"] = "different"
    elif change == "hash":
        report["prepared_npz_sha256"] = "0" * 64
    elif change == "fractional_edges":
        arrays["edges"] = arrays["edges"].astype(float) + 0.1
    elif change.endswith("shape"):
        prediction["prior_edge_log" if change == "prior_shape" else "edge_log_sd"] = np.ones(4)
    else:
        prediction["edge_log_sd"][0] = -1
    workflow_io.write_npz(args.prepared_npz, **arrays)
    workflow_io.write_npz(args.dreems_npz, **prediction)
    workflow_io.write_json(args.dreems_json, report)
    monkeypatch.setattr(
        workers.render_real_data_gallery,
        "load_topo_basemap",
        lambda *a, **k: pytest.fail("started plotting"),
    )
    with pytest.raises(ValueError):
        workers.render_real_data_gallery.render_case(args)
