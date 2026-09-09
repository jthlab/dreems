"""A report must not quietly mix cases or hide undefined results."""

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from dreems import benchmark_gallery as gallery
from dreems.workflow_io import read_npz, read_pickle, write_pickle


@pytest.mark.parametrize("value", [True, -1, "1.5", "nan", "inf", None])
def test_ids_and_counts_are_exact(value) -> None:
    with pytest.raises(ValueError):
        gallery.integer({"index": value}, "index")


def test_large_csv_id_is_not_rounded() -> None:
    assert gallery.integer({"index": "9007199254740993"}, "index") == 9007199254740993
    assert gallery.integer({"index": "12.0"}, "index") == 12
    with pytest.raises(ValueError, match="duplicate"):
        gallery._rows_by_id([{"index": "12"}, {"index": "12.0"}], "index")


@pytest.mark.parametrize(
    "change",
    [
        {"replicate": np.array([0, 0])},
        {"replicate": np.array([0.0, 1.5])},
        {"edge_offsets": np.array([1, 2, 4])},
        {"edge_offsets": np.array([0, 0, 4])},
        {"edge_offsets": np.array([0, 2, 5])},
        {"edge_offsets": np.array([0.0, 2.5, 4.0])},
        {"edge_sd": np.array([-1.0, 0.1, 0.1, 0.1])},
        {"edge_mean": np.full(4, 1j)},
        {"edge_mean": np.full(4, np.inf)},
    ],
)
def test_packed_predictions_reject_invalid_slices(tmp_path: Path, change: dict) -> None:
    arrays = dict(
        replicate=np.array([0, 1]),
        edge_offsets=np.array([0, 2, 4]),
        edge_truth=np.arange(4.0),
        edge_mean=np.arange(4.0),
        edge_sd=np.ones(4),
    )
    np.savez(tmp_path / "pack.npz", **(arrays | change))
    with pytest.raises((ValueError, TypeError)):
        gallery.prediction_slice(tmp_path / "pack.npz", 0)


def test_replacing_saved_inputs_is_visible(tmp_path: Path) -> None:
    archive = tmp_path / "case.npz"
    pack = tmp_path / "pack.pkl"
    for value in (1, 2):
        np.savez(archive, value=np.array([value]))
        write_pickle(pack, [{"value": value}])
        assert read_npz(archive)["value"][0] == value
        assert read_pickle(pack)[0]["value"] == value


@pytest.fixture
def saved_pair(tmp_path: Path) -> tuple[dict, dict]:
    common = dict(
        positions=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        edges=np.array([[0, 1], [1, 2], [2, 0]]),
        truth=np.array([-0.5, 0.0, 0.5]),
        sample_count=np.array([1.0, 0.0, 2.0]),
    )
    np.savez(tmp_path / "dreems_0.npz", **common, dreems_map=common["truth"], edge_sd=np.ones(3))
    np.savez(tmp_path / "feems_0.npz", **common, feems=common["truth"])
    kwargs = dict(
        dreems_pattern=str(tmp_path / "dreems_{example}.npz"),
        feems_pattern=str(tmp_path / "feems_{example}.npz"),
    )
    return common, kwargs


@pytest.mark.parametrize("key", ["positions", "edges", "truth", "sample_count"])
def test_methods_must_describe_the_same_case(tmp_path: Path, saved_pair: tuple, key: str) -> None:
    common, kwargs = saved_pair
    changed = dict(common)
    changed[key] = common[key][::-1]
    np.savez(tmp_path / "feems_0.npz", **changed, feems=common["truth"])
    with pytest.raises(ValueError, match=key):
        gallery._matched_predictions(0, **kwargs)


@pytest.mark.parametrize(
    "key,value",
    [
        ("edge_sd", np.array([-1.0, 1.0, 1.0])),
        ("dreems_map", np.ones(2)),
        ("dreems_map", np.ones(3) * 1j),
    ],
)
def test_surface_vectors_are_validated(tmp_path: Path, saved_pair: tuple, key: str, value) -> None:
    common, kwargs = saved_pair
    arrays = common | dict(dreems_map=common["truth"], edge_sd=np.ones(3)) | {key: value}
    np.savez(tmp_path / "dreems_0.npz", **arrays)
    with pytest.raises((ValueError, TypeError), match=key):
        gallery._matched_predictions(0, **kwargs)


def test_undefined_metrics_are_not_ranked_as_best_or_worst() -> None:
    rows = [dict(index=0, replicate=0, num_snps=100, sampled_node_fraction=0.1)]
    selections = gallery.h2h_selections(rows)
    assert selections["Random examples"] == [0]
    assert selections["Largest DREEMS correlation gains"] == []
    assert selections["Largest DREEMS MSE losses"] == []
    router = gallery.router_selections(rows)
    assert router["Router helps most"] == []
    assert router["Router hurts most"] == []


@pytest.mark.parametrize("errors", [[0.0, 0.0], [0.0, 1.0], [np.nan, np.nan], [0.1, 1.0]])
def test_scatter_handles_zero_and_undefined_metrics(tmp_path: Path, monkeypatch, errors) -> None:
    inspected = []

    def capture(fig, path):
        inspected.append(True)
        expected = "log" if np.all(np.asarray(errors) > 0) else "linear"
        assert fig.axes[1].get_xscale() == expected
        assert np.all(np.isfinite(fig.axes[0].lines[0].get_xdata()))

    monkeypatch.setattr(gallery, "save_figure", capture)
    gallery._comparison_scatter(
        np.full(2, np.nan),
        np.full(2, np.nan),
        np.array(errors),
        np.array(errors),
        np.ones(2),
        labels=("A", "B"),
        color_label="samples",
        output=tmp_path / "plot.png",
        figsize=(5.0, 3.0),
        cmap="viridis",
        size=9.0,
        alpha=0.5,
    )
    assert inspected == [True]


def test_failed_figure_write_keeps_previous_file(tmp_path: Path) -> None:
    class BrokenFigure:
        def savefig(self, stream, **kwargs):
            stream.write(b"partial")
            raise OSError("full disk")

    image = tmp_path / "plot.png"
    image.write_bytes(b"previous")
    with pytest.raises(OSError, match="full disk"):
        gallery.save_figure(BrokenFigure(), image)
    assert image.read_bytes() == b"previous"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["plot.png"]


@pytest.mark.parametrize("bad_input", ["heading", "provenance", "overlap"])
def test_invalid_reference_does_not_overwrite_gallery(tmp_path: Path, bad_input: str) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "index.html").write_text("missing" if bad_input == "heading" else "<h1>Title</h1>")
    (destination / "index.html").write_text("previous gallery")
    if bad_input == "provenance":
        (source / "provenance.json").write_text("{")
    gallery.write_rows(tmp_path / "rows.csv", [{"index": 0, "num_snps": 1000}])
    with pytest.raises(ValueError):
        gallery.command_router_h2h_gallery(
            Namespace(
                reference_gallery=source,
                reference_rows=tmp_path / "rows.csv",
                threshold=512,
                output_dir=source / "nested" if bad_input == "overlap" else destination,
            )
        )
    assert (destination / "index.html").read_text() == "previous gallery"


def test_snp_change_figure_uses_actual_count(tmp_path: Path, monkeypatch) -> None:
    labels = []
    monkeypatch.setattr(
        gallery, "save_figure", lambda fig, path: labels.append(fig.axes[0].get_ylabel())
    )
    gallery._scenario_snp_delta_figure(
        [
            dict(
                truth="barrier",
                mask="full",
                high_snps=500,
                **{
                    f"{method}_{metric}": 0.1
                    for method in ("dreems", "dreems_map", "feems")
                    for metric in ("corr_gain", "mse_reduction")
                },
            )
        ],
        tmp_path / "changes.png",
    )
    assert labels == ["Correlation gain at 500 SNPs"]
