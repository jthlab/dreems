"""QC and missing-data behavior of the public genotype adapter."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from dreems.data import genotypes


@pytest.fixture
def calls() -> xr.Dataset:
    rng = np.random.default_rng(509)
    return xr.Dataset(
        {
            "call_genotype": (
                ("variants", "samples", "ploidy"),
                rng.binomial(1, 0.4, size=(12, 6, 2)).astype(np.int8),
            ),
            "sample_id": ("samples", ["a", "b", "c", "d", "e", "f"]),
        }
    ).assign_coords(
        sample_latitude=("samples", [42.0, 42.01, 42.5, 42.51, 43.0, 43.01]),
        sample_longitude=("samples", [-84.0, -84.01, -83.5, -83.51, -83.0, -83.01]),
    )


def convert(dataset: xr.Dataset, **kwargs) -> xr.Dataset:
    return genotypes.from_genotype_dataset(
        dataset,
        node_latitude=np.asarray([42.0, 42.5, 43.0]),
        node_longitude=np.asarray([-84.0, -83.5, -83.0]),
        edges=np.asarray([[0, 1], [1, 2]], dtype=np.int32),
        **kwargs,
    )


def test_explicit_missing_mask_matches_negative_allele_codes(calls) -> None:
    masked = calls.copy(deep=True)
    mask = np.zeros_like(calls.call_genotype, dtype=bool)
    mask[::2, 0, 0] = True  # One missing allele invalidates the whole sample call.
    masked["call_genotype_mask"] = (calls.call_genotype.dims, mask)
    coded = calls.copy(deep=True)
    coded.call_genotype.values[mask] = -1
    xr.testing.assert_identical(convert(masked), convert(coded))
    assert np.all(masked.call_genotype.values >= 0)  # Conversion must not alter calls.


def test_transposed_genotype_and_mask_axes_are_supported(calls) -> None:
    calls["call_genotype_mask"] = calls.call_genotype < 0
    reordered = calls.copy()
    reordered["call_genotype"] = calls.call_genotype.transpose("samples", "ploidy", "variants")
    reordered["call_genotype_mask"] = calls.call_genotype_mask.transpose(
        "ploidy", "variants", "samples"
    )
    xr.testing.assert_identical(convert(reordered), convert(calls))


@pytest.mark.parametrize(
    "invalid",
    ["float_calls", "mask_type", "mask_dimensions", "fill", "masked_fill", "declared_mixed"],
)
def test_invalid_or_mixed_ploidy_inputs_fail_explicitly(calls, invalid: str) -> None:
    message = "mixed.ploidy"
    if invalid == "float_calls":
        calls["call_genotype"] = calls.call_genotype.astype(float) + 0.5
        message = "signed integer"
    elif invalid == "mask_type":
        calls["call_genotype_mask"] = xr.zeros_like(calls.call_genotype)
        message = "must be boolean"
    elif invalid == "mask_dimensions":
        calls["call_genotype_mask"] = xr.zeros_like(calls.call_genotype.isel(ploidy=0), dtype=bool)
        message = "dimensions"
    elif invalid == "declared_mixed":
        calls.call_genotype.attrs["mixed_ploidy"] = True
    else:
        calls.call_genotype.values[0, 0, 1] = -2
        if invalid == "masked_fill":
            calls["call_genotype_mask"] = calls.call_genotype < 0
    with pytest.raises(ValueError, match=message):
        convert(calls)


def test_entirely_missing_sample_is_not_counted_as_an_observation(calls) -> None:
    calls.call_genotype.values[:, 0, :] = -1
    with pytest.raises(ValueError, match="no called genotypes.*sample indices: \\[0\\]"):
        convert(calls)


def test_samples_must_have_calls_in_retained_variants(calls) -> None:
    calls.call_genotype.values[:, 0, :] = -1
    calls.call_genotype.values[0, 0, :] = 1
    calls.call_genotype.values[0, 1, :] = 2  # Their only observed variant is removed.
    with pytest.raises(ValueError, match="no called genotypes"):
        convert(calls)


@pytest.mark.parametrize("latitude", [np.nan, np.inf, 91.0, -91.0])
def test_existing_sample_coordinates_are_validated(calls, latitude: float) -> None:
    calls.sample_latitude.values[0] = latitude
    with pytest.raises(ValueError, match="sample latitude"):
        convert(calls)


@pytest.mark.parametrize("coordinate", ["sample_latitude", "sample_longitude"])
def test_complex_sample_coordinates_are_rejected(calls, coordinate) -> None:
    calls = calls.assign_coords({coordinate: calls[coordinate].astype(complex) + 0.01j})
    with pytest.raises(ValueError, match="real numbers"):
        convert(calls)


@pytest.mark.parametrize("coordinate", ["node_latitude", "node_longitude"])
def test_complex_node_coordinates_are_rejected(calls, coordinate) -> None:
    values = dict(
        node_latitude=np.asarray([42.0, 42.5, 43.0]),
        node_longitude=np.asarray([-84.0, -83.5, -83.0]),
        edges=np.asarray([[0, 1], [1, 2]], dtype=np.int32),
    )
    values[coordinate] = values[coordinate].astype(complex) + 0.01j
    with pytest.raises(ValueError, match="real numbers"):
        genotypes.from_genotype_dataset(calls, **values)


def test_qc_covariance_matches_a_direct_numpy_calculation(calls) -> None:
    calls.call_genotype.values[::2, 0, :] = -1
    output = convert(calls)
    raw = calls.call_genotype.values
    called = (raw >= 0).all(axis=2)
    # Xarray promotes masked int8 alleles to float32 before dosage reduction.
    dosage = np.where(called, raw.sum(axis=2).astype(np.float32), np.float32(np.nan))
    frequency = np.nansum(dosage, axis=1) / (2 * called.sum(axis=1))
    keep = (called.sum(axis=1) >= 2) & (called.mean(axis=1) >= 0.8)
    keep &= (np.minimum(frequency, 1 - frequency) >= 0.05) & (raw.max(axis=(1, 2)) <= 1)
    dosage = dosage[keep]
    imputed = np.where(np.isnan(dosage), np.nanmean(dosage, axis=1)[:, None], dosage)
    centered = imputed - imputed.mean(axis=1)[:, None]
    sd = centered.std(axis=1, ddof=1)
    standardized = centered[sd > 1e-8] / sd[sd > 1e-8, None]
    # The adapter's averaging matrix is float64.
    deme_mean = standardized.astype(np.float64).reshape(-1, 3, 2).mean(axis=2)
    expected = deme_mean.T @ deme_mean / len(deme_mean)
    np.testing.assert_allclose(output.node_covariance, expected, atol=1e-14)
    assert output.attrs["retained_variants"] == len(deme_mean)
    np.testing.assert_array_equal(output.node_sample_count, [2, 2, 2])


def test_chunked_and_eager_inputs_give_the_same_covariance(calls) -> None:
    pytest.importorskip("dask.array")
    calls.call_genotype.values[::2, 0, 0] = -1
    xr.testing.assert_allclose(convert(calls.chunk({"variants": 3, "samples": 2})), convert(calls))


def test_numpy_numeric_scalars_remain_valid_api_inputs(calls) -> None:
    output = convert(calls, min_maf=np.float32(0.05), effective_snps=np.int64(10))
    assert int(output.effective_snps) == 10
    with pytest.raises(ValueError, match="integer"):
        convert(calls, effective_snps=10.5)
