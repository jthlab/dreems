"""Construction of DREEMS datasets from sgkit-compatible genotypes."""

# ruff: noqa: F722,F821,UP037

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np
import xarray as xr
from beartype import beartype
from jaxtyping import Float, Int, Shaped, jaxtyped
from scipy.spatial import cKDTree

from dreems.data.datasets import (
    NODE_DIM,
    OBSERVED_NODE_2_DIM,
    OBSERVED_NODE_DIM,
    create_dataset,
)
from dreems.utils.validation import _effective_snp_count, _real_array

SAMPLE_ID = "sample_id"
SAMPLE_LATITUDE = "sample_latitude"
SAMPLE_LONGITUDE = "sample_longitude"


@beartype
def attach_sample_coordinates(
    dataset: xr.Dataset,
    coordinates: Mapping[str, tuple[Real, Real]],
    *,
    sample_id: str = SAMPLE_ID,
) -> xr.Dataset:
    """Attach ``sample_latitude`` and ``sample_longitude`` coordinates.

    The mapping values are ``(latitude, longitude)``. Sample IDs are matched
    against the sgkit-standard ``sample_id`` variable.
    """

    if sample_id not in dataset:
        raise ValueError(f"genotype dataset does not contain {sample_id!r}")
    if dataset[sample_id].dims != ("samples",):
        raise ValueError(f"{sample_id} must have dimensions ('samples',)")
    ids = [_sample_id_text(value) for value in np.asarray(dataset[sample_id].values)]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{sample_id} values must be unique")
    normalized = {str(key): value for key, value in coordinates.items()}
    missing = [identifier for identifier in ids if identifier not in normalized]
    if missing:
        preview = ", ".join(repr(value) for value in missing[:5])
        suffix = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
        raise ValueError(f"sample coordinates are missing for {preview}{suffix}")

    latitude = np.asarray([normalized[identifier][0] for identifier in ids], dtype=np.float64)
    longitude = np.asarray([normalized[identifier][1] for identifier in ids], dtype=np.float64)
    if not np.all(np.isfinite(latitude)) or not np.all(np.isfinite(longitude)):
        raise ValueError("sample latitude and longitude must be finite")
    if np.any(latitude < -90.0) or np.any(latitude > 90.0):
        raise ValueError("sample latitude must be between -90 and 90 degrees")
    return dataset.assign_coords(
        {
            SAMPLE_LATITUDE: (
                ("samples",),
                latitude,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            SAMPLE_LONGITUDE: (
                ("samples",),
                longitude,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
        }
    )


@beartype
def from_vcf_zarr(
    dataset: xr.Dataset,
    *,
    node_latitude: Shaped[np.ndarray, "nodes"],
    node_longitude: Shaped[np.ndarray, "nodes"],
    edges: Shaped[np.ndarray, "edges 2"],
    sample_coordinates: Mapping[str, tuple[Real, Real]] | None = None,
    sample_id: str = SAMPLE_ID,
    min_maf: Real = 0.05,
    min_variant_call_rate: Real = 0.8,
    effective_snps: Real | None = None,
    call_genotype: str = "call_genotype",
    attrs: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """Build a DREEMS dataset from sgkit-compatible calls, including opened VCF-Zarr.

    Input genotypes must be signed integer allele codes, with fixed positive
    ploidy and dimensions ``(variants, samples, ploidy)``. A -1 allele or a true
    entry in ``{call_genotype}_mask`` makes the whole sample call missing. Mixed
    ploidy (-2 padding) is not supported. Missing calls are mean-imputed after
    biallelic, call-rate, and MAF filtering. This does not prune linked SNPs or
    apply read-depth or genotype-quality thresholds. ``effective_snps`` can be supplied
    when linkage makes the retained SNP count overstate independent information.
    Supply ``sample_coordinates`` as ``sample_id -> (latitude, longitude)`` or
    attach ``sample_latitude`` and ``sample_longitude`` beforehand.
    """

    if sample_coordinates is not None:
        dataset = attach_sample_coordinates(
            dataset,
            sample_coordinates,
            sample_id=sample_id,
        )
    if not 0.0 <= float(min_maf) < 0.5:
        raise ValueError("min_maf must be between 0 and 0.5")
    if not 0.0 <= float(min_variant_call_rate) <= 1.0:
        raise ValueError("min_variant_call_rate must be between 0 and 1")
    if call_genotype not in dataset:
        raise ValueError(f"genotype dataset does not contain {call_genotype!r}")
    for coordinate in (SAMPLE_LATITUDE, SAMPLE_LONGITUDE):
        if coordinate not in dataset:
            raise ValueError(
                f"genotype dataset does not contain {coordinate!r}; "
                "use attach_sample_coordinates() first"
            )
        if dataset[coordinate].dims != ("samples",):
            raise ValueError(f"{coordinate} must have dimensions ('samples',)")

    calls = dataset[call_genotype]
    required_dims = ("variants", "samples", "ploidy")
    if set(calls.dims) != set(required_dims):
        raise ValueError(f"{call_genotype} must have dimensions {required_dims}")
    calls = calls.transpose(*required_dims)
    if calls.dtype.kind != "i":
        raise ValueError(f"{call_genotype} must contain signed integer allele codes")
    if calls.attrs.get("mixed_ploidy", False):
        raise ValueError("mixed ploidy is not supported; select samples with a common ploidy")
    ploidy = int(calls.sizes["ploidy"])
    if ploidy < 1:
        raise ValueError("genotypes must have positive ploidy")
    if calls.sizes["samples"] < 2:
        raise ValueError("at least two samples are required")

    mask_name = f"{call_genotype}_mask"
    if mask_name in dataset:
        mask = dataset[mask_name]
        if set(mask.dims) != set(required_dims) or mask.dtype.kind != "b":
            raise ValueError(f"{mask_name} must be boolean with dimensions {required_dims}")
        # Preserve fill values so the uniform-ploidy check can still reject them.
        calls = calls.where(~mask | (calls < -1), other=-1)

    sample_latitude = _real_array(dataset[SAMPLE_LATITUDE].values, name="sample latitude")
    sample_longitude = _real_array(dataset[SAMPLE_LONGITUDE].values, name="sample longitude")
    if np.any(np.abs(sample_latitude) > 90.0):
        raise ValueError("sample latitude must be between -90 and 90 degrees")
    node_latitude = _real_array(node_latitude, name="node latitude")
    node_longitude = _real_array(node_longitude, name="node longitude")
    _validate_node_geography(node_latitude, node_longitude)

    standardized, retained_snps = _standardized_genotypes(
        calls,
        min_maf=float(min_maf),
        min_variant_call_rate=float(min_variant_call_rate),
    )
    precision_snps = _effective_snp_count(
        retained_snps if effective_snps is None else effective_snps
    )

    node_positions, sample_positions, projection_attrs = _project_local_azimuthal_equidistant(
        node_latitude=node_latitude,
        node_longitude=node_longitude,
        sample_latitude=sample_latitude,
        sample_longitude=sample_longitude,
    )
    sample_nodes = np.asarray(cKDTree(node_positions).query(sample_positions)[1], dtype=np.int32)
    node_sample_count = np.bincount(
        sample_nodes,
        minlength=node_positions.shape[0],
    ).astype(np.int64)
    observed_nodes = np.flatnonzero(node_sample_count > 0).astype(np.int32)
    if observed_nodes.size < 2:
        raise ValueError("samples map to fewer than two graph nodes")

    covariance = _node_covariance(
        standardized,
        sample_nodes=sample_nodes,
        node_sample_count=node_sample_count,
        observed_nodes=observed_nodes,
    )
    output_attrs: dict[str, Any] = {
        **projection_attrs,
        "genotype_qc": (f"biallelic uniform-ploidy ({ploidy}x); mean imputed; SNP standardized"),
        "ploidy": ploidy,
        "min_maf": float(min_maf),
        "min_variant_call_rate": float(min_variant_call_rate),
        "input_variants": int(calls.sizes["variants"]),
        "retained_variants": retained_snps,
        "input_samples": int(calls.sizes["samples"]),
    }
    if attrs is not None:
        output_attrs.update(dict(attrs))
    output = create_dataset(
        node_positions=node_positions,
        edges=edges,
        node_covariance=covariance,
        observed_nodes=observed_nodes,
        node_sample_count=node_sample_count,
        effective_snps=precision_snps,
        coordinate_names=("easting", "northing"),
        attrs=output_attrs,
    )
    return output.assign_coords(
        {
            "node_latitude": (
                (NODE_DIM,),
                node_latitude,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "node_longitude": (
                (NODE_DIM,),
                node_longitude,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
        }
    )

@beartype
def _standardized_genotypes(
    calls: xr.DataArray,
    *,
    min_maf: float,
    min_variant_call_rate: float,
) -> tuple[xr.DataArray, int]:
    """Filter calls, impute variant means, and scale dosages to sample variance one.

    Reductions stay in xarray, so a chunked input does not require loading the
    full genotype tensor. Small QC summaries are computed before indexing.
    """
    sample_called = (calls >= 0).all(dim="ploidy")
    dosage = calls.where(sample_called).sum(dim="ploidy", skipna=True).where(sample_called)
    called_count = sample_called.sum(dim="samples")
    call_rate = called_count / float(calls.sizes["samples"])
    allele_frequency = dosage.sum(dim="samples", skipna=True) / (
        float(calls.sizes["ploidy"]) * called_count
    )
    maf = xr.where(allele_frequency <= 0.5, allele_frequency, 1.0 - allele_frequency)
    max_called_allele = calls.where(calls >= 0, other=0).max(dim=("samples", "ploidy"))
    keep = (
        (called_count >= 2)
        & (call_rate >= min_variant_call_rate)
        & (maf >= min_maf)
        & (max_called_allele <= 1)
    )
    qc = xr.Dataset({"keep": keep, "has_fill": (calls < -1).any()}).compute()
    if bool(qc["has_fill"]):
        raise ValueError("mixed-ploidy fill values are not supported; use fixed ploidy calls")
    retained_index = np.flatnonzero(np.asarray(qc["keep"].values, dtype=bool))
    if retained_index.size == 0:
        raise ValueError("no variants remain after biallelic, call-rate, and MAF filtering")

    filtered = dosage.isel(variants=retained_index)
    variant_mean = filtered.mean(dim="samples", skipna=True)
    imputed = filtered.fillna(variant_mean)
    centered = imputed - imputed.mean(dim="samples")
    standard_deviation = centered.std(dim="samples", ddof=1)
    variable = (standard_deviation > 1.0e-8) & np.isfinite(standard_deviation)
    qc = xr.Dataset(
        {
            "usable": variable,
            "sample_has_calls": (sample_called.isel(variants=retained_index) & variable).any(
                "variants"
            ),
        }
    ).compute()
    usable = np.asarray(qc["usable"].values, dtype=bool)
    usable_index = np.flatnonzero(usable)
    if usable_index.size == 0:
        raise ValueError("no variable SNPs remain after genotype imputation")
    missing_samples = np.flatnonzero(~np.asarray(qc["sample_has_calls"].values, dtype=bool))
    if missing_samples.size:
        raise ValueError(
            "samples have no called genotypes among retained variants; remove them before "
            f"building covariance (sample indices: {missing_samples[:10].tolist()})"
        )
    standardized = centered.isel(variants=usable_index) / standard_deviation.isel(
        variants=usable_index
    )
    return standardized, int(usable_index.size)


@jaxtyped(typechecker=beartype)
def _node_covariance(
    standardized: xr.DataArray,
    *,
    sample_nodes: Int[np.ndarray, "samples"],
    node_sample_count: Int[np.ndarray, "nodes"],
    observed_nodes: Int[np.ndarray, "observed"],
) -> Float[np.ndarray, "observed observed"]:
    """Average standardized samples by deme before forming their covariance."""
    weights = np.zeros((observed_nodes.size, sample_nodes.size), dtype=np.float64)
    observed_lookup = np.full(node_sample_count.shape[0], -1, dtype=np.int32)
    observed_lookup[observed_nodes] = np.arange(observed_nodes.size, dtype=np.int32)
    observed_row = observed_lookup[sample_nodes]
    weights[observed_row, np.arange(sample_nodes.size)] = 1.0 / node_sample_count[sample_nodes]
    weight_coords: dict[str, Any] = {OBSERVED_NODE_DIM: observed_nodes}
    if "samples" in standardized.coords:
        weight_coords["samples"] = standardized.coords["samples"]
    weight_array = xr.DataArray(
        weights,
        dims=(OBSERVED_NODE_DIM, "samples"),
        coords=weight_coords,
    )
    deme_standardized = xr.dot(weight_array, standardized, dim="samples")
    right = deme_standardized.rename({OBSERVED_NODE_DIM: OBSERVED_NODE_2_DIM})
    retained_snps = int(standardized.sizes["variants"])
    covariance_array = xr.dot(deme_standardized, right, dim="variants") / float(retained_snps)
    covariance = np.asarray(covariance_array.compute().values, dtype=np.float64)
    return 0.5 * (covariance + covariance.T)


@jaxtyped(typechecker=beartype)
def _validate_node_geography(
    node_latitude: Float[np.ndarray, "nodes"],
    node_longitude: Float[np.ndarray, "nodes"],
) -> None:
    """Validate graph coordinates before genotype reductions or projection."""
    if node_latitude.ndim != 1 or node_longitude.shape != node_latitude.shape:
        raise ValueError("node_latitude and node_longitude must be matching vectors")
    if node_latitude.shape[0] < 2:
        raise ValueError("at least two graph nodes are required")
    if np.any(node_latitude < -90.0) or np.any(node_latitude > 90.0):
        raise ValueError("node_latitude must be between -90 and 90 degrees")
    if not np.all(np.isfinite(node_latitude)) or not np.all(np.isfinite(node_longitude)):
        raise ValueError("node latitude and longitude must be finite")


@jaxtyped(typechecker=beartype)
def _project_local_azimuthal_equidistant(
    *,
    node_latitude: Float[np.ndarray, "nodes"],
    node_longitude: Float[np.ndarray, "nodes"],
    sample_latitude: Float[np.ndarray, "samples"],
    sample_longitude: Float[np.ndarray, "samples"],
) -> tuple[Float[np.ndarray, "nodes 2"], Float[np.ndarray, "samples 2"], dict[str, Any]]:
    """Project the shared geographic extent to kilometers, including dateline crossings."""
    latitude = np.concatenate((node_latitude, sample_latitude))
    longitude = np.concatenate((node_longitude, sample_longitude))
    latitude_radians = np.deg2rad(latitude)
    longitude_radians = np.deg2rad(longitude)
    mean_x = float(np.mean(np.cos(latitude_radians) * np.cos(longitude_radians)))
    mean_y = float(np.mean(np.cos(latitude_radians) * np.sin(longitude_radians)))
    mean_z = float(np.mean(np.sin(latitude_radians)))
    if np.sqrt(mean_x * mean_x + mean_y * mean_y + mean_z * mean_z) < 1.0e-8:
        raise ValueError("geographic coordinates do not define a stable projection center")
    longitude_center_radians = float(np.arctan2(mean_y, mean_x))
    latitude_center_radians = float(np.arctan2(mean_z, np.hypot(mean_x, mean_y)))
    latitude_center = float(np.rad2deg(latitude_center_radians))
    longitude_center = float(np.rad2deg(longitude_center_radians))
    if not np.isfinite(latitude_center) or not np.isfinite(longitude_center):
        raise ValueError("geographic coordinates do not define a stable projection center")
    radius_km = 6371.0088

    @jaxtyped(typechecker=beartype)
    def project(
        local_latitude: Float[np.ndarray, "locations"],
        local_longitude: Float[np.ndarray, "locations"],
    ) -> Float[np.ndarray, "locations 2"]:
        latitude_value = np.deg2rad(np.asarray(local_latitude, dtype=np.float64))
        longitude_delta = (
            np.deg2rad(np.asarray(local_longitude, dtype=np.float64))
            - longitude_center_radians
            + np.pi
        ) % (2.0 * np.pi) - np.pi
        cosine_distance = np.sin(latitude_center_radians) * np.sin(latitude_value) + np.cos(
            latitude_center_radians
        ) * np.cos(latitude_value) * np.cos(longitude_delta)
        angular_distance = np.arccos(np.clip(cosine_distance, -1.0, 1.0))
        sine_distance = np.sin(angular_distance)
        unstable = (angular_distance > 1.0e-8) & (np.abs(sine_distance) < 1.0e-8)
        if np.any(unstable):
            raise ValueError(
                "geographic extent is too broad for a single local azimuthal projection"
            )
        scale = np.ones_like(angular_distance)
        stable = angular_distance > 1.0e-8
        scale[stable] = angular_distance[stable] / sine_distance[stable]
        x = radius_km * scale * np.cos(latitude_value) * np.sin(longitude_delta)
        y = (
            radius_km
            * scale
            * (
                np.cos(latitude_center_radians) * np.sin(latitude_value)
                - np.sin(latitude_center_radians) * np.cos(latitude_value) * np.cos(longitude_delta)
            )
        )
        return np.column_stack((x, y))

    return (
        project(node_latitude, node_longitude),
        project(sample_latitude, sample_longitude),
        {
            "coordinate_reference_system": "local azimuthal equidistant projection",
            "projection_center_latitude": latitude_center,
            "projection_center_longitude": longitude_center,
            "position_units": "km",
        },
    )


@beartype
def _sample_id_text(value: object) -> str:
    """Decode byte IDs from stores without turning them into strings like b'ID'."""
    if isinstance(value, bytes | np.bytes_):
        return bytes(value).decode()
    return str(value)


__all__ = [
    "attach_sample_coordinates",
    "from_dataset",
]
