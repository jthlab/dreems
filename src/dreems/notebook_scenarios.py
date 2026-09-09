"""Reproduce the exact simulations and sampling masks in ``feems_scenarios.ipynb``.

The notebook simulates one full genotype matrix for each migration surface and
then reuses it across sampling masks. This module preserves that dependency
structure so the expensive msprime simulations run exactly three times.
"""

from __future__ import annotations

# ruff: noqa: F722, F821, UP037
import argparse
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msprime
import networkx as nx
import numpy as np
import scipy.sparse as sp
import xarray as xr
from beartype import beartype
from jaxtyping import Bool, Float, Int, Integer, Real, Shaped, UInt8, jaxtyped

from dreems import corpus, features, genotypes, workflow_io
from dreems.geography import Graph
from dreems.validation import _integer_array, _real_array, validate_graph_arrays

TRUTHS = ("barrier", "anisotropy", "homogeneous")
MASKS = ("full", "random_10pct", "outer_slabs", "left_missing")
ROWS = 8
COLUMNS = 12
NUM_NODES = ROWS * COLUMNS
DIPLOIDS_PER_NODE = 10
PLOIDY = 2
SIMULATION_SEED = 10
MIN_MAF = 0.05
MIN_CALL_RATE = 0.8

# Exact zero-based columns selected in the source notebook, in row-major order.
OUTER_SLAB_NODES = np.flatnonzero(
    np.isin(np.arange(NUM_NODES) % COLUMNS, (0, 1, 2, 9, 10, 11))
).astype(np.int32)
LEFT_MISSING_NODES = np.flatnonzero(np.arange(NUM_NODES) % COLUMNS < 3).astype(np.int32)


@dataclass(frozen=True)
class NotebookScenarioSpec:
    """One source-notebook truth and sampling case."""

    example_id: int
    truth: str
    mask: str


@beartype
def _settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate optional SNP counts without changing the notebook's defaults."""
    settings = config.get("notebook_scenarios")
    if not isinstance(settings, Mapping):
        raise ValueError("config must contain a notebook_scenarios mapping")
    settings = dict(settings)
    for key in ("num_snps", "retained_snps"):
        if key not in settings or (key == "retained_snps" and settings[key] is None):
            continue
        value = _integer_array(np.asarray(settings[key]), name=key, dtype=np.int64)
        if value.ndim != 0 or value.item() < 1:
            raise ValueError(f"notebook_scenarios.{key} must be a positive integer")
        settings[key] = value.item()
    return settings


@beartype
def scenario_count(config: Mapping[str, Any]) -> int:
    """Return the fixed number of distinct inference cells in the notebook."""

    _settings(config)
    return len(TRUTHS) * len(MASKS)


@beartype
def scenario_spec(config: Mapping[str, Any], example_id: int) -> NotebookScenarioSpec:
    """Map a stable zero-based example ID to notebook truth and mask."""

    total = scenario_count(config)
    if isinstance(example_id, bool):
        raise ValueError("example_id must be an integer, not a boolean")
    index = example_id
    if index < 0 or index >= total:
        raise ValueError(f"notebook scenario example_id must be in [0, {total}), got {index}")
    truth_index, mask_index = divmod(index, len(MASKS))
    return NotebookScenarioSpec(index, TRUTHS[truth_index], MASKS[mask_index])


@jaxtyped(typechecker=beartype)
def sampled_nodes(truth: str, mask: str) -> Int[np.ndarray, "sampled_nodes"]:
    """Return the exact node IDs selected by one notebook cell."""

    if truth not in TRUTHS:
        raise ValueError(f"unknown notebook truth: {truth!r}")
    if mask == "full":
        return np.arange(NUM_NODES, dtype=np.int32)
    if mask == "random_10pct":
        # The anisotropy notebook cell uses 20%; the other two use 10%.
        fraction = 0.20 if truth == "anisotropy" else 0.10
        count = int(NUM_NODES * fraction)
        return (
            np.random.default_rng(6).choice(NUM_NODES, size=count, replace=False).astype(np.int32)
        )
    if mask == "outer_slabs":
        return OUTER_SLAB_NODES.copy()
    if mask == "left_missing":
        return np.setdiff1d(
            np.arange(NUM_NODES, dtype=np.int32),
            LEFT_MISSING_NODES,
            assume_unique=True,
        ).astype(np.int32)
    raise ValueError(f"unknown notebook mask: {mask!r}")


@beartype
def observed_node_count(config: Mapping[str, Any], example_id: int) -> int:
    """Return the number of official FEEMS LOO folds for one notebook cell."""

    spec = scenario_spec(config, example_id)
    return int(sampled_nodes(spec.truth, spec.mask).size)


@jaxtyped(typechecker=beartype)
def notebook_graph() -> tuple[Graph, Float[np.ndarray, "nodes 2"]]:
    """Build the NetworkX lattice in the notebook's original edge order."""

    lattice = nx.generators.lattice.triangular_lattice_graph(
        ROWS - 1,
        2 * COLUMNS - 2,
        with_positions=True,
    )
    lattice = nx.convert_node_labels_to_integers(lattice)
    positions = np.asarray(list(nx.get_node_attributes(lattice, "pos").values()), dtype=np.float64)
    edges = np.asarray(lattice.edges, dtype=np.int32)
    if positions.shape != (NUM_NODES, 2) or edges.shape != (249, 2):
        raise RuntimeError("NetworkX no longer constructs the expected 8 x 12 lattice")
    graph = {
        "active_full": np.arange(NUM_NODES, dtype=np.int32),
        "source_active_full": np.arange(NUM_NODES, dtype=np.int32),
        "edges": edges,
    }
    return graph, positions


@jaxtyped(typechecker=beartype)
def edge_rates(
    truth: str, positions: Real[np.ndarray, "nodes 2"], edges: Int[np.ndarray, "edges 2"]
) -> Float[np.ndarray, "edges"]:
    """Return the raw migration rates assigned in the source notebook."""

    if truth == "barrier":
        corridor_rate = 5.0e-5
        barrier_rate = 1.0e-5
        anisotropy_scaler = 1.0
    elif truth == "anisotropy":
        corridor_rate = 1.0e-5
        barrier_rate = 1.0e-5
        anisotropy_scaler = 5.0
    elif truth == "homogeneous":
        corridor_rate = 1.0e-5
        barrier_rate = 1.0e-5
        anisotropy_scaler = 1.0
    else:
        raise ValueError(f"unknown notebook truth: {truth!r}")

    midpoint_x = positions[edges, 0].mean(axis=1)
    rates = np.where(
        (midpoint_x <= 2.5) | (midpoint_x >= 8.5),
        corridor_rate,
        barrier_rate,
    )
    horizontal = positions[edges[:, 0], 1] == positions[edges[:, 1], 1]
    rates[horizontal] *= anisotropy_scaler
    return rates.astype(np.float64)


@jaxtyped(typechecker=beartype)
def _demography(
    truth: str,
) -> tuple[msprime.Demography, Graph, Float[np.ndarray, "nodes 2"], Float[np.ndarray, "edges"]]:
    """Keep the notebook's deme order, symmetric migration, and ancestral split."""
    graph, positions = notebook_graph()
    edges = np.asarray(graph["edges"], dtype=np.int32)
    rates = edge_rates(truth, positions, edges)
    edge_rate = {tuple(edge): rate for edge, rate in zip(edges.tolist(), rates, strict=True)}

    demography = msprime.Demography()
    demography.add_population(initial_size=1.0e4, name="anc")
    for node in range(NUM_NODES):
        demography.add_population(initial_size=1.0e4, name=f"P{node}")
    demography.add_population_split(
        time=5_000_000,
        derived=[f"P{node}" for node in range(NUM_NODES)],
        ancestral="anc",
    )

    # Match the sparse upper-triangle iteration used in the notebook.
    lattice = nx.Graph()
    lattice.add_nodes_from(range(NUM_NODES))
    lattice.add_edges_from(edges.tolist())
    upper = sp.triu(nx.adjacency_matrix(lattice, np.arange(NUM_NODES)).astype(float)).tocoo()
    for left, right in zip(upper.row, upper.col, strict=True):
        key = (int(left), int(right))
        demography.set_symmetric_migration_rate(
            populations=(f"P{left}", f"P{right}"),
            rate=float(edge_rate[key]),
        )
    return demography, graph, positions, rates


@beartype
def _source_provenance(config: Mapping[str, Any]) -> tuple[Path, str]:
    """Refuse a notebook revision other than the one named in the config."""
    settings = _settings(config)
    source = Path(str(settings["source_notebook"])).expanduser().resolve()
    expected = str(settings["source_notebook_sha256"])
    observed = workflow_io.sha256(source)
    if observed != expected:
        raise ValueError(
            f"source notebook checksum changed: expected {expected}, observed {observed}"
        )
    return source, observed


@beartype
def generate_base(
    *,
    config_path: Path,
    truth: str,
    output: Path,
) -> None:
    """Simulate one full 96-deme genotype matrix exactly as the notebook does."""

    config = workflow_io.load_config(config_path)
    settings = _settings(config)
    _source, source_hash = _source_provenance(config)
    if truth not in TRUTHS:
        raise ValueError(f"unknown notebook truth: {truth!r}")
    num_snps = settings.get("num_snps", 1000)

    demography, graph, positions, rates = _demography(truth)
    samples = {f"P{node}": DIPLOIDS_PER_NODE for node in range(NUM_NODES)}
    haplotypes: list[UInt8[np.ndarray, "haploids"]] = []
    attempt = 1
    started = time.perf_counter()
    print(
        json.dumps(
            {"stage": "notebook_scenario_base", "event": "start", "truth": truth},
            sort_keys=True,
        ),
        flush=True,
    )
    while len(haplotypes) < num_snps:
        ancestry = msprime.sim_ancestry(
            samples=samples,
            demography=demography,
            recombination_rate=0,
            sequence_length=1,
            random_seed=attempt * SIMULATION_SEED * 7,
            ploidy=PLOIDY,
        )
        mutations = msprime.sim_mutations(
            ancestry,
            rate=1.0e-7,
            random_seed=attempt * SIMULATION_SEED * 2,
        )
        matrix = mutations.genotype_matrix()
        if matrix.shape[0] > 0 and not np.all(matrix[0] == 0):
            haplotypes.append(np.asarray(matrix[0], dtype=np.uint8))
            if len(haplotypes) % 100 == 0 or len(haplotypes) == num_snps:
                print(
                    json.dumps(
                        {
                            "stage": "notebook_scenario_base",
                            "event": "progress",
                            "truth": truth,
                            "accepted_snps": len(haplotypes),
                            "attempts": attempt,
                            "elapsed_seconds": time.perf_counter() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        attempt += 1

    edge_log_rates = np.log(rates)
    edge_log_rates -= edge_log_rates.mean()
    workflow_io.write_npz(
        output,
        truth=np.asarray(truth),
        positions=positions,
        edges=np.asarray(graph["edges"], dtype=np.int32),
        edge_log_rates=edge_log_rates,
        haplotypes=np.stack(haplotypes),
        attempts=np.asarray(attempt - 1, dtype=np.int64),
        source_notebook_sha256=np.asarray(source_hash),
    )
    print(
        json.dumps(
            {
                "stage": "notebook_scenario_base",
                "event": "complete",
                "truth": truth,
                "accepted_snps": num_snps,
                "attempts": attempt - 1,
                "elapsed_seconds": time.perf_counter() - started,
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


@jaxtyped(typechecker=beartype)
def _calls_for_nodes(
    haplotypes: Shaped[np.ndarray, "*haplotype_axes"], nodes: Shaped[np.ndarray, "*node_axes"]
) -> Integer[np.ndarray, "variants samples 2"]:
    """Select complete diploids in the notebook's fixed deme-major haplotype order."""
    if haplotypes.ndim != 2 or haplotypes.shape[1] != NUM_NODES * DIPLOIDS_PER_NODE * PLOIDY:
        raise ValueError("base haplotypes must contain every notebook diploid in deme order")
    if haplotypes.dtype.kind not in "iu" or np.any(haplotypes < 0):
        raise ValueError("base haplotypes must contain nonnegative integer allele codes")
    nodes = _integer_array(nodes, name="sample nodes", dtype=np.int32)
    if nodes.ndim != 1 or np.any(nodes < 0) or np.any(nodes >= NUM_NODES):
        raise ValueError("sample nodes must index the notebook lattice")
    binary = np.asarray(haplotypes)[np.max(haplotypes, axis=1) <= 1].T
    individuals = ((nodes[:, None] * DIPLOIDS_PER_NODE) + np.arange(DIPLOIDS_PER_NODE)).ravel()
    haploid_rows = np.column_stack((PLOIDY * individuals, PLOIDY * individuals + 1)).ravel()
    selected = binary[haploid_rows]
    return selected.reshape(individuals.size, PLOIDY, selected.shape[1]).transpose(2, 0, 1)


@jaxtyped(typechecker=beartype)
def maf_filtered_dosages(
    calls: Shaped[np.ndarray, "*call_axes"],
) -> tuple[Float[np.ndarray, "samples retained_snps"], Bool[np.ndarray, "variants"]]:
    """Apply the notebook's complete-data biallelic and MAF filter."""

    calls = np.asarray(calls)
    if calls.ndim != 3 or calls.shape[2] != PLOIDY:
        raise ValueError("calls must have shape (variants, samples, 2)")
    if calls.dtype.kind not in "iu" or np.any(calls < 0):
        raise ValueError("notebook calls must contain complete nonnegative integer allele codes")
    if not calls.shape[0] or not calls.shape[1]:
        raise ValueError("notebook calls must contain variants and samples")
    dosage = calls.sum(axis=2, dtype=np.float64)
    allele_frequency = dosage.mean(axis=1) / float(PLOIDY)
    maf = np.minimum(allele_frequency, 1.0 - allele_frequency)
    keep = (np.max(calls, axis=(1, 2)) <= 1) & (maf >= MIN_MAF)
    filtered = dosage[keep].T
    if filtered.shape[1] < 2:
        raise ValueError("fewer than two SNPs remain after notebook MAF filtering")
    return filtered, keep


@jaxtyped(typechecker=beartype)
def select_model_snps(
    calls: Shaped[np.ndarray, "variants samples 2"],
    *,
    target_snps: int | None,
) -> tuple[
    Integer[np.ndarray, "selected_variants samples 2"],
    Float[np.ndarray, "samples retained_snps"],
    int,
]:
    """Select a deterministic number of post-MAF SNPs for one sample mask.

    ``num_snps`` counts simulated variant rows. Subsetting samples and applying
    the MAF filter reduces this count. When a target is supplied, retain the
    first eligible SNPs in simulation order.
    """

    dosages, eligible = maf_filtered_dosages(calls)
    available = int(dosages.shape[1])
    if target_snps is None:
        return calls, dosages, available
    if isinstance(target_snps, bool) or target_snps <= 0:
        raise ValueError("notebook_scenarios.retained_snps must be positive")
    if available < target_snps:
        raise ValueError(
            f"requested {target_snps} retained SNPs, but only {available} pass "
            "the biallelic and MAF filters"
        )
    selected = np.flatnonzero(eligible)[:target_snps]
    return calls[selected], dosages[:, :target_snps], available


@jaxtyped(typechecker=beartype)
def _notebook_dataset(
    calls: Shaped[np.ndarray, "variants samples 2"],
    *,
    selected_nodes: Int[np.ndarray, "sampled_nodes"],
    raw_positions: Real[np.ndarray, "nodes 2"],
    edges: Int[np.ndarray, "edges 2"],
) -> xr.Dataset:
    """Reproduce the notebook's genotype-to-xarray path, including its projection."""
    # The archive uses unsigned bytes; the genotype API uses signed allele codes.
    calls = _integer_array(calls, name="notebook calls", dtype=np.int8)
    if np.any(calls < 0) or np.any(calls > 1):
        raise ValueError("notebook calls must be complete biallelic genotypes")
    sample_count = selected_nodes.size * DIPLOIDS_PER_NODE
    sample_ids = np.asarray([f"sample_{index:03d}" for index in range(sample_count)])
    sample_node_rows = np.repeat(selected_nodes, DIPLOIDS_PER_NODE)
    coordinates = raw_positions[sample_node_rows]
    genotype_dataset = xr.Dataset(
        data_vars={
            "call_genotype": (("variants", "samples", "ploidy"), calls),
            "sample_id": (("samples",), sample_ids),
        }
    )
    sample_coordinates = {
        sample_id: (float(position[1]), float(position[0]))
        for sample_id, position in zip(sample_ids, coordinates, strict=True)
    }
    return genotypes.from_vcf_zarr(
        genotype_dataset,
        node_latitude=raw_positions[:, 1],
        node_longitude=raw_positions[:, 0],
        edges=edges,
        sample_coordinates=sample_coordinates,
        min_maf=MIN_MAF,
        min_variant_call_rate=MIN_CALL_RATE,
        attrs={"name": "feems_scenarios_exact_h2h"},
    )


@beartype
def derive_case(
    *,
    config_path: Path,
    example_id: int,
    base: Path,
    output: Path,
    metadata: Path,
) -> None:
    """Derive one notebook sampling cell from its shared full simulation."""

    config = workflow_io.load_config(config_path)
    settings = _settings(config)
    _source, source_hash = _source_provenance(config)
    spec = scenario_spec(config, example_id)
    base_hash = workflow_io.sha256(base)
    with np.load(base, allow_pickle=False) as archive:
        base_truth = str(archive["truth"].item())
        raw_positions = _real_array(archive["positions"], name="positions")
        edges = _integer_array(archive["edges"], name="edges", dtype=np.int32)
        truth = _real_array(archive["edge_log_rates"], name="edge_log_rates")
        haplotypes = archive["haplotypes"]
        attempt_count = _integer_array(archive["attempts"], name="attempts", dtype=np.int64)
        if attempt_count.ndim != 0 or attempt_count.item() < 1:
            raise ValueError("base simulation attempts must be a positive integer")
        attempts = attempt_count.item()
        base_source_hash = str(archive["source_notebook_sha256"].item())
    if base_truth != spec.truth:
        raise ValueError(f"base truth {base_truth!r} does not match case truth {spec.truth!r}")
    if base_source_hash != source_hash:
        raise ValueError("base simulation came from a different source notebook")
    validate_graph_arrays(positions=raw_positions, edges=edges)
    if raw_positions.shape != (NUM_NODES, 2) or truth.shape != (len(edges),):
        raise ValueError("base geometry and truth must align with the notebook graph")

    selected_nodes = sampled_nodes(spec.truth, spec.mask)
    calls = _calls_for_nodes(haplotypes, selected_nodes)
    biallelic_snps = int(calls.shape[0])
    target_snps = settings.get("retained_snps")
    model_calls, genotype_matrix, available_snps = select_model_snps(
        calls,
        target_snps=target_snps,
    )
    data = _notebook_dataset(
        model_calls,
        selected_nodes=selected_nodes,
        raw_positions=raw_positions,
        edges=edges,
    )
    retained_snps = int(data.attrs["retained_variants"])
    if retained_snps != genotype_matrix.shape[1]:
        raise RuntimeError("direct MAF filter disagrees with dreems.from_vcf_zarr")

    sample_nodes = np.repeat(selected_nodes, DIPLOIDS_PER_NODE).astype(np.int32)
    standardized = features.standardized_polymorphic_genotypes(genotype_matrix)
    sample_covariance = standardized @ standardized.T / float(retained_snps)
    node_covariance, observed_nodes, node_sample_count = (
        features.sample_covariance_to_node_covariance(
            sample_covariance,
            sample_nodes,
            NUM_NODES,
        )
    )
    np.testing.assert_array_equal(observed_nodes, np.asarray(data.observed_nodes.values))
    np.testing.assert_array_equal(node_sample_count, np.asarray(data.node_sample_count.values))
    np.testing.assert_allclose(
        node_covariance,
        np.asarray(data.node_covariance.values),
        rtol=1.0e-10,
        atol=2.0e-7,
    )

    projected_positions = np.asarray(data.node_position.values, dtype=np.float64)
    graph = {
        "active_full": np.arange(NUM_NODES, dtype=np.int32),
        "source_active_full": np.arange(NUM_NODES, dtype=np.int32),
        "edges": edges,
    }
    random_fraction = (
        float(selected_nodes.size / NUM_NODES) if spec.mask == "random_10pct" else float("nan")
    )
    sample_design = (
        f"random_{int(round(100.0 * random_fraction))}pct"
        if spec.mask == "random_10pct"
        else spec.mask
    )
    surface_kind = {
        "barrier": "feems_heterogeneous_barrier",
        "anisotropy": "feems_anisotropic",
        "homogeneous": "feems_homogeneous",
    }[spec.truth]
    scenario_metadata = {
        "scenario_truth": spec.truth,
        "scenario_mask": spec.mask,
        "scenario_replicate": 0,
        "scenario_seed": SIMULATION_SEED,
        "scenario_source_notebook_sha256": source_hash,
        "scenario_raw_snps": int(haplotypes.shape[0]),
        "scenario_biallelic_snps": biallelic_snps,
        "scenario_maf_eligible_snps": available_snps,
        "scenario_requested_retained_snps": target_snps or 0,
        "scenario_random_sample_fraction": random_fraction,
        "msprime_attempts": attempts,
    }
    example: dict[str, Any] = {
        "graph": graph,
        "active_positions": projected_positions,
        "edge_log_rates": truth,
        "node_log_q": np.zeros(NUM_NODES, dtype=np.float32),
        "sample_nodes": sample_nodes,
        "genotypes": genotype_matrix,
        "num_snps": retained_snps,
        "effective_snps": retained_snps,
        "simulator": "msprime_exact_notebook",
        "surface_kind": surface_kind,
        "sample_design": sample_design,
        "sample_regime": sample_design,
        "mask_kind": "source_notebook_explicit",
        "graph_source": "networkx_triangular_lattice_source_notebook",
        "height": ROWS,
        "width": COLUMNS,
        "feems_lattice_training": True,
        "feems_lattice_rows": ROWS,
        "feems_lattice_columns": COLUMNS,
        "feems_lattice_sample_prob": float(selected_nodes.size / NUM_NODES),
        **scenario_metadata,
        "local_log_rate_std": float(np.std(truth)),
        "local_q_log_std": 0.0,
        "local_diploids_per_node": DIPLOIDS_PER_NODE,
    }
    row = corpus.example_metadata(example, spec.example_id)
    row.update(scenario_metadata)
    if workflow_io.sha256(base) != base_hash:
        raise ValueError("base simulation changed while deriving its sampling case")
    workflow_io.write_pickle(output, [example])
    workflow_io.write_json(
        metadata,
        {
            "stage": "derive_exact_notebook_scenario",
            "config": str(config_path),
            "base": str(base),
            "row": row,
        },
    )
    print(
        json.dumps(
            {
                "stage": "derive_exact_notebook_scenario",
                "example_id": spec.example_id,
                "truth": spec.truth,
                "mask": spec.mask,
                "observed_nodes": int(selected_nodes.size),
                "samples": int(sample_nodes.size),
                "raw_snps": int(haplotypes.shape[0]),
                "biallelic_snps": biallelic_snps,
                "maf_eligible_snps": available_snps,
                "retained_snps": retained_snps,
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


@beartype
def build_parser() -> argparse.ArgumentParser:
    """Separate expensive whole-graph simulation from cheap sampling views."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    base = subparsers.add_parser("generate-base")
    base.add_argument("--config", type=Path, required=True)
    base.add_argument("--truth", choices=TRUTHS, required=True)
    base.add_argument("--output", type=Path, required=True)

    case = subparsers.add_parser("derive-case")
    case.add_argument("--config", type=Path, required=True)
    case.add_argument("--example-id", type=int, required=True)
    case.add_argument("--base", type=Path, required=True)
    case.add_argument("--output", type=Path, required=True)
    case.add_argument("--metadata", type=Path, required=True)
    return parser


@beartype
def main(argv: Sequence[str] | None = None) -> None:
    """Run one notebook simulation or derive one cached sampling view."""
    args = build_parser().parse_args(argv)
    if args.command == "generate-base":
        generate_base(config_path=args.config, truth=args.truth, output=args.output)
    else:
        derive_case(
            config_path=args.config,
            example_id=args.example_id,
            base=args.base,
            output=args.output,
            metadata=args.metadata,
        )


if __name__ == "__main__":
    main()
