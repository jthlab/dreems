"""Reproduce the exact simulations and sampling masks in ``feems_scenarios.ipynb``.

The notebook simulates one full genotype matrix for each migration surface and
then reuses it across sampling masks. This module preserves that dependency
structure so the expensive msprime simulations run exactly three times.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from dreems import corpus, features, genotypes, workflow_io

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

OUTER_SLAB_NODES = np.asarray(
    [
        0,
        1,
        2,
        9,
        10,
        11,
        12,
        13,
        14,
        21,
        22,
        23,
        24,
        25,
        26,
        33,
        34,
        35,
        36,
        37,
        38,
        45,
        46,
        47,
        48,
        49,
        50,
        57,
        58,
        59,
        60,
        61,
        62,
        69,
        70,
        71,
        72,
        73,
        74,
        81,
        82,
        83,
        84,
        85,
        86,
        93,
        94,
        95,
    ],
    dtype=np.int32,
)
LEFT_MISSING_NODES = np.asarray(
    [
        0,
        1,
        2,
        12,
        13,
        14,
        24,
        25,
        26,
        36,
        37,
        38,
        48,
        49,
        50,
        60,
        61,
        62,
        72,
        73,
        74,
        84,
        85,
        86,
    ],
    dtype=np.int32,
)


@dataclass(frozen=True)
class NotebookScenarioSpec:
    """One source-notebook truth and sampling case."""

    example_id: int
    truth: str
    mask: str


def _settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    settings = config.get("notebook_scenarios")
    if not isinstance(settings, Mapping):
        raise ValueError("config must contain a notebook_scenarios mapping")
    return settings


def scenario_count(config: Mapping[str, Any]) -> int:
    """Return the fixed number of distinct inference cells in the notebook."""

    _settings(config)
    return len(TRUTHS) * len(MASKS)


def scenario_spec(config: Mapping[str, Any], example_id: int) -> NotebookScenarioSpec:
    """Map a stable zero-based example ID to notebook truth and mask."""

    total = scenario_count(config)
    index = int(example_id)
    if index < 0 or index >= total:
        raise ValueError(f"notebook scenario example_id must be in [0, {total}), got {index}")
    truth_index, mask_index = divmod(index, len(MASKS))
    return NotebookScenarioSpec(index, TRUTHS[truth_index], MASKS[mask_index])


def sampled_nodes(truth: str, mask: str) -> np.ndarray:
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


def observed_node_count(config: Mapping[str, Any], example_id: int) -> int:
    """Return the number of official FEEMS LOO folds for one notebook cell."""

    spec = scenario_spec(config, example_id)
    return int(sampled_nodes(spec.truth, spec.mask).size)


def notebook_graph() -> tuple[dict[str, np.ndarray], np.ndarray]:
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


def edge_rates(truth: str, positions: np.ndarray, edges: np.ndarray) -> np.ndarray:
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


def _demography(
    truth: str,
) -> tuple[msprime.Demography, dict[str, np.ndarray], np.ndarray, np.ndarray]:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_provenance(config: Mapping[str, Any]) -> tuple[Path, str]:
    settings = _settings(config)
    source = Path(str(settings["source_notebook"])).expanduser().resolve()
    expected = str(settings["source_notebook_sha256"])
    observed = _sha256(source)
    if observed != expected:
        raise ValueError(
            f"source notebook checksum changed: expected {expected}, observed {observed}"
        )
    return source, observed


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


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
    num_snps = int(settings.get("num_snps", 1000))
    if num_snps <= 0:
        raise ValueError("notebook_scenarios.num_snps must be positive")

    demography, graph, positions, rates = _demography(truth)
    samples = {f"P{node}": DIPLOIDS_PER_NODE for node in range(NUM_NODES)}
    haplotypes: list[np.ndarray] = []
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
    _atomic_npz(
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


def _calls_for_nodes(haplotypes: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    binary = np.asarray(haplotypes)[np.max(haplotypes, axis=1) <= 1].T
    individuals = ((nodes[:, None] * DIPLOIDS_PER_NODE) + np.arange(DIPLOIDS_PER_NODE)).ravel()
    haploid_rows = np.column_stack((PLOIDY * individuals, PLOIDY * individuals + 1)).ravel()
    selected = binary[haploid_rows]
    return selected.reshape(individuals.size, PLOIDY, selected.shape[1]).transpose(2, 0, 1)


def maf_filtered_dosages(calls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the notebook's complete-data biallelic and MAF filter."""

    calls = np.asarray(calls)
    if calls.ndim != 3 or calls.shape[2] != PLOIDY:
        raise ValueError("calls must have shape (variants, samples, 2)")
    dosage = calls.sum(axis=2, dtype=np.float64)
    allele_frequency = dosage.mean(axis=1) / float(PLOIDY)
    maf = np.minimum(allele_frequency, 1.0 - allele_frequency)
    keep = (np.max(calls, axis=(1, 2)) <= 1) & (maf >= MIN_MAF)
    filtered = dosage[keep].T
    if filtered.shape[1] < 2:
        raise ValueError("fewer than two SNPs remain after notebook MAF filtering")
    return filtered, keep


def _notebook_dataset(
    calls: np.ndarray,
    *,
    selected_nodes: np.ndarray,
    raw_positions: np.ndarray,
    edges: np.ndarray,
) -> xr.Dataset:
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
    _source, source_hash = _source_provenance(config)
    spec = scenario_spec(config, example_id)
    with np.load(base) as archive:
        base_truth = str(archive["truth"].item())
        raw_positions = np.asarray(archive["positions"], dtype=np.float64)
        edges = np.asarray(archive["edges"], dtype=np.int32)
        truth = np.asarray(archive["edge_log_rates"], dtype=np.float64)
        haplotypes = np.asarray(archive["haplotypes"], dtype=np.uint8)
        attempts = int(archive["attempts"].item())
        base_source_hash = str(archive["source_notebook_sha256"].item())
    if base_truth != spec.truth:
        raise ValueError(f"base truth {base_truth!r} does not match case truth {spec.truth!r}")
    if base_source_hash != source_hash:
        raise ValueError("base simulation came from a different source notebook")

    selected_nodes = sampled_nodes(spec.truth, spec.mask)
    calls = _calls_for_nodes(haplotypes, selected_nodes)
    genotype_matrix, retained = maf_filtered_dosages(calls)
    data = _notebook_dataset(
        calls,
        selected_nodes=selected_nodes,
        raw_positions=raw_positions,
        edges=edges,
    )
    retained_snps = int(data.attrs["retained_variants"])
    if retained_snps != genotype_matrix.shape[1] or retained_snps != int(retained.sum()):
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
        "scenario_truth": spec.truth,
        "scenario_mask": spec.mask,
        "scenario_replicate": 0,
        "scenario_seed": SIMULATION_SEED,
        "scenario_source_notebook_sha256": source_hash,
        "scenario_raw_snps": int(haplotypes.shape[0]),
        "scenario_biallelic_snps": int(calls.shape[0]),
        "scenario_random_sample_fraction": random_fraction,
        "msprime_attempts": attempts,
        "local_log_rate_std": float(np.std(truth)),
        "local_q_log_std": 0.0,
        "local_diploids_per_node": DIPLOIDS_PER_NODE,
    }
    row = corpus.example_metadata(example, spec.example_id)
    row.update(
        {
            "scenario_truth": spec.truth,
            "scenario_mask": spec.mask,
            "scenario_replicate": 0,
            "scenario_seed": SIMULATION_SEED,
            "scenario_source_notebook_sha256": source_hash,
            "scenario_raw_snps": int(haplotypes.shape[0]),
            "scenario_biallelic_snps": int(calls.shape[0]),
            "scenario_random_sample_fraction": random_fraction,
            "msprime_attempts": attempts,
        }
    )
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
                "biallelic_snps": int(calls.shape[0]),
                "retained_snps": retained_snps,
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
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


def main(argv: Sequence[str] | None = None) -> None:
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
