"""Paired sampling-mask augmentation for DREEMS training."""

from __future__ import annotations

# ruff: noqa: F722, F821, UP037
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped
from scipy.sparse import coo_array
from scipy.sparse.csgraph import dijkstra

from dreems.validation import _integer_array

COUNTERFACTUAL_MASKS = (
    "anchor",
    "random_10pct",
    "outer_regions",
    "one_side_missing",
)
COUNTERFACTUAL_MASK_INDEX = {name: index for index, name in enumerate(COUNTERFACTUAL_MASKS)}


@dataclass(frozen=True)
class BaseExample:
    """Location and audit metadata for one reusable raw simulation."""

    slot: int
    raw_path: str
    raw_index: int
    original_index: int
    surface_kind: str
    graph_source: str
    num_nodes: int
    num_sampled_nodes: int
    sampled_node_fraction: float
    effective_snps: int


@beartype
def _surface_family(surface_kind: str) -> str:
    """Group named simulation recipes for balanced base-example selection."""
    if surface_kind in {"constant", "feems_homogeneous"}:
        return "homogeneous"
    if surface_kind in {"barrier", "feems_heterogeneous_barrier"}:
        return "barrier"
    if surface_kind == "feems_anisotropic":
        return "anisotropy"
    return "broad"


@beartype
def _snp_bin(effective_snps: int) -> int:
    """Use the four SNP-count strata from the recorded training recipe."""
    for index, upper in enumerate((128, 512, 2_000)):
        if effective_snps <= upper:
            return index
    return 3


@beartype
def _raw_path(metadata_path: Path) -> Path:
    """Find the raw pack beside its metadata file."""
    suffix = ".metadata.json"
    if not metadata_path.name.endswith(suffix):
        raise ValueError(f"raw metadata path must end in {suffix}: {metadata_path}")
    return metadata_path.with_suffix("").with_suffix(".pkl")


@beartype
def select_base_examples(
    metadata_paths: Sequence[Path],
    *,
    count: int,
    seed: int,
    min_sampled_fraction: float = 0.99,
    min_sampled_nodes: int = 12,
) -> list[BaseExample]:
    """Select a deterministic, surface- and SNP-balanced dense subset."""

    if count <= 0:
        raise ValueError("count must be positive")
    if not np.isfinite(min_sampled_fraction) or not 0.0 <= min_sampled_fraction <= 1.0:
        raise ValueError("min_sampled_fraction must be finite and between zero and one")
    if min_sampled_nodes < 2:
        raise ValueError("min_sampled_nodes must be at least two")
    candidates: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for metadata_path in sorted(Path(path) for path in metadata_paths):
        payload = json.loads(metadata_path.read_text())
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"raw metadata has no rows list: {metadata_path}")
        raw_path = _raw_path(metadata_path)
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        for raw_index, row in enumerate(rows):
            sampled_fraction = float(row.get("sampled_node_fraction", 0.0))
            if not np.isfinite(sampled_fraction) or not 0.0 <= sampled_fraction <= 1.0:
                raise ValueError(
                    f"invalid sampled_node_fraction in {metadata_path}, row {raw_index}"
                )
            sampled_nodes = int(row.get("num_sampled_nodes", 0))
            if sampled_fraction < min_sampled_fraction or sampled_nodes < min_sampled_nodes:
                continue
            surface_kind = str(row.get("surface_kind", "unknown"))
            effective_snps = int(row.get("effective_snps", row.get("num_snps", 0)))
            candidates[(_surface_family(surface_kind), _snp_bin(effective_snps))].append(
                {
                    "raw_path": str(raw_path.resolve()),
                    "raw_index": raw_index,
                    "original_index": int(row.get("index", raw_index)),
                    "surface_kind": surface_kind,
                    "graph_source": str(row.get("graph_source", "unknown")),
                    "num_nodes": int(row.get("num_nodes", 0)),
                    "num_sampled_nodes": sampled_nodes,
                    "sampled_node_fraction": sampled_fraction,
                    "effective_snps": effective_snps,
                }
            )

    available = sum(len(values) for values in candidates.values())
    if available < count:
        raise ValueError(f"requested {count} dense bases but found only {available}")
    rng = np.random.default_rng(seed)
    for values in candidates.values():
        rng.shuffle(values)

    # Visit strata in a fixed order, taking one example from each per round.
    keys = sorted(candidates)
    offsets = {key: 0 for key in keys}
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progress = False
        for key in keys:
            offset = offsets[key]
            if offset >= len(candidates[key]):
                continue
            selected.append(candidates[key][offset])
            offsets[key] += 1
            progress = True
            if len(selected) == count:
                break
        if not progress:
            raise RuntimeError("base-example selection stopped before reaching the requested count")
    rng.shuffle(selected)
    return [BaseExample(slot=slot, **row) for slot, row in enumerate(selected)]


@beartype
def selection_payload(
    selected: Sequence[BaseExample],
    *,
    seed: int,
    min_sampled_fraction: float,
    min_sampled_nodes: int,
) -> dict[str, Any]:
    """Record the selection and the settings needed to reproduce it."""
    return {
        "stage": "select_counterfactual_bases",
        "seed": int(seed),
        "min_sampled_fraction": float(min_sampled_fraction),
        "min_sampled_nodes": int(min_sampled_nodes),
        "masks": list(COUNTERFACTUAL_MASKS),
        "examples": [asdict(item) for item in selected],
    }


@beartype
def read_selection(path: Path) -> list[BaseExample]:
    """Read a saved selection without reordering the chosen examples."""
    payload = json.loads(path.read_text())
    rows = payload.get("examples")
    if not isinstance(rows, list):
        raise ValueError(f"selection has no examples list: {path}")
    return [BaseExample(**row) for row in rows]


@jaxtyped(typechecker=beartype)
def topology_axis_scores(
    num_nodes: int,
    edges: np.ndarray,
    observed_nodes: np.ndarray,
    *,
    seed: int,
) -> "Float[np.ndarray, 'num_nodes']":
    """Use the difference in hop distances to two far-apart sampled nodes.

    A seeded starting node and two farthest-node searches choose the ends.
    Ties follow observed-node order, preserving the training masks across runs.
    Coordinates and migration rates do not enter this construction.
    """

    if num_nodes < 2:
        raise ValueError("num_nodes must be at least two")
    edges = _integer_array(edges, name="edges", dtype=np.int32)
    observed = _integer_array(observed_nodes, name="observed_nodes", dtype=np.int32)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (edges, 2)")
    if observed.ndim != 1 or observed.size == 0:
        raise ValueError("observed_nodes must be a nonempty vector")
    if np.any(edges < 0) or np.any(edges >= num_nodes):
        raise ValueError("edges must refer to graph nodes")
    if np.any(observed < 0) or np.any(observed >= num_nodes):
        raise ValueError("observed_nodes must refer to graph nodes")
    adjacency = coo_array(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes),
    ).tocsr()
    rng = np.random.default_rng(seed)
    start = int(rng.choice(observed))
    distance = dijkstra(adjacency, directed=False, indices=start, unweighted=True)
    if not np.all(np.isfinite(distance)):
        raise ValueError("topology masks require a connected graph")
    first = observed[np.argmax(distance[observed])]
    distance_first = dijkstra(adjacency, directed=False, indices=first, unweighted=True)
    second = observed[np.argmax(distance_first[observed])]
    distance_second = dijkstra(adjacency, directed=False, indices=second, unweighted=True)
    if rng.random() < 0.5:
        distance_first, distance_second = distance_second, distance_first
    return distance_first - distance_second


@beartype
def counterfactual_node_sets(
    example: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """Keep all nodes, a random tenth, both ends, or one end of the sampled range."""
    sample_nodes = _integer_array(
        np.asarray(example["sample_nodes"]), name="sample_nodes", dtype=np.int32
    )
    if sample_nodes.ndim != 1:
        raise ValueError("sample_nodes must be one-dimensional")
    observed = np.unique(sample_nodes)
    if observed.size < 12:
        raise ValueError("counterfactual masks require at least 12 observed nodes")
    num_nodes = int(np.asarray(example["active_positions"]).shape[0])
    edges = np.asarray(example["graph"]["edges"])
    score = topology_axis_scores(num_nodes, edges, observed, seed=seed)
    ordered = observed[np.argsort(score[observed], kind="stable")]
    quarter = max(2, int(np.ceil(0.25 * observed.size)))
    one_side = ordered[quarter:]
    outer = np.unique(np.concatenate([ordered[:quarter], ordered[-quarter:]])).astype(np.int32)
    rng = np.random.default_rng(seed + 1)
    random_count = max(2, int(np.floor(0.10 * observed.size)))
    random_sparse = np.sort(rng.choice(observed, size=random_count, replace=False)).astype(np.int32)
    return {
        "anchor": observed.astype(np.int32),
        "random_10pct": random_sparse,
        "outer_regions": outer,
        "one_side_missing": one_side.astype(np.int32),
    }


@beartype
def subset_example_to_nodes(
    example: Mapping[str, Any],
    selected_nodes: np.ndarray,
) -> dict[str, Any]:
    """Subset every stored sample representation, keeping graph and truth intact.

    Covariance axes, genotype rows, and optional sample coordinates all use the
    order of ``sample_nodes``. The SNP set and effective SNP count are retained;
    this is a sampling mask, not a fresh round of genotype filtering.
    """

    selected = _integer_array(selected_nodes, name="selected_nodes", dtype=np.int32)
    sample_nodes = _integer_array(
        np.asarray(example["sample_nodes"]), name="sample_nodes", dtype=np.int32
    )
    num_nodes = np.asarray(example["active_positions"]).shape[0]
    for name, nodes in (("selected_nodes", selected), ("sample_nodes", sample_nodes)):
        if nodes.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if np.any(nodes < 0) or np.any(nodes >= num_nodes):
            raise ValueError(f"{name} must refer to graph nodes")
    out = dict(example)
    keep = np.isin(sample_nodes, selected)
    sample_indices = np.flatnonzero(keep)
    if np.unique(sample_nodes[keep]).size < 2:
        raise ValueError("selected nodes contain fewer than two observed demes")
    out["sample_nodes"] = sample_nodes[keep]
    if "sample_covariance" not in example and "genotypes" not in example:
        raise ValueError("counterfactual augmentation requires sample covariance or genotypes")
    if "sample_covariance" in example:
        covariance = np.asarray(example["sample_covariance"])
        if covariance.shape != (sample_nodes.size, sample_nodes.size):
            raise ValueError("sample_covariance must have shape (samples, samples)")
        out["sample_covariance"] = covariance[np.ix_(sample_indices, sample_indices)]
    for name in ("genotypes", "sample_coordinates"):
        if name in example:
            values = np.asarray(example[name])
            if values.ndim != 2 or values.shape[0] != sample_nodes.size:
                raise ValueError(f"{name} must have one row per sample")
            out[name] = values[keep]
    out["num_samples"] = int(np.sum(keep))
    out["num_sampled_nodes"] = int(np.unique(out["sample_nodes"]).size)
    out["sampled_node_fraction"] = float(out["num_sampled_nodes"] / num_nodes)
    return out


@beartype
def counterfactual_variants(
    example: Mapping[str, Any],
    *,
    group: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Return paired observations of one simulation under four sampling masks."""

    node_sets = counterfactual_node_sets(example, seed=seed)
    variants = []
    for mask in COUNTERFACTUAL_MASKS:
        variant = subset_example_to_nodes(example, node_sets[mask])
        variant["sampling_base_design"] = str(example.get("sample_design", "unknown"))
        variant["sample_design"] = f"counterfactual_{mask}"
        variant["sample_regime"] = f"counterfactual_{mask}"
        variant["sampling_counterfactual_group"] = int(group)
        variant["sampling_counterfactual_mask"] = mask
        variant["sampling_counterfactual_anchor"] = mask == "anchor"
        variants.append(variant)
    return variants
