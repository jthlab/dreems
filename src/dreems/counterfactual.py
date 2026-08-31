"""Paired sampling-mask augmentation for DREEMS training."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

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


def _surface_family(surface_kind: str) -> str:
    if surface_kind in {"constant", "feems_homogeneous"}:
        return "homogeneous"
    if surface_kind in {"barrier", "feems_heterogeneous_barrier"}:
        return "barrier"
    if surface_kind == "feems_anisotropic":
        return "anisotropy"
    return "broad"


def _snp_bin(effective_snps: int) -> int:
    for index, upper in enumerate((128, 512, 2_000)):
        if effective_snps <= upper:
            return index
    return 3


def _raw_path(metadata_path: Path) -> Path:
    suffix = ".metadata.json"
    text = str(metadata_path)
    if not text.endswith(suffix):
        raise ValueError(f"raw metadata path must end in {suffix}: {metadata_path}")
    return Path(text[: -len(suffix)] + ".pkl")


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


def selection_payload(
    selected: Sequence[BaseExample],
    *,
    seed: int,
    min_sampled_fraction: float,
    min_sampled_nodes: int,
) -> dict[str, Any]:
    return {
        "stage": "select_counterfactual_bases",
        "seed": int(seed),
        "min_sampled_fraction": float(min_sampled_fraction),
        "min_sampled_nodes": int(min_sampled_nodes),
        "masks": list(COUNTERFACTUAL_MASKS),
        "examples": [asdict(item) for item in selected],
    }


def read_selection(path: Path) -> list[BaseExample]:
    payload = json.loads(path.read_text())
    rows = payload.get("examples")
    if not isinstance(rows, list):
        raise ValueError(f"selection has no examples list: {path}")
    return [BaseExample(**row) for row in rows]


def _adjacency(num_nodes: int, edges: np.ndarray) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for left, right in np.asarray(edges, dtype=np.int32):
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    return adjacency


def _distances(adjacency: Sequence[Sequence[int]], start: int) -> np.ndarray:
    distance = np.full((len(adjacency),), -1, dtype=np.int32)
    distance[start] = 0
    queue: deque[int] = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if distance[neighbor] >= 0:
                continue
            distance[neighbor] = distance[node] + 1
            queue.append(neighbor)
    return distance


def _farthest_observed(distance: np.ndarray, observed: np.ndarray) -> int:
    observed_distance = distance[observed]
    reachable = observed_distance >= 0
    if not np.any(reachable):
        raise ValueError("sampled nodes do not lie in one connected graph component")
    maximum = np.max(observed_distance[reachable])
    return int(observed[np.flatnonzero(observed_distance == maximum)[0]])


def topology_axis_scores(
    num_nodes: int,
    edges: np.ndarray,
    observed_nodes: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Construct an intrinsic end-to-end coordinate without using x or y axes."""

    observed = np.asarray(observed_nodes, dtype=np.int32)
    adjacency = _adjacency(num_nodes, edges)
    rng = np.random.default_rng(seed)
    start = int(rng.choice(observed))
    first = _farthest_observed(_distances(adjacency, start), observed)
    distance_first = _distances(adjacency, first)
    second = _farthest_observed(distance_first, observed)
    distance_second = _distances(adjacency, second)
    if rng.random() < 0.5:
        distance_first, distance_second = distance_second, distance_first
    return distance_first.astype(np.float64) - distance_second.astype(np.float64)


def counterfactual_node_sets(
    example: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    sample_nodes = np.asarray(example["sample_nodes"], dtype=np.int32)
    observed = np.unique(sample_nodes)
    if observed.size < 12:
        raise ValueError("counterfactual masks require at least 12 observed nodes")
    num_nodes = int(np.asarray(example["active_positions"]).shape[0])
    edges = np.asarray(example["graph"]["edges"], dtype=np.int32)
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


def subset_example_to_nodes(
    example: Mapping[str, Any],
    selected_nodes: np.ndarray,
) -> dict[str, Any]:
    """Restrict one observation to selected demes without changing its truth."""

    selected = np.asarray(selected_nodes, dtype=np.int32)
    if selected.size < 2:
        raise ValueError("a counterfactual observation needs at least two sampled nodes")
    out = dict(example)
    sample_nodes = np.asarray(example["sample_nodes"], dtype=np.int32)
    keep = np.isin(sample_nodes, selected)
    sample_indices = np.flatnonzero(keep)
    if np.unique(sample_nodes[keep]).size < 2:
        raise ValueError("selected nodes contain fewer than two observed demes")
    out["sample_nodes"] = sample_nodes[keep]
    if "sample_covariance" in example:
        covariance_source = np.asarray(example["sample_covariance"])
        covariance = np.asarray(covariance_source, dtype=np.float64)
        out["sample_covariance"] = covariance[np.ix_(sample_indices, sample_indices)].astype(
            np.asarray(example["sample_covariance"]).dtype,
            copy=False,
        )
    elif "genotypes" in example:
        out["genotypes"] = np.asarray(example["genotypes"])[keep]
    else:
        raise ValueError("counterfactual augmentation requires sample covariance or genotypes")
    out["num_samples"] = int(np.sum(keep))
    out["num_sampled_nodes"] = int(np.unique(out["sample_nodes"]).size)
    out["sampled_node_fraction"] = float(
        out["num_sampled_nodes"] / np.asarray(example["active_positions"]).shape[0]
    )
    return out


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
