"""
Static topology skeleton for the JINR HPC cluster.

Wraps build_routing_maps() + build_edge_indices() + build_edge_attr() from
dataset_generator — single source of truth, no duplication.

Call topology_singleton() once at startup; it caches the result.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import torch
from torch_geometric.data import HeteroData

from training_pipeline.config import (
    EDGE_DIM_DUMMY,
    EDGE_DIM_PHYSICAL,
    NUM_HOSTS,
    NUM_JOBS,
    NUM_LEAF,
    NUM_SPINE,
    PHYSICAL_EDGE_TYPES,
    LOGICAL_EDGE_TYPES,
    CONTEXT_EDGE_TYPES,
)
from training_pipeline.dataset_generator import (
    build_edge_attr,
    build_edge_indices,
    build_routing_maps,
)

EdgeType = Tuple[str, str, str]

# Expose counts for other modules
NUM_SWITCHES = NUM_LEAF + NUM_SPINE

NODE_COUNTS: Dict[str, int] = {
    "cpu":         NUM_HOSTS,
    "gpu":         NUM_HOSTS,
    "ram":         NUM_HOSTS,
    "hdd":         NUM_HOSTS,
    "switch":      NUM_SWITCHES,
    "job":         NUM_JOBS,
    "rca_context": 1,
}

EXPECTED_DIMS: Dict[str, int] = {
    "cpu": 32, "gpu": 46, "ram": 32, "hdd": 32,
    "switch": 40, "job": 40, "rca_context": 1,
}


@lru_cache(maxsize=1)
def topology_singleton() -> Tuple[
    Dict[str, int],             # host_to_leaf
    Dict[str, list],            # leaf_to_spines
    Dict[str, int],             # job_to_host
    Dict[EdgeType, torch.Tensor],  # edge_index_dict
    Dict[EdgeType, torch.Tensor],  # edge_attr_dict
]:
    """Build static topology once; subsequent calls return cached result."""
    host_to_leaf, leaf_to_spines, job_to_host, _ = build_routing_maps(seed=None)
    edge_index_dict = build_edge_indices(host_to_leaf, leaf_to_spines, job_to_host)
    edge_attr_dict  = build_edge_attr(edge_index_dict)
    return host_to_leaf, leaf_to_spines, job_to_host, edge_index_dict, edge_attr_dict


def edge_counts() -> Dict[EdgeType, int]:
    """Return {edge_type: num_edges} for inspection."""
    _, _, _, ei_dict, _ = topology_singleton()
    return {et: ei.shape[1] for et, ei in ei_dict.items()}
