"""
attention.py — Explainable-AI edge weighting for the Grafana Node Graph (L6).

The diploma (sections 5.4–5.5) requires the Node Graph to render edge thickness
and colour by the GATv2 attention coefficients, so an operator can trace the
fault-propagation path. This module produces that per-edge weighting.

Two providers, unified behind build_edge_xai():

  1. compute_edge_salience()  — DEFAULT, verified, backend-owned.
     Edge weight = geometric mean of the anomaly scores at both endpoints.
     An edge lights up only when *both* nodes it connects are anomalous, which
     is exactly the fault sub-graph an operator wants to see. Derived purely
     from predict()'s node scores — no torch internals, fully testable.

  2. extract_attention_weights() — SEAM for real GATv2 coefficients.
     Raising NotImplementedError by design: extracting raw attention requires
     calling each GATv2Conv with return_attention_weights=True (a change inside
     the GNN model, owned by the GNN author and unverifiable without a torch
     env here). build_edge_xai() catches it and falls back to salience, so the
     live pipeline can never break — it just degrades to the proxy.

Output contract (consumed by publisher.py → api/grafana_api.py /topology):
    {
      "method": "salience" | "attention",
      "edges": [
        {"source": "cpu-5", "target": "switch-2",
         "relation": "connected_to", "weight": 0.83},
        ...
      ]
    }
Node ids use "{node_type}-{index}", identical to the scheme in grafana_api.py.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

EdgeType = Tuple[str, str, str]

# Forward dependency relations worth drawing. Reverse (rev_*) and context
# (reports_to) edges are skipped — they exist only for message passing and
# would clutter the operator's view without adding causal information.
_VIZ_RELATIONS: frozenset[str] = frozenset({
    "connected_to",   # cpu/gpu → switch
    "attached_to",    # ram/hdd → cpu
    "uplink_to",      # switch  → switch (leaf → spine)
    "executes_on",    # job     → cpu
})

# Keep the artifact small and the Grafana graph readable.
_MAX_EDGES: int = 60
# Below this, an edge is noise — both endpoints are effectively healthy.
_MIN_WEIGHT: float = 0.05


def _node_id(node_type: str, index: int) -> str:
    """Canonical Grafana node id — must match grafana_api.py exactly."""
    return f"{node_type}-{index}"


def compute_edge_salience(
    scores_by_type: Dict[str, List[float]],
    edge_index_dict: Dict[EdgeType, Any],
) -> List[Dict[str, Any]]:
    """Edge importance from endpoint anomaly scores (verified provider).

    weight(u → v) = sqrt(score[u] * score[v])

    The geometric mean is high only when BOTH endpoints are anomalous, so the
    returned edges trace the fault sub-graph rather than every link touching a
    single hot node. Returns at most _MAX_EDGES edges, sorted by weight desc,
    filtered to weight >= _MIN_WEIGHT.

    edge_index_dict[et] is a [2, num_edges] tensor-like: row 0 = source indices,
    row 1 = destination indices (PyG COO layout). Accessed via .tolist() so the
    function works with torch tensors or plain nested lists (testable w/o torch).
    """
    edges: List[Dict[str, Any]] = []

    for et, ei in edge_index_dict.items():
        src_type, relation, dst_type = et
        if relation not in _VIZ_RELATIONS:
            continue

        src_scores = scores_by_type.get(src_type)
        dst_scores = scores_by_type.get(dst_type)
        if not src_scores or not dst_scores:
            continue

        # Normalise the edge index to two python lists.
        coo = ei.tolist() if hasattr(ei, "tolist") else ei
        if not coo or len(coo) != 2:
            continue
        src_idx, dst_idx = coo[0], coo[1]

        for s, d in zip(src_idx, dst_idx):
            if s >= len(src_scores) or d >= len(dst_scores):
                continue
            w = math.sqrt(max(0.0, src_scores[s]) * max(0.0, dst_scores[d]))
            if w < _MIN_WEIGHT:
                continue
            edges.append({
                "source":   _node_id(src_type, s),
                "target":   _node_id(dst_type, d),
                "relation": relation,
                "weight":   round(w, 4),
            })

    edges.sort(key=lambda e: e["weight"], reverse=True)
    return edges[:_MAX_EDGES]


def extract_attention_weights(
    model: Any,
    x_dict: Dict[str, Any],
    edge_index_dict: Dict[EdgeType, Any],
    edge_attr_dict: Dict[EdgeType, Any],
) -> List[Dict[str, Any]]:
    """SEAM for real GATv2 attention coefficients (GNN author's domain).

    To implement: bypass HeteroConv and call each per-edge-type GATv2Conv with
    return_attention_weights=True, e.g.

        out, (ei, alpha) = self.conv1.convs[et](
            (h_src, h_dst), edge_index, edge_attr=ea,
            return_attention_weights=True,
        )

    then average alpha over heads → one weight per edge, and map COO indices to
    "{type}-{idx}" ids. This touches the model's forward path and must be
    verified in a torch environment, so it is intentionally not implemented here.

    build_edge_xai() catches NotImplementedError and falls back to salience.
    """
    raise NotImplementedError(
        "Real GATv2 attention extraction is a GNN-model change (return_attention_weights); "
        "not wired yet — using the verified salience proxy instead."
    )


def build_edge_xai(
    scores_by_type: Dict[str, List[float]],
    edge_index_dict: Dict[EdgeType, Any],
    *,
    model: Optional[Any] = None,
    x_dict: Optional[Dict[str, Any]] = None,
    edge_attr_dict: Optional[Dict[EdgeType, Any]] = None,
) -> Dict[str, Any]:
    """Unified XAI edge weighting. Prefers real attention, falls back to salience.

    Any failure in the (currently unimplemented) attention path degrades to the
    verified salience proxy — the live pipeline never breaks on XAI.
    """
    if model is not None and x_dict is not None and edge_attr_dict is not None:
        try:
            edges = extract_attention_weights(model, x_dict, edge_index_dict, edge_attr_dict)
            return {"method": "attention", "edges": edges}
        except NotImplementedError:
            pass  # expected until the GNN author wires return_attention_weights
        except Exception as exc:  # defensive: never let XAI crash inference
            logger.warning("attention extraction failed, falling back to salience: %s", exc)

    edges = compute_edge_salience(scores_by_type, edge_index_dict)
    return {"method": "salience", "edges": edges}
