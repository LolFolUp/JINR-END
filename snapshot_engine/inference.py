"""
inference.py — GATv2Hetero loader and single-graph inference.

Load:
    engine = InferenceEngine.load("checkpoints/best_model.pt")

Run:
    result = engine.predict(x_dict, edge_index_dict, edge_attr_dict)

Result dict:
    rc_node       : {"type": str, "id": int}   root cause chosen by RWR (2.3.4)
    confidence    : float   (anomaly probability of the RWR-selected node)
    rc_method     : str     ("rwr" | "argmax_fallback")
    rc_node_local : {"type", "id"}             per-node argmax (audit/fallback)
    rwr_ranking   : [{"id": "type-idx", "score": float}]  RWR stationary ranking
    fault_type    : str     (placeholder — training used 3 fault types)
    top5_candidates : List[{"id": "type-idx", "score": float}]
    victim_nodes  : []      (populated downstream by publisher/RAG)
    graph_id      : int     (tick counter)
    edge_xai      : {"method": str, "edges": [{"source","target","relation","weight"}]}
                    XAI edge weighting for the Grafana Node Graph (L6).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from gnn.model import GATv2Hetero
from snapshot_engine.attention import build_edge_xai
from snapshot_engine.rwr import rank_root_causes, restart_scores_from_edges

logger = logging.getLogger(__name__)

_DEFAULT_CKPT = Path(__file__).parent.parent / "checkpoints" / "best_model.pt"
_FAULT_TYPES = ["hdd_degradation", "network_congestion", "ram_leak"]


class InferenceEngine:
    """Wraps GATv2Hetero for single-graph prediction."""

    def __init__(self, model: GATv2Hetero, device: torch.device) -> None:
        self._model = model
        self._device = device
        self._tick = 0

    @classmethod
    def load(
        cls,
        ckpt_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ) -> "InferenceEngine":
        """Load model from checkpoint.

        torch.load(..., weights_only=False) required because checkpoint
        contains non-tensor objects (node_dims dict, edge_types list).
        """
        p = Path(ckpt_path) if ckpt_path else _DEFAULT_CKPT
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(str(p), map_location=dev, weights_only=False)

        model = GATv2Hetero(
            node_dims=ckpt["node_dims"],
            edge_types=ckpt["edge_types"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(dev)
        model.eval()

        logger.info(
            "Loaded GATv2Hetero from %s  (epoch=%s, val_f1=%.3f, rca_acc=%.3f)",
            p,
            ckpt.get("epoch", "?"),
            ckpt.get("val_f1", 0.0),
            ckpt.get("rca_accuracy", 0.0),
        )
        return cls(model, dev)

    @torch.no_grad()
    def predict(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
        edge_attr_dict: Dict[Tuple, torch.Tensor],
    ) -> Dict[str, Any]:
        """Run forward pass and return structured prediction.

        RC = argmax sigmoid(logits) across all non-rca_context node types.
        top5 = top-5 global candidates.
        """
        self._tick += 1

        # Move tensors to device
        x_dev = {k: v.to(self._device) for k, v in x_dict.items()}
        ei_dev = {k: v.to(self._device) for k, v in edge_index_dict.items()}
        ea_dev = {k: v.to(self._device) for k, v in edge_attr_dict.items()}

        logits: Dict[str, torch.Tensor] = self._model(x_dev, ei_dev, ea_dev)

        # Collect all (score, node_type, node_id) candidates.
        # scores_by_type keeps the full per-node score vector for edge XAI.
        all_candidates: List[Dict[str, Any]] = []
        scores_by_type: Dict[str, List[float]] = {}
        for nt, lg in logits.items():
            scores = torch.sigmoid(lg).cpu()
            scores_by_type[nt] = scores.tolist()
            for idx in range(scores.shape[0]):
                all_candidates.append({
                    "type":  nt,
                    "id":    idx,
                    "score": float(scores[idx].item()),
                })

        # Sort descending by score
        all_candidates.sort(key=lambda c: c["score"], reverse=True)

        # Local (per-node argmax) candidate — kept for audit / fallback.
        local_rc = all_candidates[0]
        top5 = all_candidates[:5]

        # Explainable-AI edge weighting for the Grafana Node Graph (L6).
        # Uses the static inference topology (edge_index_dict) so the weighted
        # edges line up with what /topology draws. Falls back to salience until
        # real GATv2 attention is wired (see snapshot_engine/attention.py).
        edge_xai = build_edge_xai(
            scores_by_type,
            edge_index_dict,
            model=self._model,
            x_dict=x_dev,
            edge_attr_dict=ea_dev,
        )

        # Root-Cause Analysis via Random Walk with Restart (L4, diploma 2.3.4).
        # RWR aggregates the local per-node scores (restart vector) and per-edge
        # weights (transition matrix) into a global ranking, concentrating mass
        # on the cascade root. Falls back to the local argmax when the XAI graph
        # is empty (no anomalous edges this tick).
        rwr_restart = restart_scores_from_edges(edge_xai["edges"], scores_by_type)
        rwr_ranking = rank_root_causes(edge_xai["edges"], rwr_restart)

        if rwr_ranking:
            rc_id_str = rwr_ranking[0]["id"]          # "{type}-{idx}"
            rc_type, _, rc_idx_s = rc_id_str.rpartition("-")
            rc = {"type": rc_type, "id": int(rc_idx_s)}
            rc_method = "rwr"
        else:
            rc = {"type": local_rc["type"], "id": local_rc["id"]}
            rc_method = "argmax_fallback"

        # confidence stays the node's anomaly probability (interpretable),
        # not the RWR mass (which is a relative ranking score).
        rc_confidence = scores_by_type.get(rc["type"], [0.0])[rc["id"]] \
            if rc["id"] < len(scores_by_type.get(rc["type"], [])) else local_rc["score"]

        # Heuristic fault_type from the RWR-selected root-cause node type.
        fault_map = {
            "hdd":    "hdd_degradation",
            "switch": "network_congestion",
            "ram":    "ram_leak",
        }
        fault_type = fault_map.get(rc["type"], "unknown")

        return {
            "graph_id":        self._tick,
            "fault_type":      fault_type,
            "rc_node":         {"type": rc["type"], "id": rc["id"]},
            "confidence":      round(float(rc_confidence), 4),
            "rc_method":       rc_method,
            "rc_node_local":   {"type": local_rc["type"], "id": local_rc["id"]},
            "rwr_ranking":     rwr_ranking,
            "top5_candidates": [{"id": f"{c['type']}-{c['id']}", "score": round(c['score'], 4)} for c in top5],
            "victim_nodes":    [],  # filled by downstream RAG/LLM layer
            "edge_xai":        edge_xai,
        }
