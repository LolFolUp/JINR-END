"""Unified GNN RCA inference interface.

Loads the trained ``GATv2Hetero`` checkpoint (v5a_40, Hit@1 = 87.5%) and runs a
single forward pass over a ``HeteroData`` graph to localise the root-cause node
and rank the top-k RC candidates.

The output is a structured, JSON-serialisable dict consumed by
``integrations/gnn_to_incident.py`` (which adapts it into the LLM/RAG incident
contract).

Honesty notes (important — see reports/gnn_llm_end_to_end_integration_ru.md):
  * ``score`` is ``sigmoid(per-node logit)``. Candidates are ranked by this
    score within the per-graph RC-candidate pool (cpu/gpu/ram/hdd/switch).
  * The GNN localises *where* the root cause is (which node). It does **not**
    classify *what* the fault type is. ``fault_type_hint`` is taken from the
    synthetic graph's ``fault_type_idx`` when present (provenance
    ``synthetic_ground_truth``); in production this hint must come from a
    separate fault classifier or operator heuristic — never from this model.
  * ``affected_nodes`` are the *topological* physical neighbours of the
    predicted root cause (derived from the graph's edge_index), not
    ground-truth victim labels.

CLI:
    python -m gnn.inference --sample demo_data/gnn_samples/data_3.pt
    python -m gnn.inference --sample <path> --checkpoint <ckpt> --top-k 5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from gnn.model import GATv2Hetero, RC_CANDIDATE_TYPES

# ---------------------------------------------------------------------------
# Defaults — overridable via CLI / env / constructor
# ---------------------------------------------------------------------------

_PKG_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = Path(
    os.environ.get("GNN_CHECKPOINT", _PKG_DIR / "checkpoints" / "best_v5a_40_screening.pt")
)
DEFAULT_METADATA = Path(
    os.environ.get("GNN_METADATA", _PKG_DIR / "artifacts" / "metadata.json")
)

# Canonical 9 fault types (index order must match fault_type_idx in the dataset).
FAULT_TYPES: List[str] = [
    "hdd_degradation",              # 0
    "network_congestion",           # 1
    "ram_leak",                     # 2
    "cpu_frequency_drop",           # 3
    "cpu_cache_thrashing",          # 4
    "memory_bandwidth_saturation",  # 5
    "swap_thrashing",               # 6
    "gpu_thermal_throttle",         # 7
    "disk_full",                    # 8
]

# Human-readable node-id prefixes (index-based; production would resolve to CMDB).
_ID_PREFIX = {"switch": "S", "hdd": "HDD-", "ram": "RAM-", "cpu": "CPU-", "gpu": "GPU-"}

# Continuous temporal channels per feature: [value, delta_short, delta_long, rolling_var]
_DELTA_LONG_CHANNEL = 2
_CHANNELS_PER_CONT = 4


def human_node_id(node_type: str, node_id: int) -> str:
    """Index-based human label, e.g. ('switch', 3) -> 'S3'."""
    return f"{_ID_PREFIX.get(node_type, node_type + '-')}{node_id}"


class GNNInferenceEngine:
    """Load a trained GATv2Hetero checkpoint and run single-graph RCA inference."""

    def __init__(
        self,
        checkpoint_path: os.PathLike | str = DEFAULT_CHECKPOINT,
        metadata_path: os.PathLike | str = DEFAULT_METADATA,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.metadata_path = Path(metadata_path)
        self.device = torch.device(device)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"GNN checkpoint not found: {self.checkpoint_path}\n"
                "Set GNN_CHECKPOINT or pass --checkpoint. See "
                "reports/gnn_llm_end_to_end_integration_ru.md for how to obtain it."
            )

        self.metadata: Dict[str, Any] = {}
        if self.metadata_path.exists():
            with self.metadata_path.open(encoding="utf-8") as f:
                self.metadata = json.load(f)

        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        node_dims = ckpt.get("node_dims") or self.metadata.get("feature_dimensions")
        edge_types = ckpt.get("edge_types") or [tuple(et) for et in self.metadata.get("edge_types", [])]
        if node_dims is None or not edge_types:
            raise ValueError("Checkpoint/metadata missing node_dims or edge_types.")

        self.scorer_mode = ckpt.get("scorer_mode", "shared")
        self.loss_mode = ckpt.get("loss_mode", "global_ce")
        self.epoch = ckpt.get("epoch")
        self.val_hit1 = ckpt.get("hit1")
        # The checkpoint stores a schema/format tag (e.g. "v2.0.0"); the dataset
        # *codename* is carried in the filename (best_v5a_40_screening.pt).
        self.schema_version = ckpt.get("dataset_version")
        name = self.checkpoint_path.name
        if "v5a_40" in name:
            self.dataset_version = "v5a_40"
        else:
            self.dataset_version = self.schema_version or "unknown"

        self.model = GATv2Hetero(
            node_dims=node_dims,
            edge_types=edge_types,
            scorer_mode=self.scorer_mode,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())

    # -- public API ---------------------------------------------------------

    def run(
        self,
        graph: Any,
        *,
        incident_id: Optional[str] = None,
        top_k: int = 5,
        anomalous_metrics: int = 5,
    ) -> Dict[str, Any]:
        """Run inference on a HeteroData graph (or path to a .pt file)."""
        graph_path: Optional[str] = None
        if isinstance(graph, (str, os.PathLike)):
            graph_path = str(graph)
            graph = torch.load(graph_path, weights_only=False)

        t0 = time.perf_counter()
        logits = self._forward(graph)
        inference_ms = int((time.perf_counter() - t0) * 1000)

        ranked = self._rank_candidates(logits, graph)  # full ranking
        candidate_count = len(ranked)
        top = ranked[:top_k]

        fault_value, fault_prov = self._fault_hint(graph, ranked)

        rc = ranked[0] if ranked else None
        gt = self._ground_truth(graph, ranked)

        graph_id = self._graph_id(graph, incident_id, graph_path)
        inc_id = incident_id or (
            f"graph_{graph_id}_{fault_value}" if fault_value else f"graph_{graph_id}"
        )

        key_metrics: Dict[str, float] = {}
        affected: List[Dict[str, Any]] = []
        if rc is not None:
            key_metrics = self._anomalous_metrics(
                graph, rc["node_type"], rc["node_id"], top_n=anomalous_metrics
            )
            affected = self._topological_neighbours(graph, rc["node_type"], rc["node_id"])

        top_k_out: List[Dict[str, Any]] = []
        for rank, c in enumerate(top, start=1):
            top_k_out.append(
                {
                    "rank": rank,
                    "node_type": c["node_type"],
                    "node_id": c["node_id"],
                    "node_label": human_node_id(c["node_type"], c["node_id"]),
                    "score": round(c["score"], 6),
                    "logit": round(c["logit"], 6),
                    "fault_type_hint": fault_value if rank == 1 else None,
                }
            )

        notes = [
            "score = sigmoid(per-node logit); ranked within the per-graph RC-candidate pool.",
            "GNN localises the root-cause NODE; it does not classify the fault type.",
            "affected_nodes are topological neighbours of the RC (from edge_index), not victim labels.",
            "node_id is a synthetic intra-type index; production resolves it to a CMDB hostname.",
        ]

        result: Dict[str, Any] = {
            "incident_id": inc_id,
            "graph_id": graph_id,
            "source": "gnn",
            "model": {
                "name": "GATv2Hetero",
                "checkpoint": self.checkpoint_path.name,
                "dataset_version": self.dataset_version,
                "schema_version": self.schema_version,
                "scorer_mode": self.scorer_mode,
                "loss_mode": self.loss_mode,
                "trained_epoch": self.epoch,
                "val_hit1": round(float(self.val_hit1), 4) if self.val_hit1 is not None else None,
                "params": self.n_params,
            },
            "score_semantics": "sigmoid(logit)",
            "rca": {
                "root_cause": top_k_out[0] if top_k_out else None,
                "top_k": top_k_out,
                "hit_metadata": {
                    "candidate_count": candidate_count,
                    "rc_candidate_types": list(RC_CANDIDATE_TYPES),
                },
            },
            "fault_type_hint": {
                "value": fault_value,
                "provenance": fault_prov,
                "note": (
                    "GNN does not predict fault class. In production derive this from a "
                    "separate fault classifier or operator heuristic, not from the GNN."
                ),
            },
            "graph_context": {
                "affected_nodes": affected,
                "affected_counts": self._counts_by_type(affected),
                "key_metrics": key_metrics,
                "notes": notes,
            },
            "timing": {"gnn_inference_ms": inference_ms},
        }
        if gt is not None:
            result["ground_truth"] = gt
        return result

    # -- internals ----------------------------------------------------------

    def _forward(self, graph: Any) -> Dict[str, torch.Tensor]:
        x_dict = {nt: graph[nt].x.float().to(self.device) for nt in graph.node_types}
        ei_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        ea_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        for et in graph.edge_types:
            ei_dict[et] = graph[et].edge_index.to(self.device)
            ea_dict[et] = graph[et].edge_attr.float().to(self.device)
        with torch.no_grad():
            return self.model(x_dict, ei_dict, ea_dict)

    def _rank_candidates(
        self, logits: Dict[str, torch.Tensor], graph: Any
    ) -> List[Dict[str, Any]]:
        cands: List[Dict[str, Any]] = []
        for nt in RC_CANDIDATE_TYPES:
            if nt not in logits:
                continue
            lg = logits[nt].float().squeeze(-1)
            if lg.dim() == 0:
                lg = lg.unsqueeze(0)
            sc = torch.sigmoid(lg)
            for i in range(lg.shape[0]):
                cands.append(
                    {"node_type": nt, "node_id": i, "score": float(sc[i]), "logit": float(lg[i])}
                )
        cands.sort(key=lambda c: c["score"], reverse=True)
        return cands

    def _anomalous_metrics(
        self, graph: Any, node_type: str, node_id: int, top_n: int = 5
    ) -> Dict[str, float]:
        """Top-|delta_long| continuous features of the RC node (Z-normalised).

        delta_long (feature value minus its EMA) is the channel the dataset
        diagnostics use to measure how far a node has drifted from its own
        healthy baseline; large |delta_long| flags the anomalous metrics.
        """
        ordering = (self.metadata.get("feature_ordering", {}) or {}).get(node_type)
        if ordering is None:
            return {}
        cont = ordering.get("continuous", [])
        try:
            row = graph[node_type].x[node_id].float()
        except Exception:
            return {}
        scored: List[Tuple[str, float]] = []
        for i, feat in enumerate(cont):
            idx = i * _CHANNELS_PER_CONT + _DELTA_LONG_CHANNEL
            if idx < row.shape[0]:
                scored.append((feat, float(row[idx])))
        scored.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return {feat: round(val, 4) for feat, val in scored[:top_n]}

    def _topological_neighbours(
        self, graph: Any, rc_type: str, rc_id: int, cap: int = 64
    ) -> List[Dict[str, Any]]:
        """Physical neighbours of the RC node, derived from edge_index."""
        seen: set[Tuple[str, int]] = set()
        out: List[Dict[str, Any]] = []
        for et in graph.edge_types:
            src, rel, dst = et
            if rel.startswith("rev_") or rel == "reports_to" or rel == "rev_uplink_to":
                continue
            ei = graph[et].edge_index
            if src == rc_type:
                mask = ei[0] == rc_id
                for nid in ei[1][mask].tolist():
                    key = (dst, int(nid))
                    if dst != "rca_context" and key not in seen:
                        seen.add(key)
                        out.append({"node_type": dst, "node_id": int(nid), "relation": rel})
            if dst == rc_type:
                mask = ei[1] == rc_id
                for nid in ei[0][mask].tolist():
                    key = (src, int(nid))
                    if src != "rca_context" and key not in seen:
                        seen.add(key)
                        out.append({"node_type": src, "node_id": int(nid), "relation": rel})
            if len(out) >= cap:
                break
        return out[:cap]

    @staticmethod
    def _counts_by_type(affected: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for a in affected:
            counts[a["node_type"]] = counts.get(a["node_type"], 0) + 1
        return counts

    def _fault_hint(self, graph: Any, ranked: List[Dict[str, Any]]) -> Tuple[Optional[str], str]:
        """Return (fault_type_value, provenance)."""
        fi = getattr(graph, "fault_type_idx", None)
        if fi is not None:
            try:
                idx = int(fi)
            except (TypeError, ValueError):
                idx = -99
            if 0 <= idx < len(FAULT_TYPES):
                return FAULT_TYPES[idx], "synthetic_ground_truth"
            if idx == -1:
                return None, "healthy_graph"
        # Fallback heuristic from predicted RC type (lossy; documented).
        if ranked:
            rc_type = ranked[0]["node_type"]
            heuristic = {
                "switch": "network_congestion",
                "hdd": "hdd_degradation",
                "ram": "ram_leak",
                "cpu": "cpu_frequency_drop",
                "gpu": "gpu_thermal_throttle",
            }.get(rc_type)
            if heuristic:
                return heuristic, "heuristic_from_rc_type"
        return None, "unknown"

    def _ground_truth(self, graph: Any, ranked: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """If y labels exist (synthetic), report true RC and whether we hit it."""
        true_rc: Optional[Tuple[str, int]] = None
        for nt in RC_CANDIDATE_TYPES:
            store = graph[nt] if nt in graph.node_types else None
            if store is None or not hasattr(store, "y") or store.y is None:
                continue
            y = store.y
            if int(y.sum()) > 0:
                true_rc = (nt, int((y == 1).nonzero(as_tuple=True)[0][0].item()))
                break
        if true_rc is None:
            return None
        rc_rank = None
        for rank, c in enumerate(ranked, start=1):
            if c["node_type"] == true_rc[0] and c["node_id"] == true_rc[1]:
                rc_rank = rank
                break
        return {
            "rc_node_type": true_rc[0],
            "rc_node_id": true_rc[1],
            "rc_node_label": human_node_id(*true_rc),
            "rc_rank": rc_rank,
            "predicted_correct": rc_rank == 1,
            "note": "ground_truth present only because input is a synthetic labelled graph.",
        }

    @staticmethod
    def _graph_id(graph: Any, incident_id: Optional[str], graph_path: Optional[str]) -> Any:
        if getattr(graph, "graph_id", None) is not None:
            try:
                return int(graph.graph_id)
            except (TypeError, ValueError):
                return str(graph.graph_id)
        if graph_path:
            stem = Path(graph_path).stem  # data_3 -> 3
            digits = "".join(ch for ch in stem if ch.isdigit())
            if digits:
                return int(digits)
            return stem
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="GNN RCA inference (single graph)")
    parser.add_argument("--sample", required=True, help="Path to a HeteroData .pt graph")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="", help="Optional path to write JSON result")
    args = parser.parse_args()

    engine = GNNInferenceEngine(checkpoint_path=args.checkpoint, metadata_path=args.metadata)
    result = engine.run(args.sample, top_k=args.top_k)

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"\nSaved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
