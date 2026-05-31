"""Adapter: GNN RCA inference output -> LLM/RAG incident contract.

The remediation pipeline (``remediation/pipeline.run_pipeline``) consumes an
``inference`` dict with a well-defined shape (see ``remediation/pipeline.py:
normalize_inference`` and ``llm/prompt_builder.py:build_user_prompt``):

    fault_type, graph_id, rc_node{type,id,host_id}, confidence, rc_logit,
    top5_candidates[{type,id,score}], victim_nodes[{id,type}], gnn_inference_ms

This module produces exactly that contract from the structured output of
``gnn.inference.GNNInferenceEngine.run()``, plus a richer, human-readable
``incident_context`` block (root cause, top-3, anomalous metrics, affected
nodes, provenance, run id / timestamp, ``source="gnn"``) that is forwarded to
the prompt so the LLM sees *what the GNN found* and is told not to overturn the
localisation without evidence.

Design choices (honest by construction):
  * ``confidence`` is the GNN's *localisation* score = ``sigmoid(logit)`` of the
    top candidate — how strongly it beats the other RC candidates in this graph.
    It is NOT a calibrated fault probability.
  * cpu/gpu neighbours that share the same intra-type index describe the same
    physical host, so they are de-duplicated into one ``host-<id>`` victim.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

SOURCE = "gnn"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _victim_nodes(affected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse topological neighbours into deduplicated victim records.

    cpu-i and gpu-i are the same physical host -> one host-i victim.
    Other node types are kept under their own type.
    """
    hosts: set[int] = set()
    others: List[Dict[str, Any]] = []
    for a in affected:
        nt = a.get("node_type")
        nid = a.get("node_id")
        if nt in ("cpu", "gpu"):
            hosts.add(int(nid))
        else:
            others.append({"id": f"{nt}-{nid}", "type": nt})
    victims: List[Dict[str, Any]] = [
        {"id": f"host-{i:03d}", "type": "host"} for i in sorted(hosts)
    ]
    victims.extend(others)
    return victims


def build_incident_context(gnn_out: Dict[str, Any]) -> Dict[str, Any]:
    """Rich, structured incident context derived from GNN output."""
    rca = gnn_out.get("rca", {})
    rc = rca.get("root_cause") or {}
    top_k = rca.get("top_k", [])
    fault = gnn_out.get("fault_type_hint", {}) or {}
    gctx = gnn_out.get("graph_context", {}) or {}
    model = gnn_out.get("model", {}) or {}

    return {
        "run_id": f"{gnn_out.get('incident_id', 'incident')}-{uuid.uuid4().hex[:8]}",
        "timestamp": _now_iso(),
        "source": SOURCE,
        "incident_id": gnn_out.get("incident_id"),
        "graph_id": gnn_out.get("graph_id"),
        "root_cause": {
            "node_type": rc.get("node_type"),
            "node_id": rc.get("node_id"),
            "node_label": rc.get("node_label"),
            "score": rc.get("score"),
            "rank": rc.get("rank", 1),
        },
        "top3_candidates": [
            {
                "rank": c.get("rank"),
                "node_type": c.get("node_type"),
                "node_label": c.get("node_label"),
                "score": c.get("score"),
            }
            for c in top_k[:3]
        ],
        "fault_type_hint": fault.get("value"),
        "fault_type_provenance": fault.get("provenance"),
        "key_anomalous_metrics": gctx.get("key_metrics", {}),
        "affected_counts": gctx.get("affected_counts", {}),
        "affected_node_sample": gctx.get("affected_nodes", [])[:10],
        "model": {
            "name": model.get("name"),
            "checkpoint": model.get("checkpoint"),
            "dataset_version": model.get("dataset_version"),
            "val_hit1": model.get("val_hit1"),
        },
        "gnn_inference_ms": (gnn_out.get("timing", {}) or {}).get("gnn_inference_ms", 0),
    }


def gnn_to_inference(gnn_out: Dict[str, Any]) -> Dict[str, Any]:
    """Convert GNN output into the ``inference`` dict consumed by run_pipeline."""
    rca = gnn_out.get("rca", {})
    rc = rca.get("root_cause") or {}
    top_k = rca.get("top_k", [])
    fault = gnn_out.get("fault_type_hint", {}) or {}
    gctx = gnn_out.get("graph_context", {}) or {}
    model = gnn_out.get("model", {}) or {}

    rc_type = rc.get("node_type", "unknown")
    rc_label = rc.get("node_label") or rc.get("node_id")
    rc_score = float(rc.get("score", 0.0))

    top5 = [
        {
            "type": c.get("node_type"),
            "id": c.get("node_label"),
            "score": c.get("score"),
            "rank": c.get("rank"),
        }
        for c in top_k[:5]
    ]
    victim_nodes = _victim_nodes(gctx.get("affected_nodes", []))
    incident_context = build_incident_context(gnn_out)

    inference: Dict[str, Any] = {
        "source": SOURCE,
        "graph_id": gnn_out.get("graph_id", 0),
        "incident_id": gnn_out.get("incident_id"),
        "fault_type": fault.get("value", "unknown"),
        "fault_type_provenance": fault.get("provenance"),
        "rc_node": {
            "type": rc_type,
            "id": rc_label,
            "host_id": rc.get("node_id"),
            "node_index": rc.get("node_id"),
        },
        "rc_rank": rc.get("rank", 1),
        "rc_score": rc_score,
        "rc_logit": rc.get("logit"),
        "confidence": rc_score,
        "top5_candidates": top5,
        "victim_nodes": victim_nodes,
        "key_metrics": gctx.get("key_metrics", {}),
        "affected_counts": gctx.get("affected_counts", {}),
        "gnn": {
            "model": model.get("name"),
            "checkpoint": model.get("checkpoint"),
            "dataset_version": model.get("dataset_version"),
            "val_hit1": model.get("val_hit1"),
            "predicted_correct": (gnn_out.get("ground_truth") or {}).get("predicted_correct"),
        },
        "gnn_rca": incident_context,
        "gnn_inference_ms": incident_context["gnn_inference_ms"],
    }
    return inference
