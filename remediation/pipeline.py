"""End-to-end remediation pipeline orchestration."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from llm.llm_client import LLMClient
from llm.prompt_builder import (
    build_incident_id,
    build_messages,
    format_rc_node,
)
from llm.response_parser import normalize_playbook_dict
from rag.history_tickets import HistoryTicketsStore
from rag.qdrant_store import QdrantStore
from rag.redis_context import RedisContextStore
from remediation.firewall import build_fallback_playbook, validate_playbook
from remediation.incident_aggregator import IncidentAggregator
from remediation.models import RemediationPlaybook


def load_inference(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return normalize_inference(json.load(f))


def normalize_inference(inference: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt GNN branch artifacts to the RAG pipeline input contract."""
    normalized = dict(inference)

    if "rc_node" not in normalized and isinstance(normalized.get("true_rc"), dict):
        true_rc = normalized["true_rc"]
        normalized["rc_node"] = {
            "type": true_rc.get("node_type", "unknown"),
            "id": true_rc.get("label", true_rc.get("node_id", "unknown")),
            "host_id": true_rc.get("node_id"),
        }

    if "confidence" not in normalized:
        normalized["confidence"] = normalized.get("rc_score", 0.0)

    if "top5_candidates" not in normalized and "top5" in normalized:
        normalized["top5_candidates"] = normalized.get("top5", [])

    if "victim_nodes" not in normalized:
        victim_nodes = []
        for node_id in normalized.get("cpu_victim_ids", []):
            victim_nodes.append({"id": f"cpu-{node_id}", "type": "host", "subsystem": "cpu"})
        for node_id in normalized.get("gpu_victim_ids", []):
            victim_nodes.append({"id": f"gpu-{node_id}", "type": "host", "subsystem": "gpu"})
        # CPU/GPU lists describe the same affected hosts for network incidents.
        if normalized.get("cpu_victim_ids") and normalized.get("gpu_victim_ids"):
            unique_ids = sorted(set(normalized.get("cpu_victim_ids", [])))
            victim_nodes = [
                {"id": f"host-{node_id:03d}", "type": "host"}
                for node_id in unique_ids
            ]
        normalized["victim_nodes"] = victim_nodes

    return normalized


def build_retrieval_query(inference: Dict[str, Any]) -> str:
    fault = inference.get("fault_type", "")
    rc = format_rc_node(inference.get("rc_node", {}))
    victims = len(inference.get("victim_nodes", []))
    return f"{fault} root cause {rc} affected nodes {victims}"


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def run_pipeline(
    inference: Dict[str, Any],
    *,
    force_rule_based: bool = False,
    llm_client: Optional[LLMClient] = None,
    context_store: Optional[RedisContextStore] = None,
    qdrant_store: Optional[QdrantStore] = None,
    history_store: Optional[HistoryTicketsStore] = None,
) -> Tuple[RemediationPlaybook, Dict[str, Any]]:
    """Execute full RAG + LLM + firewall flow.

    Returns:
        (playbook, metadata) where metadata includes firewall_status, backend, etc.
    """
    total_start = time.perf_counter()
    inference = normalize_inference(inference)
    fault_type = inference.get("fault_type", "unknown")
    graph_id = inference.get("graph_id", 0)
    confidence = float(inference.get("confidence", 0.0))
    rc_node = inference.get("rc_node", {})
    rc_label = format_rc_node(rc_node)
    incident_id = build_incident_id(graph_id, fault_type)

    incident = IncidentAggregator().aggregate(inference)
    store = context_store or RedisContextStore()
    node_context, context_source = store.fetch_context(rc_node)

    rag_start = time.perf_counter()
    vector_store = qdrant_store or QdrantStore()
    query = build_retrieval_query(inference)
    sop_chunks, retrieval_method = vector_store.retrieve(
        fault_type=fault_type,
        rc_node_type=str(rc_node.get("type", "unknown")),
        query=query,
        top_k=3,
    )
    history = history_store or HistoryTicketsStore()
    similar_incidents = history.find_similar(fault_type, limit=3)
    rag_retrieval_ms = _elapsed_ms(rag_start)

    sop_texts = [str(chunk["text"]) for chunk in sop_chunks]
    messages = build_messages(
        inference,
        sop_texts,
        incident=incident.as_dict(),
        node_context=node_context,
        similar_incidents=similar_incidents,
    )
    client = llm_client or LLMClient(force_rule_based=force_rule_based)
    llm_start = time.perf_counter()
    raw_playbook, backend = client.generate(messages, inference)
    llm_generation_ms = _elapsed_ms(llm_start)

    defaults = {
        "incident_id": incident_id,
        "fault_type": fault_type,
        "rc_node": rc_label,
        "confidence": confidence,
        "actions": [],
        "explanation": "",
    }
    normalized = normalize_playbook_dict(raw_playbook, defaults)

    validation_start = time.perf_counter()
    playbook, passed, err = validate_playbook(normalized)
    validation_ms = _elapsed_ms(validation_start)
    firewall_status = "PASSED" if passed else "BLOCKED"

    if not passed:
        playbook = build_fallback_playbook(
            incident_id=incident_id,
            fault_type=fault_type,
            rc_node=rc_label,
            confidence=confidence,
            reason=err or "validation failed",
        )

    ttr_breakdown = {
        "gnn_inference_ms": int(inference.get("gnn_inference_ms", 0)),
        "rag_retrieval_ms": rag_retrieval_ms,
        "llm_generation_ms": llm_generation_ms,
        "validation_ms": validation_ms,
        "total_ms": _elapsed_ms(total_start),
    }
    metadata = {
        "incident": incident.as_dict(),
        "context": {
            "rc_hostname": node_context.get("hostname"),
            "rc_os": node_context.get("os"),
            "context_source": context_source,
            "node_context": node_context,
        },
        "knowledge": {
            "sop_chunks_retrieved": len(sop_chunks),
            "retrieval_method": retrieval_method,
            "sop_chunks": [
                {"title": chunk.get("title"), "fault_type": chunk.get("fault_type")}
                for chunk in sop_chunks
            ],
            "similar_incidents": similar_incidents,
        },
        "ttr_breakdown": ttr_breakdown,
        "firewall_status": firewall_status,
        "firewall_error": err,
        "llm_source": backend,
        "llm_backend": backend,
        "rule_based_fallback": backend == "rule_based_fallback"
        or raw_playbook.get("rule_based_fallback", False),
    }
    return playbook, metadata


def playbook_to_dict(playbook: RemediationPlaybook) -> Dict[str, Any]:
    return playbook.model_dump()


def remediation_report_payload(playbook: RemediationPlaybook, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Build thesis-aligned structured JSON report."""
    return {
        "incident": metadata["incident"],
        "context": {
            "rc_hostname": metadata["context"].get("rc_hostname"),
            "rc_os": metadata["context"].get("rc_os"),
            "context_source": metadata["context"].get("context_source"),
        },
        "knowledge": metadata["knowledge"],
        "actions": [action.model_dump() for action in playbook.actions],
        "explanation": playbook.explanation,
        "playbook": playbook_to_dict(playbook),
        "ttr_breakdown": metadata["ttr_breakdown"],
        "firewall_status": metadata["firewall_status"],
        "llm_source": metadata.get("llm_source"),
        "llm_backend": metadata.get("llm_backend"),
        "rule_based_fallback": metadata.get("rule_based_fallback", False),
    }
