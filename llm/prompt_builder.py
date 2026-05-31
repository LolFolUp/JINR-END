"""Construct LLM prompts from GNN inference output and RAG context."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


ALLOWED_ACTIONS_STR = (
    "CHECK_METRICS, ISOLATE_NODE, RESTART_SERVICE, MIGRATE_JOB, "
    "NOTIFY_OPERATOR, SCHEDULE_MAINTENANCE, APPLY_QOS"
)


SYSTEM_PROMPT = f"""You are an HPC cluster remediation assistant.
Given GNN root-cause analysis and SOP excerpts, produce a remediation playbook as JSON only.

Allowed action_id values (use only these): {ALLOWED_ACTIONS_STR}
priority: 1=critical, 2=high, 3=normal
estimated_ttr_seconds: realistic seconds for each step
Use the exact confidence value from the incident payload. Do not use logits as confidence.
Every action priority MUST be one of 1, 2, or 3. Never output 4 or 5.

Grounding rules:
- The root-cause node and fault_type were localised by a GNN model (source="gnn").
  Do NOT change the root-cause node or fault_type unless an SOP excerpt or the
  technical context explicitly contradicts it. If you are not convinced, your
  FIRST action must be CHECK_METRICS to gather evidence, plus NOTIFY_OPERATOR.
- Produce an engineering playbook using ONLY the allowed structured actions above.
  Never emit free-form shell/bash commands, scripts, or destructive operations.
- Prefer actions that target the identified root-cause node and protect the
  affected nodes / running jobs.

Output JSON schema:
{{
  "incident_id": "graph_<id>_<fault_type>",
  "fault_type": "<fault_type>",
  "rc_node": "<human readable RC node>",
  "confidence": <float 0-1>,
  "actions": [
    {{
      "action_id": "<ALLOWED>",
      "target_node": "<target>",
      "parameters": {{}},
      "priority": 1,
      "estimated_ttr_seconds": 30
    }}
  ],
  "explanation": "<brief summary>"
}}

Return ONLY valid JSON. No markdown. Order actions by priority (critical first).
If unsure, prefer NOTIFY_OPERATOR over actions outside the allowed list.
"""


def format_rc_node(rc_node: Dict[str, Any]) -> str:
    node_type = rc_node.get("type", "unknown")
    node_id = rc_node.get("id", "unknown")
    if node_type == "switch":
        return f"Switch {node_id}"
    if node_type == "hdd":
        return f"HDD {node_id}"
    if node_type == "ram":
        host = rc_node.get("host_id", "?")
        return f"RAM node {node_id} (host {host})"
    return f"{node_type} {node_id}"


def build_incident_id(graph_id: int | str, fault_type: str) -> str:
    return f"graph_{graph_id}_{fault_type}"


def build_user_prompt(
    inference: Dict[str, Any],
    sop_contexts: List[str],
    incident: Optional[Dict[str, Any]] = None,
    node_context: Optional[Dict[str, Any]] = None,
    similar_incidents: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build user message payload for LLM."""
    rc_node = inference.get("rc_node", {})
    victim_count = len(inference.get("victim_nodes", []))
    payload = {
        "graph_id": inference.get("graph_id"),
        "fault_type": inference.get("fault_type"),
        "fault_type_provenance": inference.get("fault_type_provenance"),
        "rc_node": rc_node,
        "rc_logit": inference.get("rc_logit"),
        "confidence": inference.get("confidence"),
        "victim_node_count": victim_count,
        "affected_counts": inference.get("affected_counts", {}),
        "key_anomalous_metrics": inference.get("key_metrics", {}),
        "top5_candidates": inference.get("top5_candidates", [])[:5],
        "gnn_provenance": inference.get("gnn_rca", {}),
        "aggregated_incident": incident or {},
        "technical_context": node_context or {},
        "sop_excerpts": sop_contexts,
        "similar_past_incidents": similar_incidents or [],
    }
    return (
        "Generate a remediation playbook for this incident:\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
    )


def build_messages(
    inference: Dict[str, Any],
    sop_contexts: List[str],
    incident: Optional[Dict[str, Any]] = None,
    node_context: Optional[Dict[str, Any]] = None,
    similar_incidents: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                inference,
                sop_contexts,
                incident=incident,
                node_context=node_context,
                similar_incidents=similar_incidents,
            ),
        },
    ]
