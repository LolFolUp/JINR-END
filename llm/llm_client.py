"""LLM inference with ollama/transformers fallback and rule-based playbook."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import requests

from llm.prompt_builder import build_incident_id, format_rc_node
from llm.response_parser import extract_json_object

logger = logging.getLogger(__name__)

RULE_BASED_PLAYBOOKS: Dict[str, List[Dict[str, Any]]] = {
    "network_congestion": [
        {
            "action_id": "CHECK_METRICS",
            "target_node": "{rc}",
            "parameters": {"metrics": ["port_utilization", "packet_loss"]},
            "priority": 1,
            "estimated_ttr_seconds": 30,
        },
        {
            "action_id": "APPLY_QOS",
            "target_node": "{rc}",
            "parameters": {"policy": "rate_limit_top_talkers"},
            "priority": 2,
            "estimated_ttr_seconds": 120,
        },
        {
            "action_id": "NOTIFY_OPERATOR",
            "target_node": "on-call",
            "parameters": {"message": "{rc} congestion detected"},
            "priority": 3,
            "estimated_ttr_seconds": 10,
        },
    ],
    "hdd_degradation": [
        {
            "action_id": "CHECK_METRICS",
            "target_node": "{rc}",
            "parameters": {"metrics": ["smart_reallocated", "io_latency_p95"]},
            "priority": 1,
            "estimated_ttr_seconds": 45,
        },
        {
            "action_id": "MIGRATE_JOB",
            "target_node": "{rc}",
            "parameters": {"threshold_percent": 70, "destination": "healthy_ost"},
            "priority": 2,
            "estimated_ttr_seconds": 600,
        },
        {
            "action_id": "SCHEDULE_MAINTENANCE",
            "target_node": "{rc}",
            "parameters": {"task": "disk_replacement"},
            "priority": 2,
            "estimated_ttr_seconds": 3600,
        },
        {
            "action_id": "NOTIFY_OPERATOR",
            "target_node": "on-call",
            "parameters": {"message": "HDD degradation on {rc}"},
            "priority": 3,
            "estimated_ttr_seconds": 10,
        },
    ],
    "ram_leak": [
        {
            "action_id": "CHECK_METRICS",
            "target_node": "{rc}",
            "parameters": {"metrics": ["rss_top_process", "oom_events"]},
            "priority": 1,
            "estimated_ttr_seconds": 30,
        },
        {
            "action_id": "RESTART_SERVICE",
            "target_node": "{rc}",
            "parameters": {"mode": "checkpoint_restart"},
            "priority": 2,
            "estimated_ttr_seconds": 180,
        },
        {
            "action_id": "MIGRATE_JOB",
            "target_node": "{rc}",
            "parameters": {"drain_node": True},
            "priority": 2,
            "estimated_ttr_seconds": 300,
        },
        {
            "action_id": "NOTIFY_OPERATOR",
            "target_node": "on-call",
            "parameters": {"message": "RAM leak suspected on {rc}"},
            "priority": 3,
            "estimated_ttr_seconds": 10,
        },
    ],
}


def _format_template(value: Any, rc_label: str) -> Any:
    if isinstance(value, str):
        return value.replace("{rc}", rc_label)
    if isinstance(value, dict):
        return {k: _format_template(v, rc_label) for k, v in value.items()}
    if isinstance(value, list):
        return [_format_template(v, rc_label) for v in value]
    return value


def generate_rule_based_playbook(inference: Dict[str, Any]) -> Dict[str, Any]:
    """Map fault_type to predefined actions without LLM."""
    fault_type = inference.get("fault_type", "unknown")
    rc_label = format_rc_node(inference.get("rc_node", {}))
    graph_id = inference.get("graph_id", 0)
    confidence = float(inference.get("confidence", 0.0))
    victim_count = len(inference.get("victim_nodes", []))

    templates = RULE_BASED_PLAYBOOKS.get(fault_type, RULE_BASED_PLAYBOOKS["network_congestion"])
    actions = []
    for tpl in templates:
        action = {k: _format_template(v, rc_label) for k, v in tpl.items()}
        actions.append(action)

    explanations = {
        "network_congestion": (
            f"Network congestion detected on {rc_label}. "
            f"{victim_count} compute nodes affected. Recommended immediate QoS intervention."
        ),
        "hdd_degradation": (
            f"HDD degradation on {rc_label}. SMART/IO checks required; "
            f"migrate data if health below threshold."
        ),
        "ram_leak": (
            f"RAM leak pattern on {rc_label}. Inspect RSS/OOM logs; "
            f"restart or migrate affected jobs."
        ),
    }

    return {
        "incident_id": build_incident_id(graph_id, fault_type),
        "fault_type": fault_type,
        "rc_node": rc_label,
        "confidence": confidence,
        "actions": actions,
        "explanation": explanations.get(
            fault_type,
            f"Rule-based remediation for {fault_type} on {rc_label}.",
        ),
        "rule_based_fallback": True,
    }


class LLMClient:
    """Try Ollama/Mistral first; fall back to rule-based generator."""

    def __init__(
        self,
        ollama_model: str = "mistral",
        transformers_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        force_rule_based: bool = False,
        enable_ollama: bool = True,
        enable_transformers: bool = False,
    ) -> None:
        self.ollama_model = ollama_model
        self.transformers_model = transformers_model
        self.force_rule_based = force_rule_based
        self.enable_ollama = enable_ollama
        self.enable_transformers = enable_transformers

    def generate(
        self,
        messages: List[Dict[str, str]],
        inference: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        """Return (playbook_dict, backend_label)."""
        if self.force_rule_based:
            return generate_rule_based_playbook(inference), "rule_based_fallback"

        if self.enable_ollama:
            raw = self._try_ollama(messages)
            if raw:
                parsed = extract_json_object(raw)
                if parsed:
                    return parsed, "ollama_mistral"

        if self.enable_transformers:
            raw = self._try_transformers(messages)
            if raw:
                parsed = extract_json_object(raw)
                if parsed:
                    return parsed, "transformers"

        logger.info("LLM unavailable; using rule-based fallback")
        return generate_rule_based_playbook(inference), "rule_based_fallback"

    def _try_ollama(self, messages: List[Dict[str, str]]) -> Optional[str]:
        prompt = self._messages_to_prompt(messages)
        return call_ollama_http(prompt, model=self.ollama_model, timeout=60)

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        parts = []
        for message in messages:
            role = message.get("role", "user").upper()
            content = message.get("content", "")
            parts.append(f"{role}:\n{content}")
        return "\n\n".join(parts)

    def _try_transformers(self, messages: List[Dict[str, str]]) -> Optional[str]:
        try:
            from transformers import pipeline

            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            generator = pipeline(
                "text-generation",
                model=self.transformers_model,
                max_new_tokens=512,
                do_sample=False,
            )
            out = generator(prompt, return_full_text=False)
            if out and isinstance(out, list):
                return out[0].get("generated_text", "")
        except Exception as exc:
            logger.debug("Transformers unavailable: %s", exc)
        return None


def call_ollama_http(prompt: str, model: str = "mistral", timeout: int = 60) -> Optional[str]:
    """Call local Ollama HTTP API; more reliable for cold starts than CLI chat."""
    try:
        resp = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "")
        logger.debug("Ollama HTTP status %s: %s", resp.status_code, resp.text)
    except Exception as exc:
        print(f"Ollama HTTP error: {exc}")
    return None
