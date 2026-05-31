"""Parse structured playbook JSON from LLM raw output."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

ACTION_ALIASES = {
    "IDENTIFY_TOP_TALKER_JOBS": "CHECK_METRICS",
    "IDENTIFY_TOP_TRAFFIC_SOURCES": "CHECK_METRICS",
    "CHECK_PORT_UTILIZATION": "CHECK_METRICS",
    "RATE_LIMIT_TOP_TALKERS": "APPLY_QOS",
}


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract first JSON object from LLM response text."""
    text = text.strip()
    if not text:
        return None

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def normalize_playbook_dict(data: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing fields from inference defaults."""
    merged = dict(defaults)
    merged.update({k: v for k, v in data.items() if v is not None})
    for key in ("incident_id", "fault_type", "rc_node", "confidence"):
        merged[key] = defaults[key]
    if "actions" in data and isinstance(data["actions"], list):
        merged["actions"] = [
            _normalize_action(action, default_target=str(defaults["rc_node"]))
            for action in data["actions"]
        ]
    if "explanation" in data:
        merged["explanation"] = data["explanation"]
    return merged


def _normalize_action(action: Any, default_target: str) -> Any:
    if not isinstance(action, dict):
        return action
    normalized = dict(action)
    action_id = normalized.get("action_id")
    if isinstance(action_id, str) and action_id in ACTION_ALIASES:
        normalized["action_id"] = ACTION_ALIASES[action_id]
    if "target_node" not in normalized and "target" in normalized:
        normalized["target_node"] = normalized["target"]
    if not isinstance(normalized.get("target_node"), str) or not normalized.get("target_node"):
        normalized["target_node"] = default_target
    if not isinstance(normalized.get("parameters"), dict):
        description = normalized.get("description")
        normalized["parameters"] = {"description": description} if description else {}
    if "priority" in normalized:
        try:
            priority = int(normalized["priority"])
            normalized["priority"] = max(1, min(priority, 3))
        except (TypeError, ValueError):
            normalized["priority"] = 3
    else:
        normalized["priority"] = 3
    if "estimated_ttr_seconds" not in normalized:
        normalized["estimated_ttr_seconds"] = 60
    return normalized
