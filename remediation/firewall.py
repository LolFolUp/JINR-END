"""Semantic firewall: validate LLM output against ActionDSL."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from remediation.models import Action, RemediationPlaybook


def validate_playbook(data: Dict[str, Any]) -> Tuple[Optional[RemediationPlaybook], bool, Optional[str]]:
    """Validate raw dict as RemediationPlaybook.

    Returns:
        (playbook, passed, error_message)
    """
    try:
        playbook = RemediationPlaybook.model_validate(data)
        return playbook, True, None
    except ValidationError as exc:
        return None, False, str(exc)


def build_fallback_playbook(
    incident_id: str,
    fault_type: str,
    rc_node: str,
    confidence: float,
    reason: str,
) -> RemediationPlaybook:
    """Return safe fallback with NOTIFY_OPERATOR only."""
    return RemediationPlaybook(
        incident_id=incident_id,
        fault_type=fault_type,
        rc_node=rc_node,
        confidence=confidence,
        actions=[
            Action(
                action_id="NOTIFY_OPERATOR",
                target_node="on-call",
                parameters={
                    "message": f"Remediation blocked by firewall: {reason}",
                    "escalation": True,
                },
                priority=1,
                estimated_ttr_seconds=10,
            )
        ],
        explanation=(
            f"LLM output failed semantic firewall validation. "
            f"Escalated to operator. Reason: {reason}"
        ),
    )


def validate_actions_list(actions: List[Dict[str, Any]]) -> List[Action]:
    """Validate each action; raises ValidationError on disallowed action_id."""
    return [Action.model_validate(a) for a in actions]
