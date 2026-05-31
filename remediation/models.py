"""Pydantic schemas for ActionDSL remediation playbooks."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator

ALLOWED_ACTIONS = {
    "CHECK_METRICS",
    "ISOLATE_NODE",
    "RESTART_SERVICE",
    "MIGRATE_JOB",
    "NOTIFY_OPERATOR",
    "SCHEDULE_MAINTENANCE",
    "APPLY_QOS",
}

PRIORITY_LABELS = {1: "CRITICAL", 2: "HIGH", 3: "NORMAL"}


class Action(BaseModel):
    """Single remediation step in the playbook."""

    action_id: str
    target_node: str
    parameters: dict = Field(default_factory=dict)
    priority: int = Field(ge=1, le=3)
    estimated_ttr_seconds: int = Field(ge=0)

    @field_validator("action_id")
    @classmethod
    def validate_action_id(cls, value: str) -> str:
        if value not in ALLOWED_ACTIONS:
            raise ValueError(
                f"action_id '{value}' not in allowed actions: {sorted(ALLOWED_ACTIONS)}"
            )
        return value


class RemediationPlaybook(BaseModel):
    """Structured remediation plan for an RCA incident."""

    incident_id: str
    fault_type: str
    rc_node: str
    confidence: float = Field(ge=0.0, le=1.0)
    actions: List[Action]
    explanation: str
