"""Aggregate GNN victim nodes into a single incident object."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class Incident:
    id: str
    severity: str
    summary: str
    affected_nodes_count: int
    aggregated_from_alerts: int
    victim_groups: Dict[str, int]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity,
            "summary": self.summary,
            "affected_nodes_count": self.affected_nodes_count,
            "aggregated_from_alerts": self.aggregated_from_alerts,
            "victim_groups": dict(self.victim_groups),
        }


class IncidentAggregator:
    """Collapse many alert-like victim nodes into one RCA incident."""

    def aggregate(self, inference: Dict[str, Any]) -> Incident:
        fault_type = str(inference.get("fault_type", "unknown"))
        graph_id = inference.get("graph_id", "unknown")
        rc_node = inference.get("rc_node", {})
        victims = inference.get("victim_nodes", []) or []
        victim_groups = self._group_victims(victims)
        affected_count = len(victims)
        severity = self._severity(fault_type, affected_count)
        aggregated_alerts = self._estimate_alert_count(fault_type, affected_count)
        summary = self._summary(fault_type, rc_node, affected_count, victim_groups)
        return Incident(
            id=f"INC-2026-graph{graph_id}",
            severity=severity,
            summary=summary,
            affected_nodes_count=affected_count,
            aggregated_from_alerts=aggregated_alerts,
            victim_groups=victim_groups,
        )

    def _group_victims(self, victim_nodes: list[Any]) -> Dict[str, int]:
        groups: Counter[str] = Counter()
        for victim in victim_nodes:
            if isinstance(victim, dict):
                node_type = str(victim.get("type") or victim.get("node_type") or "host")
            else:
                node_type = "host"
            groups[node_type] += 1
        return dict(groups)

    def _severity(self, fault_type: str, affected_count: int) -> str:
        if affected_count > 20 or fault_type == "network_congestion":
            return "CRITICAL"
        if 10 <= affected_count <= 20:
            return "HIGH"
        if 5 <= affected_count < 10:
            return "MEDIUM"
        return "LOW"

    def _estimate_alert_count(self, fault_type: str, affected_count: int) -> int:
        multiplier = 3 if fault_type == "network_congestion" else 2
        base = max(1, affected_count * multiplier)
        return base + min(affected_count, 8)

    def _summary(
        self,
        fault_type: str,
        rc_node: Dict[str, Any],
        affected_count: int,
        victim_groups: Dict[str, int],
    ) -> str:
        node_type = str(rc_node.get("type", "node"))
        node_id = str(rc_node.get("id", "unknown"))
        fault_label = fault_type.replace("_", " ")
        hosts = victim_groups.get("host", affected_count)
        jobs_estimate = max(hosts, affected_count) * 5 if affected_count else 0
        return (
            f"{fault_label.capitalize()} detected: 1 {node_type} {node_id}, "
            f"{affected_count} hosts, ~{jobs_estimate} jobs potentially affected"
        )
