"""Historical incident tickets used as an additional RAG signal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class HistoryTicket:
    ticket_id: str
    fault_type: str
    rc_node: str
    resolution: str
    resolution_time_minutes: int
    root_cause_confirmed: bool

    def as_dict(self) -> Dict[str, object]:
        return {
            "ticket_id": self.ticket_id,
            "fault_type": self.fault_type,
            "rc_node": self.rc_node,
            "resolution": self.resolution,
            "resolution_time_minutes": self.resolution_time_minutes,
            "root_cause_confirmed": self.root_cause_confirmed,
        }


HISTORY_TICKETS: List[HistoryTicket] = [
    HistoryTicket(
        "INC-2025-0847",
        "network_congestion",
        "switch",
        "Applied QoS rate limiting on ports 12-16",
        8,
        True,
    ),
    HistoryTicket(
        "INC-2025-0912",
        "network_congestion",
        "switch",
        "Identified top talker MPI job and drained affected rack",
        11,
        True,
    ),
    HistoryTicket(
        "INC-2026-0103",
        "network_congestion",
        "switch",
        "Restarted sFlow collector and rebalanced bulk transfer jobs",
        14,
        True,
    ),
    HistoryTicket(
        "INC-2025-0631",
        "hdd_degradation",
        "hdd",
        "Migrated datasets from degraded OST and scheduled disk replacement",
        42,
        True,
    ),
    HistoryTicket(
        "INC-2025-0718",
        "hdd_degradation",
        "hdd",
        "SMART pending sectors confirmed; replaced disk during hot-swap window",
        55,
        True,
    ),
    HistoryTicket(
        "INC-2026-0044",
        "hdd_degradation",
        "hdd",
        "Throttled rebuild and moved checkpoint directory to healthy OST",
        37,
        True,
    ),
    HistoryTicket(
        "INC-2025-1005",
        "ram_leak",
        "ram",
        "Checkpointed leaking simulation and migrated remaining job steps",
        18,
        True,
    ),
    HistoryTicket(
        "INC-2026-0021",
        "ram_leak",
        "ram",
        "Confirmed OOM killer events, notified owner, and drained node",
        16,
        True,
    ),
    HistoryTicket(
        "INC-2026-0188",
        "ram_leak",
        "ram",
        "Restarted service under cgroup memory cap and scheduled memtest",
        24,
        True,
    ),
]


class HistoryTicketsStore:
    """Retrieve confirmed historical tickets by matching fault type."""

    def __init__(self, tickets: List[HistoryTicket] | None = None) -> None:
        self.tickets = tickets or HISTORY_TICKETS

    def find_similar(self, fault_type: str, limit: int = 3) -> List[Dict[str, object]]:
        matches = [
            ticket
            for ticket in self.tickets
            if ticket.fault_type == fault_type and ticket.root_cause_confirmed
        ]
        matches.sort(key=lambda ticket: ticket.resolution_time_minutes)
        return [ticket.as_dict() for ticket in matches[:limit]]
