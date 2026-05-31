"""
replay_buffer.py — Thread-safe Experience Replay Buffer.

Stores engineer-labeled incidents (confirm / reject) as append-only JSONL.
Each record is one JSON object on a single line — survives crashes without
data loss, trivially readable by any tool.

Typical flow:
  1. GNN inference publishes result  → artifacts/inference_sample.json
  2. Engineer sees Grafana dashboard → clicks Confirm or Reject
  3. Grafana Action calls POST /feedback (grafana_api.py)
  4. ReplayBuffer.save() appends a LabeledIncident to replay_buffer.jsonl
  5. Offline training job calls GET /feedback/export → downloads JSONL
     and passes the verdicts into the batch fine-tuning pipeline.

Why JSONL:
  - Append-only writes hold a lock for < 1 ms — no write contention.
  - Any tool (jq, pandas, polars) reads it without a schema migration.
  - Records survive a mid-write crash (only the last line is corrupt).
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).parent.parent / "artifacts" / "replay_buffer.jsonl"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class FeedbackVerdict(str, Enum):
    CONFIRM = "confirm"   # GNN diagnosis was correct
    REJECT  = "reject"    # GNN was wrong; engineer may provide correction


class FeedbackRequest(BaseModel):
    """Body of POST /feedback — sent by Grafana Action or operator script."""

    # Optional: the server derives the canonical incident_id
    # ("graph_{graph_id}_{fault_type}") from the current inference when omitted.
    # A Grafana node-click sends the node id here, which we keep only as a
    # correlation hint — the canonical id always wins for the stored record.
    incident_id:      Optional[str]             = Field(None, description="Correlation hint; server derives canonical id if omitted")
    verdict:          FeedbackVerdict
    # Optional correction when verdict == REJECT
    correct_rc_type:  Optional[str]             = Field(None, description="True root-cause node type (cpu/gpu/ram/hdd/switch/job)")
    correct_rc_id:    Optional[int]             = Field(None, description="True root-cause node index")
    comment:          str                       = Field("", description="Free-text note from operator")


class LabeledIncident(BaseModel):
    """One record persisted in the replay buffer."""

    # Identity
    incident_id:          str
    timestamp_utc:        str       # ISO-8601

    # Engineer decision
    verdict:              FeedbackVerdict
    comment:              str

    # GNN prediction snapshot (copied from inference_sample.json at feedback time)
    predicted_rc_type:    str
    predicted_rc_id:      int
    predicted_fault_type: str
    confidence:           float
    graph_id:             Optional[int]         = None

    # Engineer correction (populated only for REJECT)
    correct_rc_type:      Optional[str]         = None
    correct_rc_id:        Optional[int]         = None

    # Derived: was prediction correct (used as training signal)
    is_correct_prediction: bool


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Append-only JSONL store for engineer-labeled incidents.

    Thread-safe: a single threading.Lock serialises all writes so that
    concurrent FastAPI requests never interleave partial JSON lines.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path  = Path(path) if path else _DEFAULT_PATH
        self._lock  = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── Write ──────────────────────────────────────────────────────────────

    def save(self, incident: LabeledIncident) -> None:
        """Append one labeled incident. Holds lock for a single .write() call."""
        line = incident.model_dump_json() + "\n"
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)
        logger.info(
            "ReplayBuffer saved: incident=%s verdict=%s correct=%s",
            incident.incident_id,
            incident.verdict.value,
            incident.is_correct_prediction,
        )

    # ── Read ───────────────────────────────────────────────────────────────

    def load_all(self) -> List[LabeledIncident]:
        """Return all records. Skips corrupt lines with a warning."""
        if not self._path.exists():
            return []
        incidents: List[LabeledIncident] = []
        with open(self._path, encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    incidents.append(LabeledIncident.model_validate_json(raw))
                except Exception as exc:
                    logger.warning("replay_buffer: corrupt line %d — %s", lineno, exc)
        return incidents

    # ── Statistics ─────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """Aggregate counts — cheap O(N) scan of the JSONL file."""
        incidents = self.load_all()
        if not incidents:
            return {
                "total": 0,
                "confirmed": 0,
                "rejected": 0,
                "accuracy_pct": None,
                "by_fault_type": {},
            }

        confirmed = sum(1 for i in incidents if i.verdict == FeedbackVerdict.CONFIRM)
        rejected  = len(incidents) - confirmed

        by_fault: Dict[str, Dict[str, int]] = {}
        for inc in incidents:
            ft = inc.predicted_fault_type
            entry = by_fault.setdefault(ft, {"confirm": 0, "reject": 0})
            entry[inc.verdict.value] += 1

        return {
            "total":        len(incidents),
            "confirmed":    confirmed,
            "rejected":     rejected,
            "accuracy_pct": round(confirmed / len(incidents) * 100, 1),
            "by_fault_type": by_fault,
        }

    # ── Export for training ─────────────────────────────────────────────────

    def export_for_training(self) -> List[Dict]:
        """Return records in the format expected by the training pipeline.

        Each dict contains:
          - graph_id            : int   — matches dataset/raw/data_{id}.pt
          - verdict             : str   — "confirm" | "reject"
          - is_correct          : bool
          - correct_rc_type     : str | None
          - correct_rc_id       : int | None
          - predicted_fault_type: str
          - timestamp_utc       : str

        Consumers (batch fine-tuning script) load the matching .pt graph,
        relabel the RC node if verdict == "reject", and add to the training set.
        Confirmed incidents reinforce the existing labels.
        """
        return [
            {
                "graph_id":             inc.graph_id,
                "verdict":              inc.verdict.value,
                "is_correct":           inc.is_correct_prediction,
                "predicted_rc_type":    inc.predicted_rc_type,
                "predicted_rc_id":      inc.predicted_rc_id,
                "correct_rc_type":      inc.correct_rc_type,
                "correct_rc_id":        inc.correct_rc_id,
                "predicted_fault_type": inc.predicted_fault_type,
                "confidence":           inc.confidence,
                "incident_id":          inc.incident_id,
                "timestamp_utc":        inc.timestamp_utc,
            }
            for inc in self.load_all()
        ]


# ---------------------------------------------------------------------------
# Module-level singleton (shared across FastAPI request handlers)
# ---------------------------------------------------------------------------

_buffer: Optional[ReplayBuffer] = None


def get_buffer() -> ReplayBuffer:
    """FastAPI dependency — returns the module-level singleton."""
    global _buffer
    if _buffer is None:
        _buffer = ReplayBuffer()
    return _buffer
