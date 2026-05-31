"""
publisher.py — writes inference result to artifacts/inference_sample.json.

Format matches what remediation/pipeline.py reads (load_inference()).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).parent.parent / "artifacts" / "inference_sample.json"


def publish(
    result: Dict[str, Any],
    path: Path | None = None,
) -> None:
    """Write inference result as JSON. Overwrites on every tick."""
    p = Path(path) if path else _DEFAULT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(
        "Published: rc=%s[%s] confidence=%.4f → %s",
        result.get("rc_node", {}).get("type"),
        result.get("rc_node", {}).get("id"),
        result.get("confidence", 0.0),
        p,
    )
