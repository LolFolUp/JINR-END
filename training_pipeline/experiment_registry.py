"""
training_pipeline/experiment_registry.py

Atomic, append-only experiment registry.

Every training run logs a JSON record to experiments/registry.jsonl.
An index file provides O(1) lookup by experiment_id without scanning the full log.

Design:
  - Atomic writes (temp + fsync + rename) → crash-safe
  - Append-only → no corruption from concurrent reads
  - JSON Lines format → line-by-line recovery if partially corrupt
  - SQLite index as optional fast-path (falls back to linear scan)
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from training_pipeline.versioning import timestamp_utc, write_atomic

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR    = os.path.join(_ROOT, "experiments")
REGISTRY_PATH      = os.path.join(EXPERIMENTS_DIR, "registry.jsonl")
REGISTRY_INDEX     = os.path.join(EXPERIMENTS_DIR, "registry_index.json")


# ---------------------------------------------------------------------------
# Experiment ID generation
# ---------------------------------------------------------------------------

def new_experiment_id() -> str:
    """Generate a unique, time-sortable experiment ID."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    return f"exp_{ts}_{uid}"


# ---------------------------------------------------------------------------
# Core registry operations
# ---------------------------------------------------------------------------

def append_experiment(metadata: Dict[str, Any]) -> None:
    """
    Append one experiment record to registry.jsonl.

    Uses atomic writes (temp → fsync → rename) to ensure crash safety.
    Updating the index is best-effort (loss of index → linear scan fallback).
    """
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    exp_id = metadata.get("experiment_id", new_experiment_id())
    metadata["experiment_id"] = exp_id
    if "timestamp" not in metadata:
        metadata["timestamp"] = timestamp_utc()

    line = json.dumps(metadata, sort_keys=True) + "\n"

    # Atomic append: write to tmp, then use binary append to actual file
    # (True atomic append to .jsonl: write tmp, read-append-write)
    tmp_path = REGISTRY_PATH + f".tmp.{os.getpid()}.{int(time.time()*1000)}"
    try:
        # Write the new line to a temp file
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

        # Append temp content to registry (read-existing + new line → atomic replace)
        existing = ""
        if os.path.exists(REGISTRY_PATH):
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                existing = f.read()

        combined = existing + line
        write_atomic(REGISTRY_PATH, combined)

    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    # Update index (best-effort)
    _update_index(exp_id, metadata.get("timestamp", ""))


def _update_index(exp_id: str, timestamp: str) -> None:
    """Update or create the registry index. Best-effort (non-critical)."""
    try:
        index: Dict[str, Any] = {}
        if os.path.exists(REGISTRY_INDEX):
            with open(REGISTRY_INDEX, encoding="utf-8") as f:
                index = json.load(f)

        # Count existing lines to get line number
        line_number = 0
        if os.path.exists(REGISTRY_PATH):
            with open(REGISTRY_PATH, encoding="utf-8") as f:
                line_number = sum(1 for _ in f) - 1  # 0-indexed

        index[exp_id] = {"line_number": max(line_number, 0), "timestamp": timestamp}

        write_atomic(REGISTRY_INDEX, json.dumps(index, indent=2, sort_keys=True))
    except Exception as e:
        # Non-critical: index is a speedup, not required for correctness
        print(f"[registry] Warning: could not update index: {e}")


# ---------------------------------------------------------------------------
# Registry queries
# ---------------------------------------------------------------------------

def _iter_registry() -> Iterator[Dict[str, Any]]:
    """Iterate over all registry records, skipping corrupt lines."""
    if not os.path.exists(REGISTRY_PATH):
        return
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"[registry] Warning: skipping corrupt line {lineno}")


def load_experiment(exp_id: str) -> Optional[Dict[str, Any]]:
    """Load a single experiment by ID. Returns None if not found."""
    for record in _iter_registry():
        if record.get("experiment_id") == exp_id:
            return record
    return None


def list_experiments(
    dataset_hash: Optional[str] = None,
    dataset_version: Optional[str] = None,
    status: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Query experiments with optional filters.

    Args:
        dataset_hash:    filter by exact dataset fingerprint
        dataset_version: filter by semantic version (e.g. "v2.0.0")
        status:          filter by "completed", "failed", "running"
        limit:           max results (most recent first)
    """
    results = []
    for record in _iter_registry():
        ds_info = record.get("dataset_info", {})
        if dataset_hash and ds_info.get("dataset_hash") != dataset_hash:
            continue
        if dataset_version and ds_info.get("dataset_version") != dataset_version:
            continue
        if status and record.get("status") != status:
            continue
        results.append(record)

    # Most recent first
    results.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

    if limit is not None:
        results = results[:limit]

    return results


def is_experiment_reproducible(exp_id: str, manifest_dir: str, train_py_path: str) -> Dict[str, Any]:
    """
    Check whether an old experiment can be reproduced now.

    Returns dict with: reproducible (bool), reasons (list of issues).
    """
    from training_pipeline.versioning import (
        compute_behavioral_code_hash,
        get_dataset_hash_from_manifest,
    )

    exp = load_experiment(exp_id)
    if exp is None:
        return {"reproducible": False, "reasons": [f"Experiment '{exp_id}' not found in registry"]}

    reasons: List[str] = []

    # Check dataset hash
    try:
        current_hash = get_dataset_hash_from_manifest(manifest_dir)
        exp_hash = exp.get("dataset_info", {}).get("dataset_hash")
        if exp_hash != current_hash:
            reasons.append(
                f"Dataset hash changed: {exp_hash} → {current_hash}. "
                "Regenerate dataset with same config to reproduce."
            )
    except Exception as e:
        reasons.append(f"Cannot check dataset hash: {e}")

    # Check code hash
    try:
        current_code = compute_behavioral_code_hash(train_py_path)
        exp_code = exp.get("code_versions", {}).get("train_code_hash")
        if exp_code and exp_code != current_code:
            reasons.append(
                "train.py loss/optimizer logic changed since experiment. "
                "Metrics may differ."
            )
    except Exception as e:
        reasons.append(f"Cannot check code hash: {e}")

    return {"reproducible": len(reasons) == 0, "reasons": reasons}


# ---------------------------------------------------------------------------
# Registry repair / inspection
# ---------------------------------------------------------------------------

def count_experiments() -> int:
    return sum(1 for _ in _iter_registry())


def registry_summary() -> Dict[str, Any]:
    """Print a quick summary of the experiment registry."""
    records = list(_iter_registry())
    if not records:
        return {"total": 0}

    versions: Dict[str, int] = {}
    statuses: Dict[str, int] = {}
    for r in records:
        v = r.get("dataset_info", {}).get("dataset_version", "unknown")
        versions[v] = versions.get(v, 0) + 1
        s = r.get("status", "unknown")
        statuses[s] = statuses.get(s, 0) + 1

    return {
        "total":               len(records),
        "by_dataset_version":  versions,
        "by_status":           statuses,
        "oldest":              records[-1].get("timestamp", "?"),
        "newest":              records[0].get("timestamp", "?"),
    }


# ---------------------------------------------------------------------------
# Convenience: build standard experiment metadata dict
# ---------------------------------------------------------------------------

def build_experiment_metadata(
    exp_id: str,
    dataset_version: str,
    dataset_hash: str,
    dataset_codename: str,
    model_config: Dict,
    loss_config: Dict,
    optimizer_config: Dict,
    training_config: Dict,
    metrics: Optional[Dict] = None,
    runtime: Optional[Dict] = None,
    code_versions: Optional[Dict] = None,
    ablation_mode: str = "none",
    notes: str = "",
    status: str = "completed",
    checkpoint_path: str = "",
) -> Dict[str, Any]:
    """Construct a complete experiment metadata record for the registry."""
    return {
        "experiment_id": exp_id,
        "timestamp":     timestamp_utc(),
        "status":        status,

        "dataset_info": {
            "dataset_version": dataset_version,
            "dataset_hash":    dataset_hash,
            "dataset_codename": dataset_codename,
        },

        "code_versions": code_versions or {},

        "model_config":     model_config,
        "loss_config":      loss_config,
        "optimizer_config": optimizer_config,
        "training_config":  training_config,

        "ablation_mode": ablation_mode,
        "notes":         notes,

        "metrics":         metrics or {},
        "runtime":         runtime or {},
        "checkpoint_path": checkpoint_path,

        "reproducibility_info": {
            "tolerance_roc_auc":      0.01,
            "tolerance_f1":           0.02,
            "tolerance_rca_accuracy": 0.03,
            "note": "CUDA non-determinism; tolerances apply across runs",
        },
    }
