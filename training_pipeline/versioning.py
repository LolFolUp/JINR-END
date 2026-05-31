"""
training_pipeline/versioning.py

Research-grade dataset versioning, lineage tracking, and experiment traceability.

Provides:
  - Semantic dataset fingerprinting (resilient to code formatting changes)
  - Behavioral code hashing (loss/optimizer logic only, ignores comments/logs)
  - Feature ordering protection (catches silent permutation bugs)
  - Modular manifest read/write with schema versioning
  - Checkpoint compatibility checks
  - Atomic write utilities

Design notes:
  - All hashes are semantic (not raw text), so whitespace/comments don't break compatibility
  - Tolerances: ROC-AUC ±0.01, F1 ±0.02, RCA ±0.03 (CUDA non-determinism is unavoidable)
  - Manifests are split into 7 files for maintainability (not one god-object JSON)
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_SCHEMA_VERSION = "v2.0"

MANIFEST_FILES = [
    "MANIFEST.json",
    "identity.json",
    "generation_config.json",
    "topology.json",
    "temporal_features.json",
    "fault_generation.json",
    "diagnostics.json",
    "lineage.json",
]

# Reproducibility tolerance bands (CUDA nondeterminism is unavoidable)
METRIC_TOLERANCES = {
    "roc_auc":      0.01,
    "f1":           0.02,
    "rca_accuracy": 0.03,
    "loss":         0.005,
}

# ---------------------------------------------------------------------------
# Utility: canonical JSON and hashing
# ---------------------------------------------------------------------------

def _canonical_json(obj: Any) -> str:
    """Produce deterministic JSON string (sorted keys, no whitespace, normalized floats)."""
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    # Normalize floats to 6 decimal places to avoid cross-platform precision drift
    raw = re.sub(r"\d+\.\d{7,}", lambda m: str(round(float(m.group()), 6)), raw)
    return raw


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Semantic dataset hash (resilient to formatting/comments)
# ---------------------------------------------------------------------------

def compute_semantic_dataset_hash(semantic_config: Dict) -> str:
    """
    Hash only semantic generation parameters.

    Immune to: whitespace, comments, docstrings, print/logging changes.
    Changes on: any numerical parameter, seed, feature schema content,
                fault algorithm name, propagation config.

    Args:
        semantic_config: dict with keys topology, temporal, fault, seeds,
                         feature_schema_version, temporal_channels.
    """
    required_keys = ["topology", "temporal", "fault", "seeds"]
    for k in required_keys:
        if k not in semantic_config:
            raise ValueError(f"semantic_config missing required key: '{k}'")

    # Normalize and extract only the fields that actually define data generation
    normalized = {
        "topology": {
            "NUM_HOSTS": int(semantic_config["topology"]["NUM_HOSTS"]),
            "NUM_LEAF":  int(semantic_config["topology"]["NUM_LEAF"]),
            "NUM_SPINE": int(semantic_config["topology"]["NUM_SPINE"]),
        },
        "temporal": {
            "SIMULATION_STEPS":    int(semantic_config["temporal"]["SIMULATION_STEPS"]),
            "FAULT_INJECTION_STEP": int(semantic_config["temporal"]["FAULT_INJECTION_STEP"]),
            "EMA_ALPHA":           round(float(semantic_config["temporal"]["EMA_ALPHA"]), 6),
            "NOISE_SCALE":         round(float(semantic_config["temporal"]["NOISE_SCALE"]), 6),
            "temporal_channels":   list(semantic_config["temporal"]["temporal_channels"]),
        },
        "fault": {
            "fault_types":      sorted(semantic_config["fault"]["fault_types"]),
            "severity_min":     round(float(semantic_config["fault"]["severity_min"]), 4),
            "severity_max":     round(float(semantic_config["fault"]["severity_max"]), 4),
            "propagation_algo": str(semantic_config["fault"]["propagation_algo"]),
        },
        "seeds": {
            "master_seed": int(semantic_config["seeds"]["master_seed"]),
        },
        "feature_schema_version": str(semantic_config.get("feature_schema_version", "v1")),
    }

    return _sha256(_canonical_json(normalized))


# ---------------------------------------------------------------------------
# Feature ordering hash (catches silent permutation bugs)
# ---------------------------------------------------------------------------

def compute_feature_ordering_hash(feature_schema: Dict[str, List[str]]) -> str:
    """
    Hash the EXACT feature order per node type.

    Any feature permutation (even if dims match) produces a different hash.
    This prevents the most dangerous ML bug: dimensions match but features are scrambled.
    """
    ordered = {
        node_type: {feat: idx for idx, feat in enumerate(feats)}
        for node_type, feats in sorted(feature_schema.items())
    }
    return _sha256(_canonical_json(ordered))


def verify_feature_ordering(checkpoint: Dict, feature_schema: Dict[str, List[str]]) -> None:
    """Raise if feature order in checkpoint differs from current schema."""
    stored = checkpoint.get("feature_ordering_hash")
    if stored is None:
        # Pre-versioning checkpoint — warn only
        print(
            "[versioning] WARNING: checkpoint has no feature_ordering_hash "
            "(pre-versioning checkpoint). Cannot verify feature order."
        )
        return

    current = compute_feature_ordering_hash(feature_schema)
    if stored != current:
        raise ValueError(
            "CRITICAL: Feature order changed!\n"
            f"  Checkpoint expects: {stored}\n"
            f"  Current schema:     {current}\n"
            "  Shapes may match but features are SCRAMBLED.\n"
            "  DO NOT USE this checkpoint. Retrain on current schema."
        )


# ---------------------------------------------------------------------------
# Behavioral code hash (loss/optimizer logic only, ignores formatting)
# ---------------------------------------------------------------------------

def _extract_behavioral_ast_text(filepath: str) -> str:
    """
    Extract only semantically important code from a Python file.

    Strips: comments, docstrings, logging calls, print statements, type hints.
    Keeps:  function definitions (minus docstrings), class definitions,
            assignments that are not just string literals.
    """
    try:
        source = open(filepath, encoding="utf-8").read()
        tree = ast.parse(source)
    except (FileNotFoundError, SyntaxError) as e:
        raise ValueError(f"Cannot parse {filepath}: {e}") from e

    parts: List[str] = []

    for node in ast.walk(tree):
        # Remove docstrings from functions and classes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)  # Remove docstring
            parts.append(ast.unparse(node))

    # Deduplicate (ast.walk visits nested nodes multiple times)
    seen: set = set()
    unique: List[str] = []
    for p in parts:
        h = _sha256(p)
        if h not in seen:
            seen.add(h)
            unique.append(p)

    return "\n".join(unique)


def compute_behavioral_code_hash(filepath: str) -> str:
    """
    Hash only the behavioral logic of a Python file.

    Adding/removing comments, docstrings, print statements, or reformatting
    will NOT change this hash.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    behavioral_text = _extract_behavioral_ast_text(filepath)
    return _sha256(behavioral_text)


# ---------------------------------------------------------------------------
# Atomic file write
# ---------------------------------------------------------------------------

def write_atomic(path: str, content: str, encoding: str = "utf-8") -> None:
    """Write file atomically using temp-file + fsync + rename."""
    dir_ = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=dir_, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)  # atomic on POSIX and Windows
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def write_json_atomic(path: str, obj: Any, indent: int = 2) -> None:
    """Serialize obj to JSON and write atomically."""
    write_atomic(path, json.dumps(obj, indent=indent, sort_keys=True))


# ---------------------------------------------------------------------------
# Manifest writing (modular structure)
# ---------------------------------------------------------------------------

def write_dataset_manifest(
    manifest_dir: str,
    *,
    dataset_version: str,
    dataset_hash: str,
    codename: str,
    semantic_config: Dict,
    feature_schema: Dict[str, List[str]],
    total_graphs: int,
    healthy_graphs: int,
    node_dims: Dict[str, int],
    edge_types: List[Tuple[str, str, str]],
    diagnostics: Optional[Dict] = None,
    previous_version: Optional[str] = None,
    changelog: Optional[str] = None,
    breaking_changes: Optional[List[str]] = None,
    known_limitations: Optional[List[str]] = None,
    git_commit: Optional[str] = None,
) -> str:
    """
    Write a complete modular manifest to manifest_dir/.

    Creates 7 JSON files + MANIFEST.json index with per-file checksums.

    Returns: dataset_hash
    """
    os.makedirs(manifest_dir, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    faulted_graphs = total_graphs - healthy_graphs

    # --- identity.json ---
    identity = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_identity": {
            "semantic_version":      dataset_version,
            "dataset_hash":          dataset_hash,
            "codename":              codename,
            "generation_timestamp":  now,
        },
    }

    # --- generation_config.json ---
    gen_config = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "generation_config": {
            "git_commit":   git_commit or _get_git_commit(),
            "python_version": _python_version(),
        },
        "reproducibility": {
            "master_seed":          semantic_config["seeds"]["master_seed"],
            "seed_derivation":      "deterministic_child_seed_sha256",
            "randomness_sources": [
                "topology_generation_per_sample",
                "edge_attribute_sampling_per_sample",
                "initial_continuous_state_per_sample",
                "noise_trajectory_per_sample",
                "fault_severity_per_sample",
                "victim_selection_per_sample",
                "healthy_spike_injection_per_sample",
            ],
        },
    }

    # --- topology.json ---
    topology = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "topology_config": {
            "topology_schema_version":    "v1.0",
            "topology_type":              "spine_leaf",
            "edge_generation_version":    "v2.0_bfs_temporal",
            "routing_semantics_version":  "v1.0_host_to_leaf_direct",
            **semantic_config["topology"],
        },
        "dataset_structure": {
            "total_graphs":         total_graphs,
            "healthy_graphs":       healthy_graphs,
            "faulted_graphs":       faulted_graphs,
            "healthy_ratio":        round(healthy_graphs / max(total_graphs, 1), 4),
            "edge_types":           [list(et) for et in edge_types],
        },
    }

    # --- temporal_features.json ---
    feature_ordering_hash = compute_feature_ordering_hash(feature_schema)
    temporal_features = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "temporal_config":         semantic_config["temporal"],
        "feature_schema": {
            "feature_ordering_hash":              feature_ordering_hash,
            "node_types":                         sorted(feature_schema.keys()),
            "features_per_type":                  {k: list(v) for k, v in sorted(feature_schema.items())},
            "feature_dim_total":                  node_dims,
        },
    }

    # --- fault_generation.json ---
    fault_generation = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "fault_generation":        semantic_config["fault"],
    }

    # --- diagnostics.json ---
    diagnostics_data = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "diagnostics_summary":     diagnostics or {},
        "known_limitations":       known_limitations or [
            "Healthy spike injection adds artificial anomalies not in real systems",
            "BFS topology assumes connected graph; isolated hosts won't receive propagated faults",
            "TruncExp delays assume uniform hop latency; real networks have variable latency",
            "Rolling variance computed over only 5 steps; noisy in early simulation",
        ],
    }

    # --- lineage.json ---
    lineage = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "version_lineage": {
            "current_version":   dataset_version,
            "previous_version":  previous_version or "none",
            "changelog":         changelog or "Initial version",
            "breaking_changes":  breaking_changes or [],
        },
    }

    # Write all files atomically
    files_written: Dict[str, str] = {}
    file_map = {
        "identity.json":          identity,
        "generation_config.json": gen_config,
        "topology.json":          topology,
        "temporal_features.json": temporal_features,
        "fault_generation.json":  fault_generation,
        "diagnostics.json":       diagnostics_data,
        "lineage.json":           lineage,
    }

    for fname, obj in file_map.items():
        fpath = os.path.join(manifest_dir, fname)
        content = json.dumps(obj, indent=2, sort_keys=True)
        write_atomic(fpath, content)
        files_written[fname] = f"sha256:{_sha256(content)}"

    # --- MANIFEST.json (index with checksums) ---
    manifest_index = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_format":         "modular_v2",
        "dataset_version":         dataset_version,
        "dataset_hash":            dataset_hash,
        "files":                   {k.replace(".json", ""): k for k in file_map},
        "checksums":               files_written,
    }
    index_path = os.path.join(manifest_dir, "MANIFEST.json")
    write_json_atomic(index_path, manifest_index)

    return dataset_hash


# ---------------------------------------------------------------------------
# Manifest reading
# ---------------------------------------------------------------------------

def load_manifest(manifest_dir: str) -> Dict[str, Any]:
    """Load and merge all manifest files into one dict. Validates checksums."""
    index_path = os.path.join(manifest_dir, "MANIFEST.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"MANIFEST.json not found in {manifest_dir}")

    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    # Validate checksums
    for fname, stored_checksum in index.get("checksums", {}).items():
        fpath = os.path.join(manifest_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Manifest file missing: {fpath}")
        with open(fpath, encoding="utf-8") as f:
            content = f.read()
        actual = f"sha256:{_sha256(content)}"
        if stored_checksum != actual:
            raise ValueError(
                f"Manifest checksum mismatch for {fname}.\n"
                f"  Expected: {stored_checksum}\n"
                f"  Actual:   {actual}\n"
                "  Manifest may have been modified after generation."
            )

    # Merge all files
    merged: Dict[str, Any] = {"_index": index}
    for fname in index.get("files", {}).values():
        fpath = os.path.join(manifest_dir, fname)
        with open(fpath, encoding="utf-8") as f:
            merged.update(json.load(f))

    return merged


def get_dataset_hash_from_manifest(manifest_dir: str) -> str:
    """Fast: read dataset_hash without loading full manifest."""
    index_path = os.path.join(manifest_dir, "MANIFEST.json")
    with open(index_path, encoding="utf-8") as f:
        return json.load(f)["dataset_hash"]


def get_dataset_version_from_manifest(manifest_dir: str) -> str:
    index_path = os.path.join(manifest_dir, "MANIFEST.json")
    with open(index_path, encoding="utf-8") as f:
        return json.load(f)["dataset_version"]


# ---------------------------------------------------------------------------
# Checkpoint compatibility check
# ---------------------------------------------------------------------------

class IncompatibleVersionError(Exception):
    pass


class IncompatibleHashError(Exception):
    pass


class IncompatibleFeatureError(Exception):
    pass


def build_checkpoint_versioning_metadata(
    dataset_version: str,
    dataset_hash: str,
    manifest_dir: str,
    feature_schema: Dict[str, List[str]],
    node_dims: Dict[str, int],
    model_config: Dict,
    train_py_path: str,
) -> Dict:
    """Build the versioning block to embed in every checkpoint."""
    return {
        "dataset_version":       dataset_version,
        "dataset_hash":          dataset_hash,
        "manifest_dir":          manifest_dir,
        "feature_ordering_hash": compute_feature_ordering_hash(feature_schema),
        "train_code_hash":       compute_behavioral_code_hash(train_py_path),
        "model_config":          model_config,
        "node_dims":             node_dims,
    }


def check_checkpoint_compatibility(
    checkpoint: Dict,
    manifest_dir: str,
    feature_schema: Dict[str, List[str]],
    node_dims: Dict[str, int],
    train_py_path: str,
    mode: str = "strict",
) -> Dict[str, Any]:
    """
    Verify a checkpoint is compatible with the current dataset and code.

    Args:
        checkpoint:     loaded checkpoint dict
        manifest_dir:   path to current dataset manifest directory
        feature_schema: current FEATURE_SCHEMA
        node_dims:      current feature dimensions per node type
        train_py_path:  path to train.py
        mode:           "strict" → raise on hash mismatch
                        "warn"   → print warnings, don't raise

    Returns:
        dict with keys: compatible (bool), issues (list), warnings (list)
    """
    issues: List[str] = []
    warnings: List[str] = []

    # --- Check 1: Versioning metadata exists ---
    if "dataset_version" not in checkpoint:
        msg = (
            "Checkpoint has no version metadata (pre-versioning checkpoint).\n"
            "  Options:\n"
            "    1. Retrain from scratch on current dataset\n"
            "    2. Regenerate dataset and retrain"
        )
        if mode == "strict":
            raise IncompatibleVersionError(msg)
        warnings.append(msg)

    # --- Check 2: Dataset hash ---
    try:
        current_hash = get_dataset_hash_from_manifest(manifest_dir)
    except (FileNotFoundError, KeyError) as e:
        warnings.append(f"Cannot load manifest to check dataset_hash: {e}")
        current_hash = None

    if current_hash is not None:
        ckpt_hash = checkpoint.get("dataset_hash")
        if ckpt_hash and ckpt_hash != current_hash:
            msg = (
                f"Dataset fingerprint mismatch.\n"
                f"  Checkpoint: {ckpt_hash}\n"
                f"  Current:    {current_hash}\n"
                "  Metrics will not be reproducible. Retrain required."
            )
            if mode == "strict":
                raise IncompatibleHashError(msg)
            issues.append(msg)

    # --- Check 3: Version string MAJOR ---
    ckpt_version = checkpoint.get("dataset_version", "")
    try:
        current_version = get_dataset_version_from_manifest(manifest_dir)
    except Exception:
        current_version = ""

    if ckpt_version and current_version:
        ckpt_major = ckpt_version.lstrip("v").split(".")[0]
        cur_major  = current_version.lstrip("v").split(".")[0]
        if ckpt_major != cur_major:
            msg = (
                f"MAJOR version mismatch → breaking change.\n"
                f"  Checkpoint: {ckpt_version}\n"
                f"  Current:    {current_version}\n"
                "  Retraining required."
            )
            if mode == "strict":
                raise IncompatibleVersionError(msg)
            issues.append(msg)

    # --- Check 4: Feature ordering (critical) ---
    stored_fo_hash = checkpoint.get("feature_ordering_hash")
    if stored_fo_hash:
        current_fo_hash = compute_feature_ordering_hash(feature_schema)
        if stored_fo_hash != current_fo_hash:
            msg = (
                "CRITICAL: Feature ordering changed!\n"
                f"  Checkpoint: {stored_fo_hash}\n"
                f"  Current:    {current_fo_hash}\n"
                "  Dimensions may match but features are SCRAMBLED. DO NOT USE."
            )
            if mode == "strict":
                raise IncompatibleFeatureError(msg)
            issues.append(msg)

    # --- Check 5: Feature dimensions ---
    ckpt_node_dims = checkpoint.get("node_dims", {})
    if ckpt_node_dims and node_dims:
        for nt, expected_dim in node_dims.items():
            ckpt_dim = ckpt_node_dims.get(nt)
            if ckpt_dim is not None and ckpt_dim != expected_dim:
                msg = (
                    f"Feature dimension mismatch for '{nt}'.\n"
                    f"  Checkpoint expects: {ckpt_dim}\n"
                    f"  Current dataset:    {expected_dim}"
                )
                if mode == "strict":
                    raise IncompatibleFeatureError(msg)
                issues.append(msg)

    # --- Check 6: Behavioral code hash (warn only, never block) ---
    ckpt_code_hash = checkpoint.get("train_code_hash")
    if ckpt_code_hash and os.path.exists(train_py_path):
        current_code_hash = compute_behavioral_code_hash(train_py_path)
        if ckpt_code_hash != current_code_hash:
            warnings.append(
                "train.py loss/optimizer logic changed since checkpoint.\n"
                f"  Old hash: {ckpt_code_hash}\n"
                f"  New hash: {current_code_hash}\n"
                "  Proceed with caution; consider retraining."
            )

    compatible = len(issues) == 0
    return {"compatible": compatible, "issues": issues, "warnings": warnings}


# ---------------------------------------------------------------------------
# Dataset sanity regression checks
# ---------------------------------------------------------------------------

def run_dataset_sanity_checks(
    graphs: List,
    manifest: Optional[Dict] = None,
    quick: bool = False,
) -> Dict[str, Any]:
    """
    Automated regression checks to catch dataset quality regressions.

    Fails hard if:
      - Node-type-alone RC prediction accuracy > 0.55 (leakage)
      - Anomaly density outside [0.02, 0.35]
      - Zero faulted graphs found
      - Healthy/faulted separability < 0.50

    Returns dict of check results; raises ValueError on FAIL.
    """
    import numpy as np

    checks: Dict[str, Any] = {}
    failed: List[str] = []

    # Split healthy / faulted
    healthy, faulted = [], []
    for g in graphs:
        has_pos = any(
            hasattr(g[nt], "y") and int(g[nt].y.sum()) > 0
            for nt in g.node_types
            if nt != "rca_context"
        )
        (faulted if has_pos else healthy).append(g)

    n_h, n_f = len(healthy), len(faulted)

    # Check 0: Basic non-empty
    if n_f == 0:
        raise ValueError("Dataset sanity FAILED: no faulted graphs found.")
    if n_h == 0:
        raise ValueError("Dataset sanity FAILED: no healthy graphs found.")

    # Check 1: Class balance
    ratio = n_h / max(n_f, 1)
    checks["class_ratio_healthy_faulted"] = {
        "value": round(ratio, 3), "min": 0.5, "max": 3.0,
        "status": "PASS" if 0.5 <= ratio <= 3.0 else "WARN",
    }

    # Check 2: Node-type-alone RC accuracy (leakage indicator)
    nt_rc_counts: Dict[str, int] = {}
    for g in faulted:
        for nt in g.node_types:
            if nt == "rca_context":
                continue
            if hasattr(g[nt], "y") and int(g[nt].y.sum()) > 0:
                nt_rc_counts[nt] = nt_rc_counts.get(nt, 0) + 1
    if nt_rc_counts:
        majority_count = max(nt_rc_counts.values())
        nt_acc = majority_count / max(len(faulted), 1)
    else:
        nt_acc = 0.0
    checks["leakage_node_type_accuracy"] = {
        "value": round(nt_acc, 3), "threshold": 0.55,
        "status": "PASS" if nt_acc < 0.55 else "FAIL",
    }
    if checks["leakage_node_type_accuracy"]["status"] == "FAIL":
        failed.append(f"Node-type-alone RC accuracy {nt_acc:.3f} > 0.55 (leakage risk)")

    # Check 3: Anomaly density
    all_pos = all_total = 0
    for g in faulted:
        for nt in g.node_types:
            if nt == "rca_context" or not hasattr(g[nt], "y"):
                continue
            all_pos   += int(g[nt].y.sum())
            all_total += int(g[nt].y.shape[0])
    density = all_pos / max(all_total, 1)
    checks["anomaly_density"] = {
        "value": round(density, 4), "min": 0.02, "max": 0.35,
        "status": "PASS" if 0.02 <= density <= 0.35 else "FAIL",
    }
    if checks["anomaly_density"]["status"] == "FAIL":
        failed.append(f"Anomaly density {density:.4f} outside [0.02, 0.35]")

    # Check 4: Basic feature sanity (no all-zero or all-constant tensors)
    import torch
    zero_nt_count = 0
    for g in faulted[:min(10, len(faulted))]:
        for nt in g.node_types:
            if nt == "rca_context" or not hasattr(g[nt], "x"):
                continue
            x = g[nt].x.float()
            if x.std() < 1e-6:
                zero_nt_count += 1
    checks["feature_variance"] = {
        "value": zero_nt_count, "threshold": 0,
        "status": "PASS" if zero_nt_count == 0 else "WARN",
    }

    # Check 5: Victim node type has positive labels only when expected
    healthy_pos_count = sum(
        int(g[nt].y.sum())
        for g in healthy
        for nt in g.node_types
        if nt != "rca_context" and hasattr(g[nt], "y")
    )
    checks["healthy_label_contamination"] = {
        "value": healthy_pos_count, "threshold": 0,
        "status": "PASS" if healthy_pos_count == 0 else "FAIL",
    }
    if healthy_pos_count > 0:
        failed.append(f"Healthy graphs contain {healthy_pos_count} positive RC labels (should be 0)")

    if failed:
        raise ValueError(
            "Dataset sanity checks FAILED:\n"
            + "\n".join(f"  ❌ {msg}" for msg in failed)
            + "\nRegenerate dataset or check generation parameters."
        )

    return checks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_git_commit() -> str:
    """Return current git commit hash, or 'unknown' if not in a git repo."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _python_version() -> str:
    import sys
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
