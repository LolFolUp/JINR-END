"""
features.py — Proto Batch → temporal state → x_dict, via Polars PIT join.

Pipeline (all Polars-native up to the stateful mapper step):
  1. _samples_to_df          : List[pb.Sample] → typed Polars DataFrame
  2. _point_in_time_join     : sort by timestamp DESC + group_by(et, eid, mn).first()
                               → one row per key, holding the temporally LATEST
                               reading (PIT-deduplicates the stream so stale
                               values from earlier Kafka batches never enter the
                               graph snapshot).
  3. .join(_TOPOLOGY_SCHEMA_DF) : inner-join the deduped frame with the static
                               topology schema frame on (entity_type, metric_name)
                               → vectorised resolution of feat_idx; unknown
                               metrics are dropped in a single pass. This is the
                               actual JOIN between telemetry and the topology
                               skeleton, executed by Polars, not by a Python loop.
  4. EntityMapper            : entity_id → node_index — stateful (free-list,
                               eviction), so this single resolution step stays
                               in Python; everything before it is columnar.

Output layout (verified against build_final_node_features):
  temporal_state[nt] shape: [num_nodes, num_features, 4]
  channels: [value, delta_short, delta_long, rolling_var]
  These come DIRECTLY from proto — Go agent already computed them.

Why Polars:
  - Sort + group_by + inner-join run on Arrow memory in a single columnar pass,
    no GIL pressure — replaces what was a per-row Python schema lookup.
  - The PIT semantics ("last-value-wins per key as of now") and the topology
    join are both auditable Polars operations, not opaque Python control flow.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Iterable, List, Optional, Tuple

import polars as pl
import torch

from training_pipeline.config import FEATURE_SCHEMA, NODE_TYPES
from training_pipeline.dataset_generator import build_final_node_features
from snapshot_engine.topology import NODE_COUNTS, EXPECTED_DIMS

logger = logging.getLogger(__name__)

# entity_type values the model accepts (link_edge is excluded by design)
_ACCEPTED_TYPES: frozenset[str] = frozenset({"cpu", "gpu", "ram", "hdd", "switch", "job"})

# Explicit Polars schema for the raw-samples frame — avoids dtype inference
# on every tick.
_SAMPLES_SCHEMA: Dict[str, pl.PolarsDataType] = {
    "entity_type":       pl.Utf8,
    "entity_id":         pl.Utf8,
    "metric_name":       pl.Utf8,
    "timestamp_unix_ns": pl.Int64,
    "value":             pl.Float32,
    "delta_short":       pl.Float32,
    "delta_long":        pl.Float32,
    "rolling_var":       pl.Float32,
    "labels_job_id":     pl.Utf8,
}

# Static topology lookup frame: (entity_type, metric_name) → feat_idx.
# Built once at import; used as the right-hand side of the Polars topology join,
# replacing what used to be a Python dict lookup inside the hot row loop.
# This is the *second half* of the diploma's point-in-time join:
#   1. PIT-dedup the telemetry stream (one row per (et, eid, mn) at the latest ts)
#   2. JOIN with the static topology schema to resolve feat_idx — Polars-native.
_TOPOLOGY_SCHEMA_DF: pl.DataFrame = pl.DataFrame(
    {
        "entity_type": [
            et for et, feats in FEATURE_SCHEMA.items() if et != "rca_context"
            for _ in feats
        ],
        "metric_name": [
            mn for et, feats in FEATURE_SCHEMA.items() if et != "rca_context"
            for mn in feats
        ],
        "feat_idx": [
            i for et, feats in FEATURE_SCHEMA.items() if et != "rca_context"
            for i, _ in enumerate(feats)
        ],
    },
    schema={"entity_type": pl.Utf8, "metric_name": pl.Utf8, "feat_idx": pl.Int32},
)


# ---------------------------------------------------------------------------
# Entity mapper: entity_id string → integer node index
# ---------------------------------------------------------------------------

class EntityMapper:
    """Stateful mapper from proto entity_ids to model node indices.

    Assigns indices incrementally on first seen, capped at NODE_COUNTS.
    Separate registries per category (host, switch, job).

    entity_id formats from Go agent:
      cpu/gpu/ram/hdd: "{hostname}:{component}"  → use hostname
      switch:          "{target}:{port}"           → use target
      job:             "{hostname}:job{id}"        → use job_id label or parsed suffix
    """

    def __init__(self) -> None:
        self._host_reg:   Dict[str, int] = {}
        self._switch_reg: Dict[str, int] = {}
        self._job_reg:    Dict[str, int] = {}
        # Job indices support eviction (reconciliation): a monotonic high-water
        # counter plus a free-list lets finished jobs release their slot without
        # the index collisions that len(registry) would cause after deletions.
        self._job_next: int = 0
        self._job_free: List[int] = []
        self._job_idx_to_key: Dict[int, str] = {}

    def _assign(self, registry: Dict[str, int], key: str, cap: int) -> Optional[int]:
        if key in registry:
            return registry[key]
        idx = len(registry)
        if idx >= cap:
            return None  # overflow, skip
        registry[key] = idx
        return idx

    def _assign_job(self, key: str, cap: int) -> Optional[int]:
        """Assign a job index with free-list reuse (eviction-safe)."""
        if key in self._job_reg:
            return self._job_reg[key]
        idx = self._job_free.pop() if self._job_free else self._job_next
        if idx >= cap:
            return None  # overflow, skip
        if idx == self._job_next:
            self._job_next += 1
        self._job_reg[key] = idx
        self._job_idx_to_key[idx] = key
        return idx

    def resolve(
        self,
        entity_type: str,
        entity_id: str,
        labels: Dict[str, str],
    ) -> Optional[int]:
        """Return integer node index, or None if entity should be dropped."""
        if entity_type not in _ACCEPTED_TYPES:
            return None

        if entity_type in ("cpu", "gpu", "ram", "hdd"):
            hostname = entity_id.split(":")[0]
            return self._assign(self._host_reg, hostname, NODE_COUNTS["cpu"])

        if entity_type == "switch":
            # entity_id like "192.168.1.1:port3" or "leaf-01:port12"
            target = entity_id.split(":")[0]
            return self._assign(self._switch_reg, target, NODE_COUNTS["switch"])

        if entity_type == "job":
            # prefer labels["job_id"], fallback to parsing entity_id suffix
            job_key = labels.get("job_id", "")
            if not job_key:
                m = re.search(r":job(.+)$", entity_id)
                job_key = m.group(1) if m else entity_id
            return self._assign_job(job_key, NODE_COUNTS["job"])

        return None

    def resolve_job_host(self, entity_id: str) -> Optional[int]:
        """Resolve a job's host index from its entity_id "{hostname}:job{id}".

        Maps the hostname through the shared host registry so the host index
        matches the cpu/gpu/ram/hdd nodes on the same machine. Used by the
        reconciliation loop to rebuild (job → executes_on → cpu) edges.
        """
        hostname = entity_id.split(":")[0]
        return self._assign(self._host_reg, hostname, NODE_COUNTS["cpu"])

    def evict_jobs(self, job_indices: Iterable[int]) -> None:
        """Release finished job indices back to the free-list (self-healing).

        Called by the reconciliation loop with job indices no longer present in
        the authoritative placement. Their slots become reusable by future jobs.
        """
        for idx in job_indices:
            key = self._job_idx_to_key.pop(idx, None)
            if key is None:
                continue
            self._job_reg.pop(key, None)
            self._job_free.append(idx)

    def reset(self) -> None:
        """Clear all registries (call between independent inference sessions)."""
        self._host_reg.clear()
        self._switch_reg.clear()
        self._job_reg.clear()
        self._job_next = 0
        self._job_free.clear()
        self._job_idx_to_key.clear()


# ---------------------------------------------------------------------------
# Proto batch → temporal state
# ---------------------------------------------------------------------------

def _make_empty_temporal() -> Dict[str, torch.Tensor]:
    """Allocate zero-filled temporal tensors for all model node types."""
    state: Dict[str, torch.Tensor] = {}
    for nt in NODE_TYPES:
        if nt == "rca_context":
            continue
        n = NODE_COUNTS[nt]
        f = len(FEATURE_SCHEMA[nt])
        state[nt] = torch.zeros(n, f, 4, dtype=torch.float32)
    return state


def _samples_to_df(samples: list) -> pl.DataFrame:
    """Convert proto Samples to a Polars DataFrame for vectorised join.

    Uses column-oriented construction (one list per column) — faster than
    row-by-row dict append because Polars can infer and allocate types once.
    """
    if not samples:
        return pl.DataFrame(schema=_SAMPLES_SCHEMA)

    entity_type_col:       List[str]   = []
    entity_id_col:         List[str]   = []
    metric_name_col:       List[str]   = []
    timestamp_col:         List[int]   = []
    value_col:             List[float] = []
    delta_short_col:       List[float] = []
    delta_long_col:        List[float] = []
    rolling_var_col:       List[float] = []
    labels_job_id_col:     List[str]   = []

    for s in samples:
        entity_type_col.append(s.entity_type)
        entity_id_col.append(s.entity_id)
        metric_name_col.append(s.metric_name)
        timestamp_col.append(s.timestamp_unix_ns)
        value_col.append(float(s.value))
        delta_short_col.append(float(s.delta_short))
        delta_long_col.append(float(s.delta_long))
        rolling_var_col.append(float(s.rolling_var))
        labels_job_id_col.append(dict(s.labels).get("job_id", ""))

    return pl.DataFrame(
        {
            "entity_type":       entity_type_col,
            "entity_id":         entity_id_col,
            "metric_name":       metric_name_col,
            "timestamp_unix_ns": timestamp_col,
            "value":             value_col,
            "delta_short":       delta_short_col,
            "delta_long":        delta_long_col,
            "rolling_var":       rolling_var_col,
            "labels_job_id":     labels_job_id_col,
        },
        schema=_SAMPLES_SCHEMA,
    )


def _point_in_time_join(df: pl.DataFrame) -> pl.DataFrame:
    """Point-in-Time Join via Polars.

    For each unique (entity_type, entity_id, metric_name) triple, keeps exactly
    the row with the maximum timestamp_unix_ns. This eliminates stale readings
    when multiple Kafka batches are consumed within a single inference tick —
    the model always sees the freshest value for every metric.

    Algorithm:
      sort descending by timestamp → group_by keys → take first row per group
      (first after descending sort == row with highest timestamp).

    Complexity: O(N log N) sort + O(N) group scan — dominated by sort.
    For typical tick sizes (< 50k rows) this runs in < 5 ms.
    """
    return (
        df
        .filter(pl.col("entity_type").is_in(list(_ACCEPTED_TYPES)))
        .sort("timestamp_unix_ns", descending=True)
        .group_by(
            ["entity_type", "entity_id", "metric_name"],
            maintain_order=False,
        )
        .first()  # first row per group = highest timestamp after descending sort
    )


def batch_to_temporal_state(
    samples: list,   # List of proto pb.Sample objects
    mapper: EntityMapper,
) -> Dict[str, torch.Tensor]:
    """Convert a list of proto Samples to temporal_state dict via Polars PIT join.

    Returns Dict[node_type, Tensor[N, F, 4]] ready for build_final_node_features().
    Missing metrics stay 0. entity_type="link" is silently dropped.

    Three Polars stages — this is the full "point-in-time join" the diploma
    describes:
      1. _samples_to_df          : proto Samples → typed columnar frame
      2. _point_in_time_join     : sort by ts desc + group_by (et, eid, mn).first()
                                    → one row per key, holding the temporally
                                    latest reading (PIT-dedup of the stream)
      3. .join(_TOPOLOGY_SCHEMA_DF) : inner join with the static schema frame to
                                    resolve feat_idx — drops unknown metrics in
                                    a single vectorised pass. This is the actual
                                    JOIN between telemetry and the topology
                                    skeleton, executed by Polars, not Python.

    Only after the Polars pipeline collapses the data to the minimum
    (deduplicated, schema-matched) set of rows do we hand off to the mapper
    (stateful, must stay in Python) for the entity_id → node_idx resolution.
    """
    state = _make_empty_temporal()

    if not samples:
        return state

    # ── Stage 1: materialise samples into a columnar DataFrame ─────────────
    raw_df = _samples_to_df(samples)

    # ── Stages 2 + 3: PIT dedup, then JOIN with static topology schema ─────
    joined = (
        _point_in_time_join(raw_df)
        .join(_TOPOLOGY_SCHEMA_DF, on=["entity_type", "metric_name"], how="inner")
    )

    n_input  = len(raw_df)
    n_joined = len(joined)
    if n_input != n_joined:
        logger.debug(
            "Polars PIT join: %d samples → %d (et, eid, mn) rows mapped to topology "
            "(dropped %d stale/unknown)",
            n_input, n_joined, n_input - n_joined,
        )

    # ── Stage 4: resolve entity_id → node_idx (stateful) and write tensors ─
    # feat_idx already came from the Polars join — no Python schema lookup left.
    skipped = 0
    for row in joined.iter_rows(named=True):
        et       = row["entity_type"]
        feat_idx = row["feat_idx"]

        labels = {"job_id": row["labels_job_id"]} if row["labels_job_id"] else {}
        node_idx = mapper.resolve(et, row["entity_id"], labels)
        if node_idx is None:
            skipped += 1
            continue

        # Go agent already computed all 4 channels — write directly
        state[et][node_idx, feat_idx, 0] = row["value"]
        state[et][node_idx, feat_idx, 1] = row["delta_short"]
        state[et][node_idx, feat_idx, 2] = row["delta_long"]
        state[et][node_idx, feat_idx, 3] = row["rolling_var"]

    if skipped:
        logger.debug(
            "batch_to_temporal_state: skipped %d rows (node index overflow)", skipped
        )

    return state


def assemble_x_dict(
    samples: list,
    mapper: EntityMapper,
) -> Dict[str, torch.Tensor]:
    """Full pipeline: samples → temporal_state → x_dict.

    Also adds rca_context = ones(1, 1) and validates dimensions.
    """
    state = batch_to_temporal_state(samples, mapper)
    x_dict = build_final_node_features(state)

    # Validate dimensions — hard stop if schema changed
    for nt, expected_dim in EXPECTED_DIMS.items():
        actual = x_dict[nt].shape[1]
        if actual != expected_dim:
            raise ValueError(
                f"[{nt}] dimension mismatch: expected {expected_dim}, got {actual}. "
                "Check FEATURE_SCHEMA alignment with build_final_node_features."
            )

    return x_dict
