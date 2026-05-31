"""
e2e_simulator/mock_producer.py
──────────────────────────────
Publishes synthetic telemetry.v1.Batch proto messages to Kafka topic
telemetry.raw — the same format as the Go edge-agent.

Key contract:
  • Values are in [0, 1]  (pre-scaled, matching dataset_generator output)
  • entity_ids are sent in ASCENDING node-index order so that
    EntityMapper in features.py assigns the correct model indices
    (e.g., "sim-host-000" → index 0, "sim-host-001" → index 1, …)
  • Each tick publishes one Batch per node_type (7 batches) to keep
    individual messages small and respect Kafka's default max.message.bytes

Fault scenario:
  Uses inject_fault(fault_type, severity, seed) from training_pipeline
  so the injected signal is identical to training distribution.

Usage:
  python -m e2e_simulator.mock_producer [--ticks 5] [--fault hdd_degradation]
  python -m e2e_simulator.mock_producer --healthy   # no fault, for baseline test
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Proto
sys.path.insert(0, str(Path(__file__).parent.parent / "proto"))
import telemetry_pb2  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2e_simulator.mock_producer")

# ── mapping: model_index → entity_id string ──────────────────────────────────

def _entity_id(node_type: str, node_idx: int) -> str:
    """Produce an entity_id that EntityMapper will map back to node_idx.

    All samples for a given node_type MUST be sent in ascending node_idx
    order within the same batch so the incremental registry assigns correct
    indices (EntityMapper.resolve assigns idx = len(registry) on first seen).
    """
    if node_type in ("cpu", "gpu", "ram", "hdd"):
        return f"sim-host-{node_idx:03d}:{node_type}0"
    if node_type == "switch":
        return f"sim-switch-{node_idx}:port0"
    if node_type == "job":
        return f"sim-host-{node_idx % 100:03d}:job{node_idx}"
    return f"sim-{node_type}-{node_idx}"


def _labels(node_type: str, node_idx: int) -> dict:
    if node_type == "job":
        return {"job_id": str(node_idx)}
    return {}


# ── temporal_state → list[pb.Sample] ─────────────────────────────────────────

def temporal_to_samples(
    temporal_state: dict,
    node_id: str = "sim-node",
) -> list:
    """Convert temporal_state[node_type][N, F, 4] → flat list of pb.Sample.

    Samples are ordered: for each node_type, node indices 0..N-1 in order,
    then each metric in FEATURE_SCHEMA order.
    """
    from training_pipeline.config import FEATURE_SCHEMA, NODE_TYPES
    from training_pipeline.dataset_generator import _classify_features

    samples = []
    ts_ns   = int(time.time() * 1e9)

    for nt in NODE_TYPES:
        if nt == "rca_context":
            continue
        t      = temporal_state[nt]          # [N, F, 4]
        n_nodes, n_feats, _ = t.shape
        cont_idx, cat_idx = _classify_features(nt)
        schema = FEATURE_SCHEMA[nt]
        cat_set = set(cat_idx)

        for node_idx in range(n_nodes):   # ascending order — EntityMapper contract
            eid    = _entity_id(nt, node_idx)
            labels = _labels(nt, node_idx)

            for feat_pos, metric_name in enumerate(schema):
                is_cat   = feat_pos in cat_set
                val      = float(t[node_idx, feat_pos, 0])
                ds       = float(t[node_idx, feat_pos, 1]) if not is_cat else 0.0
                dl       = float(t[node_idx, feat_pos, 2]) if not is_cat else 0.0
                rv       = float(t[node_idx, feat_pos, 3]) if not is_cat else 0.0

                s = telemetry_pb2.Sample(
                    timestamp_unix_ns=ts_ns,
                    node_id=node_id,
                    source_type="e2e_sim",
                    entity_type=nt,
                    entity_id=eid,
                    metric_name=metric_name,
                    unit="ratio",          # all values are [0,1] ratios
                    is_categorical=is_cat,
                    value=val,
                    delta_short=ds,
                    delta_long=dl,
                    rolling_var=rv,
                )
                s.labels.update(labels)
                samples.append(s)

    return samples


def build_batch(samples: list, node_id: str = "sim-node") -> bytes:
    batch = telemetry_pb2.Batch(
        node_id=node_id,
        batch_timestamp_unix_ns=int(time.time() * 1e9),
        samples=samples,
    )
    return batch.SerializeToString()


def temporal_to_batches_per_type(
    temporal_state: dict,
    node_id: str = "sim-node",
    max_jobs: int = 50,
) -> dict[str, bytes]:
    """Split temporal_state into one serialized Batch per node_type.

    Keeps individual messages small (< 200 KB each after gzip) so they
    fit within Kafka's default max.message.bytes = 1 MB.
    job type is capped at max_jobs nodes to avoid oversized messages.
    """
    from training_pipeline.config import FEATURE_SCHEMA, NODE_TYPES
    from training_pipeline.dataset_generator import _classify_features

    ts_ns   = int(time.time() * 1e9)
    batches = {}

    for nt in NODE_TYPES:
        if nt == "rca_context":
            continue
        t      = temporal_state[nt]          # [N, F, 4]
        n_nodes = t.shape[0]
        if nt == "job":
            n_nodes = min(n_nodes, max_jobs)  # cap for message-size safety

        cont_idx, cat_idx = _classify_features(nt)
        schema  = FEATURE_SCHEMA[nt]
        cat_set = set(cat_idx)
        samples = []

        for node_idx in range(n_nodes):       # ascending → EntityMapper contract
            eid    = _entity_id(nt, node_idx)
            labels = _labels(nt, node_idx)

            for feat_pos, metric_name in enumerate(schema):
                is_cat = feat_pos in cat_set
                s = telemetry_pb2.Sample(
                    timestamp_unix_ns=ts_ns,
                    node_id=node_id,
                    source_type="e2e_sim",
                    entity_type=nt,
                    entity_id=eid,
                    metric_name=metric_name,
                    unit="ratio",
                    is_categorical=is_cat,
                    value=float(t[node_idx, feat_pos, 0]),
                    delta_short=float(t[node_idx, feat_pos, 1]) if not is_cat else 0.0,
                    delta_long=float(t[node_idx, feat_pos, 2]) if not is_cat else 0.0,
                    rolling_var=float(t[node_idx, feat_pos, 3]) if not is_cat else 0.0,
                )
                s.labels.update(labels)
                samples.append(s)

        batch = telemetry_pb2.Batch(
            node_id=node_id,
            batch_timestamp_unix_ns=ts_ns,
            samples=samples,
        )
        batches[nt] = batch.SerializeToString()

    return batches


# ── Kafka producer ────────────────────────────────────────────────────────────

def publish(
    brokers: list[str],
    topic: str,
    n_ticks: int,
    fault_type: str | None,
    fault_seed: int,
    tick_interval: float,
    node_id: str,
) -> None:
    from kafka import KafkaProducer
    from training_pipeline.dataset_generator import (
        build_routing_maps,
        inject_fault,
        simulate_healthy_trajectory,
    )

    # Generate temporal state once (same for all ticks in demo)
    logger.info("Generating synthetic fault scenario: %s …", fault_type or "healthy")
    traj    = simulate_healthy_trajectory(seed=42)
    routing = build_routing_maps(seed=None)

    if fault_type:
        result = inject_fault(traj, fault_type, severity=0.8, routing_maps=routing, seed=fault_seed)
        state  = result["temporal_state"]
        logger.info(
            "Fault injected: RC=%s[%d]",
            result["root_cause_node_type"],
            result["root_cause_node_id"],
        )
    else:
        state = traj

    samples = temporal_to_samples(state, node_id=node_id)
    logger.info("Batch size: %d samples", len(samples))

    producer = KafkaProducer(
        bootstrap_servers=brokers,
        value_serializer=None,    # send raw bytes
        compression_type="gzip",  # reduces message size ~5-10x
        acks="all",
        retries=5,
        request_timeout_ms=15_000,
        max_request_size=2 * 1024 * 1024,  # 2 MB
    )

    # Build per-node-type batches once (reused across ticks)
    batches_per_type = temporal_to_batches_per_type(state, node_id=node_id)
    total_bytes = sum(len(v) for v in batches_per_type.values())
    logger.info(
        "Sending %d messages/tick (one per node_type), total uncompressed ~%d KB",
        len(batches_per_type), total_bytes // 1024,
    )

    for tick in range(1, n_ticks + 1):
        futures = []
        for nt, payload in batches_per_type.items():
            future = producer.send(topic, value=payload)
            futures.append((nt, future, payload))

        for nt, future, payload in futures:
            try:
                meta = future.get(timeout=15)
                logger.info(
                    "Tick %d/%d [%s] → partition=%d offset=%d  (%d bytes raw)",
                    tick, n_ticks, nt, meta.partition, meta.offset, len(payload),
                )
            except Exception as exc:
                logger.error("Tick %d [%s] send failed: %s", tick, nt, exc)

        if tick < n_ticks:
            time.sleep(tick_interval)

    producer.flush()
    producer.close()
    logger.info("Producer done. Published %d ticks × %d messages.", n_ticks, len(batches_per_type))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Publish synthetic telemetry to Kafka")
    p.add_argument("--brokers",   default="localhost:9092")
    p.add_argument("--topic",     default="telemetry.raw")
    p.add_argument("--ticks",     type=int, default=5)
    p.add_argument("--fault",     default="hdd_degradation",
                   choices=["hdd_degradation", "network_congestion", "ram_leak", "none"])
    p.add_argument("--fault-seed", type=int, default=7)
    p.add_argument("--interval",  type=float, default=2.0, help="Seconds between ticks")
    p.add_argument("--node-id",   default="sim-node-01")
    p.add_argument("--healthy",   action="store_true", help="Send healthy data (no fault)")
    args = p.parse_args()

    fault = None if (args.healthy or args.fault == "none") else args.fault

    publish(
        brokers=[b.strip() for b in args.brokers.split(",")],
        topic=args.topic,
        n_ticks=args.ticks,
        fault_type=fault,
        fault_seed=args.fault_seed,
        tick_interval=args.interval,
        node_id=args.node_id,
    )


if __name__ == "__main__":
    main()
