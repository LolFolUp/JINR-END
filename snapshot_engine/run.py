"""
run.py — snapshot-engine entry point.

Loop: consume 1 tick → assemble x_dict → normalize → infer → publish.

Usage:
    python -m snapshot_engine.run [--brokers localhost:9092] [--topic telemetry.raw]
    python -m snapshot_engine.run --replay path/to/batch.bin
    python -m snapshot_engine.run --demo          # synthetic graphs, no Kafka needed

Env overrides:
    KAFKA_BROKERS   comma-separated brokers
    KAFKA_TOPIC     topic name
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("snapshot_engine.run")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JINR snapshot-engine")
    p.add_argument("--brokers", default=os.getenv("KAFKA_BROKERS", "localhost:9092"))
    p.add_argument("--topic",   default=os.getenv("KAFKA_TOPIC",   "telemetry.raw"))
    p.add_argument("--group",   default="snapshot-engine")
    p.add_argument("--poll-ms", type=int, default=5_000)
    p.add_argument("--output",  default=None, help="Override inference_sample.json path")
    p.add_argument("--replay",  default=None, help="Binary proto Batch file for offline test")
    p.add_argument("--demo",    action="store_true", help="Use synthetic graphs (no Kafka)")
    p.add_argument("--scaler",  default=None, help="Path to scaler_stats.pt")
    return p.parse_args()


def run_once_demo(engine, topology_data, scaler_stats, output_path) -> None:
    """Run one inference tick using a synthetic faulted graph (demo/test mode)."""
    from training_pipeline.dataset_generator import (
        build_final_node_features,
        build_routing_maps,
        inject_fault,
        simulate_healthy_trajectory,
    )
    from snapshot_engine.normalizer import apply_normalization
    from snapshot_engine.publisher import publish

    traj = simulate_healthy_trajectory(seed=0)
    routing = build_routing_maps(seed=None)
    result_fault = inject_fault(traj, "hdd_degradation", 0.8, routing, seed=7)
    x_dict = build_final_node_features(result_fault["temporal_state"])
    x_dict = apply_normalization(x_dict, scaler_stats)

    _, _, _, edge_index_dict, edge_attr_dict = topology_data
    prediction = engine.predict(x_dict, edge_index_dict, edge_attr_dict)
    publish(prediction, path=output_path)

    gt_type = result_fault["root_cause_node_type"]
    gt_id   = result_fault["root_cause_node_id"]
    rc = prediction["rc_node"]
    logger.info(
        "DEMO  GT=%s[%d]  PRED=%s[%d]  conf=%.4f  match=%s",
        gt_type, gt_id, rc["type"], rc["id"],
        prediction["confidence"],
        gt_type == rc["type"] and gt_id == rc["id"],
    )


def main() -> None:
    args = _parse_args()

    # ── Load static components ─────────────────────────────────────────────
    from snapshot_engine.inference   import InferenceEngine
    from snapshot_engine.normalizer  import fit_and_save_scaler, load_scaler_stats
    from snapshot_engine.publisher   import publish
    from snapshot_engine.topology    import topology_singleton

    topology_data = topology_singleton()
    _, _, _, edge_index_dict, edge_attr_dict = topology_data

    engine = InferenceEngine.load()

    scaler_path = Path(args.scaler) if args.scaler else None
    scaler_stats = load_scaler_stats(scaler_path)
    if scaler_stats is None:
        logger.warning("No scaler found — fitting from 100 synthetic healthy graphs (one-time cost).")
        scaler_stats = fit_and_save_scaler(n_healthy_graphs=100, seed=0)

    output_path = Path(args.output) if args.output else None

    # ── Demo mode ──────────────────────────────────────────────────────────
    if args.demo:
        logger.info("Running in DEMO mode (synthetic graph, no Kafka).")
        run_once_demo(engine, topology_data, scaler_stats, output_path)
        return

    # ── Replay mode ────────────────────────────────────────────────────────
    if args.replay:
        from snapshot_engine.consumer  import ReplayConsumer
        from snapshot_engine.features  import EntityMapper, assemble_x_dict
        from snapshot_engine.normalizer import apply_normalization

        consumer = ReplayConsumer(Path(args.replay))
        mapper   = EntityMapper()
        samples  = consumer.consume_tick()
        if not samples:
            logger.warning("Replay file produced no samples.")
            return
        x_dict = assemble_x_dict(samples, mapper)
        x_dict = apply_normalization(x_dict, scaler_stats)
        prediction = engine.predict(x_dict, edge_index_dict, edge_attr_dict)
        publish(prediction, path=output_path)
        logger.info("Replay done. Result: %s", prediction)
        return

    # ── Live Kafka loop ────────────────────────────────────────────────────
    import threading

    from snapshot_engine.consumer       import TelemetryConsumer
    from snapshot_engine.features       import EntityMapper, assemble_x_dict
    from snapshot_engine.normalizer     import apply_normalization
    from snapshot_engine.reconciliation import (
        ReconciliationLoop,
        TelemetryJobPlacementSource,
        TopologyState,
        extract_job_placement,
    )

    consumer = TelemetryConsumer(
        brokers=[b.strip() for b in args.brokers.split(",")],
        topic=args.topic,
        group_id=args.group,
        timeout_ms=args.poll_ms,
    )
    mapper = EntityMapper()

    # Reconciliation: the live topology starts from the static skeleton, then a
    # background worker rebuilds dynamic job edges from observed placement every
    # RECONCILE_INTERVAL_SEC — self-healing the graph against lost job events.
    topo_state       = TopologyState(edge_index_dict, edge_attr_dict)
    placement_source = TelemetryJobPlacementSource()
    reconciler       = ReconciliationLoop(topo_state, placement_source, mapper=mapper)

    stop_event = threading.Event()
    worker = threading.Thread(
        target=reconciler.run_forever,
        args=(stop_event,),
        name="reconciliation-worker",
        daemon=True,
    )
    worker.start()

    tick = 0
    logger.info("Snapshot-engine running. Ctrl+C to stop.")
    try:
        while True:
            samples = consumer.consume_tick()
            if not samples:
                logger.debug("No samples this tick — waiting.")
                continue

            tick += 1
            try:
                # Feed observed job→host placement to the reconciliation source.
                placement_source.observe(extract_job_placement(samples, mapper), tick)

                x_dict = assemble_x_dict(samples, mapper)
                x_dict = apply_normalization(x_dict, scaler_stats)

                # Read the latest reconciled topology (worker may have swapped it).
                ei_now, ea_now = topo_state.current()
                prediction = engine.predict(x_dict, ei_now, ea_now)
                publish(prediction, path=output_path)
            except Exception as exc:
                logger.error("Tick processing failed: %s", exc, exc_info=True)

    except KeyboardInterrupt:
        logger.info("Snapshot-engine stopped.")
    finally:
        stop_event.set()
        worker.join(timeout=5.0)
        consumer.close()


if __name__ == "__main__":
    main()
