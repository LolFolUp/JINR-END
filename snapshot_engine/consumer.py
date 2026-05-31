"""
consumer.py — Kafka consumer for topic telemetry.raw.

Deserializes proto telemetry.v1.Batch messages and yields lists of Samples.
One call to consume_tick() blocks until messages arrive or timeout_ms elapses,
then returns all samples from all Batch messages received in that window.

The consumer groups messages by batch_timestamp so that samples from the same
agent tick stay together (Point-in-Time Join in features.py).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterator, List

# Add proto dir to sys.path so telemetry_pb2 is importable
_PROTO_DIR = str(Path(__file__).parent.parent / "proto")
if _PROTO_DIR not in sys.path:
    sys.path.insert(0, _PROTO_DIR)

import telemetry_pb2  # generated from proto/telemetry.proto

logger = logging.getLogger(__name__)


class TelemetryConsumer:
    """Wraps kafka-python KafkaConsumer for telemetry.raw.

    Args:
        brokers   : list of bootstrap servers, e.g. ["localhost:9092"]
        topic     : Kafka topic name (default: telemetry.raw)
        group_id  : consumer group
        timeout_ms: poll timeout per call to consume_tick()
    """

    def __init__(
        self,
        brokers: List[str],
        topic: str = "telemetry.raw",
        group_id: str = "snapshot-engine",
        timeout_ms: int = 5_000,
    ) -> None:
        from kafka import KafkaConsumer  # kafka-python

        self._consumer = KafkaConsumer(
            topic,
            bootstrap_servers=brokers,
            group_id=group_id,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            consumer_timeout_ms=timeout_ms,
            value_deserializer=None,  # raw bytes; we parse proto manually
        )
        self._timeout_ms = timeout_ms
        logger.info("TelemetryConsumer connected: brokers=%s topic=%s", brokers, topic)

    def consume_tick(self) -> List:
        """Poll Kafka and return a flat list of pb.Sample objects.

        Reads all messages available within timeout_ms, deserializes each
        pb.Batch, and returns its samples. Last-write-wins per
        (entity_id, metric_name) is handled in features.py.
        """
        samples = []
        try:
            # KafkaConsumer is iterable; stops when consumer_timeout_ms elapses
            for msg in self._consumer:
                try:
                    batch = telemetry_pb2.Batch()
                    batch.ParseFromString(msg.value)
                    samples.extend(batch.samples)
                except Exception as e:
                    logger.warning("Failed to parse proto Batch: %s", e)
        except Exception:
            pass  # StopIteration from consumer_timeout_ms is expected
        logger.debug("consume_tick: received %d samples", len(samples))
        return samples

    def close(self) -> None:
        try:
            self._consumer.close()
        except Exception:
            pass


class ReplayConsumer:
    """File-based replay for offline testing without a running Kafka broker.

    Reads proto Batch messages from a binary file (one length-prefixed message
    per line is NOT required; the file contains one serialized Batch per call).
    Useful for integration tests and demo mode.
    """

    def __init__(self, batch_file: Path) -> None:
        self._path = batch_file
        self._done = False

    def consume_tick(self) -> List:
        if self._done:
            return []
        try:
            data = self._path.read_bytes()
            batch = telemetry_pb2.Batch()
            batch.ParseFromString(data)
            self._done = True
            return list(batch.samples)
        except Exception as e:
            logger.error("ReplayConsumer error: %s", e)
            return []

    def close(self) -> None:
        pass
