"""
reconciliation.py — Topology Reconciliation Loop (Kubernetes-style Full State Sync).

Problem (split-brain):
  Job placement is dynamic — SLURM jobs schedule, finish, and migrate between
  hosts. The live graph's job edges are updated from a telemetry stream, but
  events get lost (network blips, an agent OOM-killed). A finished job then
  lingers in the graph: its (job → executes_on → cpu) edge stays, so the GNN
  keeps propagating influence through a node that no longer exists.

Fix (self-healing):
  A periodic worker performs a Full State Sync every RECONCILE_INTERVAL_SEC:
    1. Pull the authoritative job→host placement (JobPlacementSource).
    2. Rebuild the dynamic edge_index for the 4 job-related edge types.
    3. Atomically swap the new edges into the live TopologyState.
    4. Evict finished jobs from the EntityMapper registry.
  Even if every incremental event is lost, the graph converges to ground truth
  within one interval. Physical edges (cpu↔switch, ram/hdd↔cpu, switch↔switch)
  are hardware-static and left untouched; the same swap pattern applies if a
  CMDB/registry ever reports recabling (see PhysicalTopologySource seam).

Design split:
  - Edge-index math is pure Python (nested lists) → unit-testable without torch.
  - torch is imported lazily, only at the tensor boundary in TopologyState.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple

from training_pipeline.config import EDGE_DIM_DUMMY

logger = logging.getLogger(__name__)

EdgeType = Tuple[str, str, str]

# Default Full State Sync cadence — matches the diploma (5 minutes).
RECONCILE_INTERVAL_SEC: float = 300.0

# The 4 edge types whose existence depends on which jobs are currently alive.
# Everything else (physical links, switch→rca_context) is static.
_JOB_EDGE_TYPES: Tuple[EdgeType, ...] = (
    ("job", "executes_on", "cpu"),
    ("cpu", "rev_executes_on", "job"),
    ("job", "reports_to", "rca_context"),
    ("rca_context", "rev_reports_to_job", "job"),
)


# ---------------------------------------------------------------------------
# Pure-python edge construction (no torch — testable)
# ---------------------------------------------------------------------------

def build_job_edge_lists(
    job_to_host: Dict[int, int],
) -> Dict[EdgeType, List[List[int]]]:
    """Build COO edge lists for the dynamic job edges from a placement snapshot.

    job_to_host: {job_index: host_index} for currently-alive jobs only.
    Returns {edge_type: [[src...], [dst...]]} — exactly the layout produced by
    build_edge_indices() in dataset_generator, but restricted to alive jobs.

    An empty placement yields empty [[], []] edge lists (shape [2, 0] once
    tensorised), which PyG handles as "no messages on this relation".
    """
    jobs  = sorted(job_to_host.keys())
    hosts = [job_to_host[j] for j in jobs]
    zeros = [0] * len(jobs)  # rca_context has a single node at index 0

    return {
        ("job", "executes_on", "cpu"):            [jobs,  hosts],
        ("cpu", "rev_executes_on", "job"):        [hosts, jobs],
        ("job", "reports_to", "rca_context"):     [jobs,  zeros],
        ("rca_context", "rev_reports_to_job", "job"): [zeros, jobs],
    }


# ---------------------------------------------------------------------------
# Live topology state — atomically swappable, thread-safe
# ---------------------------------------------------------------------------

class TopologyState:
    """Holds the current edge_index_dict / edge_attr_dict for inference.

    The reconciliation worker rebuilds job edges and swaps them in under a lock;
    the inference loop reads a consistent snapshot via current(). Physical and
    switch-context edges are never mutated here.
    """

    def __init__(
        self,
        edge_index_dict: Dict[EdgeType, Any],
        edge_attr_dict: Dict[EdgeType, Any],
    ) -> None:
        self._ei = dict(edge_index_dict)
        self._ea = dict(edge_attr_dict)
        self._lock = threading.Lock()
        # job_to_host reflects the placement currently encoded in the edges.
        self._job_to_host: Dict[int, int] = {}

    @property
    def job_to_host(self) -> Dict[int, int]:
        with self._lock:
            return dict(self._job_to_host)

    def current(self) -> Tuple[Dict[EdgeType, Any], Dict[EdgeType, Any]]:
        """Return a consistent (edge_index_dict, edge_attr_dict) snapshot."""
        with self._lock:
            return dict(self._ei), dict(self._ea)

    def update_job_edges(self, job_to_host: Dict[int, int]) -> None:
        """Rebuild the 4 job edge types from placement and swap them in.

        torch is imported lazily so this module stays importable (and the
        non-tensor logic stays testable) in environments without torch.
        """
        import torch  # local import: only needed at the tensor boundary

        lists = build_job_edge_lists(job_to_host)
        new_ei: Dict[EdgeType, Any] = {}
        new_ea: Dict[EdgeType, Any] = {}
        for et, (src, dst) in lists.items():
            new_ei[et] = torch.tensor([src, dst], dtype=torch.long)
            # Logical/context edges carry dummy attributes of dim EDGE_DIM_DUMMY,
            # filled with ones — identical to build_edge_attr().
            new_ea[et] = torch.ones(len(src), EDGE_DIM_DUMMY, dtype=torch.float32)

        with self._lock:
            for et in _JOB_EDGE_TYPES:
                if et in new_ei:
                    self._ei[et] = new_ei[et]
                    self._ea[et] = new_ea[et]
            self._job_to_host = dict(job_to_host)


# ---------------------------------------------------------------------------
# Authoritative placement sources
# ---------------------------------------------------------------------------

class JobPlacementSource(Protocol):
    """Authoritative source of current job→host placement."""

    def snapshot(self) -> Dict[int, int]:
        """Return {job_index: host_index} for currently-alive jobs."""
        ...


class TelemetryJobPlacementSource:
    """Derives placement from observed telemetry (default, verifiable source).

    A job is "alive" while it keeps emitting metrics. observe() records the
    latest tick each job was seen on; snapshot() returns only jobs seen within
    ttl_ticks of the most recent tick and evicts the rest. This is the
    telemetry-driven self-healing signal.

    Thread-safe: observe() runs on the main inference loop, snapshot() on the
    reconciliation worker thread.

    Seam: a SlurmJobPlacementSource implementing the same snapshot() would query
    squeue / the SLURM REST API instead of inferring liveness from telemetry.
    """

    def __init__(self, ttl_ticks: int = 3) -> None:
        if ttl_ticks < 1:
            raise ValueError("ttl_ticks must be >= 1")
        self._ttl = ttl_ticks
        self._host:      Dict[int, int] = {}   # job_idx -> host_idx
        self._last_seen: Dict[int, int] = {}   # job_idx -> tick
        self._latest_tick = -1
        self._lock = threading.Lock()

    def observe(self, pairs: Iterable[Tuple[int, int]], tick: int) -> None:
        """Record (job_index, host_index) pairs seen at the given tick."""
        with self._lock:
            self._latest_tick = max(self._latest_tick, tick)
            for job_idx, host_idx in pairs:
                self._host[job_idx] = host_idx
                self._last_seen[job_idx] = tick

    def has_data(self) -> bool:
        """True once any telemetry has been observed.

        Distinguishes "no jobs running" (legit empty) from "no data yet"
        (startup) so reconcile() never wipes the initial edges before the first
        telemetry tick arrives.
        """
        with self._lock:
            return self._latest_tick >= 0

    def snapshot(self) -> Dict[int, int]:
        """Alive jobs = seen within ttl_ticks of the latest tick. Evicts stale."""
        with self._lock:
            if self._latest_tick < 0:
                return {}
            cutoff = self._latest_tick - self._ttl
            alive: Dict[int, int] = {}
            stale: List[int] = []
            for job_idx, seen in self._last_seen.items():
                if seen >= cutoff:
                    alive[job_idx] = self._host[job_idx]
                else:
                    stale.append(job_idx)
            for job_idx in stale:
                self._host.pop(job_idx, None)
                self._last_seen.pop(job_idx, None)
            return alive


# ---------------------------------------------------------------------------
# Reconciliation loop
# ---------------------------------------------------------------------------

@dataclass
class ReconcileReport:
    n_alive_jobs: int
    n_added:      int
    n_removed:    int
    duration_ms:  float


class ReconciliationLoop:
    """Periodic Full State Sync between authoritative placement and the graph.

    Use either:
      - maybe_reconcile() called once per inference tick (time-gated), or
      - run_forever(stop_event) on a dedicated background worker thread
        (the "отдельный worker" from the diploma).
    """

    def __init__(
        self,
        topo_state: TopologyState,
        placement_source: JobPlacementSource,
        interval_sec: float = RECONCILE_INTERVAL_SEC,
        mapper: Optional[Any] = None,           # EntityMapper, optional
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._topo   = topo_state
        self._source = placement_source
        self._interval = interval_sec
        self._mapper = mapper
        self._clock  = clock
        self._last_run: Optional[float] = None

    def reconcile(self) -> Optional[ReconcileReport]:
        """Force one Full State Sync. Returns a diff report, or None if skipped.

        Skipped (None) when the source has no data yet AND it reports empty —
        this avoids wiping the initial edges at startup before any telemetry
        arrives. _last_run is updated only on a real sync, so maybe_reconcile()
        keeps retrying until the source becomes ready.
        """
        t0 = self._clock()
        placement = self._source.snapshot()

        # A truly-empty source that has never seen data → not ready, keep edges.
        source_ready = getattr(self._source, "has_data", lambda: True)()
        if not placement and not source_ready:
            return None

        prev_jobs = set(self._topo.job_to_host.keys())
        new_jobs  = set(placement.keys())
        added   = new_jobs - prev_jobs
        removed = prev_jobs - new_jobs

        # Rebuild + swap the dynamic job edges.
        self._topo.update_job_edges(placement)

        # Self-heal the mapper registry so finished jobs free their indices.
        if self._mapper is not None and removed and hasattr(self._mapper, "evict_jobs"):
            self._mapper.evict_jobs(removed)

        self._last_run = t0
        report = ReconcileReport(
            n_alive_jobs=len(new_jobs),
            n_added=len(added),
            n_removed=len(removed),
            duration_ms=round((self._clock() - t0) * 1000.0, 3),
        )
        if added or removed:
            logger.info(
                "Reconcile: %d alive jobs (+%d / -%d) in %.1f ms",
                report.n_alive_jobs, report.n_added, report.n_removed, report.duration_ms,
            )
        return report

    def maybe_reconcile(self) -> Optional[ReconcileReport]:
        """Reconcile only if interval has elapsed since the last successful run."""
        now = self._clock()
        if self._last_run is None or (now - self._last_run) >= self._interval:
            return self.reconcile()
        return None

    def run_forever(
        self,
        stop_event: threading.Event,
        poll_sec: float = 1.0,
    ) -> None:
        """Background-worker loop. Reconciles every interval until stop_event.

        Sleeps in small poll_sec slices so shutdown is responsive instead of
        blocking on a full interval-length sleep.
        """
        logger.info("Reconciliation worker started (interval=%.0fs).", self._interval)
        # maybe_reconcile() fires on the first poll (last_run is None) and keeps
        # retrying every poll_sec until the source has data — so the first real
        # sync lands ~immediately after the first telemetry tick, then settles
        # into the interval cadence. No unconditional empty sync at startup.
        while not stop_event.wait(poll_sec):
            self.maybe_reconcile()
        logger.info("Reconciliation worker stopped.")


# ---------------------------------------------------------------------------
# Helper: extract job→host pairs from a telemetry tick
# ---------------------------------------------------------------------------

def extract_job_placement(samples: list, mapper: Any) -> List[Tuple[int, int]]:
    """Pull (job_index, host_index) pairs from this tick's proto samples.

    A job sample's entity_id is "{hostname}:job{id}"; the hostname resolves to a
    host index through the same registry the cpu/gpu/etc. nodes use. Requires the
    mapper to expose resolve() and resolve_job_host().
    """
    pairs: List[Tuple[int, int]] = []
    seen_jobs: set = set()
    for s in samples:
        if s.entity_type != "job":
            continue
        labels = dict(s.labels)
        job_idx = mapper.resolve("job", s.entity_id, labels)
        if job_idx is None or job_idx in seen_jobs:
            continue
        host_idx = mapper.resolve_job_host(s.entity_id)
        if host_idx is None:
            continue
        pairs.append((job_idx, host_idx))
        seen_jobs.add(job_idx)
    return pairs
