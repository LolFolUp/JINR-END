"""SOP knowledge base for HPC cluster fault remediation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SOPChunk:
    """Single retrievable SOP text chunk."""

    chunk_id: str
    fault_type: str
    title: str
    text: str
    tags: List[str]


NETWORK_CONGESTION_SOP = """
SOP: Network Congestion on Leaf Switch
======================================
Scope: Interconnect saturation, ECN marks, job-to-job latency spikes.

1. CHECK_METRICS — Switch port utilization
   - Poll SNMP/ifHCInOctets and ifHCOutOctets per uplink and server-facing port.
   - Threshold: sustained >85% utilization for 5 minutes triggers escalation.
   - Correlate with TOR buffer drops and PFC pause frames.

2. CHECK_METRICS — Identify top traffic sources
   - Use sFlow/NetFlow on spine; rank talkers by bytes/sec and flow count.
   - Map flows to scheduler job IDs via host interface counters.

3. APPLY_QOS — Rate-limit and prioritize classes
   - Apply policy "rate_limit_top_talkers" on offending ports.
   - Preserve MPI control traffic; cap bulk data movers to 70% line rate.

4. ISOLATE_NODE — Affected leaf switch (controlled)
   - Drain new jobs from rack behind switch; do not hard-power cycle during active MPI.
   - Shift new allocations to alternate leaf in same pod.

5. NOTIFY_OPERATOR — Job owners and NOC
   - Message template: congestion on {switch_id}, {victim_count} nodes affected.
   - Include top-5 talker jobs and recommended QoS policy name.

Rollback: Remove QoS after utilization <60% for 15 minutes; re-enable rack scheduling.
""".strip()

HDD_DEGRADATION_SOP = """
SOP: HDD Degradation on Storage Node
====================================
Scope: SMART reallocated sectors, elevated await, filesystem read errors.

1. CHECK_METRICS — SMART diagnostics
   - Read Reallocated_Sector_Ct, Current_Pending_Sector, UDMA_CRC_Error_Count.
   - Flag disk if reallocated >10 or pending >0.

2. CHECK_METRICS — Affected jobs and IO latency
   - List Lustre/GPFS OST bindings for degraded OSD.
   - Plot p95 read/write latency vs cluster baseline (target <20ms p95).

3. MIGRATE_JOB — Data migration when degradation >70%
   - Threshold: SMART health score <30% or reallocated growth >5/day.
   - Migrate datasets to healthy OST; throttle migration bandwidth to 40% link.

4. SCHEDULE_MAINTENANCE — Disk replacement
   - Open change ticket; hot-spare pull within maintenance window.
   - Verify RAID/replica quorum before physical swap.

5. NOTIFY_OPERATOR — Storage and workload owners
   - Include affected job list, estimated migration duration, and replacement ETA.

Post-replacement: Full SMART short+extended self-test; rejoin OST after clean scrub.
""".strip()

RAM_LEAK_SOP = """
SOP: RAM Leak on Compute Node
=============================
Scope: Monotonic RSS growth, swap thrashing, OOM killer events.

1. CHECK_METRICS — Process with highest RSS
   - Scan /proc for top RSS; correlate with Slurm/Moab job cgroup.
   - Capture heap trend over 30-minute window (slope >100MB/hr = leak suspect).

2. CHECK_METRICS — OOM killer and kernel logs
   - grep dmesg for "Out of memory" and "Killed process".
   - Record victim PID, job ID, and oom_score_adj if set.

3. RESTART_SERVICE — User workload remediation
   - Prefer graceful job checkpoint+restart on alternate node.
   - If no checkpoint: cancel job with NOTIFY to owner before SIGKILL cascade.

4. MIGRATE_JOB — Move remaining tasks off node
   - Drain node from scheduler; reschedule pending steps to healthy rack mates.

5. CHECK_METRICS — Swap and memory pressure
   - Alert if swap_used >50% or MemAvailable <5% for 10 minutes.

6. SCHEDULE_MAINTENANCE — Memory hardware test
   - Flag node for memtest86+ or EDAC-corrected error follow-up during next window.

Escalation: Repeated leaks from same binary → NOTIFY_OPERATOR with binary hash and module list.
""".strip()

SLURM_SCONTROL_NODE_DOC_SOP = """
Source: https://slurm.schedmd.com/scontrol.html
Extracted with requests + BeautifulSoup from the scontrol documentation.

Relevant NodeName command examples and interpretation:

1. scontrol show node
   - The documentation lists "scontrol show node" as a show command for node
     state inspection. Use it before remediation to confirm node state, job
     allocation, drain flags, and scheduler-visible health.

2. scontrol update NodeName=<nodes> State={POWER_UP|POWER_DOWN|POWER_DOWN_ASAP|POWER_DOWN_FORCE}
   - The documentation marks this as prior usage superseded by the power
     subcommand, but it remains a useful SOP reference for interpreting node
     power remediation state.
   - POWER_DOWN_ASAP drains nodes first: currently running jobs complete, and
     no additional jobs are allocated to the nodes.
   - POWER_DOWN_FORCE cancels jobs, powers nodes down, and resets state to IDLE.

3. delete NodeName=<nodelist>
   - NodeName is a node name or node list. Only dynamic nodes with no running
     jobs and not part of a reservation can be deleted.
   - Multiple node names may use ranges such as lx[10-20].

Operational use in remediation:
   - Before ISOLATE_NODE or MIGRATE_JOB, inspect node state with scontrol show
     node and prefer drain/ASAP semantics over forceful cancellation.
   - For network congestion affecting compute hosts, avoid destructive node
     changes while MPI jobs are active; drain future allocations first.
""".strip()

CUMULUS_MONITORING_DOC_SOP = """
Source: https://docs.nvidia.com/networking-ethernet-software/cumulus-linux/Monitoring-and-Troubleshooting/
Extracted with requests + BeautifulSoup from the Cumulus Linux monitoring page.

Relevant monitoring commands found on the page:

1. nv show system
   - Shows switch uptime, hostname, product name, health status, product
     release, system MAC, and related system identity fields.

2. nv show system memory
   - Shows physical memory total/free/buffer/cache/used/utilization and swap
     totals. Use it to check whether switch control-plane pressure is
     contributing to telemetry gaps or delayed remediation.

3. nv show system cpu
   - Shows CPU model, core count, total utilization, load averages, and per-core
     utilization.

4. nv show platform
   - Shows platform inventory such as manufacturer, CPU, memory, serial number,
     ASIC model, UUID, and system type.

5. cl-support
   - The documentation describes generating a cl-support export for diagnostics
     and support requests.

Extraction note:
   - The fetched static page did not expose literal "show interface counters"
     examples. Keep the manual network SOP counters (SNMP ifHCInOctets,
     ifHCOutOctets, buffer drops, PFC pause frames, sFlow/NetFlow) as the
     interface-counter fallback for network_congestion remediation.
""".strip()


SOP_CHUNKS: List[SOPChunk] = [
    SOPChunk(
        chunk_id="net-001",
        fault_type="network_congestion",
        title="Network congestion leaf switch playbook",
        text=NETWORK_CONGESTION_SOP,
        tags=["switch", "qos", "congestion", "sflow", "mpi"],
    ),
    SOPChunk(
        chunk_id="hdd-001",
        fault_type="hdd_degradation",
        title="HDD degradation and SMART response",
        text=HDD_DEGRADATION_SOP,
        tags=["smart", "disk", "lustre", "migration", "ost"],
    ),
    SOPChunk(
        chunk_id="ram-001",
        fault_type="ram_leak",
        title="RAM leak and OOM response",
        text=RAM_LEAK_SOP,
        tags=["memory", "oom", "rss", "slurm", "swap"],
    ),
    SOPChunk(
        chunk_id="net-002",
        fault_type="network_congestion",
        title="QoS policy reference",
        text=(
            "QoS policy rate_limit_top_talkers: cap bulk TCP to 70% port speed, "
            "preserve latency-sensitive MPI and SSH. Apply on switch ports facing "
            "noisy neighbors identified by NetFlow."
        ),
        tags=["qos", "policy"],
    ),
    SOPChunk(
        chunk_id="hdd-002",
        fault_type="hdd_degradation",
        title="Migration threshold guidance",
        text=(
            "When SMART health below 30% or reallocated sector growth exceeds 5 per day, "
            "initiate MIGRATE_JOB to alternate OST before scheduling disk replacement."
        ),
        tags=["migration", "threshold"],
    ),
    SOPChunk(
        chunk_id="ram-002",
        fault_type="ram_leak",
        title="OOM escalation",
        text=(
            "On OOM kill: capture dmesg excerpt, notify job owner, restart or migrate job, "
            "and monitor swap usage until MemAvailable recovers above 15%."
        ),
        tags=["oom", "escalation"],
    ),
    SOPChunk(
        chunk_id="slurm-doc-001",
        fault_type="ram_leak",
        title="SLURM scontrol NodeName remediation commands",
        text=SLURM_SCONTROL_NODE_DOC_SOP,
        tags=["slurm", "scontrol", "NodeName", "drain", "scheduler"],
    ),
    SOPChunk(
        chunk_id="net-doc-001",
        fault_type="network_congestion",
        title="Cumulus Linux monitoring commands",
        text=CUMULUS_MONITORING_DOC_SOP,
        tags=["cumulus", "switch", "monitoring", "nv", "cl-support"],
    ),
]


def get_chunks_for_fault(fault_type: str) -> List[SOPChunk]:
    """Return all SOP chunks tagged for a fault type."""
    return [c for c in SOP_CHUNKS if c.fault_type == fault_type]


def get_all_chunks() -> List[SOPChunk]:
    return list(SOP_CHUNKS)
