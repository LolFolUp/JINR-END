"""
training_pipeline/config.py

Cluster topology constants, feature schemas, edge type definitions,
suffix parsing rules, and validation asserts for the heterogeneous
GNN-based RCA system of a supercomputer cluster.

No computational logic, no classes, no graph generation, no PyTorch code.
"""

from typing import Final, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Cluster topology constants
# ---------------------------------------------------------------------------

NUM_SPINE: Final[int] = 2
NUM_LEAF: Final[int] = 4
NUM_HOSTS: Final[int] = 100
NUM_JOBS: Final[int] = 500
SIMULATION_STEPS: Final[int] = 90
FAULT_INJECTION_STEP: Final[int] = 50

# EMA smoothing coefficient: alpha = 2 / (N + 1), N = 65 ticks (v3.0: longer window
# preserves delta_long signal over 40-step fault ramps — see v3_migration_report.md)
EMA_ALPHA: Final[float] = 0.030

# Edge feature dimensionality
EDGE_DIM_PHYSICAL: Final[int] = 19
EDGE_DIM_DUMMY: Final[int] = 1

# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------

NODE_TYPES: Final[List[str]] = [
    "cpu",
    "gpu",
    "ram",
    "hdd",
    "switch",
    "job",
    "rca_context",
]

# ---------------------------------------------------------------------------
# Edge type definitions
# ---------------------------------------------------------------------------

# Physical edges: hardware connectivity in the Spine-Leaf topology.
# Each entry is (src_type, relation_name, dst_type).
# Forward edges followed by their mirror reverse edges.
PHYSICAL_EDGE_TYPES: Final[List[Tuple[str, str, str]]] = [
    # Forward
    ("cpu",    "connected_to",  "switch"),
    ("gpu",    "connected_to",  "switch"),
    ("ram",    "attached_to",   "cpu"),
    ("hdd",    "attached_to",   "cpu"),
    ("switch", "uplink_to",     "switch"),
    # Reverse (mirrors)
    ("switch", "rev_connected_to_cpu", "cpu"),
    ("switch", "rev_connected_to_gpu", "gpu"),
    ("cpu",    "rev_attached_to_ram",  "ram"),
    ("cpu",    "rev_attached_to_hdd",  "hdd"),
    ("switch", "rev_uplink_to",        "switch"),
]

# Logical edges: scheduler / workload relationships.
LOGICAL_EDGE_TYPES: Final[List[Tuple[str, str, str]]] = [
    # Forward
    ("job", "executes_on",     "cpu"),
    # Reverse
    ("cpu", "rev_executes_on", "job"),
]

# Context edges: aggregation into the RCA context super-node.
CONTEXT_EDGE_TYPES: Final[List[Tuple[str, str, str]]] = [
    # Forward
    ("job",    "reports_to",         "rca_context"),
    ("switch", "reports_to",         "rca_context"),
    # Reverse
    ("rca_context", "rev_reports_to_job",    "job"),
    ("rca_context", "rev_reports_to_switch", "switch"),
]

# ---------------------------------------------------------------------------
# Feature suffix rules
# ---------------------------------------------------------------------------

CONTINUOUS_SUFFIXES: Final[List[str]] = [
    "_percent",
    "_bytes",
    "_ps",
    "_ms",
    "_celsius",
    "_mhz",
    "_watts",
    "_count",
    "_rate",
    "_score",
    "_ratio",
    "_time",
    "_mb",
    "_dbm",
    "_sectors",
    "_seconds",
    "_allocated",
]

CATEGORICAL_SUFFIXES: Final[List[str]] = [
    "_encoded",
    "_flag",
    "_status",
]

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

FEATURE_SCHEMA: Final[Dict[str, List[str]]] = {
    "cpu": [
        "cpu_usage_total_percent",
        "cpu_user_percent",
        "cpu_system_percent",
        "cpu_iowait_percent",
        "cpu_frequency_mhz",
        "cpu_temperature_celsius",
        "cpu_power_watts",
        "cpu_context_switches_ps",
    ],
    "gpu": [
        "gpu_utilization_percent",
        "gpu_memory_used_mb",
        "gpu_temperature_celsius",
        "gpu_power_watts",
        "gpu_sm_clock_mhz",
        "gpu_memory_clock_mhz",
        "gpu_tensor_utilization_percent",
        "gpu_pcie_tx_mb",
        "gpu_pcie_rx_mb",
        "gpu_ecc_errors_count",
        "gpu_fan_speed_percent",
        "gpu_throttle_flag",
    ],
    "ram": [
        "ram_used_percent",
        "ram_available_mb",
        "ram_cached_mb",
        "ram_swap_used_percent",
        "ram_bandwidth_mb",
        "ram_latency_ns",
        "ram_page_faults_ps",
        "ram_fragmentation_score",
    ],
    "hdd": [
        "disk_usage_percent",
        "disk_read_mb",
        "disk_write_mb",
        "disk_read_iops",
        "disk_write_iops",
        "disk_latency_ms",
        "disk_temperature_celsius",
        "disk_reallocated_sectors",
    ],
    "switch": [
        "switch_port_utilization_percent",
        "switch_packet_loss_percent",
        "switch_crc_errors_count",
        "switch_latency_ms",
        "switch_bandwidth_usage_percent",
        "switch_temperature_celsius",
        "switch_power_watts",
        "switch_active_connections_count",
        "switch_rx_mb",
        "switch_tx_mb",
    ],
    "link_edge": [
        "link_bandwidth_usage_percent",
        "link_latency_ms",
        "link_packet_loss_percent",
        "link_jitter_ms",
        "link_crc_errors_count",
        "link_rx_mb",
        "link_tx_mb",
        "link_dropped_packets_count",
        "link_retransmits_count",
        "link_queue_depth_count",
        "link_signal_strength_dbm",
        "link_power_watts",
        "link_utilization_ratio",
        "link_congestion_score",
        "link_timeout_count",
        "link_reset_count",
        "link_availability_percent",
        "link_error_rate",
        "link_status_encoded",
    ],
    "job": [
        "job_cpu_usage_percent",
        "job_gpu_usage_percent",
        "job_ram_usage_percent",
        "job_io_read_mb",
        "job_io_write_mb",
        "job_network_rx_mb",
        "job_network_tx_mb",
        "job_runtime_seconds",
        "job_wait_time_seconds",
        "job_priority_encoded",
        "job_status_encoded",
    ],
}

# rca_context has no feature schema — it is a virtual aggregation node.

# ---------------------------------------------------------------------------
# Validation asserts
# ---------------------------------------------------------------------------

assert len(FEATURE_SCHEMA["cpu"])      == 8,  "cpu feature count mismatch"
assert len(FEATURE_SCHEMA["gpu"])      == 12, "gpu feature count mismatch"
assert len(FEATURE_SCHEMA["ram"])      == 8,  "ram feature count mismatch"
assert len(FEATURE_SCHEMA["hdd"])      == 8,  "hdd feature count mismatch"
assert len(FEATURE_SCHEMA["switch"])   == 10, "switch feature count mismatch"
assert len(FEATURE_SCHEMA["link_edge"]) == 19, "link_edge feature count mismatch"
assert len(FEATURE_SCHEMA["job"])      == 11, "job feature count mismatch"

# All feature names must be unique within each node type
for _node_type, _features in FEATURE_SCHEMA.items():
    assert len(_features) == len(set(_features)), (
        f"Duplicate feature names detected in schema for node type '{_node_type}'"
    )

# All edge groups must be non-empty
assert len(PHYSICAL_EDGE_TYPES) > 0, "PHYSICAL_EDGE_TYPES must not be empty"
assert len(LOGICAL_EDGE_TYPES)  > 0, "LOGICAL_EDGE_TYPES must not be empty"
assert len(CONTEXT_EDGE_TYPES)  > 0, "CONTEXT_EDGE_TYPES must not be empty"

# All suffix lists must be non-empty
assert len(CONTINUOUS_SUFFIXES)  > 0, "CONTINUOUS_SUFFIXES must not be empty"
assert len(CATEGORICAL_SUFFIXES) > 0, "CATEGORICAL_SUFFIXES must not be empty"
