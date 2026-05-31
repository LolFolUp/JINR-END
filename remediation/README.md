# Layers 3-4: Validation & Output

## Module: `remediation`

This module provides:
- **Layer 3**: Firewall validation (semantic security)
- **Layer 4**: Aggregation & report generation (final output)

### Purpose

Validate that LLM-generated actions are safe and produce final remediation report.

### Components

#### `firewall.py` (Layer 3)

**Purpose**: Semantic validation of remediation actions — block dangerous commands.

**Key Constraint**: LLM has NO direct terminal access. All outputs must pass firewall first.

**Functions**:

```python
def validate_playbook(playbook: RemediationPlaybook) -> str:
    """Validate all actions in playbook.
    
    Returns: "PASSED" | "BLOCKED"
    """

def build_fallback_playbook(inference: dict) -> RemediationPlaybook:
    """Generate safe rule-based playbook if LLM unavailable.
    
    Guarantees: >= 3 actions, all validated
    """
```

**Validation Rules**:

```python
# Allowed actions
ALLOWED_ACTIONS = {
    "CHECK_METRICS",
    "APPLY_QOS",
    "ISOLATE_NODE",
    "MIGRATE_JOB",
    "RESTART_SERVICE",
    "NOTIFY_OPERATOR",
    "SCHEDULE_MAINTENANCE"
}

# Blocked keywords (anywhere in action)
BLOCKED_KEYWORDS = {
    "rm", "dd", "kill", "fork",
    ":/dev/", "reboot", "shutdown",
    "mkfs", "fdisk", "parted",
    "eval", "exec", "system"
}

# Target validation
def is_valid_target(action: Action) -> bool:
    # Must match topology node types
    return action.target_node in KNOWN_NODES
```

**Validation Flow**:

```
Input action:
  {"action_id": "APPLY_QOS", "target_node": "SWITCH-5", ...}
  
  ↓ Check 1: action_id in ALLOWED_ACTIONS?
  ↓ Check 2: keywords in parameters? (blacklist)
  ↓ Check 3: target_node in topology?
  ↓ Check 4: parameters match schema?
  
  ✓ All pass → PASSED
  ✗ Any fail → BLOCKED (log reason)
```

**Example - BLOCKED Action**:

```python
action = Action(
    action_id="RUN_COMMAND",  # ✗ Not in ALLOWED_ACTIONS
    target_node="SWITCH-5",
    parameters={"cmd": "rm -rf /"}
)

status = validate_playbook(RemediationPlaybook(actions=[action]))
# → "BLOCKED"
# Reason: "action_id 'RUN_COMMAND' not in allowed actions"
```

**Example - PASSED Action**:

```python
action = Action(
    action_id="APPLY_QOS",  # ✓ Allowed
    target_node="SWITCH-5",  # ✓ Valid topology node
    parameters={"max_throughput_gbps": 80}  # ✓ Valid schema
)

status = validate_playbook(RemediationPlaybook(actions=[action]))
# → "PASSED"
```

#### `incident_aggregator.py` (Layer 4)

**Purpose**: Assemble incident metadata from all sources.

**Key Function**:

```python
def aggregate(inference: dict) -> dict:
    """Extract incident metadata from GNN inference.
    
    Returns: {
        "id": "INC-2024-123-NETWORK_CONGESTION",
        "severity": "HIGH" | "CRITICAL" | "MEDIUM" | "LOW",
        "summary": "Network switch SWITCH-5 congestion...",
        "fault_type": "network_congestion",
        "affected_count": 27
    }
    """
```

**Severity Mapping**:

```python
def get_severity(fault_type: str, affected_count: int) -> str:
    if affected_count > 100 or fault_type == "network_congestion":
        return "CRITICAL"
    elif affected_count > 50:
        return "HIGH"
    elif affected_count > 10:
        return "MEDIUM"
    else:
        return "LOW"
```

**Example**:

```python
inference = {
    "graph_id": 123,
    "fault_type": "network_congestion",
    "rc_node": {"type": "switch", "id": "SWITCH-5"},
    "victim_nodes": [{...}, ...],  # 27 nodes
    "confidence": 0.783
}

incident = IncidentAggregator().aggregate(inference)
# → {
#     "id": "INC-2024-123-NETWORK_CONGESTION",
#     "severity": "CRITICAL",
#     "summary": "Network congestion at SWITCH-5...",
#     "affected_count": 27
# }
```

#### `models.py` (Layer 4)

**Purpose**: Pydantic schemas for validation & serialization.

**Classes**:

```python
class Action(BaseModel):
    """Remediation action."""
    action_id: str  # CHECK_METRICS, APPLY_QOS, etc.
    target_node: str
    parameters: dict
    estimated_ttr_seconds: int
    priority: int = 0
    
    @field_validator("action_id")
    def validate_action_id(cls, v):
        if v not in ALLOWED_ACTIONS:
            raise ValueError(f"{v} not in allowed actions")
        return v

class RemediationPlaybook(BaseModel):
    """Complete remediation plan."""
    incident_id: str
    rc_node: dict
    fault_type: str
    confidence: float
    actions: list[Action]
    
    @field_validator("confidence")
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be 0-1")
        return v
```

#### `pipeline.py` (Layer 4)

**Purpose**: Orchestrate all 4 layers + produce final output.

**Main Function**:

```python
def run_pipeline(
    inference: dict,
    *,
    force_rule_based: bool = False,
    llm_client: Optional[LLMClient] = None,
    ...
) -> Tuple[RemediationPlaybook, Dict[str, Any]]:
    """Execute complete RAG + LLM + firewall pipeline.
    
    Returns:
    - playbook: RemediationPlaybook (actions + metadata)
    - metadata: Dict with timing, status, etc.
    """
```

**Execution Flow**:

```python
def run_pipeline(inference):
    total_start = time.perf_counter()
    
    # Layer 1: RAG
    rag_start = time.perf_counter()
    sops = retrieve_sops(inference)
    rag_ms = (time.perf_counter() - rag_start) * 1000
    
    # Layer 2: LLM
    llm_start = time.perf_counter()
    playbook_dict = llm_client.generate(...)
    llm_ms = (time.perf_counter() - llm_start) * 1000
    
    # Layer 3: Firewall
    fw_start = time.perf_counter()
    playbook = RemediationPlaybook(**playbook_dict)
    fw_status = validate_playbook(playbook)
    fw_ms = (time.perf_counter() - fw_start) * 1000
    
    # Layer 4: Metadata
    incident = IncidentAggregator().aggregate(inference)
    
    metadata = {
        "incident": incident,
        "firewall_status": fw_status,
        "ttr_breakdown": {
            "rag_ms": rag_ms,
            "llm_ms": llm_ms,
            "firewall_ms": fw_ms,
            "total_ms": (time.perf_counter() - total_start) * 1000
        }
    }
    
    return playbook, metadata
```

**Output**:

```python
playbook = RemediationPlaybook(
    incident_id="INC-2024-123-NETWORK_CONGESTION",
    rc_node={"type": "switch", "id": "SWITCH-5"},
    fault_type="network_congestion",
    confidence=0.783,
    actions=[
        Action(action_id="CHECK_METRICS", ...),
        Action(action_id="APPLY_QOS", ...),
        Action(action_id="NOTIFY_OPERATOR", ...)
    ]
)

metadata = {
    "incident": {...},
    "firewall_status": "PASSED",
    "ttr_breakdown": {
        "rag_ms": 1.5,
        "llm_ms": 6400,
        "firewall_ms": 0.8,
        "total_ms": 6402
    }
}
```

#### `run.py` (Layer 4)

**Purpose**: CLI entry point + human-readable formatting.

**Main Function**:

```python
def format_report(playbook: RemediationPlaybook, metadata: dict) -> str:
    """Convert playbook to human-readable terminal output.
    
    Returns: Multi-line string suitable for printing.
    """
```

**Example Output**:

```
=== REMEDIATION REPORT ===
Incident: INC-2024-123-NETWORK_CONGESTION
Severity: CRITICAL
Summary: Network switch SWITCH-5 has congestion on 27 nodes
Root Cause: SWITCH-5 (confidence: 78.3%)
Hostname: spine-01
Fault Type: network_congestion
Context source: gnn_inference

Actions:
  1. [HIGH] CHECK_METRICS -> SWITCH-5
     Parameters: {"metrics": ["packet_drop", "error_rate"]}
     Est. TTR: 30s

  2. [HIGH] APPLY_QOS -> SWITCH-5
     Parameters: {"max_throughput_gbps": 80}
     Est. TTR: 10s

  3. [NORMAL] NOTIFY_OPERATOR -> OPS-TEAM
     Parameters: {"severity": "HIGH"}
     Est. TTR: 0s

Retrieval method: qdrant_semantic
Firewall Status: PASSED
Total TTR: 6.4 seconds
```

### Data Structures

#### RemediationReport (JSON Output)

```json
{
  "incident": {
    "id": "INC-2024-123-NETWORK_CONGESTION",
    "severity": "CRITICAL",
    "summary": "Network congestion at SWITCH-5",
    "fault_type": "network_congestion"
  },
  "root_cause": {
    "node_type": "switch",
    "node_id": "SWITCH-5",
    "confidence": 0.783,
    "rank": 1
  },
  "context": {
    "affected_nodes": 27,
    "rc_hostname": "spine-01"
  },
  "actions": [
    {
      "action_id": "CHECK_METRICS",
      "target_node": "SWITCH-5",
      "parameters": {"metrics": ["packet_drop"]},
      "estimated_ttr_seconds": 30,
      "priority": 1
    }
  ],
  "knowledge": {
    "sop_chunks_retrieved": 5,
    "retrieval_method": "qdrant_semantic"
  },
  "firewall_status": "PASSED",
  "ttr_breakdown": {
    "rag_retrieval_ms": 1.5,
    "llm_generation_ms": 6400,
    "firewall_validation_ms": 0.8,
    "total_ms": 6402
  }
}
```

### Testing

```bash
# Full integration test
pytest tests/test_full_system_integration.py::test_full_system_integration -v

# Expected output:
# [RAG] SOPs retrieved: 5  method=qdrant_semantic
# [LLM] Backend: mistral  generation=6412ms
# [Firewall] Status: PASSED
# [Report] Saved to: artifacts/remediation_report.json
# 
# INTEGRATION TEST PASSED
```

### Integration with All Layers

```
Layer 0 (GNN)    → inference.json
         ↓
Layer 1 (RAG)    → sops context
         ↓
Layer 2 (LLM)    → raw actions
         ↓
Layer 3 (FW)     → validated playbook
         ↓
Layer 4 (Output) → report.json + terminal
```

### Error Handling

```python
try:
    playbook, metadata = run_pipeline(inference)
    
except KeyError:
    # Missing required field in inference
    playbook = build_fallback_playbook(inference)
    
except ValueError:
    # Invalid input schema
    logger.error(f"Invalid inference: {e}")
    raise
    
except TimeoutError:
    # LLM timeout
    playbook = build_fallback_playbook(inference)
    
finally:
    # Always write report
    report = remediation_report_payload(playbook, metadata)
    Path("artifacts/remediation_report.json").write_text(
        json.dumps(report, indent=2)
    )
```

### Performance Budget

| Layer | Latency | Budget |
|-------|---------|--------|
| RAG | 1.5 ms | 2% |
| LLM | 6400 ms | 97% |
| Firewall | 0.8 ms | 0.1% |
| Output | 5 ms | 0.1% |
| **Total** | **6406 ms** | **100%** |

LLM dominates. Further optimization requires caching or model quantization.

### Fallback Logic

If any layer fails:

1. **RAG fails** → use empty context
2. **LLM fails** → use rule-based playbook
3. **Firewall rejects** → block actions (error to operator)
4. **Output fails** → raise exception (infrastructure issue)

Fallback is **always better than crashing**.

### Future Extensions

1. **Action feedback**: Did action succeed? Affect next step.
2. **Confidence scoring**: LLM scores own action confidence.
3. **Rollback playbook**: If action causes new issues, auto-revert.
4. **Parallel actions**: Execute safe actions concurrently.
5. **Approval workflow**: Notify operator before high-risk actions.
