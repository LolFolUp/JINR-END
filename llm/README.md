# Layer 2: Reasoning & Generation (LLM)

## Module: `llm`

This module provides **Large Language Model inference** — generates remediation actions from incident context.

### Purpose

Given:
1. Root cause diagnosis (GNN output)
2. Relevant SOPs (RAG output)
3. Incident context

Generate a structured JSON playbook with remediation actions.

### Components

#### `llm_client.py`
- **Backend**: Mistral 7B (via Ollama, local or remote)
- **Interface**: HTTP API (OpenAI-compatible)
- **Timeout**: 30 seconds
- **Retry**: 3 attempts with exponential backoff

**Class**: `LLMClient`

```python
from llm.llm_client import LLMClient

client = LLMClient(
    base_url="http://localhost:11434",
    model="mistral",
    timeout=30,
    max_retries=3
)

response = client.generate(
    messages=[...],
    temperature=0.3,
    top_p=0.9,
    max_tokens=1024
)
# Output: {"choices": [{"message": {"content": "..."}}]}
```

**Fallback**:
```python
if client.is_available():
    playbook = client.generate(messages)
else:
    playbook = build_fallback_playbook(inference)
```

#### `prompt_builder.py`
- **Purpose**: Construct multi-turn prompts with context
- **Output**: OpenAI-compatible message format

**Key Functions**:

```python
def format_rc_node(rc_node: dict) -> str:
    """Format root cause for readability.
    
    Example output:
    "SWITCH-5 (network_congestion, confidence: 78.3%)"
    """

def build_incident_id(graph_id: int, fault_type: str) -> str:
    """Generate unique incident identifier.
    
    Example: "INC-2024-123-NETWORK_CONGESTION"
    """

def build_messages(
    inference: dict,
    sops: list,
    context: dict
) -> list:
    """Build complete prompt with system + user messages.
    
    Returns:
    [
        {
            "role": "system",
            "content": "You are RCA remediation AI..."
        },
        {
            "role": "user",
            "content": "Incident: network congestion..."
        }
    ]
    """
```

**Prompt Template**:

```
System Role:
You are an expert remediation AI for HPC infrastructure.
Your task is to generate safe, actionable remediation steps.
Output must be valid JSON.

Context:
- Incident: network_congestion
- Root Cause: SWITCH-5
- Confidence: 78.3%
- Affected nodes: 27

Relevant Procedures:
1. Check metrics on faulty switch
2. Apply QOS limits
3. Isolate bad interface
4. Reroute traffic
5. Notify operations

Generate remediation playbook with JSON structure:
{
  "actions": [
    {
      "action_id": "CHECK_METRICS",
      "target_node": "SWITCH-5",
      "parameters": {...}
    }
  ]
}
```

#### `response_parser.py`
- **Purpose**: Extract JSON from LLM response
- **Robustness**: Handle malformed JSON, markdown code blocks

**Key Functions**:

```python
def normalize_playbook_dict(response: str) -> dict:
    """Extract playbook JSON from LLM response.
    
    Handles:
    - Raw JSON
    - JSON in markdown code blocks (```json...```)
    - Incomplete JSON (trailing commas)
    - Escaped quotes
    
    Returns: {"actions": [...]}
    """

def extract_actions(playbook: dict) -> list:
    """Validate & extract action list.
    
    Each action structure:
    {
        "action_id": str,
        "target_node": str,
        "parameters": dict,
        "estimated_ttr_seconds": int
    }
    """
```

### Data Flow

```
Input from Layer 1:
  ├─ inference: {rc_node, fault_type, confidence, ...}
  └─ context: {sop_content, ...}
  
  ↓
  
prompt_builder.py::build_messages()
  → OpenAI-compatible messages
  
  ↓
  
llm_client.py::generate()
  → Mistral 7B inference
  
  Response (raw text):
  "```json
   {
     "actions": [
       {"action_id": "CHECK_METRICS", ...}
     ]
   }
   ```"
  
  ↓
  
response_parser.py::normalize_playbook_dict()
  → Clean JSON dict
  
  ↓
  
Output to Layer 3:
  {
    "actions": [
      {
        "action_id": "CHECK_METRICS",
        "target_node": "SWITCH-5",
        "parameters": {...}
      },
      ...
    ]
  }
```

### Performance

| Metric | Value |
|--------|-------|
| Latency | 6.4 seconds |
| Throughput | 1 playbook / 6.4 sec |
| Token count | ~200-300 tokens in, ~100-150 tokens out |
| Memory | ~4 GB (Mistral 7B model) |

### Configuration

```python
# Default (local Ollama)
client = LLMClient(
    base_url="http://localhost:11434",
    model="mistral",
    timeout=30
)

# Remote API (example)
client = LLMClient(
    base_url="https://api.anyscale.com/v1",
    model="mistralai/Mistral-7B-Instruct-v0.1",
    api_key=os.environ["ANYSCALE_API_KEY"],
    timeout=30
)
```

### Temperature & Sampling

```python
# Deterministic (best for safety)
temperature=0.1, top_p=0.9
→ Conservative action selection

# Balanced (recommended for RCA)
temperature=0.3, top_p=0.9
→ Some diversity, still focused

# Creative (not recommended)
temperature=0.8, top_p=0.95
→ May hallucinate actions
```

**Current setting**: `temperature=0.3` (conservative)

### Action Generation

Example output:

```json
{
  "actions": [
    {
      "action_id": "CHECK_METRICS",
      "target_node": "SWITCH-5",
      "parameters": {
        "metrics": ["packet_drop", "error_rate", "latency"]
      },
      "estimated_ttr_seconds": 30,
      "priority": 1
    },
    {
      "action_id": "APPLY_QOS",
      "target_node": "SWITCH-5",
      "parameters": {
        "max_throughput_gbps": 80,
        "priority_queue": "CRITICAL"
      },
      "estimated_ttr_seconds": 10,
      "priority": 2
    },
    {
      "action_id": "NOTIFY_OPERATOR",
      "target_node": "OPS-TEAM",
      "parameters": {
        "message": "Network switch SWITCH-5 requires attention",
        "severity": "HIGH"
      },
      "estimated_ttr_seconds": 0,
      "priority": 3
    }
  ]
}
```

### Error Handling

1. **Model unavailable**: Return rule-based fallback
2. **Invalid JSON**: Parse with lenient regex
3. **Missing fields**: Validate with Pydantic (Layer 3)
4. **Timeout**: Retry or use cached response

```python
try:
    response = client.generate(messages)
except TimeoutError:
    return build_fallback_playbook(inference)
except ValueError as e:
    logger.error(f"Parse error: {e}")
    return build_fallback_playbook(inference)
```

### Testing

```bash
# Full system test
pytest tests/test_full_system_integration.py -v

# Expected:
# [LLM] Backend: mistral  generation=6412ms
# [Firewall] Status: PASSED
```

### Integration with Layer 3 (Firewall)

Output playbook is validated by `remediation/firewall.py`:

```python
from remediation.firewall import validate_playbook

playbook_dict = {
    "actions": [
        {"action_id": "CHECK_METRICS", ...},
        {"action_id": "APPLY_QOS", ...}
    ]
}

playbook = RemediationPlaybook(**playbook_dict)  # Pydantic validation
status = validate_playbook(playbook)
# "PASSED" → actions allowed
# "BLOCKED" → dangerous action detected
```

### Degradation & Fallback

If LLM unavailable, use **rule-based fallback**:

```python
def build_fallback_playbook(inference: dict):
    fault_type = inference.get("fault_type")
    
    if fault_type == "network_congestion":
        return RemediationPlaybook(
            actions=[
                Action(action_id="CHECK_METRICS", ...),
                Action(action_id="ISOLATE_NODE", ...),
                Action(action_id="NOTIFY_OPERATOR", ...)
            ]
        )
    elif fault_type == "hdd_degradation":
        return RemediationPlaybook(...)
    # ... etc
```

Fallback guarantees at least 3 actions even if LLM fails.

### Future Extensions

1. **Multi-turn conversation**: Ask follow-up questions
2. **Larger models**: Mistral 8x7B MoE (faster)
3. **Finetuning**: Finetune Mistral on RCA incidents
4. **Confidence scoring**: LLM scores own action confidence
5. **Chain-of-Thought**: LLM explains reasoning
6. **Caching**: Cache responses for identical incidents
