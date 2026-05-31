# Layer 1: Knowledge Retrieval (RAG)

## Module: `rag`

This module provides **Retrieval-Augmented Generation** — semantic search for relevant Standard Operating Procedures (SOPs).

### Purpose

Given a root cause (e.g., "switch SWITCH-5 has network_congestion"), find the most relevant remediation SOPs from a knowledge base.

### Components

#### `retriever.py`
- **Function**: `semantic_search(query: str, top_k: int)`
- **Input**: Natural language query (e.g., "network_congestion root cause SWITCH-5 affected 27 nodes")
- **Output**: Top-K most relevant SOP chunks with scores

**Example**:
```python
from rag.retriever import SemanticRetriever

retriever = SemanticRetriever()
results = retriever.search(
    query="network_congestion root cause SWITCH-5 affected 27 nodes",
    top_k=5
)
# Output:
# [
#   {"score": 0.98, "sop_id": "NET-001", "content": "Congestion diagnosis..."},
#   {"score": 0.95, "sop_id": "NET-003", "content": "Switch failover..."},
#   ...
# ]
```

#### `qdrant_store.py`
- **Database**: Qdrant (vector database)
- **Collections**: 
  - `sops` → 7 SOP types (CHECK_METRICS, APPLY_QOS, ISOLATE_NODE, ...)
  - `topology` → HPC cluster topology information
- **Embedding**: Sentence-Transformers (multilingual)
- **Query**: Semantic similarity (cosine distance)

**Lifecycle**:
```python
from rag.qdrant_store import QdrantStore

store = QdrantStore(host="localhost", port=6333)
store.create_collection("sops")
store.upsert([
    {"id": "NET-001", "text": "...", "vector": [...]}
])
results = store.search("network_congestion", top_k=5)
```

#### `redis_context.py`
- **Cache**: Redis (in-memory)
- **Purpose**: Store fault context for quick retrieval
- **TTL**: 1 hour per incident

**Cached Keys**:
```python
{
  "fault:network_congestion": {
    "affected_nodes": 27,
    "typical_duration": "5-10 minutes",
    "priority": "HIGH"
  },
  "topology:switch": {
    "num_switches": 6,
    "failover_available": true
  }
}
```

#### `knowledge_base.py`
- **Static knowledge**: 7 Standard Operating Procedures
- **Format**: Markdown with structured steps

**SOP Types**:
1. `CHECK_METRICS` — Verify fault diagnosis with metrics
2. `APPLY_QOS` — Apply Quality-of-Service limits
3. `ISOLATE_NODE` — Quarantine faulty resource
4. `RESTART_SERVICE` — Graceful service restart
5. `MIGRATE_JOB` — Move workload to healthy node
6. `NOTIFY_OPERATOR` — Alert on-call engineer
7. `SCHEDULE_MAINTENANCE` — Schedule future repair

**Example SOP**:
```markdown
# SOP: Network Congestion Recovery

## Diagnosis
- Check packet loss on switches
- Verify interface errors
- Monitor latency spike

## Actions
1. Check metrics on SWITCH-{id}
2. Apply QOS: max_throughput = 80%
3. Isolate faulty interface
4. Reroute traffic via backup spine
5. Notify network ops team
6. Schedule maintenance

## Expected Duration
5-10 minutes
```

#### `embedder.py`
- **Model**: `sentence-transformers/multilingual-MiniLM-L12-v2`
- **Purpose**: Convert text → semantic vector (384-dim)
- **Input**: SOP text or query string
- **Output**: Vector suitable for Qdrant

```python
from rag.embedder import TextEmbedder

embedder = TextEmbedder()
vector = embedder.embed("network_congestion switch failure")
# shape: (384,)
```

#### `history_tickets.py`
- **Historical Reference**: Past incidents + resolutions
- **Lookup**: By fault type + severity
- **Purpose**: Example-based reasoning for LLM

```python
from rag.history_tickets import HistoryTicketsStore

history = HistoryTicketsStore()
similar_incidents = history.query(
    fault_type="network_congestion",
    severity="HIGH",
    limit=3
)
# Output: [
#   {"ticket_id": "INC-2024-001", "resolution": "..."},
#   {"ticket_id": "INC-2024-002", "resolution": "..."}
# ]
```

### Data Flow

```
Input: GNN inference
  ↓
remediation/pipeline.py::build_retrieval_query()
  ↓
"network_congestion root cause SWITCH-5 affected 27 nodes"
  ↓
rag/retriever.py::search()
  ├─ Check Redis cache
  ├─ Query Qdrant (semantic search)
  └─ Return top-5 SOPs
  ↓
Context: {
  "sop_chunks": 5,
  "retrieval_method": "qdrant_semantic",
  "sop_content": [...]
}
  ↓
→ Layer 2 (LLM reasoning)
```

### Performance

| Operation | Latency |
|-----------|---------|
| Redis cache hit | <1 ms |
| Qdrant search | 1-2 ms |
| Embedding | 50-100 ms (but cached) |
| Total | <2 ms (with cache) |

### Storage

```
qdrant/
├── collections/
│   ├── sops/
│   │   ├── sop_net_001.json
│   │   ├── sop_storage_001.json
│   │   └── sop_memory_001.json
│   └── topology/
│       └── topology_2024.json
└── snapshot/
    └── backup.tar.gz
```

### Configuration

```json
{
  "qdrant": {
    "host": "localhost",
    "port": 6333,
    "timeout": 5
  },
  "redis": {
    "host": "localhost",
    "port": 6379,
    "ttl_seconds": 3600
  },
  "embedder": {
    "model": "sentence-transformers/multilingual-MiniLM-L12-v2",
    "device": "cpu"
  }
}
```

### Setup (Local Development)

```bash
# Start Qdrant (Docker)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Start Redis (Docker)
docker run -p 6379:6379 redis:7

# Populate knowledge base
python -c "from rag.knowledge_base import load_sops; load_sops()"
```

### Testing

```bash
# Integration test (includes RAG)
pytest tests/test_full_system_integration.py -v

# Expected:
# [RAG] SOPs retrieved: 5  method=qdrant_semantic
# [RAG] Retrieval took: <2 ms
```

### Integration with Other Layers

**Input from Layer 0**:
```python
inference = {
    "graph_id": 123,
    "fault_type": "network_congestion",
    "rc_node": {"type": "switch", "id": "SWITCH-5"}
}
```

**Output to Layer 2**:
```python
context = {
    "sop_chunks": 5,
    "retrieval_method": "qdrant_semantic",
    "sop_content": [
        "1. Check metrics on SWITCH...",
        "2. Apply QOS limits...",
        ...
    ]
}
```

### Degradation & Fallback

If Qdrant is unavailable:
- Use empty SOP context `[]`
- LLM generates actions from zero context
- System continues (graceful degradation)

### Future Extensions

1. **Real incident history**: Integrate with ticket system (Jira/Linear)
2. **Dynamic SOP updates**: Hot-reload from runbooks
3. **Cached embeddings**: Pre-compute SOP vectors
4. **Multi-language support**: SOPs in Russian + English
5. **Semantic cache**: Deduplicate similar queries
