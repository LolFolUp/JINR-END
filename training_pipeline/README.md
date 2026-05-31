# Layer 0: GNN Inference

## Module: `training_pipeline`

This module provides the **GNN inference component** — the input layer for the remediation pipeline.

### Components

#### `train.py`
- **Class**: `GATv2Hetero` — Heterogeneous Graph Attention Network v2
- **Architecture**: 2 layers, 4 attention heads, 311,878 parameters
- **Input**: Heterogeneous graphs (6 node types, 16 edge types)
- **Output**: Logits for root cause detection

**Key Methods**:
```python
model = GATv2Hetero(node_dims={...}, edge_types=[...])
logits = model(x_dict, edge_index_dict, edge_attr_dict)
# logits[node_type] → shape [num_nodes, 1]
```

**Loss Function**:
- E1 Listwise Loss (for RCA ranking, not binary classification)
- Optimizes Hit@1 metric directly

**Metrics**:
- Hit@1: 78.3% (finds correct RC in top-1)
- Hit@3: 82.6% (finds correct RC in top-3)
- MRR: 0.823 (mean reciprocal rank)
- F1: 0.716 (balanced precision/recall)

#### `dataset_generator.py`
- **Purpose**: Generate synthetic HPC cluster faults
- **Output**: PyTorch geometric heterogeneous graphs
- **Fault Types**: network_congestion, hdd_degradation, ram_leak

**Topology**:
- 100 CPU nodes (job hosts)
- 6 network switches (2 spine, 4 leaf)
- 1000 SLURM job nodes
- Real Spine-Leaf topology

**Fault Injection**:
```python
# Cascade propagation
fault_type = "network_congestion"  # Root cause = switch
  ↓
switch packet loss ↑ 
  ↓
27 CPU nodes affected
  ↓
jobs stall (victims)
```

#### `diagnostics.py`
- **Function**: `rca_metrics(per_graph: list)` → Hit@K, MRR
- **Evaluates**: Per-graph ranking of RC candidates
- **Returns**: {Hit@1, Hit@3, Hit@5, MRR}

#### `versioning.py`
- Tracks model versions (v2.0.0, v3.0.0)
- Dataset versioning (v3.0.0 = 1000 graphs, 7 node types)
- Checkpoint metadata

### Checkpoint Format

File: `checkpoints/best_model.pt`

```python
checkpoint = {
    "epoch": 14,
    "model_state_dict": {...},
    "dataset_version": "v3.0.0",
    "node_dims": {...},
    "edge_types": [...],
    "metrics": {
        "hit_at_1": 0.783,
        "mRR": 0.823
    }
}
```

### Usage

#### Training
```bash
cd JINR-rag
python -m training_pipeline.train
```

#### Inference (on test graph)
```python
from training_pipeline.train import GATv2Hetero, RCADataset
import torch

# Load model
ckpt = torch.load("checkpoints/best_model.pt", weights_only=False)
model = GATv2Hetero(...)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Forward pass
data = torch.load("dataset/raw/data_0.pt", weights_only=False)
logits = model(data)

# Find top-1 RC
rc_logits = logits["switch"].squeeze()
rc_idx = rc_logits.argmax().item()
```

#### Generate Inference JSON
```python
from tests.test_full_system_integration import _run_gnn_inference

inference = _run_gnn_inference(model, data, ckpt)
# Output: {
#   "graph_id": 123,
#   "fault_type": "network_congestion",
#   "rc_node": {"type": "switch", "id": "SWITCH-5", "host_id": 5},
#   "confidence": 0.783,
#   "top5_candidates": [...],
#   ...
# }
```

### Dataset Structure

```
dataset/
├── raw/
│   ├── data_0.pt → HeteroData (torch_geometric)
│   ├── data_1.pt
│   └── data_999.pt
├── manifest/
│   ├── topology.json → Spine-Leaf layout
│   └── diagnostics.json → Node type definitions
└── loss_config.json → E1 loss hyperparameters
```

### Performance

| Metric | Value |
|--------|-------|
| Inference time | 14 ms / graph |
| Throughput | ~70 graphs/sec |
| Memory footprint | ~2 GB (model + data) |

### Testing

```bash
# Full system integration (uses inference)
pytest tests/test_full_system_integration.py::test_full_system_integration -v

# Expected output:
# [GNN] Loaded: epoch=14, dataset=v3.0.0, params=311,878
# [GNN] Test graph: data_123.pt  node_types=[...]
# [GNN] RC prediction: SWITCH-5  logit=1.234  score=0.7834
```

### Integration with Layer 1+

Output JSON is consumed by `remediation/pipeline.py::run_pipeline()`:

```python
inference = {
    "graph_id": 123,
    "fault_type": "network_congestion",
    "rc_node": {"type": "switch", "id": "SWITCH-5"},
    "confidence": 0.783,
    "top5_candidates": [...],
    "victim_nodes": [...]
}

# This becomes input to Layer 1 (RAG retrieval)
playbook, metadata = run_pipeline(inference)
```

### Fallback & Degradation

If GNN inference fails:
- Use null RC node
- Use empty confidence (0.0)
- Pipeline continues with rule-based actions

### Next Steps

1. **Validation**: `pytest tests/` (all 26 tests pass)
2. **Output**: `artifacts/inference_sample.json`
3. **Input to Layer 1**: RAG retrieval begins
