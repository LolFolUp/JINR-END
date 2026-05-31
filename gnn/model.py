"""Self-contained GNN model definition for RCA inference.

This module is the *inference-time* counterpart of
``training_pipeline/train.py`` from the GNN experiment line (v5a_40).
It defines exactly the same ``GATv2Hetero`` + ``SharedScorer`` architecture
that produced the ``best_v5a_40_screening.pt`` checkpoint
(Hit@1 = 87.5%, Hit@3 = 92.3%, MRR = 0.903), but **without** the training
loop, argparse CLI, optimizer or dataset machinery.

Why a separate module instead of importing ``train.py``?
  * ``train.py`` parses ``sys.argv`` at import time and pulls in the full
    training stack — undesirable when imported from the LLM/RAG app or tests.
  * The integration only needs a forward pass, so we keep a minimal,
    dependency-light surface here.

The class is intentionally identical (layer names, shapes) to the training
definition so that ``state_dict`` from the training checkpoint loads cleanly.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv

# ---------------------------------------------------------------------------
# Architecture constants — MUST match training_pipeline/train.py (v5a_40)
# ---------------------------------------------------------------------------

HIDDEN_DIM = 64
HEADS = 4
DROPOUT = 0.1

EDGE_DIM_PHYSICAL = 19
EDGE_DIM_DUMMY = 1

# RC candidate node types (job and rca_context are NOT RC candidates).
RC_CANDIDATE_TYPES: List[str] = ["cpu", "gpu", "ram", "hdd", "switch"]
RC_TYPE_TO_IDX: Dict[str, int] = {nt: i for i, nt in enumerate(RC_CANDIDATE_TYPES)}

# Physical edge relation names (carry the 19-dim physical edge_attr).
_PHYSICAL_RELATIONS = {
    "connected_to",
    "attached_to",
    "uplink_to",
    "rev_connected_to_cpu",
    "rev_connected_to_gpu",
    "rev_attached_to_ram",
    "rev_attached_to_hdd",
    "rev_uplink_to",
}


class SharedScorer(nn.Module):
    """Single MLP scoring head shared across all RC candidate node types.

    The node type is embedded as a learned vector and concatenated to the
    hidden representation before the final projection, giving the model a
    common comparison space across cpu/gpu/ram/hdd/switch candidates.
    """

    def __init__(self, hidden_dim: int, n_rc_types: int = 5):
        super().__init__()
        self.type_emb = nn.Embedding(n_rc_types, hidden_dim // 4)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.35),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, type_idx: int) -> torch.Tensor:
        """h: [N, hidden_dim], type_idx: int. Returns [N, 1]."""
        t = self.type_emb(torch.tensor(type_idx, device=h.device))
        t = t.unsqueeze(0).expand(h.size(0), -1)
        return self.mlp(torch.cat([h, t], dim=-1))


class GATv2Hetero(nn.Module):
    """Two-layer heterogeneous GATv2 with residual connections and LayerNorm.

    Input dims are supplied as ``node_dims`` (read from metadata.json or the
    checkpoint). Output: ``Dict[node_type, logits]`` of shape ``[N]`` per type.
    """

    def __init__(
        self,
        node_dims: Dict[str, int],
        edge_types: List[Tuple[str, str, str]],
        hidden_dim: int = HIDDEN_DIM,
        heads: int = HEADS,
        dropout: float = DROPOUT,
        scorer_mode: str = "shared",
    ) -> None:
        super().__init__()

        self.node_types = list(node_dims.keys())
        self.edge_types = [tuple(et) for et in edge_types]
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout_p = dropout
        self.out_dim = hidden_dim
        self.scorer_mode = scorer_mode

        # ---- Input projections (per node type) ----------------------------
        self.input_proj = nn.ModuleDict({
            f"proj_{nt}": nn.Linear(node_dims[nt], hidden_dim, bias=False)
            for nt in self.node_types
        })

        # ---- Layer norms after each GATv2 layer ---------------------------
        self.ln1 = nn.ModuleDict({
            f"ln1_{nt}": nn.LayerNorm(hidden_dim) for nt in self.node_types
        })
        self.ln2 = nn.ModuleDict({
            f"ln2_{nt}": nn.LayerNorm(hidden_dim) for nt in self.node_types
        })

        # ---- GATv2 layer 1 ------------------------------------------------
        self.conv1 = HeteroConv(self._build_conv_dict(), aggr="sum")
        # ---- GATv2 layer 2 ------------------------------------------------
        self.conv2 = HeteroConv(self._build_conv_dict(), aggr="sum")

        # ---- Classifier / scoring heads -----------------------------------
        if scorer_mode == "shared":
            self.shared_scorer = SharedScorer(
                hidden_dim, n_rc_types=len(RC_CANDIDATE_TYPES)
            )
            self.classifiers = nn.ModuleDict({
                f"cls_{nt}": nn.Linear(hidden_dim, 1)
                for nt in self.node_types
                if nt not in RC_CANDIDATE_TYPES and nt != "rca_context"
            })
        else:  # per_type
            self.classifiers = nn.ModuleDict({
                f"cls_{nt}": nn.Linear(hidden_dim, 1)
                for nt in self.node_types
                if nt != "rca_context"
            })

    def _build_conv_dict(self) -> Dict[Tuple[str, str, str], GATv2Conv]:
        conv_dict: Dict[Tuple[str, str, str], GATv2Conv] = {}
        for et in self.edge_types:
            src, rel, dst = et
            ea_dim = EDGE_DIM_PHYSICAL if rel in _PHYSICAL_RELATIONS else EDGE_DIM_DUMMY
            conv_dict[et] = GATv2Conv(
                in_channels=(self.hidden_dim, self.hidden_dim),
                out_channels=self.hidden_dim // self.heads,
                heads=self.heads,
                edge_dim=ea_dim,
                dropout=self.dropout_p,
                add_self_loops=False,
                concat=True,
            )
        return conv_dict

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor],
        return_embeddings: bool = False,
    ):
        """Returns logits dict, or (logits, h2_out) if return_embeddings."""

        # ---- Input projection ---------------------------------------------
        h: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            h[nt] = F.elu(self.input_proj[f"proj_{nt}"](x_dict[nt].float()))

        # ---- Layer 1 + residual + LayerNorm --------------------------------
        h1 = self.conv1(h, edge_index_dict, edge_attr_dict=edge_attr_dict)
        h1_out: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            if nt in h1:
                res = h1[nt] + h[nt]
                res = F.dropout(res, p=self.dropout_p, training=self.training)
                h1_out[nt] = self.ln1[f"ln1_{nt}"](res)
            else:
                h1_out[nt] = h[nt]

        # ---- Layer 2 + residual + LayerNorm --------------------------------
        h2 = self.conv2(h1_out, edge_index_dict, edge_attr_dict=edge_attr_dict)
        h2_out: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            if nt in h2:
                res = h2[nt] + h1_out[nt]
                res = F.dropout(res, p=self.dropout_p, training=self.training)
                h2_out[nt] = self.ln2[f"ln2_{nt}"](res)
            else:
                h2_out[nt] = h1_out[nt]

        # ---- Scoring heads ------------------------------------------------
        logits: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            if nt == "rca_context":
                continue
            if self.scorer_mode == "shared" and nt in RC_CANDIDATE_TYPES:
                type_idx = RC_TYPE_TO_IDX[nt]
                logits[nt] = self.shared_scorer(h2_out[nt], type_idx).squeeze(-1)
            else:
                cls_key = f"cls_{nt}"
                if cls_key in self.classifiers:
                    logits[nt] = self.classifiers[cls_key](h2_out[nt]).squeeze(-1)

        if return_embeddings:
            return logits, h2_out
        return logits
