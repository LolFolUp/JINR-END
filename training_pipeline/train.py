"""
training_pipeline/train.py

Step 10: Production-grade training pipeline for heterogeneous GATv2
RCA model on the synthetic supercomputer cluster dataset.

Node-level binary classification: identify the root-cause node.
Masking policy: victims are excluded from loss (loss_mask == True only).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, HeteroConv
from torch_geometric.nn import Linear as PyGLinear

from training_pipeline.diagnostics import (
    CANDIDATE_TYPES, full_report, per_type_report, score_loader,
    summary_line, typed_to_flat,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR  = os.path.join(_ROOT, "dataset")
RAW_DIR      = os.path.join(DATASET_DIR, "raw")
CKPT_DIR     = os.path.join(_ROOT, "checkpoints")
MANIFEST_DIR     = os.path.join(DATASET_DIR, "manifest")
METADATA_PATH    = os.path.join(DATASET_DIR, "metadata.json")
LOSS_CFG_PATH    = os.path.join(DATASET_DIR, "loss_config.json")
BEST_MODEL_PATH  = os.path.join(CKPT_DIR, "best_model.pt")
HISTORY_PATH     = os.path.join(CKPT_DIR, "training_history.json")

# Versioning support (optional — training works without manifest present)
try:
    from training_pipeline.versioning import (
        build_checkpoint_versioning_metadata,
        check_checkpoint_compatibility,
        get_dataset_hash_from_manifest,
        get_dataset_version_from_manifest,
        compute_behavioral_code_hash,
        compute_feature_ordering_hash,
    )
    from training_pipeline.experiment_registry import (
        new_experiment_id,
        append_experiment,
        build_experiment_metadata,
    )
    from training_pipeline.config import FEATURE_SCHEMA
    _VERSIONING_AVAILABLE = True
except ImportError:
    _VERSIONING_AVAILABLE = False

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

HIDDEN_DIM   = 64
HEADS        = 4
DROPOUT      = 0.1
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 30
BATCH_SIZE   = 16
PATIENCE     = 5
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
SEED         = 42

# Phase-2 recalibration (see artifacts/phase1_summary.md)
POS_WEIGHT_CAP = 20.0   # F2: raw pos_weight up to 332 saturates logits

# Phase-5 / E1: listwise root-cause objective — L = BCE + LISTWISE_WEIGHT * listwise
LISTWISE_WEIGHT = 1.0

EDGE_DIM_PHYSICAL = 19
EDGE_DIM_DUMMY    = 1

# Physical edge relation names (must match config.py)
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

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RCADataset(Dataset):
    """Lazy-loading dataset — reads .pt files on demand, no RAM accumulation."""

    def __init__(self, indices: List[int], raw_dir: str = RAW_DIR) -> None:
        self.indices = indices
        self.raw_dir = raw_dir

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> HeteroData:
        file_idx = self.indices[idx]
        path = os.path.join(self.raw_dir, f"data_{file_idx}.pt")
        return torch.load(path, weights_only=False)


def build_dataloaders(
    num_total: int,
    train_ratio: float = TRAIN_RATIO,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    val_ratio: float = VAL_RATIO,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Deterministic 3-way split -> (train, val, test) DataLoaders.

    The seed and permutation are unchanged from the original 80/20 split, so
    the test slice perm[n_train+n_val:] is a strict subset of the original
    validation region perm[800:]. A checkpoint trained on the old 80% split
    has therefore never seen the new test set — no train/test leakage.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(num_total, generator=rng).tolist()

    n_train = int(num_total * train_ratio)
    n_val   = int(num_total * val_ratio)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train + n_val]
    test_idx  = perm[n_train + n_val:]

    train_loader = DataLoader(RCADataset(train_idx), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(RCADataset(val_idx),   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(RCADataset(test_idx),  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GATv2Hetero(nn.Module):
    """Two-layer heterogeneous GATv2 with residual connections and LayerNorm.

    Input dims are read from metadata.json at construction time.
    Output: Dict[node_type, logits]  shape [N] per node type.
    """

    def __init__(
        self,
        node_dims: Dict[str, int],
        edge_types: List[Tuple[str, str, str]],
        hidden_dim: int = HIDDEN_DIM,
        heads: int = HEADS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        self.node_types  = list(node_dims.keys())
        self.edge_types  = edge_types
        self.hidden_dim  = hidden_dim
        self.heads       = heads
        self.dropout_p   = dropout
        self.out_dim     = hidden_dim  # heads are concatenated then projected

        # ---- Input projections (per node type) ----------------------------
        self.input_proj = nn.ModuleDict({
            f"proj_{nt}": nn.Linear(node_dims[nt], hidden_dim, bias=False)
            for nt in self.node_types
        })

        # ---- Layer norms after each GATv2 layer ---------------------------
        self.ln1 = nn.ModuleDict({
            f"ln1_{nt}": nn.LayerNorm(hidden_dim)
            for nt in self.node_types
        })
        self.ln2 = nn.ModuleDict({
            f"ln2_{nt}": nn.LayerNorm(hidden_dim)
            for nt in self.node_types
        })

        # ---- GATv2 layer 1 ------------------------------------------------
        conv1_dict: Dict[Tuple[str, str, str], GATv2Conv] = {}
        for et in edge_types:
            src, rel, dst = et
            ea_dim = EDGE_DIM_PHYSICAL if rel in _PHYSICAL_RELATIONS else EDGE_DIM_DUMMY
            conv1_dict[et] = GATv2Conv(
                in_channels=(hidden_dim, hidden_dim),
                out_channels=hidden_dim // heads,
                heads=heads,
                edge_dim=ea_dim,
                dropout=dropout,
                add_self_loops=False,
                concat=True,
            )
        self.conv1 = HeteroConv(conv1_dict, aggr="sum")

        # ---- GATv2 layer 2 ------------------------------------------------
        conv2_dict: Dict[Tuple[str, str, str], GATv2Conv] = {}
        for et in edge_types:
            src, rel, dst = et
            ea_dim = EDGE_DIM_PHYSICAL if rel in _PHYSICAL_RELATIONS else EDGE_DIM_DUMMY
            conv2_dict[et] = GATv2Conv(
                in_channels=(hidden_dim, hidden_dim),
                out_channels=hidden_dim // heads,
                heads=heads,
                edge_dim=ea_dim,
                dropout=dropout,
                add_self_loops=False,
                concat=True,
            )
        self.conv2 = HeteroConv(conv2_dict, aggr="sum")

        # ---- Classifier heads (per node type, excluding rca_context) ------
        self.classifiers = nn.ModuleDict({
            f"cls_{nt}": nn.Linear(hidden_dim, 1)
            for nt in self.node_types
            if nt != "rca_context"
        })

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict:  Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        # ---- Input projection ---------------------------------------------
        h: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            h[nt] = F.elu(self.input_proj[f"proj_{nt}"](x_dict[nt].float()))

        # ---- Layer 1 + residual + LayerNorm --------------------------------
        h1 = self.conv1(h, edge_index_dict, edge_attr_dict=edge_attr_dict)
        h1_out: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            if nt in h1:
                res = h1[nt] + h[nt]          # residual
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

        # ---- Classifier heads ---------------------------------------------
        logits: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            if nt == "rca_context":
                continue
            logits[nt] = self.classifiers[f"cls_{nt}"](h2_out[nt]).squeeze(-1)

        return logits


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_masked_loss(
    logits: Dict[str, torch.Tensor],
    batch: HeteroData,
    pos_weights: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """BCEWithLogitsLoss with soft victim weighting.

    When data[nt].node_weight is present (hardened dataset):
        loss = mean(BCE(logit, y) * node_weight)
        RC nodes  → weight 1.0  (full gradient)
        Healthy   → weight 1.0
        Victims   → weight U(0.2, 0.4)  (soft gradient, not excluded)

    Fallback (legacy dataset without node_weight):
        Victims (loss_mask == False) are excluded entirely — original behaviour.
    """
    total_loss   = torch.tensor(0.0, device=device)
    n_terms      = 0

    for nt, lg in logits.items():
        if nt not in CANDIDATE_TYPES:
            continue  # F3: cpu/gpu/job never root cause — skip trivial all-negative terms
        if not hasattr(batch[nt], "y") or batch[nt].y is None:
            continue
        y  = batch[nt].y.float().to(device)
        pw = pos_weights.get(nt, torch.tensor(1.0)).to(device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
        node_loss = bce(lg, y)   # [N]

        if hasattr(batch[nt], "node_weight") and batch[nt].node_weight is not None:
            # Soft weighting path (hardened dataset)
            w = batch[nt].node_weight.float().to(device)  # [N]
            denom = w.sum().clamp(min=1e-6)
            total_loss = total_loss + (node_loss * w).sum() / denom
            n_terms    += 1
        else:
            # Hard-mask fallback (legacy dataset)
            if not hasattr(batch[nt], "loss_mask"):
                continue
            mask = batch[nt].loss_mask.to(device)
            if mask.sum() == 0:
                continue
            total_loss = total_loss + node_loss[mask].mean()
            n_terms    += 1

    if n_terms == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / n_terms


# ---------------------------------------------------------------------------
# Listwise root-cause loss (Phase 5 / E1)
# ---------------------------------------------------------------------------

def _listwise_graph_loss(
    cand_logits: torch.Tensor, cand_labels: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Per-graph listwise loss: -log softmax(cand_logits)[true RC index].

    Returns None for graphs with no candidate nodes or no root cause
    (healthy graphs) — they contribute nothing to the objective.
    """
    if cand_logits.numel() == 0 or int(cand_labels.sum()) == 0:
        return None
    rc_idx = int(torch.argmax(cand_labels))
    logp = torch.log_softmax(cand_logits, dim=0)
    return -logp[rc_idx]


def compute_listwise_loss(
    logits: Dict[str, torch.Tensor],
    batch: HeteroData,
    device: torch.device,
) -> torch.Tensor:
    """E1: mean per-graph listwise root-cause loss.

    For each faulted graph, softmax over the concatenated candidate-node
    logits {hdd,switch,ram} and penalise -log p(true RC). Reuses the BCE
    head logits — no extra parameters. Healthy and malformed (no-candidate)
    graphs are skipped; safe against empty softmax and NaN.
    """
    n_graphs = int(getattr(batch, "num_graphs", 1))
    per_logits: List[List[torch.Tensor]] = [[] for _ in range(n_graphs)]
    per_labels: List[List[torch.Tensor]] = [[] for _ in range(n_graphs)]

    for nt in CANDIDATE_TYPES:
        if nt not in logits or nt not in batch.node_types:
            continue
        node = batch[nt]
        if not hasattr(node, "y") or node.y is None:
            continue
        y = node.y
        gvec = getattr(node, "batch", None)
        if gvec is None:
            gvec = torch.zeros(y.size(0), dtype=torch.long, device=y.device)
        for g in range(n_graphs):
            m = (gvec == g)
            if bool(m.any()):
                per_logits[g].append(logits[nt][m])
                per_labels[g].append(y[m])

    losses: List[torch.Tensor] = []
    for g in range(n_graphs):
        if not per_logits[g]:
            continue
        gl = _listwise_graph_loss(
            torch.cat(per_logits[g]), torch.cat(per_labels[g])
        )
        if gl is not None:
            losses.append(gl)

    if not losses:
        return torch.zeros((), device=device, requires_grad=True)
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Training / validation steps
# ---------------------------------------------------------------------------

def _batch_to_device(batch: HeteroData, device: torch.device) -> HeteroData:
    """Move all tensors in a HeteroData batch to device."""
    return batch.to(device)


def _prepare_inputs(
    batch: HeteroData,
    edge_types: List[Tuple[str, str, str]],
    device: torch.device,
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[Tuple[str, str, str], torch.Tensor],
    Dict[Tuple[str, str, str], torch.Tensor],
]:
    x_dict: Dict[str, torch.Tensor] = {}
    for nt in batch.node_types:
        x_dict[nt] = batch[nt].x.float().to(device)

    ei_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
    ea_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
    for et in edge_types:
        if et in batch.edge_types:
            ei_dict[et] = batch[et].edge_index.to(device)
            ea_dict[et] = batch[et].edge_attr.float().to(device)

    return x_dict, ei_dict, ea_dict


def train_one_epoch(
    model: GATv2Hetero,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pos_weights: Dict[str, torch.Tensor],
    device: torch.device,
    edge_types: List[Tuple[str, str, str]],
    grad_clip: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """Run one training epoch. Returns (mean_loss, grad_norm_stats).

    grad_norm_stats records the mean/std gradient norm of each candidate
    classifier head — evidence for whether the batch-size / pos_weight
    recalibration reduced gradient variance (F4).
    """
    model.train()
    total_loss = 0.0
    total_bce  = 0.0
    total_lw   = 0.0
    n_batches  = 0
    grad_norms: Dict[str, List[float]] = {nt: [] for nt in CANDIDATE_TYPES}

    for batch in loader:
        batch = _batch_to_device(batch, device)
        x_dict, ei_dict, ea_dict = _prepare_inputs(batch, edge_types, device)

        optimizer.zero_grad()
        logits = model(x_dict, ei_dict, ea_dict)
        loss_bce = compute_masked_loss(logits, batch, pos_weights, device)
        loss_lw  = compute_listwise_loss(logits, batch, device)
        loss = loss_bce + LISTWISE_WEIGHT * loss_lw

        if torch.isnan(loss) or torch.isinf(loss):
            print("  [WARN] NaN/Inf loss detected — skipping batch")
            continue

        loss.backward()
        for nt in CANDIDATE_TYPES:
            w = model.classifiers[f"cls_{nt}"].weight
            if w.grad is not None:
                grad_norms[nt].append(float(w.grad.norm()))
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_bce  += loss_bce.item()
        total_lw   += loss_lw.item()
        n_batches  += 1

    stats: Dict[str, float] = {}
    for nt, norms in grad_norms.items():
        if norms:
            t = torch.tensor(norms)
            stats[f"{nt}_mean"] = round(float(t.mean()), 5)
            stats[f"{nt}_std"]  = round(float(t.std(unbiased=False)), 5)
    n = max(n_batches, 1)
    stats["bce_mean"]      = round(total_bce / n, 5)
    stats["listwise_mean"] = round(total_lw / n, 5)
    return total_loss / n, stats


def validate(
    model: GATv2Hetero,
    loader: DataLoader,
    pos_weights: Dict[str, torch.Tensor],
    device: torch.device,
    edge_types: List[Tuple[str, str, str]],
) -> Tuple[float, Dict[str, object]]:
    """Run validation. Returns (val_loss, diagnostics report).

    The report is diagnostics.full_report() over the candidate node set —
    per-graph, candidate-restricted, threshold-swept — plus a per_type
    breakdown. Replaces the broken per-batch compute_rca_accuracy path (F1).
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            x_dict, ei_dict, ea_dict = _prepare_inputs(batch, edge_types, device)
            logits = model(x_dict, ei_dict, ea_dict)
            loss = compute_masked_loss(logits, batch, pos_weights, device)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                n_batches  += 1

    typed = score_loader(model, loader, edge_types, device)
    report = full_report(typed_to_flat(typed))
    report["per_type"] = per_type_report(typed)
    return total_loss / max(n_batches, 1), report


def evaluate_model(
    model: GATv2Hetero,
    loader: DataLoader,
    pos_weights: Dict[str, torch.Tensor],
    device: torch.device,
    edge_types: List[Tuple[str, str, str]],
    split_name: str = "val",
) -> Dict[str, object]:
    """Load the best checkpoint and print final diagnostics on `loader`."""
    if os.path.exists(BEST_MODEL_PATH):
        state = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        print(f"Loaded best checkpoint from epoch {state.get('epoch', '?')}")

    loss, report = validate(model, loader, pos_weights, device, edge_types)

    print("\n" + "=" * 60)
    print(f"FINAL EVALUATION  ({split_name} split)")
    print("=" * 60)
    print(f"  loss : {loss:.4f}")
    print(f"  {summary_line(report)}")
    print(f"  rank histogram : {report['rca']['rank_histogram']}")
    for nt, m in report.get("per_type", {}).items():
        rt = m["rca"]["top1"]
        rt_s = "N/A" if rt is None else f"{rt:.3f}"
        print(f"  {nt:<8} AUC={m['roc_auc']:.3f}  "
              f"F1@best={m['f1_at_best_threshold']:.3f}  RCA-Top1={rt_s}")
    return report


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Versioning: load dataset identity --------------------------------
    _dataset_version = "unknown"
    _dataset_hash    = "unknown"
    _dataset_codename = "unknown"
    _exp_id = None

    if _VERSIONING_AVAILABLE and os.path.exists(MANIFEST_DIR):
        try:
            _dataset_hash    = get_dataset_hash_from_manifest(MANIFEST_DIR)
            _dataset_version = get_dataset_version_from_manifest(MANIFEST_DIR)
            print(f"Dataset version : {_dataset_version}")
            print(f"Dataset hash    : {_dataset_hash[:12]}...")
            _exp_id = new_experiment_id()
            print(f"Experiment ID   : {_exp_id}")
        except Exception as e:
            print(f"[versioning] Warning: could not read manifest: {e}")
    elif _VERSIONING_AVAILABLE:
        print("[versioning] No manifest found — running without version tracking")

    # ---- Load metadata and loss config ------------------------------------
    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)
    with open(LOSS_CFG_PATH, encoding="utf-8") as f:
        loss_cfg = json.load(f)

    node_dims: Dict[str, int] = metadata["feature_dimensions"]
    edge_types: List[Tuple[str, str, str]] = [
        tuple(et) for et in metadata["edge_types"]
    ]
    num_total: int = metadata["dataset_size"]

    pos_weights: Dict[str, torch.Tensor] = {
        nt: torch.tensor([min(pw, POS_WEIGHT_CAP)], dtype=torch.float32)
        for nt, pw in loss_cfg["pos_weight"].items()
    }
    print(f"pos_weight capped at {POS_WEIGHT_CAP} "
          f"(raw max was {max(loss_cfg['pos_weight'].values()):.1f})")

    print(f"Dataset: {num_total} graphs  |  node types: {list(node_dims.keys())}")
    print(f"Edge types: {len(edge_types)}")

    # ---- DataLoaders -------------------------------------------------------
    train_loader, val_loader, test_loader = build_dataloaders(
        num_total=num_total,
        train_ratio=TRAIN_RATIO,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )
    print(
        f"Train: {len(train_loader.dataset)} graphs  |  "
        f"Val: {len(val_loader.dataset)} graphs  |  "
        f"Test: {len(test_loader.dataset)} graphs (held out)"
    )

    # ---- Model -------------------------------------------------------------
    model = GATv2Hetero(
        node_dims=node_dims,
        edge_types=edge_types,
        hidden_dim=HIDDEN_DIM,
        heads=HEADS,
        dropout=DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ---- Optimizer ---------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    # ---- Training history --------------------------------------------------
    history: Dict[str, List] = {
        "train_loss": [], "val_loss": [],
        "val_f1": [], "val_roc_auc": [], "rca_accuracy": [],
        "val_mrr": [], "grad_norms": [],
    }

    best_val_f1    = -1.0
    patience_count = 0
    best_epoch     = 0

    print()
    print("=" * 78)
    print("Training — early-stop signal: best-threshold F1 on val candidate set")
    print("=" * 78)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, grad_stats = train_one_epoch(
            model, train_loader, optimizer, pos_weights, device, edge_types
        )
        val_loss, report = validate(
            model, val_loader, pos_weights, device, edge_types
        )

        val_f1  = float(report["f1_at_best_threshold"])
        val_auc = float(report["roc_auc"])
        _rca    = report["rca"]["top1"]
        _mrr    = report["rca"]["mrr"]
        rca_acc = float(_rca) if _rca is not None else 0.0
        mrr_val = float(_mrr) if _mrr is not None else 0.0
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_roc_auc"].append(val_auc)
        history["rca_accuracy"].append(rca_acc)
        history["val_mrr"].append(mrr_val)
        history["grad_norms"].append(grad_stats)

        print(
            f"Epoch {epoch:02d} | loss={train_loss:.4f} "
            f"(bce={grad_stats.get('bce_mean', 0):.3f} "
            f"lw={grad_stats.get('listwise_mean', 0):.3f}) | "
            f"val_loss={val_loss:.4f} | AUC={val_auc:.4f} | "
            f"F1@best={val_f1:.4f} | RCA={rca_acc:.4f} | MRR={mrr_val:.4f} | "
            f"{elapsed:.1f}s"
        )

        # ---- Checkpoint on best-threshold F1 (F5: val_f1@0.5 was corrupted) ----
        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            best_epoch     = epoch
            patience_count = 0

            # Build versioning metadata block
            _versioning_block: Dict = {}
            if _VERSIONING_AVAILABLE:
                try:
                    _versioning_block = build_checkpoint_versioning_metadata(
                        dataset_version=_dataset_version,
                        dataset_hash=_dataset_hash,
                        manifest_dir=MANIFEST_DIR,
                        feature_schema=FEATURE_SCHEMA,
                        node_dims=node_dims,
                        model_config={
                            "hidden_dim": HIDDEN_DIM,
                            "heads":      HEADS,
                            "dropout":    DROPOUT,
                        },
                        train_py_path=__file__,
                    )
                except Exception as e:
                    print(f"[versioning] Warning: checkpoint metadata error: {e}")

            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_f1":           val_f1,
                    "rca_accuracy":     rca_acc,
                    "node_dims":        node_dims,
                    "edge_types":       edge_types,
                    # --- versioning block ---
                    **_versioning_block,
                },
                BEST_MODEL_PATH,
            )
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE})")
                break

    # ---- Save training history --------------------------------------------
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved -> {HISTORY_PATH}")
    print(f"Best model: epoch {best_epoch}  val_f1={best_val_f1:.4f}")

    # ---- Final evaluation on best checkpoint ------------------------------
    evaluate_model(model, val_loader,  pos_weights, device, edge_types, "val")
    evaluate_model(model, test_loader, pos_weights, device, edge_types, "test")

    # ---- Log experiment to registry ----------------------------------------
    if _VERSIONING_AVAILABLE and _exp_id is not None:
        try:
            _train_end = time.time()
            _final_metrics = history.get("val_roc_auc", [])
            _code_versions: Dict[str, str] = {}
            try:
                _code_versions["train_code_hash"] = compute_behavioral_code_hash(__file__)
                _code_versions["feature_ordering_hash"] = compute_feature_ordering_hash(FEATURE_SCHEMA)
            except Exception:
                pass

            exp_meta = build_experiment_metadata(
                exp_id=_exp_id,
                dataset_version=_dataset_version,
                dataset_hash=_dataset_hash,
                dataset_codename=_dataset_codename,
                model_config={
                    "model_type": "HeteroGATv2",
                    "hidden_dim":  HIDDEN_DIM,
                    "heads":       HEADS,
                    "dropout":     DROPOUT,
                },
                loss_config=loss_cfg.get("pos_weight", {}),
                optimizer_config={"optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY},
                training_config={
                    "epochs":          EPOCHS,
                    "batch_size":      BATCH_SIZE,
                    "patience":        PATIENCE,
                    "train_ratio":     TRAIN_RATIO,
                    "seed":            SEED,
                    "device":          str(device),
                    "pos_weight_cap":  POS_WEIGHT_CAP,
                    "listwise_weight": LISTWISE_WEIGHT,
                },
                metrics={
                    "best_val_f1":          best_val_f1,
                    "best_epoch":           best_epoch,
                    "final_val_roc_auc":    _final_metrics[-1] if _final_metrics else None,
                    "final_rca_accuracy":   history.get("rca_accuracy", [None])[-1],
                },
                runtime={"checkpoint_path": BEST_MODEL_PATH},
                code_versions=_code_versions,
                checkpoint_path=BEST_MODEL_PATH,
                status="completed",
            )
            append_experiment(exp_meta)
            print(f"Experiment logged -> experiments/registry.jsonl  [{_exp_id}]")
        except Exception as e:
            print(f"[versioning] Warning: could not log experiment: {e}")


if __name__ == "__main__":
    # Suppress the pandas/NumPy compatibility warnings from PyG datasets import
    import warnings
    warnings.filterwarnings("ignore")
    main()
