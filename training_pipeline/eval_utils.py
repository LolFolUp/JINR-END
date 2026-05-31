"""Unified RCA evaluator — single source of truth for split + metrics.

Both train.py and eval_per_fault.py import from here so that every reported
number uses the SAME val split (torch.randperm) and the SAME metric code.

History: a numpy-vs-torch split mismatch inflated Hit@1 from 60.7% to 70.1%.
This module removes that entire class of bug by centralizing the split.

Design: pure functions operate on per-graph data. The high-level
`run_evaluation()` takes a `forward_fn(batch)->logits_dict` callable so this
module does not need to import the model from train.py (avoids circular import).
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

# RCA candidate node types — only these are ranked for root-cause localization.
# job and rca_context are never candidates.
RC_CANDIDATE_TYPES: List[str] = ["cpu", "gpu", "ram", "hdd", "switch"]

# Canonical 9 fault types (must match fault_type_idx in v4.4+ graphs).
FAULT_TYPES: List[str] = [
    "hdd_degradation",             # 0
    "network_congestion",          # 1
    "ram_leak",                    # 2
    "cpu_frequency_drop",          # 3
    "cpu_cache_thrashing",         # 4
    "memory_bandwidth_saturation", # 5
    "swap_thrashing",              # 6
    "gpu_thermal_throttle",        # 7
    "disk_full",                   # 8
]


# ---------------------------------------------------------------------------
# Split — the ONE canonical split. Must match train.py:build_dataloaders.
# ---------------------------------------------------------------------------

def build_train_val_indices(
    num_total: int,
    train_ratio: float = 0.80,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Deterministic train/val index split via torch.randperm (NOT numpy).

    train.py uses exactly this; any evaluator MUST use it too or metrics
    are measured on a different val set.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(num_total, generator=rng).tolist()
    n_train = int(num_total * train_ratio)
    return perm[:n_train], perm[n_train:]


class RCADataset(Dataset):
    """Loads data_{i}.pt graphs by index from a raw dir."""

    def __init__(self, indices: List[int], raw_dir: str):
        self.indices = list(indices)
        self.raw_dir = raw_dir

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> HeteroData:
        idx = self.indices[i]
        return torch.load(
            os.path.join(self.raw_dir, f"data_{idx}.pt"), weights_only=False
        )


def build_val_loader(
    num_total: int,
    raw_dir: str,
    train_ratio: float = 0.80,
    seed: int = 42,
    batch_size: int = 8,
) -> DataLoader:
    """Build the canonical val DataLoader (shuffle=False)."""
    _, val_idx = build_train_val_indices(num_total, train_ratio, seed)
    return DataLoader(RCADataset(val_idx, raw_dir), batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Per-graph unbatching (carries fault_type_idx)
# ---------------------------------------------------------------------------

def unbatch_with_fault(
    logits: Dict[str, torch.Tensor],
    batch: HeteroData,
) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]], List[int]]:
    """Split batched logits/labels into per-graph dicts + fault_type_idx list.

    Returns (per_graph_logits, per_graph_y, per_graph_fault_idx).
    fault_type_idx is -1 for healthy graphs.
    """
    n_graphs = batch.num_graphs
    pg_logits: List[Dict[str, torch.Tensor]] = [dict() for _ in range(n_graphs)]
    pg_y:      List[Dict[str, torch.Tensor]] = [dict() for _ in range(n_graphs)]

    for nt, lg in logits.items():
        ptr = getattr(batch[nt], "ptr", None)
        if ptr is None:
            pg_logits[0][nt] = lg.detach().cpu()
            if hasattr(batch[nt], "y"):
                pg_y[0][nt] = batch[nt].y.detach().cpu()
            continue
        for gi in range(n_graphs):
            s, e = int(ptr[gi]), int(ptr[gi + 1])
            if s >= e:
                continue
            pg_logits[gi][nt] = lg[s:e].detach().cpu()
            if hasattr(batch[nt], "y"):
                pg_y[gi][nt] = batch[nt].y[s:e].detach().cpu()

    # fault_type_idx per graph
    if hasattr(batch, "fault_type_idx"):
        fault_idx = [int(x) for x in batch.fault_type_idx.detach().cpu().tolist()]
    else:
        fault_idx = [-2] * n_graphs  # -2 = unknown (old dataset without metadata)

    return pg_logits, pg_y, fault_idx


# ---------------------------------------------------------------------------
# Core ranking: produce RC rank per graph
# ---------------------------------------------------------------------------

def _rc_rank_for_graph(
    g_logits: Dict[str, torch.Tensor],
    g_y: Dict[str, torch.Tensor],
    temperature: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """Rank all RC candidate nodes; return (rank_of_true_RC, true_rc_type, pred_top_type).

    temperature: optional per-node-type divisor applied to logits before sigmoid.
    Returns (None, None, None) for healthy graphs (no positive label).
    """
    candidates: List[Tuple[float, int, str]] = []  # (score, label, node_type)
    true_rc_type: Optional[str] = None

    for nt in RC_CANDIDATE_TYPES:
        if nt not in g_logits:
            continue
        lg = g_logits[nt].float().squeeze(-1)
        if temperature is not None and nt in temperature:
            lg = lg / temperature[nt]
        sc = torch.sigmoid(lg)
        if sc.dim() == 0:
            sc = sc.unsqueeze(0)
        y_nt = g_y.get(nt, torch.zeros(sc.shape[0]))
        if int(y_nt.sum()) > 0:
            true_rc_type = nt
        for i in range(len(sc)):
            lbl = int(y_nt[i]) if i < len(y_nt) else 0
            candidates.append((float(sc[i]), lbl, nt))

    if true_rc_type is None:
        return None, None, None  # healthy

    candidates.sort(key=lambda x: x[0], reverse=True)
    pred_top_type = candidates[0][2] if candidates else None

    rc_rank: Optional[int] = None
    for r, (_, lbl, _) in enumerate(candidates):
        if lbl == 1:
            rc_rank = r + 1
            break
    return rc_rank, true_rc_type, pred_top_type


# ---------------------------------------------------------------------------
# Master report
# ---------------------------------------------------------------------------

def compute_rca_report(
    per_graph_logits: List[Dict[str, torch.Tensor]],
    per_graph_y: List[Dict[str, torch.Tensor]],
    per_graph_fault_idx: Optional[List[int]] = None,
    temperature: Optional[Dict[str, float]] = None,
) -> Dict:
    """Full RCA report: global + per-fault + per-RC + histogram.

    Healthy graphs (no positive RC label) are excluded automatically.
    """
    ranks: List[int] = []
    per_rc: Dict[str, List[int]] = defaultdict(list)
    per_ft: Dict[str, List[int]] = defaultdict(list)

    for gi, (g_logits, g_y) in enumerate(zip(per_graph_logits, per_graph_y)):
        rc_rank, true_rc_type, _ = _rc_rank_for_graph(g_logits, g_y, temperature)
        if rc_rank is None:
            continue  # healthy
        ranks.append(rc_rank)
        per_rc[true_rc_type].append(rc_rank)
        if per_graph_fault_idx is not None and gi < len(per_graph_fault_idx):
            fi = per_graph_fault_idx[gi]
            if 0 <= fi < len(FAULT_TYPES):
                per_ft[FAULT_TYPES[fi]].append(rc_rank)

    n = len(ranks)
    if n == 0:
        return {"hit1": 0.0, "hit3": 0.0, "hit5": 0.0, "mrr": 0.0,
                "median_rank": 0, "n": 0, "per_rc_type": {}, "per_fault_type": {},
                "rank_histogram": {}}

    arr = np.array(ranks, dtype=np.float32)

    def _bucket(rks: List[int]) -> Dict[str, float]:
        a = np.array(rks, dtype=np.float32)
        return {
            "hit1": float(np.mean(a == 1)),
            "hit3": float(np.mean(a <= 3)),
            "hit5": float(np.mean(a <= 5)),
            "mrr":  float(np.mean(1.0 / a)),
            "median_rank": int(np.median(a)),
            "n": len(rks),
        }

    result: Dict = {
        "hit1":        float(np.mean(arr == 1)),
        "hit3":        float(np.mean(arr <= 3)),
        "hit5":        float(np.mean(arr <= 5)),
        "mrr":         float(np.mean(1.0 / arr)),
        "median_rank": int(np.median(arr)),
        "n":           n,
        "per_rc_type":    {nt: _bucket(v) for nt, v in per_rc.items()},
        "per_fault_type": {ft: _bucket(v) for ft, v in per_ft.items()},
    }

    # Rank histogram
    hist = {
        "rank_1":     int(np.sum(arr == 1)),
        "rank_2_3":   int(np.sum((arr >= 2) & (arr <= 3))),
        "rank_4_5":   int(np.sum((arr >= 4) & (arr <= 5))),
        "rank_6_10":  int(np.sum((arr >= 6) & (arr <= 10))),
        "rank_gt10":  int(np.sum(arr > 10)),
    }
    result["rank_histogram"] = hist

    # Macro averages over fault types
    if result["per_fault_type"]:
        result["macro_hit1"] = float(np.mean([v["hit1"] for v in result["per_fault_type"].values()]))
        result["macro_hit3"] = float(np.mean([v["hit3"] for v in result["per_fault_type"].values()]))
        result["worst_fault"] = min(result["per_fault_type"].items(),
                                    key=lambda x: x[1]["hit1"])[0]
    return result


# ---------------------------------------------------------------------------
# High-level: run a model over the val loader and return the report
# ---------------------------------------------------------------------------

def run_evaluation(
    forward_fn: Callable[[HeteroData], Dict[str, torch.Tensor]],
    val_loader: DataLoader,
    device: torch.device,
    temperature: Optional[Dict[str, float]] = None,
    collect_raw: bool = False,
) -> Dict:
    """Run forward_fn over all val batches, return compute_rca_report(...).

    forward_fn(batch) -> {node_type: logits[N]}; batch is already on device.
    If collect_raw, also returns per-graph logits/y/fault for diagnostics under
    keys '_pg_logits', '_pg_y', '_pg_fault'.
    """
    all_pg_logits: List[Dict[str, torch.Tensor]] = []
    all_pg_y:      List[Dict[str, torch.Tensor]] = []
    all_pg_fault:  List[int] = []

    was_training = None
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits = forward_fn(batch)
            pg_l, pg_y, pg_f = unbatch_with_fault(logits, batch)
            all_pg_logits.extend(pg_l)
            all_pg_y.extend(pg_y)
            all_pg_fault.extend(pg_f)

    report = compute_rca_report(all_pg_logits, all_pg_y, all_pg_fault, temperature)
    if collect_raw:
        report["_pg_logits"] = all_pg_logits
        report["_pg_y"]      = all_pg_y
        report["_pg_fault"]  = all_pg_fault
    return report


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def format_report(report: Dict, title: str = "RCA EVALUATION") -> str:
    """Return a human-readable multi-line string of the report."""
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append(title)
    lines.append("=" * 80)
    lines.append(f"  Hit@1={report['hit1']:.4f}  Hit@3={report['hit3']:.4f}  "
                 f"Hit@5={report['hit5']:.4f}  MRR={report['mrr']:.4f}  "
                 f"median_rank={report['median_rank']}  n={report['n']}")

    if report.get("rank_histogram"):
        h = report["rank_histogram"]
        lines.append("  rank histogram: "
                     f"[1]={h['rank_1']} [2-3]={h['rank_2_3']} [4-5]={h['rank_4_5']} "
                     f"[6-10]={h['rank_6_10']} [>10]={h['rank_gt10']}")

    if report.get("per_fault_type"):
        lines.append("")
        lines.append("Per-fault Hit@1/3/5/MRR/median (weakest first):")
        rows = sorted(report["per_fault_type"].items(), key=lambda x: x[1]["hit1"])
        lines.append(f"  {'fault_type':<30}{'n':>4} {'H@1':>7}{'H@3':>7}{'H@5':>7}{'MRR':>7}{'med':>5}")
        for ft, v in rows:
            lines.append(f"  {ft:<30}{v['n']:>4} {v['hit1']:>6.1%} {v['hit3']:>6.1%} "
                         f"{v['hit5']:>6.1%} {v['mrr']:>6.3f} {v['median_rank']:>4d}")
        lines.append(f"  macro_hit1={report.get('macro_hit1',0):.3f}  "
                     f"macro_hit3={report.get('macro_hit3',0):.3f}  "
                     f"worst={report.get('worst_fault','?')}")

    if report.get("per_rc_type"):
        lines.append("")
        lines.append("Per-RC-type Hit@1/3/5/MRR/median:")
        for nt in RC_CANDIDATE_TYPES:
            if nt in report["per_rc_type"]:
                v = report["per_rc_type"][nt]
                lines.append(f"  {nt:<8} H@1={v['hit1']:.3f} H@3={v['hit3']:.3f} "
                             f"H@5={v['hit5']:.3f} MRR={v['mrr']:.3f} "
                             f"med={v['median_rank']} n={v['n']}")
    return "\n".join(lines)
