"""
training_pipeline/diagnostics.py

Single source of truth for RCA pipeline evaluation metrics.

Every metric here is computed PER-GRAPH first, then aggregated — never
per-batch. train.py and evaluate_pipeline.py import from this module so
that no metric logic is ever duplicated or allowed to diverge.

CANDIDATE SET
-------------
Root-cause nodes only ever occur on types {hdd, switch, ram}. cpu/gpu/job
are structurally never root cause, so they are excluded from every RCA and
classification metric. This is an inference-time structural prior, not a
label leak. All functions operate on the candidate set only.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

CANDIDATE_TYPES: Tuple[str, ...] = ("hdd", "switch", "ram")

# One graph's candidate scores: (logits_1d, labels_1d), equal length.
GraphScores = Tuple[torch.Tensor, torch.Tensor]


# ---------------------------------------------------------------------------
# Per-graph extraction
# ---------------------------------------------------------------------------

def extract_candidate_scores(
    logits: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
) -> GraphScores:
    """Concatenate logits and labels over CANDIDATE_TYPES for ONE graph."""
    ls: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    for nt in CANDIDATE_TYPES:
        if nt in logits and nt in labels:
            lg = logits[nt].detach().reshape(-1).float().cpu()
            y = labels[nt].detach().reshape(-1).long().cpu()
            if lg.numel() != y.numel():
                raise ValueError(
                    f"[{nt}] logits/labels length mismatch: {lg.numel()} vs {y.numel()}"
                )
            ls.append(lg)
            ys.append(y)
    if not ls:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(ls), torch.cat(ys)


# ---------------------------------------------------------------------------
# Primitive metrics
# ---------------------------------------------------------------------------

def roc_auc(labels: torch.Tensor, probs: torch.Tensor) -> float:
    """ROC-AUC via the trapezoidal rule (pure torch, no sklearn)."""
    labels = labels.detach().long().cpu()
    probs = probs.detach().float().cpu()
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = torch.argsort(probs, descending=True)
    ls = labels[order].float()
    tps = ls.cumsum(0)
    fps = (1.0 - ls).cumsum(0)
    tpr = torch.cat([torch.zeros(1), tps / n_pos])
    fpr = torch.cat([torch.zeros(1), fps / n_neg])
    return float(torch.clamp(torch.trapezoid(tpr, fpr), 0.0, 1.0))


def classification_metrics(
    logits: torch.Tensor, labels: torch.Tensor, threshold: float,
) -> Dict[str, float]:
    """Precision / recall / F1 at an EXPLICIT threshold on sigmoid(logits)."""
    probs = torch.sigmoid(logits.detach().float().cpu())
    labels = labels.detach().long().cpu()
    preds = (probs >= threshold).long()
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
    }


def threshold_sweep(
    logits: torch.Tensor, labels: torch.Tensor, n_steps: int = 99,
) -> Dict[str, object]:
    """Sweep the decision threshold; return the F1 curve and the best point."""
    thresholds = torch.linspace(0.01, 0.99, n_steps)
    f1_curve: List[float] = [
        classification_metrics(logits, labels, float(t))["f1"] for t in thresholds
    ]
    f1_t = torch.tensor(f1_curve)
    best_i = int(torch.argmax(f1_t))
    return {
        "thresholds": [round(float(t), 4) for t in thresholds],
        "f1_curve": [round(v, 4) for v in f1_curve],
        "best_threshold": float(thresholds[best_i]),
        "best_f1": float(f1_t[best_i]),
        "f1_at_0.5": classification_metrics(logits, labels, 0.5)["f1"],
    }


def expected_calibration_error(
    probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10,
) -> Dict[str, object]:
    """ECE over equal-width probability bins."""
    probs = probs.detach().float().cpu()
    labels = labels.detach().float().cpu()
    n = probs.numel()
    if n == 0:
        return {"ece": 0.0, "bins": []}
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows: List[Dict[str, object]] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        in_bin = (probs > lo) & (probs <= hi)
        if i == 0:
            in_bin = in_bin | (probs == lo)
        cnt = int(in_bin.sum())
        if cnt == 0:
            rows.append({"bin": [round(lo, 2), round(hi, 2)], "count": 0,
                         "confidence": None, "accuracy": None})
            continue
        conf = float(probs[in_bin].mean())
        acc = float(labels[in_bin].mean())
        ece += (cnt / n) * abs(conf - acc)
        rows.append({"bin": [round(lo, 2), round(hi, 2)], "count": cnt,
                     "confidence": round(conf, 4), "accuracy": round(acc, 4)})
    return {"ece": round(ece, 4), "bins": rows}


# ---------------------------------------------------------------------------
# RCA localization metrics (per-graph)
# ---------------------------------------------------------------------------

def rca_metrics(per_graph: Sequence[GraphScores]) -> Dict[str, object]:
    """Root-cause localization metrics over FAULTED graphs only.

    For each faulted graph (exactly one candidate node with label==1) the
    rank is the 1-indexed position of the true RC node among candidate
    nodes sorted by descending logit. Healthy graphs are excluded.
    """
    ranks: List[int] = []
    for logits, labels in per_graph:
        if labels.numel() == 0 or int(labels.sum()) == 0:
            continue  # healthy graph — excluded from RCA
        true_idx = int(torch.argmax(labels))  # the single RC node
        order = torch.argsort(logits, descending=True)
        rank = int((order == true_idx).nonzero(as_tuple=True)[0].item()) + 1
        ranks.append(rank)

    n = len(ranks)
    if n == 0:
        return {"n_faulted_graphs": 0, "top1": None, "hits@1": None,
                "hits@3": None, "hits@5": None, "mrr": None, "rank_histogram": {}}
    r = torch.tensor(ranks, dtype=torch.float32)
    hist: Dict[int, int] = {}
    for rk in ranks:
        hist[rk] = hist.get(rk, 0) + 1
    return {
        "n_faulted_graphs": n,
        "top1": float((r == 1).float().mean()),
        "hits@1": float((r <= 1).float().mean()),
        "hits@3": float((r <= 3).float().mean()),
        "hits@5": float((r <= 5).float().mean()),
        "mrr": float((1.0 / r).mean()),
        "rank_histogram": {int(k): hist[k] for k in sorted(hist)},
    }


# ---------------------------------------------------------------------------
# Aggregated report
# ---------------------------------------------------------------------------

def full_report(per_graph: Sequence[GraphScores]) -> Dict[str, object]:
    """Aggregate every metric from a list of per-graph candidate scores."""
    per_graph = list(per_graph)
    if not per_graph:
        raise ValueError("full_report received an empty graph list")
    all_logits = torch.cat([g[0] for g in per_graph])
    all_labels = torch.cat([g[1] for g in per_graph])
    probs = torch.sigmoid(all_logits)
    sweep = threshold_sweep(all_logits, all_labels)
    best_t = float(sweep["best_threshold"])
    return {
        "n_graphs": len(per_graph),
        "n_faulted_graphs": sum(1 for g in per_graph if int(g[1].sum()) > 0),
        "n_candidate_nodes": int(all_logits.numel()),
        "n_positive_nodes": int(all_labels.sum()),
        "roc_auc": roc_auc(all_labels, probs),
        "f1_at_0.5": sweep["f1_at_0.5"],
        "f1_at_best_threshold": sweep["best_f1"],
        "best_threshold": best_t,
        "metrics_at_best_threshold": classification_metrics(all_logits, all_labels, best_t),
        "threshold_sweep": sweep,
        "rca": rca_metrics(per_graph),
        "calibration": expected_calibration_error(probs, all_labels),
    }


def per_type_report(
    per_graph_typed: Sequence[Dict[str, GraphScores]],
) -> Dict[str, Dict[str, object]]:
    """Per-node-type breakdown. Input: list of {node_type: (logits, labels)}."""
    out: Dict[str, Dict[str, object]] = {}
    for nt in CANDIDATE_TYPES:
        graphs = [g[nt] for g in per_graph_typed if nt in g]
        if not graphs:
            continue
        logits = torch.cat([g[0] for g in graphs])
        labels = torch.cat([g[1] for g in graphs])
        if labels.numel() == 0:
            continue
        sweep = threshold_sweep(logits, labels)
        out[nt] = {
            "roc_auc": roc_auc(labels, torch.sigmoid(logits)),
            "f1_at_0.5": sweep["f1_at_0.5"],
            "f1_at_best_threshold": sweep["best_f1"],
            "best_threshold": float(sweep["best_threshold"]),
            "rca": rca_metrics(graphs),
        }
    return out


# ---------------------------------------------------------------------------
# Model scoring (per-graph, batch-invariant)
# ---------------------------------------------------------------------------

def score_loader(model, loader, edge_types, device) -> List[Dict[str, GraphScores]]:
    """Forward every graph individually and return one
    {candidate_type: (logits, labels)} dict per graph.

    Each graph is forwarded alone (via batch.to_data_list()), so the result
    is independent of the loader's batch size — this is the fix for the
    per-batch RCA bug (finding F1).
    """
    model.eval()
    et_set = set(edge_types)
    typed: List[Dict[str, GraphScores]] = []
    with torch.no_grad():
        for batch in loader:
            for data in batch.to_data_list():
                data = data.to(device)
                x = {nt: data[nt].x.float() for nt in data.node_types}
                ei = {et: data[et].edge_index
                      for et in data.edge_types if et in et_set}
                ea = {et: data[et].edge_attr.float()
                      for et in data.edge_types if et in et_set}
                logits = model(x, ei, ea)
                row: Dict[str, GraphScores] = {}
                for nt in CANDIDATE_TYPES:
                    node = data[nt] if nt in data.node_types else None
                    if nt in logits and node is not None \
                            and hasattr(node, "y") and node.y is not None:
                        row[nt] = (
                            logits[nt].detach().reshape(-1).float().cpu(),
                            node.y.detach().reshape(-1).long().cpu(),
                        )
                typed.append(row)
    return typed


def typed_to_flat(typed: List[Dict[str, GraphScores]]) -> List[GraphScores]:
    """Concatenate each graph's per-type scores into flat candidate scores."""
    flat: List[GraphScores] = []
    for row in typed:
        ls, ys = [], []
        for nt in CANDIDATE_TYPES:
            if nt in row:
                ls.append(row[nt][0])
                ys.append(row[nt][1])
        if ls:
            flat.append((torch.cat(ls), torch.cat(ys)))
        else:
            flat.append((torch.empty(0), torch.empty(0, dtype=torch.long)))
    return flat


def summary_line(report: Dict[str, object]) -> str:
    """One-line human summary of a full_report() result."""
    rca = report["rca"]
    top1 = rca["top1"]
    return (
        f"AUC={report['roc_auc']:.4f}  "
        f"F1@0.5={report['f1_at_0.5']:.4f}  "
        f"F1@best={report['f1_at_best_threshold']:.4f} (t={report['best_threshold']:.2f})  "
        f"RCA-Top1={'N/A' if top1 is None else f'{top1:.4f}'}  "
        f"MRR={'N/A' if rca['mrr'] is None else f'{rca['mrr']:.4f}'}"
    )
