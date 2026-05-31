"""
Unit tests for training_pipeline/diagnostics.py and the corrected evaluator.

Run directly:   python tests/test_diagnostics.py
Or with pytest: python -m pytest tests/test_diagnostics.py -v

test_batch_invariance is the permanent regression guard for the per-batch
RCA bug (finding F1): if it ever fails, per-graph unbatching has regressed.
"""

import os
import sys
import warnings

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_pipeline.diagnostics import (  # noqa: E402
    full_report, rca_metrics, score_loader, typed_to_flat,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _graph(n, rc_idx, rc_logit=10.0, other_logit=-10.0):
    """Build one (logits, labels) candidate-score pair for a synthetic graph."""
    logits = torch.full((n,), float(other_logit))
    labels = torch.zeros(n, dtype=torch.long)
    if rc_idx is not None:
        logits[rc_idx] = float(rc_logit)
        labels[rc_idx] = 1
    return (logits, labels)


def test_perfect_predictor():
    """RC always gets the highest logit -> every metric is perfect."""
    pg = [_graph(20, (i * 3) % 20) for i in range(5)]      # 5 faulted
    pg += [_graph(20, None) for _ in range(3)]             # 3 healthy
    rep = full_report(pg)
    assert abs(rep["roc_auc"] - 1.0) < 1e-6, rep["roc_auc"]
    assert rep["rca"]["top1"] == 1.0, rep["rca"]
    assert rep["rca"]["hits@1"] == 1.0
    assert rep["rca"]["mrr"] == 1.0
    assert abs(rep["f1_at_best_threshold"] - 1.0) < 1e-9, rep["f1_at_best_threshold"]


def test_random_predictor():
    """Uncorrelated logits -> AUC ~ 0.5, RCA Top-1 ~ 1/n_candidates."""
    torch.manual_seed(0)
    n_cand, n_graphs = 50, 400
    pg = []
    for _ in range(n_graphs):
        logits = torch.randn(n_cand)
        labels = torch.zeros(n_cand, dtype=torch.long)
        labels[int(torch.randint(0, n_cand, (1,)))] = 1
        pg.append((logits, labels))
    rep = full_report(pg)
    assert abs(rep["roc_auc"] - 0.5) < 0.05, rep["roc_auc"]
    assert abs(rep["rca"]["top1"] - 1.0 / n_cand) < 0.03, rep["rca"]["top1"]


def test_known_rank():
    """RC placed at rank 3 (two nodes outrank it) -> rank/hits are exact."""
    logits = torch.zeros(10)
    logits[4] = 5.0   # the RC node
    logits[1] = 9.0   # outranks RC
    logits[7] = 7.0   # outranks RC
    labels = torch.zeros(10, dtype=torch.long)
    labels[4] = 1
    rep = rca_metrics([(logits, labels)])
    assert rep["rank_histogram"] == {3: 1}, rep["rank_histogram"]
    assert rep["hits@1"] == 0.0
    assert rep["hits@3"] == 1.0
    assert rep["top1"] == 0.0


def test_batch_invariance():
    """Same graphs scored at batch_size 1/4/16 must yield identical metrics."""
    ckpt = os.path.join(_ROOT, "checkpoints", "baseline_model.pt")
    if not os.path.exists(ckpt):
        print("SKIP test_batch_invariance: checkpoint missing")
        return
    from torch_geometric.loader import DataLoader
    from training_pipeline.train import RCADataset
    from evaluate_pipeline import load_model

    device = torch.device("cpu")
    model, edge_types, _ = load_model(ckpt, device)
    idx = list(range(24))
    reports = []
    for bs in (1, 4, 16):
        loader = DataLoader(RCADataset(idx), batch_size=bs, shuffle=False)
        typed = score_loader(model, loader, edge_types, device)
        reports.append(full_report(typed_to_flat(typed)))
    base = reports[0]
    for r in reports[1:]:
        assert abs(r["roc_auc"] - base["roc_auc"]) < 1e-4, "roc_auc not batch-invariant"
        assert abs(r["f1_at_best_threshold"] - base["f1_at_best_threshold"]) < 1e-4, \
            "f1 not batch-invariant"
        assert r["rca"]["rank_histogram"] == base["rca"]["rank_histogram"], \
            "rank histogram not batch-invariant"


ALL_TESTS = [test_perfect_predictor, test_random_predictor,
             test_known_rank, test_batch_invariance]

if __name__ == "__main__":
    failed = 0
    for t in ALL_TESTS:
        try:
            t()
            print(f"PASS   {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL   {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - failed}/{len(ALL_TESTS)} tests passed")
    sys.exit(1 if failed else 0)
