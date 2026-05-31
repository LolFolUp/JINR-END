"""
Unit tests for the Phase 5 / E1 listwise root-cause loss.

Run directly:   python tests/test_listwise.py
Or with pytest: python -m pytest tests/test_listwise.py -v

Covers the robustness constraints for E1: non-empty candidate validation,
malformed / missing-candidate-type graphs, single-candidate softmax, and
NaN-safety.
"""

import os
import sys
import warnings

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_pipeline.train import (  # noqa: E402
    _listwise_graph_loss, compute_listwise_loss,
)


def test_listwise_correct_rc_low_loss():
    """RC node has the highest logit -> near-zero loss, gradient flows."""
    logits = torch.tensor([0.0, 5.0, 0.0], requires_grad=True)
    labels = torch.tensor([0, 1, 0])
    loss = _listwise_graph_loss(logits, labels)
    assert loss is not None
    assert loss.item() < 0.1
    loss.backward()
    assert logits.grad is not None


def test_listwise_wrong_rc_high_loss():
    """RC node has the lowest logit -> large loss."""
    logits = torch.tensor([-5.0, 5.0, 5.0])
    labels = torch.tensor([1, 0, 0])
    loss = _listwise_graph_loss(logits, labels)
    assert loss is not None and loss.item() > 5.0


def test_listwise_single_candidate():
    """One candidate -> degenerate softmax -> loss 0, finite (no NaN)."""
    loss = _listwise_graph_loss(torch.tensor([3.0]), torch.tensor([1]))
    assert loss is not None
    assert torch.isfinite(loss) and abs(loss.item()) < 1e-6


def test_listwise_healthy_graph_returns_none():
    """No root cause in the candidate set -> None (contributes nothing)."""
    assert _listwise_graph_loss(torch.tensor([1.0, 2.0, 3.0]),
                                torch.tensor([0, 0, 0])) is None


def test_listwise_empty_returns_none():
    """No candidate nodes at all -> None, no crash."""
    assert _listwise_graph_loss(torch.empty(0),
                                torch.empty(0, dtype=torch.long)) is None


def _make_graph(rc):
    """A hetero graph with all candidate types; rc in {'hdd','switch',None}."""
    from torch_geometric.data import HeteroData
    d = HeteroData()
    d["hdd"].x    = torch.randn(5, 2); d["hdd"].y    = torch.zeros(5, dtype=torch.long)
    d["switch"].x = torch.randn(3, 2); d["switch"].y = torch.zeros(3, dtype=torch.long)
    d["ram"].x    = torch.randn(4, 2); d["ram"].y    = torch.zeros(4, dtype=torch.long)
    if rc == "hdd":
        d["hdd"].y[0] = 1
    elif rc == "switch":
        d["switch"].y[1] = 1
    return d


def test_compute_listwise_batch_mixed():
    """A batch with hdd-faulted, switch-faulted, and healthy graphs:
    finite loss, gradient flows, healthy graph silently skipped."""
    from torch_geometric.data import Batch
    batch = Batch.from_data_list(
        [_make_graph("hdd"), _make_graph("switch"), _make_graph(None)]
    )
    logits = {nt: torch.randn(batch[nt].x.size(0), requires_grad=True)
              for nt in ("hdd", "switch", "ram")}
    loss = compute_listwise_loss(logits, batch, torch.device("cpu"))
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()


def test_compute_listwise_missing_logits_key():
    """logits dict missing the 'ram' key entirely -> graceful, no KeyError."""
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([_make_graph("hdd")])
    logits = {"hdd":    torch.randn(batch["hdd"].x.size(0), requires_grad=True),
              "switch": torch.randn(batch["switch"].x.size(0), requires_grad=True)}
    loss = compute_listwise_loss(logits, batch, torch.device("cpu"))
    assert torch.isfinite(loss)


def test_compute_listwise_all_healthy():
    """A batch with no faulted graph -> zero loss, no NaN, backward works."""
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([_make_graph(None), _make_graph(None)])
    logits = {nt: torch.randn(batch[nt].x.size(0), requires_grad=True)
              for nt in ("hdd", "switch", "ram")}
    loss = compute_listwise_loss(logits, batch, torch.device("cpu"))
    assert torch.isfinite(loss) and loss.item() == 0.0
    loss.backward()


ALL_TESTS = [
    test_listwise_correct_rc_low_loss,
    test_listwise_wrong_rc_high_loss,
    test_listwise_single_candidate,
    test_listwise_healthy_graph_returns_none,
    test_listwise_empty_returns_none,
    test_compute_listwise_batch_mixed,
    test_compute_listwise_missing_logits_key,
    test_compute_listwise_all_healthy,
]

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
