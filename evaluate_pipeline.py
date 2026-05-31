"""
evaluate_pipeline.py

Corrected, candidate-restricted, per-graph evaluator for RCA checkpoints.

Replaces the broken per-batch compute_rca_accuracy path in
train.py.validate(). Every metric comes from training_pipeline/diagnostics.py.
Scoring is per-graph (batch.to_data_list()), so results are batch-invariant
by construction.

Usage:
    python evaluate_pipeline.py [--checkpoint PATH] [--split val|test|both]
                                [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch

from training_pipeline import diagnostics
from training_pipeline.diagnostics import score_loader, typed_to_flat
from training_pipeline.train import (
    DROPOUT, HEADS, HIDDEN_DIM, METADATA_PATH, build_dataloaders, GATv2Hetero,
)

_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(_ROOT, "checkpoints", "baseline_model.pt")
DEFAULT_OUT = os.path.join(_ROOT, "artifacts", "phase1_diag.json")


def load_model(ckpt_path: str, device: torch.device):
    """Reconstruct a GATv2Hetero model from a checkpoint's stored config."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    node_dims = state["node_dims"]
    edge_types = [tuple(et) for et in state["edge_types"]]
    model = GATv2Hetero(
        node_dims=node_dims, edge_types=edge_types,
        hidden_dim=HIDDEN_DIM, heads=HEADS, dropout=DROPOUT,
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, edge_types, state


def evaluate_checkpoint(checkpoint: str, split: str, output: str) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, edge_types, state = load_model(checkpoint, device)

    with open(METADATA_PATH, encoding="utf-8") as f:
        num_total = json.load(f)["dataset_size"]
    _, val_loader, test_loader = build_dataloaders(num_total)
    loaders = {"val": val_loader, "test": test_loader}
    targets = ["val", "test"] if split == "both" else [split]

    report: Dict[str, object] = {
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_epoch": state.get("epoch"),
        "device": str(device),
        "splits": {},
    }
    print(f"Checkpoint: {checkpoint}  (epoch {state.get('epoch')})")
    for name in targets:
        typed = score_loader(model, loaders[name], edge_types, device)
        rep = diagnostics.full_report(typed_to_flat(typed))
        rep["per_type"] = diagnostics.per_type_report(typed)
        report["splits"][name] = rep
        print(f"  [{name:<4}] {diagnostics.summary_line(rep)}")
        print(f"         rank histogram: {rep['rca']['rank_histogram']}")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report written -> {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrected RCA checkpoint evaluator")
    parser.add_argument("--checkpoint", "-c", default=DEFAULT_CKPT)
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    parser.add_argument("--output", "-o", default=DEFAULT_OUT)
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.split, args.output)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
