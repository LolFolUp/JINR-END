"""
Full system integration test: GNN inference -> RAG -> Mistral -> Firewall.

Exercises the complete pipeline with a real dataset sample and the live
Mistral model (via Ollama). Falls back to rule-based if Ollama is offline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training_pipeline.diagnostics import CANDIDATE_TYPES
from training_pipeline.train import GATv2Hetero, RCADataset, SEED, TRAIN_RATIO, VAL_RATIO
from remediation.pipeline import run_pipeline, remediation_report_payload
from remediation.run import format_report

CKPT_PATH     = ROOT / "checkpoints" / "best_model.pt"
DATASET_DIR   = ROOT / "dataset" / "raw"
INFERENCE_OUT = ROOT / "artifacts" / "inference_sample.json"
REPORT_OUT    = ROOT / "artifacts" / "remediation_report.json"
NUM_TOTAL     = 1000


def _load_model(ckpt: dict) -> GATv2Hetero:
    model = GATv2Hetero(
        node_dims=ckpt["node_dims"],
        edge_types=[tuple(e) for e in ckpt["edge_types"]],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _test_split_indices() -> list[int]:
    rng = torch.Generator()
    rng.manual_seed(SEED)
    perm = torch.randperm(NUM_TOTAL, generator=rng).tolist()
    n_train = int(NUM_TOTAL * TRAIN_RATIO)
    n_val   = int(NUM_TOTAL * VAL_RATIO)
    return perm[n_train + n_val:]


def _pick_faulted_graph(indices: list[int]) -> tuple[int, object]:
    """Return the first test-split graph that has exactly one RC node."""
    for idx in indices:
        data = torch.load(DATASET_DIR / f"data_{idx}.pt", weights_only=False)
        for nt in CANDIDATE_TYPES:
            if nt in data.node_types and hasattr(data[nt], "y") and data[nt].y is not None:
                if int(data[nt].y.sum()) == 1:
                    return idx, data
    raise RuntimeError("No faulted graph found in test split")


def _run_gnn_inference(model: GATv2Hetero, data, ckpt: dict) -> dict:
    """Forward one graph and return an inference_sample dict."""
    et_set = {tuple(e) for e in ckpt["edge_types"]}
    x_dict  = {nt: data[nt].x.float() for nt in data.node_types}
    ei_dict = {et: data[et].edge_index for et in data.edge_types if et in et_set}
    ea_dict = {et: data[et].edge_attr.float() for et in data.edge_types if et in et_set}

    with torch.no_grad():
        logits = model(x_dict, ei_dict, ea_dict)

    # Find RC node: argmax logit among candidate types
    best_nt, best_idx, best_logit = None, -1, float("-inf")
    for nt in CANDIDATE_TYPES:
        if nt not in logits:
            continue
        lg = logits[nt].reshape(-1)
        val, idx = lg.max(dim=0)
        if float(val) > best_logit:
            best_logit = float(val)
            best_nt    = nt
            best_idx   = int(idx)

    # Determine fault_type from RC node type
    fault_map = {"switch": "network_congestion", "hdd": "hdd_degradation", "ram": "ram_leak"}
    fault_type = fault_map.get(best_nt, "unknown")

    # Build top-5 candidate list across all candidate types
    top5 = []
    for nt in CANDIDATE_TYPES:
        if nt not in logits:
            continue
        lg = logits[nt].reshape(-1)
        scores = torch.sigmoid(lg)
        for i, s in enumerate(scores.tolist()):
            top5.append({"type": nt, "id": f"{nt.upper()}-{i}", "score": round(s, 4)})
    top5 = sorted(top5, key=lambda x: x["score"], reverse=True)[:5]

    # Count victim nodes (nodes with y==1 but not the RC node itself)
    victim_nodes = []
    for nt in data.node_types:
        if not hasattr(data[nt], "y") or data[nt].y is None:
            continue
        y = data[nt].y.reshape(-1)
        if nt in CANDIDATE_TYPES:
            # RC nodes: the one with y==1 is the RC, skip it
            continue
        for i, label in enumerate(y.tolist()):
            if label == 1:
                victim_nodes.append({"id": f"{nt}-{i}", "type": nt})

    rc_score = float(torch.sigmoid(torch.tensor(best_logit)))

    return {
        "graph_id":         int(data.graph_id) if hasattr(data, "graph_id") else 0,
        "fault_type":       fault_type,
        "rc_node":          {"type": best_nt, "id": f"{best_nt.upper()}-{best_idx}", "host_id": best_idx},
        "rc_logit":         round(best_logit, 4),
        "rc_score":         round(rc_score, 6),
        "rc_rank":          1,
        "top5_candidates":  top5,
        "victim_nodes":     victim_nodes,
        "confidence":       round(rc_score, 6),
        "gnn_inference_ms": 0,
        "dataset_version":  ckpt.get("dataset_version", "v3.0.0"),
        "checkpoint_epoch": ckpt.get("epoch", 14),
    }


def test_full_system_integration():
    """
    End-to-end system test:
      GNN load -> dataset sample -> inference -> RAG retrieval -> Mistral -> firewall -> report
    """
    print("\n" + "=" * 60)
    print("FULL SYSTEM INTEGRATION TEST")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load GNN checkpoint
    # ------------------------------------------------------------------
    assert CKPT_PATH.exists(), f"Checkpoint not found: {CKPT_PATH}"
    ckpt = torch.load(CKPT_PATH, weights_only=False, map_location="cpu")
    model = _load_model(ckpt)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[GNN] Loaded: epoch={ckpt['epoch']}, dataset={ckpt['dataset_version']}, params={n_params:,}")

    assert n_params > 0, "Model has no parameters"

    # ------------------------------------------------------------------
    # 2. Load one test-split graph
    # ------------------------------------------------------------------
    assert DATASET_DIR.exists(), f"Dataset not found: {DATASET_DIR}"
    test_indices = _test_split_indices()
    assert len(test_indices) > 0, "Test split is empty"

    graph_idx, data = _pick_faulted_graph(test_indices)
    print(f"[GNN] Test graph: data_{graph_idx}.pt  node_types={list(data.node_types)}")

    # ------------------------------------------------------------------
    # 3. Run GNN inference
    # ------------------------------------------------------------------
    inference = _run_gnn_inference(model, data, ckpt)

    rc = inference["rc_node"]
    print(f"[GNN] RC prediction: {rc['type'].upper()}-{rc['host_id']}  "
          f"logit={inference['rc_logit']}  score={inference['rc_score']:.4f}  "
          f"fault={inference['fault_type']}")

    assert inference["rc_node"]["type"] in CANDIDATE_TYPES, "RC node type invalid"
    assert 0.0 <= inference["confidence"] <= 1.0, "Confidence out of range"
    assert inference["fault_type"] in {"network_congestion", "hdd_degradation", "ram_leak"}

    # Save inference output
    INFERENCE_OUT.parent.mkdir(parents=True, exist_ok=True)
    INFERENCE_OUT.write_text(json.dumps(inference, indent=2), encoding="utf-8")
    print(f"[GNN] Inference saved to: {INFERENCE_OUT}")

    # ------------------------------------------------------------------
    # 4 + 5 + 6. RAG retrieval -> Mistral -> Firewall
    # ------------------------------------------------------------------
    print("\n[RAG/LLM] Running remediation pipeline...")
    playbook, metadata = run_pipeline(inference)

    rag_chunks = metadata["knowledge"]["sop_chunks_retrieved"]
    retrieval   = metadata["knowledge"]["retrieval_method"]
    llm_backend = metadata.get("llm_backend", "unknown")
    fw_status   = metadata["firewall_status"]
    ttr         = metadata["ttr_breakdown"]

    print(f"[RAG] SOPs retrieved: {rag_chunks}  method={retrieval}")
    print(f"[LLM] Backend: {llm_backend}  generation={ttr['llm_generation_ms']}ms")
    print(f"[Firewall] Status: {fw_status}")

    assert rag_chunks > 0, "RAG retrieved 0 SOP chunks"
    assert len(playbook.actions) >= 1, "Playbook has no actions"
    assert fw_status == "PASSED", f"Firewall BLOCKED: {metadata.get('firewall_error')}"

    # ------------------------------------------------------------------
    # 7. Save and validate remediation report as JSON
    # ------------------------------------------------------------------
    payload = remediation_report_payload(playbook, metadata)
    report_json = json.dumps(payload, indent=2, ensure_ascii=False)
    loaded_back = json.loads(report_json)   # validates round-trip

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(report_json, encoding="utf-8")
    print(f"\n[Report] Saved to: {REPORT_OUT}")

    assert loaded_back["firewall_status"] == "PASSED"
    assert loaded_back["actions"], "Report has no actions"
    assert loaded_back["incident"]["severity"] in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}

    # ------------------------------------------------------------------
    # 8. Print human-readable summary
    # ------------------------------------------------------------------
    print("\n" + format_report(playbook, metadata))
    print("\n" + "=" * 60)
    print("INTEGRATION TEST PASSED")
    print("=" * 60)
