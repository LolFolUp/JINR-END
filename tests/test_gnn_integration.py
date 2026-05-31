"""Tests for the GNN -> LLM/RAG end-to-end integration.

Split into two tiers:
  * model-free tests (always run): adapter mapping, prompt construction,
    firewall, and the integration glue under fully offline mock services.
  * checkpoint-gated tests (skipped if the .pt checkpoint / sample are absent,
    e.g. on a fresh clone where binaries were not committed): real GATv2Hetero
    inference on a demo graph.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.demo_gnn_llm_pipeline import MockQdrantStore
from integrations.gnn_to_incident import build_incident_context, gnn_to_inference
from llm.llm_client import LLMClient
from llm.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from rag.history_tickets import HistoryTicketsStore
from rag.redis_context import RedisContextStore
from remediation.firewall import validate_playbook
from remediation.models import RemediationPlaybook
from remediation.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "gnn" / "checkpoints" / "best_v5a_40_screening.pt"
SAMPLE = ROOT / "demo_data" / "gnn_samples" / "data_3.pt"


def _synthetic_gnn_out() -> dict:
    """A hand-built GNN output mimicking engine.run() for a switch root cause."""
    return {
        "incident_id": "graph_3_network_congestion",
        "graph_id": 3,
        "source": "gnn",
        "model": {
            "name": "GATv2Hetero",
            "checkpoint": "best_v5a_40_screening.pt",
            "dataset_version": "v5a_40",
            "val_hit1": 0.8745,
        },
        "rca": {
            "root_cause": {
                "rank": 1, "node_type": "switch", "node_id": 3,
                "node_label": "S3", "score": 0.999, "logit": 21.0,
                "fault_type_hint": "network_congestion",
            },
            "top_k": [
                {"rank": 1, "node_type": "switch", "node_id": 3, "node_label": "S3",
                 "score": 0.999, "logit": 21.0, "fault_type_hint": "network_congestion"},
                {"rank": 2, "node_type": "cpu", "node_id": 13, "node_label": "CPU-13",
                 "score": 0.03, "logit": -3.3, "fault_type_hint": None},
            ],
            "hit_metadata": {"candidate_count": 406,
                             "rc_candidate_types": ["cpu", "gpu", "ram", "hdd", "switch"]},
        },
        "fault_type_hint": {"value": "network_congestion",
                            "provenance": "synthetic_ground_truth", "note": "..."},
        "graph_context": {
            "affected_nodes": [
                {"node_type": "cpu", "node_id": 2, "relation": "connected_to"},
                {"node_type": "gpu", "node_id": 2, "relation": "connected_to"},
                {"node_type": "cpu", "node_id": 8, "relation": "connected_to"},
                {"node_type": "switch", "node_id": 4, "relation": "uplink_to"},
            ],
            "affected_counts": {"cpu": 2, "gpu": 1, "switch": 1},
            "key_metrics": {"switch_packet_loss_percent": 6.04, "switch_latency_ms": 5.24},
            "notes": [],
        },
        "timing": {"gnn_inference_ms": 27},
        "ground_truth": {"rc_node_type": "switch", "rc_node_id": 3,
                         "rc_node_label": "S3", "rc_rank": 1, "predicted_correct": True},
    }


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

def test_adapter_produces_inference_contract():
    inference = gnn_to_inference(_synthetic_gnn_out())
    # required keys for run_pipeline / prompt builder
    for key in ("fault_type", "rc_node", "confidence", "top5_candidates",
                "victim_nodes", "gnn_inference_ms"):
        assert key in inference
    assert inference["fault_type"] == "network_congestion"
    assert inference["rc_node"]["type"] == "switch"
    assert inference["rc_node"]["id"] == "S3"
    assert 0.0 <= inference["confidence"] <= 1.0
    assert inference["top5_candidates"][0]["id"] == "S3"


def test_adapter_dedupes_cpu_gpu_into_single_host():
    inference = gnn_to_inference(_synthetic_gnn_out())
    victims = inference["victim_nodes"]
    # cpu-2 and gpu-2 share index 2 -> one host-002; cpu-8 -> host-008
    host_ids = sorted(v["id"] for v in victims if v["type"] == "host")
    assert host_ids == ["host-002", "host-008"]
    # switch-4 kept as its own type
    assert any(v["type"] == "switch" for v in victims)


def test_incident_context_has_run_id_and_topk():
    ctx = build_incident_context(_synthetic_gnn_out())
    assert ctx["source"] == "gnn"
    assert ctx["run_id"].startswith("graph_3_network_congestion-")
    assert ctx["root_cause"]["node_label"] == "S3"
    assert len(ctx["top3_candidates"]) >= 1


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def test_prompt_includes_rca_topk_fault_and_metrics():
    inference = gnn_to_inference(_synthetic_gnn_out())
    prompt = build_user_prompt(inference, ["SOP: apply QoS"], incident={}, node_context={})
    assert "network_congestion" in prompt
    assert "S3" in prompt                      # root-cause node label
    assert "top5_candidates" in prompt
    assert "key_anomalous_metrics" in prompt
    assert "switch_packet_loss_percent" in prompt


def test_system_prompt_has_grounding_guardrails():
    assert "Never emit free-form shell" in SYSTEM_PROMPT
    assert "Do NOT change the root-cause node" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Firewall
# ---------------------------------------------------------------------------

def test_firewall_blocks_dangerous_action():
    bad = {
        "incident_id": "graph_3_network_congestion",
        "fault_type": "network_congestion",
        "rc_node": "Switch S3",
        "confidence": 0.99,
        "actions": [{"action_id": "DROP_CLUSTER", "target_node": "all",
                     "parameters": {}, "priority": 1, "estimated_ttr_seconds": 1}],
        "explanation": "bad",
    }
    playbook, passed, err = validate_playbook(bad)
    assert passed is False and playbook is None and err is not None


# ---------------------------------------------------------------------------
# End-to-end glue (offline, no services, no checkpoint required)
# ---------------------------------------------------------------------------

def test_end_to_end_mock_runs_without_services():
    inference = gnn_to_inference(_synthetic_gnn_out())
    playbook, metadata = run_pipeline(
        inference,
        llm_client=LLMClient(force_rule_based=True),
        qdrant_store=MockQdrantStore(),
        context_store=RedisContextStore(client=None),
        history_store=HistoryTicketsStore(),
    )
    assert isinstance(playbook, RemediationPlaybook)
    assert playbook.fault_type == "network_congestion"
    assert metadata["firewall_status"] == "PASSED"
    assert len(playbook.actions) >= 1
    assert metadata["ttr_breakdown"]["total_ms"] < 5000
    # GNN inference time threads through to the TTR breakdown
    assert metadata["ttr_breakdown"]["gnn_inference_ms"] == 27


# ---------------------------------------------------------------------------
# Real GNN inference (checkpoint-gated)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (CKPT.exists() and SAMPLE.exists()),
                    reason="GNN checkpoint or demo sample not present")
def test_real_inference_localizes_switch_root_cause():
    from gnn.inference import GNNInferenceEngine

    engine = GNNInferenceEngine(checkpoint_path=CKPT)
    out = engine.run(str(SAMPLE), top_k=5)
    rc = out["rca"]["root_cause"]
    assert rc["node_type"] == "switch"
    assert 0.0 <= rc["score"] <= 1.0
    assert out["rca"]["hit_metadata"]["candidate_count"] == 406
    assert out["ground_truth"]["predicted_correct"] is True
    # structured top-k contract
    assert [c["rank"] for c in out["rca"]["top_k"]] == [1, 2, 3, 4, 5]


@pytest.mark.skipif(not (CKPT.exists() and SAMPLE.exists()),
                    reason="GNN checkpoint or demo sample not present")
def test_real_inference_feeds_pipeline_end_to_end():
    from gnn.inference import GNNInferenceEngine

    engine = GNNInferenceEngine(checkpoint_path=CKPT)
    out = engine.run(str(SAMPLE), top_k=5)
    inference = gnn_to_inference(out)
    playbook, metadata = run_pipeline(
        inference,
        llm_client=LLMClient(force_rule_based=True),
        qdrant_store=MockQdrantStore(),
        context_store=RedisContextStore(client=None),
        history_store=HistoryTicketsStore(),
    )
    assert metadata["firewall_status"] == "PASSED"
    assert playbook.fault_type == "network_congestion"
