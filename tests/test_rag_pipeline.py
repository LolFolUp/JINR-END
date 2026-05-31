"""Unit tests for RAG/LLM remediation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm.llm_client import LLMClient, generate_rule_based_playbook
from rag.history_tickets import HistoryTicketsStore
from rag.qdrant_store import QdrantStore
from rag.redis_context import RedisContextStore
from remediation.firewall import validate_playbook
from remediation.incident_aggregator import IncidentAggregator
from remediation.models import Action, RemediationPlaybook
from remediation.pipeline import remediation_report_payload, run_pipeline

MOCK_NETWORK = {
    "graph_id": 171,
    "fault_type": "network_congestion",
    "rc_node": {"type": "switch", "id": "S2", "host_id": 2},
    "rc_logit": 8.52,
    "rc_rank": 1,
    "top5_candidates": [{"type": "switch", "id": "S2", "score": 0.99}],
    "victim_nodes": [{"id": f"cn-{i}"} for i in range(27)],
    "confidence": 0.9998,
}

MOCK_HDD = {
    "graph_id": 42,
    "fault_type": "hdd_degradation",
    "rc_node": {"type": "hdd", "id": "HDD-7", "host_id": 15},
    "rc_logit": 6.1,
    "rc_rank": 1,
    "top5_candidates": [],
    "victim_nodes": [{"id": "cn-15"}],
    "confidence": 0.91,
}

MOCK_RAM = {
    "graph_id": 88,
    "fault_type": "ram_leak",
    "rc_node": {"type": "ram", "id": "RAM-3", "host_id": 3},
    "rc_logit": 5.4,
    "rc_rank": 1,
    "top5_candidates": [],
    "victim_nodes": [{"id": "cn-3"}, {"id": "cn-4"}],
    "confidence": 0.87,
}


def test_firewall_blocks_disallowed_action():
    with pytest.raises(ValidationError):
        Action.model_validate(
            {
                "action_id": "DELETE_ALL_DATA",
                "target_node": "Switch S2",
                "parameters": {},
                "priority": 1,
                "estimated_ttr_seconds": 10,
            }
        )


def test_firewall_passes_allowed_action():
    action = Action.model_validate(
        {
            "action_id": "ISOLATE_NODE",
            "target_node": "Switch S2",
            "parameters": {"mode": "drain_rack"},
            "priority": 2,
            "estimated_ttr_seconds": 120,
        }
    )
    assert action.action_id == "ISOLATE_NODE"
    assert action.target_node == "Switch S2"


@pytest.mark.parametrize(
    "inference",
    [MOCK_NETWORK, MOCK_HDD, MOCK_RAM],
    ids=["network_congestion", "hdd_degradation", "ram_leak"],
)
def test_pipeline_end_to_end_mock_input(inference):
    client = LLMClient(force_rule_based=True)
    playbook, metadata = run_pipeline(inference, llm_client=client)
    assert isinstance(playbook, RemediationPlaybook)
    assert playbook.fault_type == inference["fault_type"]
    assert len(playbook.actions) >= 1
    assert metadata["firewall_status"] == "PASSED"


def test_rule_based_fallback_notify_operator():
    playbook_dict = generate_rule_based_playbook(MOCK_NETWORK)
    assert playbook_dict.get("rule_based_fallback") is True
    action_ids = [a["action_id"] for a in playbook_dict["actions"]]
    assert "NOTIFY_OPERATOR" in action_ids

    client = LLMClient(force_rule_based=True)
    playbook, meta = run_pipeline(MOCK_NETWORK, llm_client=client)
    assert meta["rule_based_fallback"] is True
    assert any(a.action_id == "NOTIFY_OPERATOR" for a in playbook.actions)


def test_firewall_blocks_invalid_playbook_dict():
    invalid = {
        "incident_id": "graph_1_network_congestion",
        "fault_type": "network_congestion",
        "rc_node": "Switch S1",
        "confidence": 0.9,
        "actions": [
            {
                "action_id": "DROP_CLUSTER",
                "target_node": "all",
                "parameters": {},
                "priority": 1,
                "estimated_ttr_seconds": 1,
            }
        ],
        "explanation": "bad",
    }
    playbook, passed, err = validate_playbook(invalid)
    assert passed is False
    assert playbook is None
    assert err is not None


def test_write_remediation_report_tmp(tmp_path: Path):
    inference_file = tmp_path / "inference_sample.json"
    inference_file.write_text(json.dumps(MOCK_NETWORK), encoding="utf-8")
    out_file = tmp_path / "remediation_report.json"

    from remediation.pipeline import load_inference, playbook_to_dict

    inference = load_inference(inference_file)
    playbook, metadata = run_pipeline(inference, force_rule_based=True)
    payload = {"playbook": playbook_to_dict(playbook), "metadata": metadata}
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert out_file.exists()
    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["playbook"]["incident_id"] == "graph_171_network_congestion"


def test_redis_context_fetch_works_with_fakeredis_when_available():
    client = None
    expected_source = "dict_fallback"
    try:
        import fakeredis

        client = fakeredis.FakeRedis(decode_responses=True)
        expected_source = "redis"
    except Exception:
        pass

    store = RedisContextStore(client=client)
    context, source = store.fetch_context(MOCK_NETWORK["rc_node"])
    assert source == expected_source
    assert context["hostname"] == "leaf-switch-02.govorun.jinr.ru"
    assert context["node_type"] == "switch"


def test_qdrant_retrieval_returns_top_3_sops():
    store = QdrantStore(top_k=3)
    chunks, method = store.retrieve(
        fault_type="network_congestion",
        rc_node_type="switch",
        query="network congestion switch qos top talkers",
        top_k=3,
    )
    assert method in {"qdrant", "bow_fallback"}
    assert len(chunks) == 3
    assert any(chunk["fault_type"] == "network_congestion" for chunk in chunks)


def test_incident_aggregator_assigns_critical_for_many_victims():
    incident = IncidentAggregator().aggregate(MOCK_NETWORK)
    assert incident.severity == "CRITICAL"
    assert incident.affected_nodes_count == 27
    assert incident.aggregated_from_alerts >= 89


def test_history_tickets_retrieved_for_matching_fault_type():
    tickets = HistoryTicketsStore().find_similar("hdd_degradation")
    assert tickets
    assert all(ticket["fault_type"] == "hdd_degradation" for ticket in tickets)
    assert tickets[0]["ticket_id"].startswith("INC-")


def test_ttr_measured_and_under_5000ms():
    client = LLMClient(force_rule_based=True)
    _, metadata = run_pipeline(MOCK_NETWORK, llm_client=client)
    ttr = metadata["ttr_breakdown"]
    assert {"gnn_inference_ms", "rag_retrieval_ms", "llm_generation_ms", "validation_ms", "total_ms"} <= set(ttr)
    assert ttr["total_ms"] < 5000


def test_full_pipeline_with_all_components_produces_valid_report():
    client = LLMClient(force_rule_based=True)
    playbook, metadata = run_pipeline(MOCK_NETWORK, llm_client=client)
    report = remediation_report_payload(playbook, metadata)
    assert report["incident"]["id"] == "INC-2026-graph171"
    assert report["incident"]["severity"] == "CRITICAL"
    assert report["context"]["rc_hostname"] == "leaf-switch-02.govorun.jinr.ru"
    assert report["knowledge"]["sop_chunks_retrieved"] == 3
    assert report["knowledge"]["similar_incidents"]
    assert report["actions"]
    assert report["firewall_status"] == "PASSED"
