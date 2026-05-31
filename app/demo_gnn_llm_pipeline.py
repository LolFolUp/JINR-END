"""End-to-end demo: HeteroData graph -> GNN RCA -> incident -> RAG -> LLM -> firewall.

Pipeline stages
---------------
    sample .pt graph
      -> GNN inference            (gnn.inference.GNNInferenceEngine)
      -> RCA top-k + context      (integrations.gnn_to_incident)
      -> IncidentAggregator       (remediation.incident_aggregator, inside run_pipeline)
      -> RAG retrieval            (rag.qdrant_store / mock)
      -> LLM playbook             (llm.llm_client: Ollama/Mistral or rule-based)
      -> Semantic Firewall        (remediation.firewall / ActionDSL)
      -> final engineering report

Modes
-----
    (default)      real mode: tries Ollama for the LLM and in-process Qdrant/
                   embeddings for RAG; both auto-fall-back if unavailable.
    --mock-llm     force the deterministic rule-based playbook (no Ollama).
    --mock-rag     use a dependency-free static SOP/context stub (no embedder).
    --mock         shorthand for --mock-llm --mock-rag (fully offline).

Examples
--------
    python -m app.demo_gnn_llm_pipeline --sample demo_data/gnn_samples/data_3.pt --mock
    python -m app.demo_gnn_llm_pipeline --sample demo_data/gnn_samples/data_11.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gnn.inference import DEFAULT_CHECKPOINT, DEFAULT_METADATA, GNNInferenceEngine
from integrations.gnn_to_incident import build_incident_context, gnn_to_inference
from llm.llm_client import LLMClient
from rag.history_tickets import HistoryTicketsStore
from rag.qdrant_store import QdrantStore
from rag.redis_context import RedisContextStore
from remediation.pipeline import remediation_report_payload, run_pipeline
from remediation.run import format_report

DEFAULT_SAMPLE = ROOT / "demo_data" / "gnn_samples" / "data_3.pt"
DEFAULT_TRACE = ROOT / "artifacts" / "gnn_llm_demo_trace.json"

_BAR = "=" * 78


class MockQdrantStore:
    """Dependency-free SOP retrieval stub (no embedder, no vector DB).

    Used by ``--mock-rag`` so the demo runs with zero external services and no
    heavy optional dependencies.
    """

    _SOPS: Dict[str, List[Dict[str, str]]] = {
        "network_congestion": [
            {"title": "Switch congestion triage", "fault_type": "network_congestion",
             "text": "Check port utilization and packet loss; apply QoS rate limiting to top talkers; drain the rack if MPI all-to-all saturates uplinks."},
            {"title": "QoS application procedure", "fault_type": "network_congestion",
             "text": "Apply rate_limit_top_talkers policy on the affected leaf switch; verify packet loss returns below 0.1%."},
        ],
        "hdd_degradation": [
            {"title": "Degraded OST handling", "fault_type": "hdd_degradation",
             "text": "Inspect SMART reallocated sectors and IO latency p95; migrate datasets off the degraded OST and schedule disk replacement."},
        ],
        "ram_leak": [
            {"title": "RAM leak containment", "fault_type": "ram_leak",
             "text": "Inspect RSS of top processes and OOM events; checkpoint-restart or migrate the leaking job under a cgroup memory cap."},
        ],
    }
    _GENERAL = {"title": "HPC incident communication procedure", "fault_type": "general",
                "text": "Open an incident ticket, notify the NOC, record impacted jobs, preserve logs, avoid destructive remediation while workloads are active."}

    def retrieve(self, *, fault_type: str, rc_node_type: str, query: str,
                 top_k: int = 3) -> Tuple[List[Dict[str, Any]], str]:
        chunks = list(self._SOPS.get(fault_type, []))
        chunks.append(self._GENERAL)
        out = [{"chunk_id": f"mock-{i}", "score": 1.0 - i * 0.1, **c}
               for i, c in enumerate(chunks[:top_k])]
        return out, "mock_static"


def _print_header(title: str) -> None:
    print(f"\n{_BAR}\n{title}\n{_BAR}")


def _print_gnn_stage(gnn_out: Dict[str, Any]) -> None:
    _print_header("STAGE 1 - GNN ROOT-CAUSE ANALYSIS (GATv2Hetero)")
    model = gnn_out["model"]
    print(f"model={model['name']}  checkpoint={model['checkpoint']}  "
          f"dataset={model['dataset_version']}  val_Hit@1={model['val_hit1']}")
    print(f"candidate pool: {gnn_out['rca']['hit_metadata']['candidate_count']} RC candidates "
          f"({', '.join(gnn_out['rca']['hit_metadata']['rc_candidate_types'])})")
    print("\nTop-k root-cause candidates (score = sigmoid(logit)):")
    for c in gnn_out["rca"]["top_k"]:
        marker = "  <== predicted root cause" if c["rank"] == 1 else ""
        print(f"  #{c['rank']}  {c['node_label']:<8} ({c['node_type']:<6}) "
              f"score={c['score']:.4f}{marker}")
    fh = gnn_out["fault_type_hint"]
    print(f"\nfault_type_hint: {fh['value']}  (provenance: {fh['provenance']})")
    km = gnn_out["graph_context"]["key_metrics"]
    if km:
        print("key anomalous metrics of root cause (normalised delta_long, sigma):")
        for feat, val in km.items():
            print(f"    {feat:<34} {val:+.2f}")
    ac = gnn_out["graph_context"]["affected_counts"]
    if ac:
        print(f"affected (topological neighbours): "
              + ", ".join(f"{k}={v}" for k, v in ac.items()))
    gt = gnn_out.get("ground_truth")
    if gt:
        ok = "CORRECT" if gt["predicted_correct"] else "MISS"
        print(f"[synthetic ground truth] true RC = {gt['rc_node_label']} "
              f"({gt['rc_node_type']}), rank={gt['rc_rank']} -> {ok}")
    print(f"gnn_inference: {gnn_out['timing']['gnn_inference_ms']} ms")


def _print_incident_stage(ctx: Dict[str, Any]) -> None:
    _print_header("STAGE 2 - ASSEMBLED INCIDENT CONTEXT (source=gnn)")
    print(f"run_id={ctx['run_id']}  timestamp={ctx['timestamp']}")
    rc = ctx["root_cause"]
    print(f"root cause: {rc['node_label']} ({rc['node_type']}) score={rc['score']:.4f}")
    print("top-3 candidates: " + ", ".join(
        f"{c['node_label']}({c['score']:.3f})" for c in ctx["top3_candidates"]))
    print(f"fault hint: {ctx['fault_type_hint']} ({ctx['fault_type_provenance']})")
    if ctx["affected_counts"]:
        print("affected counts: " + ", ".join(
            f"{k}={v}" for k, v in ctx["affected_counts"].items()))


def _print_rag_stage(metadata: Dict[str, Any]) -> None:
    _print_header("STAGE 3 - RAG RETRIEVAL")
    k = metadata["knowledge"]
    print(f"retrieval_method: {k['retrieval_method']}   "
          f"sop_chunks_retrieved: {k['sop_chunks_retrieved']}")
    for chunk in k.get("sop_chunks", []):
        print(f"  - [{chunk.get('fault_type')}] {chunk.get('title')}")
    sims = k.get("similar_incidents", [])
    if sims:
        print("similar past incidents:")
        for s in sims:
            print(f"  - {s['ticket_id']}: {s['resolution']} "
                  f"({s['resolution_time_minutes']} min)")
    ctx = metadata.get("context", {})
    print(f"technical context: host={ctx.get('rc_hostname')} os={ctx.get('rc_os')} "
          f"(source={ctx.get('context_source')})")


def run_demo(
    sample: str,
    *,
    checkpoint: str,
    metadata_path: str,
    top_k: int,
    mock_llm: bool,
    mock_rag: bool,
    trace_path: Optional[str],
) -> int:
    # ---- Stage 1: GNN inference ------------------------------------------
    engine = GNNInferenceEngine(checkpoint_path=checkpoint, metadata_path=metadata_path)
    gnn_out = engine.run(sample, top_k=top_k)
    _print_gnn_stage(gnn_out)

    # ---- Stage 2: adapt to incident contract -----------------------------
    inference = gnn_to_inference(gnn_out)
    incident_ctx = build_incident_context(gnn_out)
    _print_incident_stage(incident_ctx)

    # ---- Stage 3+4+5: RAG + LLM + firewall via the existing pipeline ------
    qdrant_store = MockQdrantStore() if mock_rag else QdrantStore()
    context_store = RedisContextStore(client=None) if mock_rag else RedisContextStore()
    history_store = HistoryTicketsStore()
    llm_client = LLMClient(force_rule_based=mock_llm)

    playbook, metadata = run_pipeline(
        inference,
        llm_client=llm_client,
        qdrant_store=qdrant_store,
        context_store=context_store,
        history_store=history_store,
    )
    _print_rag_stage(metadata)

    _print_header("STAGE 4 - LLM REMEDIATION PLAYBOOK + SEMANTIC FIREWALL")
    print(f"LLM backend: {metadata.get('llm_backend')}   "
          f"firewall: {metadata.get('firewall_status')}")
    print()
    print(format_report(playbook, metadata))

    # ---- Persist full trace ----------------------------------------------
    if trace_path:
        trace = {
            "mode": {"mock_llm": mock_llm, "mock_rag": mock_rag},
            "sample": sample,
            "gnn_output": gnn_out,
            "incident_context": incident_ctx,
            "inference_contract": inference,
            "report": remediation_report_payload(playbook, metadata),
        }
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        Path(trace_path).write_text(
            json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nFull end-to-end trace saved -> {trace_path}")

    # Demo succeeds when the firewall produced a valid, validated playbook.
    return 0 if metadata.get("firewall_status") == "PASSED" else 2


def main() -> int:
    p = argparse.ArgumentParser(description="GNN -> RAG/LLM end-to-end demo")
    p.add_argument("--sample", default=str(DEFAULT_SAMPLE), help="Path to a .pt HeteroData graph")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--metadata", default=str(DEFAULT_METADATA))
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--mock-llm", action="store_true", help="Force rule-based playbook (no Ollama)")
    p.add_argument("--mock-rag", action="store_true", help="Static SOP/context stub (no embedder)")
    p.add_argument("--mock", action="store_true", help="Shorthand for --mock-llm --mock-rag")
    p.add_argument("--trace", default=str(DEFAULT_TRACE), help="Where to write the JSON trace ('' to skip)")
    args = p.parse_args()

    mock_llm = args.mock_llm or args.mock
    mock_rag = args.mock_rag or args.mock

    if not Path(args.sample).exists():
        print(f"ERROR: sample graph not found: {args.sample}", file=sys.stderr)
        return 1

    return run_demo(
        args.sample,
        checkpoint=args.checkpoint,
        metadata_path=args.metadata,
        top_k=args.top_k,
        mock_llm=mock_llm,
        mock_rag=mock_rag,
        trace_path=args.trace or None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
