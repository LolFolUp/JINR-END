"""
Grafana data API — serves inference + remediation artifacts for the
Infinity datasource plugin (yesoreyeram-infinity-datasource).

Endpoints:
  GET  /health              → 200 OK
  GET  /topology            → Node Graph format (nodes + edges)
  GET  /scores              → GNN confidence / fault_type time-series
  GET  /playbook            → Remediation actions table
  GET  /summary             → Top-level incident stats (Stat panels)

  POST /feedback            → Engineer confirms or rejects RCA diagnosis
                              Body: FeedbackRequest (see replay_buffer.py)
  GET  /feedback/stats      → Replay buffer statistics (accuracy, counts)
  GET  /feedback/export     → Download all labeled incidents as JSON list
                              (consumed by offline batch fine-tuning)

How to wire confirm/reject in Grafana:
  Grafana 10.3+ Node Graph panel supports "Actions" that fire HTTP requests
  when an operator clicks on a node. Configure two actions in the panel JSON:

    {
      "title": "Confirm RCA",
      "type": "post",
      "url": "http://localhost:8080/feedback",
      "body": "{\"incident_id\": \"${__data.fields.id}\", \"verdict\": \"confirm\"}"
    },
    {
      "title": "Reject RCA",
      "type": "post",
      "url": "http://localhost:8080/feedback",
      "body": "{\"incident_id\": \"${__data.fields.id}\", \"verdict\": \"reject\"}"
    }

  Alternatively, operators can call the endpoint directly from any HTTP client.

Run:
  uvicorn api.grafana_api:app --host 0.0.0.0 --port 8080 --reload
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.replay_buffer import (
    FeedbackRequest,
    FeedbackVerdict,
    LabeledIncident,
    ReplayBuffer,
    get_buffer,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
ARTIFACTS = BASE_DIR / "artifacts"
MANIFEST  = BASE_DIR / "dataset" / "manifest"

INFERENCE_FILE   = ARTIFACTS / "inference_sample.json"
REMEDIATION_FILE = ARTIFACTS / "remediation_report.json"
TOPOLOGY_FILE    = MANIFEST  / "topology.json"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="JINR RCA Grafana API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{path.name} not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


# Anomaly score colour: green (ok) → yellow → red (fault)
def _score_color(confidence: float) -> str:
    if confidence >= 0.75:
        return "red"
    if confidence >= 0.45:
        return "orange"
    return "green"


# Edge XAI weight → Node Graph thickness (px). Base 1, scales up to ~7 at w=1.
def _edge_thickness(weight: float) -> float:
    return round(1.0 + 6.0 * max(0.0, min(1.0, weight)), 2)


# Edge XAI weight → colour (same palette as nodes: green→orange→red).
def _edge_color(weight: float) -> str:
    if weight >= 0.75:
        return "red"
    if weight >= 0.45:
        return "orange"
    if weight >= 0.05:
        return "yellow"
    return "#cccccc"  # near-zero: faint grey, structural link only


def _edge_weight_lookup(inference: dict) -> dict[tuple[str, str], float]:
    """Build {(source, target): weight} from inference edge_xai, both directions.

    Direction-insensitive: the model emits directed edges (e.g. cpu→switch) but
    the Node Graph may draw either orientation, so we index both (u,v) and (v,u).
    """
    lookup: dict[tuple[str, str], float] = {}
    for e in inference.get("edge_xai", {}).get("edges", []):
        s, t = e.get("source", ""), e.get("target", "")
        w = float(e.get("weight", 0.0))
        if not s or not t:
            continue
        lookup[(s, t)] = w
        lookup.setdefault((t, s), w)
    return lookup


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": _now_ms()}


@app.get("/topology")
def topology() -> dict[str, Any]:
    """
    Returns Grafana Node Graph format: { "nodes": [...], "edges": [...] }

    Node IDs use "{type}-{model_index}" to match inference_sample.json exactly.
    Switches: model indices 0-3 = leaf, 4-5 = spine (NUM_LEAF=4, NUM_SPINE=2).
    Edges built from the same static routing as snapshot_engine/topology.py:
      host_to_leaf[h] = h % NUM_LEAF  (build_routing_maps seed=None, round-robin)
      every leaf → both spines (uplinks)

    Each edge carries XAI weighting from inference["edge_xai"] (GATv2 attention,
    or salience proxy as fallback): "thickness" and "color" trace the
    fault-propagation path so an operator sees *how* the fault spread, not just
    where. Edges with weight >= 0.45 are highlighted.
    """
    topo_manifest = _load_json(TOPOLOGY_FILE)
    inference     = _load_json(INFERENCE_FILE) if INFERENCE_FILE.exists() else {}

    rc_node    = inference.get("rc_node", {})
    rc_type    = rc_node.get("type", "")
    rc_id_num  = rc_node.get("id", -1)          # integer model index
    rc_node_id = f"{rc_type}-{rc_id_num}"        # canonical string: "hdd-82"
    confidence = float(inference.get("confidence", 0.0))
    fault_type = inference.get("fault_type", "unknown")
    top5: list = inference.get("top5_candidates", [])

    # top5_scores: {"hdd-82": 0.9999, "switch-2": 0.03, ...}
    top5_scores: dict[str, float] = {c.get("id", ""): float(c.get("score", 0.0)) for c in top5}

    # XAI edge weights (attention/salience) → drives edge thickness + colour
    edge_w = _edge_weight_lookup(inference)

    topo_cfg = topo_manifest.get("topology_config", {})
    n_leaf   = int(topo_cfg.get("NUM_LEAF",  4))
    n_spine  = int(topo_cfg.get("NUM_SPINE", 2))
    n_sw     = n_leaf + n_spine        # leaf indices: 0..n_leaf-1, spine: n_leaf..n_sw-1

    # Static routing: identical to build_routing_maps(seed=None) in dataset_generator
    # host_to_leaf[h] = h % n_leaf
    def host_leaf(h: int) -> int:
        return h % n_leaf

    nodes: list[dict] = []
    edges: list[dict] = []
    seen_node_ids: set[str] = set()

    def _add_node(nid: str, title: str, subtitle: str, score: float, is_rc: bool) -> None:
        if nid in seen_node_ids:
            return
        seen_node_ids.add(nid)
        fault_arc = confidence if is_rc else score
        nodes.append({
            "id":            nid,
            "title":         title,
            "subTitle":      subtitle,
            "mainStat":      "RC" if is_rc else (f"{score:.3f}" if score > 0.001 else "ok"),
            "secondaryStat": fault_type if is_rc else "",
            "arc__fault":    round(fault_arc, 4),
            "arc__ok":       round(max(0.0, 1.0 - fault_arc), 4),
            "color":         "red" if is_rc else _score_color(score),
        })

    # ── 1. All switches (leaf + spine) — always shown ─────────────────────────
    for sw_i in range(n_sw):
        nid      = f"switch-{sw_i}"
        is_rc    = (rc_type == "switch" and rc_id_num == sw_i)
        score    = top5_scores.get(nid, 0.0)
        role     = "leaf switch" if sw_i < n_leaf else "spine switch"
        _add_node(nid, nid, role, score, is_rc)

    # ── 2. Host-component nodes: RC + top-5 candidates + context hosts ─────────
    # Collect host indices from top5 (e.g. "hdd-35" → host 35)
    interesting_hosts: set[int] = set()
    for cid in top5_scores:
        parts = cid.rsplit("-", 1)
        if len(parts) == 2 and parts[0] in ("cpu", "gpu", "ram", "hdd", "job"):
            try:
                interesting_hosts.add(int(parts[1]))
            except ValueError:
                pass

    # Always include RC host
    if rc_type in ("cpu", "gpu", "ram", "hdd", "job"):
        interesting_hosts.add(int(rc_id_num))

    # A few context hosts (indices 0–3) so the graph is never empty
    for h in range(min(4, 100)):
        interesting_hosts.add(h)

    # Cap at 12 host nodes for readability
    for host_idx in sorted(interesting_hosts)[:12]:
        # Pick the most informative component type for this host index:
        # prefer whichever type is in top5, fall back to rc_type, then "cpu"
        best_score = 0.0
        best_nid   = f"cpu-{host_idx}"
        is_rc_host = False

        for ctype in ("hdd", "ram", "cpu", "gpu", "job"):
            candidate = f"{ctype}-{host_idx}"
            s = top5_scores.get(candidate, 0.0)
            if s > best_score:
                best_score = s
                best_nid   = candidate
            if rc_type == ctype and rc_id_num == host_idx:
                is_rc_host = True
                best_nid   = rc_node_id
                best_score = confidence

        subtitle = "RC" if is_rc_host else ("top-5" if best_score > 0.001 else "host")
        _add_node(best_nid, best_nid, subtitle, best_score, is_rc_host)

        # Edge: host-component → its leaf switch (real round-robin from topology)
        leaf_sw_idx = host_leaf(host_idx)
        leaf_nid    = f"switch-{leaf_sw_idx}"
        w = edge_w.get((best_nid, leaf_nid), 0.0)
        edges.append({
            "id":            f"e_{best_nid}_sw{leaf_sw_idx}",
            "source":        best_nid,
            "target":        leaf_nid,
            "mainStat":      "access",
            "secondaryStat": f"{w:.2f}" if w > 0.0 else "",
            "thickness":     _edge_thickness(w),
            "color":         _edge_color(w),
            "highlighted":   w >= 0.45,
        })

    # ── 3. Leaf → Spine uplinks (from build_edge_indices logic) ───────────────
    for leaf_i in range(n_leaf):
        for spine_i in range(n_leaf, n_sw):
            src_nid = f"switch-{leaf_i}"
            dst_nid = f"switch-{spine_i}"
            w = edge_w.get((src_nid, dst_nid), 0.0)
            edges.append({
                "id":            f"e_sw{leaf_i}_sw{spine_i}",
                "source":        src_nid,
                "target":        dst_nid,
                "mainStat":      "uplink",
                "secondaryStat": f"{w:.2f}" if w > 0.0 else "",
                "thickness":     _edge_thickness(w),
                "color":         _edge_color(w),
                "highlighted":   w >= 0.45,
            })

    return {"nodes": nodes, "edges": edges}


@app.get("/scores")
def scores() -> dict[str, Any]:
    """
    Returns GNN prediction summary for Stat / Gauge panels.
    """
    inference = _load_json(INFERENCE_FILE)

    rc_node    = inference.get("rc_node", {})
    top5: list = inference.get("top5_candidates", [])

    return {
        "graph_id":       inference.get("graph_id", 0),
        "fault_type":     inference.get("fault_type", "unknown"),
        "confidence":     round(float(inference.get("confidence", 0.0)), 4),
        "rc_node_type":   rc_node.get("type", ""),
        "rc_node_id":     rc_node.get("id", ""),
        "victim_count":   len(inference.get("victim_nodes", [])),
        "top5":           top5,
        "timestamp_ms":   _now_ms(),
    }


@app.get("/playbook")
def playbook() -> list[dict[str, Any]]:
    """
    Returns remediation actions as a flat list (Table panel).
    """
    report = _load_json(REMEDIATION_FILE)

    raw_actions = (
        report.get("actions")
        or report.get("playbook", {}).get("actions")
        or []
    )

    rows = []
    for i, action in enumerate(raw_actions, start=1):
        rows.append({
            "step":        i,
            "action_id":   action.get("action_id", action.get("id", f"STEP_{i}")),
            "target":      action.get("target", action.get("component", "")),
            "description": action.get("description", action.get("action", "")),
            "priority":    action.get("priority", "medium"),
            "status":      action.get("status", "pending"),
        })

    return rows


@app.get("/summary")
def summary() -> dict[str, Any]:
    """
    High-level stats for the top Stat row: fault type, RC node,
    confidence, action count, firewall status.
    """
    inference  = _load_json(INFERENCE_FILE)
    report     = {}
    if REMEDIATION_FILE.exists():
        report = _load_json(REMEDIATION_FILE)

    raw_actions = (
        report.get("actions")
        or report.get("playbook", {}).get("actions")
        or []
    )

    firewall = report.get("firewall", {})
    fw_status = firewall.get("status", report.get("validation_status", "unknown"))

    return {
        "fault_type":     inference.get("fault_type", "unknown"),
        "rc_node":        inference.get("rc_node", {}).get("id", "unknown"),
        "confidence_pct": round(float(inference.get("confidence", 0.0)) * 100, 1),
        "action_count":   len(raw_actions),
        "firewall_status": fw_status,
        "timestamp_ms":   _now_ms(),
    }


# ── Experience Replay: Human-in-the-Loop feedback ─────────────────────────────

@app.post("/feedback", status_code=201)
def post_feedback(
    req: FeedbackRequest,
    buf: ReplayBuffer = Depends(get_buffer),
) -> dict[str, Any]:
    """
    Record engineer verdict (confirm / reject) for the current RCA diagnosis.

    Reads the latest inference_sample.json to snapshot the prediction state,
    then appends a LabeledIncident to artifacts/replay_buffer.jsonl.

    Grafana 10.3+ Node Graph Actions example (panel JSON):
      {
        "title": "Confirm RCA",
        "type": "post",
        "url": "http://localhost:8080/feedback",
        "body": "{\\"incident_id\\": \\"${__data.fields.id}\\", \\"verdict\\": \\"confirm\\"}"
      }
    """
    # ── Snapshot current inference state ──────────────────────────────────
    if not INFERENCE_FILE.exists():
        raise HTTPException(
            status_code=409,
            detail="No active inference found. Run the GNN pipeline first.",
        )
    inference = _load_json(INFERENCE_FILE)

    rc_node    = inference.get("rc_node", {})
    pred_type  = str(rc_node.get("type", "unknown"))
    pred_id    = int(rc_node.get("id", -1))
    fault_type = str(inference.get("fault_type", "unknown"))
    confidence = float(inference.get("confidence", 0.0))
    graph_id   = inference.get("graph_id")

    # Canonical incident_id — identical to build_incident_id() in llm/prompt_builder.py
    # so feedback records correlate with remediation reports. The stored id is always
    # canonical; a differing client hint (e.g. a Grafana node id) is folded into the comment.
    incident_id = f"graph_{graph_id}_{fault_type}"
    comment = req.comment
    if req.incident_id and req.incident_id != incident_id:
        hint = f"[grafana_ref={req.incident_id}]"
        comment = f"{hint} {comment}".strip()

    # ── Determine if prediction was correct ───────────────────────────────
    if req.verdict == FeedbackVerdict.CONFIRM:
        is_correct = True
    else:
        # Reject: correct if engineer supplied the same node as the model
        if req.correct_rc_type and req.correct_rc_id is not None:
            is_correct = (
                req.correct_rc_type == pred_type
                and req.correct_rc_id == pred_id
            )
        else:
            is_correct = False

    # ── Build and persist labeled incident ────────────────────────────────
    incident = LabeledIncident(
        incident_id           = incident_id,
        timestamp_utc         = datetime.now(timezone.utc).isoformat(),
        verdict               = req.verdict,
        comment               = comment,
        predicted_rc_type     = pred_type,
        predicted_rc_id       = pred_id,
        predicted_fault_type  = fault_type,
        confidence            = confidence,
        graph_id              = graph_id,
        correct_rc_type       = req.correct_rc_type,
        correct_rc_id         = req.correct_rc_id,
        is_correct_prediction = is_correct,
    )
    buf.save(incident)

    return {
        "status":         "saved",
        "incident_id":    incident_id,
        "verdict":        req.verdict.value,
        "is_correct":     is_correct,
        "buffer_total":   buf.stats()["total"],
    }


@app.get("/feedback/stats")
def feedback_stats(buf: ReplayBuffer = Depends(get_buffer)) -> dict[str, Any]:
    """
    Replay buffer statistics: total count, accuracy, per-fault-type breakdown.

    Use as a Stat panel data source in Grafana to monitor labeling progress.
    """
    return buf.stats()


@app.get("/feedback/export")
def feedback_export(buf: ReplayBuffer = Depends(get_buffer)) -> list[dict[str, Any]]:
    """
    Export all labeled incidents for offline batch fine-tuning.

    Each record maps to a graph file via graph_id:
      dataset/raw/data_{graph_id}.pt

    Batch retraining workflow:
      1. Download: GET http://localhost:8080/feedback/export > labeled.json
      2. For each record where is_correct=False:
           load data_{graph_id}.pt, relabel RC node, add to fine-tune set.
      3. For each record where is_correct=True:
           add to fine-tune set with original labels (positive reinforcement).
      4. Run training_pipeline/train.py with --finetune-from labeled.json
    """
    return buf.export_for_training()
