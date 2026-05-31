"""
e2e_simulator/local_smoke.py
─────────────────────────────
In-process smoke test that exercises the FULL proto→features→normalize→infer→publish
pipeline WITHOUT Kafka or Docker. Uses mock_producer to build proto payloads, then
feeds them directly into snapshot_engine internals.

Reports each stage pass/fail exactly like smoke_test.py would.

Usage:
  python -m e2e_simulator.local_smoke
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "proto"))

def _ok(s): return f"[PASS] {s}"
def _fail(s): return f"[FAIL] {s}"


def run() -> dict[str, bool]:
    results = {}

    # ── Stage 1: Proto generation ──────────────────────────────────────────────
    try:
        from e2e_simulator.mock_producer import temporal_to_batches_per_type
        from training_pipeline.dataset_generator import (
            build_routing_maps, inject_fault, simulate_healthy_trajectory,
        )
        traj    = simulate_healthy_trajectory(seed=42)
        routing = build_routing_maps(seed=None)
        fault   = inject_fault(traj, "hdd_degradation", 0.8, routing, seed=7)
        batches = temporal_to_batches_per_type(fault["temporal_state"])
        total_kb = sum(len(v) for v in batches.values()) // 1024
        gt_type  = fault["root_cause_node_type"]
        gt_id    = fault["root_cause_node_id"]
        print(_ok(f"Stage 1 proto_gen: {len(batches)} batches {total_kb} KB  GT={gt_type}[{gt_id}]"))
        results["proto_gen"] = True
    except Exception as exc:
        print(_fail(f"Stage 1 proto_gen: {exc}"))
        results["proto_gen"] = False
        return results

    # ── Stage 2: Proto deserialization + all values [0,1] ─────────────────────
    try:
        import telemetry_pb2
        all_samples = []
        for payload in batches.values():
            b = telemetry_pb2.Batch()
            b.ParseFromString(payload)
            all_samples.extend(b.samples)
        cont_vals = [s.value for s in all_samples if not s.is_categorical]
        outside   = [v for v in cont_vals if v < -0.01 or v > 1.01]
        ok = not outside
        print((_ok if ok else _fail)(
            f"Stage 2 proto_deser: {len(all_samples)} samples, "
            f"cont values in [0,1]: {len(cont_vals)-len(outside)}/{len(cont_vals)}"
        ))
        results["proto_deser"] = ok
    except Exception as exc:
        print(_fail(f"Stage 2 proto_deser: {exc}"))
        results["proto_deser"] = False
        return results

    # ── Stage 3: EntityMapper → x_dict dimensions ─────────────────────────────
    try:
        from snapshot_engine.features import EntityMapper, assemble_x_dict
        from snapshot_engine.topology import EXPECTED_DIMS
        mapper   = EntityMapper()
        x_dict   = assemble_x_dict(all_samples, mapper)
        dim_ok   = all(x_dict[nt].shape[1] == exp for nt, exp in EXPECTED_DIMS.items())
        print((_ok if dim_ok else _fail)(
            f"Stage 3 features: dims=" +
            " ".join(f"{nt}={x_dict[nt].shape}" for nt in list(EXPECTED_DIMS)[:3])
        ))
        results["features"] = dim_ok
    except Exception as exc:
        print(_fail(f"Stage 3 features: {exc}"))
        results["features"] = False
        return results

    # ── Stage 4: Normalization ─────────────────────────────────────────────────
    try:
        from snapshot_engine.normalizer import apply_normalization, load_scaler_stats
        scaler = load_scaler_stats()
        ok     = scaler is not None
        x_norm = apply_normalization(x_dict, scaler)
        # Check cpu mean is roughly centred (not 269x off)
        cpu_mean = float(x_norm["cpu"].mean().item())
        cpu_ok   = abs(cpu_mean) < 5.0   # after Z-score should be near 0
        print((_ok if ok and cpu_ok else _fail)(
            f"Stage 4 normalize: scaler={'ok' if ok else 'MISSING'}  "
            f"cpu_mean={cpu_mean:.4f} (expect ~0)"
        ))
        results["normalize"] = ok and cpu_ok
    except Exception as exc:
        print(_fail(f"Stage 4 normalize: {exc}"))
        results["normalize"] = False

    # ── Stage 5: Inference ─────────────────────────────────────────────────────
    try:
        from snapshot_engine.inference  import InferenceEngine
        from snapshot_engine.topology   import topology_singleton
        _, _, _, ei, ea = topology_singleton()
        engine = InferenceEngine.load()
        result = engine.predict(x_norm, ei, ea)
        rc     = result["rc_node"]
        conf   = result["confidence"]
        pred_ok = conf > 0.1 and rc["type"] in ("hdd", "ram", "switch", "cpu", "gpu", "job")
        match   = rc["type"] == gt_type and rc["id"] == gt_id
        print((_ok if pred_ok else _fail)(
            f"Stage 5 inference: PRED={rc['type']}[{rc['id']}] conf={conf:.4f}  "
            f"GT={gt_type}[{gt_id}]  hit@1={'YES' if match else 'NO'}"
        ))
        results["inference"] = pred_ok
    except Exception as exc:
        print(_fail(f"Stage 5 inference: {exc}"))
        results["inference"] = False
        return results

    # ── Stage 6: Publish → inference_sample.json ──────────────────────────────
    try:
        from snapshot_engine.publisher import publish
        publish(result)
        inf_path = BASE_DIR / "artifacts" / "inference_sample.json"
        content  = json.loads(inf_path.read_text(encoding="utf-8"))
        ok       = "rc_node" in content and "top5_candidates" in content
        print((_ok if ok else _fail)(
            f"Stage 6 publish: {inf_path.name} written, "
            f"RC={content.get('rc_node')} conf={content.get('confidence')}"
        ))
        results["publish"] = ok
    except Exception as exc:
        print(_fail(f"Stage 6 publish: {exc}"))
        results["publish"] = False

    # ── Stage 7: Grafana API /topology (in-process, no Docker) ────────────────
    try:
        from fastapi.testclient import TestClient
        from api.grafana_api import app
        client = TestClient(app)
        resp   = client.get("/topology")
        data   = resp.json()
        nodes  = data.get("nodes", [])
        edges  = data.get("edges", [])
        n_sw   = sum(1 for n in nodes if n["id"].startswith("switch-"))
        n_up   = sum(1 for e in edges if e.get("mainStat") == "uplink")
        has_rc = any(n.get("mainStat") == "RC" for n in nodes)
        nids   = {n["id"] for n in nodes}
        broken = [e for e in edges if e["source"] not in nids or e["target"] not in nids]
        ok     = n_sw == 6 and n_up == 8 and not broken
        print((_ok if ok else _fail)(
            f"Stage 7 /topology: nodes={len(nodes)} edges={len(edges)} "
            f"switches={n_sw}/6 uplinks={n_up}/8 RC_marked={has_rc} broken={len(broken)}"
        ))
        results["topology_api"] = ok
    except Exception as exc:
        print(_fail(f"Stage 7 /topology: {exc}"))
        results["topology_api"] = False

    # ── Stage 8: Grafana API /scores ──────────────────────────────────────────
    try:
        from fastapi.testclient import TestClient
        from api.grafana_api import app
        client = TestClient(app)
        resp   = client.get("/scores")
        data   = resp.json()
        ok     = data.get("confidence", 0) > 0 and data.get("fault_type", "unknown") != "unknown"
        print((_ok if ok else _fail)(
            f"Stage 8 /scores: fault={data.get('fault_type')} conf={data.get('confidence')}"
        ))
        results["scores_api"] = ok
    except Exception as exc:
        print(_fail(f"Stage 8 /scores: {exc}"))
        results["scores_api"] = False

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("LOCAL SMOKE TEST  (in-process, no Kafka/Docker needed)")
    print("=" * 60)
    results = run()
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = failed = 0
    for step, ok in results.items():
        status = "PASS" if ok else "FAIL  ←"
        print(f"  {step:20s}  {status}")
        if ok: passed += 1
        else:  failed += 1
    print(f"\nPASS={passed}  FAIL={failed}")
    sys.exit(0 if failed == 0 else 1)
