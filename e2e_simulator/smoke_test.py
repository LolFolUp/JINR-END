"""
e2e_simulator/smoke_test.py
────────────────────────────
End-to-end smoke test: Kafka → mock_producer → snapshot_engine → inference_sample.json
                                             → jinr_api → Grafana

Steps
  1  docker compose up -d kafka
  2  Wait for Kafka broker to accept connections
  3  Run mock_producer (N ticks of hdd_degradation in background thread)
  4  Run snapshot_engine.run in background thread
  5  Wait for artifacts/inference_sample.json to be updated
  6  docker compose up -d jinr_api grafana
  7  Wait for jinr_api to become healthy (/health)
  8  GET /topology  → validate nodes + edges
  9  GET /scores    → validate confidence + fault_type
 10  Report: which steps PASS / FAIL / SKIP

Usage:
  python -m e2e_simulator.smoke_test [--brokers localhost:9092] [--timeout 60]
  python -m e2e_simulator.smoke_test --no-docker   # skip docker steps
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).parent.parent

# ── Colour helpers ────────────────────────────────────────────────────────────
def _ok(msg: str)  -> str: return f"[PASS] {msg}"
def _fail(msg: str) -> str: return f"[FAIL] {msg}"
def _skip(msg: str) -> str: return f"[SKIP] {msg}"
def _info(msg: str) -> str: return f"[INFO] {msg}"


# ── Docker helpers ────────────────────────────────────────────────────────────

def docker_up(*services: str, timeout: int = 120) -> bool:
    """docker compose up -d <services>. Returns True on success."""
    cmd = ["docker", "compose", "-f", str(BASE_DIR / "docker-compose.yml"),
           "up", "-d", "--no-recreate"] + list(services)
    print(_info(f"docker compose up -d {' '.join(services)}"))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(BASE_DIR))
        if r.returncode != 0:
            print(_fail(f"docker compose failed: {r.stderr[-500:]}"))
            return False
        return True
    except subprocess.TimeoutExpired:
        print(_fail(f"docker compose timed out after {timeout}s"))
        return False


def docker_build(*services: str, timeout: int = 300) -> bool:
    """docker compose build <services>."""
    cmd = ["docker", "compose", "-f", str(BASE_DIR / "docker-compose.yml"),
           "build"] + list(services)
    print(_info(f"docker compose build {' '.join(services)} (may take a while on first run)"))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(BASE_DIR))
        if r.returncode != 0:
            print(_fail(f"docker compose build failed:\n{r.stderr[-1000:]}"))
            return False
        print(_ok("docker compose build succeeded"))
        return True
    except subprocess.TimeoutExpired:
        print(_fail(f"docker compose build timed out after {timeout}s"))
        return False


def wait_kafka(brokers: str, timeout: int = 60) -> bool:
    """Poll until Kafka accepts connections."""
    import socket
    host, port = brokers.split(":")[0], int(brokers.split(":")[1])
    deadline = time.time() + timeout
    print(_info(f"Waiting for Kafka at {host}:{port} …"))
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                print(_ok(f"Kafka reachable at {host}:{port}"))
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(2)
    print(_fail(f"Kafka not reachable after {timeout}s"))
    return False


def wait_http(url: str, timeout: int = 60) -> bool:
    """Poll until HTTP GET url returns 2xx."""
    import urllib.request
    deadline = time.time() + timeout
    print(_info(f"Waiting for {url} …"))
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status < 300:
                    print(_ok(f"{url} → HTTP {resp.status}"))
                    return True
        except Exception:
            time.sleep(3)
    print(_fail(f"{url} not reachable after {timeout}s"))
    return False


# ── Producer thread ───────────────────────────────────────────────────────────

def run_producer(brokers: str, ticks: int, errors: list) -> None:
    try:
        from e2e_simulator.mock_producer import publish
        publish(
            brokers=[brokers],
            topic="telemetry.raw",
            n_ticks=ticks,
            fault_type="hdd_degradation",
            fault_seed=7,
            tick_interval=2.0,
            node_id="smoke-sim",
        )
    except Exception as exc:
        errors.append(str(exc))


# ── snapshot_engine thread ────────────────────────────────────────────────────

def run_snapshot_engine(brokers: str, stop_event: threading.Event, errors: list) -> None:
    """Run snapshot_engine in-process until stop_event is set."""
    try:
        from snapshot_engine.consumer    import TelemetryConsumer
        from snapshot_engine.features    import EntityMapper, assemble_x_dict
        from snapshot_engine.inference   import InferenceEngine
        from snapshot_engine.normalizer  import apply_normalization, load_scaler_stats
        from snapshot_engine.publisher   import publish as publish_result
        from snapshot_engine.topology    import topology_singleton

        _, _, _, edge_index_dict, edge_attr_dict = topology_singleton()
        engine  = InferenceEngine.load()
        scaler  = load_scaler_stats()
        consumer = TelemetryConsumer(
            brokers=[brokers],
            topic="telemetry.raw",
            group_id="smoke-engine",
            timeout_ms=4_000,
        )
        mapper = EntityMapper()

        while not stop_event.is_set():
            samples = consumer.consume_tick()
            if not samples:
                continue
            x_dict = assemble_x_dict(samples, mapper)
            x_dict = apply_normalization(x_dict, scaler)
            result = engine.predict(x_dict, edge_index_dict, edge_attr_dict)
            publish_result(result)
            print(_info(f"snapshot_engine: RC={result['rc_node']['type']}[{result['rc_node']['id']}] conf={result['confidence']:.4f}"))

        consumer.close()
    except Exception as exc:
        errors.append(str(exc))


# ── Grafana API checks ────────────────────────────────────────────────────────

def check_topology(api_base: str) -> tuple[bool, str]:
    import urllib.request
    try:
        with urllib.request.urlopen(f"{api_base}/topology", timeout=5) as r:
            data = json.loads(r.read())
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        n_sw  = sum(1 for n in nodes if n["id"].startswith("switch-"))
        n_ups = sum(1 for e in edges if e.get("mainStat") == "uplink")
        has_rc = any(n.get("mainStat") == "RC" for n in nodes)
        node_ids = {n["id"] for n in nodes}
        broken  = [e for e in edges if e["source"] not in node_ids or e["target"] not in node_ids]
        ok = n_sw == 6 and n_ups == 8 and not broken
        detail = (f"nodes={len(nodes)} edges={len(edges)} "
                  f"switches={n_sw}/6 uplinks={n_ups}/8 RC_marked={has_rc} broken_refs={len(broken)}")
        return ok, detail
    except Exception as exc:
        return False, str(exc)


def check_scores(api_base: str) -> tuple[bool, str]:
    import urllib.request
    try:
        with urllib.request.urlopen(f"{api_base}/scores", timeout=5) as r:
            data = json.loads(r.read())
        conf  = data.get("confidence", 0)
        fault = data.get("fault_type", "unknown")
        ok    = conf > 0 and fault != "unknown"
        return ok, f"fault_type={fault} confidence={conf}"
    except Exception as exc:
        return False, str(exc)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="E2E smoke test")
    p.add_argument("--brokers",   default="localhost:9092")
    p.add_argument("--api",       default="http://localhost:8080")
    p.add_argument("--grafana",   default="http://localhost:3000")
    p.add_argument("--timeout",   type=int, default=90,  help="Max wait per step (s)")
    p.add_argument("--ticks",     type=int, default=4,   help="Ticks to publish")
    p.add_argument("--no-docker", action="store_true",   help="Skip docker steps")
    p.add_argument("--no-build",  action="store_true",   help="Skip docker build")
    args = p.parse_args()

    results: dict[str, Optional[bool]] = {}
    inference_path = BASE_DIR / "artifacts" / "inference_sample.json"

    # ── Step 1: Start Kafka ────────────────────────────────────────────────────
    if args.no_docker:
        print(_skip("Step 1: Start Kafka (--no-docker)"))
        results["kafka_up"] = None
    else:
        ok = docker_up("kafka")
        results["kafka_up"] = ok
        if not ok:
            print(_fail("Cannot continue without Kafka"))
            _report(results)
            sys.exit(1)

    # ── Step 2: Wait for Kafka ─────────────────────────────────────────────────
    ok = wait_kafka(args.brokers, timeout=args.timeout)
    results["kafka_ready"] = ok
    if not ok:
        print(_fail("Kafka did not start in time"))
        _report(results)
        sys.exit(1)

    # ── Step 3+4: Producer + snapshot_engine ──────────────────────────────────
    print(_info("Starting mock_producer and snapshot_engine …"))
    producer_errors  = []
    engine_errors    = []
    stop_engine      = threading.Event()

    t_prod   = threading.Thread(target=run_producer,
                                args=(args.brokers, args.ticks, producer_errors),
                                daemon=True)
    t_engine = threading.Thread(target=run_snapshot_engine,
                                args=(args.brokers, stop_engine, engine_errors),
                                daemon=True)

    # Record mtime before
    mtime_before = inference_path.stat().st_mtime if inference_path.exists() else 0

    t_engine.start()
    time.sleep(2)          # give engine time to connect
    t_prod.start()
    t_prod.join(timeout=args.ticks * 3 + 30)

    results["producer_ok"] = len(producer_errors) == 0
    if producer_errors:
        print(_fail(f"Producer errors: {producer_errors}"))

    # ── Step 5: Wait for inference_sample.json ─────────────────────────────────
    deadline = time.time() + args.timeout
    updated  = False
    while time.time() < deadline:
        if inference_path.exists():
            mtime = inference_path.stat().st_mtime
            if mtime > mtime_before:
                updated = True
                break
        time.sleep(2)

    stop_engine.set()
    t_engine.join(timeout=10)

    results["engine_ok"] = len(engine_errors) == 0
    results["inference_written"] = updated

    if engine_errors:
        print(_fail(f"snapshot_engine errors: {engine_errors}"))

    if updated:
        try:
            sample = json.loads(inference_path.read_text(encoding="utf-8"))
            rc = sample.get("rc_node", {})
            conf = sample.get("confidence", 0)
            print(_ok(f"inference_sample.json: RC={rc.get('type')}[{rc.get('id')}] conf={conf:.4f}"))
            results["rc_plausible"] = rc.get("type") in ("hdd", "ram", "switch") and conf > 0.1
        except Exception as exc:
            print(_fail(f"inference_sample.json parse error: {exc}"))
            results["rc_plausible"] = False
    else:
        print(_fail("inference_sample.json was NOT updated within timeout"))
        results["rc_plausible"] = False

    # ── Step 6: Start jinr_api + grafana ──────────────────────────────────────
    if args.no_docker:
        print(_skip("Step 6: Start jinr_api + grafana (--no-docker)"))
        results["api_up"] = results["grafana_up"] = None
    else:
        # Build image first if not skipped
        if not args.no_build:
            build_ok = docker_build("jinr_api")
            results["docker_build"] = build_ok
            if not build_ok:
                print(_fail("Docker build failed — skipping API/Grafana checks"))
                _report(results)
                sys.exit(1)

        ok_api     = docker_up("jinr_api")
        ok_grafana = docker_up("grafana")
        results["api_up"]     = ok_api
        results["grafana_up"] = ok_grafana

    # ── Step 7: Wait for jinr_api ─────────────────────────────────────────────
    if results.get("api_up") is not False:
        ok = wait_http(f"{args.api}/health", timeout=args.timeout)
        results["api_health"] = ok
    else:
        results["api_health"] = False

    # ── Step 8: /topology check ────────────────────────────────────────────────
    if results.get("api_health"):
        ok, detail = check_topology(args.api)
        results["topology_ok"] = ok
        print((_ok if ok else _fail)(f"/topology: {detail}"))
    else:
        results["topology_ok"] = False
        print(_skip("/topology (API not healthy)"))

    # ── Step 9: /scores check ──────────────────────────────────────────────────
    if results.get("api_health"):
        ok, detail = check_scores(args.api)
        results["scores_ok"] = ok
        print((_ok if ok else _fail)(f"/scores: {detail}"))
    else:
        results["scores_ok"] = False
        print(_skip("/scores (API not healthy)"))

    # ── Step 10: Grafana health ────────────────────────────────────────────────
    if results.get("grafana_up") is not False:
        ok = wait_http(f"{args.grafana}/api/health", timeout=60)
        results["grafana_health"] = ok
    else:
        results["grafana_health"] = None

    _report(results)


def _report(results: dict) -> None:
    print()
    print("=" * 60)
    print("SMOKE TEST REPORT")
    print("=" * 60)
    passed = failed = skipped = 0
    for step, ok in results.items():
        if ok is None:
            print(f"  {step:30s}  SKIP")
            skipped += 1
        elif ok:
            print(f"  {step:30s}  PASS")
            passed += 1
        else:
            print(f"  {step:30s}  FAIL  ←")
            failed += 1
    print("=" * 60)
    print(f"  PASS={passed}  FAIL={failed}  SKIP={skipped}")
    print("=" * 60)

    if failed:
        print("\nFAIL — see [FAIL] lines above for root cause")
        sys.exit(1)
    else:
        print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
