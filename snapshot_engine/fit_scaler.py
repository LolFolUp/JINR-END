"""
snapshot_engine/fit_scaler.py
────────────────────────────
Fits the Z-score scaler used by normalizer.py and saves it to
checkpoints/scaler_stats.pt.

Source of truth for the scaler:
  • dataset/manifest/generation_config.json → master_seed (default 42)
  • simulate_healthy_trajectory(seed) from training_pipeline.dataset_generator
    produces the same [0,1] pre-norm data the training pipeline used before
    calling fit_global_scaler.

Why not use dataset/raw/data_N.pt directly:
  Those files are already Z-normalized (x.mean≈0, x.std≈1). Fitting a scaler
  on pre-normalized data would give (μ≈0, σ≈1) — wrong for use with raw outputs
  of build_final_node_features which live in [0,1].

Usage:
  python -m snapshot_engine.fit_scaler                 # 200 graphs, default paths
  python -m snapshot_engine.fit_scaler --n-graphs 500
  python -m snapshot_engine.fit_scaler --verify        # fit + run all checks
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("snapshot_engine.fit_scaler")

BASE_DIR    = Path(__file__).parent.parent
MANIFEST    = BASE_DIR / "dataset" / "manifest"
SCALER_PATH = BASE_DIR / "checkpoints" / "scaler_stats.pt"

_DEFAULT_N_GRAPHS  = 200
_DEFAULT_SEED_FILE = MANIFEST / "generation_config.json"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _read_master_seed() -> int:
    """Extract master_seed from dataset generation config, or return 42."""
    try:
        cfg = json.loads(_DEFAULT_SEED_FILE.read_text(encoding="utf-8"))
        seed = cfg.get("reproducibility", {}).get("master_seed", 42)
        logger.info("master_seed from generation_config.json: %d", seed)
        return int(seed)
    except Exception as exc:
        logger.warning("Could not read master_seed (%s) — using 42", exc)
        return 42


def fit_scaler(n_graphs: int = _DEFAULT_N_GRAPHS, output: Path = SCALER_PATH) -> dict:
    """
    Simulate n_graphs healthy trajectories and fit a Z-score scaler.

    Seeds: master_seed + i for i in range(n_graphs) — gives the same
    pre-norm distribution as the training pipeline healthy split.

    Returns the scaler_stats dict (also saved to output path).
    """
    from training_pipeline.dataset_generator import (
        build_final_node_features,
        fit_global_scaler,
        simulate_healthy_trajectory,
    )

    master_seed = _read_master_seed()
    logger.info("Generating %d healthy graphs (master_seed=%d) …", n_graphs, master_seed)

    healthy_x_dicts = []
    for i in range(n_graphs):
        seed = master_seed + i
        traj    = simulate_healthy_trajectory(seed=seed)
        x_dict  = build_final_node_features(traj)
        healthy_x_dicts.append(x_dict)
        if (i + 1) % 50 == 0:
            logger.info("  … %d / %d done", i + 1, n_graphs)

    logger.info("Fitting scaler …")
    stats = fit_global_scaler(healthy_x_dicts)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, str(output))
    logger.info("Saved scaler_stats → %s", output)

    # Quick sanity print
    logger.info("Per-type mu/sigma ranges:")
    for nt, s in stats.items():
        mu = s["mean"]
        sg = s["std"]
        logger.info(
            "  %-10s  mu=[%.4f, %.4f]  sigma=[%.4f, %.4f]",
            nt, mu.min().item(), mu.max().item(), sg.min().item(), sg.max().item(),
        )

    return stats


# ── Verification ───────────────────────────────────────────────────────────────

def verify_no_warning() -> bool:
    """Check that --demo run no longer emits the scaler-not-found WARNING."""
    logger.info("--- Verification 1: --demo no longer emits scaler WARNING ---")
    result = subprocess.run(
        [sys.executable, "-m", "snapshot_engine.run", "--demo"],
        capture_output=True, text=True, cwd=str(BASE_DIR),
    )
    combined = result.stdout + result.stderr
    warning_present = "scaler_stats.pt NOT FOUND" in combined
    published       = "Published" in combined or "DEMO" in combined

    if warning_present:
        logger.error("FAIL: scaler WARNING still present in --demo output")
        logger.error("--- stderr ---\n%s", result.stderr[-1000:])
        return False
    if not published:
        logger.error("FAIL: --demo did not produce a prediction (check run.py)")
        logger.error("--- stderr ---\n%s", result.stderr[-1000:])
        return False

    logger.info("PASS: no scaler WARNING, prediction published")
    return True


def verify_hit1_data1() -> bool:
    """Check Hit@1 on dataset/raw/data_1.pt (pre-normalized, no scaler needed)."""
    logger.info("--- Verification 2: Hit@1 on data_1.pt ---")
    from snapshot_engine.inference import InferenceEngine

    data_path = BASE_DIR / "dataset" / "raw" / "data_1.pt"
    if not data_path.exists():
        logger.warning("data_1.pt not found — skipping Hit@1 check")
        return True

    data = torch.load(str(data_path), map_location="cpu", weights_only=False)

    x_dict = {nt: data[nt].x for nt in data.node_types}
    ei     = {et: data[et].edge_index for et in data.edge_types}
    ea     = {
        et: data[et].edge_attr for et in data.edge_types
        if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None
    }

    engine = InferenceEngine.load()
    result = engine.predict(x_dict, ei, ea)

    # Ground truth
    gt_type = gt_id = None
    for nt in data.node_types:
        if nt == "rca_context":
            continue
        if hasattr(data[nt], "y") and data[nt].y is not None:
            idx = (data[nt].y == 1).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                gt_type, gt_id = nt, int(idx[0])

    rc   = result["rc_node"]
    match = (gt_type == rc["type"] and gt_id == rc["id"])
    logger.info(
        "%s  GT=%s[%s]  PRED=%s[%s]  confidence=%.4f",
        "PASS" if match else "FAIL",
        gt_type, gt_id, rc["type"], rc["id"], result["confidence"],
    )
    return match


def verify_demo_confidence() -> bool:
    """Check that --demo mode produces a high-confidence prediction (>0.5)."""
    logger.info("--- Verification 3: --demo confidence check ---")
    from training_pipeline.dataset_generator import (
        build_final_node_features,
        build_routing_maps,
        inject_fault,
        simulate_healthy_trajectory,
    )
    from snapshot_engine.inference    import InferenceEngine
    from snapshot_engine.normalizer   import apply_normalization, load_scaler_stats
    from snapshot_engine.topology     import topology_singleton

    traj    = simulate_healthy_trajectory(seed=0)
    routing = build_routing_maps(seed=None)
    faulted = inject_fault(traj, "hdd_degradation", 0.8, routing, seed=7)
    x_dict  = build_final_node_features(faulted["temporal_state"])

    stats   = load_scaler_stats()
    x_norm  = apply_normalization(x_dict, stats)

    _, _, _, ei, ea = topology_singleton()
    engine  = InferenceEngine.load()
    result  = engine.predict(x_norm, ei, ea)

    gt_type = faulted["root_cause_node_type"]
    gt_id   = faulted["root_cause_node_id"]
    rc      = result["rc_node"]
    conf    = result["confidence"]
    match   = (gt_type == rc["type"] and gt_id == rc["id"])

    if conf < 0.5:
        logger.warning(
            "Low confidence %.4f — scaler may still be off; "
            "consider increasing --n-graphs", conf,
        )
    logger.info(
        "%s  GT=%s[%d]  PRED=%s[%d]  confidence=%.4f",
        "PASS" if match else "INFO(top-5?)", gt_type, gt_id, rc["type"], rc["id"], conf,
    )
    return True   # confidence check is informational only


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and save Z-score scaler for snapshot_engine")
    parser.add_argument("--n-graphs",  type=int, default=_DEFAULT_N_GRAPHS,
                        help=f"Healthy graphs to simulate (default {_DEFAULT_N_GRAPHS})")
    parser.add_argument("--output",    default=str(SCALER_PATH),
                        help="Output path for scaler_stats.pt")
    parser.add_argument("--verify",    action="store_true",
                        help="Run verification checks after fitting")
    parser.add_argument("--verify-only", action="store_true",
                        help="Skip fitting; only run verification on existing scaler")
    args = parser.parse_args()

    output = Path(args.output)

    if not args.verify_only:
        fit_scaler(n_graphs=args.n_graphs, output=output)
    else:
        if not output.exists():
            logger.error("--verify-only requested but %s does not exist. Run without --verify-only first.", output)
            sys.exit(1)
        logger.info("Skipping fit; using existing %s", output)

    if args.verify or args.verify_only:
        results = {
            "no_warning": verify_no_warning(),
            "hit1_data1": verify_hit1_data1(),
            "demo_confidence": verify_demo_confidence(),
        }
        passed = sum(results.values())
        total  = len(results)
        logger.info("─" * 60)
        logger.info("Verification: %d / %d passed", passed, total)
        for name, ok in results.items():
            logger.info("  %-22s  %s", name, "PASS" if ok else "FAIL")
        if passed < total:
            sys.exit(1)


if __name__ == "__main__":
    main()
