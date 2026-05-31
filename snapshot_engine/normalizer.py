"""
normalizer.py — Z-score normalization of x_dict.

Loads scaler_stats.pt produced during training.
If the file is absent, passes x_dict through UNCHANGED and logs a loud WARNING.

scaler_stats format (matches dataset_generator.fit_global_scaler output):
  { node_type: {"mean": Tensor[D], "std": Tensor[D]} }

_HEALTHY_MEAN/_STD from dataset_generator are generative params — NOT used here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import torch

from training_pipeline.dataset_generator import normalize_features

logger = logging.getLogger(__name__)

_NORM_EPS = 1e-6
_DEFAULT_PATH = Path(__file__).parent.parent / "checkpoints" / "scaler_stats.pt"

ScalerStats = Dict[str, Dict[str, torch.Tensor]]


def load_scaler_stats(path: Optional[Path] = None) -> Optional[ScalerStats]:
    """Load scaler_stats from disk. Returns None if file not found."""
    p = Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        msg = (
            "\n" + "=" * 70 + "\n"
            f"  NORMALIZER: scaler_stats.pt NOT FOUND at {p}\n"
            "  Running WITHOUT normalization — model accuracy will be degraded.\n"
            "  To fix: run training_pipeline/train.py with --save-scaler flag\n"
            "  or call snapshot_engine.normalizer.fit_and_save_scaler().\n"
            + "=" * 70
        )
        logger.warning(msg)
        return None
    stats = torch.load(str(p), map_location="cpu", weights_only=True)
    logger.info("Loaded scaler_stats from %s (%d node types)", p, len(stats))
    return stats


def apply_normalization(
    x_dict: Dict[str, torch.Tensor],
    scaler_stats: Optional[ScalerStats],
) -> Dict[str, torch.Tensor]:
    """Z-score normalize x_dict. Pass-through if scaler_stats is None."""
    if scaler_stats is None:
        return x_dict
    return normalize_features(x_dict, scaler_stats)


def fit_and_save_scaler(
    n_healthy_graphs: int = 200,
    save_path: Optional[Path] = None,
    seed: int = 42,
) -> ScalerStats:
    """Generate healthy graphs on-the-fly and fit scaler. Saves result.

    Use this when training_pipeline/train.py has not yet saved scaler_stats.pt.
    """
    from training_pipeline.dataset_generator import (
        build_final_node_features,
        fit_global_scaler,
        simulate_healthy_trajectory,
    )

    logger.info("Fitting scaler from %d healthy synthetic graphs…", n_healthy_graphs)
    healthy_x_dicts = []
    for i in range(n_healthy_graphs):
        traj = simulate_healthy_trajectory(seed=seed + i)
        healthy_x_dicts.append(build_final_node_features(traj))

    stats = fit_global_scaler(healthy_x_dicts)

    save_p = Path(save_path) if save_path else _DEFAULT_PATH
    save_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, str(save_p))
    logger.info("scaler_stats saved to %s", save_p)
    return stats
