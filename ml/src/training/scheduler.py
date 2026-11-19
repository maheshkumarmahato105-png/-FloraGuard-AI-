"""Learning rate scheduler factory (CosineAnnealingLR default)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR


def build_scheduler(
    optimizer: optim.Optimizer,
    training_config: Dict[str, Any],
) -> Optional[Any]:
    """Create configured LR scheduler (CosineAnnealingLR, ReduceLROnPlateau, StepLR, or None)."""
    sched_cfg = training_config.get("scheduler", {})
    name = str(sched_cfg.get("name", "cosine_annealing")).strip().lower()
    epochs = max(1, int(training_config["training"].get("epochs", 30)))

    if name in ("none", ""):
        return None
    if name == "cosine_annealing":
        eta_min = float(sched_cfg.get("eta_min", 1e-6))
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    if name == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(sched_cfg.get("plateau_factor", 0.5)),
            patience=int(sched_cfg.get("plateau_patience", 3)),
            min_lr=float(sched_cfg.get("eta_min", 1e-6)),
        )
    if name == "step":
        return StepLR(
            optimizer,
            step_size=int(sched_cfg.get("step_size", 10)),
            gamma=float(sched_cfg.get("step_gamma", 0.5)),
        )

    raise ValueError(
        f"Unsupported scheduler '{name}'. Supported: cosine_annealing, reduce_on_plateau, step, none."
    )


def step_scheduler(scheduler: Optional[Any], monitor_metric: Optional[float] = None) -> None:
    """Step the scheduler appropriately whether epoch-based or metric-based."""
    if scheduler is None:
        return
    if isinstance(scheduler, ReduceLROnPlateau):
        if monitor_metric is None:
            raise ValueError("monitor_metric is required for ReduceLROnPlateau.")
        scheduler.step(monitor_metric)
    else:
        scheduler.step()
