"""Optimizer factory for PlantDiseaseAI (AdamW default)."""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn
import torch.optim as optim


def build_optimizer(model: nn.Module, training_config: Dict[str, Any]) -> optim.Optimizer:
    """Create configured optimizer (default: AdamW with lr=0.001, weight_decay=0.0001)."""
    opt_cfg = training_config["optimizer"]
    name = str(opt_cfg.get("name", "adamw")).strip().lower()
    lr = float(opt_cfg.get("lr", 1e-3))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-4))

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Model has no trainable parameters.")

    if name == "adamw":
        betas = tuple(float(b) for b in opt_cfg.get("betas", [0.9, 0.999]))
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "adam":
        betas = tuple(float(b) for b in opt_cfg.get("betas", [0.9, 0.999]))
        return optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)

    raise ValueError(f"Unsupported optimizer '{name}'. Supported: adamw, adam, sgd.")
