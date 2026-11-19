"""Loss function factory with controlled class-imbalance weighting and label smoothing."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


def compute_class_weights(
    class_counts: Sequence[int],
    strategy: str = "inverse_sqrt",
    min_weight: float = 0.5,
    max_weight: float = 3.0,
) -> Optional[torch.Tensor]:
    """Compute bounded per-class loss weights strictly from training split counts.

    Strategies:
      - `none`: unweighted standard CrossEntropyLoss.
      - `inverse_frequency`: total / (num_classes * count_c), clipped to [min_weight, max_weight].
      - `inverse_sqrt`: dampened inverse square-root weighting (well-suited for 36x imbalance
        without destabilizing gradients on rare classes), normalized to mean 1.0 and clipped.
    """
    key = strategy.strip().lower()
    if key == "none":
        return None

    counts = np.asarray(class_counts, dtype=np.float64)
    if counts.ndim != 1 or len(counts) == 0:
        raise ValueError("class_counts must be a non-empty 1D sequence.")
    safe_counts = np.maximum(counts, 1.0)
    total = float(safe_counts.sum())
    num_classes = float(len(safe_counts))

    if key == "inverse_frequency":
        weights = total / (num_classes * safe_counts)
    elif key == "inverse_sqrt":
        weights = np.sqrt(total / (num_classes * safe_counts))
        weights = weights / np.mean(weights)
    else:
        raise ValueError(
            f"Unsupported class_weighting strategy '{strategy}'. "
            "Use 'none', 'inverse_frequency', or 'inverse_sqrt'."
        )

    weights = np.clip(weights, float(min_weight), float(max_weight))
    return torch.tensor(weights, dtype=torch.float32)


def build_criterion(
    training_config: Dict[str, Any],
    device: torch.device,
    train_class_counts: Optional[Sequence[int]] = None,
) -> nn.Module:
    """Build CrossEntropyLoss from training_config.yaml, guarding against conflicting imbalance fixes."""
    loss_cfg = training_config["loss"]
    data_cfg = training_config.get("data", {})

    loss_name = str(loss_cfg.get("name", "cross_entropy")).strip().lower()
    if loss_name != "cross_entropy":
        raise ValueError(f"Unsupported loss '{loss_name}'. Expected 'cross_entropy'.")

    weighting = str(loss_cfg.get("class_weighting", "none")).strip().lower()
    use_weighted_sampler = bool(data_cfg.get("use_weighted_sampler", False))
    if weighting != "none" and use_weighted_sampler:
        raise ValueError(
            "Do not combine class-weighted loss with a weighted random sampler simultaneously. "
            "Disable either `loss.class_weighting` or `data.use_weighted_sampler`."
        )

    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError(f"label_smoothing must be in [0.0, 1.0), got {label_smoothing}")

    weight_tensor = None
    if weighting != "none":
        if train_class_counts is None:
            raise ValueError(
                "train_class_counts must be provided when loss.class_weighting is enabled."
            )
        weight_tensor = compute_class_weights(
            class_counts=train_class_counts,
            strategy=weighting,
            min_weight=float(loss_cfg.get("class_weight_min", 0.5)),
            max_weight=float(loss_cfg.get("class_weight_max", 3.0)),
        )
        if weight_tensor is not None:
            weight_tensor = weight_tensor.to(device)

    return nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
