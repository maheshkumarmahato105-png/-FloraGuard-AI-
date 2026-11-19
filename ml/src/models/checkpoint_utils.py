"""Reusable, reproducible checkpoint save/load utilities for PlantDiseaseAI.

Supports storing:
  - model_state_dict
  - model_config
  - class_mapping (ordered class list + optional full taxonomy dict)
  - random_seed
  - architecture_name
  - optional training metadata (epoch, optimizer_state_dict, metrics — for Phase 4+)

Note: Phase 3 provides these utilities only; no fake trained checkpoints are created.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


def save_model_checkpoint(
    filepath: Path | str,
    model: nn.Module,
    architecture_name: str,
    model_config: Dict[str, Any],
    class_names: List[str],
    random_seed: int,
    class_mapping: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a structured, reproducible PyTorch model checkpoint."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture_name": architecture_name,
        "random_seed": int(random_seed),
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "class_mapping": class_mapping or {},
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
    }
    if extra_metadata:
        payload["extra_metadata"] = extra_metadata

    torch.save(payload, out_path)
    return out_path


def load_model_checkpoint(
    filepath: Path | str,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Load and validate a structured checkpoint dictionary from disk."""
    ckpt_path = Path(filepath)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    required_keys = {
        "architecture_name",
        "random_seed",
        "class_names",
        "model_config",
        "model_state_dict",
    }
    missing = required_keys - set(checkpoint.keys())
    if missing:
        raise ValueError(f"Checkpoint '{ckpt_path}' is missing required keys: {sorted(missing)}")
    return checkpoint
