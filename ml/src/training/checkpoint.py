"""Training checkpoint persistence and safe resume utilities (Phase 4).

Saves `latest_checkpoint.pth` and `best_model.pth` containing:
  - model_state_dict
  - optimizer_state_dict
  - scheduler_state_dict
  - scaler_state_dict (when AMP GradScaler is active)
  - early_stopping_state
  - epoch & best_val_macro_f1
  - training_config, model_config, class_names, class_mapping, random_seed
  - architecture_name & history_records

Never labels an untrained model (epoch < 1) as a trained checkpoint, and guards
against silently overwriting existing checkpoints unless explicitly permitted.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim


class CheckpointOverwriteError(FileExistsError):
    """Raised when attempting to overwrite existing checkpoints without explicit confirmation."""


class CheckpointManager:
    """Manages latest, best, and emergency OOM checkpoints for a training run."""

    def __init__(
        self,
        checkpoint_dir: Path | str,
        allow_overwrite: bool = False,
        is_resuming: bool = False,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        self.best_path = self.checkpoint_dir / "best_model.pth"
        self.emergency_path = self.checkpoint_dir / "emergency_oom_checkpoint.pth"

        if not allow_overwrite and not is_resuming:
            if self.latest_path.exists() or self.best_path.exists():
                raise CheckpointOverwriteError(
                    f"Existing checkpoints found in '{self.checkpoint_dir}'. "
                    "Pass `--resume <checkpoint_path>` to continue training or `--overwrite` to confirm replacing them."
                )

    def _build_payload(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any],
        scaler: Optional[torch.amp.GradScaler],
        early_stopping_state: Dict[str, Any],
        epoch: int,
        best_val_macro_f1: float,
        training_config: Dict[str, Any],
        model_config: Dict[str, Any],
        architecture_name: str,
        class_names: List[str],
        class_mapping: Dict[str, Any],
        random_seed: int,
        history_records: List[Dict[str, Any]],
        is_emergency: bool = False,
    ) -> Dict[str, Any]:
        if epoch < 1 and not is_emergency:
            raise ValueError(
                "Refusing to save a trained checkpoint before completing at least 1 training epoch."
            )
        return {
            "schema_version": "1.0",
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "is_trained": bool(epoch >= 1),
            "is_emergency_checkpoint": bool(is_emergency),
            "architecture_name": architecture_name,
            "epoch": int(epoch),
            "best_val_macro_f1": float(best_val_macro_f1),
            "random_seed": int(random_seed),
            "num_classes": len(class_names),
            "class_names": list(class_names),
            "class_mapping": class_mapping,
            "training_config": training_config,
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "early_stopping_state": early_stopping_state,
            "history_records": history_records,
        }

    def save_latest(self, **kwargs: Any) -> Path:
        payload = self._build_payload(**kwargs, is_emergency=False)
        torch.save(payload, self.latest_path)
        return self.latest_path

    def save_best(self, **kwargs: Any) -> Path:
        payload = self._build_payload(**kwargs, is_emergency=False)
        torch.save(payload, self.best_path)
        return self.best_path

    def save_emergency(self, **kwargs: Any) -> Path:
        payload = self._build_payload(**kwargs, is_emergency=True)
        torch.save(payload, self.emergency_path)
        return self.emergency_path


def load_training_checkpoint(
    checkpoint_path: Path | str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    early_stopping: Optional[Any] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Load full training state for exact resumption."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Training checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if early_stopping is not None and ckpt.get("early_stopping_state") is not None:
        early_stopping.load_state_dict(ckpt["early_stopping_state"])

    return ckpt
