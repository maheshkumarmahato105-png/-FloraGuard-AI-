"""Reusable PyTorch training and validation epoch execution engine (Phase 4).

Supports:
  - GPU acceleration with Automatic Mixed Precision (AMP) & GradScaler
  - Safe CPU fallback
  - Non-blocking host-to-device transfers
  - Optional gradient norm clipping
  - Strict failure on NaN/Inf loss (never silently skips broken batches)
  - Explicit propagation of torch.cuda.OutOfMemoryError for safe recovery
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_classification_metrics


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve execution device ('auto', 'cuda', or 'cpu')."""
    req = requested.strip().lower()
    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device was explicitly requested, but torch.cuda.is_available() is False."
            )
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
    grad_clip_norm: Optional[float] = None,
    non_blocking: bool = True,
    epoch_num: int = 1,
    total_epochs: int = 1,
    show_progress: bool = True,
) -> Dict[str, float]:
    """Run one complete training epoch using raw logits with CrossEntropyLoss."""
    model.train()
    amp_enabled = bool(use_amp and device.type == "cuda")

    running_loss = 0.0
    total_samples = 0
    all_preds: List[int] = []
    all_targets: List[int] = []

    iterator = (
        tqdm(dataloader, desc=f"Epoch {epoch_num}/{total_epochs} [Train]", leave=False)
        if show_progress
        else dataloader
    )

    for batch_idx, (images, targets) in enumerate(iterator):
        images = images.to(device, non_blocking=(non_blocking and device.type == "cuda"))
        targets = targets.to(device, non_blocking=(non_blocking and device.type == "cuda"))

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        if not torch.isfinite(loss).item():
            raise FloatingPointError(
                f"Non-finite training loss ({loss.item()}) encountered at epoch {epoch_num}, batch {batch_idx}."
            )

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            optimizer.step()

        batch_size = int(targets.size(0))
        running_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        preds = torch.argmax(logits.detach(), dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())

        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{running_loss / max(total_samples, 1):.4f}")

    if total_samples == 0:
        raise RuntimeError("Training dataloader yielded 0 samples.")

    avg_loss = running_loss / total_samples
    metrics = compute_classification_metrics(all_targets, all_preds)
    return {
        "loss": round(avg_loss, 6),
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "samples": total_samples,
    }


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    non_blocking: bool = True,
    epoch_num: int = 1,
    total_epochs: int = 1,
    split_label: str = "Val",
    show_progress: bool = True,
) -> Dict[str, float]:
    """Run deterministic evaluation epoch (model.eval(), no gradients, no augmentation)."""
    model.eval()
    amp_enabled = bool(use_amp and device.type == "cuda")

    running_loss = 0.0
    total_samples = 0
    all_preds: List[int] = []
    all_targets: List[int] = []

    iterator = (
        tqdm(dataloader, desc=f"Epoch {epoch_num}/{total_epochs} [{split_label}]", leave=False)
        if show_progress
        else dataloader
    )

    for images, targets in iterator:
        images = images.to(device, non_blocking=(non_blocking and device.type == "cuda"))
        targets = targets.to(device, non_blocking=(non_blocking and device.type == "cuda"))

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        if not torch.isfinite(loss).item():
            raise FloatingPointError(
                f"Non-finite {split_label} loss ({loss.item()}) encountered at epoch {epoch_num}."
            )

        batch_size = int(targets.size(0))
        running_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    if total_samples == 0:
        raise RuntimeError(f"{split_label} dataloader yielded 0 samples.")

    avg_loss = running_loss / total_samples
    metrics = compute_classification_metrics(all_targets, all_preds)
    return {
        "loss": round(avg_loss, 6),
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "samples": total_samples,
    }
