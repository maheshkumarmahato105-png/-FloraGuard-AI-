"""Two-stage Transfer Learning Pipeline for Plant Disease Detection.

Trains a MobileNetV2 (default) or EfficientNet-B0 PyTorch model on the real
PlantVillage train/validation splits with data augmentation, mixed-precision
GPU acceleration (when CUDA is available), class-balanced loss, learning rate
scheduling, and early stopping.

Usage:
    python training/train.py --arch mobilenet_v2 --warmup-epochs 2 --finetune-epochs 6 --batch-size 64
"""

import argparse
from datetime import datetime, timezone
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    CLASS_NAMES_PATH,
    MODEL_WEIGHTS_PATH,
    SAVED_MODELS_DIR,
    TRAINING_REPORTS_DIR,
)
from backend.services.model_service import build_model_architecture
from training.preprocess import load_splits_and_dataloaders


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
    scaler=None,
    use_amp: bool = False,
    desc: str = "Epoch",
) -> Tuple[float, float, float]:
    """Run one training or validation epoch. Returns (loss, accuracy, macro_f1)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    total_samples = 0
    all_preds: List[int] = []
    all_targets: List[int] = []

    pbar = tqdm(loader, desc=desc, leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            if is_train:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())

        pbar.set_postfix(loss=f"{running_loss / max(total_samples, 1):.4f}")

    avg_loss = running_loss / max(total_samples, 1)
    correct = sum(int(p == t) for p, t in zip(all_preds, all_targets))
    accuracy = correct / max(total_samples, 1)
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    return avg_loss, accuracy, macro_f1


def plot_training_curves(history: List[Dict], output_path: Path) -> None:
    """Save clean training/validation loss, accuracy, and macro F1 curves."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] * 100.0 for h in history]
    val_acc = [h["val_acc"] * 100.0 for h in history]
    val_f1 = [h["val_macro_f1"] * 100.0 for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, train_loss, marker="o", label="Train Loss", color="#059669", linewidth=2)
    axes[0].plot(epochs, val_loss, marker="s", label="Val Loss", color="#dc2626", linewidth=2)
    axes[0].set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", label="Train Accuracy (%)", color="#059669", linewidth=2)
    axes[1].plot(epochs, val_acc, marker="s", label="Val Accuracy (%)", color="#2563eb", linewidth=2)
    axes[1].plot(epochs, val_f1, marker="^", linestyle="--", label="Val Macro F1 (%)", color="#d97706", linewidth=2)
    axes[1].set_title("Accuracy & Macro F1 Progression", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def train_model(
    arch: str = "mobilenet_v2",
    warmup_epochs: int = 2,
    finetune_epochs: int = 6,
    batch_size: int = 64,
    num_workers: int = 4,
    lr_head: float = 1e-3,
    lr_backbone: float = 2e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> None:
    """Execute 2-stage transfer learning and save best checkpoint to backend/models/saved/."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print(f"=== Starting Plant Disease Model Training ===")
    print(f"Architecture:     {arch}")
    print(f"Device:           {device}" + (f" ({torch.cuda.get_device_name(0)})" if use_amp else ""))
    print(f"Mixed Precision:  {use_amp}")
    print(f"Warmup Epochs:    {warmup_epochs} | Fine-tune Epochs: {finetune_epochs} | Batch Size: {batch_size}")

    train_loader, val_loader, _, class_names, class_weights = load_splits_and_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=use_amp,
    )
    num_classes = len(class_names)
    print(f"Loaded {len(train_loader.dataset)} train / {len(val_loader.dataset)} val images across {num_classes} classes.")

    model = build_model_architecture(architecture=arch, num_classes=num_classes, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=0.05,
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_f1 = -1.0
    best_val_acc = 0.0
    history: List[Dict] = []
    epoch_counter = 0
    train_start = time.time()

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save class_names.json immediately
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    def save_if_best(val_acc: float, val_f1: float, epoch_num: int) -> None:
        nonlocal best_val_f1, best_val_acc
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            checkpoint = {
                "architecture": arch,
                "num_classes": num_classes,
                "class_names": class_names,
                "model_state_dict": model.state_dict(),
                "best_val_accuracy": round(val_acc * 100.0, 4),
                "best_val_macro_f1": round(val_f1 * 100.0, 4),
                "best_epoch": epoch_num,
                "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            torch.save(checkpoint, MODEL_WEIGHTS_PATH)
            print(
                f"  -> Saved new best checkpoint to {MODEL_WEIGHTS_PATH.name} "
                f"(Val Acc: {val_acc * 100:.2f}%, Val Macro F1: {val_f1 * 100:.2f}%)"
            )

    # ---------------- STAGE 1: Classifier Head Warmup ----------------
    if warmup_epochs > 0:
        print("\n--- Stage 1: Classifier Head Warmup (Frozen Backbone) ---")
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("classifier")

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr_head,
            weight_decay=weight_decay,
        )
        for s1_epoch in range(1, warmup_epochs + 1):
            epoch_counter += 1
            t0 = time.time()
            tr_loss, tr_acc, tr_f1 = run_epoch(
                model, train_loader, criterion, device, optimizer, scaler, use_amp,
                desc=f"Stage1 Train {s1_epoch}/{warmup_epochs}",
            )
            va_loss, va_acc, va_f1 = run_epoch(
                model, val_loader, criterion, device, None, None, use_amp,
                desc=f"Stage1 Val   {s1_epoch}/{warmup_epochs}",
            )
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch_counter:02d} [Warmup]   ({elapsed:5.1f}s) | "
                f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:6.2f}% | "
                f"Val Loss: {va_loss:.4f} Acc: {va_acc*100:6.2f}% F1: {va_f1*100:6.2f}%"
            )
            history.append(
                {
                    "epoch": epoch_counter,
                    "stage": "warmup",
                    "train_loss": round(tr_loss, 5),
                    "train_acc": round(tr_acc, 5),
                    "train_macro_f1": round(tr_f1, 5),
                    "val_loss": round(va_loss, 5),
                    "val_acc": round(va_acc, 5),
                    "val_macro_f1": round(va_f1, 5),
                    "duration_sec": round(elapsed, 2),
                }
            )
            save_if_best(va_acc, va_f1, epoch_counter)

    # ---------------- STAGE 2: Full Backbone Fine-Tuning ----------------
    if finetune_epochs > 0:
        print("\n--- Stage 2: Full Backbone Fine-Tuning ---")
        for param in model.parameters():
            param.requires_grad = True

        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
        head_params = [p for n, p in model.named_parameters() if n.startswith("classifier")]

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": lr_backbone},
                {"params": head_params, "lr": lr_head * 0.6},
            ],
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=finetune_epochs, eta_min=1e-6
        )

        for s2_epoch in range(1, finetune_epochs + 1):
            epoch_counter += 1
            t0 = time.time()
            tr_loss, tr_acc, tr_f1 = run_epoch(
                model, train_loader, criterion, device, optimizer, scaler, use_amp,
                desc=f"Stage2 Train {s2_epoch}/{finetune_epochs}",
            )
            va_loss, va_acc, va_f1 = run_epoch(
                model, val_loader, criterion, device, None, None, use_amp,
                desc=f"Stage2 Val   {s2_epoch}/{finetune_epochs}",
            )
            scheduler.step()
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch_counter:02d} [FineTune] ({elapsed:5.1f}s) | "
                f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:6.2f}% | "
                f"Val Loss: {va_loss:.4f} Acc: {va_acc*100:6.2f}% F1: {va_f1*100:6.2f}%"
            )
            history.append(
                {
                    "epoch": epoch_counter,
                    "stage": "finetune",
                    "train_loss": round(tr_loss, 5),
                    "train_acc": round(tr_acc, 5),
                    "train_macro_f1": round(tr_f1, 5),
                    "val_loss": round(va_loss, 5),
                    "val_acc": round(va_acc, 5),
                    "val_macro_f1": round(va_f1, 5),
                    "duration_sec": round(elapsed, 2),
                }
            )
            save_if_best(va_acc, va_f1, epoch_counter)

    total_time = time.time() - train_start
    history_path = TRAINING_REPORTS_DIR / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "architecture": arch,
                "device": str(device),
                "total_training_seconds": round(total_time, 2),
                "best_val_accuracy_pct": round(best_val_acc * 100.0, 2),
                "best_val_macro_f1_pct": round(best_val_f1 * 100.0, 2),
                "epochs": history,
            },
            f,
            indent=2,
        )

    plot_training_curves(history, TRAINING_REPORTS_DIR / "training_curves.png")

    print("\n=== Training Completed ===")
    print(f"Total Duration:     {total_time / 60:.2f} minutes")
    print(f"Best Val Accuracy:  {best_val_acc * 100:.2f}%")
    print(f"Best Val Macro F1:  {best_val_f1 * 100:.2f}%")
    print(f"Saved Checkpoint:   {MODEL_WEIGHTS_PATH}")
    print(f"Training Curves:    {TRAINING_REPORTS_DIR / 'training_curves.png'}")
    print("Next step: Run `python training/evaluate.py` to evaluate on the held-out test set.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Plant Disease Detection CNN model.")
    parser.add_argument("--arch", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "efficientnet_b0"])
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--finetune-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_model(
        arch=args.arch,
        warmup_epochs=args.warmup_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
