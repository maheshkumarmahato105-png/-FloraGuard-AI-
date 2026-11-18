"""Evaluate the trained plant disease detection model on the held-out test set.

Generates:
  - training/reports/evaluation_metrics.json
  - training/reports/evaluation_report.md
  - training/reports/confusion_matrix.png
  - backend/models/saved/model_metadata.json (displayed live in the web UI)

Usage:
    python training/evaluate.py
"""

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    DATASET_DIR,
    MODEL_METADATA_PATH,
    MODEL_WEIGHTS_PATH,
    TRAINING_REPORTS_DIR,
)
from backend.services.disease_knowledge import parse_class_label
from backend.services.model_service import build_model_architecture
from training.preprocess import load_splits_and_dataloaders


def _short_label(raw_label: str) -> str:
    plant, disease, _ = parse_class_label(raw_label)
    short_d = disease if len(disease) <= 22 else disease[:20] + "…"
    return f"{plant}: {short_d}"


def save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    """Render a normalized confusion matrix heatmap."""
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), np.maximum(row_sums, 1.0))

    n = len(class_names)
    fig_size = max(12, min(26, int(n * 0.55)))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.88))

    labels = [_short_label(c) for c in class_names]
    sns.heatmap(
        cm_norm,
        xticklabels=labels,
        yticklabels=labels,
        cmap="Greens",
        vmin=0.0,
        vmax=1.0,
        square=True,
        cbar_kws={"shrink": 0.75, "label": "Normalized Recall"},
        ax=ax,
    )
    ax.set_title("Plant Disease Classification — Normalized Test Confusion Matrix", fontsize=15, fontweight="bold", pad=16)
    ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=12, fontweight="bold")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def write_markdown_report(metrics: dict, output_path: Path) -> None:
    """Write a readable Markdown evaluation report."""
    lines = [
        "# Plant Disease Detection — Model Evaluation Report",
        "",
        f"- **Evaluated At (UTC):** `{metrics['evaluated_at_utc']}`",
        f"- **Model Architecture:** `{metrics['architecture']}`",
        f"- **Number of Classes:** `{metrics['num_classes']}`",
        f"- **Test Set Size:** `{metrics['test_samples']}` images",
        "",
        "## Overall Test Set Performance",
        "",
        "| Metric | Score |",
        "| --- | ---: |",
        f"| **Test Accuracy** | **{metrics['accuracy_pct']:.2f}%** |",
        f"| **Top-3 Accuracy** | **{metrics['top3_accuracy_pct']:.2f}%** |",
        f"| **Macro Precision** | **{metrics['macro_precision_pct']:.2f}%** |",
        f"| **Macro Recall** | **{metrics['macro_recall_pct']:.2f}%** |",
        f"| **Macro F1-Score** | **{metrics['macro_f1_pct']:.2f}%** |",
        f"| **Weighted Precision** | **{metrics['weighted_precision_pct']:.2f}%** |",
        f"| **Weighted Recall** | **{metrics['weighted_recall_pct']:.2f}%** |",
        f"| **Weighted F1-Score** | **{metrics['weighted_f1_pct']:.2f}%** |",
        "",
        "## Per-Class Classification Metrics",
        "",
        "| Plant | Condition | Precision | Recall | F1-Score | Support |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for row in metrics["per_class"]:
        lines.append(
            f"| {row['plant_name']} | {row['disease_name']} | "
            f"{row['precision_pct']:.2f}% | {row['recall_pct']:.2f}% | "
            f"{row['f1_pct']:.2f}% | {row['support']} |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "- Confusion Matrix: `training/reports/confusion_matrix.png`",
            "- Training Curves: `training/reports/training_curves.png`",
            "- Raw Metrics JSON: `training/reports/evaluation_metrics.json`",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_model(batch_size: int = 64, num_workers: int = 4) -> dict:
    """Load saved checkpoint, run evaluation on the held-out test set, and export reports."""
    if not MODEL_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_WEIGHTS_PATH}'. Run `python training/train.py` first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=False)
    arch = checkpoint.get("architecture", "mobilenet_v2")
    class_names = checkpoint["class_names"]

    _, _, test_loader, split_classes, _ = load_splits_and_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    if split_classes != class_names:
        raise ValueError("Mismatch between saved model class names and dataset split class names.")

    model = build_model_architecture(architecture=arch, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    top3_correct = 0
    total = 0

    print(f"=== Evaluating {arch} on Test Set ({len(test_loader.dataset)} images) ===")
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            k = min(3, logits.size(1))
            top3 = torch.topk(logits, k=k, dim=1).indices
            top3_correct += sum(
                int(targets[i].item() in top3[i].tolist()) for i in range(targets.size(0))
            )
            total += targets.size(0)

            y_true.extend(targets.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    top3_acc = top3_correct / max(total, 1)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    cls_report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class = []
    for raw_label in class_names:
        stats = cls_report[raw_label]
        plant, disease, is_healthy = parse_class_label(raw_label)
        per_class.append(
            {
                "class_label": raw_label,
                "plant_name": plant,
                "disease_name": disease,
                "is_healthy": is_healthy,
                "precision_pct": round(stats["precision"] * 100.0, 2),
                "recall_pct": round(stats["recall"] * 100.0, 2),
                "f1_pct": round(stats["f1-score"] * 100.0, 2),
                "support": int(stats["support"]),
            }
        )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_path = TRAINING_REPORTS_DIR / "confusion_matrix.png"
    save_confusion_matrix_plot(cm, class_names, cm_path)

    # Load dataset split summary if available
    dataset_splits = {}
    summary_file = DATASET_DIR / "splits" / "dataset_summary.json"
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            dataset_splits = json.load(f).get("splits", {})

    metrics = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": arch,
        "dataset": "PlantVillage (Color)",
        "num_classes": len(class_names),
        "test_samples": int(total),
        "dataset_splits": dataset_splits,
        "accuracy_pct": round(accuracy * 100.0, 2),
        "top3_accuracy_pct": round(top3_acc * 100.0, 2),
        "macro_precision_pct": round(float(macro_p) * 100.0, 2),
        "macro_recall_pct": round(float(macro_r) * 100.0, 2),
        "macro_f1_pct": round(float(macro_f1) * 100.0, 2),
        "weighted_precision_pct": round(float(weighted_p) * 100.0, 2),
        "weighted_recall_pct": round(float(weighted_r) * 100.0, 2),
        "weighted_f1_pct": round(float(weighted_f1) * 100.0, 2),
        "best_val_accuracy_pct": checkpoint.get("best_val_accuracy"),
        "best_val_macro_f1_pct": checkpoint.get("best_val_macro_f1"),
        "per_class": per_class,
    }

    TRAINING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_REPORTS_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    write_markdown_report(metrics, TRAINING_REPORTS_DIR / "evaluation_report.md")

    # Compact metadata for live backend /api/health and About page
    summary_meta = {
        "architecture": arch,
        "dataset": "PlantVillage (Color)",
        "num_classes": len(class_names),
        "test_samples": int(total),
        "dataset_splits": dataset_splits,
        "test_accuracy_pct": metrics["accuracy_pct"],
        "top3_accuracy_pct": metrics["top3_accuracy_pct"],
        "macro_precision_pct": metrics["macro_precision_pct"],
        "macro_recall_pct": metrics["macro_recall_pct"],
        "macro_f1_pct": metrics["macro_f1_pct"],
        "weighted_f1_pct": metrics["weighted_f1_pct"],
        "evaluated_at_utc": metrics["evaluated_at_utc"],
    }
    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_meta, f, indent=2)

    print("\n=== Held-Out Test Set Evaluation Results ===")
    print(f"Test Accuracy:      {metrics['accuracy_pct']:.2f}%")
    print(f"Top-3 Accuracy:     {metrics['top3_accuracy_pct']:.2f}%")
    print(f"Macro Precision:    {metrics['macro_precision_pct']:.2f}%")
    print(f"Macro Recall:       {metrics['macro_recall_pct']:.2f}%")
    print(f"Macro F1-Score:     {metrics['macro_f1_pct']:.2f}%")
    print(f"Weighted F1-Score:  {metrics['weighted_f1_pct']:.2f}%")
    print(f"Confusion Matrix:   {cm_path}")
    print(f"Markdown Report:    {TRAINING_REPORTS_DIR / 'evaluation_report.md'}")
    print(f"Backend Metadata:   {MODEL_METADATA_PATH}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained Plant Disease Detection model.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    evaluate_model(batch_size=args.batch_size, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
