"""Classification metrics computation, CSV logging, overfitting tracking, and curve plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Dict[str, float]:
    """Compute accuracy, macro precision, macro recall, and macro F1-score."""
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on empty predictions.")

    acc = float(accuracy_score(y_true, y_pred))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": round(acc, 6),
        "macro_precision": round(float(macro_p), 6),
        "macro_recall": round(float(macro_r), 6),
        "macro_f1": round(float(macro_f1), 6),
    }


class MetricsTracker:
    """Tracks per-epoch metrics, detects overfitting signals, exports CSV, and plots curves."""

    def __init__(self, initial_records: List[Dict[str, Any]] | None = None) -> None:
        self.records: List[Dict[str, Any]] = list(initial_records) if initial_records else []

    def add_epoch(self, record: Dict[str, Any]) -> None:
        # Compute overfitting gap indicators automatically
        enriched = dict(record)
        enriched["acc_gap_train_minus_val"] = round(
            float(record["train_accuracy"]) - float(record["val_accuracy"]), 6
        )
        enriched["loss_gap_val_minus_train"] = round(
            float(record["val_loss"]) - float(record["train_loss"]), 6
        )
        self.records.append(enriched)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def save_csv(self, csv_path: Path | str) -> Path:
        out = Path(csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(out, index=False)
        return out

    def detect_overfitting_warning(self) -> Dict[str, Any]:
        """Inspect train/val divergence on the latest epoch for overfitting diagnostics."""
        if not self.records:
            return {"flagged": False, "reason": "No epochs completed."}
        last = self.records[-1]
        acc_gap = float(last["acc_gap_train_minus_val"])
        loss_gap = float(last["loss_gap_val_minus_train"])
        flagged = (acc_gap > 0.10) or (loss_gap > 0.35)
        return {
            "flagged": flagged,
            "latest_acc_gap": round(acc_gap, 4),
            "latest_loss_gap": round(loss_gap, 4),
            "summary": (
                f"Potential overfitting detected (train-val acc gap={acc_gap*100:.2f}%, "
                f"val-train loss gap={loss_gap:.4f})."
                if flagged
                else f"No severe overfitting signal (train-val acc gap={acc_gap*100:.2f}%, "
                f"val-train loss gap={loss_gap:.4f})."
            ),
        }

    def save_training_charts(self, charts_dir: Path | str, run_prefix: str = "training") -> Dict[str, Path]:
        """Generate training/validation loss, accuracy, macro-F1, and learning-rate curves."""
        if not self.records:
            return {}

        out_dir = Path(charts_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        epochs = df["epoch"].tolist()

        saved: Dict[str, Path] = {}

        # 1. Combined 2x2 overview figure (Loss, Accuracy, Macro-F1, Learning Rate)
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        axes[0, 0].plot(epochs, df["train_loss"], marker="o", label="Train Loss", color="#059669", lw=2)
        axes[0, 0].plot(epochs, df["val_loss"], marker="s", label="Val Loss", color="#dc2626", lw=2)
        axes[0, 0].set_title("Training vs. Validation Loss", fontweight="bold")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Cross-Entropy Loss")
        axes[0, 0].grid(True, alpha=0.25)
        axes[0, 0].legend()

        axes[0, 1].plot(epochs, df["train_accuracy"] * 100.0, marker="o", label="Train Accuracy (%)", color="#059669", lw=2)
        axes[0, 1].plot(epochs, df["val_accuracy"] * 100.0, marker="s", label="Val Accuracy (%)", color="#2563eb", lw=2)
        axes[0, 1].set_title("Training vs. Validation Accuracy", fontweight="bold")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy (%)")
        axes[0, 1].grid(True, alpha=0.25)
        axes[0, 1].legend()

        axes[1, 0].plot(epochs, df["val_macro_f1"] * 100.0, marker="^", label="Val Macro-F1 (%)", color="#d97706", lw=2)
        axes[1, 0].plot(epochs, df["val_macro_precision"] * 100.0, linestyle="--", label="Val Macro Precision (%)", color="#0284c7", lw=1.5)
        axes[1, 0].plot(epochs, df["val_macro_recall"] * 100.0, linestyle=":", label="Val Macro Recall (%)", color="#7c3aed", lw=1.5)
        axes[1, 0].set_title("Validation Macro-F1, Precision & Recall", fontweight="bold")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Score (%)")
        axes[1, 0].grid(True, alpha=0.25)
        axes[1, 0].legend()

        axes[1, 1].plot(epochs, df["learning_rate"], marker="d", label="Learning Rate", color="#0f766e", lw=2)
        axes[1, 1].set_title("Learning Rate Schedule", fontweight="bold")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].grid(True, alpha=0.25)
        axes[1, 1].legend()

        plt.suptitle(f"PlantDiseaseAI Training Diagnostics ({run_prefix})", fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        overview_path = out_dir / f"{run_prefix}_curves.png"
        plt.savefig(overview_path, dpi=160)
        plt.close(fig)
        saved["overview_curves"] = overview_path

        # 2. Individual focused plots for research figures
        def _single_plot(fname: str, title: str, ylabel: str, series: List[tuple[str, Any, str]]) -> Path:
            f, ax = plt.subplots(figsize=(7.5, 4.5))
            for lbl, vals, col in series:
                ax.plot(epochs, vals, marker="o", label=lbl, color=col, lw=2)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.legend()
            plt.tight_layout()
            p = out_dir / fname
            plt.savefig(p, dpi=150)
            plt.close(f)
            return p

        saved["loss_curve"] = _single_plot(
            f"{run_prefix}_loss.png",
            "Training & Validation Loss",
            "Cross-Entropy Loss",
            [("Train Loss", df["train_loss"], "#059669"), ("Val Loss", df["val_loss"], "#dc2626")],
        )
        saved["accuracy_curve"] = _single_plot(
            f"{run_prefix}_accuracy.png",
            "Training & Validation Accuracy",
            "Accuracy (%)",
            [
                ("Train Accuracy", df["train_accuracy"] * 100.0, "#059669"),
                ("Val Accuracy", df["val_accuracy"] * 100.0, "#2563eb"),
            ],
        )
        saved["macro_f1_curve"] = _single_plot(
            f"{run_prefix}_macro_f1.png",
            "Validation Macro-F1 Progression",
            "Macro-F1 (%)",
            [("Val Macro-F1", df["val_macro_f1"] * 100.0, "#d97706")],
        )
        saved["lr_curve"] = _single_plot(
            f"{run_prefix}_learning_rate.png",
            "Learning Rate Schedule",
            "Learning Rate",
            [("Learning Rate", df["learning_rate"], "#0f766e")],
        )
        return saved
