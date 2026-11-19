"""Raw and normalized confusion matrix computation and high-resolution heatmap generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def _shorten_class_label(name: str, max_len: int = 28) -> str:
    cleaned = name.replace("__", ": ").replace("_", " ").strip()
    return cleaned if len(cleaned) <= max_len else (cleaned[: max_len - 1] + "…")


def compute_confusion_matrices(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Compute raw count and row-normalized confusion matrices + summary stats."""
    labels = list(range(len(class_names)))
    raw_cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = raw_cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(raw_cm.astype(np.float64), np.maximum(row_sums, 1.0))

    total = int(raw_cm.sum())
    correct = int(np.trace(raw_cm))
    misclassified = total - correct

    # Extract top off-diagonal confused pairs
    confused_pairs: List[Dict[str, Any]] = []
    n = len(class_names)
    for i in range(n):
        for j in range(n):
            if i != j and raw_cm[i, j] > 0:
                confused_pairs.append(
                    {
                        "true_class": str(class_names[i]),
                        "predicted_class": str(class_names[j]),
                        "count": int(raw_cm[i, j]),
                        "true_class_error_rate": round(float(norm_cm[i, j]), 4),
                    }
                )
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    summary = {
        "total_predictions": total,
        "correct_predictions": correct,
        "misclassified_predictions": misclassified,
        "accuracy_from_cm": round(correct / max(total, 1), 6),
        "top_confused_pairs": confused_pairs[:25],
    }
    return raw_cm, norm_cm, summary


def save_confusion_matrix_artifacts(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    charts_dir: Path | str,
    metrics_dir: Path | str,
    title_prefix: str = "Model",
) -> Dict[str, Any]:
    """Compute and export raw/normalized confusion matrix CSVs and high-resolution heatmaps."""
    charts_path = Path(charts_dir)
    metrics_path = Path(metrics_dir)
    charts_path.mkdir(parents=True, exist_ok=True)
    metrics_path.mkdir(parents=True, exist_ok=True)

    raw_cm, norm_cm, summary = compute_confusion_matrices(y_true, y_pred, class_names)
    short_labels = [_shorten_class_label(c) for c in class_names]
    n = len(class_names)

    # Save CSVs preserving full canonical class names
    pd.DataFrame(raw_cm, index=class_names, columns=class_names).to_csv(
        metrics_path / "confusion_matrix_raw.csv"
    )
    pd.DataFrame(np.round(norm_cm, 4), index=class_names, columns=class_names).to_csv(
        metrics_path / "confusion_matrix_normalized.csv"
    )

    def _plot_heatmap(matrix: np.ndarray, out_file: Path, title: str, normalized: bool) -> None:
        fig_side = max(10, min(26, int(n * 0.56)))
        fig, ax = plt.subplots(figsize=(fig_side, fig_side * 0.88))
        im = ax.imshow(
            matrix,
            interpolation="nearest",
            cmap="Greens" if normalized else "Blues",
            vmin=0.0 if normalized else 0,
            vmax=1.0 if normalized else max(1, int(matrix.max())),
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
        cbar.set_label("Row-Normalized Recall" if normalized else "Sample Count", fontsize=10)

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        font_sz = 8 if n <= 20 else 7
        ax.set_xticklabels(short_labels, rotation=90, fontsize=font_sz)
        ax.set_yticklabels(short_labels, fontsize=font_sz)
        ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Class", fontsize=11, fontweight="bold")
        ax.set_title(
            f"{title}\nCorrect: {summary['correct_predictions']:,} | Misclassified: {summary['misclassified_predictions']:,}",
            fontsize=12.5,
            fontweight="bold",
            pad=14,
        )

        # Annotate cells when compact (<= 15 classes) or annotate non-zero diagonal/errors cleanly
        if n <= 15:
            thresh = matrix.max() * 0.55
            for i in range(n):
                for j in range(n):
                    val = matrix[i, j]
                    if val > 0 or i == j:
                        txt = f"{val:.2f}" if normalized else f"{int(val)}"
                        ax.text(
                            j,
                            i,
                            txt,
                            ha="center",
                            va="center",
                            fontsize=7.5,
                            color="white" if val > thresh else "#0f172a",
                        )

        plt.tight_layout()
        plt.savefig(out_file, dpi=180)
        plt.close(fig)

    raw_png = charts_path / "confusion_matrix.png"
    norm_png = charts_path / "normalized_confusion_matrix.png"
    _plot_heatmap(raw_cm, raw_png, f"{title_prefix} — Raw Confusion Matrix", normalized=False)
    _plot_heatmap(norm_cm, norm_png, f"{title_prefix} — Normalized Confusion Matrix", normalized=True)

    summary["raw_png"] = str(raw_png)
    summary["normalized_png"] = str(norm_png)
    summary["raw_csv"] = str(metrics_path / "confusion_matrix_raw.csv")
    summary["normalized_csv"] = str(metrics_path / "confusion_matrix_normalized.csv")
    return summary
