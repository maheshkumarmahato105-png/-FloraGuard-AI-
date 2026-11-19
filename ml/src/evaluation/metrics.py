"""Comprehensive multiclass classification metrics with safe missing-class handling (Phase 5).

Computes:
  - Accuracy, Top-3 Accuracy
  - Macro & Weighted Precision, Recall, F1-Score
  - Per-class Precision, Recall, F1-Score, Support
  - Cohen's Kappa
  - Matthews Correlation Coefficient (MCC)
  - Multiclass One-vs-Rest (OvR) ROC-AUC (macro & weighted) with safe handling and
    explicit documentation whenever classes are missing or ROC-AUC is undefined.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_multiclass_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Optional[np.ndarray],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    """Compute complete multiclass evaluation metrics and per-class statistics."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    n_samples = int(len(y_true_arr))
    num_classes = len(class_names)

    if n_samples == 0:
        raise ValueError("Cannot compute metrics on an empty evaluation set.")
    if len(y_pred_arr) != n_samples:
        raise ValueError("y_true and y_pred must have the same length.")

    present_classes = sorted(set(y_true_arr.tolist()))
    missing_classes = [
        class_names[i] for i in range(num_classes) if i not in present_classes
    ]

    acc = float(accuracy_score(y_true_arr, y_pred_arr))

    # Macro metrics evaluated over present classes if some classes are absent (documented clearly)
    eval_labels = present_classes if missing_classes else list(range(num_classes))

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=eval_labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=eval_labels, average="weighted", zero_division=0
    )

    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )

    # Cohen's Kappa & MCC (can be undefined if only 1 class is present)
    undefined_metrics: Dict[str, str] = {}
    if len(present_classes) < 2:
        kappa: Optional[float] = None
        mcc: Optional[float] = None
        undefined_metrics["cohens_kappa"] = "Undefined when fewer than 2 ground-truth classes are present."
        undefined_metrics["matthews_corrcoef"] = "Undefined when fewer than 2 ground-truth classes are present."
    else:
        kappa = round(float(cohen_kappa_score(y_true_arr, y_pred_arr)), 6)
        mcc = round(float(matthews_corrcoef(y_true_arr, y_pred_arr)), 6)

    # Top-3 accuracy & Multiclass OvR ROC-AUC
    top3_acc: Optional[float] = None
    roc_auc_macro: Optional[float] = None
    roc_auc_weighted: Optional[float] = None

    if y_prob is not None:
        probs = np.asarray(y_prob, dtype=np.float64)
        if probs.ndim != 2 or probs.shape != (n_samples, num_classes):
            raise ValueError(
                f"Expected y_prob shape ({n_samples}, {num_classes}), got {probs.shape}."
            )
        k = min(3, num_classes)
        top_k_idx = np.argsort(probs, axis=1)[:, -k:]
        top3_hits = [int(y_true_arr[i] in top_k_idx[i]) for i in range(n_samples)]
        top3_acc = round(float(np.mean(top3_hits)), 6)

        if len(present_classes) < 2:
            undefined_metrics["roc_auc_ovr"] = "Undefined when fewer than 2 ground-truth classes exist."
        elif missing_classes:
            # Safely compute OvR ROC-AUC over present classes only (renormalized probability slice)
            try:
                sub_probs = probs[:, present_classes]
                row_sums = np.maximum(sub_probs.sum(axis=1, keepdims=True), 1e-12)
                sub_probs = sub_probs / row_sums
                remap = {orig: new_i for new_i, orig in enumerate(present_classes)}
                remapped_true = np.array([remap[int(c)] for c in y_true_arr], dtype=np.int64)
                if len(present_classes) == 2:
                    auc_val = float(roc_auc_score(remapped_true, sub_probs[:, 1]))
                    roc_auc_macro = round(auc_val, 6)
                    roc_auc_weighted = round(auc_val, 6)
                else:
                    roc_auc_macro = round(
                        float(roc_auc_score(remapped_true, sub_probs, multi_class="ovr", average="macro")),
                        6,
                    )
                    roc_auc_weighted = round(
                        float(roc_auc_score(remapped_true, sub_probs, multi_class="ovr", average="weighted")),
                        6,
                    )
                undefined_metrics["roc_auc_ovr_note"] = (
                    f"Computed over {len(present_classes)}/{num_classes} present classes "
                    f"({len(missing_classes)} classes absent in this split)."
                )
            except ValueError as exc:
                undefined_metrics["roc_auc_ovr"] = f"Undefined: {exc}"
        else:
            try:
                if num_classes == 2:
                    auc_val = float(roc_auc_score(y_true_arr, probs[:, 1]))
                    roc_auc_macro = round(auc_val, 6)
                    roc_auc_weighted = round(auc_val, 6)
                else:
                    roc_auc_macro = round(
                        float(roc_auc_score(y_true_arr, probs, multi_class="ovr", average="macro")),
                        6,
                    )
                    roc_auc_weighted = round(
                        float(roc_auc_score(y_true_arr, probs, multi_class="ovr", average="weighted")),
                        6,
                    )
            except ValueError as exc:
                undefined_metrics["roc_auc_ovr"] = f"Undefined: {exc}"
    else:
        undefined_metrics["roc_auc_ovr"] = "Probabilities (y_prob) were not provided."

    per_class: List[Dict[str, Any]] = []
    for idx, cls_name in enumerate(class_names):
        sup = int(per_sup[idx])
        per_class.append(
            {
                "class_index": idx,
                "class_name": str(cls_name),
                "present_in_split": sup > 0,
                "support": sup,
                "precision": round(float(per_p[idx]), 6) if sup > 0 else None,
                "recall": round(float(per_r[idx]), 6) if sup > 0 else None,
                "f1_score": round(float(per_f1[idx]), 6) if sup > 0 else None,
            }
        )

    return {
        "num_samples": n_samples,
        "num_classes_total": num_classes,
        "num_classes_present": len(present_classes),
        "missing_classes": missing_classes,
        "accuracy": round(acc, 6),
        "top3_accuracy": top3_acc,
        "macro_precision": round(float(macro_p), 6),
        "macro_recall": round(float(macro_r), 6),
        "macro_f1": round(float(macro_f1), 6),
        "weighted_precision": round(float(weighted_p), 6),
        "weighted_recall": round(float(weighted_r), 6),
        "weighted_f1": round(float(weighted_f1), 6),
        "cohens_kappa": kappa,
        "matthews_corrcoef": mcc,
        "roc_auc_ovr_macro": roc_auc_macro,
        "roc_auc_ovr_weighted": roc_auc_weighted,
        "undefined_metrics": undefined_metrics,
        "per_class": per_class,
    }
