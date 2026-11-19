"""Confidence calibration: ECE, Brier score, validation-only temperature scaling, and plots.

IMPORTANT SCIENTIFIC GUARANTEE:
  - Temperature scaling (`T`) is fitted STRICTLY on validation-set logits by minimizing
    Negative Log-Likelihood (NLL) via LBFGS.
  - Test-set logits are NEVER used to fit temperature scaling; test metrics evaluate
    both uncalibrated (`T=1.0`) and validation-calibrated (`T=T_val`) probabilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None
import numpy as np
import torch
import torch.nn as nn


def softmax_numpy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature scaling."""
    scaled = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-4)
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.maximum(np.sum(exp_vals, axis=1, keepdims=True), 1e-12)


def compute_ece(
    y_true: Sequence[int],
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, Dict[str, Any]]:
    """Compute Expected Calibration Error (ECE) and per-bin reliability statistics."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(y_prob, dtype=np.float64)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true_arr).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins_info = []
    n_total = float(len(y_true_arr))

    for b in range(n_bins):
        low, high = bin_edges[b], bin_edges[b + 1]
        if b == 0:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences > low) & (confidences <= high)
        count = int(np.sum(in_bin))
        if count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            gap = abs(bin_acc - bin_conf)
            ece += (count / max(n_total, 1.0)) * gap
        else:
            bin_acc = 0.0
            bin_conf = 0.0
            gap = 0.0

        bins_info.append(
            {
                "bin_index": b,
                "low": round(float(low), 4),
                "high": round(float(high), 4),
                "count": count,
                "avg_confidence": round(bin_conf, 6),
                "avg_accuracy": round(bin_acc, 6),
                "abs_gap": round(gap, 6),
            }
        )

    return round(float(ece), 6), {
        "n_bins": n_bins,
        "ece": round(float(ece), 6),
        "mean_confidence": round(float(np.mean(confidences)), 6),
        "empirical_accuracy": round(float(np.mean(accuracies)), 6),
        "bins": bins_info,
    }


def compute_multiclass_brier_score(y_true: Sequence[int], y_prob: np.ndarray) -> float:
    """Compute multiclass Brier score: mean_i sum_c (p_{i,c} - y_{i,c})^2."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(y_prob, dtype=np.float64)
    n_samples, num_classes = probs.shape
    one_hot = np.zeros((n_samples, num_classes), dtype=np.float64)
    one_hot[np.arange(n_samples), y_true_arr] = 1.0
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return round(float(brier), 6)


def fit_temperature_scaling(
    val_logits: np.ndarray,
    val_targets: Sequence[int],
    max_iter: int = 100,
) -> Dict[str, float]:
    """Fit a single positive scalar temperature `T` strictly on VALIDATION logits via NLL."""
    logits_t = torch.tensor(np.asarray(val_logits, dtype=np.float32), dtype=torch.float32)
    targets_t = torch.tensor(np.asarray(val_targets, dtype=np.int64), dtype=torch.long)
    criterion = nn.CrossEntropyLoss()

    nll_before = float(criterion(logits_t, targets_t).item())
    # Parameterize log_temp so temperature is guaranteed strictly positive
    log_temp = nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=max_iter)

    def _closure():
        optimizer.zero_grad()
        temp = torch.exp(log_temp).clamp(min=0.05, max=20.0)
        loss = criterion(logits_t / temp, targets_t)
        loss.backward()
        return loss

    optimizer.step(_closure)
    optimal_temp = float(torch.exp(log_temp).clamp(min=0.05, max=20.0).item())
    with torch.no_grad():
        nll_after = float(criterion(logits_t / optimal_temp, targets_t).item())

    return {
        "fitted_on": "validation_split_only",
        "optimal_temperature": round(optimal_temp, 6),
        "val_nll_before": round(nll_before, 6),
        "val_nll_after": round(nll_after, 6),
    }


def save_calibration_plots(
    uncal_bins: Dict[str, Any],
    cal_bins: Dict[str, Any],
    uncal_confidences: np.ndarray,
    cal_confidences: np.ndarray,
    temperature: float,
    charts_dir: Path | str,
    model_name: str = "Model",
) -> Dict[str, str]:
    """Generate reliability diagram and confidence histogram plots."""
    out = Path(charts_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Reliability Diagram (Uncalibrated vs. Temperature-Scaled)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, binfo, title_tag in (
        (axes[0], uncal_bins, f"Uncalibrated (T=1.00, ECE={uncal_bins['ece']:.4f})"),
        (axes[1], cal_bins, f"Temperature-Scaled (T={temperature:.2f}, ECE={cal_bins['ece']:.4f})"),
    ):
        bins = binfo["bins"]
        centers = [(b["low"] + b["high"]) / 2.0 for b in bins]
        widths = [b["high"] - b["low"] for b in bins]
        accs = [b["avg_accuracy"] for b in bins]
        confs = [b["avg_confidence"] for b in bins]

        ax.plot([0, 1], [0, 1], "--", color="#64748b", lw=1.4, label="Perfect Calibration")
        ax.bar(centers, accs, width=widths, alpha=0.75, color="#059669", edgecolor="#065f46", label="Empirical Accuracy")
        ax.bar(
            centers,
            [abs(a - c) for a, c in zip(accs, confs)],
            bottom=[min(a, c) for a, c in zip(accs, confs)],
            width=widths,
            alpha=0.30,
            color="#e11d48",
            edgecolor="#be123c",
            hatch="//",
            label="Calibration Gap",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted Confidence")
        ax.set_ylabel("Observed Accuracy")
        ax.set_title(f"{model_name}\n{title_tag}", fontsize=10.5, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    rel_path = out / "reliability_diagram.png"
    plt.savefig(rel_path, dpi=160)
    plt.close(fig)

    # 2. Confidence Histogram
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(uncal_confidences, bins=20, range=(0, 1), alpha=0.55, color="#0284c7", label="Uncalibrated (T=1.00)")
    ax.hist(cal_confidences, bins=20, range=(0, 1), alpha=0.55, color="#059669", label=f"Calibrated (T={temperature:.2f})")
    ax.set_title(f"{model_name} — Test Prediction Confidence Distribution", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Maximum Predicted Probability (Confidence)")
    ax.set_ylabel("Number of Test Images")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    hist_path = out / "confidence_histogram.png"
    plt.savefig(hist_path, dpi=160)
    plt.close(fig)

    return {
        "reliability_diagram": str(rel_path),
        "confidence_histogram": str(hist_path),
    }


def evaluate_calibration(
    val_logits: np.ndarray,
    val_targets: Sequence[int],
    test_logits: np.ndarray,
    test_targets: Sequence[int],
    charts_dir: Path | str,
    model_name: str = "Model",
    n_bins: int = 15,
) -> Dict[str, Any]:
    """Fit temperature scaling on validation set only, then evaluate calibration on test set."""
    temp_info = fit_temperature_scaling(val_logits, val_targets)
    opt_temp = temp_info["optimal_temperature"]

    test_probs_uncal = softmax_numpy(test_logits, temperature=1.0)
    test_probs_cal = softmax_numpy(test_logits, temperature=opt_temp)

    ece_uncal, bins_uncal = compute_ece(test_targets, test_probs_uncal, n_bins=n_bins)
    ece_cal, bins_cal = compute_ece(test_targets, test_probs_cal, n_bins=n_bins)

    brier_uncal = compute_multiclass_brier_score(test_targets, test_probs_uncal)
    brier_cal = compute_multiclass_brier_score(test_targets, test_probs_cal)

    plots = save_calibration_plots(
        uncal_bins=bins_uncal,
        cal_bins=bins_cal,
        uncal_confidences=np.max(test_probs_uncal, axis=1),
        cal_confidences=np.max(test_probs_cal, axis=1),
        temperature=opt_temp,
        charts_dir=charts_dir,
        model_name=model_name,
    )

    return {
        "temperature_scaling": temp_info,
        "test_uncalibrated": {
            "ece": ece_uncal,
            "brier_score": brier_uncal,
            "mean_confidence": bins_uncal["mean_confidence"],
            "empirical_accuracy": bins_uncal["empirical_accuracy"],
        },
        "test_calibrated": {
            "temperature": opt_temp,
            "ece": ece_cal,
            "brier_score": brier_cal,
            "mean_confidence": bins_cal["mean_confidence"],
            "empirical_accuracy": bins_cal["empirical_accuracy"],
        },
        "ece_reduction": round(ece_uncal - ece_cal, 6),
        "brier_reduction": round(brier_uncal - brier_cal, 6),
        "plots": plots,
    }
