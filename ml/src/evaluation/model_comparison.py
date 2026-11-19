"""Multi-architecture comparison tables, CSV/JSON reports, and visualization charts (Phase 5).

Compares Simple CNN, AgriVision-CNN, and ResNet18 under identical evaluation conditions
across accuracy, macro precision/recall/F1, weighted F1, parameters, model file size,
latency, throughput, and GPU VRAM — without automatically declaring a winner on accuracy alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None
import numpy as np
import pandas as pd


def build_model_comparison(
    model_records: List[Dict[str, Any]],
    metrics_dir: Path | str,
    charts_dir: Path | str,
) -> Dict[str, Any]:
    """Export side-by-side comparison CSV, JSON, and multi-metric comparison charts."""
    m_dir = Path(metrics_dir)
    c_dir = Path(charts_dir)
    m_dir.mkdir(parents=True, exist_ok=True)
    c_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(model_records)
    csv_path = m_dir / "model_comparison.csv"
    json_path = m_dir / "model_comparison.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(model_records, f, indent=2)

    charts: Dict[str, str] = {}
    if len(df) == 0:
        return {"csv_path": str(csv_path), "json_path": str(json_path), "charts": charts}

    labels = df["architecture_name"].tolist()
    x = np.arange(len(labels))

    # 1. Efficiency & Hardware Comparison Chart (always available from structural + HW benchmark)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    colors = ["#0284c7", "#059669", "#d97706"][: len(labels)]

    params_m = [float(p) / 1e6 for p in df["total_params"].fillna(0)]
    axes[0].bar(x, params_m, color=colors, width=0.55)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, fontsize=9)
    axes[0].set_ylabel("Parameters (Millions)")
    axes[0].set_title("Model Parameter Count (M)", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.25)

    cpu_lat = df["cpu_mean_latency_ms"].fillna(0).tolist()
    axes[1].bar(x, cpu_lat, color=colors, width=0.55)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, fontsize=9)
    axes[1].set_ylabel("Latency (ms / image)")
    axes[1].set_title("CPU Inference Latency (bs=1, FP32)", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.25)

    if "gpu_bs16_fp16_throughput_ips" in df.columns and df["gpu_bs16_fp16_throughput_ips"].notna().any():
        gpu_tps = df["gpu_bs16_fp16_throughput_ips"].fillna(0).tolist()
        axes[2].bar(x, gpu_tps, color=colors, width=0.55)
        axes[2].set_ylabel("Throughput (images / sec)")
        axes[2].set_title("GPU Throughput (bs=16, AMP FP16)", fontweight="bold")
    else:
        sizes_mb = df["model_size_mb"].fillna(0).tolist()
        axes[2].bar(x, sizes_mb, color=colors, width=0.55)
        axes[2].set_ylabel("Size (MB)")
        axes[2].set_title("FP32 Model Footprint (MB)", fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=15, fontsize=9)
    axes[2].grid(axis="y", alpha=0.25)

    plt.suptitle("PlantDiseaseAI — Architectural & Hardware Efficiency Comparison", fontsize=12.5, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    eff_chart = c_dir / "model_comparison_efficiency.png"
    plt.savefig(eff_chart, dpi=160)
    plt.close(fig)
    charts["efficiency_chart"] = str(eff_chart)

    # 2. Classification Metrics Comparison Chart (ONLY when real trained test metrics exist)
    evaluated_df = df[df["test_macro_f1"].notna()] if "test_macro_f1" in df.columns else pd.DataFrame()
    if len(evaluated_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        e_labels = evaluated_df["architecture_name"].tolist()
        ex = np.arange(len(e_labels))
        width = 0.24
        ax.bar(ex - width, evaluated_df["test_accuracy"] * 100.0, width, label="Test Accuracy (%)", color="#0284c7")
        ax.bar(ex, evaluated_df["test_macro_f1"] * 100.0, width, label="Macro F1 (%)", color="#059669")
        ax.bar(ex + width, evaluated_df["test_weighted_f1"] * 100.0, width, label="Weighted F1 (%)", color="#d97706")
        ax.set_xticks(ex)
        ax.set_xticklabels(e_labels)
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Internal Test Set Classification Comparison (Trained Checkpoints)", fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        plt.tight_layout()
        met_chart = c_dir / "model_comparison_metrics.png"
        plt.savefig(met_chart, dpi=160)
        plt.close(fig)
        charts["metrics_chart"] = str(met_chart)

    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "charts": charts,
    }
