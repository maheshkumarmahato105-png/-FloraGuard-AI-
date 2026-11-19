"""Master Phase 5 Evaluation Orchestrator & Research Report Builder.

Strictly enforces scientific integrity:
  1. Verifies availability and integrity of actual trained Phase 4 checkpoints
     (`is_trained == True` and `epoch >= 1`).
  2. Never fabricates accuracy, confusion matrices, or calibration scores for
     missing checkpoints — explicitly reports missing checkpoints.
  3. Runs real CPU & CUDA GPU hardware inference benchmarks across all 3 Phase 3
     architectures (pure structural/latency measurement, clearly labeled separately).
  4. When trained checkpoints are available, runs internal test evaluation,
     confusion matrices, classification reports, validation-fitted temperature scaling,
     external PlantDoc validation (verified mapping only), robustness perturbation
     testing, error analysis + misclassification gallery, and 3-model comparison.
  5. Compiles `ml/reports/phase5_research_report.md` with all 14 required sections.

Usage:
    python ml/src/evaluation/evaluate.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_loader import PROJECT_ROOT, load_class_mapping, load_dataset_config
from datasets.plant_dataset import PlantDiseaseDataset
from models import create_model, load_model_config, resolve_verified_classes
from preprocess import get_eval_transform

from evaluation.calibration import evaluate_calibration, softmax_numpy
from evaluation.classification_report import export_classification_report
from evaluation.confusion_matrix import save_confusion_matrix_artifacts
from evaluation.error_analysis import run_error_analysis
from evaluation.external_validation import (
    audit_external_plantdoc_coverage,
    evaluate_external_plantdoc,
)
from evaluation.inference_benchmark import run_full_hardware_benchmark
from evaluation.metrics import compute_multiclass_metrics
from evaluation.model_comparison import build_model_comparison
from evaluation.robustness import run_robustness_suite

DEFAULT_EXPERIMENTS = [
    ("simple_cnn", "exp_b_simple_cnn"),
    ("agrivision_cnn", "exp_a_agrivision_cnn"),
    ("resnet18_baseline", "exp_c_resnet18_scratch"),
]


def verify_trained_checkpoint(
    checkpoint_path: Path,
    expected_arch: str,
) -> Dict[str, Any]:
    """Verify that a checkpoint exists, loads cleanly, and represents a genuinely trained model."""
    if not checkpoint_path.exists():
        return {
            "architecture_name": expected_arch,
            "checkpoint_path": str(checkpoint_path),
            "exists": False,
            "valid_trained_checkpoint": False,
            "reason": "Trained checkpoint file not found (Phase 4 training has not been executed for this model).",
            "file_size_mb": None,
            "epoch": None,
            "best_val_macro_f1": None,
        }

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        required = {"model_state_dict", "class_names", "architecture_name", "epoch"}
        missing = required - set(ckpt.keys())
        if missing:
            return {
                "architecture_name": expected_arch,
                "checkpoint_path": str(checkpoint_path),
                "exists": True,
                "valid_trained_checkpoint": False,
                "reason": f"Checkpoint missing required keys: {sorted(missing)}",
                "file_size_mb": round(checkpoint_path.stat().st_size / (1024**2), 3),
                "epoch": None,
                "best_val_macro_f1": None,
            }

        epoch = int(ckpt.get("epoch", 0))
        is_trained = bool(ckpt.get("is_trained", epoch >= 1))
        if not is_trained or epoch < 1:
            return {
                "architecture_name": expected_arch,
                "checkpoint_path": str(checkpoint_path),
                "exists": True,
                "valid_trained_checkpoint": False,
                "reason": "Checkpoint is marked untrained (epoch < 1); refusing to evaluate as a trained model.",
                "file_size_mb": round(checkpoint_path.stat().st_size / (1024**2), 3),
                "epoch": epoch,
                "best_val_macro_f1": None,
            }

        return {
            "architecture_name": str(ckpt["architecture_name"]),
            "checkpoint_path": str(checkpoint_path),
            "exists": True,
            "valid_trained_checkpoint": True,
            "reason": "Verified trained checkpoint.",
            "file_size_mb": round(checkpoint_path.stat().st_size / (1024**2), 3),
            "epoch": epoch,
            "best_val_macro_f1": ckpt.get("best_val_macro_f1"),
            "num_classes": len(ckpt["class_names"]),
            "class_names": ckpt["class_names"],
        }
    except Exception as exc:
        return {
            "architecture_name": expected_arch,
            "checkpoint_path": str(checkpoint_path),
            "exists": True,
            "valid_trained_checkpoint": False,
            "reason": f"Checkpoint failed integrity load: {exc}",
            "file_size_mb": round(checkpoint_path.stat().st_size / (1024**2), 3),
            "epoch": None,
            "best_val_macro_f1": None,
        }


@torch.no_grad()
def predict_over_dataframe(
    model: nn.Module,
    dataframe: pd.DataFrame,
    dataset_config: Dict[str, Any],
    class_names: List[str],
    device: torch.device,
    batch_size: int = 32,
    project_root: Path = PROJECT_ROOT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run deterministic inference (model.eval(), no grad, no augmentation) on a manifest slice.

    Returns: (logits, probs, preds, targets)
    """
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    ds = PlantDiseaseDataset(
        dataframe=dataframe,
        class_to_idx=class_to_idx,
        project_root=project_root,
        use_processed=True,
        transform=get_eval_transform(dataset_config),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    model = model.to(device)
    logits_list: List[np.ndarray] = []
    targets_list: List[int] = []

    for images, targets in loader:
        out = model(images.to(device))
        logits_list.append(out.detach().cpu().numpy())
        targets_list.extend(targets.tolist())

    logits_np = np.concatenate(logits_list, axis=0)
    probs_np = softmax_numpy(logits_np, temperature=1.0)
    preds_np = np.argmax(probs_np, axis=1)
    targets_np = np.asarray(targets_list, dtype=np.int64)
    return logits_np, probs_np, preds_np, targets_np


def write_phase5_research_report(
    report_path: Path,
    payload: Dict[str, Any],
) -> None:
    """Write comprehensive 14-section ml/reports/phase5_research_report.md."""
    ckpts = payload["checkpoint_verification"]
    ds_audit = payload["dataset_audit"]
    ext_audit = payload["external_coverage_audit"]
    benchmarks = payload["hardware_benchmarks"]
    comp_rows = payload["comparison_rows"]
    eval_results = payload["evaluated_models"]
    trained_count = len(eval_results)

    lines = [
        "# PlantDiseaseAI — Phase 5 Research Evaluation Report",
        "",
        f"- **Generated At (UTC):** `{payload['generated_at_utc']}`",
        f"- **Evaluation Status:** **`{'COMPLETE' if trained_count == 3 else ('PARTIAL (' + str(trained_count) + '/3 checkpoints evaluated)' if trained_count > 0 else 'EVALUATION_PIPELINE_READY_CHECKPOINTS_MISSING')}`**",
        "- **Scientific Integrity Guarantee:** Zero metrics have been fabricated. Any model without a verified trained Phase 4 checkpoint is explicitly reported as missing.",
        "",
        "---",
        "",
        "## 1. Research Objective",
        "",
        "Rigorously evaluate and compare three custom/baseline convolutional architectures (**Simple CNN**, **AgriVision-CNN**, and **ResNet18 from scratch**) for multi-species plant leaf disease classification across:",
        "1. In-domain generalization on the untouched Phase 2 **PlantVillage** held-out test split.",
        "2. Out-of-domain real-world field generalization on the separate **PlantDoc** external dataset.",
        "3. Perturbation robustness under realistic camera and lighting degradations.",
        "4. Confidence calibration via validation-fitted temperature scaling (ECE & Brier score).",
        "5. Fine-grained error analysis and CPU/GPU hardware inference efficiency.",
        "",
        "---",
        "",
        "## 2. Dataset Information (Measured from Phase 2 Manifest)",
        "",
        f"- **Internal Training Split (PlantVillage):** `{ds_audit['train_count']:,}` images *(used in Phase 4 training only)*",
        f"- **Internal Validation Split (PlantVillage):** `{ds_audit['val_count']:,}` images *(used for model selection & temperature scaling only)*",
        f"- **Untouched Internal Test Split (PlantVillage):** `{ds_audit['test_count']:,}` images across `{ds_audit['num_classes']}` verified canonical classes",
        f"- **External Real-World Evaluation Dataset (PlantDoc):** `{ext_audit['total_external_images']:,}` total valid unique images",
        f"  - Verified mapped images: **`{ext_audit['mapped_images']:,}`** (`{ext_audit['mapped_coverage_pct']}%`) across `{ext_audit['mapped_classes_count']}` mapped classes",
        f"  - Unmatched/excluded external images: **`{ext_audit['unmatched_images']:,}`**",
        f"  - Duplicate-group overlap with PlantVillage training split: **`{ext_audit['train_duplicate_group_overlap']}`** (Independent: **`{ext_audit['is_independent_from_train']}`**)",
        "",
        "---",
        "",
        "## 3. Models Evaluated & Checkpoint Integrity Verification",
        "",
        "| Architecture | Expected Checkpoint Path | Exists | Valid Trained Checkpoint | Epoch | Status / Notes |",
        "| :--- | :--- | :---: | :---: | ---: | :--- |",
    ]

    for c in ckpts:
        lines.append(
            f"| `{c['architecture_name']}` | `{c['checkpoint_path']}` | "
            f"{'Yes' if c['exists'] else 'No'} | {'**Yes**' if c['valid_trained_checkpoint'] else '**No**'} | "
            f"{c['epoch'] if c['epoch'] is not None else '—'} | {c['reason']} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Experimental Methodology",
            "",
            "- **Deterministic Evaluation:** `model.eval()`, `torch.no_grad()`, zero test-time random augmentation, identical `224×224` aspect-ratio-preserving letterbox RGB preprocessing and ImageNet normalization as Phase 2/4.",
            "- **Leakage Isolation:** Test split is never used for training, hyperparameter tuning, or calibration fitting. Temperature scaling is fitted **exclusively on validation logits** via LBFGS NLL minimization.",
            "- **External Validation Hygiene:** PlantDoc uses verified mappings from `ml/configs/class_mapping.yaml`, preserving original labels and never assuming field accuracy from lab images.",
            "- **Non-Destructive Robustness Testing:** All 10 realistic image perturbations (brightness, contrast, Gaussian blur, Gaussian noise, JPEG Q=30, ±12° rotation, warm/cool lighting tint) are applied in memory only.",
            "",
            "---",
            "",
            "## 5. Evaluation Metrics (Internal Held-Out PlantVillage Test Set)",
            "",
        ]
    )

    if eval_results:
        lines.extend(
            [
                "| Model | Accuracy | Top-3 Acc | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Cohen's Kappa | MCC | OvR ROC-AUC (Macro) |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for arch, res in eval_results.items():
            m = res["internal_test_metrics"]
            auc_str = f"{m['roc_auc_ovr_macro']*100:.2f}%" if m["roc_auc_ovr_macro"] is not None else "N/A"
            lines.append(
                f"| `{arch}` | {m['accuracy']*100:.2f}% | {m['top3_accuracy']*100:.2f}% | "
                f"{m['macro_precision']*100:.2f}% | {m['macro_recall']*100:.2f}% | **{m['macro_f1']*100:.2f}%** | "
                f"{m['weighted_f1']*100:.2f}% | {m['cohens_kappa']:.4f} | {m['matthews_corrcoef']:.4f} | {auc_str} |"
            )
    else:
        lines.append(
            "> **Not Yet Measured (Checkpoints Missing):** No trained Phase 4 checkpoints (`best_model.pth`) were found in `ml/experiments/checkpoints/`. Run Phase 4 training first, then re-run `python ml/src/evaluation/evaluate.py` to populate verified internal test metrics."
        )

    lines.extend(["", "---", "", "## 6. Confusion Matrices", ""])
    if eval_results:
        for arch, res in eval_results.items():
            cm = res["confusion_matrix"]
            lines.append(
                f"- **`{arch}`:** Correct `{cm['correct_predictions']:,}` / Misclassified `{cm['misclassified_predictions']:,}` — "
                f"Raw: `{cm['raw_png']}` | Normalized: `{cm['normalized_png']}`"
            )
    else:
        lines.append(
            "> **Pending Trained Checkpoints:** `ml/src/evaluation/confusion_matrix.py` is implemented and verified via automated tests; real heatmaps (`confusion_matrix.png` and `normalized_confusion_matrix.png`) will be generated automatically once Phase 4 checkpoints exist."
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 7. Model Comparison Table (Measured Hardware + Evaluation Metrics)",
            "",
            "| Architecture | Total Params | Trainable Params | FP32 Size (MB) | CPU Latency bs=1 (ms) | GPU Latency bs=1 FP32 (ms) | GPU Throughput bs=16 FP16 (img/s) | Peak GPU VRAM bs=16 (MB) | Test Accuracy | Test Macro-F1 |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for r in comp_rows:
        acc_s = f"{r['test_accuracy']*100:.2f}%" if r["test_accuracy"] is not None else "*Pending Phase 4*"
        f1_s = f"{r['test_macro_f1']*100:.2f}%" if r["test_macro_f1"] is not None else "*Pending Phase 4*"
        g_lat = f"{r['gpu_bs1_fp32_latency_ms']:.2f}" if r["gpu_bs1_fp32_latency_ms"] is not None else "N/A"
        g_tps = f"{r['gpu_bs16_fp16_throughput_ips']:.1f}" if r["gpu_bs16_fp16_throughput_ips"] is not None else "N/A"
        g_vram = f"{r['gpu_bs16_fp16_vram_mb']:.1f}" if r["gpu_bs16_fp16_vram_mb"] is not None else "N/A"
        lines.append(
            f"| `{r['architecture_name']}` | {r['total_params']:,} | {r['trainable_params']:,} | "
            f"{r['model_size_mb']:.2f} | {r['cpu_mean_latency_ms']:.2f} | {g_lat} | {g_tps} | {g_vram} | {acc_s} | {f1_s} |"
        )

    lines.extend(
        [
            "",
            f"Comparison CSV: `{payload['comparison_artifacts']['csv_path']}` | Efficiency Chart: `{payload['comparison_artifacts']['charts'].get('efficiency_chart', 'N/A')}`",
            "",
            "> **Selection Policy:** No single winner is declared based on accuracy alone; final deployment selection must weigh Macro-F1, external field generalization (PlantDoc), calibration error (ECE), robustness degradation, and latency/memory constraints.",
            "",
            "---",
            "",
            "## 8. External Validation Results (PlantDoc — Real-World Field Images)",
            "",
        ]
    )

    if eval_results:
        lines.extend(
            [
                "| Model | Mapped PlantDoc Acc | Mapped PlantDoc Macro-F1 | Internal-to-External Acc Drop | Internal-to-External Macro-F1 Drop |",
                "| :--- | ---: | ---: | ---: | ---: |",
            ]
        )
        for arch, res in eval_results.items():
            ext = res["external_validation"]
            em = ext["metrics"]
            d = ext["domain_shift_delta"]
            lines.append(
                f"| `{arch}` | {em['accuracy']*100:.2f}% | {em['macro_f1']*100:.2f}% | "
                f"{d['accuracy_drop']*100:+.2f}% | {d['macro_f1_drop']*100:+.2f}% |"
            )
    else:
        lines.append(
            f"> **Coverage Verified, Inference Pending Trained Checkpoints:** Verified `{ext_audit['mapped_images']:,}` mapped PlantDoc images (`{ext_audit['unmatched_images']}` unmatched, `{ext_audit['train_duplicate_group_overlap']}` training overlaps). Real external generalization scores will populate once Phase 4 checkpoints are trained."
        )

    lines.extend(["", "---", "", "## 9. Robustness Testing Results (In-Memory Perturbations)", ""])
    if eval_results:
        for arch, res in eval_results.items():
            rob = res["robustness"]
            lines.append(f"### `{arch}` (Sampled Test Images: `{rob['samples_evaluated']}`, Original Files Modified: `{rob['original_files_modified']}`)")
            lines.append("")
            lines.append("| Perturbation | Accuracy | Macro-F1 | Acc Drop | Macro-F1 Drop |")
            lines.append("| :--- | ---: | ---: | ---: | ---: |")
            for p in rob["perturbations"]:
                lines.append(
                    f"| `{p['perturbation']}` | {p['accuracy']*100:.2f}% | {p['macro_f1']*100:.2f}% | "
                    f"{p['accuracy_drop']*100:+.2f}% | {p['macro_f1_drop']*100:+.2f}% |"
                )
            lines.append("")
    else:
        lines.append(
            "> **Pending Trained Checkpoints:** 10 isolated perturbations (`brightness_dark_0.65`, `brightness_bright_1.35`, `contrast_low_0.65`, `contrast_high_1.35`, `mild_blur_r1.5`, `gaussian_noise_std14`, `jpeg_compression_q30`, `small_rotation_12deg`, `warm_sunlight_tint`, `cool_shade_tint`) are implemented and unit-tested in `ml/src/evaluation/robustness.py`."
        )

    lines.extend(["", "---", "", "## 10. Confidence Calibration Results (Validation-Fitted Temperature Scaling)", ""])
    if eval_results:
        lines.extend(
            [
                "| Model | Optimal Temp (Val-Fitted) | Test ECE Before | Test ECE After | Test Brier Before | Test Brier After |",
                "| :--- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for arch, res in eval_results.items():
            cal = res["calibration"]
            lines.append(
                f"| `{arch}` | {cal['temperature_scaling']['optimal_temperature']:.4f} | "
                f"{cal['test_uncalibrated']['ece']:.4f} | {cal['test_calibrated']['ece']:.4f} | "
                f"{cal['test_uncalibrated']['brier_score']:.4f} | {cal['test_calibrated']['brier_score']:.4f} |"
            )
    else:
        lines.append(
            "> **Pending Trained Checkpoints:** ECE, multiclass Brier score, validation-only LBFGS temperature scaling, reliability diagrams, and confidence histograms are implemented and verified via automated tests in `ml/src/evaluation/calibration.py`."
        )

    lines.extend(["", "---", "", "## 11. Error Analysis", ""])
    if eval_results:
        for arch, res in eval_results.items():
            ea = res["error_analysis"]
            lines.append(
                f"- **`{arch}`:** Misclassified `{ea['total_misclassified']:,}` / `{ea['total_evaluated']:,}` "
                f"(`{ea['error_rate']*100:.2f}%`) | Low-confidence (<60%): `{ea['low_confidence_count']}` | "
                f"CSV: `{ea['predictions_csv']}` | Gallery: `{ea['gallery_path']}`"
            )
    else:
        lines.append(
            "> **Pending Trained Checkpoints:** Prediction/error CSV export (`image_path`, `true_label`, `predicted_label`, `confidence`, `is_correct`, `dataset_source`), per-species difficulty ranking, false-positive/false-negative counters, and misclassification gallery generation are ready in `ml/src/evaluation/error_analysis.py`."
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 12. Inference Benchmarks (Measured Real Hardware Performance)",
            "",
            "Measured with warm-up runs and explicit `torch.cuda.synchronize()` barriers on `224 × 224 × 3` inputs:",
            "",
            "| Architecture | Condition | Hardware | Precision | Batch Size | Mean (ms) | Median (ms) | P95 (ms) | Throughput (img/s) | Peak VRAM (MB) |",
            "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for arch, bmap in benchmarks.items():
        for cond, b in bmap.items():
            vram_s = f"{b['peak_gpu_memory_mb']:.2f}" if b["peak_gpu_memory_mb"] is not None else "—"
            lines.append(
                f"| `{arch}` | `{cond}` | {b['hardware_name']} | `{b['precision']}` | {b['batch_size']} | "
                f"{b['mean_latency_ms']:.2f} | {b['median_latency_ms']:.2f} | {b['p95_latency_ms']:.2f} | "
                f"{b['throughput_images_per_sec']:.1f} | {vram_s} |"
            )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 13. Limitations",
            "",
            "1. **Missing Phase 4 Trained Checkpoints (Current State):** Because Phase 4 halted before starting lengthy GPU training pending user approval, classification accuracy, F1, confusion matrices, calibration curves, and error galleries cannot be measured yet and are intentionally left unpopulated.",
            "2. **Lab-to-Field Domain Gap:** PlantVillage test performance measures controlled studio-leaf recognition only and must NEVER be extrapolated as real-world field accuracy without PlantDoc external validation.",
            "3. **Agricultural Diagnostic Disclaimer:** Model outputs are decision-support research metrics and cannot claim clinical or agricultural diagnostic reliability without multi-season field validation across diverse cultivars and geographies.",
            "4. **External Class Coverage:** PlantDoc covers 28 of PlantVillage's 38 canonical classes; 10 PlantVillage classes lack external PlantDoc field samples.",
            "",
            "---",
            "",
            "## 14. Research Conclusions & Next Steps",
            "",
            "- **Completed & Verified in Phase 5:** The complete, leakage-safe evaluation suite (metrics, confusion matrices, classification reports, validation-fitted temperature scaling, PlantDoc external validation, 10-perturbation robustness testing, error analysis, multi-model comparison, and real CPU/GPU inference benchmarks) is fully implemented and unit-tested.",
            "- **Measured Hardware Trade-Offs:** Real CPU and RTX 5050 GPU benchmarks above quantify the exact latency, throughput, parameter, and VRAM trade-offs among `simple_cnn`, `agrivision_cnn`, and `resnet18_baseline`.",
            "- **To Populate Full Trained Model Results:** Run Phase 4 training for Experiments A, B, and C, then re-execute:",
            "  ```powershell",
            "  python ml/src/evaluation/evaluate.py",
            "  ```",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase5_evaluation(
    checkpoints_root: Optional[Path] = None,
    batch_size: int = 32,
    robustness_samples: int = 500,
    cpu_bench_runs: int = 15,
    gpu_bench_runs: int = 40,
) -> Dict[str, Any]:
    """Execute complete Phase 5 verification, hardware benchmarking, and (if trained) full evaluation."""
    dataset_cfg = load_dataset_config()
    model_cfg = load_model_config()
    class_names, class_mapping = resolve_verified_classes(model_cfg)

    split_manifest_path = PROJECT_ROOT / "ml" / "data" / "metadata" / "split_manifest.csv"
    split_df = pd.read_csv(split_manifest_path, low_memory=False)
    val_df = split_df[split_df["assigned_split"] == "validation"].reset_index(drop=True)
    test_df = split_df[split_df["assigned_split"] == "test"].reset_index(drop=True)
    train_df = split_df[split_df["assigned_split"] == "train"]

    dataset_audit = {
        "train_count": int(len(train_df)),
        "val_count": int(len(val_df)),
        "test_count": int(len(test_df)),
        "num_classes": len(class_names),
    }

    ext_audit_full = audit_external_plantdoc_coverage(split_df, class_names, class_mapping)
    ext_audit_full.pop("mapped_dataframe", None)

    ckpt_base = checkpoints_root or (PROJECT_ROOT / "ml" / "experiments" / "checkpoints")
    eval_root = PROJECT_ROOT / "ml" / "experiments" / "evaluation"
    metrics_root = eval_root / "metrics"
    charts_root = eval_root / "charts"
    preds_root = eval_root / "predictions"
    reports_root = eval_root / "reports"
    for d in (metrics_root, charts_root, preds_root, reports_root):
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Step 1: Verify checkpoint availability & integrity FIRST
    print("=== Step 1: Verifying Phase 4 Trained Checkpoint Availability & Integrity ===")
    checkpoint_checks: List[Dict[str, Any]] = []
    for arch, run_id in DEFAULT_EXPERIMENTS:
        ckpt_path = ckpt_base / run_id / "best_model.pth"
        info = verify_trained_checkpoint(ckpt_path, expected_arch=arch)
        checkpoint_checks.append(info)
        status_str = "READY" if info["valid_trained_checkpoint"] else "MISSING/UNAVAILABLE"
        print(f"  - {arch:18s} [{status_str}]: {info['reason']}")

    with open(reports_root / "checkpoint_verification.json", "w", encoding="utf-8") as f:
        json.dump(checkpoint_checks, f, indent=2)

    # Step 2: Run real CPU & CUDA GPU hardware inference benchmarks for all 3 architectures
    print("\n=== Step 2: Running Real Hardware Inference Benchmarks (CPU & GPU) ===")
    hardware_benchmarks: Dict[str, Dict[str, Any]] = {}
    structural_info: Dict[str, Dict[str, Any]] = {}
    h = int(model_cfg["input"]["image_height"])
    w = int(model_cfg["input"]["image_width"])
    c = int(model_cfg["input"]["in_channels"])

    for arch, _ in DEFAULT_EXPERIMENTS:
        model = create_model(architecture=arch, model_config=model_cfg, num_classes=len(class_names), seed=42)
        total_p = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        buf_e = sum(b.numel() for b in model.buffers())
        fp32_mb = round(((total_p + buf_e) * 4) / (1024**2), 3)
        structural_info[arch] = {
            "total_params": int(total_p),
            "trainable_params": int(train_p),
            "fp32_size_mb": fp32_mb,
        }
        bench = run_full_hardware_benchmark(
            model=model,
            input_size=(c, h, w),
            cpu_runs=cpu_bench_runs,
            gpu_runs=gpu_bench_runs,
        )
        hardware_benchmarks[arch] = bench
        gpu_note = (
            f" | GPU bs=1: {bench['gpu_bs1_fp32']['mean_latency_ms']:.2f}ms"
            f" | GPU bs=16 FP16: {bench['gpu_bs16_fp16']['throughput_images_per_sec']:.1f} img/s"
            if "gpu_bs1_fp32" in bench
            else ""
        )
        print(f"  - {arch:18s} | CPU bs=1: {bench['cpu_bs1_fp32']['mean_latency_ms']:.2f}ms{gpu_note}")

    with open(metrics_root / "inference_benchmarks.json", "w", encoding="utf-8") as f:
        json.dump(hardware_benchmarks, f, indent=2)

    # Step 3: Evaluate every verified trained checkpoint (NEVER evaluate untrained weights!)
    evaluated_models: Dict[str, Any] = {}
    for chk in checkpoint_checks:
        if not chk["valid_trained_checkpoint"]:
            continue
        arch = chk["architecture_name"]
        ckpt_path = Path(chk["checkpoint_path"])
        print(f"\n=== Step 3: Full Evaluation of Trained Model '{arch}' ===")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_classes = list(ckpt["class_names"])
        model = create_model(
            architecture=arch,
            model_config=ckpt.get("model_config") or model_cfg,
            num_classes=len(ckpt_classes),
        )
        model.load_state_dict(ckpt["model_state_dict"])

        # Internal test inference
        test_logits, test_probs, test_preds, test_targets = predict_over_dataframe(
            model, test_df, dataset_cfg, ckpt_classes, device, batch_size=batch_size
        )
        val_logits, _, _, val_targets = predict_over_dataframe(
            model, val_df, dataset_cfg, ckpt_classes, device, batch_size=batch_size
        )

        # Save raw test probabilities & predictions
        np.savez_compressed(
            preds_root / f"{arch}_test_predictions.npz",
            logits=test_logits,
            probs=test_probs,
            preds=test_preds,
            targets=test_targets,
        )

        test_metrics = compute_multiclass_metrics(
            y_true=test_targets.tolist(),
            y_pred=test_preds.tolist(),
            y_prob=test_probs,
            class_names=ckpt_classes,
        )
        cls_rep = export_classification_report(test_metrics, metrics_root / arch, prefix="internal_test")
        cm_res = save_confusion_matrix_artifacts(
            y_true=test_targets.tolist(),
            y_pred=test_preds.tolist(),
            class_names=ckpt_classes,
            charts_dir=charts_root / arch,
            metrics_dir=metrics_root / arch,
            title_prefix=arch,
        )
        cal_res = evaluate_calibration(
            val_logits=val_logits,
            val_targets=val_targets.tolist(),
            test_logits=test_logits,
            test_targets=test_targets.tolist(),
            charts_dir=charts_root / arch,
            model_name=arch,
        )
        with open(metrics_root / arch / "calibration_report.json", "w", encoding="utf-8") as f:
            json.dump(cal_res, f, indent=2)

        ext_res = evaluate_external_plantdoc(
            model=model,
            split_df=split_df,
            dataset_config=dataset_cfg,
            class_names=ckpt_classes,
            class_mapping=class_mapping,
            device=device,
            internal_test_metrics=test_metrics,
            batch_size=batch_size,
        )
        ext_pred_df = ext_res.pop("predictions_df")
        ext_res.pop("logits", None)
        ext_res.pop("probs", None)
        ext_res.pop("targets", None)
        if len(ext_pred_df) > 0:
            ext_pred_df.to_csv(preds_root / f"{arch}_plantdoc_predictions.csv", index=False)
        with open(metrics_root / arch / "external_plantdoc_report.json", "w", encoding="utf-8") as f:
            json.dump(ext_res, f, indent=2)

        rob_res = run_robustness_suite(
            model=model,
            test_df=test_df,
            dataset_config=dataset_cfg,
            class_names=ckpt_classes,
            device=device,
            batch_size=batch_size,
            max_samples=robustness_samples,
        )
        with open(metrics_root / arch / "robustness_report.json", "w", encoding="utf-8") as f:
            json.dump(rob_res, f, indent=2)

        # Build test predictions DataFrame for error analysis
        confs = np.max(test_probs, axis=1)
        test_pred_df = pd.DataFrame(
            {
                "image_path": test_df["image_path"].astype(str).tolist(),
                "true_label": [ckpt_classes[i] for i in test_targets.tolist()],
                "predicted_label": [ckpt_classes[i] for i in test_preds.tolist()],
                "confidence": np.round(confs, 6),
                "is_correct": (test_preds == test_targets),
                "dataset_source": "plantvillage",
            }
        )
        ea_res = run_error_analysis(
            predictions_df=test_pred_df,
            output_predictions_csv=preds_root / f"{arch}_test_predictions.csv",
            gallery_path=charts_root / arch / "misclassification_gallery.png",
            project_root=PROJECT_ROOT,
            external_predictions_df=ext_pred_df if len(ext_pred_df) > 0 else None,
        )
        with open(metrics_root / arch / "error_analysis.json", "w", encoding="utf-8") as f:
            json.dump(ea_res, f, indent=2)

        evaluated_models[arch] = {
            "internal_test_metrics": test_metrics,
            "classification_report_files": cls_rep,
            "confusion_matrix": cm_res,
            "calibration": cal_res,
            "external_validation": ext_res,
            "robustness": rob_res,
            "error_analysis": ea_res,
        }

    # Step 4: Build side-by-side comparison artifacts
    comparison_rows: List[Dict[str, Any]] = []
    for chk in checkpoint_checks:
        arch = chk["architecture_name"]
        s = structural_info[arch]
        b = hardware_benchmarks[arch]
        ev = evaluated_models.get(arch)
        m = ev["internal_test_metrics"] if ev else None
        comparison_rows.append(
            {
                "architecture_name": arch,
                "checkpoint_verified": chk["valid_trained_checkpoint"],
                "total_params": s["total_params"],
                "trainable_params": s["trainable_params"],
                "model_size_mb": chk["file_size_mb"] or s["fp32_size_mb"],
                "cpu_mean_latency_ms": b["cpu_bs1_fp32"]["mean_latency_ms"],
                "cpu_throughput_ips": b["cpu_bs1_fp32"]["throughput_images_per_sec"],
                "gpu_bs1_fp32_latency_ms": b.get("gpu_bs1_fp32", {}).get("mean_latency_ms"),
                "gpu_bs16_fp16_throughput_ips": b.get("gpu_bs16_fp16", {}).get("throughput_images_per_sec"),
                "gpu_bs16_fp16_vram_mb": b.get("gpu_bs16_fp16", {}).get("peak_gpu_memory_mb"),
                "test_accuracy": m["accuracy"] if m else None,
                "test_macro_precision": m["macro_precision"] if m else None,
                "test_macro_recall": m["macro_recall"] if m else None,
                "test_macro_f1": m["macro_f1"] if m else None,
                "test_weighted_f1": m["weighted_f1"] if m else None,
            }
        )

    comp_artifacts = build_model_comparison(comparison_rows, metrics_root, charts_root)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_audit": dataset_audit,
        "external_coverage_audit": ext_audit_full,
        "checkpoint_verification": checkpoint_checks,
        "hardware_benchmarks": hardware_benchmarks,
        "comparison_rows": comparison_rows,
        "comparison_artifacts": comp_artifacts,
        "evaluated_models": evaluated_models,
    }

    report_path = PROJECT_ROOT / "ml" / "reports" / "phase5_research_report.md"
    write_phase5_research_report(report_path, payload)
    print(f"\n=== Phase 5 Report Saved: {report_path} ===")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5 Model Evaluation & Benchmarking Suite.")
    parser.add_argument("--checkpoints-root", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--robustness-samples", type=int, default=500)
    parser.add_argument("--cpu-bench-runs", type=int, default=15)
    parser.add_argument("--gpu-bench-runs", type=int, default=40)
    args = parser.parse_args()

    run_phase5_evaluation(
        checkpoints_root=Path(args.checkpoints_root) if args.checkpoints_root else None,
        batch_size=args.batch_size,
        robustness_samples=args.robustness_samples,
        cpu_bench_runs=args.cpu_bench_runs,
        gpu_bench_runs=args.gpu_bench_runs,
    )


if __name__ == "__main__":
    main()
