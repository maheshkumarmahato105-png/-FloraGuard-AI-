"""Automated unit & integration tests for Phase 5 Evaluation Suite (`ml/src/evaluation/`).

Tests all evaluation modules using synthetic checkpoints/images inside pytest `tmp_path`
without creating fake trained checkpoints or modifying Phase 2 datasets:
  - Checkpoint loading & untrained-checkpoint rejection
  - Prediction & probability shapes + determinism/reproducibility
  - Multiclass metrics (Accuracy, Macro/Weighted P/R/F1, Kappa, MCC, OvR ROC-AUC)
  - Missing-class safe handling & documented undefined metrics
  - Raw & normalized confusion matrix CSV + PNG generation
  - Confidence calibration (ECE, Brier score, validation-only temperature scaling, plots)
  - External PlantDoc label mapping (mapped vs. unmatched counts & independence check)
  - In-memory robustness perturbations (verifies original image file untouched)
  - Error analysis CSV export & misclassification gallery PNG
  - Inference benchmark (CPU & GPU when available) + comparison CSV/charts
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_SRC = PROJECT_ROOT / "ml" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

from config_loader import load_class_mapping, load_dataset_config
from evaluation import (
    audit_external_plantdoc_coverage,
    benchmark_model_inference,
    build_model_comparison,
    compute_confusion_matrices,
    compute_ece,
    compute_multiclass_brier_score,
    compute_multiclass_metrics,
    evaluate_calibration,
    export_classification_report,
    fit_temperature_scaling,
    get_robustness_perturbations,
    run_error_analysis,
    run_robustness_suite,
    save_confusion_matrix_artifacts,
    softmax_numpy,
    verify_trained_checkpoint,
)
from evaluation.evaluate import predict_over_dataframe
from models import create_model, load_model_config


def _make_synthetic_eval_data(tmp_path: Path, class_names: list[str], n_per_class: int = 3):
    rng = np.random.default_rng(42)
    rows = []
    idx = 0
    for c_i, cname in enumerate(class_names):
        for _ in range(n_per_class):
            rel = Path("eval_imgs") / cname / f"sample_{idx:03d}.jpg"
            abs_p = tmp_path / rel
            abs_p.parent.mkdir(parents=True, exist_ok=True)
            arr = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
            Image.fromarray(arr, mode="RGB").save(abs_p, format="JPEG", quality=92)
            rows.append(
                {
                    "image_path": rel.as_posix(),
                    "processed_path": rel.as_posix(),
                    "original_label": cname,
                    "canonical_class": cname,
                    "assigned_split": "test",
                    "duplicate_group_id": f"G_{idx:03d}",
                }
            )
            idx += 1
    return pd.DataFrame(rows)


def test_checkpoint_verification_missing_untrained_and_valid(tmp_path: Path):
    # 1. Missing checkpoint
    miss = verify_trained_checkpoint(tmp_path / "does_not_exist.pth", "agrivision_cnn")
    assert miss["exists"] is False
    assert miss["valid_trained_checkpoint"] is False

    # 2. Untrained checkpoint (epoch=0) must be rejected
    model = create_model("agrivision_cnn", num_classes=3, seed=42)
    untrained_path = tmp_path / "untrained.pth"
    torch.save(
        {
            "architecture_name": "agrivision_cnn",
            "epoch": 0,
            "is_trained": False,
            "class_names": ["a", "b", "c"],
            "model_state_dict": model.state_dict(),
        },
        untrained_path,
    )
    untrained = verify_trained_checkpoint(untrained_path, "agrivision_cnn")
    assert untrained["exists"] is True
    assert untrained["valid_trained_checkpoint"] is False

    # 3. Valid trained checkpoint (epoch >= 1)
    valid_path = tmp_path / "best_model.pth"
    torch.save(
        {
            "architecture_name": "agrivision_cnn",
            "epoch": 5,
            "is_trained": True,
            "best_val_macro_f1": 0.91,
            "class_names": ["a", "b", "c"],
            "model_state_dict": model.state_dict(),
        },
        valid_path,
    )
    valid = verify_trained_checkpoint(valid_path, "agrivision_cnn")
    assert valid["valid_trained_checkpoint"] is True
    assert valid["epoch"] == 5
    assert valid["num_classes"] == 3


def test_prediction_shapes_and_reproducibility(tmp_path: Path):
    dataset_cfg = load_dataset_config()
    class_names = ["tomato__early_blight", "tomato__healthy", "potato__late_blight"]
    df = _make_synthetic_eval_data(tmp_path, class_names, n_per_class=2)

    model = create_model("agrivision_cnn", num_classes=3, seed=42)
    device = torch.device("cpu")

    logits1, probs1, preds1, targets1 = predict_over_dataframe(
        model, df, dataset_cfg, class_names, device, batch_size=3, project_root=tmp_path
    )
    logits2, probs2, preds2, targets2 = predict_over_dataframe(
        model, df, dataset_cfg, class_names, device, batch_size=3, project_root=tmp_path
    )

    assert logits1.shape == (6, 3)
    assert probs1.shape == (6, 3)
    assert np.allclose(probs1.sum(axis=1), np.ones(6), atol=1e-5)
    assert np.array_equal(preds1, preds2)
    assert np.allclose(logits1, logits2, atol=1e-6)
    assert np.array_equal(targets1, targets2)


def test_multiclass_metrics_and_missing_class_handling(tmp_path: Path):
    class_names = ["apple__healthy", "apple__apple_scab", "tomato__early_blight", "grape__black_rot"]
    # Class 3 ("grape__black_rot") intentionally absent from y_true to verify safe missing-class handling
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 2]
    logits = np.array(
        [
            [3.0, 0.2, 0.1, -1.0],
            [0.8, 1.5, 0.2, -1.0],
            [0.1, 3.2, 0.2, -1.0],
            [0.2, 2.8, 0.1, -1.0],
            [0.1, 0.1, 3.5, -1.0],
            [0.1, 0.3, 2.9, -1.0],
        ]
    )
    probs = softmax_numpy(logits, temperature=1.0)
    m = compute_multiclass_metrics(y_true, y_pred, probs, class_names)

    assert m["num_samples"] == 6
    assert m["num_classes_present"] == 3
    assert m["missing_classes"] == ["grape__black_rot"]
    assert abs(m["accuracy"] - (5 / 6)) < 1e-5
    assert m["cohens_kappa"] is not None
    assert m["matthews_corrcoef"] is not None
    assert m["roc_auc_ovr_macro"] is not None
    assert "roc_auc_ovr_note" in m["undefined_metrics"]

    # Export classification report CSV & JSON
    rep = export_classification_report(m, tmp_path / "metrics", prefix="unit")
    assert Path(rep["per_class_csv"]).exists()
    assert Path(rep["metrics_json"]).exists()
    rep_df = pd.read_csv(rep["per_class_csv"])
    assert len(rep_df) == 4


def test_confusion_matrix_artifacts_and_calibration(tmp_path: Path):
    class_names = ["plant__a", "plant__b", "plant__c"]
    rng = np.random.default_rng(7)
    y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    y_pred = [0, 0, 1, 1, 1, 2, 2, 2, 2]

    raw_cm, norm_cm, summary = compute_confusion_matrices(y_true, y_pred, class_names)
    assert raw_cm.shape == (3, 3)
    assert summary["correct_predictions"] == 7
    assert summary["misclassified_predictions"] == 2
    assert np.allclose(norm_cm.sum(axis=1), np.ones(3))

    cm_art = save_confusion_matrix_artifacts(
        y_true, y_pred, class_names, tmp_path / "charts", tmp_path / "metrics", title_prefix="Unit"
    )
    assert Path(cm_art["raw_png"]).exists()
    assert Path(cm_art["normalized_png"]).exists()
    assert Path(cm_art["raw_csv"]).exists()

    # Overconfident validation & test logits -> temperature scaling should fit T > 1 and reduce/maintain NLL
    val_logits = np.zeros((30, 3), dtype=np.float32)
    val_targets = np.array([i % 3 for i in range(30)], dtype=np.int64)
    for i, t in enumerate(val_targets):
        val_logits[i] = rng.normal(0.0, 1.0, size=3)
        val_logits[i, t] += 4.5 if i % 5 != 0 else -2.0  # overconfident errors

    test_logits = val_logits.copy()
    test_targets = val_targets.copy()

    t_fit = fit_temperature_scaling(val_logits, val_targets)
    assert t_fit["fitted_on"] == "validation_split_only"
    assert t_fit["val_nll_after"] <= t_fit["val_nll_before"] + 1e-4

    cal = evaluate_calibration(
        val_logits, val_targets, test_logits, test_targets, tmp_path / "cal_charts", model_name="Unit"
    )
    assert 0.0 <= cal["test_uncalibrated"]["ece"] <= 1.0
    assert 0.0 <= cal["test_calibrated"]["ece"] <= 1.0
    assert cal["test_uncalibrated"]["brier_score"] >= 0.0
    assert Path(cal["plots"]["reliability_diagram"]).exists()
    assert Path(cal["plots"]["confidence_histogram"]).exists()


def test_external_mapping_audit_handles_mapped_and_unmatched():
    dataset_cfg = load_dataset_config()
    class_mapping = load_class_mapping(dataset_cfg)
    model_classes = ["tomato__early_blight", "apple__apple_scab"]

    synth_split = pd.DataFrame(
        [
            {"image_path": "a.jpg", "original_label": "Tomato___Early_blight", "canonical_class": "tomato__early_blight", "assigned_split": "train", "duplicate_group_id": "G1"},
            {"image_path": "ext1.jpg", "original_label": "Tomato Early blight leaf", "canonical_class": "tomato__early_blight", "assigned_split": "external_plantdoc", "duplicate_group_id": "G2"},
            {"image_path": "ext2.jpg", "original_label": "Unknown Exotic Weed Leaf", "canonical_class": "unmapped__unknown", "assigned_split": "external_plantdoc", "duplicate_group_id": "G3"},
            {"image_path": "ext3.jpg", "original_label": "Blueberry leaf", "canonical_class": "blueberry__healthy", "assigned_split": "external_plantdoc", "duplicate_group_id": "G4"},
        ]
    )
    audit = audit_external_plantdoc_coverage(synth_split, model_classes, class_mapping)
    assert audit["total_external_images"] == 3
    assert audit["mapped_images"] == 1
    assert audit["unmatched_images"] == 2
    assert audit["is_independent_from_train"] is True


def test_robustness_non_destructive_and_error_analysis_csv_and_benchmark(tmp_path: Path):
    dataset_cfg = load_dataset_config()
    class_names = ["tomato__early_blight", "tomato__healthy"]
    df = _make_synthetic_eval_data(tmp_path, class_names, n_per_class=2)

    # Record original SHA-256 of first test image before robustness testing
    first_img = tmp_path / df.iloc[0]["image_path"]
    sha_before = hashlib.sha256(first_img.read_bytes()).hexdigest()

    model = create_model("simple_cnn", num_classes=2, seed=42)
    rob = run_robustness_suite(
        model=model,
        test_df=df,
        dataset_config=dataset_cfg,
        class_names=class_names,
        device=torch.device("cpu"),
        batch_size=2,
        max_samples=4,
        project_root=tmp_path,
    )
    sha_after = hashlib.sha256(first_img.read_bytes()).hexdigest()
    assert sha_before == sha_after, "Robustness testing must NEVER modify original test files!"
    assert len(rob["perturbations"]) == len(get_robustness_perturbations())

    # Error analysis CSV & gallery
    pred_df = pd.DataFrame(
        [
            {"image_path": df.iloc[0]["image_path"], "true_label": "tomato__early_blight", "predicted_label": "tomato__healthy", "confidence": 0.85, "is_correct": False, "dataset_source": "plantvillage"},
            {"image_path": df.iloc[1]["image_path"], "true_label": "tomato__early_blight", "predicted_label": "tomato__early_blight", "confidence": 0.92, "is_correct": True, "dataset_source": "plantvillage"},
            {"image_path": df.iloc[2]["image_path"], "true_label": "tomato__healthy", "predicted_label": "tomato__early_blight", "confidence": 0.55, "is_correct": False, "dataset_source": "plantvillage"},
            {"image_path": df.iloc[3]["image_path"], "true_label": "tomato__healthy", "predicted_label": "tomato__healthy", "confidence": 0.95, "is_correct": True, "dataset_source": "plantvillage"},
        ]
    )
    ea = run_error_analysis(
        predictions_df=pred_df,
        output_predictions_csv=tmp_path / "preds.csv",
        gallery_path=tmp_path / "gallery.png",
        project_root=tmp_path,
    )
    assert ea["total_misclassified"] == 2
    assert ea["low_confidence_count"] == 1
    assert Path(ea["predictions_csv"]).exists()
    assert Path(ea["gallery_path"]).exists()

    # Inference benchmark & comparison export
    cpu_bench = benchmark_model_inference(model, torch.device("cpu"), batch_size=1, warmup_runs=1, measured_runs=3)
    assert cpu_bench["mean_latency_ms"] > 0 and cpu_bench["throughput_images_per_sec"] > 0

    comp = build_model_comparison(
        [
            {
                "architecture_name": "simple_cnn",
                "total_params": 1000,
                "trainable_params": 1000,
                "model_size_mb": 1.0,
                "cpu_mean_latency_ms": cpu_bench["mean_latency_ms"],
                "test_accuracy": 0.5,
                "test_macro_f1": 0.5,
                "test_weighted_f1": 0.5,
            }
        ],
        tmp_path / "comp_m",
        tmp_path / "comp_c",
    )
    assert Path(comp["csv_path"]).exists()
    assert Path(comp["charts"]["efficiency_chart"]).exists()
    assert Path(comp["charts"]["metrics_chart"]).exists()
