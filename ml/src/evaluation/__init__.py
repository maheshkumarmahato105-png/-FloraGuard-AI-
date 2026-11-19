"""Phase 5 Model Evaluation, Calibration, Robustness, Error Analysis & Benchmarking Suite."""

from .calibration import (
    compute_ece,
    compute_multiclass_brier_score,
    evaluate_calibration,
    fit_temperature_scaling,
    softmax_numpy,
)
from .classification_report import export_classification_report
from .confusion_matrix import compute_confusion_matrices, save_confusion_matrix_artifacts
from .error_analysis import run_error_analysis
from .evaluate import run_phase5_evaluation, verify_trained_checkpoint
from .external_validation import audit_external_plantdoc_coverage, evaluate_external_plantdoc
from .inference_benchmark import benchmark_model_inference, run_full_hardware_benchmark
from .metrics import compute_multiclass_metrics
from .model_comparison import build_model_comparison
from .robustness import get_robustness_perturbations, run_robustness_suite

__all__ = [
    "compute_multiclass_metrics",
    "compute_confusion_matrices",
    "save_confusion_matrix_artifacts",
    "export_classification_report",
    "softmax_numpy",
    "compute_ece",
    "compute_multiclass_brier_score",
    "fit_temperature_scaling",
    "evaluate_calibration",
    "audit_external_plantdoc_coverage",
    "evaluate_external_plantdoc",
    "get_robustness_perturbations",
    "run_robustness_suite",
    "run_error_analysis",
    "benchmark_model_inference",
    "run_full_hardware_benchmark",
    "build_model_comparison",
    "verify_trained_checkpoint",
    "run_phase5_evaluation",
]
