"""Health check, supported plants, and model evaluation metadata endpoints."""

import json
from fastapi import APIRouter

from backend.config import CONFIDENCE_THRESHOLD, TRAINING_REPORTS_DIR
from backend.database import check_db_health
from backend.models.schemas import HealthResponse
from backend.services.model_service import model_service

router = APIRouter(prefix="/api", tags=["Health & Metadata"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Check backend, database, and AI model availability."""
    db_ok = check_db_health()
    model_ok = model_service.is_loaded

    if db_ok and model_ok:
        overall_status = "healthy"
        message = "Backend, database, and trained AI model are ready."
    elif db_ok and not model_ok:
        overall_status = "degraded"
        message = model_service.load_error or "Trained model not loaded yet."
    else:
        overall_status = "unhealthy"
        message = "Database connectivity check failed."

    return HealthResponse(
        status=overall_status,
        database_connected=db_ok,
        model_loaded=model_ok,
        model_architecture=model_service.architecture if model_ok else None,
        device=str(model_service.device),
        num_classes=len(model_service.class_names),
        supported_plants=model_service.get_supported_plants(),
        confidence_threshold=round(CONFIDENCE_THRESHOLD * 100.0, 1),
        evaluation_summary=model_service.metadata or None,
        message=message,
    )


@router.get("/plants")
def list_supported_plants() -> dict:
    """Return all plants and diseases supported by the currently loaded AI model."""
    return {
        "model_loaded": model_service.is_loaded,
        "num_classes": len(model_service.class_names),
        "plants": model_service.get_plants_detail(),
    }


@router.get("/evaluation")
def get_evaluation_report() -> dict:
    """Return real evaluation metrics and report assets produced by training/evaluate.py."""
    metrics_path = TRAINING_REPORTS_DIR / "evaluation_metrics.json"
    cm_path = TRAINING_REPORTS_DIR / "confusion_matrix.png"
    curves_path = TRAINING_REPORTS_DIR / "training_curves.png"

    metrics = None
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    return {
        "available": metrics is not None,
        "metrics": metrics,
        "confusion_matrix_url": "/reports/confusion_matrix.png" if cm_path.exists() else None,
        "training_curves_url": "/reports/training_curves.png" if curves_path.exists() else None,
    }
