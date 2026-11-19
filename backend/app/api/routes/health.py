"""Health check route (GET /api/v1/health)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ...schemas.common import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API, database, and AI model readiness",
)
def get_health(request: Request) -> HealthResponse:
    """Return operational status for API, SQLite database, and AgriVision-CNN model."""
    settings = request.app.state.settings
    db_manager = request.app.state.db_manager
    model_service = request.app.state.model_service

    db_ok = db_manager.check_health()
    model_status = model_service.get_status()
    model_ready = bool(model_status["model_ready"])

    if db_ok and model_ready:
        overall = "healthy"
    elif db_ok and not model_ready:
        overall = "degraded"
    else:
        overall = "unhealthy"

    return HealthResponse(
        status=overall,
        api_status="online",
        database_status="connected" if db_ok else "unavailable",
        model_ready=model_ready,
        model_status_detail=model_status["status_detail"],
        inference_device=model_status["device"],
        app_version=settings.app_version,
    )
