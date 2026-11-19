"""System technical metadata route (GET /api/v1/system/info)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ...schemas.common import SystemInfoResponse
from ...services.explanation_service import GRADCAM_DISCLAIMER

router = APIRouter(tags=["System"])


@router.get(
    "/system/info",
    response_model=SystemInfoResponse,
    summary="Retrieve safe technical metadata about model, formats, and capabilities",
)
def get_system_info(request: Request) -> SystemInfoResponse:
    """Expose safe runtime capabilities without leaking paths, env vars, or secrets."""
    settings = request.app.state.settings
    model_service = request.app.state.model_service
    status = model_service.get_status()

    return SystemInfoResponse(
        app_name=settings.app_name,
        app_version=settings.app_version,
        model_name=status["model_name"],
        num_classes=status["num_classes"],
        device_type=status["device"],
        checkpoint_available=status["model_ready"],
        checkpoint_id=status["checkpoint_id"],
        supported_image_formats=sorted(settings.supported_formats),
        max_upload_size_bytes=settings.max_upload_size_bytes,
        explanation_available=status["model_ready"],
        explanation_method="Grad-CAM",
        explanation_disclaimer=GRADCAM_DISCLAIMER,
        image_retention_enabled=settings.retain_uploaded_images,
    )
