"""Prediction and controlled Grad-CAM artifact delivery routes."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ...core.exceptions import RecordNotFoundError
from ...schemas.prediction import PredictionResponse

router = APIRouter(tags=["Prediction"])


def _get_db_session(request: Request):
    yield from request.app.state.db_manager.get_session()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify an uploaded plant leaf image with uncertainty checks and optional Grad-CAM",
)
async def predict_plant_disease(
    request: Request,
    file: UploadFile = File(..., description="Plant leaf image (JPG, JPEG, PNG, or WEBP; max 10 MB)"),
    generate_explanation: bool = Query(
        default=False,
        description="When true, generates approximate Grad-CAM heatmap and overlay URLs.",
    ),
    db: Session = Depends(_get_db_session),
) -> PredictionResponse:
    """Validate upload, run Phase 2 preprocessing + AgriVision-CNN inference, apply uncertainty
    rejection policy, optionally generate Grad-CAM, persist history, and return JSON."""
    request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
    prediction_service = request.app.state.prediction_service
    return await prediction_service.predict(
        upload=file,
        request_id=request_id,
        db=db,
        generate_explanation=generate_explanation,
    )


@router.get(
    "/explanations/{filename}",
    summary="Serve a generated Grad-CAM heatmap or overlay PNG from the controlled results folder",
    response_class=FileResponse,
)
def get_explanation_artifact(request: Request, filename: str) -> FileResponse:
    """Serve explanation PNGs strictly from the controlled results directory (blocks path traversal)."""
    explanation_service = request.app.state.explanation_service
    artifact_path = explanation_service.resolve_safe_artifact_path(filename)
    if artifact_path is None:
        raise RecordNotFoundError("Requested explanation image was not found.")
    return FileResponse(path=str(artifact_path), media_type="image/png")
