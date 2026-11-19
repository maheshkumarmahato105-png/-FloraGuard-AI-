"""Clean response builders mapping internal service results to sanitized Pydantic schemas."""

from __future__ import annotations

from datetime import datetime
import json
from typing import Any, Dict, List, Optional

from ..database.models import PredictionHistory
from ..schemas.history import HistoryRecordResponse
from ..schemas.prediction import (
    DiseaseInformation,
    ExplanationOutput,
    PredictionDetail,
    PredictionResponse,
    TopPredictionItem,
    UncertaintyMetrics,
)


def build_explanation_url(api_prefix: str, filename: Optional[str]) -> Optional[str]:
    """Return a safe relative API URL for a generated Grad-CAM artifact (never a local path)."""
    if not filename:
        return None
    safe_name = filename.split("/")[-1].split("\\")[-1]
    return f"{api_prefix}/explanations/{safe_name}"


def build_prediction_response(
    request_id: str,
    created_at: datetime,
    model_name: str,
    checkpoint_id: str,
    inference_device: str,
    uncertainty: Dict[str, Any],
    prediction: Optional[Dict[str, Any]] = None,
    explanation: Optional[Dict[str, Any]] = None,
    disease_info: Optional[Dict[str, Any]] = None,
    api_prefix: str = "/api/v1",
) -> PredictionResponse:
    """Build either a verified `success` or rejected `uncertain` PredictionResponse."""
    uncertainty_schema = UncertaintyMetrics(
        entropy=float(uncertainty["entropy"]),
        margin=float(uncertainty["margin"]),
        max_probability=float(uncertainty["max_probability"]),
        is_uncertain=bool(uncertainty["is_uncertain"]),
        rejection_reasons=list(uncertainty.get("rejection_reasons", [])),
    )

    if uncertainty_schema.is_uncertain or prediction is None:
        return PredictionResponse(
            request_id=request_id,
            status="uncertain",
            message=(
                "The image could not be classified reliably. "
                "Please capture a clear, well-lit close-up of a single plant leaf."
            ),
            prediction=None,
            uncertainty=uncertainty_schema,
            explanation=None,
            disease_information=None,
            model_name=model_name,
            checkpoint_id=checkpoint_id,
            inference_device=inference_device,
            created_at=created_at,
        )

    top_items = [TopPredictionItem(**item) for item in prediction.get("top_predictions", [])]
    pred_schema = PredictionDetail(
        class_name=prediction["class_name"],
        display_name=prediction["display_name"],
        plant_name=prediction["plant_name"],
        disease_name=prediction["disease_name"],
        confidence=float(prediction["confidence"]),
        top_predictions=top_items,
        is_healthy=bool(prediction["is_healthy"]),
    )

    exp_schema = None
    if explanation is not None:
        exp_schema = ExplanationOutput(
            available=bool(explanation.get("available", False)),
            requested=bool(explanation.get("requested", False)),
            method=str(explanation.get("method", "Grad-CAM")),
            heatmap_url=build_explanation_url(api_prefix, explanation.get("heatmap_filename")),
            overlay_url=build_explanation_url(api_prefix, explanation.get("overlay_filename")),
            note=str(
                explanation.get(
                    "note",
                    "Grad-CAM highlights approximate AI attention regions and is NOT a verified "
                    "pixel-level disease segmentation mask.",
                )
            ),
        )

    disease_schema = DiseaseInformation(**disease_info) if disease_info else None

    return PredictionResponse(
        request_id=request_id,
        status="success",
        message=None,
        prediction=pred_schema,
        uncertainty=uncertainty_schema,
        explanation=exp_schema,
        disease_information=disease_schema,
        model_name=model_name,
        checkpoint_id=checkpoint_id,
        inference_device=inference_device,
        created_at=created_at,
    )


def serialize_history_record(
    record: PredictionHistory,
    api_prefix: str = "/api/v1",
    disease_info: Optional[Dict[str, Any]] = None,
) -> HistoryRecordResponse:
    """Convert a SQLAlchemy PredictionHistory row into a safe API response."""
    top_preds: List[TopPredictionItem] = []
    if record.top_predictions_json:
        try:
            raw = json.loads(record.top_predictions_json)
            top_preds = [TopPredictionItem(**item) for item in raw]
        except Exception:
            top_preds = []

    disease_schema = (
        DiseaseInformation(**disease_info)
        if (disease_info and record.status == "success" and not record.is_uncertain)
        else None
    )

    return HistoryRecordResponse(
        request_id=record.request_id,
        created_at=record.created_at,
        status=record.status,
        predicted_class=record.predicted_class,
        is_healthy=record.is_healthy,
        confidence=record.confidence,
        top_predictions=top_preds,
        entropy=float(record.entropy),
        margin=float(record.margin),
        is_uncertain=bool(record.is_uncertain),
        rejection_reason=record.rejection_reason,
        model_name=record.model_name,
        checkpoint_id=record.checkpoint_id,
        image_id=record.image_id,
        explanation_available=bool(record.explanation_available),
        explanation_method=record.explanation_method,
        heatmap_url=build_explanation_url(api_prefix, record.heatmap_filename),
        overlay_url=build_explanation_url(api_prefix, record.overlay_filename),
        disease_information=disease_schema,
    )
