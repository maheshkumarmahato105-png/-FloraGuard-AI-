"""History service coordinating DB persistence, pagination, and safe deletion."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from ..core.exceptions import RecordNotFoundError
from ..database.models import PredictionHistory
from ..database.repository import PredictionHistoryRepository
from .explanation_service import ExplanationService
from .image_service import ImageService


class HistoryService:
    """Business logic for creating, listing, inspecting, and deleting prediction history."""

    def __init__(
        self,
        image_service: ImageService,
        explanation_service: ExplanationService,
    ) -> None:
        self.image_service = image_service
        self.explanation_service = explanation_service

    def record_prediction(
        self,
        db: Session,
        request_id: str,
        status: str,
        uncertainty: Dict[str, Any],
        model_name: str,
        checkpoint_id: str,
        image_id: str,
        image_retained: bool,
        prediction: Optional[Dict[str, Any]] = None,
        explanation: Optional[Dict[str, Any]] = None,
    ) -> PredictionHistory:
        repo = PredictionHistoryRepository(db)
        reasons = uncertainty.get("rejection_reasons", [])

        row = PredictionHistory(
            request_id=request_id,
            created_at=datetime.now(timezone.utc),
            status=status,
            predicted_class=prediction["class_name"] if prediction else None,
            is_healthy=bool(prediction["is_healthy"]) if prediction else None,
            confidence=float(prediction["confidence"]) if prediction else None,
            top_predictions_json=(
                json.dumps(prediction.get("top_predictions", [])) if prediction else None
            ),
            entropy=float(uncertainty["entropy"]),
            margin=float(uncertainty["margin"]),
            is_uncertain=bool(uncertainty["is_uncertain"]),
            rejection_reason="; ".join(reasons) if reasons else None,
            model_name=model_name,
            checkpoint_id=checkpoint_id,
            image_id=image_id,
            image_retained=bool(image_retained),
            explanation_available=bool(explanation and explanation.get("available")),
            explanation_method=(explanation.get("method") if explanation and explanation.get("available") else None),
            heatmap_filename=(explanation.get("heatmap_filename") if explanation else None),
            overlay_filename=(explanation.get("overlay_filename") if explanation else None),
        )
        return repo.create(row)

    def list_history(
        self,
        db: Session,
        limit: int = 20,
        offset: int = 0,
        status_filter: Optional[str] = None,
        predicted_class: Optional[str] = None,
    ) -> Tuple[int, List[PredictionHistory]]:
        repo = PredictionHistoryRepository(db)
        return repo.list_records(
            limit=limit,
            offset=offset,
            status_filter=status_filter,
            predicted_class=predicted_class,
        )

    def get_record(self, db: Session, request_id: str) -> PredictionHistory:
        repo = PredictionHistoryRepository(db)
        record = repo.get_by_request_id(request_id)
        if record is None:
            raise RecordNotFoundError(f"Prediction history record '{request_id}' was not found.")
        return record

    def delete_record(self, db: Session, request_id: str) -> bool:
        """Delete a prediction record and clean up its associated explanation/retained files."""
        repo = PredictionHistoryRepository(db)
        deleted = repo.delete_by_request_id(request_id)
        if deleted is None:
            raise RecordNotFoundError(f"Prediction history record '{request_id}' was not found.")

        self.explanation_service.delete_artifacts(
            deleted.heatmap_filename,
            deleted.overlay_filename,
        )
        if deleted.image_retained:
            self.image_service.delete_retained_image(deleted.image_id)
        return True
