"""Detection history endpoints (GET /api/history, GET/DELETE /api/history/{id})."""

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import desc, or_
from sqlalchemy.orm import Session

from backend.config import UPLOADS_DIR
from backend.database import get_db
from backend.models.db_models import PredictionRecord
from backend.models.schemas import (
    DiseaseInfo,
    HistoryItem,
    HistoryListResponse,
    TopPrediction,
)
from backend.services.disease_knowledge import get_disease_info

router = APIRouter(prefix="/api", tags=["History"])


def _serialize_record(record: PredictionRecord) -> HistoryItem:
    top_preds = []
    if record.top_predictions_json:
        try:
            raw_preds = json.loads(record.top_predictions_json)
            top_preds = [TopPrediction(**p) for p in raw_preds]
        except Exception:
            top_preds = []

    disease_info_obj = None
    if record.status != "Uncertain":
        info = get_disease_info(record.raw_class_label)
        disease_info_obj = DiseaseInfo(
            description=info["description"],
            symptoms=info["symptoms"],
            causes=info["causes"],
            prevention=info["prevention"],
            treatment=info["treatment"],
            severity=info["severity"],
        )

    return HistoryItem(
        id=record.id,
        created_at=record.created_at,
        image_filename=record.image_filename,
        image_url=record.image_url,
        raw_class_label=record.raw_class_label,
        plant_name=record.plant_name,
        disease_name=record.disease_name,
        status=record.status,
        confidence=record.confidence,
        input_source=record.input_source,
        top_predictions=top_preds,
        disease_info=disease_info_obj,
    )


@router.get("/history", response_model=HistoryListResponse)
def get_detection_history(
    plant: Optional[str] = Query(default=None, description="Filter by plant name"),
    status_filter: Optional[str] = Query(
        default=None, alias="status", description="Filter by Healthy, Infected, or Uncertain"
    ),
    search: Optional[str] = Query(default=None, description="Search plant or disease name"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> HistoryListResponse:
    """Return paginated plant disease detection history sorted newest first."""
    query = db.query(PredictionRecord)

    if plant:
        query = query.filter(PredictionRecord.plant_name.ilike(f"%{plant.strip()}%"))
    if status_filter:
        query = query.filter(PredictionRecord.status.ilike(status_filter.strip()))
    if search:
        term = f"%{search.strip()}%"
        query = query.filter(
            or_(
                PredictionRecord.plant_name.ilike(term),
                PredictionRecord.disease_name.ilike(term),
                PredictionRecord.status.ilike(term),
            )
        )

    total = query.count()
    rows = (
        query.order_by(desc(PredictionRecord.created_at), desc(PredictionRecord.id))
        .offset(offset)
        .limit(limit)
        .all()
    )

    return HistoryListResponse(
        total=total,
        items=[_serialize_record(row) for row in rows],
    )


@router.get("/history/{record_id}", response_model=HistoryItem)
def get_history_record(record_id: int, db: Session = Depends(get_db)) -> HistoryItem:
    """Retrieve a single detection record by ID."""
    record = db.query(PredictionRecord).filter(PredictionRecord.id == record_id).first()
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Detection record #{record_id} was not found.",
        )
    return _serialize_record(record)


@router.delete("/history/{record_id}")
def delete_history_record(record_id: int, db: Session = Depends(get_db)) -> dict:
    """Delete a detection record and its stored image if no longer referenced."""
    record = db.query(PredictionRecord).filter(PredictionRecord.id == record_id).first()
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Detection record #{record_id} was not found.",
        )

    filename = record.image_filename
    db.delete(record)
    db.commit()

    # Remove uploaded file if no other record references it
    still_used = (
        db.query(PredictionRecord)
        .filter(PredictionRecord.image_filename == filename)
        .first()
        is not None
    )
    if not still_used and filename:
        file_path = UPLOADS_DIR / filename
        if file_path.exists() and file_path.parent.resolve() == UPLOADS_DIR.resolve():
            try:
                file_path.unlink()
            except OSError:
                pass

    return {"deleted": True, "id": record_id}
