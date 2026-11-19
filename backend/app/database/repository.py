"""Data-access repository for prediction history CRUD, pagination, and filtering."""

from __future__ import annotations

from typing import List, Optional, Tuple

from sqlalchemy import desc
from sqlalchemy.orm import Session

from .models import PredictionHistory


class PredictionHistoryRepository:
    """Encapsulates all SQL queries on `prediction_history`."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, record: PredictionHistory) -> PredictionHistory:
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def get_by_request_id(self, request_id: str) -> Optional[PredictionHistory]:
        return (
            self.db.query(PredictionHistory)
            .filter(PredictionHistory.request_id == request_id)
            .first()
        )

    def list_records(
        self,
        limit: int = 20,
        offset: int = 0,
        status_filter: Optional[str] = None,
        predicted_class: Optional[str] = None,
    ) -> Tuple[int, List[PredictionHistory]]:
        query = self.db.query(PredictionHistory)

        if status_filter:
            query = query.filter(PredictionHistory.status == status_filter.strip().lower())
        if predicted_class:
            query = query.filter(
                PredictionHistory.predicted_class.ilike(f"%{predicted_class.strip()}%")
            )

        total = query.count()
        items = (
            query.order_by(desc(PredictionHistory.created_at), desc(PredictionHistory.id))
            .offset(offset)
            .limit(limit)
            .all()
        )
        return total, items

    def delete_by_request_id(self, request_id: str) -> Optional[PredictionHistory]:
        record = self.get_by_request_id(request_id)
        if record is None:
            return None
        self.db.delete(record)
        self.db.commit()
        return record
