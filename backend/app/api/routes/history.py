"""Prediction history routes (GET /api/v1/history, GET/DELETE /api/v1/history/{request_id})."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from ...schemas.history import (
    DeleteHistoryResponse,
    HistoryListResponse,
    HistoryRecordResponse,
)
from ...utils.response_builder import serialize_history_record

router = APIRouter(tags=["History"])


def _get_db_session(request: Request):
    yield from request.app.state.db_manager.get_session()


@router.get(
    "/history",
    response_model=HistoryListResponse,
    summary="List paginated prediction history sorted newest first",
)
def list_prediction_history(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100, description="Maximum records to return"),
    offset: int = Query(default=0, ge=0, description="Number of records to skip"),
    status: Optional[str] = Query(
        default=None,
        description="Filter by prediction status ('success' or 'uncertain')",
    ),
    predicted_class: Optional[str] = Query(
        default=None,
        description="Filter by predicted canonical class name (substring match)",
    ),
    db: Session = Depends(_get_db_session),
) -> HistoryListResponse:
    settings = request.app.state.settings
    history_service = request.app.state.history_service
    total, rows = history_service.list_history(
        db=db,
        limit=limit,
        offset=offset,
        status_filter=status,
        predicted_class=predicted_class,
    )
    return HistoryListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[serialize_history_record(r, api_prefix=settings.api_v1_prefix) for r in rows],
    )


@router.get(
    "/history/{request_id}",
    response_model=HistoryRecordResponse,
    summary="Retrieve a single prediction history record by request_id",
)
def get_prediction_history_item(
    request: Request,
    request_id: str,
    db: Session = Depends(_get_db_session),
) -> HistoryRecordResponse:
    settings = request.app.state.settings
    history_service = request.app.state.history_service
    record = history_service.get_record(db=db, request_id=request_id)

    disease_info = None
    if record.status == "success" and not record.is_uncertain and record.predicted_class:
        model_service = request.app.state.model_service
        disease_info_service = request.app.state.disease_info_service
        meta = model_service.get_class_metadata(record.predicted_class)
        disease_info = disease_info_service.get_information(
            canonical_class=meta["class_name"],
            plant_name=meta["plant_name"],
            disease_name=meta["disease_name"],
            is_healthy=bool(record.is_healthy) if record.is_healthy is not None else meta["is_healthy"],
        )

    return serialize_history_record(
        record,
        api_prefix=settings.api_v1_prefix,
        disease_info=disease_info,
    )


@router.delete(
    "/history/{request_id}",
    response_model=DeleteHistoryResponse,
    summary="Delete a prediction history record and its associated artifacts",
)
def delete_prediction_history_item(
    request: Request,
    request_id: str,
    db: Session = Depends(_get_db_session),
) -> DeleteHistoryResponse:
    history_service = request.app.state.history_service
    history_service.delete_record(db=db, request_id=request_id)
    return DeleteHistoryResponse(request_id=request_id, deleted=True)
