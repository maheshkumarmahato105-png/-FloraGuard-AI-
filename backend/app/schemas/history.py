"""Pydantic schemas for prediction history endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from .prediction import DiseaseInformation, TopPredictionItem


class HistoryRecordResponse(BaseModel):
    request_id: str
    created_at: datetime
    status: str
    predicted_class: Optional[str] = None
    is_healthy: Optional[bool] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_predictions: List[TopPredictionItem] = Field(default_factory=list)
    entropy: float
    margin: float
    is_uncertain: bool
    rejection_reason: Optional[str] = None
    model_name: str
    checkpoint_id: str
    image_id: str
    explanation_available: bool
    explanation_method: Optional[str] = None
    heatmap_url: Optional[str] = None
    overlay_url: Optional[str] = None
    disease_information: Optional[DiseaseInformation] = None


class HistoryListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: List[HistoryRecordResponse] = Field(default_factory=list)


class DeleteHistoryResponse(BaseModel):
    request_id: str
    deleted: bool
