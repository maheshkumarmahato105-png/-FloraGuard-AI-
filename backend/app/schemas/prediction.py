"""Pydantic schemas for POST /api/v1/predict responses (success & uncertain rejection)."""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class TopPredictionItem(BaseModel):
    class_name: str
    display_name: str
    plant_name: str
    disease_name: str
    is_healthy: bool
    confidence: float = Field(..., ge=0.0, le=1.0)


class PredictionDetail(BaseModel):
    class_name: str
    display_name: str
    plant_name: str
    disease_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    top_predictions: List[TopPredictionItem] = Field(default_factory=list)
    is_healthy: bool


class UncertaintyMetrics(BaseModel):
    entropy: float = Field(..., ge=0.0)
    margin: float = Field(..., ge=0.0, le=1.0)
    max_probability: float = Field(..., ge=0.0, le=1.0)
    is_uncertain: bool
    rejection_reasons: List[str] = Field(default_factory=list)


class ExplanationOutput(BaseModel):
    available: bool
    requested: bool
    method: str = "Grad-CAM"
    heatmap_url: Optional[str] = None
    overlay_url: Optional[str] = None
    note: str = (
        "Grad-CAM highlights approximate AI attention regions and is NOT a verified "
        "pixel-level disease segmentation mask."
    )


class DiseaseInformation(BaseModel):
    disease_name: str
    affected_plant: str
    is_healthy: bool
    summary: str
    symptoms: List[str] = Field(default_factory=list)
    prevention: List[str] = Field(default_factory=list)
    treatment: List[str] = Field(default_factory=list)
    disclaimer: str = (
        "General educational information only — not a substitute for professional "
        "agricultural diagnosis or local extension guidance."
    )


class PredictionResponse(BaseModel):
    request_id: str
    status: Literal["success", "uncertain"]
    message: Optional[str] = None
    prediction: Optional[PredictionDetail] = None
    uncertainty: UncertaintyMetrics
    explanation: Optional[ExplanationOutput] = None
    disease_information: Optional[DiseaseInformation] = None
    model_name: str
    checkpoint_id: str
    inference_device: str
    created_at: datetime
