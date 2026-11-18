"""Pydantic request and response schemas for the REST API."""

from datetime import datetime
from typing import Any, List, Optional
from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    class_label: str
    plant_name: str
    disease_name: str
    status: str
    confidence: float = Field(..., ge=0.0, le=100.0)


class ImageDiagnostics(BaseModel):
    original_width: int
    original_height: int
    blur_score: float
    mean_brightness: float
    leaf_coverage_ratio: float
    leaf_focused: bool
    quality_warnings: List[str] = Field(default_factory=list)


class DiseaseInfo(BaseModel):
    description: str
    symptoms: List[str]
    causes: List[str]
    prevention: List[str]
    treatment: List[str]
    severity: str


class PredictionResponse(BaseModel):
    record_id: Optional[int] = None
    timestamp: datetime
    image_url: str
    image_filename: str
    plant_name: str
    disease_name: str
    raw_class_label: str
    status: str  # "Healthy", "Infected", or "Uncertain"
    is_healthy: bool
    is_uncertain: bool
    confidence: float = Field(..., ge=0.0, le=100.0)
    confidence_threshold: float
    uncertainty_message: Optional[str] = None
    disease_info: Optional[DiseaseInfo] = None
    top_predictions: List[TopPrediction] = Field(default_factory=list)
    diagnostics: ImageDiagnostics
    model_architecture: str
    inference_time_ms: float


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    image_url: str
    size_bytes: int
    diagnostics: ImageDiagnostics


class HistoryItem(BaseModel):
    id: int
    created_at: datetime
    image_filename: str
    image_url: str
    raw_class_label: str
    plant_name: str
    disease_name: str
    status: str
    confidence: float
    input_source: str
    top_predictions: List[TopPrediction] = Field(default_factory=list)
    disease_info: Optional[DiseaseInfo] = None


class HistoryListResponse(BaseModel):
    total: int
    items: List[HistoryItem]


class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    model_loaded: bool
    model_architecture: Optional[str] = None
    device: str
    num_classes: int
    supported_plants: List[str]
    confidence_threshold: float
    evaluation_summary: Optional[dict[str, Any]] = None
    message: Optional[str] = None
