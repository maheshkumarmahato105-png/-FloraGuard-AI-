"""Common health, system info, and error Pydantic schemas."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall API health: healthy | degraded | unhealthy")
    api_status: str
    database_status: str
    model_ready: bool
    model_status_detail: str
    inference_device: str
    app_version: str


class SystemInfoResponse(BaseModel):
    app_name: str
    app_version: str
    model_name: str
    num_classes: int
    device_type: str
    checkpoint_available: bool
    checkpoint_id: Optional[str] = None
    supported_image_formats: List[str]
    max_upload_size_bytes: int
    explanation_available: bool
    explanation_method: str
    explanation_disclaimer: str
    image_retention_enabled: bool


class ErrorResponse(BaseModel):
    request_id: str
    status: str = "error"
    error_code: str
    detail: str
