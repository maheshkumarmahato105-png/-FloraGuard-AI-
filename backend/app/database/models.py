"""SQLAlchemy ORM models for PlantDiseaseAI prediction history."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text

from .database import Base


class PredictionHistory(Base):
    """Stores metadata and results for each processed prediction request."""

    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64), unique=True, index=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
    status = Column(String(32), nullable=False, index=True)  # success | uncertain
    predicted_class = Column(String(200), nullable=True, index=True)
    is_healthy = Column(Boolean, nullable=True)
    confidence = Column(Float, nullable=True)  # 0.0 to 1.0
    top_predictions_json = Column(Text, nullable=True)

    # Uncertainty metrics
    entropy = Column(Float, nullable=False, default=0.0)
    margin = Column(Float, nullable=False, default=0.0)
    is_uncertain = Column(Boolean, nullable=False, default=False)
    rejection_reason = Column(Text, nullable=True)

    # Model & provenance metadata (never exposing private filesystem paths)
    model_name = Column(String(120), nullable=False)
    checkpoint_id = Column(String(128), nullable=False)
    image_id = Column(String(64), nullable=False)
    image_retained = Column(Boolean, nullable=False, default=False)

    # Explanation metadata
    explanation_available = Column(Boolean, nullable=False, default=False)
    explanation_method = Column(String(64), nullable=True)
    heatmap_filename = Column(String(255), nullable=True)
    overlay_filename = Column(String(255), nullable=True)
