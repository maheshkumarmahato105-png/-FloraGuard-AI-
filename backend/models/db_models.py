"""SQLAlchemy ORM models for prediction history storage."""

from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from backend.database import Base


class PredictionRecord(Base):
    """Stores each plant disease detection result."""

    __tablename__ = "prediction_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
    image_filename = Column(String(255), nullable=False)
    image_url = Column(String(512), nullable=False)
    raw_class_label = Column(String(255), nullable=False)
    plant_name = Column(String(120), nullable=False, index=True)
    disease_name = Column(String(200), nullable=False)
    status = Column(String(32), nullable=False, index=True)  # Healthy | Infected | Uncertain
    confidence = Column(Float, nullable=False)  # 0.0 to 100.0
    top_predictions_json = Column(Text, nullable=True)
    input_source = Column(String(64), default="upload", nullable=False)
    notes = Column(Text, nullable=True)
