"""Plant disease AI inference endpoint (POST /api/predict)."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from PIL import Image
from sqlalchemy.orm import Session

from backend.config import UPLOADS_DIR
from backend.database import get_db
from backend.models.db_models import PredictionRecord
from backend.models.schemas import (
    DiseaseInfo,
    ImageDiagnostics,
    PredictionResponse,
    TopPrediction,
)
from backend.services.image_processor import (
    ImageValidationError,
    save_uploaded_image,
    validate_and_decode_image,
)
from backend.services.model_service import ModelNotReadyError, model_service

router = APIRouter(prefix="/api", tags=["Prediction"])


def _find_uploaded_file(file_id: str) -> Optional[Path]:
    """Safely look up a previously uploaded image by UUID hex file_id."""
    clean_id = "".join(ch for ch in file_id if ch.isalnum())
    if len(clean_id) < 16:
        return None
    for candidate in UPLOADS_DIR.glob(f"{clean_id}.*"):
        if candidate.is_file() and candidate.parent.resolve() == UPLOADS_DIR.resolve():
            return candidate
    return None


@router.post("/predict", response_model=PredictionResponse)
async def predict_plant_disease(
    file: Optional[UploadFile] = File(default=None),
    file_id: Optional[str] = Form(default=None),
    input_source: str = Form(default="upload"),
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """Accept an uploaded image (or previously uploaded file_id), run real AI inference,
    save the record to the database, and return structured diagnostic results."""
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                model_service.load_error
                or "The trained AI model is not loaded. Run the training pipeline first."
            ),
        )

    if file is None and not file_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide an image file ('file') or an uploaded 'file_id'.",
        )

    try:
        if file is not None:
            file_bytes = await file.read()
            rgb_img, ext = validate_and_decode_image(
                file_bytes=file_bytes,
                filename=file.filename,
                content_type=file.content_type,
            )
            _, stored_filename, _ = save_uploaded_image(rgb_img, ext=ext)
        else:
            existing_path = _find_uploaded_file(file_id or "")
            if existing_path is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Referenced uploaded image was not found. Please upload the image again.",
                )
            with Image.open(existing_path) as img:
                rgb_img = img.convert("RGB")
            stored_filename = existing_path.name

        image_url = f"/uploads/{stored_filename}"
        prediction = model_service.predict(rgb_img, top_k=3)

        # Persist detection record to SQLite/PostgreSQL
        safe_source = (input_source or "upload").strip()[:64]
        record = PredictionRecord(
            image_filename=stored_filename,
            image_url=image_url,
            raw_class_label=prediction["raw_class_label"],
            plant_name=prediction["plant_name"],
            disease_name=prediction["disease_name"],
            status=prediction["status"],
            confidence=prediction["confidence"],
            top_predictions_json=json.dumps(prediction["top_predictions"]),
            input_source=safe_source,
            notes=prediction.get("uncertainty_message"),
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        disease_info_obj = (
            DiseaseInfo(**prediction["disease_info"])
            if prediction.get("disease_info")
            else None
        )

        return PredictionResponse(
            record_id=record.id,
            timestamp=record.created_at or datetime.now(timezone.utc),
            image_url=image_url,
            image_filename=stored_filename,
            plant_name=prediction["plant_name"],
            disease_name=prediction["disease_name"],
            raw_class_label=prediction["raw_class_label"],
            status=prediction["status"],
            is_healthy=prediction["is_healthy"],
            is_uncertain=prediction["is_uncertain"],
            confidence=prediction["confidence"],
            confidence_threshold=prediction["confidence_threshold"],
            uncertainty_message=prediction.get("uncertainty_message"),
            disease_info=disease_info_obj,
            top_predictions=[TopPrediction(**item) for item in prediction["top_predictions"]],
            diagnostics=ImageDiagnostics(**prediction["diagnostics"]),
            model_architecture=prediction["model_architecture"],
            inference_time_ms=prediction["inference_time_ms"],
        )

    except HTTPException:
        raise
    except ImageValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ModelNotReadyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed due to an internal error: {exc}",
        ) from exc
