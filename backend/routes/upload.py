"""Secure image upload endpoint (POST /api/upload)."""

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.models.schemas import ImageDiagnostics, UploadResponse
from backend.services.image_processor import (
    ImageValidationError,
    analyze_and_focus_leaf,
    save_uploaded_image,
    validate_and_decode_image,
)

router = APIRouter(prefix="/api", tags=["Upload"])


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_image(file: UploadFile = File(...)) -> UploadResponse:
    """Validate, inspect, and securely store an uploaded plant leaf image."""
    try:
        file_bytes = await file.read()
        rgb_img, ext = validate_and_decode_image(
            file_bytes=file_bytes,
            filename=file.filename,
            content_type=file.content_type,
        )
        _, diagnostics = analyze_and_focus_leaf(rgb_img)
        file_id, stored_filename, _ = save_uploaded_image(rgb_img, ext=ext)

        return UploadResponse(
            file_id=file_id,
            filename=stored_filename,
            image_url=f"/uploads/{stored_filename}",
            size_bytes=len(file_bytes),
            diagnostics=ImageDiagnostics(**diagnostics),
        )
    except ImageValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process uploaded image: {exc}",
        ) from exc
