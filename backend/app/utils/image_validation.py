"""Strict server-side image validation, magic-byte inspection, and safe decoding.

Validates actual image content using Pillow AND OpenCV (never trusting extensions
or uploaded filenames as filesystem paths), handles EXIF orientation, converts
grayscale/RGBA/CMYK cleanly to RGB, and enforces size and resolution limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath, PureWindowsPath
import re
import uuid

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ..core.config import Settings
from ..core.exceptions import (
    ImageValidationError,
    PayloadTooLargeError,
    UnsupportedMediaFormatError,
)


@dataclass(frozen=True)
class DecodedImage:
    """Validated, safe in-memory RGB image with server-assigned identifier."""

    image_id: str
    rgb_image: Image.Image
    detected_format: str
    original_mode: str
    width: int
    height: int
    size_bytes: int


def sanitize_filename_for_logging(raw_filename: str | None) -> str:
    """Strip any directory components or unsafe characters; never used for filesystem paths."""
    if not raw_filename:
        return "unnamed"
    base = PureWindowsPath(PurePosixPath(raw_filename).name).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return cleaned[:80] or "unnamed"


def detect_magic_format(data: bytes) -> str | None:
    """Inspect leading magic bytes to identify JPEG, PNG, or WEBP independently of filename."""
    if len(data) < 12:
        return None
    if data[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return None


def validate_and_decode_bytes(
    file_bytes: bytes,
    settings: Settings,
    filename: str | None = None,
    content_type: str | None = None,
) -> DecodedImage:
    """Validate upload size, magic bytes, Pillow & OpenCV decode integrity, and dimensions."""
    if not file_bytes:
        raise ImageValidationError("Uploaded file is empty. Please upload a valid plant leaf image.")

    size_bytes = len(file_bytes)
    if size_bytes > settings.max_upload_size_bytes:
        max_mb = settings.max_upload_size_bytes / (1024 * 1024)
        raise PayloadTooLargeError(
            f"Uploaded image exceeds the maximum allowed size of {max_mb:.1f} MB."
        )

    # Reject explicit non-image MIME types early (allow generic octet-stream if magic bytes match)
    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        if ct and ct != "application/octet-stream" and ct not in settings.supported_mime_types:
            raise UnsupportedMediaFormatError(
                f"Unsupported media type '{ct}'. Supported formats: JPG, JPEG, PNG, WEBP."
            )

    # Magic-byte check guarantees we validate actual content, not just extension
    magic_fmt = detect_magic_format(file_bytes)
    if magic_fmt is None or magic_fmt not in settings.supported_formats:
        # Try Pillow format inspection to produce an accurate 415 vs 400 error
        try:
            with Image.open(BytesIO(file_bytes)) as probe:
                fmt = (probe.format or "").upper()
            if fmt and fmt not in settings.supported_formats:
                raise UnsupportedMediaFormatError(
                    f"Unsupported image format '{fmt}'. Supported formats: JPG, JPEG, PNG, WEBP."
                )
        except UnsupportedMediaFormatError:
            raise
        except Exception as exc:
            raise ImageValidationError(
                "Uploaded file is corrupted or not a valid JPG, PNG, or WEBP image."
            ) from exc
        raise UnsupportedMediaFormatError(
            "Unsupported image format. Supported formats: JPG, JPEG, PNG, WEBP."
        )

    # Full Pillow verification + EXIF orientation + RGB conversion (handles grayscale/RGBA/CMYK)
    try:
        with Image.open(BytesIO(file_bytes)) as img:
            detected_format = (img.format or magic_fmt).upper()
            if detected_format not in settings.supported_formats:
                raise UnsupportedMediaFormatError(
                    f"Unsupported image format '{detected_format}'. Supported formats: JPG, JPEG, PNG, WEBP."
                )
            img.verify()

        with Image.open(BytesIO(file_bytes)) as img:
            img = ImageOps.exif_transpose(img)
            img.load()
            original_mode = img.mode
            rgb_image = img.convert("RGB")
    except UnsupportedMediaFormatError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageValidationError(
            "Image file is corrupted and could not be decoded safely."
        ) from exc

    # Cross-verify readability via OpenCV
    try:
        np_buf = np.frombuffer(file_bytes, dtype=np.uint8)
        cv_img = cv2.imdecode(np_buf, cv2.IMREAD_UNCHANGED)
        if cv_img is None or cv_img.size == 0:
            raise ImageValidationError("Image failed OpenCV decoding verification.")
    except ImageValidationError:
        raise
    except Exception as exc:
        raise ImageValidationError("Image failed OpenCV decoding verification.") from exc

    width, height = rgb_image.size
    if width < settings.min_image_dimension or height < settings.min_image_dimension:
        raise ImageValidationError(
            f"Image dimensions ({width}x{height}) are smaller than minimum allowed "
            f"({settings.min_image_dimension}x{settings.min_image_dimension})."
        )
    if width > settings.max_image_dimension or height > settings.max_image_dimension:
        raise ImageValidationError(
            f"Image dimensions ({width}x{height}) exceed maximum allowed "
            f"({settings.max_image_dimension}x{settings.max_image_dimension})."
        )

    return DecodedImage(
        image_id=uuid.uuid4().hex,
        rgb_image=rgb_image,
        detected_format=detected_format,
        original_mode=original_mode,
        width=width,
        height=height,
        size_bytes=size_bytes,
    )
