"""Secure image upload ingestion, size enforcement, decoding, and optional retention."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

from ..core.config import Settings
from ..core.exceptions import PayloadTooLargeError
from ..utils.image_validation import (
    DecodedImage,
    sanitize_filename_for_logging,
    validate_and_decode_bytes,
)

logger = logging.getLogger("plantdiseaseai.backend")

_READ_CHUNK_SIZE = 256 * 1024  # 256 KB chunks to abort oversized uploads early


class ImageService:
    """Reads, validates, decodes, and optionally retains uploaded leaf images safely."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def process_upload(self, upload: UploadFile, request_id: str) -> DecodedImage:
        """Read multipart upload safely (bounded memory) and decode into validated RGB."""
        safe_log_name = sanitize_filename_for_logging(upload.filename)
        max_bytes = self.settings.max_upload_size_bytes

        chunks: list[bytes] = []
        total = 0
        try:
            while True:
                chunk = await upload.read(_READ_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    max_mb = max_bytes / (1024 * 1024)
                    raise PayloadTooLargeError(
                        f"Uploaded image exceeds the maximum allowed size of {max_mb:.1f} MB."
                    )
                chunks.append(chunk)
        finally:
            await upload.close()

        raw_bytes = b"".join(chunks)
        decoded = validate_and_decode_bytes(
            file_bytes=raw_bytes,
            settings=self.settings,
            filename=safe_log_name,
            content_type=upload.content_type,
        )

        # Retain only when explicitly enabled via RETAIN_UPLOADED_IMAGES=true
        if self.settings.retain_uploaded_images:
            self._save_retained_image(decoded)

        logger.info(
            f"Validated upload '{safe_log_name}' ({decoded.width}x{decoded.height}, {decoded.detected_format})",
            extra={"request_id": request_id, "event": "image_validated"},
        )
        return decoded

    def _save_retained_image(self, decoded: DecodedImage) -> Optional[Path]:
        """Store validated RGB copy using server-generated UUID filename (never client filename)."""
        self.settings.retained_uploads_dir.mkdir(parents=True, exist_ok=True)
        dest = self.settings.retained_uploads_dir / f"{decoded.image_id}.jpg"
        decoded.rgb_image.save(dest, format="JPEG", quality=92)
        return dest

    def delete_retained_image(self, image_id: str) -> None:
        """Safely delete a retained image by server-generated hex ID if present."""
        clean_id = "".join(ch for ch in image_id if ch.isalnum())
        if len(clean_id) < 16:
            return
        candidate = (self.settings.retained_uploads_dir / f"{clean_id}.jpg").resolve()
        root = self.settings.retained_uploads_dir.resolve()
        if candidate.parent == root and candidate.exists():
            try:
                candidate.unlink()
            except OSError:
                pass
