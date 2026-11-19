"""Automated tests for strict image upload validation, magic bytes, EXIF, and size limits."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

from PIL import Image
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import Settings
from backend.app.core.exceptions import (
    ImageValidationError,
    PayloadTooLargeError,
    UnsupportedMediaFormatError,
)
from backend.app.utils.image_validation import (
    sanitize_filename_for_logging,
    validate_and_decode_bytes,
)


def _img_bytes(mode="RGB", size=(128, 128), fmt="JPEG", color=(40, 150, 50)) -> bytes:
    img = Image.new(mode, size, color=color if mode == "RGB" else 128)
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_valid_jpeg_png_webp_and_grayscale_decode(tmp_path: Path):
    settings = Settings(results_dir=tmp_path / "r", temp_upload_dir=tmp_path / "t")

    for fmt, mime in (("JPEG", "image/jpeg"), ("PNG", "image/png"), ("WEBP", "image/webp")):
        raw = _img_bytes(mode="RGB", size=(160, 120), fmt=fmt)
        decoded = validate_and_decode_bytes(raw, settings, filename=f"leaf.{fmt.lower()}", content_type=mime)
        assert decoded.rgb_image.mode == "RGB"
        assert (decoded.width, decoded.height) == (160, 120)
        assert len(decoded.image_id) == 32

    # Grayscale PNG must decode safely and convert to 3-channel RGB
    gray_raw = _img_bytes(mode="L", size=(128, 128), fmt="PNG")
    gray_decoded = validate_and_decode_bytes(gray_raw, settings, filename="gray.png", content_type="image/png")
    assert gray_decoded.original_mode == "L"
    assert gray_decoded.rgb_image.mode == "RGB"


def test_rejects_empty_corrupted_unsupported_oversized_and_tiny_files(tmp_path: Path):
    settings = Settings(
        max_upload_size_bytes=20_000,
        results_dir=tmp_path / "r",
        temp_upload_dir=tmp_path / "t",
    )

    # Empty
    with pytest.raises(ImageValidationError):
        validate_and_decode_bytes(b"", settings, filename="empty.jpg", content_type="image/jpeg")

    # Unsupported MIME type
    with pytest.raises(UnsupportedMediaFormatError):
        validate_and_decode_bytes(b"hello world", settings, filename="doc.txt", content_type="text/plain")

    # Unsupported image format disguised with .jpg filename (BMP magic bytes)
    bmp_bytes = _img_bytes(mode="RGB", size=(64, 64), fmt="BMP")
    with pytest.raises(UnsupportedMediaFormatError):
        validate_and_decode_bytes(bmp_bytes, settings, filename="fake.jpg", content_type="image/jpeg")

    # Corrupted JPEG (valid JPEG magic header followed by garbage)
    with pytest.raises(ImageValidationError):
        validate_and_decode_bytes(b"\xff\xd8\xff\xe0corrupted-garbage-bytes", settings, filename="bad.jpg", content_type="image/jpeg")

    # Oversized file (> 20,000 bytes limit)
    big_bytes = b"\xff\xd8\xff\xe0" + (b"\x00" * 25_000)
    assert len(big_bytes) > 20_000
    with pytest.raises(PayloadTooLargeError):
        validate_and_decode_bytes(big_bytes, settings, filename="big.jpg", content_type="image/jpeg")

    # Tiny image (< 32x32)
    tiny_bytes = _img_bytes(mode="RGB", size=(16, 16), fmt="JPEG")
    with pytest.raises(ImageValidationError):
        validate_and_decode_bytes(tiny_bytes, settings, filename="tiny.jpg", content_type="image/jpeg")


def test_path_traversal_filename_sanitization():
    assert sanitize_filename_for_logging("../../etc/passwd") == "passwd"
    assert sanitize_filename_for_logging(r"C:\Windows\System32\evil.jpg") == "evil.jpg"
    assert ".." not in sanitize_filename_for_logging("..\\..\\leaf.jpg")
