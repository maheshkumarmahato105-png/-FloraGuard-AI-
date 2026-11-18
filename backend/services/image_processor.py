"""Image validation, quality inspection, OpenCV leaf focusing, and tensor preprocessing."""

from io import BytesIO
from pathlib import Path
from typing import List, Tuple
import uuid

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torchvision import transforms

from backend.config import (
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_BRIGHTNESS,
    MIN_BLUR_VARIANCE,
    MIN_BRIGHTNESS,
    MIN_IMAGE_DIMENSION,
    MODEL_INPUT_SIZE,
    UPLOADS_DIR,
)


class ImageValidationError(ValueError):
    """Raised when an uploaded image fails security or integrity validation."""


def get_inference_transform() -> transforms.Compose:
    """Standard deterministic inference transform matching validation/test preprocessing."""
    return transforms.Compose(
        [
            transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def validate_and_decode_image(
    file_bytes: bytes,
    filename: str | None = None,
    content_type: str | None = None,
) -> Tuple[Image.Image, str]:
    """Validate file size, extension, MIME type, and actual decoded image integrity.

    Returns:
        (rgb_pil_image, normalized_extension)
    """
    if not file_bytes:
        raise ImageValidationError("Uploaded file is empty. Please choose a valid plant leaf image.")

    if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
        max_mb = MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
        raise ImageValidationError(
            f"Image exceeds the maximum allowed size of {max_mb:.0f} MB."
        )

    ext = Path(filename or "upload.jpg").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        if content_type and content_type.lower() in ALLOWED_MIME_TYPES:
            mime_to_ext = {
                "image/jpeg": ".jpg",
                "image/jpg": ".jpg",
                "image/png": ".png",
                "image/webp": ".webp",
            }
            ext = mime_to_ext[content_type.lower()]
        else:
            raise ImageValidationError(
                "Unsupported file format. Please upload a JPG, JPEG, PNG, or WEBP image."
            )

    if content_type and content_type.lower() not in ALLOWED_MIME_TYPES and content_type != "application/octet-stream":
        raise ImageValidationError(
            f"Invalid file MIME type '{content_type}'. Only JPG, PNG, and WEBP images are supported."
        )

    try:
        with Image.open(BytesIO(file_bytes)) as img:
            img.verify()
        with Image.open(BytesIO(file_bytes)) as img:
            # Handle EXIF orientation from mobile cameras automatically
            img = ImageOps.exif_transpose(img)
            rgb_img = img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError(
            "The uploaded file is corrupted or not a valid image. Please try another photo."
        ) from exc

    width, height = rgb_img.size
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        raise ImageValidationError(
            f"Image resolution ({width}x{height}) is too small. Please provide an image at least {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION} pixels."
        )

    return rgb_img, ext


def save_uploaded_image(rgb_img: Image.Image, ext: str = ".jpg") -> Tuple[str, str, Path]:
    """Save a validated image safely with a UUID filename inside uploads/.

    Returns:
        (file_id, stored_filename, full_path)
    """
    safe_ext = ext if ext in ALLOWED_EXTENSIONS else ".jpg"
    file_id = uuid.uuid4().hex
    stored_filename = f"{file_id}{safe_ext}"
    full_path = UPLOADS_DIR / stored_filename

    save_format = {
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".png": "PNG",
        ".webp": "WEBP",
    }.get(safe_ext, "JPEG")

    save_kwargs = {"quality": 92} if save_format in ("JPEG", "WEBP") else {}
    rgb_img.save(full_path, format=save_format, **save_kwargs)
    return file_id, stored_filename, full_path


def analyze_and_focus_leaf(rgb_img: Image.Image) -> Tuple[Image.Image, dict]:
    """Assess image quality (blur, brightness) and optionally detect/focus on the plant leaf.

    Uses OpenCV HSV color segmentation for plant foliage (green, yellow-green,
    brownish diseased leaf regions) to estimate leaf coverage and crop away
    excessive background clutter ONLY when a clear leaf contour occupies a
    moderate portion of the frame (10%–72%), preserving full lesions and margins.
    """
    np_rgb = np.array(rgb_img)
    height, width = np_rgb.shape[:2]
    np_bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)

    # 1. Image sharpness (Laplacian variance on standardized size for consistent scoring)
    gray = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2GRAY)
    gray_std = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    blur_score = float(cv2.Laplacian(gray_std, cv2.CV_64F).var())

    # 2. Mean luminance / brightness
    mean_brightness = float(np.mean(gray_std))

    # 3. Leaf / foliage segmentation in HSV space (captures healthy green + yellow/brown diseased leaf tissue)
    hsv = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2HSV)
    # Green to yellow-green foliage range
    mask_green = cv2.inRange(hsv, np.array([22, 25, 25]), np.array([98, 255, 255]))
    # Diseased yellow/orange/brown leaf tissue (only moderate-high saturation to avoid neutral gray backgrounds)
    mask_lesion = cv2.inRange(hsv, np.array([8, 45, 30]), np.array([25, 255, 235]))
    leaf_mask = cv2.bitwise_or(mask_green, mask_lesion)

    # Morphological cleanup to remove isolated noise specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)

    total_pixels = float(height * width)
    leaf_pixels = float(np.count_nonzero(leaf_mask))
    leaf_coverage_ratio = round(leaf_pixels / max(total_pixels, 1.0), 4)

    quality_warnings: List[str] = []
    if blur_score < MIN_BLUR_VARIANCE:
        quality_warnings.append(
            "Image appears blurry or out of focus. Hold the camera steady for a sharper shot."
        )
    if mean_brightness < MIN_BRIGHTNESS:
        quality_warnings.append(
            "Image is too dark. Move to brighter, indirect natural light."
        )
    elif mean_brightness > MAX_BRIGHTNESS:
        quality_warnings.append(
            "Image is overexposed. Avoid harsh direct glare on the leaf."
        )
    if leaf_coverage_ratio < 0.04:
        quality_warnings.append(
            "Very little plant leaf region detected. Center a single leaf clearly in the frame."
        )

    # 4. Gentle leaf ROI focusing when background dominates (leaf covers 10% - 72%)
    focused_img = rgb_img
    leaf_focused = False
    if 0.10 <= leaf_coverage_ratio <= 0.72:
        contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            contour_ratio = cv2.contourArea(largest) / max(total_pixels, 1.0)
            if contour_ratio >= 0.08:
                x, y, w, h = cv2.boundingRect(largest)
                # Add generous 18% margin so diseased edges and full context are retained
                pad_x = int(w * 0.18)
                pad_y = int(h * 0.18)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(width, x + w + pad_x)
                y2 = min(height, y + h + pad_y)
                crop_w, crop_h = x2 - x1, y2 - y1
                # Only crop if it meaningfully focuses while staying >= 96px
                if crop_w >= 96 and crop_h >= 96 and (crop_w * crop_h) < (0.88 * total_pixels):
                    cropped_rgb = np_rgb[y1:y2, x1:x2]
                    # Mild edge-preserving bilateral denoise to reduce camera sensor grain
                    denoised_rgb = cv2.bilateralFilter(cropped_rgb, d=5, sigmaColor=25, sigmaSpace=25)
                    focused_img = Image.fromarray(denoised_rgb)
                    leaf_focused = True

    diagnostics = {
        "original_width": int(width),
        "original_height": int(height),
        "blur_score": round(blur_score, 2),
        "mean_brightness": round(mean_brightness, 2),
        "leaf_coverage_ratio": leaf_coverage_ratio,
        "leaf_focused": leaf_focused,
        "quality_warnings": quality_warnings,
    }
    return focused_img, diagnostics


def preprocess_for_model(rgb_img: Image.Image) -> Tuple[torch.Tensor, dict]:
    """Run quality analysis, leaf focus, resizing, and ImageNet normalization.

    Returns:
        (batched_tensor [1, 3, 224, 224], diagnostics_dict)
    """
    processed_img, diagnostics = analyze_and_focus_leaf(rgb_img)
    transform = get_inference_transform()
    tensor = transform(processed_img).unsqueeze(0)
    return tensor, diagnostics
