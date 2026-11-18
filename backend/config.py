"""Centralized configuration for the Plant Disease Detection backend."""

from pathlib import Path
import os

# Base directories
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

FRONTEND_DIR = PROJECT_ROOT / "frontend"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
DATABASE_DIR = PROJECT_ROOT / "database"
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAINING_REPORTS_DIR = PROJECT_ROOT / "training" / "reports"
SAVED_MODELS_DIR = BACKEND_DIR / "models" / "saved"

# Ensure runtime directories exist
for _dir in (UPLOADS_DIR, DATABASE_DIR, SAVED_MODELS_DIR, TRAINING_REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# Model artifacts
MODEL_WEIGHTS_PATH = Path(
    os.getenv("MODEL_WEIGHTS_PATH", str(SAVED_MODELS_DIR / "plant_disease_model.pth"))
)
CLASS_NAMES_PATH = Path(
    os.getenv("CLASS_NAMES_PATH", str(SAVED_MODELS_DIR / "class_names.json"))
)
MODEL_METADATA_PATH = Path(
    os.getenv("MODEL_METADATA_PATH", str(SAVED_MODELS_DIR / "model_metadata.json"))
)

# Database configuration (SQLite by default; set DATABASE_URL for PostgreSQL)
DEFAULT_SQLITE_URL = f"sqlite:///{(DATABASE_DIR / 'plant_disease.db').as_posix()}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_SQLITE_URL)

# Upload & Image constraints
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(10 * 1024 * 1024)))  # 10 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}
MIN_IMAGE_DIMENSION = 64
MODEL_INPUT_SIZE = 224

# Inference uncertainty thresholds
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
MIN_BLUR_VARIANCE = float(os.getenv("MIN_BLUR_VARIANCE", "18.0"))
MIN_BRIGHTNESS = float(os.getenv("MIN_BRIGHTNESS", "18.0"))
MAX_BRIGHTNESS = float(os.getenv("MAX_BRIGHTNESS", "245.0"))

# ImageNet normalization constants (used by torchvision MobileNetV2 / EfficientNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
