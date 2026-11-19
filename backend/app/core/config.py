"""Environment-driven application configuration for PlantDiseaseAI Backend (Phase 7).

Loads optional `backend/.env` via python-dotenv and resolves all filesystem paths
relative to the PlantDiseaseAI root so zero machine-specific paths are hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Set

from dotenv import load_dotenv

# PlantDiseaseAI/
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"

# Load local .env if present (never overrides explicit environment variables)
_env_file = BACKEND_ROOT / ".env"
if _env_file.exists():
    load_dotenv(_env_file, override=False)


def _resolve_project_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _parse_bool(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _parse_origins(raw: str | None) -> List[str]:
    default_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    if not raw:
        return default_origins
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or default_origins


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings for the FastAPI backend."""

    app_name: str = "PlantDiseaseAI — AgriVision-CNN Backend API"
    app_version: str = field(default_factory=lambda: os.getenv("APP_VERSION", "1.0.0"))
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    api_v1_prefix: str = "/api/v1"
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())

    cors_origins: List[str] = field(
        default_factory=lambda: _parse_origins(os.getenv("CORS_ORIGINS"))
    )

    # Phase 2–4 ML paths
    model_checkpoint_path: Path = field(
        default_factory=lambda: _resolve_project_path(
            os.getenv(
                "MODEL_CHECKPOINT_PATH",
                "ml/experiments/checkpoints/exp_a_agrivision_cnn/best_model.pth",
            )
        )
    )
    model_config_path: Path = field(
        default_factory=lambda: _resolve_project_path(
            os.getenv("MODEL_CONFIG_PATH", "ml/configs/model_config.yaml")
        )
    )
    dataset_config_path: Path = field(
        default_factory=lambda: _resolve_project_path(
            os.getenv("DATASET_CONFIG_PATH", "ml/configs/dataset_config.yaml")
        )
    )
    class_mapping_path: Path = field(
        default_factory=lambda: _resolve_project_path(
            os.getenv("CLASS_MAPPING_PATH", "ml/configs/class_mapping.yaml")
        )
    )

    # Inference & Uncertainty policy
    inference_device: str = field(
        default_factory=lambda: os.getenv("INFERENCE_DEVICE", "auto").strip().lower()
    )
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
    )
    entropy_threshold: float = field(
        default_factory=lambda: float(os.getenv("ENTROPY_THRESHOLD", "1.60"))
    )
    min_margin_threshold: float = field(
        default_factory=lambda: float(os.getenv("MIN_MARGIN_THRESHOLD", "0.10"))
    )
    top_k_predictions: int = field(
        default_factory=lambda: int(os.getenv("TOP_K_PREDICTIONS", "3"))
    )

    # Upload validation & storage
    max_upload_size_bytes: int = field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(10 * 1024 * 1024)))
    )
    min_image_dimension: int = 32
    max_image_dimension: int = 10000
    supported_extensions: Set[str] = field(
        default_factory=lambda: {".jpg", ".jpeg", ".png", ".webp"}
    )
    supported_formats: Set[str] = field(
        default_factory=lambda: {"JPEG", "JPG", "PNG", "WEBP"}
    )
    supported_mime_types: Set[str] = field(
        default_factory=lambda: {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    )
    retain_uploaded_images: bool = field(
        default_factory=lambda: _parse_bool(os.getenv("RETAIN_UPLOADED_IMAGES"), default=False)
    )

    # Controlled runtime directories (never exposed directly in API responses)
    results_dir: Path = field(
        default_factory=lambda: _resolve_project_path(
            os.getenv("RESULTS_DIR", "backend/runtime/results")
        )
    )
    temp_upload_dir: Path = field(
        default_factory=lambda: _resolve_project_path(
            os.getenv("TEMP_UPLOAD_DIR", "backend/runtime/tmp")
        )
    )
    retained_uploads_dir: Path = field(
        default_factory=lambda: _resolve_project_path(
            os.getenv("RETAINED_UPLOADS_DIR", "backend/runtime/uploads")
        )
    )

    # Database URL
    database_url: str = field(default_factory=lambda: _build_database_url())

    # Rate limit
    rate_limit_per_minute: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    )

    def ensure_runtime_dirs(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.temp_upload_dir.mkdir(parents=True, exist_ok=True)
        if self.retain_uploaded_images:
            self.retained_uploads_dir.mkdir(parents=True, exist_ok=True)
        if self.database_url.startswith("sqlite:///"):
            db_path = Path(self.database_url.replace("sqlite:///", "", 1))
            db_path.parent.mkdir(parents=True, exist_ok=True)


def _build_database_url() -> str:
    raw = os.getenv("DATABASE_URL", "").strip()
    if not raw:
        default_db = PROJECT_ROOT / "backend" / "runtime" / "plantdiseaseai.db"
        return f"sqlite:///{default_db.as_posix()}"
    if raw.startswith("sqlite:///") and not raw.startswith("sqlite:///:memory:"):
        rel_or_abs = raw.replace("sqlite:///", "", 1)
        p = Path(rel_or_abs)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return f"sqlite:///{p.as_posix()}"
    return raw


def get_settings() -> Settings:
    """Create a fresh Settings instance from the current environment."""
    settings = Settings()
    if settings.app_env.lower() == "production" and "*" in settings.cors_origins:
        raise ValueError("Unrestricted wildcard CORS ('*') is forbidden in production.")
    return settings
