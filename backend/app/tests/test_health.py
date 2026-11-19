"""Automated tests for application startup, health, system info, DB, and Swagger docs."""

from __future__ import annotations

from pathlib import Path
import sys

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import Settings
from backend.app.main import create_app


def _make_test_settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{(tmp_path / 'test_health.db').as_posix()}",
        model_checkpoint_path=tmp_path / "nonexistent_checkpoint.pth",
        results_dir=tmp_path / "results",
        temp_upload_dir=tmp_path / "tmp",
        retained_uploads_dir=tmp_path / "uploads",
        inference_device="cpu",
    )


def test_startup_health_and_swagger_without_checkpoint(tmp_path: Path):
    """App must start cleanly without a trained checkpoint and report model_ready=False."""
    app = create_app(settings=_make_test_settings(tmp_path))
    with TestClient(app) as client:
        res = client.get("/api/v1/health")
        assert res.status_code == 200
        data = res.json()
        assert data["api_status"] == "online"
        assert data["database_status"] == "connected"
        assert data["model_ready"] is False
        assert data["status"] == "degraded"
        assert "unavailable" in data["model_status_detail"].lower()
        assert "X-Request-ID" in res.headers

        # Ensure internal local paths are never leaked in health response
        assert str(tmp_path) not in res.text

        # Swagger UI & OpenAPI schema available
        docs_res = client.get("/docs")
        assert docs_res.status_code == 200
        schema_res = client.get("/openapi.json")
        assert schema_res.status_code == 200
        assert "/api/v1/predict" in schema_res.json()["paths"]


def test_system_info_returns_safe_technical_metadata(tmp_path: Path):
    app = create_app(settings=_make_test_settings(tmp_path))
    with TestClient(app) as client:
        res = client.get("/api/v1/system/info")
        assert res.status_code == 200
        data = res.json()
        assert data["model_name"] == "agrivision_cnn"
        assert data["num_classes"] == 38
        assert data["checkpoint_available"] is False
        assert data["checkpoint_id"] is None
        assert set(data["supported_image_formats"]) == {"JPEG", "JPG", "PNG", "WEBP"}
        assert data["max_upload_size_bytes"] == 10 * 1024 * 1024
        assert data["image_retention_enabled"] is False
        assert "approximate" in data["explanation_disclaimer"].lower()
        assert str(tmp_path) not in res.text
