"""Automated tests for POST /api/v1/predict, uncertainty rejection, Grad-CAM, CPU & GPU.

Clearly separates:
  - Mocked API unit tests (using synthetic probability vectors; never reported as real AI predictions).
  - Isolated synthetic-checkpoint pipeline integration tests (inside pytest `tmp_path`, testing
    real AgriVision-CNN forward pass, Grad-CAM hook cleanup, and CPU/CUDA execution).
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SRC = PROJECT_ROOT / "ml" / "src"
for p in (PROJECT_ROOT, ML_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.app.core.config import Settings
from backend.app.main import create_app
from models import create_model, load_model_config, resolve_verified_classes


def _sample_jpeg_bytes(size=(224, 224), color=(35, 145, 45)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _make_settings(tmp_path: Path, ckpt_path: Path, device: str = "cpu", max_bytes: int = 10 * 1024 * 1024) -> Settings:
    return Settings(
        database_url=f"sqlite:///{(tmp_path / 'test_pred.db').as_posix()}",
        model_checkpoint_path=ckpt_path,
        results_dir=tmp_path / "results",
        temp_upload_dir=tmp_path / "tmp",
        retained_uploads_dir=tmp_path / "uploads",
        inference_device=device,
        max_upload_size_bytes=max_bytes,
    )


def test_predict_returns_503_when_checkpoint_missing(tmp_path: Path):
    app = create_app(settings=_make_settings(tmp_path, tmp_path / "missing.pth"))
    with TestClient(app) as client:
        res = client.post(
            "/api/v1/predict",
            files={"file": ("leaf.jpg", _sample_jpeg_bytes(), "image/jpeg")},
        )
        assert res.status_code == 503
        body = res.json()
        assert body["error_code"] == "model_unavailable"


def test_predict_mocked_confident_and_uncertain_rejection_and_upload_errors(tmp_path: Path):
    """Mocked unit tests for API schema, uncertainty rejection, and upload error codes."""
    app = create_app(settings=_make_settings(tmp_path, tmp_path / "missing.pth", max_bytes=200_000))
    with TestClient(app) as client:
        ms = app.state.model_service
        ms.is_ready = True
        ms.checkpoint_id = "mock-unit-test-only"

        # 1. Mocked confident Tomato Early Blight prediction
        target_idx = ms.class_names.index("tomato__early_blight")
        confident_probs = np.full(len(ms.class_names), 0.001, dtype=np.float64)
        confident_probs[target_idx] = 0.963
        confident_probs /= confident_probs.sum()
        ms.predict_tensor = lambda _tensor: (np.zeros(len(ms.class_names)), confident_probs)

        res = client.post(
            "/api/v1/predict?generate_explanation=false",
            files={"file": ("leaf.jpg", _sample_jpeg_bytes(), "image/jpeg")},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        assert body["prediction"]["class_name"] == "tomato__early_blight"
        assert body["prediction"]["is_healthy"] is False
        assert body["prediction"]["confidence"] > 0.90
        assert len(body["prediction"]["top_predictions"]) == 3
        assert body["uncertainty"]["is_uncertain"] is False
        assert body["disease_information"] is not None
        assert len(body["disease_information"]["symptoms"]) > 0
        assert "educational" in body["disease_information"]["disclaimer"].lower()

        # 2. Mocked uniform (high-entropy, low-confidence) prediction -> MUST reject as `uncertain`
        uncertain_probs = np.full(len(ms.class_names), 1.0 / len(ms.class_names), dtype=np.float64)
        ms.predict_tensor = lambda _tensor: (np.zeros(len(ms.class_names)), uncertain_probs)

        res_unc = client.post(
            "/api/v1/predict",
            files={"file": ("blurry.jpg", _sample_jpeg_bytes(), "image/jpeg")},
        )
        assert res_unc.status_code == 200
        unc = res_unc.json()
        assert unc["status"] == "uncertain"
        assert unc["prediction"] is None
        assert unc["disease_information"] is None
        assert unc["uncertainty"]["is_uncertain"] is True
        assert len(unc["uncertainty"]["rejection_reasons"]) >= 1

        # 3. Invalid uploads -> 415 / 400 / 413
        res_415 = client.post("/api/v1/predict", files={"file": ("a.txt", b"hello", "text/plain")})
        assert res_415.status_code == 415

        res_400 = client.post("/api/v1/predict", files={"file": ("bad.jpg", b"\xff\xd8\xff\xe0broken", "image/jpeg")})
        assert res_400.status_code == 400

        oversized = b"\xff\xd8\xff" + (b"\x00" * 210_000)
        res_413 = client.post("/api/v1/predict", files={"file": ("huge.jpg", oversized, "image/jpeg")})
        assert res_413.status_code == 413


def _create_temp_synthetic_checkpoint(tmp_path: Path) -> Path:
    """Create a temporary synthetic checkpoint strictly inside pytest `tmp_path` to test
    real PyTorch AgriVision-CNN loading, Grad-CAM hooks, and CPU/CUDA execution."""
    model_cfg = load_model_config()
    class_names, _ = resolve_verified_classes(model_cfg)
    model = create_model("agrivision_cnn", model_config=model_cfg, num_classes=len(class_names), seed=42)
    # Bias class 0 logit slightly so confidence passes threshold deterministically for Grad-CAM pipeline test
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.fill_(-5.0)
        model.classifier.bias[0] = 5.0

    ckpt_path = tmp_path / "synthetic_pipeline_test.pth"
    torch.save(
        {
            "architecture_name": "agrivision_cnn",
            "epoch": 1,
            "is_trained": True,
            "best_val_macro_f1": 0.0,
            "class_names": class_names,
            "model_config": model_cfg,
            "model_state_dict": model.state_dict(),
        },
        ckpt_path,
    )
    return ckpt_path


@pytest.mark.parametrize("device_opt", ["cpu", "cuda"])
def test_real_agrivision_pipeline_and_gradcam_cpu_and_cuda(tmp_path: Path, device_opt: str):
    """Integration test using temporary synthetic weights in tmp_path to verify real
    AgriVision-CNN loading, forward pass, Grad-CAM heatmap/overlay generation, hook cleanup,
    and safe artifact serving on CPU and CUDA."""
    if device_opt == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available.")

    run_dir = tmp_path / f"run_{device_opt}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = _create_temp_synthetic_checkpoint(run_dir)
    app = create_app(settings=_make_settings(run_dir, ckpt_path, device=device_opt))

    with TestClient(app) as client:
        h_res = client.get("/api/v1/health").json()
        assert h_res["model_ready"] is True
        assert h_res["inference_device"] == device_opt

        res = client.post(
            "/api/v1/predict?generate_explanation=true",
            files={"file": ("leaf.jpg", _sample_jpeg_bytes(), "image/jpeg")},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        assert body["explanation"]["available"] is True
        assert body["explanation"]["heatmap_url"].startswith("/api/v1/explanations/")
        assert body["explanation"]["overlay_url"].startswith("/api/v1/explanations/")
        assert "approximate" in body["explanation"]["note"].lower()

        # Verify generated heatmap & overlay PNGs are served cleanly
        hm_res = client.get(body["explanation"]["heatmap_url"])
        ov_res = client.get(body["explanation"]["overlay_url"])
        assert hm_res.status_code == 200 and hm_res.headers["content-type"] == "image/png"
        assert ov_res.status_code == 200 and ov_res.headers["content-type"] == "image/png"

        # Verify path traversal against explanation endpoint is blocked
        trav_res = client.get("/api/v1/explanations/..%2F..%2Fmain.py")
        assert trav_res.status_code == 404

        # Verify zero dangling hooks remain on AgriVision-CNN after Grad-CAM
        for mod in app.state.model_service.model.modules():
            assert len(mod._forward_hooks) == 0
            assert len(mod._backward_hooks) == 0
            assert len(mod._backward_pre_hooks) == 0
