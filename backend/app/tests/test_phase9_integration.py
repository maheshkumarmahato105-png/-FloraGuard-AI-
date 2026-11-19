"""Phase 9 Full-Stack Integration, Preprocessing Parity, Grad-CAM Concurrency & Security Tests.

Note on Model Weights:
  - Tests involving synthetic weights create temporary files strictly inside pytest's `tmp_path`
    to verify real PyTorch architecture wiring, preprocessing parity, Grad-CAM hook cleanup,
    concurrency, and error resilience.
  - Synthetic pipeline tests NEVER represent trained model classification accuracy.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image, ImageFilter
import pytest
from sqlalchemy.exc import OperationalError
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SRC = PROJECT_ROOT / "ml" / "src"
for p in (PROJECT_ROOT, ML_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.utils.image_validation import validate_and_decode_bytes
from models import create_model, load_model_config, resolve_verified_classes
from preprocess import preprocess_image_tensor


def _encode_image(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=95)
    else:
        img.save(buf, format=fmt)
    return buf.getvalue()


def _make_settings(tmp_path: Path, ckpt_path: Path, device: str = "cpu") -> Settings:
    return Settings(
        database_url=f"sqlite:///{(tmp_path / 'phase9_int.db').as_posix()}",
        model_checkpoint_path=ckpt_path,
        results_dir=tmp_path / "results",
        temp_upload_dir=tmp_path / "tmp",
        retained_uploads_dir=tmp_path / "uploads",
        inference_device=device,
    )


def _create_temp_synthetic_checkpoint(tmp_path: Path, confident_class_idx: int = 0) -> Path:
    """Isolated temporary synthetic checkpoint in tmp_path (pipeline verification only)."""
    model_cfg = load_model_config()
    class_names, _ = resolve_verified_classes(model_cfg)
    model = create_model("agrivision_cnn", model_config=model_cfg, num_classes=len(class_names), seed=7)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.fill_(-5.0)
        model.classifier.bias[confident_class_idx] = 5.0

    ckpt_path = tmp_path / "phase9_temp_pipeline.pth"
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


def test_preprocessing_parity_between_phase2_and_phase7_backend(tmp_path: Path):
    """Prove bit-for-bit tensor parity between Phase 2 `preprocess_image_tensor(..., mode='eval')`
    and Phase 7 `validate_and_decode_bytes` + `ModelService.preprocess_image` across varied
    image modes, aspect ratios, and real dataset leaf images when present."""
    settings = _make_settings(tmp_path, tmp_path / "missing.pth")
    app = create_app(settings=settings)
    ms = app.state.model_service

    test_cases: list[tuple[Path, str]] = []

    # 1. Synthetic non-square RGB JPEG (tests RGB/BGR order & aspect-ratio letterboxing)
    arr = np.zeros((360, 640, 3), dtype=np.uint8)
    arr[:, :, 0] = 210  # strong R channel to catch any accidental RGB<->BGR swap
    arr[:, :, 1] = 95
    arr[:, :, 2] = 30
    rgb_path = tmp_path / "nonsquare_rgb.jpg"
    Image.fromarray(arr, mode="RGB").save(rgb_path, format="JPEG", quality=95)
    test_cases.append((rgb_path, "image/jpeg"))

    # 2. RGBA PNG (tests alpha stripping & RGB conversion parity)
    rgba = Image.new("RGBA", (300, 480), color=(40, 180, 65, 220))
    rgba_path = tmp_path / "alpha_leaf.png"
    rgba.save(rgba_path, format="PNG")
    test_cases.append((rgba_path, "image/png"))

    # 3. Grayscale PNG (tests single-channel -> 3-channel RGB parity)
    gray = Image.new("L", (256, 192), color=120)
    gray_path = tmp_path / "gray_leaf.png"
    gray.save(gray_path, format="PNG")
    test_cases.append((gray_path, "image/png"))

    # 4. Real PlantVillage test split images (healthy & diseased) if present on disk
    test_split_dir = PROJECT_ROOT / "ml" / "data" / "processed" / "test"
    if test_split_dir.exists():
        for cls_dir in ("tomato__healthy", "tomato__early_blight"):
            d = test_split_dir / cls_dir
            if d.exists():
                first_img = next((p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")), None)
                if first_img is not None:
                    test_cases.append((first_img, "image/jpeg"))

    assert len(test_cases) >= 3

    for img_path, content_type in test_cases:
        phase2_tensor = preprocess_image_tensor(img_path, ms.dataset_config, mode="eval").unsqueeze(0)
        decoded = validate_and_decode_bytes(
            file_bytes=img_path.read_bytes(),
            settings=settings,
            filename=img_path.name,
            content_type=content_type,
        )
        phase7_tensor = ms.preprocess_image(decoded.rgb_image)

        assert phase7_tensor.shape == (1, 3, 224, 224)
        assert phase7_tensor.dtype == torch.float32
        assert torch.allclose(phase2_tensor, phase7_tensor, atol=1e-6), (
            f"Preprocessing mismatch detected for {img_path.name}"
        )


def test_gradcam_concurrency_dimensions_cors_and_failure_resilience(tmp_path: Path):
    """Verify Grad-CAM output dimensions (224x224), CORS headers on explanation images,
    concurrent thread safety with zero dangling hooks, and graceful prediction preservation
    when Grad-CAM fails."""
    ckpt_path = _create_temp_synthetic_checkpoint(tmp_path, confident_class_idx=0)
    app = create_app(settings=_make_settings(tmp_path, ckpt_path, device="cpu"))

    with TestClient(app) as client:
        # 1. Verify CORS header & exact 224x224 PNG dimensions on /api/v1/explanations/{filename}
        sample_rgb = Image.new("RGB", (400, 300), color=(45, 160, 55))
        res = client.post(
            "/api/v1/predict?generate_explanation=true",
            files={"file": ("leaf.jpg", _encode_image(sample_rgb), "image/jpeg")},
            headers={"Origin": "http://127.0.0.1:5173"},
        )
        assert res.status_code == 200
        assert res.headers.get("access-control-allow-origin") == "http://127.0.0.1:5173"
        body = res.json()
        assert body["explanation"]["available"] is True

        for url_key in ("heatmap_url", "overlay_url"):
            art_res = client.get(
                body["explanation"][url_key],
                headers={"Origin": "http://127.0.0.1:5173"},
            )
            assert art_res.status_code == 200
            assert art_res.headers["content-type"] == "image/png"
            assert art_res.headers.get("access-control-allow-origin") == "http://127.0.0.1:5173"
            with Image.open(BytesIO(art_res.content)) as art_img:
                assert art_img.size == (224, 224)
                assert art_img.mode == "RGB"

        # 2. Concurrent multi-threaded Grad-CAM safety & hook cleanup verification
        exp_service = app.state.explanation_service
        ms = app.state.model_service
        tensor = ms.preprocess_image(sample_rgb)

        def _run_cam(idx: int):
            return exp_service.generate_gradcam(
                rgb_image=sample_rgb,
                input_tensor=tensor,
                target_class_idx=0,
                request_id=f"concurrentreq{idx:04d}abcdef",
            )

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(_run_cam, range(4)))

        assert all(r["available"] is True for r in results)
        for mod in ms.model.modules():
            assert len(mod._forward_hooks) == 0
            assert len(mod._backward_hooks) == 0
            assert len(mod._backward_pre_hooks) == 0

        # 3. Explanation failure resilience: prediction still succeeds and is returned cleanly
        original_compute = exp_service._compute_cam

        def _failing_cam(*_args, **_kwargs):
            raise RuntimeError("Simulated Grad-CAM failure")

        exp_service._compute_cam = _failing_cam
        try:
            res_fail_cam = client.post(
                "/api/v1/predict?generate_explanation=true",
                files={"file": ("leaf2.jpg", _encode_image(sample_rgb), "image/jpeg")},
            )
            assert res_fail_cam.status_code == 200
            fail_body = res_fail_cam.json()
            assert fail_body["status"] == "success"
            assert fail_body["prediction"] is not None
            assert fail_body["explanation"]["available"] is False
            assert fail_body["explanation"]["requested"] is True
        finally:
            exp_service._compute_cam = original_compute


def test_history_detail_includes_disease_info_for_success_and_null_for_uncertain(tmp_path: Path):
    """Verify BUG-02 fix + edge-case inputs (blurry, dark, non-plant) uncertainty rejection."""
    app = create_app(settings=_make_settings(tmp_path, tmp_path / "missing.pth"))

    with TestClient(app) as client:
        ms = app.state.model_service
        ms.is_ready = True
        ms.checkpoint_id = "mock-phase9-history-check"
        n_cls = len(ms.class_names)
        idx_target = ms.class_names.index("tomato__early_blight")

        # Accepted prediction
        conf = np.full(n_cls, 0.001, dtype=np.float64)
        conf[idx_target] = 0.95
        conf /= conf.sum()
        ms.predict_tensor = lambda _t: (np.zeros(n_cls), conf)

        r_ok = client.post(
            "/api/v1/predict",
            files={"file": ("leaf.jpg", _encode_image(Image.new("RGB", (128, 128), (30, 130, 40))), "image/jpeg")},
        )
        assert r_ok.status_code == 200
        req_ok = r_ok.json()["request_id"]

        # Reopen from history -> MUST include structured disease_information
        hist_ok = client.get(f"/api/v1/history/{req_ok}")
        assert hist_ok.status_code == 200
        hist_ok_data = hist_ok.json()
        assert hist_ok_data["disease_information"] is not None
        assert hist_ok_data["disease_information"]["disease_name"] == "Early Blight"
        assert len(hist_ok_data["disease_information"]["symptoms"]) > 0

        # Edge-case uploads (blurry, very dark, non-plant checkerboard) with uncertain probability distribution
        unc_probs = np.full(n_cls, 1.0 / n_cls, dtype=np.float64)
        ms.predict_tensor = lambda _t: (np.zeros(n_cls), unc_probs)

        blurry = Image.new("RGB", (128, 128), (80, 90, 80)).filter(ImageFilter.GaussianBlur(radius=6))
        very_dark = Image.new("RGB", (128, 128), (4, 5, 4))
        checker = Image.fromarray(
            np.indices((128, 128)).sum(axis=0).astype(np.uint8) % 2 * 255
        ).convert("RGB")

        seen_ids = {req_ok}
        for name, img in (("blurry.jpg", blurry), ("dark.jpg", very_dark), ("nonplant.png", checker)):
            fmt = "PNG" if name.endswith(".png") else "JPEG"
            mime = "image/png" if fmt == "PNG" else "image/jpeg"
            r_u = client.post(
                "/api/v1/predict",
                files={"file": (name, _encode_image(img, fmt=fmt), mime)},
            )
            assert r_u.status_code == 200
            u_body = r_u.json()
            assert u_body["request_id"] not in seen_ids
            seen_ids.add(u_body["request_id"])
            assert u_body["status"] == "uncertain"
            assert u_body["prediction"] is None
            assert u_body["disease_information"] is None

            # Reopen uncertain record from history -> MUST NOT fabricate disease_information
            hist_u = client.get(f"/api/v1/history/{u_body['request_id']}")
            assert hist_u.status_code == 200
            assert hist_u.json()["disease_information"] is None
            assert hist_u.json()["predicted_class"] is None


def test_database_unavailable_returns_sanitized_503_and_cors_blocks_unlisted_origin(tmp_path: Path):
    """Verify BUG-03 fix (sanitized 503 database_unavailable without stack traces) and CORS allowlist."""
    app = create_app(settings=_make_settings(tmp_path, tmp_path / "missing.pth"))

    with TestClient(app, raise_server_exceptions=False) as client:
        # Disallowed CORS origin must NOT be echoed
        r_cors = client.get(
            "/api/v1/health",
            headers={"Origin": "http://untrusted-origin.example"},
        )
        assert r_cors.status_code == 200
        assert r_cors.headers.get("access-control-allow-origin") is None

        # Simulate SQLAlchemy OperationalError on history listing
        original_list = app.state.history_service.list_history

        def _broken_db(*_args, **_kwargs):
            raise OperationalError("SELECT * FROM prediction_history", {}, Exception("disk I/O error"))

        app.state.history_service.list_history = _broken_db
        try:
            r_db = client.get("/api/v1/history")
            assert r_db.status_code == 503
            err_body = r_db.json()
            assert err_body["status"] == "error"
            assert err_body["error_code"] == "database_unavailable"
            assert "SELECT" not in err_body["detail"]
            assert "Traceback" not in err_body["detail"]
        finally:
            app.state.history_service.list_history = original_list
