"""End-to-end automated tests for the Plant Disease Detection REST API."""

from io import BytesIO
from pathlib import Path
import sys

from fastapi.testclient import TestClient
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import app
from backend.config import DATASET_DIR


def _make_sample_bytes(color=(30, 140, 40), size=(256, 256), fmt="JPEG") -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_health_and_plants_endpoints():
    with TestClient(app) as client:
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["database_connected"] is True
        assert data["model_loaded"] is True
        assert data["num_classes"] >= 7
        assert len(data["supported_plants"]) >= 7

        plants_res = client.get("/api/plants")
        assert plants_res.status_code == 200
        plants_data = plants_res.json()
        assert plants_data["model_loaded"] is True
        assert len(plants_data["plants"]) >= 7


def test_frontend_index_served():
    with TestClient(app) as client:
        res = client.get("/")
        assert res.status_code == 200
        assert "FloraGuard" in res.text


def test_upload_validation_rejects_invalid_files():
    with TestClient(app) as client:
        # Reject non-image text file
        res = client.post(
            "/api/upload",
            files={"file": ("notes.txt", b"not an image", "text/plain")},
        )
        assert res.status_code == 400

        # Reject corrupted JPEG bytes
        res = client.post(
            "/api/upload",
            files={"file": ("broken.jpg", b"not-real-jpeg-bytes", "image/jpeg")},
        )
        assert res.status_code == 400

        # Reject tiny image below minimum dimension
        tiny = _make_sample_bytes(size=(24, 24))
        res = client.post(
            "/api/upload",
            files={"file": ("tiny.jpg", tiny, "image/jpeg")},
        )
        assert res.status_code == 400


def test_upload_and_predict_with_real_test_leaf_and_history_lifecycle():
    """Tests POST /api/upload, POST /api/predict, GET /api/history, and DELETE /api/history/{id}
    using a real held-out test image from PlantVillage."""
    test_csv = DATASET_DIR / "splits" / "test.csv"
    assert test_csv.exists(), "Run training/preprocess.py before running full API tests."

    import pandas as pd
    df = pd.read_csv(test_csv)
    # Pick a real Tomato Early Blight or first available test image
    tomato_rows = df[df["class_label"].str.contains("Tomato___Early_blight", case=False, na=False)]
    row = tomato_rows.iloc[0] if len(tomato_rows) > 0 else df.iloc[0]
    sample_path = DATASET_DIR / row["relative_path"]
    assert sample_path.exists()
    img_bytes = sample_path.read_bytes()

    with TestClient(app) as client:
        # 1. Test POST /api/upload
        up_res = client.post(
            "/api/upload",
            files={"file": (sample_path.name, img_bytes, "image/jpeg")},
        )
        assert up_res.status_code == 201
        up_data = up_res.json()
        assert up_data["file_id"]
        assert up_data["image_url"].startswith("/uploads/")

        # 2. Test POST /api/predict using uploaded file
        pred_res = client.post(
            "/api/predict",
            files={"file": (sample_path.name, img_bytes, "image/jpeg")},
            data={"input_source": "test_suite"},
        )
        assert pred_res.status_code == 200
        pred = pred_res.json()
        assert pred["record_id"] is not None
        assert pred["status"] in ("Healthy", "Infected")
        assert pred["confidence"] >= 60.0
        assert pred["disease_info"] is not None
        assert len(pred["disease_info"]["symptoms"]) > 0
        assert len(pred["disease_info"]["prevention"]) > 0
        assert len(pred["top_predictions"]) >= 3

        record_id = pred["record_id"]

        # 3. Verify GET /api/history contains the new record
        hist_res = client.get("/api/history")
        assert hist_res.status_code == 200
        hist = hist_res.json()
        assert hist["total"] >= 1
        assert any(item["id"] == record_id for item in hist["items"])

        # 4. Clean up test record via DELETE /api/history/{id}
        del_res = client.delete(f"/api/history/{record_id}")
        assert del_res.status_code == 200
        assert del_res.json()["deleted"] is True


def test_uncertain_safeguard_on_dark_blank_image():
    """Verify the AI refuses to fake a confident diagnosis on a pitch-black image."""
    black_bytes = _make_sample_bytes(color=(2, 2, 2), size=(256, 256))
    with TestClient(app) as client:
        res = client.post(
            "/api/predict",
            files={"file": ("black.jpg", black_bytes, "image/jpeg")},
            data={"input_source": "test_suite"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["is_uncertain"] is True
        assert data["status"] == "Uncertain"
        assert data["uncertainty_message"]

        # Clean up test record
        if data.get("record_id"):
            client.delete(f"/api/history/{data['record_id']}")
