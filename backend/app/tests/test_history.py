"""Automated tests for prediction history CRUD, pagination, filtering, and deletion."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import Settings
from backend.app.main import create_app


def _sample_jpeg() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (128, 128), color=(30, 140, 40)).save(buf, format="JPEG")
    return buf.getvalue()


def test_history_creation_pagination_filtering_and_deletion(tmp_path: Path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'test_hist.db').as_posix()}",
        model_checkpoint_path=tmp_path / "missing.pth",
        results_dir=tmp_path / "results",
        temp_upload_dir=tmp_path / "tmp",
        retained_uploads_dir=tmp_path / "uploads",
        inference_device="cpu",
    )
    app = create_app(settings=settings)

    with TestClient(app) as client:
        ms = app.state.model_service
        ms.is_ready = True
        ms.checkpoint_id = "mock-history-test"
        n_cls = len(ms.class_names)

        # Create 3 records: 2 success (different classes) + 1 uncertain
        idx_tomato = ms.class_names.index("tomato__early_blight")
        idx_apple = ms.class_names.index("apple__healthy")

        def _make_conf_probs(idx: int):
            p = np.full(n_cls, 0.001, dtype=np.float64)
            p[idx] = 0.96
            return p / p.sum()

        created_ids = []
        for probs in (
            _make_conf_probs(idx_tomato),
            _make_conf_probs(idx_apple),
            np.full(n_cls, 1.0 / n_cls, dtype=np.float64),
        ):
            ms.predict_tensor = lambda _t, p=probs: (np.zeros(n_cls), p)
            r = client.post(
                "/api/v1/predict",
                files={"file": ("leaf.jpg", _sample_jpeg(), "image/jpeg")},
            )
            assert r.status_code == 200
            created_ids.append(r.json()["request_id"])

        # 1. List all (newest first)
        all_res = client.get("/api/v1/history?limit=10&offset=0")
        assert all_res.status_code == 200
        all_data = all_res.json()
        assert all_data["total"] == 3
        assert len(all_data["items"]) == 3
        assert all_data["items"][0]["request_id"] == created_ids[2]

        # 2. Pagination (limit=2, offset=1)
        page_res = client.get("/api/v1/history?limit=2&offset=1").json()
        assert page_res["total"] == 3
        assert len(page_res["items"]) == 2
        assert page_res["items"][0]["request_id"] == created_ids[1]

        # 3. Filter by status
        unc_res = client.get("/api/v1/history?status=uncertain").json()
        assert unc_res["total"] == 1
        assert unc_res["items"][0]["status"] == "uncertain"

        # 4. Filter by predicted_class
        tom_res = client.get("/api/v1/history?predicted_class=tomato__early_blight").json()
        assert tom_res["total"] == 1
        assert tom_res["items"][0]["predicted_class"] == "tomato__early_blight"

        # 5. Get single record by request_id
        one_res = client.get(f"/api/v1/history/{created_ids[0]}")
        assert one_res.status_code == 200
        assert one_res.json()["predicted_class"] == "tomato__early_blight"

        # 6. Delete record explicitly
        del_res = client.delete(f"/api/v1/history/{created_ids[0]}")
        assert del_res.status_code == 200
        assert del_res.json() == {"request_id": created_ids[0], "deleted": True}

        # Verify deleted record returns 404
        gone_res = client.get(f"/api/v1/history/{created_ids[0]}")
        assert gone_res.status_code == 404

        # Verify total decremented
        after_res = client.get("/api/v1/history").json()
        assert after_res["total"] == 2
