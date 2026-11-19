# Phase 7 — FastAPI Backend Development & AI Model Integration Plan

**Project:** `PlantDiseaseAI`  
**Verified Project Directory:** `c:\Users\Mahesh Kumar Mahato\Desktop\Plant Disease Detection\PlantDiseaseAI` *(note: `C:\PlantDiseaseAI` does not exist on this machine; all paths resolve dynamically relative to the project root)*.

---

## 1. Inspection of Existing Components (Phases 1–5)

| Area | Status | Verified Assets Reused in Phase 7 |
| :--- | :--- | :--- |
| **Python Environment** | Ready | Python `3.11.18` (`.venv`), PyTorch `2.11.0+cu128` (CUDA on RTX 5050), FastAPI `0.163.2`, Uvicorn `0.54.0`, Pydantic `2.14.3`, OpenCV `5.1.0`, Pillow `13.0.0`, `python-dotenv`, `python-multipart`, `pytest` |
| **Phase 2 Config & Taxonomy** | Ready | `ml/configs/dataset_config.yaml`, `ml/configs/class_mapping.yaml` (38 verified canonical classes), `ml/src/config_loader.py` |
| **Phase 2 Preprocessing** | Ready | `ml/src/preprocess.py` (`safe_load_rgb`, `LetterboxResize`, `get_eval_transform`) — reused directly for exact train/inference parity |
| **Phase 3 Model Architecture** | Ready | `ml/src/models/agrivision_cnn.py` (`AgriVisionCNN`), `simple_cnn.py`, `resnet_baseline.py`, `model_factory.py` (`create_model`, `load_model_config`, `resolve_verified_classes`) |
| **Phase 5 Checkpoint Verification & Softmax** | Ready | `ml/src/evaluation/evaluate.py` (`verify_trained_checkpoint`), `ml/src/evaluation/calibration.py` (`softmax_numpy`) |
| **Backend Scaffolding** | Empty | `backend/__init__.py`, `backend/app/{__init__,api/__init__,core/__init__,models/__init__,services/__init__}.py` |

---

## 2. Missing Components Identified (Verified Honestly)

1. **Trained Phase 4 Checkpoints (`ml/experiments/checkpoints/`):**
   - **Status:** `0` `.pth` files currently present (Phase 4 paused prior to lengthy GPU training awaiting approval).
   - **Phase 7 Handling:** Never fabricate weights or run an untrained random model as a real predictor. The backend starts cleanly, reports `model_ready: false` with a clear diagnostic message on `GET /api/v1/health`, and returns HTTP `503 Service Unavailable` on `POST /api/v1/predict` until a verified trained checkpoint (`is_trained == True`, `epoch >= 1`) is configured.
2. **Phase 6 Explainability & Uncertainty Modules (`ml/src/explainability/`):**
   - **Status:** Phase 6 has not yet been created in `ml/src/`.
   - **Phase 7 Handling:** Implement real, mathematically grounded, thread-safe (`threading.Lock`) PyTorch Grad-CAM in `backend/app/services/explanation_service.py` (with strict forward/backward hook cleanup and clear labeling as *approximate AI attention regions*, never as a verified segmentation mask) and entropy/margin/confidence rejection policy in `backend/app/services/uncertainty_service.py`. Both services are architected for zero-friction delegation to Phase 6 modules once added.
3. **Python Packages (`sqlalchemy`, `httpx`):**
   - **Status:** Not yet installed in `PlantDiseaseAI/.venv` (`fastapi`, `uvicorn`, `pydantic` are already installed).
   - **Action:** Install `sqlalchemy` (for maintainable SQLite ORM) and `httpx` (for FastAPI `TestClient`) into `PlantDiseaseAI/.venv` and record dependencies in `backend/app/requirements.txt`.
4. **Disease Knowledge Base:**
   - **Status:** Not yet present inside `PlantDiseaseAI/`.
   - **Action:** Create structured, educational local agronomic knowledge base in `backend/app/services/disease_information_service.py` covering all 38 canonical classes plus healthy & unknown handling — without inventing chemical dosages, and explicitly omitted (`null`) on uncertain/rejected predictions.

---

## 3. Target Phase 7 Backend Architecture

```text
PlantDiseaseAI/backend/
├── .env.example
└── app/
    ├── __init__.py
    ├── main.py                              # FastAPI app, lifespan, CORS, middleware, exception handlers, /api/v1 routers
    ├── requirements.txt
    ├── api/
    │   ├── __init__.py
    │   └── routes/
    │       ├── __init__.py
    │       ├── health.py                    # GET /api/v1/health
    │       ├── system.py                    # GET /api/v1/system/info
    │       ├── prediction.py                # POST /api/v1/predict + safe GET /api/v1/explanations/{filename}
    │       └── history.py                   # GET/DELETE /api/v1/history & /api/v1/history/{request_id}
    ├── core/
    │   ├── __init__.py
    │   ├── config.py                        # Env-driven Settings (paths, CORS, limits, thresholds, retention)
    │   ├── logging.py                       # Structured JSON/key-value logging with request_id
    │   └── exceptions.py                    # Typed domain errors & sanitized global exception handlers
    ├── schemas/
    │   ├── __init__.py
    │   ├── common.py
    │   ├── prediction.py
    │   └── history.py
    ├── services/
    │   ├── __init__.py
    │   ├── model_service.py                 # Singleton checkpoint loader & inference engine (reuses Phase 2/3/5)
    │   ├── image_service.py                 # Secure multipart read, decode, temp cleanup, optional retention
    │   ├── uncertainty_service.py           # Entropy, margin, confidence & rejection policy
    │   ├── explanation_service.py           # Lock-protected PyTorch Grad-CAM heatmap & overlay generator
    │   ├── disease_information_service.py   # Structured educational knowledge base (no pesticide dosages)
    │   ├── history_service.py               # Business logic coordinating DB repository & artifact cleanup
    │   └── prediction_service.py            # End-to-end orchestration pipeline
    ├── database/
    │   ├── __init__.py
    │   ├── database.py                      # SQLAlchemy engine, SessionLocal, init_db, health check
    │   ├── models.py                        # PredictionHistory ORM model
    │   └── repository.py                    # CRUD, pagination, status/class filtering
    ├── utils/
    │   ├── __init__.py
    │   ├── image_validation.py              # Magic-byte/Pillow/OpenCV validation, EXIF, size & path safety
    │   └── response_builder.py              # Clean response assembly without leaking internal paths
    └── tests/
        ├── __init__.py
        ├── test_health.py
        ├── test_image_validation.py
        ├── test_prediction.py
        └── test_history.py
```

---

## 4. Step-by-Step Implementation & Verification Plan

1. **Dependencies:** Install `sqlalchemy` and `httpx` in `PlantDiseaseAI/.venv` and write `backend/app/requirements.txt` + `backend/.env.example`.
2. **Core & Database:** Implement `core/{config,logging,exceptions}.py` and `database/{database,models,repository}.py`.
3. **Schemas & Utils:** Implement `schemas/{common,prediction,history}.py` and `utils/{image_validation,response_builder}.py`.
4. **Services:** Implement `model_service.py`, `image_service.py`, `uncertainty_service.py`, `explanation_service.py`, `disease_information_service.py`, `history_service.py`, and `prediction_service.py`.
5. **Routes & App:** Implement `api/routes/{health,system,prediction,history}.py` and `backend/app/main.py` (with rate-limiting middleware, strict dev CORS, request-ID headers, and sanitized errors).
6. **Tests & Docs:** Write comprehensive mocked unit tests + separate real-model/Grad-CAM tests (`backend/app/tests/`), write `docs/phase7_api_documentation.md`, run all backend tests + live Uvicorn verification + Phase 1–5 regression check, and generate `docs/phase7_completion_report.md`.
