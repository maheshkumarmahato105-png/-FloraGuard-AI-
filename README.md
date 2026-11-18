# FloraGuard AI — Plant Disease Detection System

> **Workspace Organization:** This workspace contains two cleanly separated layers:
> 1. **Live Web Application (Workspace Root):** Ready-to-run **FloraGuard AI** FastAPI + Tailwind CSS website backed by a trained MobileNetV2 model (`99.57%` test accuracy across 38 PlantVillage classes). Launch anytime with `.\run_website.ps1`.
> 2. **Research ML Engineering Suite ([PlantDiseaseAI/](PlantDiseaseAI/README.md)):** Modular Phases 1–5 research project featuring the custom **AgriVision-CNN** architecture, leak-free PlantVillage + PlantDoc pipeline, training engine, and evaluation suite (`37/37` tests passing). See [PlantDiseaseAI/docs/PROJECT_STRUCTURE.md](PlantDiseaseAI/docs/PROJECT_STRUCTURE.md).

An end-to-end, full-stack AI-powered **Plant Disease Detection System** built with **PyTorch (MobileNetV2 / EfficientNet-B0)**, **OpenCV**, **FastAPI**, **SQLite / PostgreSQL**, and a responsive **Tailwind CSS + ES6 JavaScript** web dashboard.

Supports real-time detection from **laptop/desktop webcams**, **mobile rear/front cameras**, **drag-and-drop uploads**, and **device galleries** across **38 PlantVillage crop-disease classes** (including Tomato, Potato, Apple, Corn, Grape, Pepper, Strawberry, and more).

---

## Features

- **Multi-Modal Image Input**
  - **Live Camera (`getUserMedia`)**: Works on desktop webcams and mobile cameras, defaults to the rear (`environment`) camera on mobile, and supports one-tap camera switching.
  - **Drag-and-Drop & File Upload**: Accepts `.jpg`, `.jpeg`, `.png`, and `.webp` images up to 10 MB with instant preview and client-side mobile image optimization.
  - **Mobile Native Camera Fallback**: Direct `<input capture="environment">` capture for mobile browsers.
- **Computer Vision & AI Pipeline**
  - Strict server-side image validation (MIME type, extension, PIL integrity, EXIF orientation, minimum resolution).
  - OpenCV sharpness (Laplacian variance) and exposure checks, plus adaptive HSV leaf ROI focusing and edge-preserving denoising.
  - Real PyTorch transfer-learning CNN inference (`MobileNetV2` / `EfficientNet-B0`) with ImageNet normalization.
  - **No Fake Predictions**: Low-confidence predictions (<60%) or severely dark/unclear images return an `Uncertain` status with actionable photography guidance.
- **Agronomic Knowledge Base & Persistent History**
  - Full disease profiles covering all 38 PlantVillage classes: description, visible symptoms, possible causes, prevention methods, treatment guidance, and severity level.
  - Persistent history stored in **SQLite** via SQLAlchemy (scalable to **PostgreSQL** via `DATABASE_URL`) with search, status filtering, detail replay, JSON export, and print support.

---

## Project Structure

```text
Plant Disease Detection/
├── frontend/
│   ├── index.html                  # Responsive 5-page SPA dashboard (Home, Detect, Results, History, About)
│   ├── css/styles.css              # Custom animations, scanner overlay, print styles
│   ├── js/
│   │   ├── api.js                  # REST API client
│   │   ├── camera.js               # Desktop/mobile camera controller & image optimizer
│   │   └── app.js                  # Main UI state and workflow controller
│   ├── components/
│   │   ├── navbar.js               # Navigation bar & live backend health pill
│   │   ├── resultCard.js           # Confidence ring, top-3 chart & agronomic report
│   │   └── historyTable.js         # Responsive desktop table & mobile history cards
│   └── assets/logo.svg
│
├── backend/
│   ├── app.py                      # FastAPI application entrypoint (serves API + Frontend)
│   ├── config.py                   # Paths, thresholds, and DB settings
│   ├── database.py                 # SQLAlchemy engine & session management
│   ├── routes/
│   │   ├── health.py               # GET /api/health, /api/plants, /api/evaluation
│   │   ├── upload.py               # POST /api/upload
│   │   ├── predict.py              # POST /api/predict
│   │   └── history.py              # GET /api/history, GET/DELETE /api/history/{id}
│   ├── services/
│   │   ├── image_processor.py      # Validation, OpenCV leaf focus & tensor transform
│   │   ├── model_service.py        # PyTorch model loader & uncertainty-aware inference
│   │   └── disease_knowledge.py    # 38-class agronomic symptom/treatment knowledge base
│   ├── models/
│   │   ├── db_models.py            # SQLAlchemy PredictionRecord ORM model
│   │   ├── schemas.py              # Pydantic API schemas
│   │   └── saved/                  # Trained .pth checkpoint, class_names.json, model_metadata.json
│   └── requirements.txt
│
├── training/
│   ├── download_dataset.py         # Automated PlantVillage dataset downloader
│   ├── preprocess.py               # Image validation, leak-free train/val/test split & DataLoaders
│   ├── train.py                    # Two-stage MobileNetV2 / EfficientNet-B0 transfer learning
│   ├── evaluate.py                 # Held-out test evaluation & confusion matrix generator
│   └── reports/                    # evaluation_report.md, evaluation_metrics.json, plots
│
├── tests/test_api.py               # End-to-end FastAPI test suite
├── dataset/                        # PlantVillage raw images and split manifests
├── uploads/                        # Saved uploaded/captured leaf images
├── database/                       # plant_disease.db (SQLite)
├── README.md
└── .gitignore
```

---

## Quick Start (Local Setup)

### 1. Environment & Dependencies

Create a Python 3.12 virtual environment and install dependencies:

```powershell
# Using uv (recommended)
uv venv --python 3.12 .venv

# Install PyTorch (CUDA 12.8 for NVIDIA GPUs, or drop --index-url for CPU)
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install remaining backend and ML dependencies
uv pip install --python .venv\Scripts\python.exe -r backend/requirements.txt
```

Or with standard `venv` + `pip`:
```powershell
python -m venv .venv
.venv\Scripts\pip.exe install -r backend/requirements.txt
```

### 2. Download Dataset, Preprocess, Train & Evaluate Model

```powershell
# Step 1: Download the 38-class PlantVillage dataset (color images)
.venv\Scripts\python.exe training/download_dataset.py

# Step 2: Validate images and create stratified 70/15/15 Train/Val/Test splits
.venv\Scripts\python.exe training/preprocess.py

# Step 3: Train MobileNetV2 via 2-stage transfer learning (uses GPU automatically if available)
.venv\Scripts\python.exe training/train.py --arch mobilenet_v2 --warmup-epochs 2 --finetune-epochs 6 --batch-size 64

# Step 4: Evaluate on the held-out test set and generate reports & confusion matrix
.venv\Scripts\python.exe training/evaluate.py
```

### 3. Run the Application

```powershell
.venv\Scripts\uvicorn.exe backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at:
- **Web Dashboard**: `http://127.0.0.1:8000`
- **Interactive API Docs (Swagger UI)**: `http://127.0.0.1:8000/docs`

> **Mobile Camera Note**: Modern mobile browsers allow `getUserMedia` live camera access on `localhost` or over **HTTPS**. On a local LAN IP over HTTP, the **Mobile Camera Capture** button (`capture="environment"`) works natively as a seamless fallback, or serve behind HTTPS/tunnel for WebRTC live preview.

### 4. Scaling from SQLite to PostgreSQL

The backend defaults to `database/plant_disease.db`. To switch to PostgreSQL:
1. Install `psycopg2-binary`.
2. Set `DATABASE_URL` before starting Uvicorn:
   ```powershell
   $env:DATABASE_URL = "postgresql+psycopg2://user:password@localhost:5432/plant_disease"
   .venv\Scripts\uvicorn.exe backend.app:app --host 0.0.0.0 --port 8000
   ```

---

## REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Backend, database, and trained model availability + evaluation summary |
| `GET` | `/api/plants` | Supported crops and disease classes loaded from the trained model |
| `GET` | `/api/evaluation` | Real held-out test evaluation metrics and plot URLs |
| `POST` | `/api/upload` | Securely validate, inspect, and store an uploaded image |
| `POST` | `/api/predict` | Run OpenCV preprocessing + PyTorch inference and save result to DB |
| `GET` | `/api/history` | List past detection records (filterable by `plant`, `status`, `search`) |
| `GET` | `/api/history/{id}` | Retrieve full details for a single detection record |
| `DELETE` | `/api/history/{id}` | Delete a detection record and clean up its unreferenced image |

---

## Verified Model Evaluation Report

Trained and evaluated on the full **38-class PlantVillage (Color)** dataset (**54,381 images**: 38,065 train / 8,158 validation / 8,158 held-out test) using **MobileNetV2** two-stage transfer learning (2 warmup epochs + 6 fine-tuning epochs, mixed-precision AMP on NVIDIA GeForce RTX 5050 Laptop GPU):

| Metric | Held-Out Test Set Score |
| :--- | ---: |
| **Test Accuracy** | **99.57%** |
| **Top-3 Accuracy** | **100.00%** |
| **Macro Precision** | **99.39%** |
| **Macro Recall** | **99.57%** |
| **Macro F1-Score** | **99.47%** |
| **Weighted F1-Score** | **99.57%** |

Full per-class metrics and visualizations are generated in:
- `training/reports/evaluation_report.md`
- `training/reports/evaluation_metrics.json`
- `training/reports/confusion_matrix.png`
- `training/reports/training_curves.png`

---

## Automated Tests

```powershell
.venv\Scripts\pytest.exe tests/test_api.py -v
```
#   - F l o r a G u a r d - A I - P l a n t - D i s e a s e - D e t e c t i o n - S y s t e m  
 