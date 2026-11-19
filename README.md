# PlantDiseaseAI — Intelligent Plant Disease Detection System

**PlantDiseaseAI** is a research-grade, modular deep learning system for plant leaf disease classification using a custom PyTorch architecture (**AgriVision-CNN**), leak-free multi-dataset preprocessing (**PlantVillage** + external **PlantDoc** + **PlantSeg** roadmap), reproducible training, and rigorous evaluation.

> **Completed Milestones:** Phases 1–5 (Environment Setup, Dataset Pipeline, Custom `AgriVision-CNN` Architecture, Training & Optimization Engine, Evaluation & Hardware Benchmarking). Full multi-epoch GPU training and Phase 6+ (Explainable AI / Grad-CAM / Backend / React Frontend) await user approval.

---

## Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Runtime** | Python 3.11.18 (isolated `.venv` virtual environment) |
| **Deep Learning** | PyTorch `2.11.0+cu128`, Torchvision `0.26.0+cu128` (CUDA 12.8 on RTX 5050) |
| **Computer Vision** | OpenCV (`opencv-python`), Pillow (`PIL`) |
| **Data Science & Tracking** | NumPy, Pandas, scikit-learn, Matplotlib, PyYAML, tqdm, TensorBoard |
| **Backend API (Reserved)** | FastAPI, Uvicorn, `python-multipart`, `python-dotenv` |
| **Frontend (Reserved)** | React.js, JavaScript (Node.js `v24.19.0`) |
| **Testing & VCS** | `pytest` (`37/37` tests passing), Git |

---

## Organized Project Structure

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for the complete file-by-file directory index.

```text
PlantDiseaseAI/
├── ml/
│   ├── configs/            # dataset_config, class_mapping, model_config, training_config, experiment_config
│   ├── data/
│   │   ├── raw/            # plantvillage (54,381), plantdoc (2,573), plantseg (manual instructions)
│   │   ├── interim/        # Non-destructive quarantine folder (2 flagged PlantDoc files)
│   │   ├── processed/      # Deterministic 224x224 RGB letterbox splits: train/validation/test/external_plantdoc
│   │   └── metadata/       # image_manifest, split_manifest, class_distribution, cleaning/duplicate/summary JSONs
│   ├── src/
│   │   ├── datasets/       # plant_dataset.py (Phase 2 manifest DataLoaders)
│   │   ├── models/         # agrivision_cnn, simple_cnn, resnet_baseline, attention, residual_block, model_factory, checkpoint_utils
│   │   ├── training/       # train, trainer, engine, optimizer, scheduler, loss, checkpoint, metrics, early_stopping
│   │   ├── evaluation/     # evaluate, metrics, confusion_matrix, classification_report, calibration, external_validation, robustness, error_analysis, inference_benchmark, model_comparison
│   │   └── *.py            # Phase 2–3 pipeline scripts (download, validate, detect_duplicates, explore, split, preprocess, dataset_statistics, analyze_model)
│   ├── experiments/        # logs/, checkpoints/, results/, evaluation/{metrics,charts,predictions,reports}/
│   ├── reports/            # dataset_report.md, model_architecture_report.md, training_report.md, phase5_research_report.md & charts/
│   └── tests/              # 7 test suites (37 passing tests)
├── backend/app/            # Reserved FastAPI structure (api, core, services, models)
├── frontend/               # Reserved React.js frontend directory
├── artifacts/              # Reserved exported model artifacts
├── notebooks/              # Reserved EDA & research notebooks
├── docs/                   # PROJECT_STRUCTURE.md, project_cleanup_report.md
├── .venv/                  # Python 3.11.18 virtual environment (Git-ignored)
├── requirements.txt        # Direct dependencies
├── requirements.lock.txt   # Exact pinned lockfile
├── .gitignore
└── README.md
```

---

## Quick Commands (Windows PowerShell)

Activate the Python 3.11 virtual environment from inside `PlantDiseaseAI/`:

```powershell
cd PlantDiseaseAI
.\.venv\Scripts\Activate.ps1
```

### Phase 1 — Verify GPU & Environment
```powershell
python ml/tests/test_gpu.py
```

### Phase 2 — Dataset Pipeline (Already Executed)
```powershell
python ml/src/download_dataset.py
python ml/src/validate_images.py
python ml/src/detect_duplicates.py
python ml/src/explore_dataset.py
python ml/src/split_dataset.py
python ml/src/preprocess.py
python ml/src/dataset_statistics.py
```

### Phase 3 — Architecture Profiling & Diagram
```powershell
python ml/src/analyze_model.py
```

### Phase 4 — Pre-Flight Check & Controlled Training
```powershell
# Safe single-batch real-dataset & GPU VRAM dry-run
python ml/src/training/train.py --experiment experiment_a_agrivision --dry-run

# Full training runs (when ready)
python ml/src/training/train.py --experiment experiment_a_agrivision
python ml/src/training/train.py --experiment experiment_b_simple_cnn
python ml/src/training/train.py --experiment experiment_c_resnet18

# TensorBoard live monitoring
tensorboard --logdir ml/experiments/logs
```

### Phase 5 — Evaluation & Benchmarking
```powershell
python ml/src/evaluation/evaluate.py
```

### Run Complete Automated Test Suite (37 Tests)
```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
pytest ml/tests -p no:cacheprovider -v
```
