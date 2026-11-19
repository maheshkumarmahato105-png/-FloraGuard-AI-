# PlantDiseaseAI — Complete File & Module Index

A single-reference index of every organized file across Phases 1–5.

## 1. Configurations (`ml/configs/`)
| File | Phase | Purpose |
| :--- | :---: | :--- |
| `dataset_config.yaml` | 2 | Dataset paths, image dimensions (`224×224`), validation bounds, split ratios (`70/15/15`), normalization, training-only augmentation |
| `class_mapping.yaml` | 2 | Verified 38-class canonical taxonomy mapping PlantVillage and external PlantDoc labels |
| `model_config.yaml` | 3 | Architecture parameters for `AgriVision-CNN`, `SimpleCNN`, and `ResNet18Baseline` |
| `training_config.yaml` | 4 | Optimizer (`AdamW`), scheduler (`CosineAnnealingLR`), class-weighted `CrossEntropyLoss`, AMP, early stopping, checkpoint policy |
| `experiment_config.yaml` | 4 | Controlled Experiment A (`AgriVision-CNN`), B (`Simple CNN`), and C (`ResNet18` from scratch) definitions |

## 2. Source Code (`ml/src/`)
- **Data Pipeline (`ml/src/*.py`):** `config_loader.py`, `download_dataset.py`, `validate_images.py`, `detect_duplicates.py`, `explore_dataset.py`, `split_dataset.py`, `preprocess.py`, `dataset_statistics.py`
- **Dataset Loader (`ml/src/datasets/`):** `plant_dataset.py`
- **Model Architectures (`ml/src/models/`):** `agrivision_cnn.py`, `simple_cnn.py`, `resnet_baseline.py`, `attention.py`, `residual_block.py`, `model_factory.py`, `checkpoint_utils.py`, plus `ml/src/analyze_model.py`
- **Training Engine (`ml/src/training/`):** `train.py`, `trainer.py`, `engine.py`, `optimizer.py`, `scheduler.py`, `loss.py`, `checkpoint.py`, `metrics.py`, `early_stopping.py`
- **Evaluation Suite (`ml/src/evaluation/`):** `evaluate.py`, `metrics.py`, `confusion_matrix.py`, `classification_report.py`, `calibration.py`, `external_validation.py`, `robustness.py`, `error_analysis.py`, `inference_benchmark.py`, `model_comparison.py`

## 3. Research Reports & Charts (`ml/reports/` & `docs/`)
| Report | Phase | Summary |
| :--- | :---: | :--- |
| `ml/reports/dataset_report.md` | 2 | 56,954 scanned files, 414 duplicate clusters, zero-leakage splits, 4 EDA/split charts |
| `ml/reports/model_architecture_report.md` | 3 | Layer summary, FLOPs/MACs, parameter & size comparison, architecture diagram |
| `ml/reports/training_report.md` | 4 | Configured hyperparameters, RTX 5050 pre-flight VRAM probe (`534.16 MB` at `bs=16`) |
| `ml/reports/phase5_research_report.md` | 5 | Checkpoint verification, real CPU & RTX 5050 latency/throughput benchmarks, PlantDoc coverage |
| `docs/project_cleanup_report.md` | Maint. | Full pre/post cleanup audit (`76` files removed, `~933.55 MB` freed, `37/37` tests verified) |

## 4. Automated Test Suites (`ml/tests/`)
| Test File | Tests | Coverage |
| :--- | :---: | :--- |
| `test_gpu.py` | 1 | PyTorch CUDA detection & GPU matrix multiplication |
| `test_dataset.py` | 4 | Manifests, readability, class mapping consistency, summary statistics |
| `test_preprocessing.py` | 4 | Letterbox aspect-ratio preservation, deterministic eval, train augmentation, processed images |
| `test_splits.py` | 3 | Zero leakage across splits & PlantDoc, 70/15/15 ratios, 38-class coverage |
| `test_model.py` | 12 | CPU & GPU forward/backward/gradients, dynamic classes, submodules, checkpoint round-trip |
| `test_training.py` | 7 | Synthetic loaders, optimizer updates, early stopping, overwrite guard, resume, GPU AMP |
| `test_evaluation.py` | 6 | Checkpoint checks, metrics, missing classes, CM, calibration, PlantDoc audit, robustness, error CSV, benchmarks |
