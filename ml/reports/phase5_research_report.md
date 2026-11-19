# PlantDiseaseAI — Phase 5 Research Evaluation Report

- **Generated At (UTC):** `2026-11-18T06:09:03.237625+00:00`
- **Evaluation Status:** **`EVALUATION_PIPELINE_READY_CHECKPOINTS_MISSING`**
- **Scientific Integrity Guarantee:** Zero metrics have been fabricated. Any model without a verified trained Phase 4 checkpoint is explicitly reported as missing.

---

## 1. Research Objective

Rigorously evaluate and compare three custom/baseline convolutional architectures (**Simple CNN**, **AgriVision-CNN**, and **ResNet18 from scratch**) for multi-species plant leaf disease classification across:
1. In-domain generalization on the untouched Phase 2 **PlantVillage** held-out test split.
2. Out-of-domain real-world field generalization on the separate **PlantDoc** external dataset.
3. Perturbation robustness under realistic camera and lighting degradations.
4. Confidence calibration via validation-fitted temperature scaling (ECE & Brier score).
5. Fine-grained error analysis and CPU/GPU hardware inference efficiency.

---

## 2. Dataset Information (Measured from Phase 2 Manifest)

- **Internal Training Split (PlantVillage):** `37,844` images *(used in Phase 4 training only)*
- **Internal Validation Split (PlantVillage):** `8,127` images *(used for model selection & temperature scaling only)*
- **Untouched Internal Test Split (PlantVillage):** `8,123` images across `38` verified canonical classes
- **External Real-World Evaluation Dataset (PlantDoc):** `2,554` total valid unique images
  - Verified mapped images: **`2,554`** (`100.0%`) across `28` mapped classes
  - Unmatched/excluded external images: **`0`**
  - Duplicate-group overlap with PlantVillage training split: **`0`** (Independent: **`True`**)

---

## 3. Models Evaluated & Checkpoint Integrity Verification

| Architecture | Expected Checkpoint Path | Exists | Valid Trained Checkpoint | Epoch | Status / Notes |
| :--- | :--- | :---: | :---: | ---: | :--- |
| `simple_cnn` | `C:\Users\Mahesh Kumar Mahato\Desktop\Plant Disease Detection\PlantDiseaseAI\ml\experiments\checkpoints\exp_b_simple_cnn\best_model.pth` | No | **No** | — | Trained checkpoint file not found (Phase 4 training has not been executed for this model). |
| `agrivision_cnn` | `C:\Users\Mahesh Kumar Mahato\Desktop\Plant Disease Detection\PlantDiseaseAI\ml\experiments\checkpoints\exp_a_agrivision_cnn\best_model.pth` | No | **No** | — | Trained checkpoint file not found (Phase 4 training has not been executed for this model). |
| `resnet18_baseline` | `C:\Users\Mahesh Kumar Mahato\Desktop\Plant Disease Detection\PlantDiseaseAI\ml\experiments\checkpoints\exp_c_resnet18_scratch\best_model.pth` | No | **No** | — | Trained checkpoint file not found (Phase 4 training has not been executed for this model). |

---

## 4. Experimental Methodology

- **Deterministic Evaluation:** `model.eval()`, `torch.no_grad()`, zero test-time random augmentation, identical `224×224` aspect-ratio-preserving letterbox RGB preprocessing and ImageNet normalization as Phase 2/4.
- **Leakage Isolation:** Test split is never used for training, hyperparameter tuning, or calibration fitting. Temperature scaling is fitted **exclusively on validation logits** via LBFGS NLL minimization.
- **External Validation Hygiene:** PlantDoc uses verified mappings from `ml/configs/class_mapping.yaml`, preserving original labels and never assuming field accuracy from lab images.
- **Non-Destructive Robustness Testing:** All 10 realistic image perturbations (brightness, contrast, Gaussian blur, Gaussian noise, JPEG Q=30, ±12° rotation, warm/cool lighting tint) are applied in memory only.

---

## 5. Evaluation Metrics (Internal Held-Out PlantVillage Test Set)

> **Not Yet Measured (Checkpoints Missing):** No trained Phase 4 checkpoints (`best_model.pth`) were found in `ml/experiments/checkpoints/`. Run Phase 4 training first, then re-run `python ml/src/evaluation/evaluate.py` to populate verified internal test metrics.

---

## 6. Confusion Matrices

> **Pending Trained Checkpoints:** `ml/src/evaluation/confusion_matrix.py` is implemented and verified via automated tests; real heatmaps (`confusion_matrix.png` and `normalized_confusion_matrix.png`) will be generated automatically once Phase 4 checkpoints exist.

---

## 7. Model Comparison Table (Measured Hardware + Evaluation Metrics)

| Architecture | Total Params | Trainable Params | FP32 Size (MB) | CPU Latency bs=1 (ms) | GPU Latency bs=1 FP32 (ms) | GPU Throughput bs=16 FP16 (img/s) | Peak GPU VRAM bs=16 (MB) | Test Accuracy | Test Macro-F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `simple_cnn` | 398,662 | 398,662 | 1.52 | 8.39 | 0.56 | 3602.6 | 131.1 | *Pending Phase 4* | *Pending Phase 4* |
| `agrivision_cnn` | 2,871,350 | 2,871,350 | 10.97 | 11.43 | 2.61 | 2333.2 | 139.6 | *Pending Phase 4* | *Pending Phase 4* |
| `resnet18_baseline` | 11,196,006 | 11,196,006 | 42.75 | 13.19 | 2.82 | 2532.6 | 121.1 | *Pending Phase 4* | *Pending Phase 4* |

Comparison CSV: `C:\Users\Mahesh Kumar Mahato\Desktop\Plant Disease Detection\PlantDiseaseAI\ml\experiments\evaluation\metrics\model_comparison.csv` | Efficiency Chart: `C:\Users\Mahesh Kumar Mahato\Desktop\Plant Disease Detection\PlantDiseaseAI\ml\experiments\evaluation\charts\model_comparison_efficiency.png`

> **Selection Policy:** No single winner is declared based on accuracy alone; final deployment selection must weigh Macro-F1, external field generalization (PlantDoc), calibration error (ECE), robustness degradation, and latency/memory constraints.

---

## 8. External Validation Results (PlantDoc — Real-World Field Images)

> **Coverage Verified, Inference Pending Trained Checkpoints:** Verified `2,554` mapped PlantDoc images (`0` unmatched, `0` training overlaps). Real external generalization scores will populate once Phase 4 checkpoints are trained.

---

## 9. Robustness Testing Results (In-Memory Perturbations)

> **Pending Trained Checkpoints:** 10 isolated perturbations (`brightness_dark_0.65`, `brightness_bright_1.35`, `contrast_low_0.65`, `contrast_high_1.35`, `mild_blur_r1.5`, `gaussian_noise_std14`, `jpeg_compression_q30`, `small_rotation_12deg`, `warm_sunlight_tint`, `cool_shade_tint`) are implemented and unit-tested in `ml/src/evaluation/robustness.py`.

---

## 10. Confidence Calibration Results (Validation-Fitted Temperature Scaling)

> **Pending Trained Checkpoints:** ECE, multiclass Brier score, validation-only LBFGS temperature scaling, reliability diagrams, and confidence histograms are implemented and verified via automated tests in `ml/src/evaluation/calibration.py`.

---

## 11. Error Analysis

> **Pending Trained Checkpoints:** Prediction/error CSV export (`image_path`, `true_label`, `predicted_label`, `confidence`, `is_correct`, `dataset_source`), per-species difficulty ranking, false-positive/false-negative counters, and misclassification gallery generation are ready in `ml/src/evaluation/error_analysis.py`.

---

## 12. Inference Benchmarks (Measured Real Hardware Performance)

Measured with warm-up runs and explicit `torch.cuda.synchronize()` barriers on `224 × 224 × 3` inputs:

| Architecture | Condition | Hardware | Precision | Batch Size | Mean (ms) | Median (ms) | P95 (ms) | Throughput (img/s) | Peak VRAM (MB) |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `simple_cnn` | `cpu_bs1_fp32` | Intel64 Family 6 Model 183 Stepping 1, GenuineIntel | `fp32` | 1 | 8.39 | 8.37 | 9.88 | 119.2 | — |
| `simple_cnn` | `gpu_bs1_fp32` | NVIDIA GeForce RTX 5050 Laptop GPU | `fp32` | 1 | 0.56 | 0.53 | 0.60 | 1790.0 | 23.48 |
| `simple_cnn` | `gpu_bs16_fp32` | NVIDIA GeForce RTX 5050 Laptop GPU | `fp32` | 16 | 7.11 | 7.11 | 7.16 | 2249.5 | 215.84 |
| `simple_cnn` | `gpu_bs16_fp16` | NVIDIA GeForce RTX 5050 Laptop GPU | `amp_fp16` | 16 | 4.44 | 4.43 | 4.48 | 3602.6 | 131.09 |
| `agrivision_cnn` | `cpu_bs1_fp32` | Intel64 Family 6 Model 183 Stepping 1, GenuineIntel | `fp32` | 1 | 11.43 | 11.74 | 12.56 | 87.5 | — |
| `agrivision_cnn` | `gpu_bs1_fp32` | NVIDIA GeForce RTX 5050 Laptop GPU | `fp32` | 1 | 2.61 | 2.54 | 2.96 | 383.4 | 35.41 |
| `agrivision_cnn` | `gpu_bs16_fp32` | NVIDIA GeForce RTX 5050 Laptop GPU | `fp32` | 16 | 10.65 | 10.48 | 11.26 | 1501.7 | 252.80 |
| `agrivision_cnn` | `gpu_bs16_fp16` | NVIDIA GeForce RTX 5050 Laptop GPU | `amp_fp16` | 16 | 6.86 | 6.79 | 7.51 | 2333.2 | 139.63 |
| `resnet18_baseline` | `cpu_bs1_fp32` | Intel64 Family 6 Model 183 Stepping 1, GenuineIntel | `fp32` | 1 | 13.19 | 13.21 | 14.36 | 75.8 | — |
| `resnet18_baseline` | `gpu_bs1_fp32` | NVIDIA GeForce RTX 5050 Laptop GPU | `fp32` | 1 | 2.82 | 2.45 | 4.35 | 354.2 | 68.38 |
| `resnet18_baseline` | `gpu_bs16_fp32` | NVIDIA GeForce RTX 5050 Laptop GPU | `fp32` | 16 | 8.76 | 8.73 | 9.00 | 1825.7 | 169.80 |
| `resnet18_baseline` | `gpu_bs16_fp16` | NVIDIA GeForce RTX 5050 Laptop GPU | `amp_fp16` | 16 | 6.32 | 6.08 | 7.26 | 2532.6 | 121.09 |

---

## 13. Limitations

1. **Missing Phase 4 Trained Checkpoints (Current State):** Because Phase 4 halted before starting lengthy GPU training pending user approval, classification accuracy, F1, confusion matrices, calibration curves, and error galleries cannot be measured yet and are intentionally left unpopulated.
2. **Lab-to-Field Domain Gap:** PlantVillage test performance measures controlled studio-leaf recognition only and must NEVER be extrapolated as real-world field accuracy without PlantDoc external validation.
3. **Agricultural Diagnostic Disclaimer:** Model outputs are decision-support research metrics and cannot claim clinical or agricultural diagnostic reliability without multi-season field validation across diverse cultivars and geographies.
4. **External Class Coverage:** PlantDoc covers 28 of PlantVillage's 38 canonical classes; 10 PlantVillage classes lack external PlantDoc field samples.

---

## 14. Research Conclusions & Next Steps

- **Completed & Verified in Phase 5:** The complete, leakage-safe evaluation suite (metrics, confusion matrices, classification reports, validation-fitted temperature scaling, PlantDoc external validation, 10-perturbation robustness testing, error analysis, multi-model comparison, and real CPU/GPU inference benchmarks) is fully implemented and unit-tested.
- **Measured Hardware Trade-Offs:** Real CPU and RTX 5050 GPU benchmarks above quantify the exact latency, throughput, parameter, and VRAM trade-offs among `simple_cnn`, `agrivision_cnn`, and `resnet18_baseline`.
- **To Populate Full Trained Model Results:** Run Phase 4 training for Experiments A, B, and C, then re-execute:
  ```powershell
  python ml/src/evaluation/evaluate.py
  ```
