# PlantDiseaseAI — Phase 4 Model Training Report

- **Generated At (UTC):** `2026-11-18T17:41:44.986167+00:00`
- **Run ID / Experiment:** `exp_a_agrivision_cnn`
- **Pipeline Status:** **`training_completed`**

---

## 1. Model & Dataset Configuration

- **Architecture Used:** `agrivision_cnn` (`2,871,350` total parameters)
- **Dataset Version:** `PlantVillage-Color-38-Classes (Phase 2 split_manifest.csv)`
- **Training Images (Measured from Phase 2 manifest):** `37,844`
- **Validation Images (Measured from Phase 2 manifest):** `8,127`
- **Held-Out Test Images (Strictly Untouched during Phase 4):** `8,123`
- **External Evaluation Set (PlantDoc — Strictly Untouched):** `2,554`
- **Number of Verified Classes:** `38`

---

## 2. Configured Hyperparameters (Targets)

| Parameter | Configured Value |
| :--- | :--- |
| Optimizer | `adamw` |
| Initial Learning Rate | `0.001` |
| Weight Decay | `0.0001` |
| LR Scheduler | `cosine_annealing` |
| Loss Function | `cross_entropy` |
| Class Weighting | `inverse_sqrt` |
| Label Smoothing | `0.0` |
| Batch Size (Train / Eval) | `16` / `32` |
| Max Epochs | `30` |
| Early Stopping | Monitor `val_macro_f1`, patience `8` |
| Mixed Precision (AMP) | `True` |
| Gradient Clipping (`max_norm`) | `1.0` |
| Random Seed | `42` |

---

## 3. Hardware & GPU Telemetry (Measured)

- **Execution Device:** `cuda:0`
- **CUDA Available:** `True`
- **GPU Name:** `NVIDIA GeForce RTX 5050 Laptop GPU`
- **Compute Capability:** `sm_120 (12.0)`
- **Total VRAM:** `7.93 GB`
- **PyTorch / CUDA Build:** `2.11.0+cu128` / `12.8`

---

## 4. Actual Training Results

| Metric | Measured Result |
| :--- | ---: |
| **Epochs Completed** | `30` / `30` |
| **Best Validation Macro-F1** | **99.61%** (Epoch `26`) |
| **Best Validation Accuracy** | **99.67%** |
| **Total Training Duration** | `2225.98 s` (`37.10 min`) |
| **Early Stopping Status** | `Not triggered` |
| **Overfitting Diagnostic** | No severe overfitting signal (train-val acc gap=-0.09%, val-train loss gap=0.0024). |

### Saved Checkpoints & Artifacts
- Best Model Checkpoint: `C:\Users\Mahesh Kumar Mahato\Desktop\Plant Disease Detection\PlantDiseaseAI\ml\experiments\checkpoints\exp_a_agrivision_cnn\best_model.pth`
- Latest Checkpoint: `C:\Users\Mahesh Kumar Mahato\Desktop\Plant Disease Detection\PlantDiseaseAI\ml\experiments\checkpoints\exp_a_agrivision_cnn\latest_checkpoint.pth`
- Epoch Metrics CSV: `C:\Users\Mahesh Kumar Mahato\Desktop\Plant Disease Detection\PlantDiseaseAI\ml\experiments\results\exp_a_agrivision_cnn\metrics.csv`
- TensorBoard Log Dir: `C:\Users\Mahesh Kumar Mahato\Desktop\Plant Disease Detection\PlantDiseaseAI\ml\experiments\logs\exp_a_agrivision_cnn`
- Training Curves: `C:\Users\Mahesh Kumar Mahato\Desktop\Plant Disease Detection\PlantDiseaseAI\ml\experiments\results\exp_a_agrivision_cnn\charts\exp_a_agrivision_cnn_curves.png`

---

## 5. Planned Controlled Experiments (`ml/configs/experiment_config.yaml`)

| Experiment | Architecture | Initialization | Selection Metric |
| :--- | :--- | :--- | :--- |
| **Experiment A** | `agrivision_cnn` | From scratch (seed 42) | `val_macro_f1` |
| **Experiment B** | `simple_cnn` | From scratch (seed 42) | `val_macro_f1` |
| **Experiment C** | `resnet18_baseline` | From scratch (`weights=None`, seed 42) | `val_macro_f1` |

---

## 6. Training Limitations & Notes

1. **Class Imbalance (36.23× in PlantVillage):** Addressed via bounded `inverse_sqrt` class weights (`[0.5, 3.0]`) inside `CrossEntropyLoss` without stacking a second conflicting sampler.
2. **Hardware Budget (8 GB Laptop GPU):** Initial batch size is conservatively set to `16` with AMP enabled; automatic CUDA OOM handling saves an emergency checkpoint and suggests reducing batch size.
3. **In-Domain vs. Real-World Domain Shift:** Validation metrics during Phase 4 reflect PlantVillage in-domain performance only; real-world generalization on `external_plantdoc` and held-out `test` evaluation are reserved for **Phase 5**.
