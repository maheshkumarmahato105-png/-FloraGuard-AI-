# PlantDiseaseAI — Phase 3 Model Architecture & Computational Analysis Report

- **Generated At (UTC):** `2026-11-18T05:57:19.761707+00:00`
- **Input Tensor Shape:** `(B, 3, 224, 224)` (profiled at batch size `1` for MACs/FLOPs)
- **Dynamic Output Classes:** `38` (resolved automatically from verified Phase 2 `ml/configs/class_mapping.yaml`)
- **Random Seed:** `42`
- **Training Status:** **Phase 3 Architecture Development Only — No model training has been performed, and no accuracy claims are made.**

---

## 1. Proposed Architecture: AgriVision-CNN

Visual diagram: `ml/reports/charts/agrivision_cnn_architecture.png`

AgriVision-CNN is a custom, mobile-conscious multi-scale residual convolutional architecture tailored for plant leaf pathology:

1. **Fine-Grained Initial Stem (`3×3`, 32 channels, stride 2, BN, ReLU):** Uses a `3×3` kernel instead of ResNet's aggressive `7×7` + MaxPool stem to preserve small foliar lesions, pinpoint fungal pycnidia, and early necrotic edges.
2. **Multi-Scale Feature Extraction Block (Parallel `3×3` & `5×5` Branches → Concat → BN → ReLU → `1×1` Projection):** Simultaneously captures fine disease spots (`3×3`) and wider yellow chlorosis halos / blight blotches (`5×5`), fusing cross-scale maps into 64 channels.
3. **Residual Learning Stages (`64 → 128 → 256` channels, 2 blocks/stage):** Deep residual blocks with `1×1` projection shortcuts whenever channel depth or spatial resolution changes, preventing vanishing gradients during training from scratch.
4. **Squeeze-and-Excitation (SE) Channel Attention (`r=8`):** Lightweight GAP → `1×1` Conv → ReLU → `1×1` Conv → Sigmoid channel reweighting embedded inside every residual block to emphasize disease-discriminative color/texture channels while suppressing background clutter (important for PlantDoc generalization).
5. **Classification Head (`GlobalAvgPool2d → Dropout(0.30) → Linear`):** Resolution-flexible pooling returning **raw logits** (no softmax in `forward()`), keeping memory footprint low for future mobile deployment.

---

## 2. Three-Architecture Comparison (Measured Structural & Compute Metrics)

| Architecture | Total Params | Trainable Params | FP32 Size (MB) | FP16 Size (MB) | GMACs (1×3×224×224) | GFLOPs (2×MACs) | Pretrained Weights |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Simple CNN (`simple_cnn`)** | 398,662 | 398,662 | 1.52 | 0.76 | 0.740 | 1.480 | None (from scratch) |
| **AgriVision-CNN (`agrivision_cnn`)** | 2,871,350 | 2,871,350 | 10.97 | 5.49 | 1.800 | 3.600 | None (from scratch) |
| **ResNet18 Baseline (`resnet18_baseline`)** | 11,196,006 | 11,196,006 | 42.75 | 21.37 | 1.816 | 3.632 | None (from scratch) |

### Architectural Differences Summary

| Dimension | Simple CNN | AgriVision-CNN (Proposed) | ResNet18 Baseline |
| :--- | :--- | :--- | :--- |
| **Stem** | `3×3` Conv (32ch) + MaxPool | `3×3` Conv (32ch, s=2) | `7×7` Conv (64ch, s=2) + MaxPool |
| **Multi-Scale Feature Block** | No | **Yes (Parallel `3×3` + `5×5` + `1×1` proj)** | No |
| **Residual Connections** | No | **Yes (`64 → 128 → 256` with proj shortcuts)** | Yes (`64 → 128 → 256 → 512`) |
| **Channel Attention (SE)** | No | **Yes (SE in every residual block, `r=8`)** | No |
| **Parameter Efficiency vs. ResNet18** | `3.6%` of ResNet18 params | **`25.6%`** of ResNet18 params (**3.90× smaller**) | `100.0%` (reference) |
| **Accuracy / F1** | *Pending Phase 4 training* | *Pending Phase 4 training* | *Pending Phase 4 training* |

> **Note on Empirical Performance:** No accuracy, precision, recall, or F1 claims are made in Phase 3. Controlled apples-to-apples training and evaluation on PlantVillage (`train`/`validation`/`test`) and external real-world `external_plantdoc` will be conducted in **Phase 4**.

---

## 3. AgriVision-CNN Layer-by-Layer Summary (`1 × 3 × 224 × 224`)

- **Input Tensor:** `1×3×224×224`
- **Output Logits Tensor:** `1×38`
- **Profiled Layers:** `58`

| Layer Name | Layer Type | Input Shape | Output Shape | Parameters | MACs |
| :--- | :--- | :--- | :--- | ---: | ---: |
| `stem.0` | `Conv2d` | `1×3×224×224` | `1×32×112×112` | 864 | 10,838,016 |
| `stem.1` | `BatchNorm2d` | `1×32×112×112` | `1×32×112×112` | 64 | 401,408 |
| `multiscale.branch_3x3` | `Conv2d` | `1×32×112×112` | `1×32×112×112` | 9,216 | 115,605,504 |
| `multiscale.branch_5x5` | `Conv2d` | `1×32×112×112` | `1×32×112×112` | 25,600 | 321,126,400 |
| `multiscale.concat_bn` | `BatchNorm2d` | `1×64×112×112` | `1×64×112×112` | 128 | 802,816 |
| `multiscale.projection.0` | `Conv2d` | `1×64×112×112` | `1×64×112×112` | 4,096 | 51,380,224 |
| `multiscale.projection.1` | `BatchNorm2d` | `1×64×112×112` | `1×64×112×112` | 128 | 802,816 |
| `stages.0.0.shortcut.0` | `Conv2d` | `1×64×112×112` | `1×64×56×56` | 4,096 | 12,845,056 |
| `stages.0.0.shortcut.1` | `BatchNorm2d` | `1×64×56×56` | `1×64×56×56` | 128 | 200,704 |
| `stages.0.0.conv1` | `Conv2d` | `1×64×112×112` | `1×64×56×56` | 36,864 | 115,605,504 |
| `stages.0.0.bn1` | `BatchNorm2d` | `1×64×56×56` | `1×64×56×56` | 128 | 200,704 |
| `stages.0.0.conv2` | `Conv2d` | `1×64×56×56` | `1×64×56×56` | 36,864 | 115,605,504 |
| `stages.0.0.bn2` | `BatchNorm2d` | `1×64×56×56` | `1×64×56×56` | 128 | 200,704 |
| `stages.0.0.attention.squeeze` | `AdaptiveAvgPool2d` | `1×64×56×56` | `1×64×1×1` | 0 | 0 |
| `stages.0.0.attention.excitation.0` | `Conv2d` | `1×64×1×1` | `1×8×1×1` | 520 | 512 |
| `stages.0.0.attention.excitation.2` | `Conv2d` | `1×8×1×1` | `1×64×1×1` | 576 | 512 |
| `stages.0.1.conv1` | `Conv2d` | `1×64×56×56` | `1×64×56×56` | 36,864 | 115,605,504 |
| `stages.0.1.bn1` | `BatchNorm2d` | `1×64×56×56` | `1×64×56×56` | 128 | 200,704 |
| `stages.0.1.conv2` | `Conv2d` | `1×64×56×56` | `1×64×56×56` | 36,864 | 115,605,504 |
| `stages.0.1.bn2` | `BatchNorm2d` | `1×64×56×56` | `1×64×56×56` | 128 | 200,704 |
| `stages.0.1.attention.squeeze` | `AdaptiveAvgPool2d` | `1×64×56×56` | `1×64×1×1` | 0 | 0 |
| `stages.0.1.attention.excitation.0` | `Conv2d` | `1×64×1×1` | `1×8×1×1` | 520 | 512 |
| `stages.0.1.attention.excitation.2` | `Conv2d` | `1×8×1×1` | `1×64×1×1` | 576 | 512 |
| `stages.1.0.shortcut.0` | `Conv2d` | `1×64×56×56` | `1×128×28×28` | 8,192 | 6,422,528 |
| `stages.1.0.shortcut.1` | `BatchNorm2d` | `1×128×28×28` | `1×128×28×28` | 256 | 100,352 |
| `stages.1.0.conv1` | `Conv2d` | `1×64×56×56` | `1×128×28×28` | 73,728 | 57,802,752 |
| `stages.1.0.bn1` | `BatchNorm2d` | `1×128×28×28` | `1×128×28×28` | 256 | 100,352 |
| `stages.1.0.conv2` | `Conv2d` | `1×128×28×28` | `1×128×28×28` | 147,456 | 115,605,504 |
| `stages.1.0.bn2` | `BatchNorm2d` | `1×128×28×28` | `1×128×28×28` | 256 | 100,352 |
| `stages.1.0.attention.squeeze` | `AdaptiveAvgPool2d` | `1×128×28×28` | `1×128×1×1` | 0 | 0 |
| `stages.1.0.attention.excitation.0` | `Conv2d` | `1×128×1×1` | `1×16×1×1` | 2,064 | 2,048 |
| `stages.1.0.attention.excitation.2` | `Conv2d` | `1×16×1×1` | `1×128×1×1` | 2,176 | 2,048 |
| `stages.1.1.conv1` | `Conv2d` | `1×128×28×28` | `1×128×28×28` | 147,456 | 115,605,504 |
| `stages.1.1.bn1` | `BatchNorm2d` | `1×128×28×28` | `1×128×28×28` | 256 | 100,352 |
| `stages.1.1.conv2` | `Conv2d` | `1×128×28×28` | `1×128×28×28` | 147,456 | 115,605,504 |
| `stages.1.1.bn2` | `BatchNorm2d` | `1×128×28×28` | `1×128×28×28` | 256 | 100,352 |
| `stages.1.1.attention.squeeze` | `AdaptiveAvgPool2d` | `1×128×28×28` | `1×128×1×1` | 0 | 0 |
| `stages.1.1.attention.excitation.0` | `Conv2d` | `1×128×1×1` | `1×16×1×1` | 2,064 | 2,048 |
| `stages.1.1.attention.excitation.2` | `Conv2d` | `1×16×1×1` | `1×128×1×1` | 2,176 | 2,048 |
| `stages.2.0.shortcut.0` | `Conv2d` | `1×128×28×28` | `1×256×14×14` | 32,768 | 6,422,528 |
| `stages.2.0.shortcut.1` | `BatchNorm2d` | `1×256×14×14` | `1×256×14×14` | 512 | 50,176 |
| `stages.2.0.conv1` | `Conv2d` | `1×128×28×28` | `1×256×14×14` | 294,912 | 57,802,752 |
| `stages.2.0.bn1` | `BatchNorm2d` | `1×256×14×14` | `1×256×14×14` | 512 | 50,176 |
| `stages.2.0.conv2` | `Conv2d` | `1×256×14×14` | `1×256×14×14` | 589,824 | 115,605,504 |
| `stages.2.0.bn2` | `BatchNorm2d` | `1×256×14×14` | `1×256×14×14` | 512 | 50,176 |
| `stages.2.0.attention.squeeze` | `AdaptiveAvgPool2d` | `1×256×14×14` | `1×256×1×1` | 0 | 0 |
| `stages.2.0.attention.excitation.0` | `Conv2d` | `1×256×1×1` | `1×32×1×1` | 8,224 | 8,192 |
| `stages.2.0.attention.excitation.2` | `Conv2d` | `1×32×1×1` | `1×256×1×1` | 8,448 | 8,192 |
| `stages.2.1.conv1` | `Conv2d` | `1×256×14×14` | `1×256×14×14` | 589,824 | 115,605,504 |
| `stages.2.1.bn1` | `BatchNorm2d` | `1×256×14×14` | `1×256×14×14` | 512 | 50,176 |
| `stages.2.1.conv2` | `Conv2d` | `1×256×14×14` | `1×256×14×14` | 589,824 | 115,605,504 |
| `stages.2.1.bn2` | `BatchNorm2d` | `1×256×14×14` | `1×256×14×14` | 512 | 50,176 |
| `stages.2.1.attention.squeeze` | `AdaptiveAvgPool2d` | `1×256×14×14` | `1×256×1×1` | 0 | 0 |
| `stages.2.1.attention.excitation.0` | `Conv2d` | `1×256×1×1` | `1×32×1×1` | 8,224 | 8,192 |
| `stages.2.1.attention.excitation.2` | `Conv2d` | `1×32×1×1` | `1×256×1×1` | 8,448 | 8,192 |
| `global_pool` | `AdaptiveAvgPool2d` | `1×256×14×14` | `1×256×1×1` | 0 | 0 |
| `dropout` | `Dropout` | `1×256` | `1×256` | 0 | 0 |
| `classifier` | `Linear` | `1×256` | `1×38` | 9,766 | 9,728 |

---

## 4. Checkpoint & Reproducibility Design

Reusable checkpoint utilities are implemented in `ml/src/models/checkpoint_utils.py` (`save_model_checkpoint` and `load_model_checkpoint`), storing:
- `model_state_dict`
- `model_config`
- `class_names` and `class_mapping`
- `random_seed`
- `architecture_name`

No trained weights have been created in `artifacts/` during Phase 3.
