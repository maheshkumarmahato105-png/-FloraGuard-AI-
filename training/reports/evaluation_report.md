# Plant Disease Detection — Model Evaluation Report

- **Evaluated At (UTC):** `2026-11-18T05:26:41.895463+00:00`
- **Model Architecture:** `mobilenet_v2`
- **Number of Classes:** `38`
- **Test Set Size:** `8158` images

## Overall Test Set Performance

| Metric | Score |
| --- | ---: |
| **Test Accuracy** | **99.57%** |
| **Top-3 Accuracy** | **100.00%** |
| **Macro Precision** | **99.39%** |
| **Macro Recall** | **99.57%** |
| **Macro F1-Score** | **99.47%** |
| **Weighted Precision** | **99.58%** |
| **Weighted Recall** | **99.57%** |
| **Weighted F1-Score** | **99.57%** |

## Per-Class Classification Metrics

| Plant | Condition | Precision | Recall | F1-Score | Support |
| --- | --- | ---: | ---: | ---: | ---: |
| Apple | Apple Scab | 100.00% | 100.00% | 100.00% | 95 |
| Apple | Black Rot | 100.00% | 100.00% | 100.00% | 93 |
| Apple | Cedar Apple Rust | 100.00% | 100.00% | 100.00% | 43 |
| Apple | Healthy | 100.00% | 100.00% | 100.00% | 250 |
| Blueberry | Healthy | 100.00% | 100.00% | 100.00% | 225 |
| Cherry | Powdery Mildew | 100.00% | 100.00% | 100.00% | 158 |
| Cherry | Healthy | 99.22% | 100.00% | 99.61% | 128 |
| Corn (Maize) | Cercospora Leaf Spot (Gray Leaf Spot) | 89.53% | 100.00% | 94.48% | 77 |
| Corn (Maize) | Common Rust | 100.00% | 100.00% | 100.00% | 179 |
| Corn (Maize) | Northern Leaf Blight | 100.00% | 93.92% | 96.86% | 148 |
| Corn (Maize) | Healthy | 100.00% | 100.00% | 100.00% | 174 |
| Grape | Black Rot | 100.00% | 100.00% | 100.00% | 177 |
| Grape | Esca (Black Measles) | 100.00% | 100.00% | 100.00% | 207 |
| Grape | Leaf Blight (Isariopsis Leaf Spot) | 100.00% | 100.00% | 100.00% | 161 |
| Grape | Healthy | 100.00% | 100.00% | 100.00% | 63 |
| Orange (Citrus) | Huanglongbing (Citrus Greening) | 100.00% | 100.00% | 100.00% | 826 |
| Peach | Bacterial Spot | 100.00% | 99.71% | 99.85% | 345 |
| Peach | Healthy | 98.18% | 100.00% | 99.08% | 54 |
| Pepper (Bell) | Bacterial Spot | 100.00% | 100.00% | 100.00% | 149 |
| Pepper (Bell) | Healthy | 100.00% | 100.00% | 100.00% | 222 |
| Potato | Early Blight | 100.00% | 100.00% | 100.00% | 150 |
| Potato | Late Blight | 99.34% | 100.00% | 99.67% | 150 |
| Potato | Healthy | 100.00% | 100.00% | 100.00% | 23 |
| Raspberry | Healthy | 100.00% | 100.00% | 100.00% | 56 |
| Soybean | Healthy | 100.00% | 99.87% | 99.93% | 764 |
| Squash | Powdery Mildew | 100.00% | 100.00% | 100.00% | 275 |
| Strawberry | Leaf Scorch | 100.00% | 100.00% | 100.00% | 166 |
| Strawberry | Healthy | 100.00% | 100.00% | 100.00% | 68 |
| Tomato | Bacterial Spot | 99.37% | 98.13% | 98.75% | 321 |
| Tomato | Early Blight | 97.32% | 96.67% | 96.99% | 150 |
| Tomato | Late Blight | 98.59% | 97.90% | 98.25% | 286 |
| Tomato | Leaf Mold | 100.00% | 99.30% | 99.65% | 143 |
| Tomato | Septoria Leaf Spot | 99.63% | 100.00% | 99.81% | 266 |
| Tomato | Spider Mites (Two-Spotted Spider Mite) | 97.67% | 100.00% | 98.82% | 251 |
| Tomato | Target Spot | 98.11% | 98.58% | 98.35% | 211 |
| Tomato | Tomato Yellow Leaf Curl Virus (TYLCV) | 99.75% | 99.63% | 99.69% | 804 |
| Tomato | Tomato Mosaic Virus (ToMV) | 100.00% | 100.00% | 100.00% | 56 |
| Tomato | Healthy | 100.00% | 100.00% | 100.00% | 244 |

## Artifacts
- Confusion Matrix: `training/reports/confusion_matrix.png`
- Training Curves: `training/reports/training_curves.png`
- Raw Metrics JSON: `training/reports/evaluation_metrics.json`
