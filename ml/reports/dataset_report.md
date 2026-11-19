# PlantDiseaseAI — Phase 2 Dataset Research Report

- **Generated At (UTC):** `2026-11-18T05:52:58.545786+00:00`
- **Random Seed:** `42`
- **Phase Scope:** Dataset Collection, Exploration, Validation/Cleaning, Duplicate & Leakage Prevention, and Preprocessing (**Model training has NOT started yet**).

> **Methodological Note:** All numerical counts, image resolutions, duplicate clusters, RGB channel statistics, and leakage audit checks below are **measured results** computed directly from the acquired raw and processed datasets. Any forward-looking notes for PlantSeg or Phase 3 CNN training are explicitly labeled as **Expected / Future Phase**.

---

## 1. Dataset Sources

| Dataset | Role | Status (Measured) | Images Verified | Source Reference |
| :--- | :--- | :--- | ---: | :--- |
| **plantvillage** | `primary_train_val_test` | `unknown` | 0 | N/A |
| **PlantDoc** | `external_real_world_evaluation` | `downloaded_git_windows_safe_archive` | 2,573 | https://github.com/pratikkayal/PlantDoc-Dataset |
| **plantseg** | `future_segmentation_research` | `unknown` | 0 | N/A |

### Dataset Isolation Policy
- **PlantVillage** serves as the primary labeled classification corpus split into `train` / `validation` / `test`.
- **PlantDoc** is preserved strictly as a **separate external real-world evaluation dataset** (`external_plantdoc`) with verified label mapping in `ml/configs/class_mapping.yaml`. Incompatible PlantDoc classes are never silently merged into PlantVillage.
- **PlantSeg** is reserved for **future pixel-level disease segmentation research** (manual acquisition instructions saved to `ml/data/raw/plantseg/DOWNLOAD_INSTRUCTIONS.md`; no access controls bypassed).

---

## 2. Dataset Licensing and Attribution

| Dataset | License & Citation / Attribution |
| :--- | :--- |
| **PlantVillage** | CC0 / Academic Research — Hughes, D., & Salathé, M. (2015), *An open access repository of images on plant health to enable the development of mobile disease diagnostics*. |
| **PlantDoc** | CC BY 4.0 (Singh et al., ACM CoDS-COMAD 2020) — Singh, D., et al. (2020), *PlantDoc: A Dataset for Visual Plant Disease Detection*, ACM India Joint International Conference on Data Science & Management of Data (CoDS-COMAD). |
| **PlantSeg** | See Nature Scientific Data descriptor — Nature Scientific Data (2025), DOI: `s41597-025-06513-4`. |

---

## 3. Total Image Counts (Measured)

- **Total raw files scanned:** `56,954`
- **Total valid images (passed Pillow + OpenCV + dimension checks):** `56,952`
- **Total quarantined files (copied to `ml/data/interim/quarantine/`, raw untouched):** `2`

### PlantVillage (Primary Classification Dataset — Measured)
- Valid images: **54,381**
- Plant species: **14** (Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange (Citrus), Peach, Pepper (Bell), Potato, Raspberry, Soybean, Squash, Strawberry, Tomato)
- Total classes: **38** (**12** healthy, **26** diseased)
- Images per class: min **152**, median **1092.5**, max **5,507**
- Class imbalance ratio (max / min): **36.23x**

### PlantDoc (External Real-World Evaluation Dataset — Measured)
- Valid images: **2,571**
- Plant species: **13** (Apple, Blueberry, Cherry, Corn (Maize), Grape, Peach, Pepper (Bell), Potato, Raspberry, Soybean, Squash, Strawberry, Tomato)
- Total classes: **28** (**10** healthy, **18** diseased)
- Images per class: min **2**, median **91.0**, max **191**
- Class imbalance ratio (max / min): **95.5x**

---

## 4. Class Distribution (Measured)

Visual chart: `ml/reports/charts/class_distribution.png` | Full CSV: `ml/data/metadata/class_distribution.csv`

| Dataset | Canonical Class | Plant | Disease | Healthy | Mapping Status | Valid Images |
| :--- | :--- | :--- | :--- | :---: | :--- | ---: |
| plantdoc | `corn__northern_leaf_blight` | Corn (Maize) | Northern Leaf Blight | No | `verified_compatible` | 191 |
| plantdoc | `tomato__septoria_leaf_spot` | Tomato | Septoria Leaf Spot | No | `verified_compatible` | 151 |
| plantdoc | `squash__powdery_mildew` | Squash | Powdery Mildew | No | `verified_compatible` | 130 |
| plantdoc | `raspberry__healthy` | Raspberry | Healthy | Yes | `verified_compatible` | 119 |
| plantdoc | `corn__common_rust` | Corn (Maize) | Common Rust | No | `verified_compatible` | 116 |
| plantdoc | `potato__early_blight` | Potato | Early Blight | No | `verified_compatible` | 116 |
| plantdoc | `blueberry__healthy` | Blueberry | Healthy | Yes | `verified_compatible` | 115 |
| plantdoc | `peach__healthy` | Peach | Healthy | Yes | `verified_compatible` | 111 |
| plantdoc | `tomato__late_blight` | Tomato | Late Blight | No | `verified_compatible` | 111 |
| plantdoc | `tomato__bacterial_spot` | Tomato | Bacterial Spot | No | `verified_compatible` | 110 |
| plantdoc | `potato__late_blight` | Potato | Late Blight | No | `verified_compatible` | 104 |
| plantdoc | `strawberry__healthy` | Strawberry | Healthy | Yes | `verified_compatible` | 96 |
| plantdoc | `apple__apple_scab` | Apple | Apple Scab | No | `verified_compatible` | 93 |
| plantdoc | `apple__healthy` | Apple | Healthy | Yes | `verified_compatible` | 91 |
| plantdoc | `tomato__leaf_mold` | Tomato | Leaf Mold | No | `verified_compatible` | 91 |
| plantdoc | `apple__cedar_apple_rust` | Apple | Cedar Apple Rust | No | `verified_compatible` | 88 |
| plantdoc | `tomato__early_blight` | Tomato | Early Blight | No | `verified_compatible` | 88 |
| plantdoc | `tomato__tomato_yellow_leaf_curl_virus` | Tomato | Tomato Yellow Leaf Curl Virus | No | `verified_compatible` | 76 |
| plantdoc | `pepper_bell__bacterial_spot` | Pepper (Bell) | Bacterial Spot | No | `verified_compatible` | 71 |
| plantdoc | `grape__healthy` | Grape | Healthy | Yes | `verified_compatible` | 69 |
| plantdoc | `corn__cercospora_gray_leaf_spot` | Corn (Maize) | Cercospora Leaf Spot (Gray Leaf Spot) | No | `verified_compatible` | 68 |
| plantdoc | `soybean__healthy` | Soybean | Healthy | Yes | `verified_compatible` | 65 |
| plantdoc | `grape__black_rot` | Grape | Black Rot | No | `verified_compatible` | 64 |
| plantdoc | `tomato__healthy` | Tomato | Healthy | Yes | `verified_compatible` | 63 |
| plantdoc | `pepper_bell__healthy` | Pepper (Bell) | Healthy | Yes | `verified_compatible` | 61 |
| plantdoc | `cherry__healthy` | Cherry | Healthy | Yes | `verified_compatible` | 57 |
| plantdoc | `tomato__tomato_mosaic_virus` | Tomato | Tomato Mosaic Virus | No | `verified_compatible` | 54 |
| plantdoc | `tomato__spider_mites_two_spotted` | Tomato | Spider Mites (Two-Spotted Spider Mite) | No | `verified_compatible` | 2 |
| plantvillage | `orange__huanglongbing_citrus_greening` | Orange (Citrus) | Huanglongbing (Citrus Greening) | No | `verified_compatible` | 5,507 |
| plantvillage | `tomato__tomato_yellow_leaf_curl_virus` | Tomato | Tomato Yellow Leaf Curl Virus | No | `verified_compatible` | 5,357 |
| plantvillage | `soybean__healthy` | Soybean | Healthy | Yes | `verified_compatible` | 5,090 |
| plantvillage | `peach__bacterial_spot` | Peach | Bacterial Spot | No | `verified_compatible` | 2,297 |
| plantvillage | `tomato__bacterial_spot` | Tomato | Bacterial Spot | No | `verified_compatible` | 2,137 |
| plantvillage | `tomato__late_blight` | Tomato | Late Blight | No | `verified_compatible` | 1,909 |
| plantvillage | `squash__powdery_mildew` | Squash | Powdery Mildew | No | `verified_compatible` | 1,835 |
| plantvillage | `tomato__septoria_leaf_spot` | Tomato | Septoria Leaf Spot | No | `verified_compatible` | 1,771 |
| plantvillage | `tomato__spider_mites_two_spotted` | Tomato | Spider Mites (Two-Spotted Spider Mite) | No | `verified_compatible` | 1,676 |
| plantvillage | `apple__healthy` | Apple | Healthy | Yes | `verified_compatible` | 1,665 |
| plantvillage | `tomato__healthy` | Tomato | Healthy | Yes | `verified_compatible` | 1,625 |
| plantvillage | `blueberry__healthy` | Blueberry | Healthy | Yes | `verified_compatible` | 1,502 |
| plantvillage | `pepper_bell__healthy` | Pepper (Bell) | Healthy | Yes | `verified_compatible` | 1,478 |
| plantvillage | `tomato__target_spot` | Tomato | Target Spot | No | `verified_compatible` | 1,404 |
| plantvillage | `grape__esca_black_measles` | Grape | Esca (Black Measles) | No | `verified_compatible` | 1,383 |
| plantvillage | `corn__common_rust` | Corn (Maize) | Common Rust | No | `verified_compatible` | 1,192 |
| plantvillage | `grape__black_rot` | Grape | Black Rot | No | `verified_compatible` | 1,180 |
| plantvillage | `corn__healthy` | Corn (Maize) | Healthy | Yes | `verified_compatible` | 1,162 |
| plantvillage | `strawberry__leaf_scorch` | Strawberry | Leaf Scorch | No | `verified_compatible` | 1,109 |
| plantvillage | `grape__leaf_blight_isariopsis` | Grape | Leaf Blight (Isariopsis Leaf Spot) | No | `verified_compatible` | 1,076 |
| plantvillage | `cherry__powdery_mildew` | Cherry | Powdery Mildew | No | `verified_compatible` | 1,052 |
| plantvillage | `potato__early_blight` | Potato | Early Blight | No | `verified_compatible` | 1,000 |
| plantvillage | `potato__late_blight` | Potato | Late Blight | No | `verified_compatible` | 1,000 |
| plantvillage | `tomato__early_blight` | Tomato | Early Blight | No | `verified_compatible` | 1,000 |
| plantvillage | `pepper_bell__bacterial_spot` | Pepper (Bell) | Bacterial Spot | No | `verified_compatible` | 997 |
| plantvillage | `corn__northern_leaf_blight` | Corn (Maize) | Northern Leaf Blight | No | `verified_compatible` | 985 |
| plantvillage | `tomato__leaf_mold` | Tomato | Leaf Mold | No | `verified_compatible` | 952 |
| plantvillage | `cherry__healthy` | Cherry | Healthy | Yes | `verified_compatible` | 854 |
| plantvillage | `apple__apple_scab` | Apple | Apple Scab | No | `verified_compatible` | 631 |
| plantvillage | `apple__black_rot` | Apple | Black Rot | No | `verified_compatible` | 621 |
| plantvillage | `corn__cercospora_gray_leaf_spot` | Corn (Maize) | Cercospora Leaf Spot (Gray Leaf Spot) | No | `verified_compatible` | 514 |
| plantvillage | `strawberry__healthy` | Strawberry | Healthy | Yes | `verified_compatible` | 456 |
| plantvillage | `grape__healthy` | Grape | Healthy | Yes | `verified_compatible` | 423 |
| plantvillage | `tomato__tomato_mosaic_virus` | Tomato | Tomato Mosaic Virus | No | `verified_compatible` | 373 |
| plantvillage | `raspberry__healthy` | Raspberry | Healthy | Yes | `verified_compatible` | 371 |
| plantvillage | `peach__healthy` | Peach | Healthy | Yes | `verified_compatible` | 360 |
| plantvillage | `apple__cedar_apple_rust` | Apple | Cedar Apple Rust | No | `verified_compatible` | 285 |
| plantvillage | `potato__healthy` | Potato | Healthy | Yes | `verified_compatible` | 152 |

---

## 5. Image Quality Statistics (Measured)

Visual charts: `ml/reports/charts/resolution_distribution.png`, `ml/reports/charts/sample_gallery.png`

| Dataset | Avg Dimensions (W × H) | Min Dimensions | Max Dimensions | Formats | Color Modes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **plantdoc** | 1062.73 × 916.45 px | 115 × 69 px | 6000 × 6000 px | `{'JPEG': 2564, 'PNG': 5, 'MPO': 2}` | `{'RGB': 2560, 'CMYK': 5, 'RGBA': 5, 'L': 1}` |
| **plantvillage** | 256.0 × 256.0 px | 256 × 256 px | 256 × 256 px | `{'JPEG': 54381}` | `{'RGB': 54381}` |

### Cleaning & Quarantine Audit (Measured)
- Raw files modified or deleted: **`False`** (strictly non-destructive)
- Validation issues detected: `{'missing_label': 1, 'unusual_aspect_ratio': 1}`
- Quarantine directory: `ml/data/interim/quarantine`

---

## 6. Duplicate Detection Results (Measured)

- Hashing method: `sha256 (file + decoded RGB pixels)` + `64-bit pHash (DCT) + 64-bit dHash`
- Perceptual thresholds: pHash Hamming $\le$ `4`, dHash Hamming $\le$ `6`
- Total valid images hashed: **`56,952`**
- Total duplicate/near-duplicate clusters: **`414`**
  - Clusters containing exact duplicates: **`304`**
  - Near-duplicate-only clusters: **`110`**
- Role counts: `{'unique': 56121, 'primary': 414, 'exact_duplicate': 304, 'near_duplicate': 113}`
- Cross-dataset duplicate clusters (PlantVillage vs. PlantDoc): **`0`**
- Cross-label duplicate clusters: **`31`**

---

## 7. Preprocessing Methods

- Target dimensions: **`224 × 224` RGB**
- Aspect ratio preservation: **`True`** using letterbox padding (`mode=constant`, `fill_rgb=[124, 116, 104]`) so leaf lesion geometry is never distorted.
- Offline preprocessing applied to disk: **Deterministic resize + pad ONLY (`offline_augmentation_applied=False`)** — total processed images saved: **`56,648`**.
- Training-only online augmentation (configured in `ml/configs/dataset_config.yaml`, applied **only** during Phase 3 training):
  - Random horizontal flip (`p=0.5`), small-angle rotation (`±15°`), random resized crop (`scale=[0.85, 1.0]`), brightness (`0.20`), contrast (`0.20`), saturation (`0.15`), and mild hue (`0.03`) jitter.
- Validation, test, and live inference transform: **Deterministic LetterboxResize(224×224) → ToTensor → Normalize** (identical between validation, test, and production inference).
- Configured ImageNet normalization: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
- Measured empirical RGB stats on processed **training split only** (`n=4,000` sampled): `mean=[0.468, 0.4909, 0.4117]`, `std=[0.1956, 0.1708, 0.2129]`.

---

## 8. Train / Validation / Test Distribution (Measured)

Visual chart: `ml/reports/charts/split_distribution.png` | Full manifest: `ml/data/metadata/split_manifest.csv`

| Assigned Split | Image Count | Notes |
| :--- | ---: | :--- |
| `train` | 37,844 | Primary PlantVillage training split (~70%, group-aware stratified) |
| `validation` | 8,127 | Primary PlantVillage validation split (~15%, group-aware stratified) |
| `test` | 8,123 | Primary PlantVillage held-out test split (~15%, group-aware stratified) |
| `external_plantdoc` | 2,554 | External real-world PlantDoc evaluation set (never mixed into training) |
| `excluded_exact_duplicate` | 304 | Exact duplicate secondary copies excluded from train/val/test (raw files preserved) |
| `quarantined` | 2 | Failed Pillow/OpenCV/dimension/format validation (copied to quarantine) |

- Actual PlantVillage core split ratios: **Train `69.96%` / Validation `15.02%` / Test `15.02%`** (target: 70% / 15% / 15%).
- Classes below minimum sample threshold (`10`): **`0`**.

---

## 9. Data Leakage Checks (Measured)

- Overall zero-leakage status: **`PASSED`**

| Leakage Audit Dimension | Train ∩ Val | Train ∩ Test | Val ∩ Test | Status |
| :--- | ---: | ---: | ---: | :---: |
| `image_path` | 0 | 0 | 0 | **PASS** |
| `file_sha256` | 0 | 0 | 0 | **PASS** |
| `pixel_sha256` | 0 | 0 | 0 | **PASS** |
| `duplicate_group_id` | 0 | 0 | 0 | **PASS** |
| `external_plantdoc vs plantvillage groups` | — | — | Overlap: 0 | **PASS** |

---

## 10. Dataset Limitations & Research Considerations

### Measured Limitations in Current Data
1. **Class Imbalance (PlantVillage):** Class sizes range from `152` to `5507` images (`36.23x` ratio). Phase 3 training should use class-weighted loss or balanced sampling and report Macro F1 alongside accuracy.
2. **Domain Shift (Lab vs. Field):** PlantVillage images are mostly uniform `256 × 256` single-leaf studio/plain-background photographs, whereas PlantDoc images vary dramatically in resolution, aspect ratio, lighting, multi-leaf clutter, and natural field backgrounds.
3. **Label Taxonomy Differences:** PlantDoc contains several broad or distinct labels (e.g., general species leaf categories or non-PlantVillage diseases) that do **not** map 1-to-1 to PlantVillage classes; these are explicitly marked `verified_external_only` in `ml/configs/class_mapping.yaml`.
4. **Duplicate & Near-Duplicate Prevalence:** Perceptual hashing revealed exact and near-duplicate leaf images within raw collections; keeping entire clusters inside a single split prevents artificially inflated validation/test scores.

### Expected / Future Phase Considerations
- **PlantSeg Integration (Future Phase):** Once acquired manually under the Nature Scientific Data license (`s41597-025-06513-4`), PlantSeg pixel masks will enable disease lesion localization and foreground leaf segmentation to bridge the lab-to-field domain gap.
- **Phase 3 Model Training (Not Yet Started):** Expected baseline experiments will train a custom PyTorch CNN on `ml/data/processed/train`, tune hyperparameters on `validation`, evaluate in-domain generalization on `test`, and benchmark out-of-domain real-world generalization on compatible classes of `external_plantdoc`.
