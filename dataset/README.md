# PlantVillage Dataset Directory

This folder stores the PlantVillage dataset downloaded and split by the training pipeline.

## Automated Download (Recommended)

From the project root with the virtual environment activated:

```powershell
.venv\Scripts\python.exe training/download_dataset.py
.venv\Scripts\python.exe training/preprocess.py
```

By default, `download_dataset.py` downloads the 38-class **PlantVillage (color)** dataset from Hugging Face (`mohanty/PlantVillage`) into `dataset/raw/<class_name>/` and writes `dataset/metadata.csv`.

## Manual Download Alternative (Kaggle)

If you prefer Kaggle:
1. Download the PlantVillage Dataset (color images, 38 class folders) from Kaggle.
2. Extract the 38 class folders directly into `dataset/raw/` so each folder looks like `dataset/raw/Tomato___Early_blight/*.jpg`.
3. Run `.venv\Scripts\python.exe training/preprocess.py` to validate images and generate the stratified `train.csv`, `val.csv`, and `test.csv` splits in `dataset/splits/`.
