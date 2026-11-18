"""PyTorch model loading and real-time plant disease inference service.

Never fabricates or hardcodes predictions. Requires a genuinely trained model
checkpoint created by `training/train.py`.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
import torch
import torch.nn as nn
from torchvision import models

from backend.config import (
    CLASS_NAMES_PATH,
    CONFIDENCE_THRESHOLD,
    MODEL_METADATA_PATH,
    MODEL_WEIGHTS_PATH,
)
from backend.services.disease_knowledge import get_disease_info, get_supported_plants_summary
from backend.services.image_processor import preprocess_for_model


def build_model_architecture(architecture: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """Construct MobileNetV2 or EfficientNet-B0 classification backbone."""
    arch = architecture.lower().strip()
    if arch == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(in_features, num_classes),
        )
        return model
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(in_features, num_classes),
        )
        return model
    else:
        raise ValueError(f"Unsupported architecture: {architecture}. Use 'mobilenet_v2' or 'efficientnet_b0'.")


class ModelNotReadyError(RuntimeError):
    """Raised when inference is requested before a trained model checkpoint exists."""


class PlantDiseaseModelService:
    """Singleton-style service managing the trained PyTorch model lifecycle."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.class_names: List[str] = []
        self.architecture: str = "mobilenet_v2"
        self.metadata: Dict[str, Any] = {}
        self.load_error: Optional[str] = None
        self.load_model()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and len(self.class_names) > 0

    def load_model(self) -> bool:
        """Load trained model weights and class mapping from disk."""
        self.model = None
        self.class_names = []
        self.metadata = {}
        self.load_error = None

        if not MODEL_WEIGHTS_PATH.exists():
            self.load_error = (
                f"Trained model file not found at '{MODEL_WEIGHTS_PATH}'. "
                "Please run the training pipeline (training/download_dataset.py and training/train.py) first."
            )
            return False

        try:
            checkpoint = torch.load(
                MODEL_WEIGHTS_PATH,
                map_location=self.device,
                weights_only=False,
            )

            # Support both full checkpoint dict and raw state_dict + class_names.json
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.architecture = checkpoint.get("architecture", "mobilenet_v2")
                self.class_names = checkpoint.get("class_names", [])
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
                self.architecture = "mobilenet_v2"

            if not self.class_names and CLASS_NAMES_PATH.exists():
                with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                    self.class_names = json.load(f)

            if not self.class_names:
                raise ValueError("Class names mapping is missing from checkpoint and class_names.json.")

            model = build_model_architecture(
                architecture=self.architecture,
                num_classes=len(self.class_names),
                pretrained=False,
            )
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.model = model

            if MODEL_METADATA_PATH.exists():
                with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

            return True
        except Exception as exc:
            self.load_error = f"Failed to load trained model: {exc}"
            self.model = None
            return False

    def get_supported_plants(self) -> List[str]:
        """Return sorted unique list of plant categories supported by the loaded model."""
        if not self.class_names:
            return []
        summary = get_supported_plants_summary(self.class_names)
        return [entry["plant_name"] for entry in summary]

    def get_plants_detail(self) -> List[Dict]:
        """Return grouped plant categories and their detectable conditions."""
        if not self.class_names:
            return []
        return get_supported_plants_summary(self.class_names)

    def predict(self, rgb_img: Image.Image, top_k: int = 3) -> Dict[str, Any]:
        """Run real AI inference on a validated PIL RGB image."""
        if not self.is_loaded or self.model is None:
            raise ModelNotReadyError(
                self.load_error or "AI model is not loaded. Train the model before predicting."
            )

        start_time = time.perf_counter()
        input_tensor, diagnostics = preprocess_for_model(rgb_img)
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)

        inference_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        k = min(top_k, len(self.class_names))
        top_probs, top_indices = torch.topk(probabilities, k=k)

        top_predictions = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            raw_label = self.class_names[idx]
            info = get_disease_info(raw_label)
            top_predictions.append(
                {
                    "class_label": raw_label,
                    "plant_name": info["plant_name"],
                    "disease_name": info["disease_name"],
                    "status": "Healthy" if info["is_healthy"] else "Infected",
                    "confidence": round(prob * 100.0, 2),
                }
            )

        best_prob = float(top_probs[0].item())
        best_idx = int(top_indices[0].item())
        best_raw_label = self.class_names[best_idx]
        best_info = get_disease_info(best_raw_label)
        confidence_pct = round(best_prob * 100.0, 2)

        # Uncertainty / out-of-distribution safeguard — never fabricate confident predictions
        severe_image_issue = (
            diagnostics["leaf_coverage_ratio"] < 0.03 and best_prob < 0.85
        ) or (
            diagnostics["mean_brightness"] < 15.0 or diagnostics["mean_brightness"] > 248.0
        )

        is_uncertain = best_prob < CONFIDENCE_THRESHOLD or severe_image_issue

        if is_uncertain:
            reasons = []
            if best_prob < CONFIDENCE_THRESHOLD:
                reasons.append(
                    f"Model confidence ({confidence_pct:.1f}%) is below the reliability threshold ({CONFIDENCE_THRESHOLD * 100:.0f}%)."
                )
            if diagnostics["leaf_coverage_ratio"] < 0.03:
                reasons.append("A clear plant leaf could not be reliably detected in the image.")
            if diagnostics["quality_warnings"]:
                reasons.extend(diagnostics["quality_warnings"])

            uncertainty_msg = (
                "The AI model could not make a reliable diagnosis from this image. "
                + " ".join(dict.fromkeys(reasons))
                + " Please capture or upload a sharp, well-lit close-up photo of a single plant leaf against a plain background."
            )

            return {
                "raw_class_label": best_raw_label,
                "plant_name": "Uncertain",
                "disease_name": "Inconclusive — Please Capture Another Image",
                "status": "Uncertain",
                "is_healthy": False,
                "is_uncertain": True,
                "confidence": confidence_pct,
                "confidence_threshold": round(CONFIDENCE_THRESHOLD * 100.0, 1),
                "uncertainty_message": uncertainty_msg,
                "disease_info": None,
                "top_predictions": top_predictions,
                "diagnostics": diagnostics,
                "model_architecture": self.architecture,
                "inference_time_ms": inference_time_ms,
            }

        return {
            "raw_class_label": best_raw_label,
            "plant_name": best_info["plant_name"],
            "disease_name": best_info["disease_name"],
            "status": "Healthy" if best_info["is_healthy"] else "Infected",
            "is_healthy": best_info["is_healthy"],
            "is_uncertain": False,
            "confidence": confidence_pct,
            "confidence_threshold": round(CONFIDENCE_THRESHOLD * 100.0, 1),
            "uncertainty_message": None,
            "disease_info": {
                "description": best_info["description"],
                "symptoms": best_info["symptoms"],
                "causes": best_info["causes"],
                "prevention": best_info["prevention"],
                "treatment": best_info["treatment"],
                "severity": best_info["severity"],
            },
            "top_predictions": top_predictions,
            "diagnostics": diagnostics,
            "model_architecture": self.architecture,
            "inference_time_ms": inference_time_ms,
        }


# Shared service instance
model_service = PlantDiseaseModelService()
