"""Model loading and inference service reusing Phase 2, Phase 3, and Phase 5 modules.

Guarantees:
  - Loads ONLY a verified trained AgriVision-CNN (or configured Phase 3) checkpoint.
  - Uses Phase 3 `create_model` factory and Phase 2 `class_mapping.yaml` + `get_eval_transform`.
  - Never invents model weights or runs an untrained random model as a real predictor.
  - Handles missing or incompatible checkpoints gracefully without crashing startup.
  - Never exposes private filesystem paths through public status properties.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple
import urllib.request

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from ..core.config import PROJECT_ROOT, Settings
from ..core.exceptions import ModelUnavailableError

ML_SRC_DIR = PROJECT_ROOT / "ml" / "src"
if str(ML_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SRC_DIR))

try:
    from ml.src.config_loader import load_class_mapping, load_dataset_config
    from ml.src.evaluation.calibration import softmax_numpy
    from ml.src.evaluation.evaluate import verify_trained_checkpoint
    from ml.src.models.model_factory import create_model, load_model_config, resolve_verified_classes
    from ml.src.preprocess import get_eval_transform
except ImportError:
    from config_loader import load_class_mapping, load_dataset_config
    from evaluation.calibration import softmax_numpy
    from evaluation.evaluate import verify_trained_checkpoint
    from models.model_factory import create_model, load_model_config, resolve_verified_classes
    from preprocess import get_eval_transform

logger = logging.getLogger("plantdiseaseai.backend")


def _select_device(requested: str) -> torch.device:
    req = requested.strip().lower()
    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        logger.warning("CUDA requested via INFERENCE_DEVICE=cuda but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _compute_safe_checkpoint_id(ckpt_path: Path, epoch: int) -> str:
    """Create a non-path fingerprint (short SHA-256 + epoch) safe for API exposure."""
    hasher = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return f"ckpt-{hasher.hexdigest()[:12]}-ep{epoch}"


class ModelService:
    """Singleton service managing verified model lifecycle and deterministic inference."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device: torch.device = _select_device(settings.inference_device)
        self.lock = threading.Lock()
        if self.device.type == "cpu":
            try:
                torch.set_num_threads(1)
            except Exception:
                pass

        # Load verified Phase 2 configs & class mapping dynamically (zero hardcoding)
        self.dataset_config: Dict[str, Any] = load_dataset_config(settings.dataset_config_path)
        self.model_config: Dict[str, Any] = load_model_config(settings.model_config_path)
        self.class_mapping: Dict[str, Any] = load_class_mapping(self.dataset_config)
        self.class_names: List[str]
        self.class_names, _ = resolve_verified_classes(
            self.model_config, class_mapping_path=settings.class_mapping_path
        )
        self.canonical_defs: Dict[str, Dict[str, Any]] = self.class_mapping.get(
            "canonical_classes", {}
        )
        self.eval_transform = get_eval_transform(self.dataset_config)

        # Model runtime state
        self.model: Optional[nn.Module] = None
        self.model_name: str = str(self.model_config["model"]["architecture"])
        self.checkpoint_id: Optional[str] = None
        self.checkpoint_epoch: Optional[int] = None
        self.temperature: float = 1.0
        self.is_ready: bool = False
        self.status_detail: str = "Model service not initialized."

    def initialize(self) -> bool:
        """Verify and load the configured trained checkpoint once at startup."""
        ckpt_path = self.settings.model_checkpoint_path

        # Safe auto-download or chunk reassembly (never crashes startup if remote is unreachable)
        try:
            if not ckpt_path.exists():
                chunk_dir = ckpt_path.parent / "chunks"
                if chunk_dir.exists():
                    chunks = sorted(chunk_dir.glob("chunk_*.bin"))
                    if chunks:
                        logger.info("Reassembling model checkpoint from %d binary chunks...", len(chunks))
                        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(ckpt_path, "wb") as outfile:
                            for chunk_file in chunks:
                                with open(chunk_file, "rb") as infile:
                                    outfile.write(infile.read())
                        logger.info("Model checkpoint successfully reassembled at %s", ckpt_path)

            if not ckpt_path.exists():
                download_url = os.getenv(
                    "MODEL_DOWNLOAD_URL",
                    "https://github.com/maheshkumarmahato105-png/-FloraGuard-AI-/releases/download/v1.0.0/best_model.pth",
                )
                if download_url:
                    logger.info("Checking for model checkpoint download from %s...", download_url)
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    temp_file = ckpt_path.with_suffix(".tmp")
                    urllib.request.urlretrieve(download_url, str(temp_file))
                    if temp_file.exists() and temp_file.stat().st_size > 1000000:
                        temp_file.replace(ckpt_path)
                        logger.info("Successfully downloaded and installed checkpoint to %s", ckpt_path)
                    else:
                        temp_file.unlink(missing_ok=True)
        except Exception as setup_err:
            logger.warning("Optional checkpoint preparation skipped or failed: %s", setup_err)

        check = verify_trained_checkpoint(ckpt_path, expected_arch=self.model_name)

        if not check["valid_trained_checkpoint"]:
            self.model = None
            self.is_ready = False
            self.checkpoint_id = None
            self.checkpoint_epoch = None
            if not check["exists"]:
                self.status_detail = (
                    "Prediction unavailable: no trained AgriVision-CNN checkpoint is configured or present. "
                    "Complete Phase 4 model training and configure MODEL_CHECKPOINT_PATH."
                )
            else:
                self.status_detail = (
                    f"Prediction unavailable: checkpoint failed validation ({check['reason']})."
                )
            logger.warning(self.status_detail, extra={"event": "model_not_ready"})
            return False

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ckpt_classes = list(ckpt["class_names"])
            ckpt_arch = str(ckpt.get("architecture_name", self.model_name))
            ckpt_model_cfg = ckpt.get("model_config") or self.model_config

            # Verify compatibility with Phase 2 verified class mapping
            unknown_classes = [c for c in ckpt_classes if c not in self.canonical_defs]
            if unknown_classes and len(ckpt_classes) != len(self.class_names):
                logger.warning(
                    "Checkpoint uses a custom class list not identical to full Phase 2 mapping; "
                    "preserving checkpoint class order."
                )

            model = create_model(
                architecture=ckpt_arch,
                model_config=ckpt_model_cfg,
                num_classes=len(ckpt_classes),
            )
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            model.to(self.device)
            model.eval()

            self.model = model
            self.model_name = ckpt_arch
            self.class_names = ckpt_classes
            self.checkpoint_epoch = int(ckpt["epoch"])
            self.checkpoint_id = _compute_safe_checkpoint_id(ckpt_path, self.checkpoint_epoch)
            self.temperature = float(ckpt.get("calibration_temperature", 1.0))
            self.is_ready = True
            self.status_detail = (
                f"Verified trained '{self.model_name}' checkpoint loaded on {self.device.type}."
            )
            logger.info(self.status_detail, extra={"event": "model_loaded"})
            return True
        except Exception as exc:
            self.model = None
            self.is_ready = False
            self.checkpoint_id = None
            self.status_detail = (
                "Prediction unavailable: configured checkpoint could not be loaded safely."
            )
            logger.exception("Failed to load configured model checkpoint", exc_info=exc)
            return False

    def preprocess_image(self, rgb_image: Image.Image) -> torch.Tensor:
        """Apply exact Phase 2 deterministic evaluation transform -> (1, 3, 224, 224)."""
        return self.eval_transform(rgb_image).unsqueeze(0)

    def predict_tensor(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Run deterministic forward pass returning (logits_1d, calibrated_probs_1d)."""
        if not self.is_ready or self.model is None:
            raise ModelUnavailableError(self.status_detail)

        with self.lock:
            self.model.eval()
            with torch.no_grad():
                logits = self.model(input_tensor.to(self.device))
            logits_np = logits.detach().cpu().numpy()

        probs_np = softmax_numpy(logits_np, temperature=self.temperature)
        return logits_np[0], probs_np[0]

    def get_class_metadata(self, class_name: str) -> Dict[str, Any]:
        """Resolve human-readable plant/disease/healthy metadata from verified Phase 2 mapping."""
        cdef = self.canonical_defs.get(class_name)
        if cdef:
            plant = str(cdef["plant_species"])
            disease = str(cdef["disease_label"])
            is_healthy = bool(cdef["is_healthy"])
            return {
                "class_name": class_name,
                "plant_name": plant,
                "disease_name": disease,
                "display_name": f"{plant} — {disease}",
                "is_healthy": is_healthy,
            }
        # Clean fallback if checkpoint uses a custom subset
        parts = class_name.split("__", 1)
        plant = parts[0].replace("_", " ").title()
        disease = parts[1].replace("_", " ").title() if len(parts) > 1 else "Unknown"
        is_healthy = "healthy" in disease.lower()
        return {
            "class_name": class_name,
            "plant_name": plant,
            "disease_name": "Healthy" if is_healthy else disease,
            "display_name": f"{plant} — {'Healthy' if is_healthy else disease}",
            "is_healthy": is_healthy,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "model_ready": self.is_ready,
            "model_name": self.model_name,
            "checkpoint_id": self.checkpoint_id,
            "device": self.device.type,
            "num_classes": len(self.class_names),
            "status_detail": self.status_detail,
        }
