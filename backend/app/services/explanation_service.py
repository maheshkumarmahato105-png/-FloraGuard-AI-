"""Thread-safe PyTorch Grad-CAM explanation service (Phase 7).

Features:
  - Uses the loaded AgriVision-CNN (or compatible CNN) model's last convolutional layer.
  - Serializes hook registration + backward pass under `model_service.lock` so concurrent
    API requests NEVER interfere with shared model hooks.
  - Always removes registered forward/backward hooks inside a `finally:` block.
  - Saves heatmap and overlay PNGs strictly inside `settings.results_dir`.
  - Handles failures gracefully without discarding a valid prediction.
  - Explicitly labels outputs as approximate AI attention regions (NOT verified
    pixel-level disease segmentation masks).
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from ..core.config import PROJECT_ROOT, Settings
from .model_service import ModelService

ML_SRC_DIR = PROJECT_ROOT / "ml" / "src"
if str(ML_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SRC_DIR))

try:
    from ml.src.preprocess import resize_and_pad_rgb
except ImportError:
    from preprocess import resize_and_pad_rgb

logger = logging.getLogger("plantdiseaseai.backend")

GRADCAM_DISCLAIMER = (
    "Grad-CAM highlights approximate AI attention regions used for classification "
    "and is NOT a verified pixel-level disease segmentation mask."
)


def _find_target_conv_layer(model: nn.Module) -> nn.Module:
    """Locate the final convolutional feature layer for AgriVision-CNN or baseline CNNs."""
    # Preferred target for AgriVision-CNN: last conv2 inside final ResidualBlock
    if hasattr(model, "stages") and isinstance(model.stages, nn.ModuleList) and len(model.stages) > 0:
        last_stage = model.stages[-1]
        if isinstance(last_stage, nn.Sequential) and len(last_stage) > 0:
            last_block = last_stage[-1]
            if hasattr(last_block, "conv2") and isinstance(last_block.conv2, nn.Conv2d):
                return last_block.conv2

    # Generic fallback: last Conv2d module in the network
    last_conv: Optional[nn.Conv2d] = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found in model for Grad-CAM.")
    return last_conv


class ExplanationService:
    """Generates Grad-CAM heatmaps and leaf overlays under a controlled results directory."""

    def __init__(self, settings: Settings, model_service: ModelService) -> None:
        self.settings = settings
        self.model_service = model_service

    def generate_gradcam(
        self,
        rgb_image: Image.Image,
        input_tensor: torch.Tensor,
        target_class_idx: int,
        request_id: str,
    ) -> Dict[str, Any]:
        """Generate heatmap and overlay PNG files safely. Never raises into prediction route."""
        if not self.model_service.is_ready or self.model_service.model is None:
            return {
                "available": False,
                "requested": True,
                "method": "Grad-CAM",
                "heatmap_filename": None,
                "overlay_filename": None,
                "note": "Explanation unavailable because the AI model is not loaded.",
            }

        try:
            cam_map = self._compute_cam(
                model=self.model_service.model,
                input_tensor=input_tensor,
                target_class_idx=target_class_idx,
                device=self.model_service.device,
            )

            img_cfg = self.model_service.dataset_config["image"]
            target_w = int(img_cfg["target_width"])
            target_h = int(img_cfg["target_height"])
            base_rgb = resize_and_pad_rgb(
                rgb_image,
                target_size=(target_w, target_h),
                preserve_aspect_ratio=bool(img_cfg.get("preserve_aspect_ratio", True)),
                padding_mode=str(img_cfg.get("padding_mode", "constant")),
                fill_rgb=tuple(img_cfg.get("padding_fill_rgb", [124, 116, 104])),
            )
            base_np_rgb = np.asarray(base_rgb, dtype=np.uint8)

            cam_resized = cv2.resize(cam_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            cam_uint8 = np.uint8(np.clip(cam_resized * 255.0, 0, 255))
            heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
            overlay_rgb = np.uint8(np.clip(0.58 * base_np_rgb + 0.42 * heatmap_rgb, 0, 255))

            self.settings.results_dir.mkdir(parents=True, exist_ok=True)
            safe_id = "".join(ch for ch in request_id if ch.isalnum())[:32]
            heatmap_name = f"{safe_id}_heatmap.png"
            overlay_name = f"{safe_id}_overlay.png"

            Image.fromarray(heatmap_rgb, mode="RGB").save(
                self.settings.results_dir / heatmap_name, format="PNG"
            )
            Image.fromarray(overlay_rgb, mode="RGB").save(
                self.settings.results_dir / overlay_name, format="PNG"
            )

            return {
                "available": True,
                "requested": True,
                "method": "Grad-CAM",
                "heatmap_filename": heatmap_name,
                "overlay_filename": overlay_name,
                "note": GRADCAM_DISCLAIMER,
            }
        except Exception as exc:
            logger.exception(
                "Grad-CAM explanation generation failed; preserving primary prediction",
                extra={"request_id": request_id, "event": "gradcam_failed"},
                exc_info=exc,
            )
            return {
                "available": False,
                "requested": True,
                "method": "Grad-CAM",
                "heatmap_filename": None,
                "overlay_filename": None,
                "note": "Approximate Grad-CAM explanation could not be generated for this request.",
            }

    def _compute_cam(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_class_idx: int,
        device: torch.device,
    ) -> np.ndarray:
        """Compute normalized [0, 1] 2D Grad-CAM map under model lock with guaranteed hook cleanup."""
        target_layer = _find_target_conv_layer(model)
        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        def _fwd_hook(_module, _inp, output):
            activations.append(output.detach())

        def _bwd_hook(_module, _grad_in, grad_out):
            gradients.append(grad_out[0].detach())

        with self.model_service.lock:
            fwd_handle = target_layer.register_forward_hook(_fwd_hook)
            bwd_handle = target_layer.register_full_backward_hook(_bwd_hook)
            try:
                model.eval()
                model.zero_grad(set_to_none=True)
                x = input_tensor.to(device).detach().requires_grad_(True)
                with torch.enable_grad():
                    logits = model(x)
                    score = logits[0, int(target_class_idx)]
                    score.backward()

                if not activations or not gradients:
                    raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

                act = activations[0][0]   # (C, H, W)
                grad = gradients[0][0]    # (C, H, W)
                weights = grad.mean(dim=(1, 2), keepdim=True)
                cam = torch.relu((weights * act).sum(dim=0))
                cam_np = cam.detach().cpu().numpy().astype(np.float32)

                max_val = float(cam_np.max())
                if max_val > 1e-8:
                    cam_np = cam_np / max_val
                else:
                    cam_np = np.zeros_like(cam_np)
                return cam_np
            finally:
                fwd_handle.remove()
                bwd_handle.remove()
                activations.clear()
                gradients.clear()
                model.zero_grad(set_to_none=True)
                import gc
                gc.collect()

    def resolve_safe_artifact_path(self, filename: str) -> Optional[Path]:
        """Resolve an explanation filename strictly inside `settings.results_dir` (blocks traversal)."""
        if not filename or "/" in filename or "\\" in filename or ".." in filename:
            return None
        if not filename.endswith(".png"):
            return None
        root = self.settings.results_dir.resolve()
        candidate = (root / filename).resolve()
        if candidate.parent != root or not candidate.is_file():
            return None
        return candidate

    def delete_artifacts(self, heatmap_filename: Optional[str], overlay_filename: Optional[str]) -> None:
        for fname in (heatmap_filename, overlay_filename):
            if fname:
                p = self.resolve_safe_artifact_path(fname)
                if p is not None:
                    try:
                        p.unlink()
                    except OSError:
                        pass
