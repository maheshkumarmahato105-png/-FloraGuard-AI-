"""Robustness Stress-Testing under Realistic Field & Camera Perturbations (Phase 5).

Evaluates model degradation under isolated, reproducible in-memory perturbations
(NEVER modifying original test images on disk):
  - Brightness decrease / increase
  - Contrast decrease / increase
  - Mild Gaussian blur (camera defocus)
  - Additive Gaussian sensor noise
  - JPEG compression artifacts (Q=30)
  - Small rotation (+12 degrees)
  - Warm / cool ambient lighting tint shifts
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_loader import PROJECT_ROOT
from preprocess import get_eval_transform, safe_load_rgb

from .calibration import softmax_numpy
from .metrics import compute_multiclass_metrics


def _perturb_brightness(factor: float) -> Callable[[Image.Image], Image.Image]:
    return lambda img: ImageEnhance.Brightness(img).enhance(factor)


def _perturb_contrast(factor: float) -> Callable[[Image.Image], Image.Image]:
    return lambda img: ImageEnhance.Contrast(img).enhance(factor)


def _perturb_gaussian_blur(radius: float = 1.5) -> Callable[[Image.Image], Image.Image]:
    return lambda img: img.filter(ImageFilter.GaussianBlur(radius=radius))


def _perturb_gaussian_noise(std: float = 14.0, seed: int = 42) -> Callable[[Image.Image], Image.Image]:
    def _fn(img: Image.Image) -> Image.Image:
        rng = np.random.default_rng(seed)
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        noisy = np.clip(arr + rng.normal(0.0, std, size=arr.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(noisy, mode="RGB")

    return _fn


def _perturb_jpeg_compression(quality: int = 30) -> Callable[[Image.Image], Image.Image]:
    def _fn(img: Image.Image) -> Image.Image:
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        with Image.open(buf) as decoded:
            return decoded.convert("RGB").copy()

    return _fn


def _perturb_rotation(angle: float = 12.0, fill: tuple[int, int, int] = (124, 116, 104)) -> Callable[[Image.Image], Image.Image]:
    return lambda img: img.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=fill)


def _perturb_lighting_tint(r_scale: float, g_scale: float, b_scale: float) -> Callable[[Image.Image], Image.Image]:
    def _fn(img: Image.Image) -> Image.Image:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        arr[:, :, 0] *= r_scale
        arr[:, :, 1] *= g_scale
        arr[:, :, 2] *= b_scale
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")

    return _fn


def get_robustness_perturbations(seed: int = 42) -> Dict[str, Callable[[Image.Image], Image.Image]]:
    """Return ordered dictionary of isolated realistic camera/field perturbations."""
    return {
        "brightness_dark_0.65": _perturb_brightness(0.65),
        "brightness_bright_1.35": _perturb_brightness(1.35),
        "contrast_low_0.65": _perturb_contrast(0.65),
        "contrast_high_1.35": _perturb_contrast(1.35),
        "mild_blur_r1.5": _perturb_gaussian_blur(radius=1.5),
        "gaussian_noise_std14": _perturb_gaussian_noise(std=14.0, seed=seed),
        "jpeg_compression_q30": _perturb_jpeg_compression(quality=30),
        "small_rotation_12deg": _perturb_rotation(angle=12.0),
        "warm_sunlight_tint": _perturb_lighting_tint(1.12, 1.03, 0.88),
        "cool_shade_tint": _perturb_lighting_tint(0.90, 0.98, 1.12),
    }


class _PerturbedEvalDataset(Dataset):
    """Applies a single deterministic in-memory perturbation before standard eval preprocessing."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        class_to_idx: Dict[str, int],
        eval_transform: Callable,
        perturbation_fn: Optional[Callable[[Image.Image], Image.Image]] = None,
        project_root: Path = PROJECT_ROOT,
    ) -> None:
        self.project_root = Path(project_root)
        self.eval_transform = eval_transform
        self.perturbation_fn = perturbation_fn
        self.paths: List[Path] = []
        self.labels: List[int] = []

        for _, row in dataframe.iterrows():
            rel = str(row.get("processed_path") or row["image_path"])
            p = Path(rel)
            self.paths.append(p if p.is_absolute() else (self.project_root / p))
            self.labels.append(class_to_idx[str(row["canonical_class"])])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        rgb = safe_load_rgb(self.paths[idx])
        if self.perturbation_fn is not None:
            rgb = self.perturbation_fn(rgb)
        return self.eval_transform(rgb), self.labels[idx]


@torch.no_grad()
def run_robustness_suite(
    model: nn.Module,
    test_df: pd.DataFrame,
    dataset_config: Dict[str, Any],
    class_names: Sequence[str],
    device: torch.device,
    batch_size: int = 32,
    max_samples: int = 1000,
    seed: int = 42,
    project_root: Path = PROJECT_ROOT,
) -> Dict[str, Any]:
    """Run clean baseline + each perturbation separately and measure performance degradation."""
    if len(test_df) > max_samples:
        eval_df = test_df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    else:
        eval_df = test_df.reset_index(drop=True)

    class_to_idx = {c: i for i, c in enumerate(class_names)}
    eval_tf = get_eval_transform(dataset_config)
    model = model.to(device)
    model.eval()

    def _run_condition(perturb_fn: Optional[Callable[[Image.Image], Image.Image]]) -> Dict[str, float]:
        ds = _PerturbedEvalDataset(
            dataframe=eval_df,
            class_to_idx=class_to_idx,
            eval_transform=eval_tf,
            perturbation_fn=perturb_fn,
            project_root=project_root,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        logits_list, targets_list = [], []
        for imgs, tgts in loader:
            logits_list.append(model(imgs.to(device)).detach().cpu().numpy())
            targets_list.extend(tgts.tolist())
        logits_np = np.concatenate(logits_list, axis=0)
        probs_np = softmax_numpy(logits_np, temperature=1.0)
        preds_np = np.argmax(probs_np, axis=1)
        m = compute_multiclass_metrics(targets_list, preds_np.tolist(), probs_np, class_names)
        return {
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "mean_confidence": round(float(np.mean(np.max(probs_np, axis=1))), 6),
        }

    clean = _run_condition(None)
    rows = []
    for name, fn in get_robustness_perturbations(seed=seed).items():
        res = _run_condition(fn)
        rows.append(
            {
                "perturbation": name,
                "accuracy": res["accuracy"],
                "macro_f1": res["macro_f1"],
                "mean_confidence": res["mean_confidence"],
                "accuracy_drop": round(clean["accuracy"] - res["accuracy"], 6),
                "macro_f1_drop": round(clean["macro_f1"] - res["macro_f1"], 6),
            }
        )

    return {
        "samples_evaluated": int(len(eval_df)),
        "original_files_modified": False,
        "clean_baseline": clean,
        "perturbations": rows,
    }
