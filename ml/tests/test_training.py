"""Automated synthetic-dataset pipeline tests for Phase 4 training & optimization.

Runs completely independently of the full 54k PlantVillage dataset using tiny
in-memory/temporary synthetic images inside pytest's `tmp_path`, verifying:
  - Manifest-based Dataset & DataLoader creation, batch shapes, and label mapping
  - Forward pass, CrossEntropyLoss backprop, and verifiable optimizer weight updates
  - Class-weighted loss & guard against conflicting imbalance methods
  - Checkpoint saving (`latest_checkpoint.pth`, `best_model.pth`), overwrite protection,
    loading, and exact epoch-to-epoch training resumption
  - Early stopping trigger & state restoration
  - CSV metrics, TensorBoard event files, visualization charts, and report generation
  - Both CPU and CUDA GPU (AMP + GradScaler) compatibility
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_SRC = PROJECT_ROOT / "ml" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

from config_loader import load_dataset_config
from datasets.plant_dataset import create_dataloaders
from models import SUPPORTED_ARCHITECTURES, create_model, load_model_config
from training import (
    CheckpointOverwriteError,
    EarlyStopping,
    PlantDiseaseTrainer,
    build_criterion,
    build_optimizer,
)
from training.train import load_training_config


def _create_synthetic_dataset(tmp_path: Path, num_classes: int = 3, train_per_cls: int = 4, val_per_cls: int = 2):
    """Create a tiny deterministic synthetic image dataset + Phase 2-compatible split_manifest.csv."""
    class_names = [f"synth_plant__cond_{i}" for i in range(num_classes)]
    rows = []
    rng = np.random.default_rng(42)

    def _make_img(path: Path, base_color: tuple[int, int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        noise = rng.integers(0, 35, size=(128, 128, 3), dtype=np.uint8)
        arr = np.clip(np.array(base_color, dtype=np.uint16) + noise, 0, 255).astype(np.uint8)
        Image.fromarray(arr, mode="RGB").save(path, format="JPEG", quality=90)

    colors = [(30, 160, 40), (170, 90, 30), (140, 140, 40), (60, 90, 170)]
    idx = 0
    for c_idx, cls_name in enumerate(class_names):
        col = colors[c_idx % len(colors)]
        for split_name, count in (("train", train_per_cls), ("validation", val_per_cls)):
            for _ in range(count):
                rel = Path("imgs") / split_name / cls_name / f"img_{idx:03d}.jpg"
                _make_img(tmp_path / rel, col)
                rows.append(
                    {
                        "image_filename": rel.name,
                        "image_path": rel.as_posix(),
                        "processed_path": rel.as_posix(),
                        "canonical_class": cls_name,
                        "assigned_split": split_name,
                    }
                )
                idx += 1

    manifest_path = tmp_path / "split_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path, class_names


def _make_synthetic_configs(device: str = "cpu", epochs: int = 2):
    dataset_cfg = deepcopy(load_dataset_config())
    model_cfg = deepcopy(load_model_config())
    train_cfg = deepcopy(load_training_config())

    train_cfg["data"]["batch_size"] = 4
    train_cfg["data"]["eval_batch_size"] = 4
    train_cfg["data"]["num_workers"] = 0
    train_cfg["training"]["epochs"] = epochs
    train_cfg["training"]["device"] = device
    train_cfg["training"]["use_amp"] = (device == "cuda")
    train_cfg["early_stopping"]["patience"] = 4
    return dataset_cfg, model_cfg, train_cfg


def test_synthetic_dataset_loading_batch_shapes_and_label_mapping(tmp_path: Path):
    manifest_path, class_names = _create_synthetic_dataset(tmp_path, num_classes=3)
    dataset_cfg, _, train_cfg = _make_synthetic_configs(device="cpu")

    loaders = create_dataloaders(
        dataset_config=dataset_cfg,
        training_config=train_cfg,
        class_names=class_names,
        manifest_path=manifest_path,
        project_root=tmp_path,
    )
    assert loaders["train_size"] == 12
    assert loaders["val_size"] == 6
    assert loaders["class_to_idx"] == {c: i for i, c in enumerate(class_names)}
    assert loaders["train_class_counts"] == [4, 4, 4]

    tr_x, tr_y = next(iter(loaders["train_loader"]))
    va_x, va_y = next(iter(loaders["val_loader"]))
    assert tr_x.shape == (4, 3, 224, 224)
    assert va_x.shape == (4, 3, 224, 224)
    assert tr_y.dtype == torch.long and int(tr_y.max()) < 3 and int(tr_y.min()) >= 0
    assert va_y.dtype == torch.long


def test_optimizer_updates_weights_and_loss_guardrail():
    _, model_cfg, train_cfg = _make_synthetic_configs(device="cpu")
    model = create_model(architecture="agrivision_cnn", model_config=model_cfg, num_classes=3, seed=42)
    criterion = build_criterion(train_cfg, torch.device("cpu"), train_class_counts=[10, 4, 2])
    optimizer = build_optimizer(model, train_cfg)

    before_weight = model.classifier.weight.detach().clone()
    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

    assert not torch.equal(before_weight, model.classifier.weight.detach()), (
        "Optimizer step did not update model weights!"
    )

    # Guardrail: combining class_weighting + use_weighted_sampler must raise ValueError
    bad_cfg = deepcopy(train_cfg)
    bad_cfg["loss"]["class_weighting"] = "inverse_sqrt"
    bad_cfg["data"]["use_weighted_sampler"] = True
    with pytest.raises(ValueError):
        build_criterion(bad_cfg, torch.device("cpu"), train_class_counts=[10, 4, 2])


def test_early_stopping_behavior_and_state_roundtrip():
    es = EarlyStopping(patience=2, min_delta=1e-4, mode="max")
    assert es.step(0.50, epoch=1) == (True, False)
    assert es.step(0.50, epoch=2) == (False, False)
    assert es.step(0.49, epoch=3) == (False, True)
    assert es.early_stopped is True
    assert es.best_epoch == 1
    assert es.stopped_epoch == 3

    es2 = EarlyStopping(patience=2, min_delta=1e-4, mode="max")
    es2.load_state_dict(es.state_dict())
    assert es2.best_score == es.best_score
    assert es2.stopped_epoch == 3


def test_end_to_end_cpu_train_checkpoint_overwrite_guard_and_resume(tmp_path: Path):
    """Run 2 synthetic epochs on CPU, verify checkpoints/CSV/charts/TensorBoard/report,
    verify overwrite guard, then resume cleanly to epoch 3."""
    manifest_path, class_names = _create_synthetic_dataset(tmp_path, num_classes=3)
    dataset_cfg, model_cfg, train_cfg = _make_synthetic_configs(device="cpu", epochs=2)

    loaders = create_dataloaders(
        dataset_config=dataset_cfg,
        training_config=train_cfg,
        class_names=class_names,
        manifest_path=manifest_path,
        project_root=tmp_path,
    )

    out_root = tmp_path / "experiments"
    report_file = tmp_path / "training_report.md"

    model = create_model(architecture="agrivision_cnn", model_config=model_cfg, num_classes=3, seed=42)
    trainer = PlantDiseaseTrainer(
        model=model,
        architecture_name="agrivision_cnn",
        train_loader=loaders["train_loader"],
        val_loader=loaders["val_loader"],
        class_names=class_names,
        class_mapping={"classes": class_names},
        train_class_counts=loaders["train_class_counts"],
        training_config=train_cfg,
        model_config=model_cfg,
        run_name="synth_cpu_run",
        output_root=out_root,
        report_path=report_file,
        show_progress=False,
    )
    summary = trainer.fit()

    assert summary["actual_results"]["epochs_completed"] == 2
    latest_ckpt = Path(summary["checkpoints"]["latest_checkpoint_path"])
    best_ckpt = Path(summary["checkpoints"]["best_model_path"])
    metrics_csv = Path(summary["checkpoints"]["metrics_csv_path"])
    tb_dir = Path(summary["checkpoints"]["tensorboard_log_dir"])
    chart_png = Path(summary["checkpoints"]["overview_chart_path"])

    assert latest_ckpt.exists() and best_ckpt.exists()
    assert metrics_csv.exists() and len(pd.read_csv(metrics_csv)) == 2
    assert chart_png.exists() and report_file.exists()
    assert any(tb_dir.glob("events.out.tfevents.*")), "TensorBoard event file not written"

    # Overwrite protection must block replacing existing checkpoints without allow_overwrite or resume
    with pytest.raises(CheckpointOverwriteError):
        PlantDiseaseTrainer(
            model=create_model(architecture="agrivision_cnn", model_config=model_cfg, num_classes=3, seed=42),
            architecture_name="agrivision_cnn",
            train_loader=loaders["train_loader"],
            val_loader=loaders["val_loader"],
            class_names=class_names,
            class_mapping={"classes": class_names},
            train_class_counts=loaders["train_class_counts"],
            training_config=train_cfg,
            model_config=model_cfg,
            run_name="synth_cpu_run",
            output_root=out_root,
            show_progress=False,
        )

    # Resume training from epoch 2 -> epoch 3
    resume_cfg = deepcopy(train_cfg)
    resume_cfg["training"]["epochs"] = 3
    resumed_trainer = PlantDiseaseTrainer(
        model=create_model(architecture="agrivision_cnn", model_config=model_cfg, num_classes=3, seed=99),
        architecture_name="agrivision_cnn",
        train_loader=loaders["train_loader"],
        val_loader=loaders["val_loader"],
        class_names=class_names,
        class_mapping={"classes": class_names},
        train_class_counts=loaders["train_class_counts"],
        training_config=resume_cfg,
        model_config=model_cfg,
        run_name="synth_cpu_run",
        output_root=out_root,
        resume_path=latest_ckpt,
        show_progress=False,
    )
    assert resumed_trainer.start_epoch == 3
    resumed_summary = resumed_trainer.fit()
    assert resumed_summary["actual_results"]["epochs_completed"] == 3
    assert len(pd.read_csv(metrics_csv)) == 3


@pytest.mark.parametrize("arch", SUPPORTED_ARCHITECTURES)
def test_gpu_amp_training_step_for_all_three_architectures(tmp_path: Path, arch: str):
    """Verify CUDA GPU + AMP GradScaler 1-epoch synthetic training for AgriVision-CNN,
    Simple CNN, and ResNet18 from scratch."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available in this environment.")

    manifest_path, class_names = _create_synthetic_dataset(tmp_path, num_classes=3, train_per_cls=4, val_per_cls=2)
    dataset_cfg, model_cfg, train_cfg = _make_synthetic_configs(device="cuda", epochs=1)

    loaders = create_dataloaders(
        dataset_config=dataset_cfg,
        training_config=train_cfg,
        class_names=class_names,
        manifest_path=manifest_path,
        project_root=tmp_path,
    )
    model = create_model(architecture=arch, model_config=model_cfg, num_classes=3, seed=42)
    trainer = PlantDiseaseTrainer(
        model=model,
        architecture_name=arch,
        train_loader=loaders["train_loader"],
        val_loader=loaders["val_loader"],
        class_names=class_names,
        class_mapping={"classes": class_names},
        train_class_counts=loaders["train_class_counts"],
        training_config=train_cfg,
        model_config=model_cfg,
        run_name=f"synth_gpu_{arch}",
        output_root=tmp_path / "experiments",
        show_progress=False,
    )
    summary = trainer.fit()
    assert summary["gpu_details"]["cuda_available"] is True
    assert summary["hyperparameters"]["use_amp"] is True
    assert summary["actual_results"]["epochs_completed"] == 1
