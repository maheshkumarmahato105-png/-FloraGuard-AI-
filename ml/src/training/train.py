"""CLI entrypoint for Phase 4 Training, Controlled Experiments, Resumption, and Pre-Flight Dry Run.

Examples:
    # Verify real dataset, model factory, and single-batch GPU VRAM without starting long training:
    python ml/src/training/train.py --experiment experiment_a_agrivision --dry-run

    # Launch Experiment A (AgriVision-CNN):
    python ml/src/training/train.py --experiment experiment_a_agrivision

    # Resume training from a saved checkpoint:
    python ml/src/training/train.py --experiment experiment_a_agrivision --resume ml/experiments/checkpoints/exp_a_agrivision_cnn/latest_checkpoint.pth
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional

import pandas as pd
import torch
import yaml

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_loader import PROJECT_ROOT, load_dataset_config
from datasets.plant_dataset import create_dataloaders
from models import (
    SUPPORTED_ARCHITECTURES,
    create_model,
    load_model_config,
    resolve_verified_classes,
    set_reproducible_seed,
)
from training.engine import resolve_device
from training.loss import build_criterion
from training.optimizer import build_optimizer
from training.trainer import PlantDiseaseTrainer, collect_gpu_details, write_training_report


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = deepcopy(v)
    return merged


def load_training_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else (PROJECT_ROOT / "ml" / "configs" / "training_config.yaml")
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_experiment_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else (PROJECT_ROOT / "ml" / "configs" / "experiment_config.yaml")
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_preflight_dry_run(
    architecture: str,
    run_name: str,
    training_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    model_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Verify real Phase 2 splits, model instantiation, and 1-batch GPU/CPU memory probe
    WITHOUT running multi-epoch training or saving fake checkpoints."""
    seed = int(training_config.get("seed", 42))
    set_reproducible_seed(seed)

    class_names, class_mapping = resolve_verified_classes(model_config)
    manifest_file = PROJECT_ROOT / training_config["paths"]["split_manifest"]
    manifest_df = pd.read_csv(manifest_file, low_memory=False)
    split_counts = manifest_df["assigned_split"].value_counts().to_dict()

    # Load DataLoaders with num_workers=0 for instant single-batch verification
    probe_train_cfg = deepcopy(training_config)
    probe_train_cfg["data"]["num_workers"] = 0
    loaders = create_dataloaders(
        dataset_config=dataset_config,
        training_config=probe_train_cfg,
        class_names=class_names,
        manifest_path=manifest_file,
        project_root=PROJECT_ROOT,
        include_test=False,
    )

    device = resolve_device(str(training_config["training"].get("device", "auto")))
    use_amp = bool(training_config["training"].get("use_amp", True)) and device.type == "cuda"
    gpu_details = collect_gpu_details(device)

    model = create_model(architecture=architecture, model_config=model_config, seed=seed).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    criterion = build_criterion(training_config, device, loaders["train_class_counts"])
    optimizer = build_optimizer(model, training_config)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Fetch 1 real training batch & 1 real validation batch
    tr_images, tr_targets = next(iter(loaders["train_loader"]))
    va_images, va_targets = next(iter(loaders["val_loader"]))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    model.train()
    tr_images = tr_images.to(device)
    tr_targets = tr_targets.to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
        logits = model(tr_images)
        loss = criterion(logits, tr_targets)
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    # Deterministic val batch forward check
    model.eval()
    with torch.no_grad():
        va_logits = model(va_images.to(device))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_vram_mb = round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
    else:
        peak_vram_mb = 0.0
    step_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    # Read dataset provenance if available
    ds_version = "PlantVillage-Color-38-Classes (Phase 2 split_manifest.csv)"
    prov_path = PROJECT_ROOT / training_config["paths"]["dataset_sources"]
    if prov_path.exists():
        try:
            with open(prov_path, "r", encoding="utf-8") as f:
                prov = json.load(f)
            pv_ver = prov.get("datasets", {}).get("plantvillage", {}).get("version")
            if pv_ver:
                ds_version = str(pv_ver)
        except Exception:
            pass

    summary: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "status": "pre_flight_verified_awaiting_training_approval",
        "architecture_name": architecture,
        "total_params": int(total_params),
        "dataset_info": {
            "dataset_version": ds_version,
            "train_images": int(loaders["train_size"]),
            "val_images": int(loaders["val_size"]),
            "test_images": int(split_counts.get("test", 0)),
            "external_plantdoc_images": int(split_counts.get("external_plantdoc", 0)),
            "num_classes": len(class_names),
        },
        "hyperparameters": {
            "optimizer": training_config["optimizer"]["name"],
            "lr": training_config["optimizer"]["lr"],
            "weight_decay": training_config["optimizer"]["weight_decay"],
            "scheduler": training_config["scheduler"]["name"],
            "loss": training_config["loss"]["name"],
            "class_weighting": training_config["loss"]["class_weighting"],
            "label_smoothing": training_config["loss"]["label_smoothing"],
            "batch_size": training_config["data"]["batch_size"],
            "eval_batch_size": training_config["data"]["eval_batch_size"],
            "epochs": training_config["training"]["epochs"],
            "early_stopping_monitor": training_config["early_stopping"]["monitor"],
            "early_stopping_patience": training_config["early_stopping"]["patience"],
            "use_amp": use_amp,
            "grad_clip_norm": training_config["training"]["grad_clip_norm"],
            "random_seed": seed,
        },
        "gpu_details": gpu_details,
        "preflight_gpu_probe": {
            "batch_size": int(tr_images.shape[0]),
            "train_batch_shape": list(tr_images.shape),
            "val_batch_shape": list(va_images.shape),
            "train_logits_shape": list(logits.shape),
            "val_logits_shape": list(va_logits.shape),
            "use_amp": use_amp,
            "sample_batch_loss": round(float(loss.item()), 4),
            "finite": bool(torch.isfinite(loss).item() and torch.isfinite(va_logits).all().item()),
            "peak_vram_mb": peak_vram_mb,
            "step_time_ms": step_ms,
        },
        "actual_results": None,
    }

    report_path = PROJECT_ROOT / training_config["paths"]["training_report"]
    write_training_report(report_path, summary)

    print("=== Phase 4 Pre-Flight Verification Complete (No Full Training Started) ===")
    print(f"Architecture:       {architecture} ({total_params:,} params)")
    print(f"Device / GPU:       {gpu_details['device']} ({gpu_details['gpu_name']}, {gpu_details['total_vram_gb']} GB)")
    print(f"Phase 2 Splits:     Train={loaders['train_size']:,} | Val={loaders['val_size']:,} | Test={split_counts.get('test', 0):,} (untouched) | PlantDoc={split_counts.get('external_plantdoc', 0):,}")
    print(f"Batch Probe:        train={list(tr_images.shape)} -> logits={list(logits.shape)}, loss={loss.item():.4f}")
    print(f"Peak GPU Memory:    {peak_vram_mb:.2f} MB (at batch_size={tr_images.shape[0]}, AMP={use_amp})")
    print(f"Report Updated:     {report_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AgriVision-CNN or baseline architectures (Phase 4).")
    parser.add_argument("--training-config", type=str, default=None)
    parser.add_argument("--experiment-config", type=str, default=None)
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment_a_agrivision",
        choices=["experiment_a_agrivision", "experiment_b_simple_cnn", "experiment_c_resnet18"],
    )
    parser.add_argument("--arch", type=str, default=None, choices=list(SUPPORTED_ARCHITECTURES))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training.")
    parser.add_argument("--overwrite", action="store_true", help="Confirm overwriting existing checkpoints.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify dataset availability, batch shapes, and single-batch GPU VRAM without starting training.",
    )
    args = parser.parse_args()

    training_cfg = load_training_config(args.training_config)
    exp_cfg_all = load_experiment_config(args.experiment_config)
    exp_entry = exp_cfg_all["experiments"][args.experiment]

    if "overrides" in exp_entry:
        training_cfg = _deep_merge(training_cfg, exp_entry["overrides"])

    architecture = args.arch or exp_entry["architecture"]
    run_name = args.run_name or exp_entry["id"]

    if args.epochs is not None:
        training_cfg["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        training_cfg["data"]["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        training_cfg["optimizer"]["lr"] = float(args.lr)
    if args.device is not None:
        training_cfg["training"]["device"] = args.device

    dataset_cfg = load_dataset_config(PROJECT_ROOT / training_cfg["paths"]["dataset_config"])
    model_cfg = load_model_config(PROJECT_ROOT / training_cfg["paths"]["model_config"])

    if args.dry_run:
        run_preflight_dry_run(
            architecture=architecture,
            run_name=run_name,
            training_config=training_cfg,
            dataset_config=dataset_cfg,
            model_config=model_cfg,
        )
        return

    seed = int(training_cfg.get("seed", 42))
    set_reproducible_seed(seed)
    class_names, class_mapping = resolve_verified_classes(model_cfg)

    manifest_file = PROJECT_ROOT / training_cfg["paths"]["split_manifest"]
    manifest_df = pd.read_csv(manifest_file, low_memory=False)
    split_counts = manifest_df["assigned_split"].value_counts().to_dict()

    loaders = create_dataloaders(
        dataset_config=dataset_cfg,
        training_config=training_cfg,
        class_names=class_names,
        manifest_path=manifest_file,
        project_root=PROJECT_ROOT,
        include_test=False,  # Test set NEVER used during training or model selection
    )

    model = create_model(architecture=architecture, model_config=model_cfg, seed=seed)
    trainer = PlantDiseaseTrainer(
        model=model,
        architecture_name=architecture,
        train_loader=loaders["train_loader"],
        val_loader=loaders["val_loader"],
        class_names=class_names,
        class_mapping=class_mapping,
        train_class_counts=loaders["train_class_counts"],
        training_config=training_cfg,
        model_config=model_cfg,
        run_name=run_name,
        output_root=PROJECT_ROOT / "ml" / "experiments",
        resume_path=args.resume,
        allow_overwrite=args.overwrite,
        dataset_info={
            "dataset_version": "PlantVillage-Color-38-Classes (Phase 2 split_manifest.csv)",
            "train_images": int(loaders["train_size"]),
            "val_images": int(loaders["val_size"]),
            "test_images": int(split_counts.get("test", 0)),
            "external_plantdoc_images": int(split_counts.get("external_plantdoc", 0)),
            "num_classes": len(class_names),
        },
        report_path=PROJECT_ROOT / training_cfg["paths"]["training_report"],
    )
    trainer.fit()


if __name__ == "__main__":
    main()
