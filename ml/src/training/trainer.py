"""End-to-end Trainer orchestrating training, validation, TensorBoard, OOM safety,
checkpointing, resumption, early stopping, charts, and Markdown reporting (Phase 4)."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .checkpoint import CheckpointManager, load_training_checkpoint
from .early_stopping import EarlyStopping
from .engine import evaluate_one_epoch, resolve_device, train_one_epoch
from .loss import build_criterion
from .metrics import MetricsTracker
from .optimizer import build_optimizer
from .scheduler import build_scheduler, step_scheduler


def collect_gpu_details(device: torch.device) -> Dict[str, Any]:
    """Collect verified runtime GPU/CPU hardware telemetry."""
    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        return {
            "device": str(device),
            "cuda_available": True,
            "gpu_name": torch.cuda.get_device_name(device),
            "compute_capability": f"sm_{props.major}{props.minor} ({props.major}.{props.minor})",
            "total_vram_gb": round(props.total_memory / (1024**3), 2),
            "torch_version": torch.__version__,
            "cuda_build": torch.version.cuda,
        }
    return {
        "device": "cpu",
        "cuda_available": False,
        "gpu_name": "None (CPU fallback)",
        "compute_capability": "N/A",
        "total_vram_gb": 0.0,
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
    }


def write_training_report(
    report_path: Path | str,
    summary: Dict[str, Any],
) -> Path:
    """Write ml/reports/training_report.md distinguishing actual results from configured targets."""
    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    status = summary.get("status", "pre_flight_verified_not_trained")
    gpu = summary.get("gpu_details", {})
    hp = summary.get("hyperparameters", {})
    ds = summary.get("dataset_info", {})
    ckpts = summary.get("checkpoints", {})
    es = summary.get("early_stopping", {})
    actual = summary.get("actual_results")

    lines = [
        "# PlantDiseaseAI — Phase 4 Model Training Report",
        "",
        f"- **Generated At (UTC):** `{summary.get('generated_at_utc', datetime.now(timezone.utc).isoformat())}`",
        f"- **Run ID / Experiment:** `{summary.get('run_name', 'N/A')}`",
        f"- **Pipeline Status:** **`{status}`**",
        "",
        "---",
        "",
        "## 1. Model & Dataset Configuration",
        "",
        f"- **Architecture Used:** `{summary.get('architecture_name', 'agrivision_cnn')}` (`{summary.get('total_params', 0):,}` total parameters)",
        f"- **Dataset Version:** `{ds.get('dataset_version', 'PlantVillage-Color-38-Classes (Phase 2 split_manifest.csv)')}`",
        f"- **Training Images (Measured from Phase 2 manifest):** `{ds.get('train_images', 0):,}`",
        f"- **Validation Images (Measured from Phase 2 manifest):** `{ds.get('val_images', 0):,}`",
        f"- **Held-Out Test Images (Strictly Untouched during Phase 4):** `{ds.get('test_images', 0):,}`",
        f"- **External Evaluation Set (PlantDoc — Strictly Untouched):** `{ds.get('external_plantdoc_images', 0):,}`",
        f"- **Number of Verified Classes:** `{ds.get('num_classes', 0)}`",
        "",
        "---",
        "",
        "## 2. Configured Hyperparameters (Targets)",
        "",
        "| Parameter | Configured Value |",
        "| :--- | :--- |",
        f"| Optimizer | `{hp.get('optimizer', 'adamw')}` |",
        f"| Initial Learning Rate | `{hp.get('lr', 0.001)}` |",
        f"| Weight Decay | `{hp.get('weight_decay', 0.0001)}` |",
        f"| LR Scheduler | `{hp.get('scheduler', 'cosine_annealing')}` |",
        f"| Loss Function | `{hp.get('loss', 'cross_entropy')}` |",
        f"| Class Weighting | `{hp.get('class_weighting', 'inverse_sqrt')}` |",
        f"| Label Smoothing | `{hp.get('label_smoothing', 0.0)}` |",
        f"| Batch Size (Train / Eval) | `{hp.get('batch_size', 16)}` / `{hp.get('eval_batch_size', 32)}` |",
        f"| Max Epochs | `{hp.get('epochs', 30)}` |",
        f"| Early Stopping | Monitor `{hp.get('early_stopping_monitor', 'val_macro_f1')}`, patience `{hp.get('early_stopping_patience', 8)}` |",
        f"| Mixed Precision (AMP) | `{hp.get('use_amp', True)}` |",
        f"| Gradient Clipping (`max_norm`) | `{hp.get('grad_clip_norm', 1.0)}` |",
        f"| Random Seed | `{hp.get('random_seed', 42)}` |",
        "",
        "---",
        "",
        "## 3. Hardware & GPU Telemetry (Measured)",
        "",
        f"- **Execution Device:** `{gpu.get('device', 'unknown')}`",
        f"- **CUDA Available:** `{gpu.get('cuda_available', False)}`",
        f"- **GPU Name:** `{gpu.get('gpu_name', 'Unknown')}`",
        f"- **Compute Capability:** `{gpu.get('compute_capability', 'N/A')}`",
        f"- **Total VRAM:** `{gpu.get('total_vram_gb', 0.0)} GB`",
        f"- **PyTorch / CUDA Build:** `{gpu.get('torch_version', '')}` / `{gpu.get('cuda_build', '')}`",
    ]

    if summary.get("preflight_gpu_probe"):
        probe = summary["preflight_gpu_probe"]
        lines.extend(
            [
                f"- **Single-Batch Pre-Flight Probe (`batch_size={probe.get('batch_size')}`, AMP=`{probe.get('use_amp')}`):**",
                f"  - Forward + Backward Loss: `{probe.get('sample_batch_loss')}` (finite=`{probe.get('finite')}`)",
                f"  - Peak GPU Memory Allocated: `{probe.get('peak_vram_mb')} MB`",
                f"  - Single-Batch Step Latency: `{probe.get('step_time_ms')} ms`",
            ]
        )

    lines.extend(["", "---", "", "## 4. Actual Training Results"])

    if actual and actual.get("epochs_completed", 0) > 0:
        lines.extend(
            [
                "",
                "| Metric | Measured Result |",
                "| :--- | ---: |",
                f"| **Epochs Completed** | `{actual['epochs_completed']}` / `{hp.get('epochs', 30)}` |",
                f"| **Best Validation Macro-F1** | **{actual['best_val_macro_f1'] * 100:.2f}%** (Epoch `{actual['best_epoch']}`) |",
                f"| **Best Validation Accuracy** | **{actual['best_val_accuracy'] * 100:.2f}%** |",
                f"| **Total Training Duration** | `{actual['total_duration_sec']:.2f} s` (`{actual['total_duration_sec'] / 60:.2f} min`) |",
                f"| **Early Stopping Status** | `{'Triggered at epoch ' + str(es.get('stopped_epoch')) if es.get('early_stopped') else 'Not triggered'}` |",
                f"| **Overfitting Diagnostic** | {actual.get('overfitting_summary', 'N/A')} |",
                "",
                "### Saved Checkpoints & Artifacts",
                f"- Best Model Checkpoint: `{ckpts.get('best_model_path', 'None')}`",
                f"- Latest Checkpoint: `{ckpts.get('latest_checkpoint_path', 'None')}`",
                f"- Epoch Metrics CSV: `{ckpts.get('metrics_csv_path', 'None')}`",
                f"- TensorBoard Log Dir: `{ckpts.get('tensorboard_log_dir', 'None')}`",
                f"- Training Curves: `{ckpts.get('overview_chart_path', 'None')}`",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "> **Full Training Not Yet Executed (Awaiting User Approval):** Per Phase 4 safety requirements, automated synthetic-dataset pipeline tests and a single-batch real-dataset/GPU pre-flight check have been verified, and **no full multi-epoch training run has been started automatically**. Consequently, **no trained checkpoints (`best_model.pth` / `latest_checkpoint.pth`) exist yet** and **no accuracy or F1 metrics are reported or fabricated**.",
                "",
                "- **Epochs Completed:** `0` (configured target: `" + str(hp.get("epochs", 30)) + "`)",
                "- **Best Validation Macro-F1:** *Pending approved training run*",
                "- **Best Validation Accuracy:** *Pending approved training run*",
                "- **Training Duration:** *Pending approved training run*",
                "- **Early Stopping Status:** Configured (`patience=" + str(hp.get("early_stopping_patience", 8)) + "`, monitor=`val_macro_f1`)",
                "- **Saved Checkpoints:** None created yet (no untrained weights saved as trained checkpoints).",
            ]
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Planned Controlled Experiments (`ml/configs/experiment_config.yaml`)",
            "",
            "| Experiment | Architecture | Initialization | Selection Metric |",
            "| :--- | :--- | :--- | :--- |",
            "| **Experiment A** | `agrivision_cnn` | From scratch (seed 42) | `val_macro_f1` |",
            "| **Experiment B** | `simple_cnn` | From scratch (seed 42) | `val_macro_f1` |",
            "| **Experiment C** | `resnet18_baseline` | From scratch (`weights=None`, seed 42) | `val_macro_f1` |",
            "",
            "---",
            "",
            "## 6. Training Limitations & Notes",
            "",
            "1. **Class Imbalance (36.23× in PlantVillage):** Addressed via bounded `inverse_sqrt` class weights (`[0.5, 3.0]`) inside `CrossEntropyLoss` without stacking a second conflicting sampler.",
            "2. **Hardware Budget (8 GB Laptop GPU):** Initial batch size is conservatively set to `16` with AMP enabled; automatic CUDA OOM handling saves an emergency checkpoint and suggests reducing batch size.",
            "3. **In-Domain vs. Real-World Domain Shift:** Validation metrics during Phase 4 reflect PlantVillage in-domain performance only; real-world generalization on `external_plantdoc` and held-out `test` evaluation are reserved for **Phase 5**.",
        ]
    )

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


class PlantDiseaseTrainer:
    """Complete reproducible trainer for AgriVision-CNN and baseline architectures."""

    def __init__(
        self,
        model: nn.Module,
        architecture_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_names: List[str],
        class_mapping: Dict[str, Any],
        train_class_counts: List[int],
        training_config: Dict[str, Any],
        model_config: Dict[str, Any],
        run_name: str,
        output_root: Path,
        resume_path: Optional[Path | str] = None,
        allow_overwrite: bool = False,
        dataset_info: Optional[Dict[str, Any]] = None,
        report_path: Optional[Path | str] = None,
        show_progress: bool = True,
    ) -> None:
        self.architecture_name = architecture_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = list(class_names)
        self.class_mapping = class_mapping
        self.training_config = training_config
        self.model_config = model_config
        self.run_name = run_name
        self.output_root = Path(output_root)
        self.show_progress = show_progress
        self.dataset_info = dataset_info or {}
        self.report_path = Path(report_path) if report_path else None

        train_cfg = training_config["training"]
        self.device = resolve_device(str(train_cfg.get("device", "auto")))
        self.use_amp = bool(train_cfg.get("use_amp", True)) and self.device.type == "cuda"
        self.grad_clip_norm = train_cfg.get("grad_clip_norm", 1.0)
        self.non_blocking = bool(train_cfg.get("non_blocking_transfer", True))
        self.total_epochs = int(train_cfg.get("epochs", 30))
        self.seed = int(training_config.get("seed", 42))

        self.model = model.to(self.device)
        self.criterion = build_criterion(
            training_config=training_config,
            device=self.device,
            train_class_counts=train_class_counts,
        )
        self.optimizer = build_optimizer(self.model, training_config)
        self.scheduler = build_scheduler(self.optimizer, training_config)
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        es_cfg = training_config.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=int(es_cfg.get("patience", 8)),
            min_delta=float(es_cfg.get("min_delta", 1e-4)),
            mode=str(es_cfg.get("mode", "max")),
            monitor=str(es_cfg.get("monitor", "val_macro_f1")),
            enabled=bool(es_cfg.get("enabled", True)),
        )

        # Output directories isolated per run
        self.checkpoint_dir = self.output_root / "checkpoints" / run_name
        self.log_dir = self.output_root / "logs" / run_name
        self.results_dir = self.output_root / "results" / run_name
        self.charts_dir = self.results_dir / "charts"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            allow_overwrite=allow_overwrite or bool(training_config.get("checkpointing", {}).get("allow_overwrite", False)),
            is_resuming=resume_path is not None,
        )

        self.start_epoch = 1
        self.best_val_macro_f1 = -1.0
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        self.tracker = MetricsTracker()

        if resume_path is not None:
            ckpt = load_training_checkpoint(
                checkpoint_path=resume_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                early_stopping=self.early_stopping,
                map_location=self.device,
            )
            self.start_epoch = int(ckpt["epoch"]) + 1
            self.best_val_macro_f1 = float(ckpt.get("best_val_macro_f1", -1.0))
            prior_history = ckpt.get("history_records", [])
            if prior_history:
                self.tracker = MetricsTracker(prior_history)
                self.best_val_accuracy = max(float(r["val_accuracy"]) for r in prior_history)
            self.best_epoch = int(self.early_stopping.best_epoch)

    def _checkpoint_kwargs(self, epoch: int) -> Dict[str, Any]:
        return {
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "scaler": self.scaler,
            "early_stopping_state": self.early_stopping.state_dict(),
            "epoch": epoch,
            "best_val_macro_f1": max(self.best_val_macro_f1, 0.0),
            "training_config": self.training_config,
            "model_config": self.model_config,
            "architecture_name": self.architecture_name,
            "class_names": self.class_names,
            "class_mapping": self.class_mapping,
            "random_seed": self.seed,
            "history_records": self.tracker.records,
        }

    def fit(self) -> Dict[str, Any]:
        """Execute full training & validation loop with OOM safety, TensorBoard, and reporting."""
        writer = SummaryWriter(log_dir=str(self.log_dir))
        train_start = time.perf_counter()
        completed_epoch = self.start_epoch - 1

        try:
            for epoch in range(self.start_epoch, self.total_epochs + 1):
                ep_t0 = time.perf_counter()
                current_lr = float(self.optimizer.param_groups[0]["lr"])

                try:
                    train_metrics = train_one_epoch(
                        model=self.model,
                        dataloader=self.train_loader,
                        criterion=self.criterion,
                        optimizer=self.optimizer,
                        device=self.device,
                        scaler=self.scaler,
                        use_amp=self.use_amp,
                        grad_clip_norm=self.grad_clip_norm,
                        non_blocking=self.non_blocking,
                        epoch_num=epoch,
                        total_epochs=self.total_epochs,
                        show_progress=self.show_progress,
                    )
                    val_metrics = evaluate_one_epoch(
                        model=self.model,
                        dataloader=self.val_loader,
                        criterion=self.criterion,
                        device=self.device,
                        use_amp=self.use_amp,
                        non_blocking=self.non_blocking,
                        epoch_num=epoch,
                        total_epochs=self.total_epochs,
                        split_label="Val",
                        show_progress=self.show_progress,
                    )
                except torch.cuda.OutOfMemoryError as oom_err:
                    torch.cuda.empty_cache()
                    cur_bs = int(self.training_config["data"]["batch_size"])
                    suggested_bs = max(2, cur_bs // 2)
                    saved_emergency = None
                    try:
                        saved_emergency = str(
                            self.checkpoint_manager.save_emergency(**self._checkpoint_kwargs(max(completed_epoch, 0)))
                        )
                    except Exception:
                        saved_emergency = None
                    raise RuntimeError(
                        f"CUDA Out-Of-Memory at epoch {epoch} (batch_size={cur_bs}). "
                        f"Emergency checkpoint: {saved_emergency}. "
                        f"Suggestion: Reduce batch size to --batch-size {suggested_bs} and resume via "
                        f"--resume \"{self.checkpoint_manager.latest_path}\"."
                    ) from oom_err

                step_scheduler(self.scheduler, monitor_metric=val_metrics["macro_f1"])
                epoch_duration = round(time.perf_counter() - ep_t0, 2)
                completed_epoch = epoch

                record = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_precision": val_metrics["macro_precision"],
                    "val_macro_recall": val_metrics["macro_recall"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "learning_rate": current_lr,
                    "epoch_duration_sec": epoch_duration,
                }
                self.tracker.add_epoch(record)

                # TensorBoard live monitoring
                writer.add_scalar("Loss/train", record["train_loss"], epoch)
                writer.add_scalar("Loss/val", record["val_loss"], epoch)
                writer.add_scalar("Accuracy/train", record["train_accuracy"], epoch)
                writer.add_scalar("Accuracy/val", record["val_accuracy"], epoch)
                writer.add_scalar("Validation/macro_precision", record["val_macro_precision"], epoch)
                writer.add_scalar("Validation/macro_recall", record["val_macro_recall"], epoch)
                writer.add_scalar("Validation/macro_f1", record["val_macro_f1"], epoch)
                writer.add_scalar("LearningRate", current_lr, epoch)
                writer.flush()

                # Select best model strictly by validation macro-F1
                is_best, should_stop = self.early_stopping.step(val_metrics["macro_f1"], epoch)
                if is_best:
                    self.best_val_macro_f1 = val_metrics["macro_f1"]
                    self.best_val_accuracy = max(self.best_val_accuracy, val_metrics["accuracy"])
                    self.best_epoch = epoch
                    self.checkpoint_manager.save_best(**self._checkpoint_kwargs(epoch))
                else:
                    self.best_val_accuracy = max(self.best_val_accuracy, val_metrics["accuracy"])

                # Always save latest checkpoint after each completed epoch
                self.checkpoint_manager.save_latest(**self._checkpoint_kwargs(epoch))

                print(
                    f"Epoch {epoch:02d}/{self.total_epochs:02d} ({epoch_duration:5.1f}s) | "
                    f"LR: {current_lr:.6f} | "
                    f"TrLoss: {record['train_loss']:.4f} TrAcc: {record['train_accuracy']*100:5.2f}% | "
                    f"VaLoss: {record['val_loss']:.4f} VaAcc: {record['val_accuracy']*100:5.2f}% "
                    f"VaMacroF1: {record['val_macro_f1']*100:5.2f}%"
                    + (" [BEST]" if is_best else "")
                )

                if should_stop:
                    print(
                        f"[EarlyStopping] No val_macro_f1 improvement for {self.early_stopping.patience} epochs. "
                        f"Stopped at epoch {epoch} (best epoch: {self.best_epoch})."
                    )
                    break
        finally:
            writer.close()

        total_duration = round(time.perf_counter() - train_start, 2)
        metrics_csv = self.tracker.save_csv(self.results_dir / "metrics.csv")
        charts = self.tracker.save_training_charts(self.charts_dir, run_prefix=self.run_name)
        overfit_info = self.tracker.detect_overfitting_warning()

        summary: Dict[str, Any] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": self.run_name,
            "status": "training_completed",
            "architecture_name": self.architecture_name,
            "total_params": int(sum(p.numel() for p in self.model.parameters())),
            "dataset_info": self.dataset_info,
            "hyperparameters": {
                "optimizer": self.training_config["optimizer"]["name"],
                "lr": self.training_config["optimizer"]["lr"],
                "weight_decay": self.training_config["optimizer"]["weight_decay"],
                "scheduler": self.training_config["scheduler"]["name"],
                "loss": self.training_config["loss"]["name"],
                "class_weighting": self.training_config["loss"]["class_weighting"],
                "label_smoothing": self.training_config["loss"]["label_smoothing"],
                "batch_size": self.training_config["data"]["batch_size"],
                "eval_batch_size": self.training_config["data"]["eval_batch_size"],
                "epochs": self.total_epochs,
                "early_stopping_monitor": self.early_stopping.monitor,
                "early_stopping_patience": self.early_stopping.patience,
                "use_amp": self.use_amp,
                "grad_clip_norm": self.grad_clip_norm,
                "random_seed": self.seed,
            },
            "gpu_details": collect_gpu_details(self.device),
            "early_stopping": self.early_stopping.state_dict(),
            "checkpoints": {
                "best_model_path": str(self.checkpoint_manager.best_path),
                "latest_checkpoint_path": str(self.checkpoint_manager.latest_path),
                "metrics_csv_path": str(metrics_csv),
                "tensorboard_log_dir": str(self.log_dir),
                "overview_chart_path": str(charts.get("overview_curves", "")),
            },
            "actual_results": {
                "epochs_completed": len(self.tracker.records),
                "best_epoch": self.best_epoch,
                "best_val_macro_f1": round(max(self.best_val_macro_f1, 0.0), 6),
                "best_val_accuracy": round(self.best_val_accuracy, 6),
                "total_duration_sec": total_duration,
                "overfitting_flagged": overfit_info["flagged"],
                "overfitting_summary": overfit_info["summary"],
            },
        }

        summary_json_path = self.results_dir / "training_summary.json"
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if self.report_path is not None:
            write_training_report(self.report_path, summary)

        return summary
