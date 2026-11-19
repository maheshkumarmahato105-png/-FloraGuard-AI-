"""Phase 4 modular PyTorch training, optimization, checkpointing, and monitoring package."""

from .checkpoint import CheckpointManager, CheckpointOverwriteError, load_training_checkpoint
from .early_stopping import EarlyStopping
from .engine import evaluate_one_epoch, resolve_device, train_one_epoch
from .loss import build_criterion, compute_class_weights
from .metrics import MetricsTracker, compute_classification_metrics
from .optimizer import build_optimizer
from .scheduler import build_scheduler, step_scheduler
from .trainer import PlantDiseaseTrainer, collect_gpu_details, write_training_report

__all__ = [
    "CheckpointManager",
    "CheckpointOverwriteError",
    "load_training_checkpoint",
    "EarlyStopping",
    "evaluate_one_epoch",
    "resolve_device",
    "train_one_epoch",
    "build_criterion",
    "compute_class_weights",
    "MetricsTracker",
    "compute_classification_metrics",
    "build_optimizer",
    "build_scheduler",
    "step_scheduler",
    "PlantDiseaseTrainer",
    "collect_gpu_details",
    "write_training_report",
]
