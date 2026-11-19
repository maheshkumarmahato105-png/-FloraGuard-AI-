"""Early stopping monitoring validation macro-F1 with state serialization for resumption."""

from __future__ import annotations

from typing import Any, Dict, Optional


class EarlyStopping:
    """Stops training when the monitored validation metric stops improving."""

    def __init__(
        self,
        patience: int = 8,
        min_delta: float = 1e-4,
        mode: str = "max",
        monitor: str = "val_macro_f1",
        enabled: bool = True,
    ) -> None:
        if patience <= 0:
            raise ValueError(f"patience must be positive, got {patience}")
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode}")

        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.monitor = monitor
        self.enabled = bool(enabled)

        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.counter: int = 0
        self.early_stopped: bool = False
        self.stopped_epoch: Optional[int] = None

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > (self.best_score + self.min_delta)
        return score < (self.best_score - self.min_delta)

    def step(self, score: float, epoch: int) -> tuple[bool, bool]:
        """Update early-stopping state.

        Returns:
            (is_best, should_stop)
        """
        is_best = self._is_improvement(score)
        if is_best:
            self.best_score = float(score)
            self.best_epoch = int(epoch)
            self.counter = 0
        else:
            self.counter += 1
            if self.enabled and self.counter >= self.patience:
                self.early_stopped = True
                self.stopped_epoch = int(epoch)

        return is_best, self.early_stopped

    def state_dict(self) -> Dict[str, Any]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "monitor": self.monitor,
            "enabled": self.enabled,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "early_stopped": self.early_stopped,
            "stopped_epoch": self.stopped_epoch,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.best_score = state.get("best_score")
        self.best_epoch = int(state.get("best_epoch", 0))
        self.counter = int(state.get("counter", 0))
        self.early_stopped = bool(state.get("early_stopped", False))
        self.stopped_epoch = state.get("stopped_epoch")
