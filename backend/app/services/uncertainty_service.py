"""Uncertainty quantification and rejection policy service (Phase 7).

Computes Shannon predictive entropy, top-1 vs. top-2 probability margin, and
maximum predicted probability. Rejects unreliable predictions as `uncertain`
rather than returning misleading disease diagnoses.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..core.config import Settings


class UncertaintyService:
    """Evaluates predictive uncertainty and applies configured rejection rules."""

    def __init__(self, settings: Settings) -> None:
        self.confidence_threshold = float(settings.confidence_threshold)
        self.entropy_threshold = float(settings.entropy_threshold)
        self.min_margin_threshold = float(settings.min_margin_threshold)

    def evaluate(self, probabilities: np.ndarray) -> Dict[str, Any]:
        """Compute entropy, top-2 margin, max probability, and rejection decision."""
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.ndim != 1 or len(probs) == 0:
            raise ValueError("probabilities must be a non-empty 1D array.")

        probs = np.clip(probs, 1e-12, 1.0)
        probs = probs / np.sum(probs)

        entropy = float(-np.sum(probs * np.log(probs)))
        sorted_probs = np.sort(probs)[::-1]
        max_prob = float(sorted_probs[0])
        second_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        margin = max(0.0, max_prob - second_prob)

        reasons: List[str] = []
        if max_prob < self.confidence_threshold:
            reasons.append(
                f"Top confidence ({max_prob:.3f}) is below minimum threshold ({self.confidence_threshold:.2f})."
            )
        if entropy > self.entropy_threshold:
            reasons.append(
                f"Predictive entropy ({entropy:.3f}) exceeds reliability limit ({self.entropy_threshold:.2f})."
            )
        if margin < self.min_margin_threshold:
            reasons.append(
                f"Top-2 probability margin ({margin:.3f}) is below minimum separation ({self.min_margin_threshold:.2f})."
            )

        return {
            "entropy": round(entropy, 6),
            "margin": round(margin, 6),
            "max_probability": round(max_prob, 6),
            "is_uncertain": len(reasons) > 0,
            "rejection_reasons": reasons,
        }
