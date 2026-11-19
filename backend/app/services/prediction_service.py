"""End-to-end prediction orchestration service (Phase 7).

Coordinates image validation -> Phase 2 preprocessing -> AgriVision-CNN inference ->
top-K ranking -> uncertainty rejection policy -> optional Grad-CAM -> disease info ->
SQLite history logging -> sanitized response construction.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import UploadFile
import numpy as np
from sqlalchemy.orm import Session

from ..core.config import Settings
from ..core.exceptions import ModelUnavailableError
from ..schemas.prediction import PredictionResponse
from ..utils.response_builder import build_prediction_response
from .disease_information_service import DiseaseInformationService
from .explanation_service import ExplanationService
from .history_service import HistoryService
from .image_service import ImageService
from .model_service import ModelService
from .uncertainty_service import UncertaintyService


class PredictionService:
    """Orchestrates the complete /api/v1/predict request lifecycle."""

    def __init__(
        self,
        settings: Settings,
        model_service: ModelService,
        image_service: ImageService,
        uncertainty_service: UncertaintyService,
        explanation_service: ExplanationService,
        disease_info_service: DiseaseInformationService,
        history_service: HistoryService,
    ) -> None:
        self.settings = settings
        self.model_service = model_service
        self.image_service = image_service
        self.uncertainty_service = uncertainty_service
        self.explanation_service = explanation_service
        self.disease_info_service = disease_info_service
        self.history_service = history_service

    async def predict(
        self,
        upload: UploadFile,
        request_id: str,
        db: Session,
        generate_explanation: bool = False,
    ) -> PredictionResponse:
        # 1. Fail fast with HTTP 503 if no verified trained model is loaded
        if not self.model_service.is_ready:
            raise ModelUnavailableError(self.model_service.status_detail)

        # 2. Validate & decode uploaded image safely
        decoded = await self.image_service.process_upload(upload, request_id=request_id)

        # 3. Apply exact Phase 2 deterministic preprocessing & run model inference
        input_tensor = self.model_service.preprocess_image(decoded.rgb_image)
        _, probs = self.model_service.predict_tensor(input_tensor)

        # 4. Evaluate uncertainty & rejection policy
        uncertainty = self.uncertainty_service.evaluate(probs)

        checkpoint_id = self.model_service.checkpoint_id or "unavailable"
        model_name = self.model_service.model_name
        device_type = self.model_service.device.type

        # 5. If rejected due to uncertainty, NEVER claim a disease diagnosis or treatment
        if uncertainty["is_uncertain"]:
            record = self.history_service.record_prediction(
                db=db,
                request_id=request_id,
                status="uncertain",
                uncertainty=uncertainty,
                model_name=model_name,
                checkpoint_id=checkpoint_id,
                image_id=decoded.image_id,
                image_retained=self.settings.retain_uploaded_images,
                prediction=None,
                explanation=None,
            )
            return build_prediction_response(
                request_id=request_id,
                created_at=record.created_at or datetime.now(timezone.utc),
                model_name=model_name,
                checkpoint_id=checkpoint_id,
                inference_device=device_type,
                uncertainty=uncertainty,
                prediction=None,
                explanation=None,
                disease_info=None,
                api_prefix=self.settings.api_v1_prefix,
            )

        # 6. Build top-K verified predictions
        k = min(self.settings.top_k_predictions, len(self.model_service.class_names))
        top_indices = np.argsort(probs)[::-1][:k]
        top_predictions = []
        for idx in top_indices:
            cls_name = self.model_service.class_names[int(idx)]
            meta = self.model_service.get_class_metadata(cls_name)
            top_predictions.append(
                {
                    "class_name": meta["class_name"],
                    "display_name": meta["display_name"],
                    "plant_name": meta["plant_name"],
                    "disease_name": meta["disease_name"],
                    "is_healthy": meta["is_healthy"],
                    "confidence": round(float(probs[int(idx)]), 6),
                }
            )

        best_idx = int(top_indices[0])
        best_pred = top_predictions[0]
        prediction_payload = {
            "class_name": best_pred["class_name"],
            "display_name": best_pred["display_name"],
            "plant_name": best_pred["plant_name"],
            "disease_name": best_pred["disease_name"],
            "confidence": best_pred["confidence"],
            "is_healthy": best_pred["is_healthy"],
            "top_predictions": top_predictions,
        }

        # 7. Optional Grad-CAM explanation (failure-tolerant)
        if generate_explanation:
            explanation_payload = self.explanation_service.generate_gradcam(
                rgb_image=decoded.rgb_image,
                input_tensor=input_tensor,
                target_class_idx=best_idx,
                request_id=request_id,
            )
        else:
            explanation_payload = {
                "available": False,
                "requested": False,
                "method": "Grad-CAM",
                "heatmap_filename": None,
                "overlay_filename": None,
                "note": "Explanation was not requested. Pass generate_explanation=true to compute approximate Grad-CAM attention maps.",
            }

        # 8. Retrieve educational disease metadata
        disease_info = self.disease_info_service.get_information(
            canonical_class=best_pred["class_name"],
            plant_name=best_pred["plant_name"],
            disease_name=best_pred["disease_name"],
            is_healthy=best_pred["is_healthy"],
        )

        # 9. Persist history record
        record = self.history_service.record_prediction(
            db=db,
            request_id=request_id,
            status="success",
            uncertainty=uncertainty,
            model_name=model_name,
            checkpoint_id=checkpoint_id,
            image_id=decoded.image_id,
            image_retained=self.settings.retain_uploaded_images,
            prediction=prediction_payload,
            explanation=explanation_payload,
        )

        return build_prediction_response(
            request_id=request_id,
            created_at=record.created_at or datetime.now(timezone.utc),
            model_name=model_name,
            checkpoint_id=checkpoint_id,
            inference_device=device_type,
            uncertainty=uncertainty,
            prediction=prediction_payload,
            explanation=explanation_payload,
            disease_info=disease_info,
            api_prefix=self.settings.api_v1_prefix,
        )
