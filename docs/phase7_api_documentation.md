# PlantDiseaseAI — Phase 7 REST API Documentation (`/api/v1`)

**Base URL (Local Development):** `http://127.0.0.1:8000`  
**Interactive Swagger UI:** `http://127.0.0.1:8000/docs`  
**OpenAPI JSON Schema:** `http://127.0.0.1:8000/openapi.json`

All responses include an `X-Request-ID` response header for end-to-end log tracing. Internal filesystem paths, environment variables, and raw stack traces are never exposed.

---

## 1. Health Check

- **Method & URL:** `GET /api/v1/health`
- **Purpose:** Verify API availability, SQLite database connectivity, and trained model readiness.
- **Status Codes:** `200 OK`

### Example Response (Checkpoint Not Yet Trained)
```json
{
  "status": "degraded",
  "api_status": "online",
  "database_status": "connected",
  "model_ready": false,
  "model_status_detail": "Prediction unavailable: no trained AgriVision-CNN checkpoint is configured or present. Complete Phase 4 model training and configure MODEL_CHECKPOINT_PATH.",
  "inference_device": "cuda",
  "app_version": "1.0.0"
}
```

---

## 2. System Technical Info

- **Method & URL:** `GET /api/v1/system/info`
- **Purpose:** Return safe model, format, upload limit, and explanation metadata for frontend clients.
- **Status Codes:** `200 OK`

### Example Response
```json
{
  "app_name": "PlantDiseaseAI — AgriVision-CNN Backend API",
  "app_version": "1.0.0",
  "model_name": "agrivision_cnn",
  "num_classes": 38,
  "device_type": "cuda",
  "checkpoint_available": false,
  "checkpoint_id": null,
  "supported_image_formats": ["JPEG", "JPG", "PNG", "WEBP"],
  "max_upload_size_bytes": 10485760,
  "explanation_available": false,
  "explanation_method": "Grad-CAM",
  "explanation_disclaimer": "Grad-CAM highlights approximate AI attention regions used for classification and is NOT a verified pixel-level disease segmentation mask.",
  "image_retention_enabled": false
}
```

---

## 3. Predict Plant Disease

- **Method & URL:** `POST /api/v1/predict`
- **Content-Type:** `multipart/form-data`
- **Query Parameters:**
  - `generate_explanation` (`boolean`, default `false`): When `true`, computes approximate Grad-CAM heatmap and overlay PNGs and returns safe API URLs.
- **Form Fields:**
  - `file` (required): Plant leaf image (`JPG`, `JPEG`, `PNG`, `WEBP`; max `10 MB`; validated via magic bytes + Pillow + OpenCV).
- **Status Codes:**
  - `200 OK`: Successful classification (`status: "success"`) OR reliable uncertainty rejection (`status: "uncertain"`)
  - `400 Bad Request`: Empty, corrupted, or invalid-dimension image
  - `413 Content Too Large`: File exceeds `MAX_UPLOAD_SIZE_BYTES` (default 10 MB)
  - `415 Unsupported Media Type`: Unsupported MIME type or decoded image format
  - `429 Too Many Requests`: Per-IP rate limit exceeded
  - `503 Service Unavailable`: No verified trained model checkpoint loaded

### Example Request (`curl`)
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/predict?generate_explanation=true" \
  -F "file=@leaf.jpg"
```

### Example Response (`status: "success"` — Schema Illustration Only)
```json
{
  "request_id": "a1b2c3d4e5f6...",
  "status": "success",
  "message": null,
  "prediction": {
    "class_name": "tomato__early_blight",
    "display_name": "Tomato — Early Blight",
    "plant_name": "Tomato",
    "disease_name": "Early Blight",
    "confidence": 0.945,
    "top_predictions": [],
    "is_healthy": false
  },
  "uncertainty": {
    "entropy": 0.21,
    "margin": 0.89,
    "max_probability": 0.945,
    "is_uncertain": false,
    "rejection_reasons": []
  },
  "explanation": {
    "available": true,
    "requested": true,
    "method": "Grad-CAM",
    "heatmap_url": "/api/v1/explanations/a1b2c3d4e5f6_heatmap.png",
    "overlay_url": "/api/v1/explanations/a1b2c3d4e5f6_overlay.png",
    "note": "Grad-CAM highlights approximate AI attention regions used for classification and is NOT a verified pixel-level disease segmentation mask."
  },
  "disease_information": {
    "disease_name": "Early Blight",
    "affected_plant": "Tomato",
    "is_healthy": false,
    "summary": "...",
    "symptoms": ["..."],
    "prevention": ["..."],
    "treatment": ["..."],
    "disclaimer": "General educational information only — not a professional agricultural diagnosis..."
  },
  "model_name": "agrivision_cnn",
  "checkpoint_id": "ckpt-...",
  "inference_device": "cuda",
  "created_at": "2026-11-18T12:00:00Z"
}
```

### Example Response (`status: "uncertain"` — Rejected Due to Uncertainty)
```json
{
  "request_id": "f6e5d4c3b2a1...",
  "status": "uncertain",
  "message": "The image could not be classified reliably. Please capture a clear, well-lit close-up of a single plant leaf.",
  "prediction": null,
  "uncertainty": {
    "entropy": 3.41,
    "margin": 0.02,
    "max_probability": 0.14,
    "is_uncertain": true,
    "rejection_reasons": [
      "Top confidence (0.140) is below minimum threshold (0.60)."
    ]
  },
  "explanation": null,
  "disease_information": null,
  "model_name": "agrivision_cnn",
  "checkpoint_id": "ckpt-...",
  "inference_device": "cuda",
  "created_at": "2026-11-18T12:01:00Z"
}
```

---

## 4. Serve Grad-CAM Explanation Artifact

- **Method & URL:** `GET /api/v1/explanations/{filename}`
- **Purpose:** Serve generated heatmap or overlay PNG strictly from the controlled `RESULTS_DIR` (blocks path traversal and non-PNG requests).
- **Status Codes:** `200 OK` (`image/png`), `404 Not Found`

---

## 5. List Prediction History

- **Method & URL:** `GET /api/v1/history`
- **Query Parameters:**
  - `limit` (`int`, `1..100`, default `20`)
  - `offset` (`int`, `>=0`, default `0`)
  - `status` (`string`, optional): `success` or `uncertain`
  - `predicted_class` (`string`, optional): Case-insensitive substring filter
- **Sorting:** Newest first (`created_at DESC`)
- **Status Codes:** `200 OK`

### Example Response
```json
{
  "total": 1,
  "limit": 20,
  "offset": 0,
  "items": [
    {
      "request_id": "a1b2c3d4e5f6...",
      "created_at": "2026-11-18T12:00:00Z",
      "status": "success",
      "predicted_class": "tomato__early_blight",
      "is_healthy": false,
      "confidence": 0.945,
      "top_predictions": [],
      "entropy": 0.21,
      "margin": 0.89,
      "is_uncertain": false,
      "rejection_reason": null,
      "model_name": "agrivision_cnn",
      "checkpoint_id": "ckpt-...",
      "image_id": "9876543210abcdef...",
      "explanation_available": true,
      "explanation_method": "Grad-CAM",
      "heatmap_url": "/api/v1/explanations/a1b2c3d4e5f6_heatmap.png",
      "overlay_url": "/api/v1/explanations/a1b2c3d4e5f6_overlay.png"
    }
  ]
}
```

---

## 6. Get Single History Record

- **Method & URL:** `GET /api/v1/history/{request_id}`
- **Status Codes:** `200 OK`, `404 Not Found`

---

## 7. Delete History Record

- **Method & URL:** `DELETE /api/v1/history/{request_id}`
- **Purpose:** Permanently delete a history record and clean up any associated Grad-CAM PNGs or retained image.
- **Status Codes:** `200 OK` (`{"request_id": "...", "deleted": true}`), `404 Not Found`

---

## Standard Error Response Format

```json
{
  "request_id": "uuid-hex",
  "status": "error",
  "error_code": "model_unavailable",
  "detail": "Clear user-safe message without internal paths or stack traces."
}
```
