# Phase 9 Known Issues & Recommended Fixes

**Project:** PlantDiseaseAI  
**Date:** 2026-11-18  

---

## 1. Critical Unresolved Issues (Operational Blockers)

### KI-01 — No Trained Phase 4 `AgriVision-CNN` Checkpoint Present
- **Severity:** Critical (Operational Blocker for Real Predictions)
- **Location:** `ml/experiments/checkpoints/exp_a_agrivision_cnn/best_model.pth` (directory currently contains only `.gitkeep`)
- **Cause:** Full multi-epoch Phase 4 training was intentionally paused pending explicit user approval; Phase 9 strictly forbids retraining without permission or creating fake checkpoints.
- **Observed Impact:**
  - `GET /api/v1/health` returns `status: "degraded"`, `model_ready: false`.
  - `POST /api/v1/predict` returns HTTP `503` (`error_code: "model_unavailable"`).
  - Acceptance Tests **TEST E** and **TEST F** remain `BLOCKED` for live real-leaf predictions.
- **Recommended Fix:**
  1. Run Phase 4 training on the prepared PlantVillage dataset (`ml/src/training/train.py`) to produce a verified `best_model.pth` checkpoint.
  2. Configure `MODEL_CHECKPOINT_PATH` in `backend/.env` (or place the checkpoint at the default path `ml/experiments/checkpoints/exp_a_agrivision_cnn/best_model.pth`).
  3. Restart the Phase 7 FastAPI backend and re-run **TEST E** and **TEST F** with real healthy, diseased, blurry, dark, and non-plant photographs.

---

## 2. Medium Unresolved Issues (Verification & Feature Gaps)

### KI-02 — Automated Multi-Viewport Browser E2E Suite Not Executed
- **Severity:** Medium
- **Location:** Local workstation tooling (Playwright not installed; Chrome remote debugging `DevToolsActivePort` not enabled)
- **Cause:** Neither Playwright nor an active Chrome DevTools remote debugging instance was available during Phase 9 execution.
- **Observed Impact:** Acceptance Test **TEST N** is recorded as `BLOCKED` for visual pixel-level viewport verification across `360px`, `390px`, `768px`, `1024px`, and `1440px` (code/CSS/unit-test verification passed).
- **Recommended Fix:** Install Playwright (`npm --prefix frontend install -D @playwright/test && npx playwright install chromium`) or enable Chrome remote debugging (`chrome://inspect/#remote-debugging`) to run automated visual viewport assertions and capture real screenshots.

### KI-03 — Physical Webcam & Mobile Device Camera Capture Not Executed
- **Severity:** Medium
- **Location:** Live browser hardware / physical mobile device testing (`CameraCapture.jsx`)
- **Cause:** No interactive webcam browser session or physical mobile device was connected during Phase 9 verification.
- **Observed Impact:** Acceptance Test **TEST O** is recorded as `NOT TESTED` for live hardware capture (unit tests verified opt-in activation, error states, upload fallback, and `track.stop()` cleanup).
- **Recommended Fix:** Open `http://127.0.0.1:5173/detection?mode=camera` in a desktop browser with a webcam (and over HTTPS on a mobile phone) to test Start Camera, front/rear camera switching, capture, retake, and track release end-to-end.

### KI-04 — Dedicated OOD Detector Beyond 3-Signal Uncertainty Policy
- **Severity:** Medium (Research Enhancement)
- **Location:** `backend/app/services/uncertainty_service.py`
- **Cause:** Out-of-distribution rejection currently relies on softmax confidence (`< 0.60`), Shannon entropy (`> 1.60`), and top-2 margin (`< 0.10`), rather than a dedicated feature-space Mahalanobis or energy-based OOD detector.
- **Recommended Fix:** After Phase 4 training completes, calibrate temperature scaling on the validation set (never the final test set) and optionally add an energy-based OOD score benchmarked against PlantDoc and non-plant validation images.

---

## 3. Low Unresolved Issues

### KI-05 — Server-Side History Sort Order Parameter
- **Severity:** Low
- **Location:** `backend/app/api/routes/history.py` & `frontend/src/pages/History.jsx`
- **Cause:** `GET /api/v1/history` always returns records ordered newest-first (`created_at DESC`) with no `sort` query parameter, so selecting "Oldest first" in the frontend reorders only the currently loaded page client-side.
- **Recommended Fix:** Add an optional `sort: Literal["newest", "oldest"] = "newest"` query parameter to `GET /api/v1/history` in a future minor backend update if multi-page oldest-first browsing is needed.

### KI-06 — Original Uploaded Images Not Retained on Reopened History Records
- **Severity:** Low (Intentional Privacy Default)
- **Location:** `backend/app/core/config.py` (`RETAIN_UPLOADED_IMAGES=false`) & `frontend/src/pages/Results.jsx`
- **Cause:** By default, the backend deletes raw uploaded images after inference for privacy. Fresh predictions show the original preview via `sessionStorage`, whereas reopening older scans from `/history` in a new session shows the Grad-CAM overlay/heatmap (if generated) and a clear notice that the raw upload was not retained.
- **Recommended Fix:** Keep disabled by default for privacy; if persistence of original images is desired in a future release, add an authenticated/controlled endpoint when `RETAIN_UPLOADED_IMAGES=true`.

### KI-07 — Starlette `TestClient` `httpx` Deprecation Warning in Backend Pytest
- **Severity:** Low
- **Location:** Third-party dependency (`fastapi`/`starlette` + `httpx`)
- **Cause:** Installed `starlette` emits `StarletteDeprecationWarning: Using httpx with starlette.testclient is deprecated; install httpx2 instead` during pytest collection.
- **Recommended Fix:** Upgrade FastAPI/Starlette/httpx together in a future dependency maintenance cycle; does not affect production Uvicorn runtime.
