# Phase 9 Integration Plan — Full-Stack Integration, E2E Testing & Bug Fixing

**Project:** PlantDiseaseAI (AgriVision-CNN Research Platform)  
**Project Root:** `C:\Users\Mahesh Kumar Mahato\Desktop\Plant Disease Detection\PlantDiseaseAI`  
**Phase Scope:** Phase 9 ONLY — Full-Stack Audit, Integration, Bug Fixing, Performance Benchmarking & Final Acceptance Verification

---

## 1. Complete Project Audit (Phases 1–8)

| Phase | Inspected Components | Verified Status | Audit Findings |
|---|---|---|---|
| **Phase 1: Environment** | `.venv` (Python `3.11.18`), PyTorch `2.11.0+cu128`, CUDA RTX 5050 Laptop GPU (8 GB), Node.js `v24.19.0`, npm `11.17.0`, `.gitignore`, `ml/tests/test_gpu.py` | **Operational** | Both Python and Node environments are healthy and isolated. |
| **Phase 2: Dataset & Preprocessing** | `ml/configs/dataset_config.yaml`, `ml/configs/class_mapping.yaml` (38 canonical classes), `ml/src/preprocess.py`, `ml/data/processed/` (37,844 train / 8,127 val / 8,123 test / 2,554 PlantDoc) | **Complete** | Deterministic `LetterboxResize(224x224, fill=[124,116,104], BICUBIC) -> ToTensor -> Normalize(ImageNet)` verified. Canonical IDs use lowercase double-underscore format (`tomato__early_blight`). |
| **Phase 3: AgriVision-CNN** | `ml/configs/model_config.yaml`, `ml/src/models/{agrivision_cnn,residual_block,attention,simple_cnn,resnet_baseline,model_factory}.py` | **Complete** | `AgriVision-CNN` (`2,871,350` params) verified cleanly for 38 classes. |
| **Phase 4: Training** | `ml/src/training/*`, `ml/src/datasets/plant_dataset.py`, `ml/experiments/checkpoints/` | **Pipeline Ready / No Trained Checkpoint** | Training pipeline is verified, but **0 trained `.pth` checkpoints** exist in `ml/experiments/checkpoints/` (only `.gitkeep`) because full multi-epoch training requires explicit user approval. |
| **Phase 5: Evaluation** | `ml/src/evaluation/*`, `ml/reports/phase5_research_report.md` | **Pipeline Ready** | Evaluation scripts and hardware benchmarks exist; accuracy metrics await a trained Phase 4 checkpoint. |
| **Phase 6: Explainability & Uncertainty** | Integrated into Phase 7 `backend/app/services/{explanation_service,uncertainty_service}.py` | **Integrated in Phase 7** | No separate `ml/src/explainability/` folder exists; thread-safe Grad-CAM (`model_service.lock`, guaranteed hook removal in `finally:`, BGR→RGB conversion) and 3-signal uncertainty rejection (confidence < 0.60, entropy > 1.60, margin < 0.10) are implemented in Phase 7. Standalone OOD distance/energy scoring is not implemented. |
| **Phase 7: FastAPI Backend** | `backend/app/main.py`, routes (`/api/v1/{health,system/info,predict,explanations,history}`), SQLite (`backend/runtime/plantdiseaseai.db`), `backend/app/tests/` | **Operational (port 8001)** | Runs live on `http://127.0.0.1:8001`. Uses exact Phase 2 `get_eval_transform` (100% preprocessing parity). Returns `503 model_unavailable` cleanly until a real Phase 4 checkpoint exists. |
| **Phase 8: React Frontend** | `frontend/` (Vite 6, React 19, React Router v7, CSS design system, 25 Vitest unit tests, production `dist/`) | **Operational (port 5173)** | Runs live on `http://127.0.0.1:5173`. Configured with `VITE_API_BASE_URL=http://127.0.0.1:8001`. CORS verified with Phase 7 backend. |

---

## 2. Issues & Gaps Discovered During Audit

### Critical (Operational Blocker — Not a Code Defect)
- **ISSUE-01 (Critical Blocker):** No trained `AgriVision-CNN` checkpoint exists at `ml/experiments/checkpoints/exp_a_agrivision_cnn/best_model.pth`.
  - **Impact:** Live `POST /api/v1/predict` returns HTTP `503` (`model_unavailable`); real accuracy / real-leaf classification / live Grad-CAM generation against trained weights cannot be claimed until Phase 4 training is executed with user approval.
  - **Policy:** Never create fake checkpoints or silently run random weights in production. Test full pipeline mechanics (preprocessing parity, CPU/CUDA forward pass, Grad-CAM hooks, uncertainty rejection, history CRUD) inside isolated temporary pytest fixtures strictly labeled as synthetic pipeline tests.

### Medium Severity Code Bugs (Will Be Fixed in Phase 9)
- **BUG-01 (Medium — Frontend/Backend Canonical Class Formatting Mismatch):** `frontend/src/utils/formatters.js` (`formatClassLabel`) only parsed triple-underscore (`___`) class names, whereas Phase 2 `class_mapping.yaml` and Phase 7 SQLite history (`predicted_class`) use double-underscore lowercase canonical identifiers (e.g., `tomato__early_blight`, `corn__northern_leaf_blight`). Additionally, `HistoryTable.jsx` and `RecentScans.jsx` ignored `record.top_predictions[0]?.display_name` even when returned by the backend.
- **BUG-02 (Medium — Reopened History Detail Missing Disease Information):** `GET /api/v1/history/{request_id}` did not return `disease_information` for accepted (`status: "success"`) records even though `DiseaseInformationService` already holds the complete 38-class educational reference in memory, causing users reopening saved predictions from `/history` to see a missing-guidance placeholder. (Must remain strictly `null` for `uncertain` records.)
- **BUG-03 (Medium — Unhandled `SQLAlchemyError` Mapped to Generic 500):** Database connectivity/lock errors in `backend/app/core/exceptions.py` fell through to HTTP `500` instead of returning a sanitized HTTP `503` (`database_unavailable`) response.

### Low Severity UI Issues (Will Be Fixed in Phase 9)
- **BUG-04 (Low — Search Placeholder Mismatch):** `HistoryFilters.jsx` suggested `Early_blight` instead of canonical lowercase hints (`tomato`, `early_blight`, `healthy`).

### Environment / Tooling Constraints
- **ENV-01:** Neither Playwright (`playwright` Python/Node) nor an active Chrome DevTools remote debugging session (`chrome://inspect/#remote-debugging`) is available in this environment, and no physical mobile phone is attached. Browser automation and physical mobile camera tests will be honestly marked `BLOCKED` / `NOT TESTED` rather than fabricated.

---

## 3. Planned Phase 9 Verification & Bug-Fixing Procedure

1. **Fix Discovered Bugs (BUG-01 through BUG-04)** cleanly without breaking existing API contracts or modifying Phases 1–6 model/dataset assets.
2. **Add Comprehensive Phase 9 Integration & Verification Tests (`backend/app/tests/test_phase9_integration.py` + Frontend Vitest updates):**
   - **Preprocessing Parity Test:** Prove bit-for-bit tensor equality between Phase 2 `preprocess_image_tensor(..., mode="eval")` and Phase 7 `validate_and_decode_bytes` + `ModelService.preprocess_image` across RGB, RGBA, grayscale, non-square aspect ratios, and EXIF-rotated JPEGs.
   - **Grad-CAM Safety & Concurrency Test:** Verify heatmap & overlay dimensions (`224x224`), valid PNG headers, CORS header on `/api/v1/explanations/{filename}`, zero dangling forward/backward hooks, thread-safe concurrent Grad-CAM requests, and graceful degradation when Grad-CAM raises an internal error (primary prediction still returned & saved).
   - **Uncertainty & Edge-Case Input Test:** Test real PlantVillage healthy/diseased sample images plus synthetic blurry, very dark, and non-plant pattern images through preprocessing, uncertainty math (confidence, Shannon entropy, top-2 margin), and rejection policy (verifying zero fabricated diagnosis or treatment on uncertain responses).
   - **Security & Error Handling Test:** Verify corrupted image (`400`), unsupported type (`415`), oversized upload (`413`), missing file (`422`), missing checkpoint (`503`), simulated database unavailability (`503 database_unavailable` with zero leaked paths/stack traces), path traversal blocking (`404`), and unique `request_id` per prediction.
   - **History Consistency & Disease Info Test:** Verify `GET /api/v1/history/{request_id}` returns `disease_information` for accepted records and `null` for uncertain records, and verify frontend canonical class formatting + `display_name` rendering.
3. **Run All Automated Test Suites:**
   - Phase 1–5 ML pytest suite (`ml/tests/`)
   - Phase 7 + Phase 9 Backend pytest suite (`backend/app/tests/`)
   - Phase 8 + Phase 9 Frontend Vitest suite (`frontend/src/__tests__/`) + `npm run build`
4. **Live Full-Stack Verification & Performance Benchmarking:**
   - Measure real timings (no fabricated metrics!) on this exact workstation (CPU + NVIDIA RTX 5050 Laptop GPU): backend app startup time, `AgriVision-CNN` model build/load time, preprocessing time, single-image CPU & CUDA inference time, Grad-CAM generation time, total API response time, and process memory footprint.
   - Probe live servers (`http://127.0.0.1:8001` & `http://127.0.0.1:5173`) across all endpoints and CORS rules.
5. **Execute Final Acceptance Tests (TEST A through TEST O)** and document honest `PASS` / `FAIL` / `BLOCKED` / `NOT TESTED` outcomes in `docs/phase9_integration_report.md` and `docs/phase9_known_issues.md`.
