# Phase 8 Implementation Plan — Complete React Frontend for PlantDiseaseAI

**Project:** PlantDiseaseAI (AgriVision-CNN Research Platform)  
**Location:** `C:\Users\Mahesh Kumar Mahato\Desktop\Plant Disease Detection\PlantDiseaseAI`  
**Phase Scope:** Phase 8 ONLY — React Frontend Development & Phase 7 Backend Integration  

---

## 1. Project Inspection Findings

### 1.1 Existing Frontend Directory (`PlantDiseaseAI/frontend/`)
- **Current Contents:** Only a placeholder `.gitkeep` file exists inside `PlantDiseaseAI/frontend/`.
- **Existing `package.json` / Framework / CSS:** None initialized yet inside `PlantDiseaseAI/frontend/`.
- **Local Toolchain Verified:**
  - Node.js `v24.19.0`
  - npm `11.17.0`
  - `.gitignore` at `PlantDiseaseAI/.gitignore` already excludes `frontend/node_modules/`, `frontend/dist/`, `frontend/build/`, `frontend/.vite/`, and `.env.*` (while keeping `.env.example`).
- **Selected Frontend Stack (clean, fast, zero-bloat):**
  - **Framework & Bundler:** React 19 + Vite 6 (JavaScript / `.jsx`, matching the requested file structure).
  - **Routing:** `react-router-dom` v7 (`BrowserRouter`, `Routes`, `Route`, `NavLink`, `useNavigate`, `useParams`, `useLocation`).
  - **Icons:** `lucide-react` (clean, crisp scientific/SaaS iconography).
  - **Styling:** Dedicated modular CSS design system (`src/styles/global.css`, `src/styles/components.css`) using CSS custom properties for exact color tokens, dark-first + light theme switching, glass surfaces, responsive grids, skeleton loaders, subtle transitions, and `prefers-reduced-motion` support.
  - **Testing:** Vitest + `@testing-library/react` + `@testing-library/jest-dom` + `@testing-library/user-event` + `jsdom`.

### 1.2 Inspected Phase 7 FastAPI Backend (`PlantDiseaseAI/backend/`)
Inspected `backend/app/main.py`, `backend/app/core/config.py`, `backend/app/api/routes/{health,system,prediction,history}.py`, and `backend/app/schemas/{common,prediction,history}.py`.

- **Active Backend URL:** `http://127.0.0.1:8001` (live Uvicorn instance started in Phase 7; configurable via `VITE_API_BASE_URL`).
- **CORS Configuration (`backend/app/core/config.py` & `main.py`):**
  - Default `CORS_ORIGINS`: `http://localhost:3000`, `http://127.0.0.1:3000`, `http://localhost:5173`, `http://127.0.0.1:5173`.
  - Allowed methods: `GET`, `POST`, `DELETE`, `OPTIONS`.
  - Allowed headers: `Content-Type`, `Accept`, `X-Request-ID`.
  - Vite's default port `5173` matches the Phase 7 CORS allowlist without modifying backend configuration.

### 1.3 Verified Backend API Endpoints & Response Schemas (`/api/v1`)

| Method & Endpoint | Query / Body | Verified Response Schema |
|---|---|---|
| `GET /api/v1/health` | — | `{ status ("healthy"|"degraded"|"unhealthy"), api_status, database_status, model_ready, model_status_detail, inference_device, app_version }` |
| `GET /api/v1/system/info` | — | `{ app_name, app_version, model_name, num_classes, device_type, checkpoint_available, checkpoint_id, supported_image_formats (["JPEG","JPG","PNG","WEBP"]), max_upload_size_bytes (10485760), explanation_available, explanation_method ("Grad-CAM"), explanation_disclaimer, image_retention_enabled }` |
| `POST /api/v1/predict` | `multipart/form-data` (`file`) + `?generate_explanation=true\|false` | **Success (`status: "success"`):** `{ request_id, status, message, prediction: { class_name, display_name, plant_name, disease_name, confidence (0..1), top_predictions[], is_healthy }, uncertainty: { entropy, margin, max_probability, is_uncertain, rejection_reasons[] }, explanation: { available, requested, method, heatmap_url, overlay_url, note }, disease_information: { disease_name, affected_plant, is_healthy, summary, symptoms[], prevention[], treatment[], disclaimer }, model_name, checkpoint_id, inference_device, created_at }`<br>**Uncertain (`status: "uncertain"`):** `prediction: null`, `explanation: null`, `disease_information: null`, populated `message` & `uncertainty`. |
| `GET /api/v1/explanations/{filename}` | Path param `filename` | `image/png` file served from controlled `backend/runtime/results/` directory |
| `GET /api/v1/history` | `?limit=20&offset=0&status=&predicted_class=` | `{ total, limit, offset, items: [HistoryRecordResponse] }` |
| `GET /api/v1/history/{request_id}` | Path param `request_id` | `{ request_id, created_at, status, predicted_class, is_healthy, confidence (0..1\|null), top_predictions[], entropy, margin, is_uncertain, rejection_reason, model_name, checkpoint_id, image_id, explanation_available, explanation_method, heatmap_url, overlay_url }` |
| `DELETE /api/v1/history/{request_id}` | Path param `request_id` | `{ request_id, deleted: true }` |

**Unified Error Schema (HTTP `400`, `404`, `413`, `415`, `422`, `429`, `500`, `503`):**
```json
{
  "request_id": "...",
  "status": "error",
  "error_code": "...",
  "detail": "..."
}
```

### 1.4 Current Backend Runtime Caveat & Missing/Nuanced API Functionality
Rather than modifying Phase 7 backend contracts unnecessarily, the frontend will cleanly accommodate these verified backend behaviors:

1. **No Trained Phase 4 Checkpoint Currently Present:**
   - `GET /api/v1/health` currently reports `status: "degraded"`, `model_ready: false`, and `POST /api/v1/predict` returns `503` (`error_code: "model_unavailable"`).
   - The frontend will surface this transparently in the status indicator, Dashboard banner, Detection page notice, and Settings page without pretending inference works.
2. **Confidence Scale (`0.0`–`1.0`):**
   - Backend returns confidence as a normalized probability in `[0, 1]`. Frontend formatters will convert to percentages cleanly (`(value * 100).toFixed(1)%`).
3. **History Detail vs. Live Prediction Response:**
   - `GET /api/v1/history/{request_id}` stores prediction metrics and Grad-CAM URLs, but does **not** persist `disease_information` or serve the original uploaded leaf image (since `RETAIN_UPLOADED_IMAGES=false` by default).
   - **Frontend Strategy:** When navigating to `/results/:requestId` immediately after `POST /api/v1/predict`, pass the full API payload and local image preview (data URL) via router state + `sessionStorage` cache (keyed by `request_id`). When opening an older record from `/history` or a direct URL, fetch `GET /api/v1/history/{request_id}` and display all persisted fields accurately, showing a clear, honest notice for fields not retained by the backend.
4. **History Sorting (`newest` / `oldest`):**
   - `GET /api/v1/history` returns records sorted newest-first (`created_at DESC`) with no `sort` query parameter.
   - **Frontend Strategy:** Request records from `/api/v1/history` with backend filtering (`status`, `predicted_class`, `limit`, `offset`), and apply client-side ascending/descending sort cleanly when the user selects "Oldest first".
5. **Approximate Attention Bounding Boxes (`LocalizationViewer.jsx`):**
   - Backend returns `heatmap_url` and `overlay_url` (Grad-CAM PNGs), not pixel-level segmentation masks or bounding-box JSON coordinates.
   - **Frontend Strategy:** `LocalizationViewer.jsx` renders the overlay/original view and computes approximate high-attention bounding boxes client-side from the Grad-CAM heatmap pixels via `<canvas>` (with graceful fallback if cross-origin/load fails), explicitly labeled as **approximate AI attention regions** and never as ground-truth disease segmentation.
6. **Dashboard Aggregate Statistics:**
   - No dedicated `/api/v1/stats` endpoint exists. Dashboard fetches `GET /api/v1/history` (`limit=100`, plus status-filtered total counts via `GET /api/v1/history?limit=1&status=uncertain` and `status=success`) to compute real Total Scans, Healthy Detections, Disease Detections, and Uncertain Predictions strictly from backend records—displaying zeros and an empty state when no records exist.

---

## 2. Design System & Visual Architecture

- **Theme Philosophy:** Apple-inspired clarity + modern scientific AI SaaS workbench. Dark-first palette with full Light Theme toggle persisted in `localStorage`.
- **Dark Theme Tokens (exact match to specification):**
  - Primary background: `#0A0F0D`
  - Secondary background: `#111A15`
  - Card background: `#16221B`
  - Primary green accent: `#4ADE80`
  - Secondary green accent: `#86EFAC`
  - Main text: `#F5F7F5`
  - Secondary text: `#A1ADA4`
  - Border: `#26372C`
- **Light Theme Tokens:** High-contrast botanical alabaster surfaces (`#F6F9F7`, `#EDF3EF`, `#FFFFFF`, `#15803D`, `#16A34A`, `#0F1A14`, `#4B5C52`, `#D5E2DA`) maintaining WCAG AA contrast.
- **Typography:** Clean geometric sans-serif (`Inter`, system-ui fallback) with tabular numerals (`font-variant-numeric: tabular-nums`) for confidence scores, entropy, and margins, plus monospace for Request IDs and API URLs.
- **Responsive Breakpoints:** Verified across `360px`, `390px`, `768px`, `1024px`, and `1440px` with zero horizontal overflow, adaptive desktop sidebar + mobile bottom navigation bar, stacked mobile cards for history tables, and viewport-constrained camera previews.

---

## 3. Planned Frontend Directory & Component Structure

```text
PlantDiseaseAI/frontend/
├── .env.example
├── .env                         # local gitignored config pointing to http://127.0.0.1:8001
├── index.html
├── package.json
├── vite.config.js
└── src/
    ├── main.jsx
    ├── App.jsx
    ├── styles/
    │   ├── global.css
    │   └── components.css
    ├── context/
    │   ├── ThemeContext.jsx
    │   └── ToastContext.jsx
    ├── utils/
    │   ├── formatters.js
    │   └── validators.js
    ├── services/
    │   ├── api.js
    │   ├── predictionService.js
    │   ├── historyService.js
    │   └── systemService.js
    ├── hooks/
    │   ├── useBackendHealth.js
    │   └── usePrediction.js
    ├── components/
    │   ├── layout/
    │   │   ├── Sidebar.jsx
    │   │   ├── Navbar.jsx
    │   │   ├── MobileNavigation.jsx
    │   │   └── PageContainer.jsx
    │   ├── common/
    │   │   ├── LoadingSpinner.jsx
    │   │   ├── ErrorMessage.jsx
    │   │   ├── EmptyState.jsx
    │   │   ├── StatusIndicator.jsx
    │   │   ├── ConfirmModal.jsx
    │   │   └── ToastContainer.jsx
    │   ├── dashboard/
    │   │   ├── DashboardStats.jsx
    │   │   ├── RecentScans.jsx
    │   │   └── QuickActions.jsx
    │   ├── detection/
    │   │   ├── ImageUploader.jsx
    │   │   ├── CameraCapture.jsx
    │   │   ├── ImagePreview.jsx
    │   │   └── DetectionProgress.jsx
    │   ├── results/
    │   │   ├── PredictionCard.jsx
    │   │   ├── ConfidenceDisplay.jsx
    │   │   ├── TopPredictions.jsx
    │   │   ├── DiseaseInformation.jsx
    │   │   ├── GradCAMViewer.jsx
    │   │   └── LocalizationViewer.jsx
    │   └── history/
    │       ├── HistoryFilters.jsx
    │       └── HistoryTable.jsx
    ├── pages/
    │   ├── Dashboard.jsx
    │   ├── Detection.jsx
    │   ├── Results.jsx
    │   ├── History.jsx
    │   ├── Settings.jsx
    │   └── NotFound.jsx
    └── __tests__/
        ├── setup.js
        ├── AppRouting.test.jsx
        ├── ValidatorsAndApi.test.js
        ├── DetectionFlow.test.jsx
        ├── CameraCapture.test.jsx
        ├── ResultsView.test.jsx
        └── HistoryAndSettings.test.jsx
```

---

## 4. Implementation Milestones

1. **Scaffold Project & Dependencies:** Configure `package.json`, `vite.config.js`, `index.html`, `.env.example`, and install packages.
2. **Design System & Core Infrastructure:** Build `global.css`, `components.css`, `ThemeContext`, `ToastContext`, `formatters`, `validators`, `api.js` (XHR upload progress + `fetch` timeout + error normalization), service modules, and hooks.
3. **Layout & Common Components:** Implement `Sidebar`, `Navbar`, `MobileNavigation`, `PageContainer`, `LoadingSpinner`, `ErrorMessage`, `EmptyState`, `StatusIndicator`, `ConfirmModal`, and `ToastContainer`.
4. **Feature Components & Pages:** Implement Dashboard, Detection (Uploader + Live Camera + Preview + Progress), Results (Prediction + Confidence + Top-3 + Disease Info + Grad-CAM + Attention Localization), History (Filters + Table/Cards + Confirm Delete), Settings, and NotFound.
5. **Automated Unit Testing (Mocked API):** Run Vitest suite covering rendering, routing, validation, preview, camera errors/track cleanup, API errors (`400`/`413`/`415`/`422`/`503`/timeout/offline), prediction submission, success vs. uncertain rendering, history loading/filtering/deletion, empty/error states, and responsive layout structure.
6. **Production Build & Live Backend Integration Verification:** Run `npm run build`, verify CORS and real endpoint responses against the live Phase 7 backend on `http://127.0.0.1:8001`, and verify responsive rendering across viewports (`360px`, `390px`, `768px`, `1024px`, `1440px`).
7. **Documentation:** Write `docs/phase8_frontend_documentation.md` and `docs/phase8_completion_report.md`.
