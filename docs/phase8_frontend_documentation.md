# Phase 8 — PlantDiseaseAI Frontend Documentation

**Project:** PlantDiseaseAI (AgriVision-CNN Research Platform)  
**Frontend Root:** `PlantDiseaseAI/frontend/`  
**Tech Stack:** React 19, Vite 6, React Router v7, Lucide React, CSS Custom Properties (Dark & Light Themes), Vitest + React Testing Library

---

## 1. Frontend Architecture

The Phase 8 frontend is a modular, single-page React application built with Vite and organized by clear domain boundaries:

```text
PlantDiseaseAI/frontend/
├── .env.example                 # Environment variable template
├── index.html                   # SPA mount entry + Inter & JetBrains Mono fonts
├── package.json                 # Verified npm scripts and dependencies
├── vite.config.js               # Vite dev server (127.0.0.1:5173) + Vitest jsdom config
└── src/
    ├── main.jsx                 # React root mounting + global stylesheets
    ├── App.jsx                  # ThemeProvider, ToastProvider, BrowserRouter, AppShell routes
    ├── styles/
    │   ├── global.css           # Exact palette tokens, light/dark themes, reduced-motion, app shell
    │   └── components.css       # Cards, navigation, dropzone, camera, XAI, modals, tables, toasts
    ├── context/
    │   ├── ThemeContext.jsx     # Dark (default) & Light theme state persisted in localStorage
    │   └── ToastContext.jsx     # Auto-dismissing accessible toast notifications
    ├── services/
    │   ├── api.js               # Central HTTP client (fetch timeout, XHR FormData upload progress, ApiError)
    │   ├── systemService.js     # GET /api/v1/health, GET /api/v1/system/info
    │   ├── predictionService.js # POST /api/v1/predict + sessionStorage result caching
    │   └── historyService.js    # GET/DELETE /api/v1/history + real dashboard aggregation
    ├── hooks/
    │   ├── useBackendHealth.js  # Live backend health & system metadata hook
    │   └── usePrediction.js     # Duplicate-guarded submission, upload progress, abort cleanup
    ├── utils/
    │   ├── formatters.js        # 0..1 confidence -> %, timestamps, bytes, class label formatter
    │   └── validators.js        # File type & max-size validation synced with backend system info
    ├── components/
    │   ├── layout/              # Sidebar, Navbar, MobileNavigation, PageContainer
    │   ├── common/              # LoadingSpinner, ErrorMessage, EmptyState, StatusIndicator, ConfirmModal, ToastContainer
    │   ├── dashboard/           # DashboardStats, RecentScans, QuickActions
    │   ├── detection/           # ImageUploader, CameraCapture, ImagePreview, DetectionProgress
    │   ├── results/             # PredictionCard, ConfidenceDisplay, TopPredictions, DiseaseInformation, GradCAMViewer, LocalizationViewer
    │   └── history/             # HistoryFilters, HistoryTable (desktop table + mobile card list)
    ├── pages/
    │   ├── Dashboard.jsx
    │   ├── Detection.jsx
    │   ├── Results.jsx
    │   ├── History.jsx
    │   ├── Settings.jsx
    │   └── NotFound.jsx
    └── __tests__/               # Vitest + React Testing Library unit test suite (mocked API)
```

### Design System & Accessibility
- **Dark Theme Palette (Default):** Primary `#0A0F0D`, Secondary `#111A15`, Card `#16221B`, Primary Green Accent `#4ADE80`, Secondary Green `#86EFAC`, Main Text `#F5F7F5`, Secondary Text `#A1ADA4`, Border `#26372C`.
- **Light Theme Support:** Accessible high-contrast light mode toggled via the top Navbar or Settings page (`[data-theme="light"]`).
- **Responsive Breakpoints:** Designed with zero horizontal overflow across `360px`, `390px`, `768px`, `1024px`, and `1440px`:
  - `>= 1024px`: Fixed left `Sidebar` + multi-column dashboard/results grids + full desktop `HistoryTable`.
  - `< 1024px`: `Sidebar` collapses automatically into bottom `MobileNavigation` bar, top `Navbar` displays brand mark, grids stack vertically.
  - `< 768px`: `HistoryTable` switches seamlessly from a horizontal table to stacked mobile cards.
- **Accessibility:** Skip-to-main link, semantic landmarks (`aside`, `header`, `nav`, `main`, `section`, `article`), `role="alert"` / `aria-live` status updates, keyboard-accessible dropzone and dialogs (`Escape` closes modals), visible `:focus-visible` outlines, and `@media (prefers-reduced-motion: reduce)` support.

---

## 2. Routes

| Path | Page Component | Purpose |
|---|---|---|
| `/` | `Dashboard.jsx` | Welcome header, real backend statistics, recent scans, quick actions, and live API/model status |
| `/detection` | `Detection.jsx` | Drag-and-drop file upload, opt-in live camera capture, image preview, Grad-CAM toggle, upload progress & inference submission |
| `/results/:requestId` | `Results.jsx` | Accepted diagnosis or uncertain rejection notice, confidence & uncertainty telemetry, top-3 predictions, Grad-CAM viewer, approximate AI attention region viewer, and agronomic disease guidance |
| `/history` | `History.jsx` | Searchable, filterable, sortable, paginated SQLite prediction history with view details & confirm-before-delete modal |
| `/settings` | `Settings.jsx` | Backend connection status, API base URL, safe model metadata, upload constraints, and Dark/Light theme selector |
| `*` | `NotFound.jsx` | Clean 404 recovery screen with navigation back to Dashboard and Detection |

---

## 3. Environment Configuration

Copy `frontend/.env.example` to `frontend/.env` and update as needed:

```dotenv
# Base URL of the PlantDiseaseAI FastAPI backend (no trailing slash)
VITE_API_BASE_URL=http://127.0.0.1:8001

# Request timeout in milliseconds
VITE_API_TIMEOUT_MS=30000
```

- Only public non-secret Vite variables (`VITE_*`) are used on the client.
- `frontend/.env` is excluded from Git by `PlantDiseaseAI/.gitignore`, while `frontend/.env.example` is committed.

---

## 4. Phase 7 Backend Integration

All backend calls flow through `src/services/api.js` and the domain services:

| Endpoint | Frontend Service | Notes |
|---|---|---|
| `GET /api/v1/health` | `systemService.getHealth()` | Checks `api_status`, `database_status`, `model_ready`, `inference_device`, and `model_status_detail` |
| `GET /api/v1/system/info` | `systemService.getSystemInfo()` | Syncs `max_upload_size_bytes`, `supported_image_formats`, `model_name`, `num_classes`, `device_type`, and Grad-CAM metadata |
| `POST /api/v1/predict?generate_explanation=true\|false` | `predictionService.submitPrediction()` | Uploads image via `FormData` + `XMLHttpRequest` for real upload progress without manually setting multipart boundaries |
| `GET /api/v1/explanations/{filename}` | `GradCAMViewer.jsx` / `LocalizationViewer.jsx` | Resolved against `VITE_API_BASE_URL` via `resolveApiUrl()` |
| `GET /api/v1/history` | `historyService.listHistory()` & `fetchDashboardSummary()` | Supports `limit`, `offset`, `status`, and `predicted_class` substring search |
| `GET /api/v1/history/{request_id}` | `historyService.getHistoryRecord()` | Loads persisted prediction details when navigating directly or from History |
| `DELETE /api/v1/history/{request_id}` | `historyService.deleteHistoryRecord()` | Triggered only after explicit confirmation in `ConfirmModal` |

### Error Handling Matrix
`src/services/api.js` normalizes all error responses into `ApiError` without leaking internal stack traces:
- **HTTP `400` / `422`:** Invalid image payload or malformed request message.
- **HTTP `413`:** File exceeds backend `MAX_UPLOAD_SIZE_BYTES` limit.
- **HTTP `415`:** Unsupported media type (only JPG, JPEG, PNG, WEBP allowed).
- **HTTP `429`:** Rate limit exceeded notice.
- **HTTP `503`:** Model unavailable (e.g., Phase 4 checkpoint not yet trained) with explicit recovery instructions.
- **Timeouts & Network Failures:** Clear offline/timeout messaging with Retry controls.

### Scientific Safety & Explainability Rules Enforced in UI
- **Zero Hardcoded Disease Classes:** All class labels and guidance come dynamically from backend responses.
- **Uncertain Predictions (`status: "uncertain"`):** Displays a prominent warning, suppresses unverified disease diagnosis, suppresses top predictions, and **never** displays treatment guidance.
- **Grad-CAM & Attention Bounding Box:** Clearly labeled as **approximate AI attention regions** and explicitly distinguished from verified pixel-level disease segmentation.

---

## 5. Live Camera Requirements

Implemented in `src/components/detection/CameraCapture.jsx`:
- **Opt-In Only:** Never calls `navigator.mediaDevices.getUserMedia()` on page load; activates strictly when the user clicks **Start Camera**.
- **Mobile & Desktop Support:** Defaults to `facingMode: { ideal: "environment" }` for rear mobile cameras, supports front/rear switching (`user` / `environment`), and enumerates multiple video input devices after permission is granted.
- **Resource Cleanup:** Automatically calls `track.stop()` on all active `MediaStreamTrack`s when capturing a photo, clicking **Stop Camera**, switching tabs, or unmounting the component.
- **Browser Security Context:** Live camera access requires `http://localhost`, `http://127.0.0.1`, or `https://` per browser WebRTC security rules.
- **Graceful Fallbacks:** Handles `NotAllowedError`, `NotFoundError`, `NotReadableError`, and missing `navigator.mediaDevices` with recovery instructions and a one-click **Upload Image Instead** fallback.

---

## 6. Running Instructions (Windows PowerShell)

Start the **Phase 7 FastAPI Backend** in one PowerShell terminal:

```powershell
cd "C:\Users\Mahesh Kumar Mahato\Desktop\Plant Disease Detection\PlantDiseaseAI"
.\.venv\Scripts\uvicorn.exe backend.app.main:app --reload --host 127.0.0.1 --port 8001
```

Start the **Phase 8 React Frontend** in a second PowerShell terminal (verified against `frontend/package.json`):

```powershell
cd "C:\Users\Mahesh Kumar Mahato\Desktop\Plant Disease Detection\PlantDiseaseAI\frontend"
npm install
npm run dev
```

Open **`http://127.0.0.1:5173/`** in your browser.

### Verified `package.json` Scripts
- `npm run dev` — Starts Vite dev server at `http://127.0.0.1:5173`
- `npm run build` — Builds optimized production bundle into `frontend/dist/`
- `npm run preview` — Previews production build at `http://127.0.0.1:4173`
- `npm test` — Runs the Vitest unit test suite once (`vitest run`)
- `npm run test:watch` — Runs Vitest in interactive watch mode

---

## 7. Testing Instructions

```powershell
cd "C:\Users\Mahesh Kumar Mahato\Desktop\Plant Disease Detection\PlantDiseaseAI\frontend"
npm test
```

Frontend unit tests (`src/__tests__/*.test.*`) use **mocked API responses** to verify UI logic deterministically. Live backend integration testing is kept strictly separate from mocked unit tests.

---

## 8. Known Limitations

1. **No Trained Phase 4 Checkpoint Currently Present:** Until Phase 4 multi-epoch training produces `best_model.pth`, the live backend reports `model_ready: false` (`status: "degraded"`) and returns HTTP `503` (`model_unavailable`) on `POST /api/v1/predict`. The frontend reports this transparently rather than fabricating predictions.
2. **History Record Detail Scope:** The Phase 7 backend SQLite table stores prediction metrics and Grad-CAM URLs, not full `disease_information` objects or original uploaded images (when `RETAIN_UPLOADED_IMAGES=false`). Fresh scans display both via router state / `sessionStorage` cache, while direct lookups of older records display all persisted fields with an honest notice.
3. **Client-Side History Sorting:** Because `GET /api/v1/history` returns records sorted newest-first without a server-side `sort` parameter, the "Oldest first" sort option reorders the loaded page records client-side.
4. **Approximate Attention Localization:** `LocalizationViewer.jsx` derives an approximate bounding box around peak Grad-CAM heatmap activations on the client canvas; it is an interpretability indicator, not pixel-accurate segmentation.
