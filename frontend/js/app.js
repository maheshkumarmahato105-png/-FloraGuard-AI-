/**
 * Main FloraGuard AI application controller.
 * Wires navigation, live camera, drag-and-drop/file upload, mobile capture,
 * AI inference, results rendering, history filtering, and model evaluation metrics.
 */

import {
  deleteHistoryRecord,
  fetchEvaluationReport,
  fetchHealth,
  fetchHistory,
  fetchSupportedPlants,
  predictPlantDisease,
  resolveAssetUrl,
} from "./api.js";
import { CameraController, optimizeImageForUpload } from "./camera.js";
import { initNavbar, setActiveNav, updateHealthPill } from "../components/navbar.js";
import { renderResultCard } from "../components/resultCard.js";
import { renderHistoryTable } from "../components/historyTable.js";

const VALID_PAGES = new Set(["home", "detect", "results", "history", "about"]);
const ALLOWED_MIME = new Set(["image/jpeg", "image/jpg", "image/png", "image/webp"]);
const MAX_SIZE_BYTES = 10 * 1024 * 1024;

const state = {
  currentPage: "home",
  selectedFile: null,
  selectedPreviewUrl: null,
  inputSource: "upload", // upload | live_camera | mobile_camera
  lastResult: null,
  health: null,
  plants: [],
  historyItems: [],
  isAnalyzing: false,
  activeInputTab: "upload",
};

let cameraController = null;

function esc(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const colors = {
    success: "bg-emerald-600 text-white",
    error: "bg-rose-600 text-white",
    warning: "bg-amber-500 text-white",
    info: "bg-slate-900 text-white",
  };
  const toast = document.createElement("div");
  toast.className = `toast-enter px-4 py-3 rounded-xl shadow-lg text-sm font-medium flex items-center gap-2.5 max-w-sm ${
    colors[type] || colors.info
  }`;
  toast.innerHTML = `<span class="flex-1">${esc(message)}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transition = "opacity 0.25s ease";
    setTimeout(() => toast.remove(), 260);
  }, 4000);
}

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

// ==================== NAVIGATION ====================

function navigateTo(pageId) {
  const target = VALID_PAGES.has(pageId) ? pageId : "home";
  // Stop camera automatically when leaving detection page
  if (state.currentPage === "detect" && target !== "detect" && cameraController?.isActive) {
    stopLiveCamera();
  }

  state.currentPage = target;
  document.querySelectorAll(".page-section").forEach((section) => {
    section.classList.toggle("active", section.id === `page-${target}`);
  });
  setActiveNav(target);

  if (window.location.hash !== `#${target}`) {
    history.replaceState(null, "", `#${target}`);
  }
  window.scrollTo({ top: 0, behavior: "smooth" });

  if (target === "results") {
    renderCurrentResult();
  } else if (target === "history") {
    loadHistory();
  } else if (target === "about") {
    loadAboutEvaluation();
  }
}

// ==================== BACKEND HEALTH & SUPPORTED PLANTS ====================

async function loadSystemHealth() {
  try {
    const health = await fetchHealth();
    state.health = health;
    updateHealthPill(health);
    updateHomeMetrics(health);

    const modelBanner = document.getElementById("detect-model-warning");
    if (modelBanner) {
      if (!health.model_loaded) {
        modelBanner.classList.remove("hidden");
        modelBanner.querySelector("[data-warning-text]").textContent =
          health.message ||
          "The AI model is not trained yet. Run training/download_dataset.py and training/train.py to enable predictions.";
      } else {
        modelBanner.classList.add("hidden");
      }
    }
  } catch (err) {
    state.health = null;
    updateHealthPill(null, err);
    const modelBanner = document.getElementById("detect-model-warning");
    if (modelBanner) {
      modelBanner.classList.remove("hidden");
      modelBanner.querySelector("[data-warning-text]").textContent =
        "Cannot connect to the Python backend API. Make sure FastAPI is running on port 8000.";
    }
  }
}

function updateHomeMetrics(health) {
  const accEl = document.getElementById("home-stat-accuracy");
  const classesEl = document.getElementById("home-stat-classes");
  const plantsEl = document.getElementById("home-stat-plants");
  const archEl = document.getElementById("home-stat-arch");

  if (classesEl && health.num_classes) {
    classesEl.textContent = String(health.num_classes);
  }
  if (plantsEl && health.supported_plants?.length) {
    plantsEl.textContent = String(health.supported_plants.length);
  }
  if (archEl && health.model_architecture) {
    archEl.textContent =
      health.model_architecture === "mobilenet_v2" ? "MobileNetV2" : health.model_architecture;
  }
  if (accEl) {
    const testAcc = health.evaluation_summary?.test_accuracy_pct;
    accEl.textContent = testAcc != null ? `${Number(testAcc).toFixed(1)}%` : "Evaluating";
  }
}

async function loadSupportedPlants() {
  const grid = document.getElementById("home-plants-grid");
  if (!grid) return;

  try {
    const data = await fetchSupportedPlants();
    state.plants = data.plants || [];
    if (state.plants.length === 0) {
      renderFallbackPlants(grid);
      return;
    }

    const plantIcons = {
      Tomato: "🍅",
      Potato: "🥔",
      Apple: "🍎",
      "Corn (Maize)": "🌽",
      Grape: "🍇",
      "Pepper (Bell)": "🫑",
      Strawberry: "🍓",
      Cherry: "🍒",
      Peach: "🍑",
      "Orange (Citrus)": "🍊",
      Blueberry: "🫐",
      Raspberry: "🍓",
      Soybean: "🌱",
      Squash: "🎃",
    };

    grid.innerHTML = state.plants
      .map((p) => {
        const diseases = p.conditions.filter((c) => !c.is_healthy);
        const icon = plantIcons[p.plant_name] || "🌿";
        return `
          <div class="bg-white rounded-2xl border border-slate-200 p-5 hover:border-emerald-300 hover:shadow-md transition group">
            <div class="flex items-center justify-between mb-3">
              <div class="flex items-center gap-2.5">
                <span class="w-10 h-10 rounded-xl bg-emerald-50 flex items-center justify-center text-xl">${icon}</span>
                <div>
                  <h4 class="font-bold text-slate-900 group-hover:text-emerald-700 transition">${esc(p.plant_name)}</h4>
                  <p class="text-xs text-slate-500">${p.condition_count} trained ${p.condition_count === 1 ? "class" : "classes"}</p>
                </div>
              </div>
              <span class="px-2 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-50 text-emerald-700">Active</span>
            </div>
            <div class="flex flex-wrap gap-1.5">
              ${
                diseases.length > 0
                  ? diseases
                      .slice(0, 4)
                      .map(
                        (d) =>
                          `<span class="px-2 py-0.5 rounded-md bg-slate-100 text-slate-700 text-[11px] font-medium">${esc(
                            d.disease_name
                          )}</span>`
                      )
                      .join("") +
                    (diseases.length > 4
                      ? `<span class="px-2 py-0.5 rounded-md bg-emerald-50 text-emerald-700 text-[11px] font-semibold">+${
                          diseases.length - 4
                        } more</span>`
                      : "")
                  : `<span class="px-2 py-0.5 rounded-md bg-emerald-50 text-emerald-700 text-[11px] font-medium">Healthy Leaf Verification</span>`
              }
            </div>
          </div>
        `;
      })
      .join("");
  } catch {
    renderFallbackPlants(grid);
  }
}

function renderFallbackPlants(grid) {
  const core = [
    { name: "Tomato", icon: "🍅", info: "Early & Late Blight, Bacterial Spot, Leaf Mold, Viruses + Healthy" },
    { name: "Potato", icon: "🥔", info: "Early Blight, Late Blight & Healthy" },
    { name: "Apple", icon: "🍎", info: "Apple Scab, Black Rot, Cedar Apple Rust & Healthy" },
    { name: "Corn (Maize)", icon: "🌽", info: "Gray Leaf Spot, Common Rust, Northern Leaf Blight & Healthy" },
    { name: "Grape", icon: "🍇", info: "Black Rot, Esca (Black Measles), Leaf Blight & Healthy" },
    { name: "Pepper (Bell)", icon: "🫑", info: "Bacterial Spot & Healthy" },
    { name: "Strawberry", icon: "🍓", info: "Leaf Scorch & Healthy" },
  ];
  grid.innerHTML = core
    .map(
      (c) => `
      <div class="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm">
        <div class="flex items-center gap-3 mb-2">
          <span class="w-10 h-10 rounded-xl bg-emerald-50 flex items-center justify-center text-xl">${c.icon}</span>
          <h4 class="font-bold text-slate-900">${c.name}</h4>
        </div>
        <p class="text-xs text-slate-600 leading-relaxed">${c.info}</p>
      </div>`
    )
    .join("");
}

// ==================== DETECTION PAGE: UPLOAD & CAMERA ====================

function switchInputTab(tabName) {
  state.activeInputTab = tabName;
  document.querySelectorAll("[data-input-tab]").forEach((btn) => {
    const active = btn.getAttribute("data-input-tab") === tabName;
    btn.classList.toggle("bg-emerald-600", active);
    btn.classList.toggle("text-white", active);
    btn.classList.toggle("shadow-sm", active);
    btn.classList.toggle("text-slate-600", !active);
    btn.classList.toggle("hover:bg-slate-100", !active);
  });

  const uploadPanel = document.getElementById("panel-upload");
  const cameraPanel = document.getElementById("panel-camera");
  if (uploadPanel) uploadPanel.classList.toggle("hidden", tabName !== "upload");
  if (cameraPanel) cameraPanel.classList.toggle("hidden", tabName !== "camera");

  if (tabName === "camera") {
    startLiveCamera();
  } else {
    stopLiveCamera();
  }
}

async function handleSelectedImage(rawFile, source = "upload") {
  if (!rawFile) return;

  // Validate file type and size
  const ext = (rawFile.name || "").toLowerCase();
  const extOk = /\.(jpg|jpeg|png|webp)$/.test(ext);
  const mimeOk = ALLOWED_MIME.has(rawFile.type?.toLowerCase());
  if (!mimeOk && !extOk) {
    showToast("Unsupported format. Please choose a JPG, JPEG, PNG, or WEBP image.", "error");
    return;
  }
  if (rawFile.size > MAX_SIZE_BYTES) {
    showToast("Image exceeds the 10 MB size limit. Please pick a smaller image.", "error");
    return;
  }

  // Optimize huge mobile photos smoothly on canvas
  const file = await optimizeImageForUpload(rawFile);

  if (state.selectedPreviewUrl && state.selectedPreviewUrl.startsWith("blob:")) {
    URL.revokeObjectURL(state.selectedPreviewUrl);
  }

  state.selectedFile = file;
  state.inputSource = source;
  state.selectedPreviewUrl = URL.createObjectURL(file);

  updatePreviewPanel(file, state.selectedPreviewUrl, source);
}

function updatePreviewPanel(file, previewUrl, source) {
  const emptyState = document.getElementById("preview-empty");
  const filledState = document.getElementById("preview-filled");
  const previewImg = document.getElementById("preview-image");
  const nameEl = document.getElementById("preview-filename");
  const metaEl = document.getElementById("preview-meta");
  const sourceBadge = document.getElementById("preview-source-badge");
  const analyzeBtn = document.getElementById("btn-analyze-plant");
  const errorBox = document.getElementById("detect-error-box");

  if (errorBox) errorBox.classList.add("hidden");

  if (!file || !previewUrl) {
    emptyState?.classList.remove("hidden");
    filledState?.classList.add("hidden");
    if (analyzeBtn) analyzeBtn.disabled = true;
    return;
  }

  emptyState?.classList.add("hidden");
  filledState?.classList.remove("hidden");
  if (previewImg) {
    previewImg.src = previewUrl;
    previewImg.onload = () => {
      if (metaEl) {
        metaEl.textContent = `${previewImg.naturalWidth} × ${previewImg.naturalHeight}px · ${formatFileSize(
          file.size
        )}`;
      }
    };
  }
  if (nameEl) nameEl.textContent = file.name || "captured-leaf.jpg";
  if (metaEl) metaEl.textContent = formatFileSize(file.size);

  const sourceLabels = {
    upload: "Uploaded Image",
    live_camera: "Live Camera Capture",
    mobile_camera: "Mobile Camera",
  };
  if (sourceBadge) sourceBadge.textContent = sourceLabels[source] || "Image Ready";
  if (analyzeBtn) analyzeBtn.disabled = false;
}

function clearSelectedImage() {
  if (state.selectedPreviewUrl && state.selectedPreviewUrl.startsWith("blob:")) {
    URL.revokeObjectURL(state.selectedPreviewUrl);
  }
  state.selectedFile = null;
  state.selectedPreviewUrl = null;
  const fileInput = document.getElementById("file-upload-input");
  const mobileInput = document.getElementById("mobile-camera-input");
  if (fileInput) fileInput.value = "";
  if (mobileInput) mobileInput.value = "";
  updatePreviewPanel(null, null, "upload");
}

async function startLiveCamera() {
  const video = document.getElementById("camera-video");
  const canvas = document.getElementById("camera-canvas");
  const statusEl = document.getElementById("camera-status-text");
  const errBox = document.getElementById("camera-error-box");
  const captureBtn = document.getElementById("btn-capture-photo");
  const switchBtn = document.getElementById("btn-switch-camera");
  const startBtn = document.getElementById("btn-start-camera");
  const stopBtn = document.getElementById("btn-stop-camera");

  if (!video || !canvas) return;
  if (!cameraController) {
    cameraController = new CameraController(video, canvas);
  }

  errBox?.classList.add("hidden");
  if (statusEl) statusEl.textContent = "Requesting camera permission...";

  try {
    const info = await cameraController.start();
    if (statusEl) {
      statusEl.textContent = `Active: ${info.label}`;
    }
    if (captureBtn) captureBtn.disabled = false;
    if (switchBtn) switchBtn.disabled = false;
    startBtn?.classList.add("hidden");
    stopBtn?.classList.remove("hidden");
  } catch (err) {
    if (statusEl) statusEl.textContent = "Camera unavailable";
    if (errBox) {
      errBox.textContent = err.message;
      errBox.classList.remove("hidden");
    }
    if (captureBtn) captureBtn.disabled = true;
    if (switchBtn) switchBtn.disabled = true;
    startBtn?.classList.remove("hidden");
    stopBtn?.classList.add("hidden");
  }
}

function stopLiveCamera() {
  if (cameraController) {
    cameraController.stop();
  }
  const statusEl = document.getElementById("camera-status-text");
  const captureBtn = document.getElementById("btn-capture-photo");
  const switchBtn = document.getElementById("btn-switch-camera");
  const startBtn = document.getElementById("btn-start-camera");
  const stopBtn = document.getElementById("btn-stop-camera");

  if (statusEl) statusEl.textContent = "Camera stopped";
  if (captureBtn) captureBtn.disabled = true;
  if (switchBtn) switchBtn.disabled = true;
  startBtn?.classList.remove("hidden");
  stopBtn?.classList.add("hidden");
}

async function handleSwitchCamera() {
  if (!cameraController) return;
  const statusEl = document.getElementById("camera-status-text");
  const errBox = document.getElementById("camera-error-box");
  errBox?.classList.add("hidden");
  if (statusEl) statusEl.textContent = "Switching camera...";

  try {
    const info = await cameraController.switchCamera();
    if (statusEl) statusEl.textContent = `Active: ${info.label}`;
    showToast(`Switched to ${info.label}`, "info");
  } catch (err) {
    if (errBox) {
      errBox.textContent = err.message;
      errBox.classList.remove("hidden");
    }
  }
}

async function handleCapturePhoto() {
  if (!cameraController?.isActive) {
    showToast("Start the camera first to capture a plant leaf.", "warning");
    return;
  }
  try {
    const captured = await cameraController.captureImage();
    state.selectedFile = captured.file;
    state.inputSource = "live_camera";
    state.selectedPreviewUrl = captured.dataUrl;
    updatePreviewPanel(captured.file, captured.dataUrl, "live_camera");
    showToast("Leaf image captured! Click 'Analyze Plant' to diagnose.", "success");
  } catch (err) {
    showToast(err.message || "Failed to capture image.", "error");
  }
}

async function runPlantAnalysis() {
  if (!state.selectedFile || state.isAnalyzing) return;

  state.isAnalyzing = true;
  const analyzeBtn = document.getElementById("btn-analyze-plant");
  const scanOverlay = document.getElementById("preview-scan-overlay");
  const scanStepText = document.getElementById("scan-step-text");
  const errorBox = document.getElementById("detect-error-box");

  if (errorBox) errorBox.classList.add("hidden");
  if (analyzeBtn) {
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = `
      <svg class="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
      </svg>
      Analyzing Leaf...
    `;
  }
  scanOverlay?.classList.remove("hidden");

  const steps = [
    "Validating image integrity...",
    "Detecting & focusing plant leaf (OpenCV)...",
    "Running MobileNetV2 deep learning inference...",
  ];
  let stepIdx = 0;
  if (scanStepText) scanStepText.textContent = steps[0];
  const stepTimer = setInterval(() => {
    stepIdx = Math.min(stepIdx + 1, steps.length - 1);
    if (scanStepText) scanStepText.textContent = steps[stepIdx];
  }, 450);

  try {
    const result = await predictPlantDisease(state.selectedFile, state.inputSource);
    state.lastResult = result;

    if (result.is_uncertain) {
      showToast("Model confidence was too low. Please review advice and retake photo.", "warning");
    } else if (result.is_healthy) {
      showToast(`Healthy ${result.plant_name} leaf detected (${result.confidence.toFixed(1)}%).`, "success");
    } else {
      showToast(
        `${result.plant_name}: ${result.disease_name} detected (${result.confidence.toFixed(1)}%).`,
        "info"
      );
    }

    navigateTo("results");
  } catch (err) {
    if (errorBox) {
      errorBox.textContent = err.message || "Prediction request failed. Please try again.";
      errorBox.classList.remove("hidden");
    }
    showToast(err.message || "Analysis failed.", "error");
  } finally {
    clearInterval(stepTimer);
    state.isAnalyzing = false;
    scanOverlay?.classList.add("hidden");
    if (analyzeBtn) {
      analyzeBtn.disabled = !state.selectedFile;
      analyzeBtn.innerHTML = `
        <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z"/>
        </svg>
        Analyze Plant
      `;
    }
  }
}

function initDetectionControls() {
  // Input tabs
  document.querySelectorAll("[data-input-tab]").forEach((btn) => {
    btn.addEventListener("click", () => {
      switchInputTab(btn.getAttribute("data-input-tab"));
    });
  });

  // Drag & Drop zone
  const dropzone = document.getElementById("upload-dropzone");
  const fileInput = document.getElementById("file-upload-input");
  const mobileInput = document.getElementById("mobile-camera-input");

  if (dropzone && fileInput) {
    dropzone.addEventListener("click", (e) => {
      if (e.target.closest("#btn-mobile-native-capture")) return;
      fileInput.click();
    });

    ["dragenter", "dragover"].forEach((evt) => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("dropzone-active");
      });
    });

    ["dragleave", "drop"].forEach((evt) => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("dropzone-active");
      });
    });

    dropzone.addEventListener("drop", (e) => {
      const files = e.dataTransfer?.files;
      if (files && files.length > 0) {
        handleSelectedImage(files[0], "upload");
      }
    });

    fileInput.addEventListener("change", () => {
      if (fileInput.files && fileInput.files.length > 0) {
        handleSelectedImage(fileInput.files[0], "upload");
      }
    });
  }

  document.getElementById("btn-mobile-native-capture")?.addEventListener("click", (e) => {
    e.stopPropagation();
    mobileInput?.click();
  });

  mobileInput?.addEventListener("change", () => {
    if (mobileInput.files && mobileInput.files.length > 0) {
      handleSelectedImage(mobileInput.files[0], "mobile_camera");
    }
  });

  // Camera buttons
  document.getElementById("btn-start-camera")?.addEventListener("click", startLiveCamera);
  document.getElementById("btn-stop-camera")?.addEventListener("click", stopLiveCamera);
  document.getElementById("btn-switch-camera")?.addEventListener("click", handleSwitchCamera);
  document.getElementById("btn-capture-photo")?.addEventListener("click", handleCapturePhoto);

  // Preview & Analyze buttons
  document.getElementById("btn-clear-preview")?.addEventListener("click", clearSelectedImage);
  document.getElementById("btn-analyze-plant")?.addEventListener("click", runPlantAnalysis);
}

// ==================== RESULTS PAGE ====================

function renderCurrentResult() {
  const container = document.getElementById("results-container");
  if (!container) return;
  renderResultCard(container, state.lastResult, {
    onScanAgain: () => {
      clearSelectedImage();
      navigateTo("detect");
    },
    onOpenHistory: () => navigateTo("history"),
  });
}

// ==================== HISTORY PAGE ====================

async function loadHistory() {
  const container = document.getElementById("history-table-container");
  const searchInput = document.getElementById("history-search-input");
  const statusSelect = document.getElementById("history-status-filter");
  if (!container) return;

  container.innerHTML = `
    <div class="bg-white rounded-2xl border border-slate-200 p-10 text-center text-slate-500 text-sm">
      Loading detection history...
    </div>
  `;

  try {
    const data = await fetchHistory({
      search: searchInput?.value?.trim() || "",
      status: statusSelect?.value || "",
      limit: 100,
    });
    state.historyItems = data.items || [];
    updateHistoryStats(state.historyItems, data.total);

    renderHistoryTable(container, state.historyItems, {
      onView: (item) => {
        state.lastResult = {
          record_id: item.id,
          timestamp: item.created_at,
          image_url: item.image_url,
          image_filename: item.image_filename,
          plant_name: item.plant_name,
          disease_name: item.disease_name,
          raw_class_label: item.raw_class_label,
          status: item.status,
          is_healthy: item.status === "Healthy",
          is_uncertain: item.status === "Uncertain",
          confidence: item.confidence,
          confidence_threshold: state.health?.confidence_threshold || 60,
          disease_info: item.disease_info,
          top_predictions: item.top_predictions || [],
          model_architecture: state.health?.model_architecture || "mobilenet_v2",
        };
        navigateTo("results");
      },
      onDelete: async (recordId) => {
        try {
          await deleteHistoryRecord(recordId);
          showToast(`Deleted record #${recordId}.`, "success");
          loadHistory();
        } catch (err) {
          showToast(err.message || "Failed to delete record.", "error");
        }
      },
      onStartDetect: () => navigateTo("detect"),
    });
  } catch (err) {
    container.innerHTML = `
      <div class="bg-rose-50 border border-rose-200 rounded-2xl p-6 text-rose-700 text-sm">
        Failed to load history: ${esc(err.message)}
      </div>
    `;
  }
}

function updateHistoryStats(items, total) {
  const totalEl = document.getElementById("hist-stat-total");
  const healthyEl = document.getElementById("hist-stat-healthy");
  const infectedEl = document.getElementById("hist-stat-infected");
  const avgConfEl = document.getElementById("hist-stat-confidence");

  const healthyCount = items.filter((i) => i.status === "Healthy").length;
  const infectedCount = items.filter((i) => i.status === "Infected").length;
  const avgConf =
    items.length > 0
      ? items.reduce((acc, i) => acc + Number(i.confidence || 0), 0) / items.length
      : 0;

  if (totalEl) totalEl.textContent = String(total ?? items.length);
  if (healthyEl) healthyEl.textContent = String(healthyCount);
  if (infectedEl) infectedEl.textContent = String(infectedCount);
  if (avgConfEl) avgConfEl.textContent = items.length > 0 ? `${avgConf.toFixed(1)}%` : "—";
}

function initHistoryControls() {
  const searchInput = document.getElementById("history-search-input");
  const statusSelect = document.getElementById("history-status-filter");
  const refreshBtn = document.getElementById("history-refresh-btn");

  let debounceTimer = null;
  searchInput?.addEventListener("input", () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => loadHistory(), 260);
  });
  statusSelect?.addEventListener("change", () => loadHistory());
  refreshBtn?.addEventListener("click", () => loadHistory());
}

// ==================== ABOUT PAGE: REAL EVALUATION REPORT ====================

async function loadAboutEvaluation() {
  const container = document.getElementById("about-evaluation-panel");
  if (!container) return;

  try {
    const data = await fetchEvaluationReport();
    if (!data.available || !data.metrics) {
      container.innerHTML = `
        <div class="bg-amber-50 border border-amber-200 rounded-2xl p-6 text-sm text-amber-800">
          Evaluation report not generated yet. Run <code class="font-mono bg-amber-100 px-1.5 py-0.5 rounded">python training/evaluate.py</code> after training to populate real held-out test metrics and confusion matrix charts.
        </div>
      `;
      return;
    }

    const m = data.metrics;
    const splits = m.dataset_splits || {};
    container.innerHTML = `
      <div class="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
        <div class="bg-emerald-50 border border-emerald-200 rounded-2xl p-4">
          <div class="text-xs font-semibold text-emerald-700 uppercase">Test Accuracy</div>
          <div class="text-2xl font-extrabold text-emerald-900 mt-1">${Number(m.accuracy_pct).toFixed(2)}%</div>
        </div>
        <div class="bg-white border border-slate-200 rounded-2xl p-4">
          <div class="text-xs font-semibold text-slate-500 uppercase">Top-3 Accuracy</div>
          <div class="text-2xl font-extrabold text-slate-900 mt-1">${Number(m.top3_accuracy_pct).toFixed(2)}%</div>
        </div>
        <div class="bg-white border border-slate-200 rounded-2xl p-4">
          <div class="text-xs font-semibold text-slate-500 uppercase">Macro Precision</div>
          <div class="text-2xl font-extrabold text-slate-900 mt-1">${Number(m.macro_precision_pct).toFixed(2)}%</div>
        </div>
        <div class="bg-white border border-slate-200 rounded-2xl p-4">
          <div class="text-xs font-semibold text-slate-500 uppercase">Macro Recall</div>
          <div class="text-2xl font-extrabold text-slate-900 mt-1">${Number(m.macro_recall_pct).toFixed(2)}%</div>
        </div>
        <div class="bg-white border border-slate-200 rounded-2xl p-4">
          <div class="text-xs font-semibold text-slate-500 uppercase">Macro F1-Score</div>
          <div class="text-2xl font-extrabold text-slate-900 mt-1">${Number(m.macro_f1_pct).toFixed(2)}%</div>
        </div>
      </div>

      <div class="text-xs text-slate-500 mb-6 flex flex-wrap gap-4">
        <span>Architecture: <strong class="text-slate-800">${esc(m.architecture)}</strong></span>
        <span>Dataset: <strong class="text-slate-800">${esc(m.dataset)}</strong></span>
        <span>Classes: <strong class="text-slate-800">${esc(m.num_classes)}</strong></span>
        ${
          splits.train
            ? `<span>Splits: <strong class="text-slate-800">${splits.train.toLocaleString()} Train / ${splits.val.toLocaleString()} Val / ${splits.test.toLocaleString()} Test</strong></span>`
            : `<span>Test Images: <strong class="text-slate-800">${esc(m.test_samples)}</strong></span>`
        }
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        ${
          data.training_curves_url
            ? `<div class="bg-white rounded-2xl border border-slate-200 p-4">
                <h4 class="text-sm font-bold text-slate-900 mb-3">Training & Validation Progression</h4>
                <a href="${esc(resolveAssetUrl(data.training_curves_url))}" target="_blank" rel="noopener">
                  <img src="${esc(resolveAssetUrl(data.training_curves_url))}" alt="Training Curves" class="w-full rounded-xl border border-slate-100 hover:opacity-95 transition" />
                </a>
              </div>`
            : ""
        }
        ${
          data.confusion_matrix_url
            ? `<div class="bg-white rounded-2xl border border-slate-200 p-4">
                <h4 class="text-sm font-bold text-slate-900 mb-3">Held-Out Test Confusion Matrix (Click to Enlarge)</h4>
                <a href="${esc(resolveAssetUrl(data.confusion_matrix_url))}" target="_blank" rel="noopener">
                  <img src="${esc(resolveAssetUrl(data.confusion_matrix_url))}" alt="Confusion Matrix" class="w-full max-h-96 object-contain rounded-xl border border-slate-100 hover:opacity-95 transition" />
                </a>
              </div>`
            : ""
        }
      </div>
    `;
  } catch (err) {
    container.innerHTML = `
      <div class="bg-slate-50 border border-slate-200 rounded-2xl p-6 text-sm text-slate-600">
        Unable to load live evaluation report right now (${esc(err.message)}).
      </div>
    `;
  }
}

// ==================== BOOTSTRAP ====================

function initApp() {
  initNavbar(navigateTo);
  initDetectionControls();
  initHistoryControls();

  // Wire hero & CTA navigation buttons
  document.querySelectorAll("[data-goto]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-goto");
      if (target) navigateTo(target);
    });
  });

  // Initial route from URL hash
  const initialHash = window.location.hash.replace("#", "").trim();
  navigateTo(VALID_PAGES.has(initialHash) ? initialHash : "home");

  window.addEventListener("hashchange", () => {
    const hash = window.location.hash.replace("#", "").trim();
    if (VALID_PAGES.has(hash) && hash !== state.currentPage) {
      navigateTo(hash);
    }
  });

  loadSystemHealth();
  loadSupportedPlants();
}

document.addEventListener("DOMContentLoaded", initApp);
