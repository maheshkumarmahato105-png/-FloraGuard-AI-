/**
 * Renders the full diagnostic results view with confidence gauge, status banner,
 * top-3 breakdown, symptoms, causes, prevention, and treatment cards.
 */

import { resolveAssetUrl } from "../js/api.js";

function esc(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatDateTime(isoString) {
  if (!isoString) return "Just now";
  const d = new Date(isoString);
  if (Number.isNaN(d.getTime())) return isoString;
  return d.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

function renderListItems(items = [], bulletColor = "bg-emerald-500") {
  if (!items || items.length === 0) {
    return `<li class="text-slate-500 text-sm">No items available.</li>`;
  }
  return items
    .map(
      (item) => `
      <li class="flex items-start gap-2.5 text-sm text-slate-700 leading-relaxed">
        <span class="w-1.5 h-1.5 rounded-full ${bulletColor} mt-2 shrink-0"></span>
        <span>${esc(item)}</span>
      </li>`
    )
    .join("");
}

export function renderResultCard(container, result, { onScanAgain, onOpenHistory } = {}) {
  if (!result) {
    container.innerHTML = `
      <div class="bg-white rounded-2xl border border-slate-200 p-12 text-center shadow-sm">
        <div class="w-16 h-16 rounded-2xl bg-emerald-50 text-emerald-600 flex items-center justify-center mx-auto mb-4">
          <svg class="w-8 h-8" fill="none" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-4.35-4.35m0 0A7.5 7.5 0 1010.5 18a7.5 7.5 0 006.15-3.35z"/>
          </svg>
        </div>
        <h3 class="text-xl font-bold text-slate-900 mb-2">No Diagnostic Result Yet</h3>
        <p class="text-slate-600 max-w-md mx-auto mb-6">
          Capture a leaf with your camera or upload a photo on the Detection page to view detailed AI disease diagnostics.
        </p>
        <button id="empty-result-detect-btn" class="px-6 py-3 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-semibold shadow-sm transition">
          Start Plant Detection
        </button>
      </div>
    `;
    const btn = container.querySelector("#empty-result-detect-btn");
    if (btn && onScanAgain) btn.addEventListener("click", onScanAgain);
    return;
  }

  const isHealthy = result.status === "Healthy";
  const isUncertain = result.status === "Uncertain" || result.is_uncertain;
  const confidence = Number(result.confidence || 0);
  const clampedConf = Math.max(0, Math.min(100, confidence));

  // SVG circular confidence gauge calculations
  const radius = 48;
  const circumference = 2 * Math.PI * radius;
  const strokeOffset = circumference * (1 - clampedConf / 100);

  let statusTheme = {
    badgeBg: "bg-rose-100 text-rose-800 border-rose-200",
    bannerBg: "bg-gradient-to-r from-rose-50 to-orange-50 border-rose-200",
    ringStroke: "#e11d48",
    barBg: "bg-rose-600",
    iconColor: "text-rose-600",
    label: "Infected",
  };
  if (isHealthy) {
    statusTheme = {
      badgeBg: "bg-emerald-100 text-emerald-800 border-emerald-200",
      bannerBg: "bg-gradient-to-r from-emerald-50 to-teal-50 border-emerald-200",
      ringStroke: "#059669",
      barBg: "bg-emerald-600",
      iconColor: "text-emerald-600",
      label: "Healthy",
    };
  } else if (isUncertain) {
    statusTheme = {
      badgeBg: "bg-amber-100 text-amber-800 border-amber-200",
      bannerBg: "bg-gradient-to-r from-amber-50 to-yellow-50 border-amber-200",
      ringStroke: "#d97706",
      barBg: "bg-amber-500",
      iconColor: "text-amber-600",
      label: "Uncertain — Retake Needed",
    };
  }

  const imageUrl = resolveAssetUrl(result.image_url);
  const diagnostics = result.diagnostics || null;
  const diseaseInfo = result.disease_info || null;
  const topPredictions = result.top_predictions || [];

  const topPredHtml =
    topPredictions.length > 0
      ? topPredictions
          .map(
            (pred, idx) => `
          <div class="space-y-1.5">
            <div class="flex items-center justify-between text-xs sm:text-sm">
              <span class="font-medium text-slate-800 truncate pr-2">
                ${idx + 1}. ${esc(pred.plant_name)} — ${esc(pred.disease_name)}
              </span>
              <span class="font-semibold text-slate-700 shrink-0">${Number(pred.confidence).toFixed(1)}%</span>
            </div>
            <div class="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
              <div class="h-full rounded-full ${
                idx === 0 ? statusTheme.barBg : "bg-slate-400"
              } transition-all duration-700" style="width: ${Math.max(2, Math.min(100, Number(pred.confidence)))}%"></div>
            </div>
          </div>`
          )
          .join("")
      : `<p class="text-sm text-slate-500">Top prediction breakdown unavailable for this archived record.</p>`;

  const diagnosticsHtml = diagnostics
    ? `
      <div class="mt-4 pt-4 border-t border-slate-100 grid grid-cols-2 gap-2.5 text-xs">
        <div class="bg-slate-50 rounded-lg p-2.5">
          <div class="text-slate-500">Resolution</div>
          <div class="font-semibold text-slate-800">${diagnostics.original_width} × ${diagnostics.original_height}px</div>
        </div>
        <div class="bg-slate-50 rounded-lg p-2.5">
          <div class="text-slate-500">Leaf Focus (OpenCV)</div>
          <div class="font-semibold text-slate-800">${diagnostics.leaf_focused ? "Applied (ROI Crop)" : "Full Frame"}</div>
        </div>
        <div class="bg-slate-50 rounded-lg p-2.5">
          <div class="text-slate-500">Sharpness Score</div>
          <div class="font-semibold text-slate-800">${Number(diagnostics.blur_score).toFixed(1)}</div>
        </div>
        <div class="bg-slate-50 rounded-lg p-2.5">
          <div class="text-slate-500">Inference Time</div>
          <div class="font-semibold text-slate-800">${result.inference_time_ms ? `${result.inference_time_ms} ms` : "—"}</div>
        </div>
      </div>
      ${
        diagnostics.quality_warnings && diagnostics.quality_warnings.length > 0
          ? `<div class="mt-3 p-3 rounded-xl bg-amber-50 border border-amber-200 text-xs text-amber-800 space-y-1">
              <div class="font-semibold">Image Quality Advisory:</div>
              ${diagnostics.quality_warnings.map((w) => `<div>• ${esc(w)}</div>`).join("")}
            </div>`
          : ""
      }
    `
    : "";

  const knowledgeSectionHtml = isUncertain
    ? `
      <div class="bg-amber-50 border border-amber-200 rounded-2xl p-6 space-y-4">
        <div class="flex items-start gap-3">
          <div class="w-10 h-10 rounded-xl bg-amber-100 text-amber-700 flex items-center justify-center shrink-0">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"/>
            </svg>
          </div>
          <div>
            <h3 class="text-lg font-bold text-amber-900">Unable to Confirm Diagnosis Reliably</h3>
            <p class="text-sm text-amber-800 mt-1 leading-relaxed">
              ${esc(
                result.uncertainty_message ||
                  "The model's confidence was too low to provide a trustworthy diagnosis. Please capture another clearer image."
              )}
            </p>
          </div>
        </div>
        <div class="bg-white/80 rounded-xl p-4 border border-amber-200/80">
          <h4 class="text-sm font-semibold text-slate-800 mb-2">Tips for a Clearer Plant Leaf Photo</h4>
          <ul class="space-y-1.5 text-sm text-slate-700">
            <li>• Photograph a single leaf close up so it fills 40%–75% of the frame.</li>
            <li>• Use bright, indirect natural daylight without harsh shadows or glare.</li>
            <li>• Keep the camera steady and tap to focus directly on any visible spots or lesions.</li>
            <li>• Place a plain neutral background behind the leaf if surrounding foliage is cluttered.</li>
          </ul>
        </div>
      </div>
    `
    : diseaseInfo
    ? `
      <!-- Disease Overview -->
      <div class="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
        <div class="flex flex-wrap items-center justify-between gap-2 mb-3">
          <h3 class="text-base font-bold text-slate-900 flex items-center gap-2">
            <span class="w-2 h-5 rounded-full bg-emerald-500"></span>
            Condition Overview
          </h3>
          <span class="px-3 py-1 rounded-full text-xs font-semibold ${
            isHealthy
              ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
              : "bg-slate-100 text-slate-700 border border-slate-200"
          }">
            Severity: ${esc(diseaseInfo.severity)}
          </span>
        </div>
        <p class="text-sm sm:text-base text-slate-700 leading-relaxed">
          ${esc(diseaseInfo.description)}
        </p>
      </div>

      <!-- 2x2 Agronomic Guidance Cards -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div class="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm">
          <h4 class="text-sm font-bold text-slate-900 uppercase tracking-wider mb-3 flex items-center gap-2">
            <span class="w-7 h-7 rounded-lg bg-rose-50 text-rose-600 flex items-center justify-center">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z"/>
                <path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
              </svg>
            </span>
            Visible Symptoms
          </h4>
          <ul class="space-y-2.5">${renderListItems(diseaseInfo.symptoms, "bg-rose-500")}</ul>
        </div>

        <div class="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm">
          <h4 class="text-sm font-bold text-slate-900 uppercase tracking-wider mb-3 flex items-center gap-2">
            <span class="w-7 h-7 rounded-lg bg-amber-50 text-amber-600 flex items-center justify-center">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9 5.25h.008v.008H12v-.008z"/>
              </svg>
            </span>
            Possible Causes
          </h4>
          <ul class="space-y-2.5">${renderListItems(diseaseInfo.causes, "bg-amber-500")}</ul>
        </div>

        <div class="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm">
          <h4 class="text-sm font-bold text-slate-900 uppercase tracking-wider mb-3 flex items-center gap-2">
            <span class="w-7 h-7 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z"/>
              </svg>
            </span>
            Recommended Prevention
          </h4>
          <ul class="space-y-2.5">${renderListItems(diseaseInfo.prevention, "bg-emerald-500")}</ul>
        </div>

        <div class="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm">
          <h4 class="text-sm font-bold text-slate-900 uppercase tracking-wider mb-3 flex items-center gap-2">
            <span class="w-7 h-7 rounded-lg bg-sky-50 text-sky-600 flex items-center justify-center">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23-.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5"/>
              </svg>
            </span>
            Suggested Treatment
          </h4>
          <ul class="space-y-2.5">${renderListItems(diseaseInfo.treatment, "bg-sky-500")}</ul>
        </div>
      </div>
    `
    : "";

  container.innerHTML = `
    <!-- Top Summary Banner -->
    <div class="rounded-2xl border p-6 sm:p-7 ${statusTheme.bannerBg} shadow-sm mb-6">
      <div class="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
        <div class="space-y-2.5">
          <div class="flex flex-wrap items-center gap-2.5">
            <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold border ${statusTheme.badgeBg}">
              <span class="w-2 h-2 rounded-full ${statusTheme.barBg}"></span>
              Status: ${esc(statusTheme.label)}
            </span>
            <span class="text-xs text-slate-600 font-medium">
              Analyzed ${esc(formatDateTime(result.timestamp || result.created_at))}
            </span>
            ${
              result.record_id || result.id
                ? `<span class="text-xs text-slate-500 font-mono">Record #${esc(result.record_id || result.id)}</span>`
                : ""
            }
          </div>

          <div class="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-2 pt-1">
            <div>
              <div class="text-xs uppercase tracking-wider font-semibold text-slate-500">Detected Plant</div>
              <div class="text-2xl sm:text-3xl font-extrabold text-slate-900">${esc(result.plant_name)}</div>
            </div>
            <div>
              <div class="text-xs uppercase tracking-wider font-semibold text-slate-500">Identified Condition</div>
              <div class="text-2xl sm:text-3xl font-extrabold ${
                isHealthy ? "text-emerald-700" : isUncertain ? "text-amber-700" : "text-rose-700"
              }">
                ${esc(result.disease_name)}
              </div>
            </div>
          </div>
        </div>

        <!-- Circular Confidence Indicator -->
        <div class="flex items-center gap-4 bg-white/90 rounded-2xl p-4 border border-white shadow-sm shrink-0">
          <div class="relative w-28 h-28 flex items-center justify-center">
            <svg class="w-28 h-28 -rotate-90" viewBox="0 0 112 112">
              <circle cx="56" cy="56" r="${radius}" fill="none" stroke="#e2e8f0" stroke-width="10" />
              <circle
                cx="56"
                cy="56"
                r="${radius}"
                fill="none"
                stroke="${statusTheme.ringStroke}"
                stroke-width="10"
                stroke-linecap="round"
                stroke-dasharray="${circumference.toFixed(2)}"
                stroke-dashoffset="${strokeOffset.toFixed(2)}"
                class="confidence-ring-progress"
              />
            </svg>
            <div class="absolute inset-0 flex flex-col items-center justify-center">
              <span class="text-xl font-extrabold text-slate-900">${clampedConf.toFixed(1)}%</span>
              <span class="text-[10px] uppercase tracking-wider font-semibold text-slate-500">Confidence</span>
            </div>
          </div>
          <div class="text-xs space-y-1 text-slate-600 max-w-[130px]">
            <div class="font-semibold text-slate-800">AI Model Output</div>
            <div>Architecture: <span class="font-mono text-slate-800">${esc(result.model_architecture || "mobilenet_v2")}</span></div>
            <div>Threshold: <span class="font-semibold">${esc(result.confidence_threshold || 60)}%</span></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Main Two-Column Content -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
      <!-- Left Column: Original Image + Top-3 Probabilities -->
      <div class="lg:col-span-4 space-y-5">
        <div class="bg-white rounded-2xl border border-slate-200 p-4 shadow-sm">
          <div class="flex items-center justify-between mb-3">
            <h3 class="text-sm font-bold text-slate-900">Original Leaf Image</h3>
            <span class="text-xs text-slate-500 truncate max-w-[160px]">${esc(result.image_filename || "")}</span>
          </div>
          <div class="rounded-xl overflow-hidden bg-slate-100 border border-slate-200 aspect-square flex items-center justify-center">
            <img src="${esc(imageUrl)}" alt="Analyzed plant leaf" class="w-full h-full object-contain" />
          </div>
          ${diagnosticsHtml}
        </div>

        <div class="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm">
          <h3 class="text-sm font-bold text-slate-900 mb-4">Top Model Predictions</h3>
          <div class="space-y-3.5">${topPredHtml}</div>
        </div>
      </div>

      <!-- Right Column: Detailed Symptoms, Causes, Prevention & Treatment -->
      <div class="lg:col-span-8 space-y-5">
        ${knowledgeSectionHtml}

        <!-- Action Bar -->
        <div class="flex flex-wrap items-center justify-between gap-3 pt-2 no-print">
          <div class="flex flex-wrap items-center gap-3">
            <button id="res-scan-another-btn" class="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-semibold text-sm shadow-sm transition">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M6.827 6.175A2.31 2.31 0 015.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 00-1.134-.175 2.31 2.31 0 01-1.64-1.055l-.822-1.316a2.192 2.192 0 00-1.736-1.039 48.774 48.774 0 00-5.232 0 2.192 2.192 0 00-1.736 1.039l-.821 1.316z"/>
                <path stroke-linecap="round" stroke-linejoin="round" d="M16.5 12.75a4.5 4.5 0 11-9 0 4.5 4.5 0 019 0z"/>
              </svg>
              Scan Another Leaf
            </button>
            <button id="res-view-history-btn" class="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 font-semibold text-sm transition">
              View History
            </button>
          </div>
          <div class="flex items-center gap-2.5">
            <button id="res-export-json-btn" class="inline-flex items-center gap-1.5 px-4 py-2.5 rounded-xl bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 font-medium text-sm transition">
              Export JSON
            </button>
            <button id="res-print-btn" class="inline-flex items-center gap-1.5 px-4 py-2.5 rounded-xl bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 font-medium text-sm transition">
              Print Report
            </button>
          </div>
        </div>
      </div>
    </div>
  `;

  container.querySelector("#res-scan-another-btn")?.addEventListener("click", () => {
    if (onScanAgain) onScanAgain();
  });
  container.querySelector("#res-view-history-btn")?.addEventListener("click", () => {
    if (onOpenHistory) onOpenHistory();
  });
  container.querySelector("#res-print-btn")?.addEventListener("click", () => {
    window.print();
  });
  container.querySelector("#res-export-json-btn")?.addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `plant-diagnosis-${result.record_id || result.id || "report"}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });
}
