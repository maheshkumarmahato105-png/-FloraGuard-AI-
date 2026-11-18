/**
 * Renders the responsive Detection History table (desktop) & card list (mobile).
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
  if (!isoString) return "—";
  const d = new Date(isoString);
  if (Number.isNaN(d.getTime())) return isoString;
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function statusBadge(status) {
  if (status === "Healthy") {
    return `<span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200"><span class="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>Healthy</span>`;
  }
  if (status === "Uncertain") {
    return `<span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-amber-50 text-amber-700 border border-amber-200"><span class="w-1.5 h-1.5 rounded-full bg-amber-500"></span>Uncertain</span>`;
  }
  return `<span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-rose-50 text-rose-700 border border-rose-200"><span class="w-1.5 h-1.5 rounded-full bg-rose-500"></span>Infected</span>`;
}

export function renderHistoryTable(container, items, { onView, onDelete, onStartDetect } = {}) {
  if (!items || items.length === 0) {
    container.innerHTML = `
      <div class="bg-white rounded-2xl border border-slate-200 p-12 text-center shadow-sm">
        <div class="w-14 h-14 rounded-2xl bg-slate-100 text-slate-400 flex items-center justify-center mx-auto mb-4">
          <svg class="w-7 h-7" fill="none" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z"/>
          </svg>
        </div>
        <h3 class="text-lg font-bold text-slate-900 mb-1">No Detection Records Found</h3>
        <p class="text-sm text-slate-600 max-w-md mx-auto mb-5">
          Analyzed plant leaf diagnostics are saved automatically in the database and will appear here.
        </p>
        <button id="history-empty-detect-btn" class="px-5 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-semibold text-sm transition">
          Analyze First Plant Leaf
        </button>
      </div>
    `;
    container.querySelector("#history-empty-detect-btn")?.addEventListener("click", () => {
      if (onStartDetect) onStartDetect();
    });
    return;
  }

  const desktopRows = items
    .map((item) => {
      const conf = Number(item.confidence || 0);
      const barColor =
        item.status === "Healthy"
          ? "bg-emerald-500"
          : item.status === "Uncertain"
          ? "bg-amber-500"
          : "bg-rose-500";
      return `
        <tr class="hover:bg-slate-50/80 transition border-b border-slate-100 last:border-none">
          <td class="py-3.5 pl-5 pr-3">
            <div class="flex items-center gap-3">
              <img src="${esc(resolveAssetUrl(item.image_url))}" alt="${esc(item.plant_name)}" class="w-12 h-12 rounded-xl object-cover border border-slate-200 bg-slate-100 shrink-0" />
              <div>
                <div class="font-semibold text-slate-900 text-sm">${esc(item.plant_name)}</div>
                <div class="text-xs text-slate-500">${esc(formatDateTime(item.created_at))}</div>
              </div>
            </div>
          </td>
          <td class="py-3.5 px-3 text-sm font-medium text-slate-800">${esc(item.disease_name)}</td>
          <td class="py-3.5 px-3">${statusBadge(item.status)}</td>
          <td class="py-3.5 px-3">
            <div class="flex items-center gap-2.5 min-w-[120px]">
              <div class="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                <div class="h-full ${barColor} rounded-full" style="width: ${Math.min(100, Math.max(4, conf))}%"></div>
              </div>
              <span class="text-xs font-bold text-slate-700 w-11 text-right">${conf.toFixed(1)}%</span>
            </div>
          </td>
          <td class="py-3.5 px-3 text-xs text-slate-500 capitalize">${esc(item.input_source || "upload")}</td>
          <td class="py-3.5 pl-3 pr-5 text-right whitespace-nowrap">
            <button data-action="view" data-id="${item.id}" class="px-3 py-1.5 rounded-lg bg-emerald-50 hover:bg-emerald-100 text-emerald-700 text-xs font-semibold mr-1.5 transition">
              View
            </button>
            <button data-action="delete" data-id="${item.id}" class="px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-rose-50 hover:text-rose-700 text-slate-600 text-xs font-medium transition">
              Delete
            </button>
          </td>
        </tr>
      `;
    })
    .join("");

  const mobileCards = items
    .map((item) => {
      const conf = Number(item.confidence || 0);
      return `
        <div class="bg-white rounded-2xl border border-slate-200 p-4 shadow-sm space-y-3">
          <div class="flex items-start gap-3">
            <img src="${esc(resolveAssetUrl(item.image_url))}" alt="${esc(item.plant_name)}" class="w-16 h-16 rounded-xl object-cover border border-slate-200 bg-slate-100 shrink-0" />
            <div class="flex-1 min-w-0">
              <div class="flex items-center justify-between gap-2">
                <h4 class="font-bold text-slate-900 truncate">${esc(item.plant_name)}</h4>
                ${statusBadge(item.status)}
              </div>
              <div class="text-sm font-medium text-slate-700 truncate mt-0.5">${esc(item.disease_name)}</div>
              <div class="text-xs text-slate-500 mt-1">${esc(formatDateTime(item.created_at))} · Confidence: <strong class="text-slate-700">${conf.toFixed(1)}%</strong></div>
            </div>
          </div>
          <div class="flex items-center justify-end gap-2 pt-2 border-t border-slate-100">
            <button data-action="view" data-id="${item.id}" class="px-3.5 py-1.5 rounded-lg bg-emerald-600 text-white text-xs font-semibold">
              View Report
            </button>
            <button data-action="delete" data-id="${item.id}" class="px-3 py-1.5 rounded-lg bg-slate-100 text-slate-700 text-xs font-medium">
              Delete
            </button>
          </div>
        </div>
      `;
    })
    .join("");

  container.innerHTML = `
    <!-- Desktop Table -->
    <div class="hidden md:block bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
      <div class="overflow-x-auto">
        <table class="w-full text-left border-collapse">
          <thead>
            <tr class="bg-slate-50 border-b border-slate-200 text-xs uppercase tracking-wider text-slate-500 font-semibold">
              <th class="py-3.5 pl-5 pr-3">Plant & Timestamp</th>
              <th class="py-3.5 px-3">Detected Condition</th>
              <th class="py-3.5 px-3">Status</th>
              <th class="py-3.5 px-3">Confidence</th>
              <th class="py-3.5 px-3">Source</th>
              <th class="py-3.5 pl-3 pr-5 text-right">Actions</th>
            </tr>
          </thead>
          <tbody>${desktopRows}</tbody>
        </table>
      </div>
    </div>

    <!-- Mobile Cards -->
    <div class="md:hidden space-y-3">${mobileCards}</div>
  `;

  container.querySelectorAll("[data-action='view']").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = Number(btn.getAttribute("data-id"));
      const item = items.find((x) => x.id === id);
      if (item && onView) onView(item);
    });
  });

  container.querySelectorAll("[data-action='delete']").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = Number(btn.getAttribute("data-id"));
      if (onDelete) onDelete(id);
    });
  });
}
