/**
 * API client for communicating with the Python FastAPI backend.
 * Automatically uses the current origin when served via FastAPI, or falls back
 * to http://127.0.0.1:8000 if opened from a separate static dev server.
 */

const API_BASE = (() => {
  if (window.location.protocol === "file:") {
    return "http://127.0.0.1:8000";
  }
  // If running on a different dev port (e.g., Live Server 5500), target FastAPI on 8000
  if (window.location.port && !["8000", "80", "443"].includes(window.location.port)) {
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }
  return "";
})();

export function resolveAssetUrl(path) {
  if (!path) return "";
  if (path.startsWith("http://") || path.startsWith("https://") || path.startsWith("data:")) {
    return path;
  }
  return `${API_BASE}${path}`;
}

async function handleResponse(response) {
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    const detail =
      payload?.detail ||
      payload?.message ||
      `Request failed with status ${response.status}`;
    const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    err.status = response.status;
    err.payload = payload;
    throw err;
  }
  return payload;
}

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/api/health`, { cache: "no-store" });
  return handleResponse(res);
}

export async function fetchSupportedPlants() {
  const res = await fetch(`${API_BASE}/api/plants`, { cache: "no-store" });
  return handleResponse(res);
}

export async function fetchEvaluationReport() {
  const res = await fetch(`${API_BASE}/api/evaluation`, { cache: "no-store" });
  return handleResponse(res);
}

export async function uploadPlantImage(file) {
  const formData = new FormData();
  formData.append("file", file, file.name || "leaf.jpg");
  const res = await fetch(`${API_BASE}/api/upload`, {
    method: "POST",
    body: formData,
  });
  return handleResponse(res);
}

export async function predictPlantDisease(file, inputSource = "upload") {
  const formData = new FormData();
  formData.append("file", file, file.name || "leaf.jpg");
  formData.append("input_source", inputSource);
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    body: formData,
  });
  return handleResponse(res);
}

export async function fetchHistory({ plant = "", status = "", search = "", limit = 50, offset = 0 } = {}) {
  const params = new URLSearchParams();
  if (plant) params.set("plant", plant);
  if (status) params.set("status", status);
  if (search) params.set("search", search);
  params.set("limit", String(limit));
  params.set("offset", String(offset));

  const res = await fetch(`${API_BASE}/api/history?${params.toString()}`, {
    cache: "no-store",
  });
  return handleResponse(res);
}

export async function fetchHistoryRecord(recordId) {
  const res = await fetch(`${API_BASE}/api/history/${recordId}`, {
    cache: "no-store",
  });
  return handleResponse(res);
}

export async function deleteHistoryRecord(recordId) {
  const res = await fetch(`${API_BASE}/api/history/${recordId}`, {
    method: "DELETE",
  });
  return handleResponse(res);
}
