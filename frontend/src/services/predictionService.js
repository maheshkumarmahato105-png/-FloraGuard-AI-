import { API_V1_PREFIX, apiUpload, DEFAULT_TIMEOUT_MS } from './api';

const CACHE_PREFIX = 'plantdiseaseai_result_';

/**
 * Submit a plant leaf image to POST /api/v1/predict using multipart FormData.
 * Never manually sets multipart Content-Type boundaries.
 */
export function submitPrediction(
  file,
  { generateExplanation = true, onProgress, timeoutMs = DEFAULT_TIMEOUT_MS, signal } = {}
) {
  const formData = new FormData();
  const filename = file?.name || 'captured-leaf.jpg';
  formData.append('file', file, filename);

  const query = `?generate_explanation=${generateExplanation ? 'true' : 'false'}`;
  return apiUpload(`${API_V1_PREFIX}/predict${query}`, formData, {
    onProgress,
    timeoutMs,
    signal,
  });
}

/**
 * Cache a real prediction response + local image preview in sessionStorage keyed by request_id
 * so refreshing /results/:requestId preserves disease_information and the original preview
 * (which the backend does not retain when RETAIN_UPLOADED_IMAGES=false).
 */
export function savePredictionCache(requestId, payload) {
  if (!requestId || !payload) return;
  try {
    window.sessionStorage.setItem(`${CACHE_PREFIX}${requestId}`, JSON.stringify(payload));
  } catch {
    // Ignore storage quota errors (e.g., very large data URLs)
    try {
      const { previewDataUrl, ...withoutImage } = payload;
      window.sessionStorage.setItem(`${CACHE_PREFIX}${requestId}`, JSON.stringify(withoutImage));
    } catch {
      // Give up silently
    }
  }
}

export function readPredictionCache(requestId) {
  if (!requestId) return null;
  try {
    const raw = window.sessionStorage.getItem(`${CACHE_PREFIX}${requestId}`);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function removePredictionCache(requestId) {
  if (!requestId) return;
  try {
    window.sessionStorage.removeItem(`${CACHE_PREFIX}${requestId}`);
  } catch {
    // Ignore
  }
}
