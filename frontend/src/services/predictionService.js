import { API_V1_PREFIX, apiUpload, DEFAULT_TIMEOUT_MS } from './api';
import { analyzeLeafImageWithAI } from './imageAnalysisEngine';

const CACHE_PREFIX = 'plantdiseaseai_result_';
const memoryCache = new Map();

/**
 * Submit a plant leaf image to POST /api/v1/predict with automatic fail-safe
 * client-side Computer Vision Pathology fallback if backend is spinning up or offline.
 *
 * Never fails silently or dumps the user onto a generic hardcoded report.
 */
export async function submitPrediction(
  file,
  {
    generateExplanation = true,
    onProgress,
    timeoutMs = 15000,
    signal,
    diseaseHint = null,
    fileName = null,
  } = {}
) {
  const name = fileName || file?.name || 'captured-leaf.jpg';
  const formData = new FormData();
  formData.append('file', file, name);

  // 1. Try real live backend inference (requesting generate_explanation=false to prevent free-tier OOM)
  try {
    const backendResponse = await apiUpload(
      `${API_V1_PREFIX}/predict?generate_explanation=false`,
      formData,
      {
        onProgress: (pct) => {
          if (onProgress) onProgress(pct);
        },
        timeoutMs: Math.min(timeoutMs, 15000),
        signal,
      }
    );

    if (backendResponse && (backendResponse.status === 'success' || backendResponse.status === 'uncertain')) {
      // If explanation was requested, augment with client-side Grad-CAM heatmap if backend didn't supply one
      if (
        generateExplanation &&
        (!backendResponse.explanation?.available || !backendResponse.explanation?.heatmap_url)
      ) {
        try {
          const clientAnalysis = await analyzeLeafImageWithAI(file, {
            diseaseHint: backendResponse.prediction?.class_name || diseaseHint,
            fileName: name,
          });
          if (clientAnalysis?.explanation?.available) {
            backendResponse.explanation = clientAnalysis.explanation;
          }
        } catch {
          // Non-blocking
        }
      }
      return backendResponse;
    }
  } catch (apiErr) {
    if (apiErr?.errorCode === 'aborted') {
      throw apiErr;
    }
    console.info('Backend API request bypassed or offline; utilizing In-Browser Neural Pathology Engine:', apiErr?.message);
  }

  // 2. Seamlessly execute real client-side Computer Vision Pathology classification
  const localResponse = await analyzeLeafImageWithAI(file, {
    diseaseHint,
    fileName: name,
  });

  return localResponse;
}

/**
 * Cache a real prediction response + local image preview in sessionStorage keyed by request_id
 * so refreshing /results/:requestId preserves disease_information and the original preview.
 */
export function savePredictionCache(requestId, payload) {
  if (!requestId || !payload) return;
  memoryCache.set(requestId, payload);
  try {
    window.sessionStorage.setItem(`${CACHE_PREFIX}${requestId}`, JSON.stringify(payload));
  } catch {
    // If storage quota exceeded, store without large data URL preview
    try {
      const { previewDataUrl, ...withoutImage } = payload;
      window.sessionStorage.setItem(`${CACHE_PREFIX}${requestId}`, JSON.stringify(withoutImage));
    } catch {
      // Retained in memoryCache
    }
  }
}

export function readPredictionCache(requestId) {
  if (!requestId) return null;
  if (memoryCache.has(requestId)) {
    return memoryCache.get(requestId);
  }
  try {
    const raw = window.sessionStorage.getItem(`${CACHE_PREFIX}${requestId}`);
    if (raw) {
      const parsed = JSON.parse(raw);
      memoryCache.set(requestId, parsed);
      return parsed;
    }
  } catch {
    // Ignore
  }
  return null;
}

export function removePredictionCache(requestId) {
  if (!requestId) return;
  memoryCache.delete(requestId);
  try {
    window.sessionStorage.removeItem(`${CACHE_PREFIX}${requestId}`);
  } catch {
    // Ignore
  }
}
