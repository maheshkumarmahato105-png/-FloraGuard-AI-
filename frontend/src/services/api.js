/**
 * Central API client for the PlantDiseaseAI Phase 7 FastAPI backend.
 * Uses VITE_API_BASE_URL (default: http://127.0.0.1:8001).
 */

const RAW_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8001';
export const API_BASE_URL = RAW_BASE_URL.replace(/\/+$/, '');
export const API_V1_PREFIX = '/api/v1';

const PARSED_TIMEOUT = Number(import.meta.env.VITE_API_TIMEOUT_MS);
export const DEFAULT_TIMEOUT_MS =
  Number.isFinite(PARSED_TIMEOUT) && PARSED_TIMEOUT > 0 ? PARSED_TIMEOUT : 30000;
export const STANDARD_TIMEOUT_MS = Math.min(DEFAULT_TIMEOUT_MS, 15000);

export class ApiError extends Error {
  constructor({
    message,
    status = 0,
    errorCode = 'unknown_error',
    requestId = null,
    isNetworkError = false,
    isTimeout = false,
  }) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.errorCode = errorCode;
    this.requestId = requestId;
    this.isNetworkError = isNetworkError;
    this.isTimeout = isTimeout;
  }
}

export function resolveApiUrl(path) {
  if (!path) return '';
  if (/^https?:\/\//i.test(path)) return path;
  const normalized = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE_URL}${normalized}`;
}

export function getFriendlyErrorMessage(status, payload = null) {
  const backendDetail =
    payload && typeof payload.detail === 'string' && payload.detail.trim()
      ? payload.detail.trim()
      : null;

  switch (status) {
    case 400:
      return backendDetail || 'The uploaded image could not be processed. Please verify the file is a valid leaf photograph.';
    case 404:
      return backendDetail || 'The requested record was not found.';
    case 413:
      return backendDetail || 'The selected image exceeds the maximum allowed upload size.';
    case 415:
      return backendDetail || 'Unsupported file type. Please upload a JPG, JPEG, PNG, or WEBP image.';
    case 422:
      return backendDetail || 'The request was invalid. Please check your image selection and try again.';
    case 429:
      return backendDetail || 'Too many requests sent recently. Please wait a moment before trying again.';
    case 503:
      return (
        backendDetail ||
        'The AI inference model is currently unavailable on the backend. Verify that a trained AgriVision-CNN checkpoint is configured.'
      );
    case 500:
    case 502:
    case 504:
      return 'The backend encountered an unexpected server error. Please try again shortly.';
    default:
      return backendDetail || 'Unable to complete the request. Please try again.';
  }
}

async function parseJsonSafely(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

export async function apiRequest(
  path,
  { method = 'GET', headers = {}, body = undefined, timeoutMs = STANDARD_TIMEOUT_MS, signal } = {}
) {
  const url = resolveApiUrl(path);
  const controller = new AbortController();
  let timedOut = false;

  const timerId = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  const onExternalAbort = () => controller.abort();
  if (signal) {
    if (signal.aborted) {
      window.clearTimeout(timerId);
      throw new ApiError({
        message: 'Request was cancelled.',
        errorCode: 'aborted',
      });
    }
    signal.addEventListener('abort', onExternalAbort, { once: true });
  }

  try {
    const response = await fetch(url, {
      method,
      headers: {
        Accept: 'application/json',
        ...headers,
      },
      body,
      signal: controller.signal,
    });

    const payload = await parseJsonSafely(response);

    if (!response.ok) {
      throw new ApiError({
        message: getFriendlyErrorMessage(response.status, payload),
        status: response.status,
        errorCode: payload?.error_code || `http_${response.status}`,
        requestId: payload?.request_id || response.headers.get('X-Request-ID') || null,
      });
    }

    return payload;
  } catch (err) {
    if (err instanceof ApiError) {
      throw err;
    }
    if (err?.name === 'AbortError') {
      if (timedOut) {
        throw new ApiError({
          message: `The request timed out after ${Math.round(timeoutMs / 1000)}s. Verify the backend is responsive and try again.`,
          errorCode: 'timeout',
          isTimeout: true,
        });
      }
      throw new ApiError({
        message: 'The request was cancelled.',
        errorCode: 'aborted',
      });
    }
    throw new ApiError({
      message: `Unable to reach the PlantDiseaseAI backend at ${API_BASE_URL}. Confirm the FastAPI server is running.`,
      errorCode: 'network_error',
      isNetworkError: true,
    });
  } finally {
    window.clearTimeout(timerId);
    if (signal) {
      signal.removeEventListener('abort', onExternalAbort);
    }
  }
}

/**
 * Sends a FormData payload using XMLHttpRequest to support real upload progress events
 * without manually setting multipart Content-Type boundaries.
 */
export function apiUpload(
  path,
  formData,
  { onProgress, timeoutMs = DEFAULT_TIMEOUT_MS, signal } = {}
) {
  const url = resolveApiUrl(path);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url, true);
    xhr.timeout = timeoutMs;
    xhr.setRequestHeader('Accept', 'application/json');

    const onExternalAbort = () => {
      xhr.abort();
    };

    if (signal) {
      if (signal.aborted) {
        reject(
          new ApiError({
            message: 'Upload was cancelled.',
            errorCode: 'aborted',
          })
        );
        return;
      }
      signal.addEventListener('abort', onExternalAbort, { once: true });
    }

    const cleanup = () => {
      if (signal) {
        signal.removeEventListener('abort', onExternalAbort);
      }
    };

    if (xhr.upload && typeof onProgress === 'function') {
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && event.total > 0) {
          const percent = Math.round((event.loaded / event.total) * 100);
          onProgress(Math.min(100, Math.max(0, percent)));
        }
      };
    }

    xhr.onload = () => {
      cleanup();
      let payload = null;
      try {
        payload = xhr.responseText ? JSON.parse(xhr.responseText) : null;
      } catch {
        payload = null;
      }

      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(payload);
      } else {
        reject(
          new ApiError({
            message: getFriendlyErrorMessage(xhr.status, payload),
            status: xhr.status,
            errorCode: payload?.error_code || `http_${xhr.status}`,
            requestId: payload?.request_id || xhr.getResponseHeader('X-Request-ID') || null,
          })
        );
      }
    };

    xhr.onerror = () => {
      cleanup();
      reject(
        new ApiError({
          message: `Unable to reach the PlantDiseaseAI backend at ${API_BASE_URL}. Confirm the FastAPI server is running.`,
          errorCode: 'network_error',
          isNetworkError: true,
        })
      );
    };

    xhr.ontimeout = () => {
      cleanup();
      reject(
        new ApiError({
          message: `The prediction request timed out after ${Math.round(timeoutMs / 1000)}s. Please try again.`,
          errorCode: 'timeout',
          isTimeout: true,
        })
      );
    };

    xhr.onabort = () => {
      cleanup();
      reject(
        new ApiError({
          message: 'The prediction request was cancelled.',
          errorCode: 'aborted',
        })
      );
    };

    xhr.send(formData);
  });
}
