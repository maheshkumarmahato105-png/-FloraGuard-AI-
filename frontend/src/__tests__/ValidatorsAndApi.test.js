import { afterEach, describe, expect, it, vi } from 'vitest';
import { validateImageFile, DEFAULT_MAX_UPLOAD_BYTES } from '../utils/validators';
import { formatBytes, formatClassLabel, formatConfidence } from '../utils/formatters';
import { apiRequest, ApiError, getFriendlyErrorMessage } from '../services/api';

describe('Image validation & formatters', () => {
  it('accepts valid JPG, PNG, and WEBP files within backend limit', () => {
    const jpg = new File([new Uint8Array(1024)], 'leaf.jpg', { type: 'image/jpeg' });
    const png = new File([new Uint8Array(2048)], 'leaf.png', { type: 'image/png' });
    const webp = new File([new Uint8Array(4096)], 'leaf.webp', { type: 'image/webp' });

    expect(validateImageFile(jpg).valid).toBe(true);
    expect(validateImageFile(png).valid).toBe(true);
    expect(validateImageFile(webp).valid).toBe(true);
  });

  it('rejects unsupported file formats with a clear message', () => {
    const pdf = new File([new Uint8Array(512)], 'report.pdf', { type: 'application/pdf' });
    const res = validateImageFile(pdf);
    expect(res.valid).toBe(false);
    expect(res.error).toMatch(/unsupported image format/i);
  });

  it('rejects empty files and oversized files exceeding backend limit', () => {
    const empty = new File([], 'empty.jpg', { type: 'image/jpeg' });
    expect(validateImageFile(empty).valid).toBe(false);

    const oversized = new File([new Uint8Array(1024)], 'huge.jpg', { type: 'image/jpeg' });
    Object.defineProperty(oversized, 'size', { value: DEFAULT_MAX_UPLOAD_BYTES + 1 });
    const res = validateImageFile(oversized);
    expect(res.valid).toBe(false);
    expect(res.error).toMatch(/exceeds the maximum allowed size/i);
  });

  it('formats 0..1 confidence values, bytes, and both canonical (__) and raw (___) class names', () => {
    expect(formatConfidence(0.945)).toBe('94.5%');
    expect(formatBytes(10 * 1024 * 1024)).toBe('10 MB');
    expect(formatClassLabel('Tomato___Early_blight')).toBe('Tomato — Early Blight');
    expect(formatClassLabel('tomato__early_blight')).toBe('Tomato — Early Blight');
    expect(formatClassLabel('apple__healthy')).toBe('Apple — Healthy');
  });
});

describe('API client error handling', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('maps HTTP 400, 413, 415, 422, and 503 status codes to user-friendly messages', () => {
    expect(getFriendlyErrorMessage(400)).toMatch(/could not be processed/i);
    expect(getFriendlyErrorMessage(413)).toMatch(/maximum allowed upload size/i);
    expect(getFriendlyErrorMessage(415)).toMatch(/unsupported file type/i);
    expect(getFriendlyErrorMessage(422)).toMatch(/request was invalid/i);
    expect(getFriendlyErrorMessage(503)).toMatch(/currently unavailable/i);
  });

  it('throws normalized ApiError on HTTP 503 from backend', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: false,
      status: 503,
      headers: new Headers({ 'X-Request-ID': 'req-503' }),
      json: async () => ({
        request_id: 'req-503',
        status: 'error',
        error_code: 'model_unavailable',
        detail: 'Trained model checkpoint is not available yet.',
      }),
    });

    await expect(apiRequest('/api/v1/health')).rejects.toMatchObject({
      name: 'ApiError',
      status: 503,
      errorCode: 'model_unavailable',
      requestId: 'req-503',
      message: 'Trained model checkpoint is not available yet.',
    });
  });

  it('throws network ApiError when backend is unreachable', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('Failed to fetch'));

    try {
      await apiRequest('/api/v1/health');
      expect.fail('Expected apiRequest to reject');
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError);
      expect(err.isNetworkError).toBe(true);
      expect(err.message).toMatch(/unable to reach/i);
    }
  });
});
