import { formatBytes } from './formatters';

export const DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // Matches backend default (10 MB)
export const DEFAULT_ALLOWED_FORMATS = ['JPG', 'JPEG', 'PNG', 'WEBP'];

const EXTENSION_MAP = {
  JPG: '.jpg',
  JPEG: '.jpeg',
  PNG: '.png',
  WEBP: '.webp',
};

const MIME_MAP = {
  JPG: 'image/jpeg',
  JPEG: 'image/jpeg',
  PNG: 'image/png',
  WEBP: 'image/webp',
};

/**
 * Validates a File/Blob against supported image formats and max upload size
 * returned by GET /api/v1/system/info (with safe fallback matching backend defaults).
 */
export function validateImageFile(
  file,
  {
    maxBytes = DEFAULT_MAX_UPLOAD_BYTES,
    supportedFormats = DEFAULT_ALLOWED_FORMATS,
  } = {}
) {
  if (!file) {
    return { valid: false, error: 'Please select or capture a plant leaf image first.' };
  }

  const normalizedFormats = (supportedFormats?.length ? supportedFormats : DEFAULT_ALLOWED_FORMATS).map((f) =>
    String(f).toUpperCase()
  );

  const allowedExtensions = new Set(
    normalizedFormats.map((f) => EXTENSION_MAP[f]).filter(Boolean)
  );
  if (allowedExtensions.has('.jpeg')) allowedExtensions.add('.jpg');
  if (allowedExtensions.has('.jpg')) allowedExtensions.add('.jpeg');

  const allowedMimes = new Set(
    normalizedFormats.map((f) => MIME_MAP[f]).filter(Boolean)
  );
  if (allowedMimes.has('image/jpeg')) allowedMimes.add('image/jpg');

  const fileName = (file.name || '').toLowerCase();
  const extMatch = fileName.includes('.') ? `.${fileName.split('.').pop()}` : '';
  const mimeType = (file.type || '').toLowerCase();

  const extValid = extMatch ? allowedExtensions.has(extMatch) : false;
  const mimeValid = mimeType ? allowedMimes.has(mimeType) : false;

  if (!extValid && !mimeValid) {
    return {
      valid: false,
      error: `Unsupported image format. Please use ${normalizedFormats.join(', ')}.`,
    };
  }

  if (file.size === 0) {
    return {
      valid: false,
      error: 'The selected image file is empty. Please choose a valid image.',
    };
  }

  if (file.size > maxBytes) {
    return {
      valid: false,
      error: `Image exceeds the maximum allowed size of ${formatBytes(maxBytes)} (selected: ${formatBytes(file.size)}).`,
    };
  }

  return { valid: true, error: null };
}

export function buildAcceptAttribute(supportedFormats = DEFAULT_ALLOWED_FORMATS) {
  const normalized = (supportedFormats?.length ? supportedFormats : DEFAULT_ALLOWED_FORMATS).map((f) =>
    String(f).toUpperCase()
  );
  const tokens = new Set();
  normalized.forEach((fmt) => {
    if (EXTENSION_MAP[fmt]) tokens.add(EXTENSION_MAP[fmt]);
    if (MIME_MAP[fmt]) tokens.add(MIME_MAP[fmt]);
  });
  return Array.from(tokens).join(',');
}
