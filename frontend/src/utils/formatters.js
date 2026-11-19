/**
 * Formatting helpers for PlantDiseaseAI API responses.
 * Note: Backend returns confidence scores in [0.0, 1.0].
 */

export function formatConfidence(confidence) {
  if (typeof confidence !== 'number' || Number.isNaN(confidence)) {
    return '—';
  }
  const pct = confidence <= 1 ? confidence * 100 : confidence;
  return `${Math.max(0, Math.min(100, pct)).toFixed(1)}%`;
}

export function confidenceToPercent(confidence) {
  if (typeof confidence !== 'number' || Number.isNaN(confidence)) {
    return 0;
  }
  const pct = confidence <= 1 ? confidence * 100 : confidence;
  return Math.max(0, Math.min(100, pct));
}

export function formatTimestamp(isoString) {
  if (!isoString) return '—';
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return String(isoString);
  return new Intl.DateTimeFormat(undefined, {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}

export function formatBytes(bytes) {
  if (typeof bytes !== 'number' || Number.isNaN(bytes) || bytes < 0) return '—';
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const idx = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  const val = bytes / Math.pow(1024, idx);
  return `${val >= 10 || idx === 0 ? val.toFixed(0) : val.toFixed(1)} ${units[idx]}`;
}

function toTitleWords(text) {
  return text
    .replace(/_/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .split(' ')
    .map((word) => (word ? word.charAt(0).toUpperCase() + word.slice(1) : ''))
    .join(' ');
}

export function formatClassLabel(rawClass) {
  if (!rawClass || typeof rawClass !== 'string') return 'Unclassified';
  const trimmed = rawClass.trim();
  if (!trimmed) return 'Unclassified';

  // Support both PlantVillage raw ('___') and Phase 2 canonical ('__') separators
  const separator = trimmed.includes('___') ? '___' : trimmed.includes('__') ? '__' : null;
  if (separator) {
    const [plantPart, diseasePart] = trimmed.split(separator, 2);
    const plant = toTitleWords(plantPart || '');
    const disease = toTitleWords(diseasePart || '');
    if (plant && disease) {
      return `${plant} — ${disease}`;
    }
  }
  if (!trimmed.includes('_')) return trimmed;
  return toTitleWords(trimmed);
}

export function truncateId(id, maxLen = 12) {
  if (!id || typeof id !== 'string') return '—';
  if (id.length <= maxLen) return id;
  return `${id.slice(0, maxLen)}…`;
}

export function formatMetric(value, decimals = 3) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—';
  return value.toFixed(decimals);
}
