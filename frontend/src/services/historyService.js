import { API_V1_PREFIX, apiRequest } from './api';
import { removePredictionCache } from './predictionService';

export function listHistory({
  limit = 20,
  offset = 0,
  status = '',
  predictedClass = '',
  signal,
} = {}) {
  const params = new URLSearchParams();
  params.set('limit', String(limit));
  params.set('offset', String(offset));
  if (status && status !== 'all') {
    params.set('status', status);
  }
  if (predictedClass && predictedClass.trim()) {
    params.set('predicted_class', predictedClass.trim());
  }
  return apiRequest(`${API_V1_PREFIX}/history?${params.toString()}`, {
    method: 'GET',
    signal,
  });
}

export function getHistoryRecord(requestId, { signal } = {}) {
  return apiRequest(`${API_V1_PREFIX}/history/${encodeURIComponent(requestId)}`, {
    method: 'GET',
    signal,
  });
}

export async function deleteHistoryRecord(requestId) {
  const result = await apiRequest(`${API_V1_PREFIX}/history/${encodeURIComponent(requestId)}`, {
    method: 'DELETE',
  });
  removePredictionCache(requestId);
  return result;
}

/**
 * Computes real Dashboard statistics exclusively from actual backend history records.
 * Never invents or estimates fake numbers.
 */
export async function fetchDashboardSummary({ signal } = {}) {
  const [recentRes, uncertainRes, successRes] = await Promise.all([
    listHistory({ limit: 100, offset: 0, signal }),
    listHistory({ limit: 1, offset: 0, status: 'uncertain', signal }),
    listHistory({ limit: 100, offset: 0, status: 'success', signal }),
  ]);

  const totalScans = recentRes?.total ?? 0;
  const uncertainCount = uncertainRes?.total ?? 0;
  const successItems = successRes?.items ?? [];
  const healthyCount = successItems.filter((item) => item.is_healthy === true).length;
  const diseasedCount = successItems.filter((item) => item.is_healthy === false).length;

  return {
    totalScans,
    healthyCount,
    diseasedCount,
    uncertainCount,
    successTotal: successRes?.total ?? 0,
    statsSampled: (successRes?.total ?? 0) > successItems.length,
    recentScans: (recentRes?.items ?? []).slice(0, 6),
  };
}
