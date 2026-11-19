import { API_V1_PREFIX, apiRequest } from './api';

export function getHealth({ signal } = {}) {
  return apiRequest(`${API_V1_PREFIX}/health`, { method: 'GET', signal });
}

export function getSystemInfo({ signal } = {}) {
  return apiRequest(`${API_V1_PREFIX}/system/info`, { method: 'GET', signal });
}
