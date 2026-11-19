import { useCallback, useEffect, useState } from 'react';
import { getHealth, getSystemInfo } from '../services/systemService';

/**
 * Polls / fetches real backend health and system info from Phase 7 FastAPI endpoints.
 */
export function useBackendHealth({ autoFetch = true } = {}) {
  const [health, setHealth] = useState(null);
  const [systemInfo, setSystemInfo] = useState(null);
  const [loading, setLoading] = useState(autoFetch);
  const [error, setError] = useState(null);
  const [lastCheckedAt, setLastCheckedAt] = useState(null);

  const refresh = useCallback(async (signal) => {
    setLoading(true);
    setError(null);
    try {
      const [healthData, infoData] = await Promise.all([
        getHealth({ signal }),
        getSystemInfo({ signal }),
      ]);
      setHealth(healthData);
      setSystemInfo(infoData);
      setLastCheckedAt(new Date().toISOString());
    } catch (err) {
      if (err?.errorCode === 'aborted') return;
      setHealth(null);
      setSystemInfo(null);
      setError(err?.message || 'Unable to connect to the backend service.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!autoFetch) return undefined;
    const controller = new AbortController();
    refresh(controller.signal);
    return () => controller.abort();
  }, [autoFetch, refresh]);

  const isOnline = Boolean(health && health.api_status === 'online');
  const isModelReady = Boolean(health?.model_ready);

  return {
    health,
    systemInfo,
    loading,
    error,
    isOnline,
    isModelReady,
    lastCheckedAt,
    refresh: () => refresh(),
  };
}
