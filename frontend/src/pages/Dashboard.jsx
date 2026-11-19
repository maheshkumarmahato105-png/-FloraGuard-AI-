import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { AlertTriangle, RefreshCw, ScanSearch, Sparkles } from 'lucide-react';
import PageContainer from '../components/layout/PageContainer';
import DashboardStats from '../components/dashboard/DashboardStats';
import RecentScans from '../components/dashboard/RecentScans';
import QuickActions from '../components/dashboard/QuickActions';
import ErrorMessage from '../components/common/ErrorMessage';
import { fetchDashboardSummary } from '../services/historyService';

export function Dashboard({ healthState }) {
  const { health, systemInfo, loading: healthLoading, error: healthError, refresh: refreshHealth } = healthState;

  const [summary, setSummary] = useState(null);
  const [summaryLoading, setSummaryLoading] = useState(true);
  const [summaryError, setSummaryError] = useState(null);

  const loadSummary = useCallback(async (signal) => {
    setSummaryLoading(true);
    setSummaryError(null);
    try {
      const data = await fetchDashboardSummary({ signal });
      setSummary(data);
    } catch (err) {
      if (err?.errorCode === 'aborted') return;
      setSummary(null);
      setSummaryError(err?.message || 'Unable to load scan statistics from the backend.');
    } finally {
      setSummaryLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    loadSummary(controller.signal);
    return () => controller.abort();
  }, [loadSummary]);

  const handleRefreshAll = () => {
    refreshHealth();
    loadSummary();
  };

  return (
    <PageContainer>
      <section className="welcome-banner" aria-label="Welcome to PlantDiseaseAI">
        <div style={{ position: 'relative', zIndex: 1 }}>
          <div className="welcome-eyebrow">
            <Sparkles size={14} aria-hidden="true" />
            <span>AgriVision-CNN Research Workbench</span>
          </div>
          <h1 className="welcome-title">Plant Disease Detection Platform</h1>
          <p className="welcome-desc">
            Analyze plant leaf photographs with calibrated uncertainty checks, Grad-CAM model
            attention maps, and persistent SQLite prediction history.
          </p>
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, position: 'relative', zIndex: 1 }}>
          <Link to="/detection" className="btn btn-primary">
            <ScanSearch size={17} aria-hidden="true" />
            Quick Scan
          </Link>
          <button type="button" className="btn btn-secondary" onClick={handleRefreshAll}>
            <RefreshCw size={15} aria-hidden="true" />
            Refresh Data
          </button>
        </div>
      </section>

      {healthError ? (
        <div style={{ marginBottom: 20 }}>
          <ErrorMessage
            title="Backend connection offline"
            message={healthError}
            recoveryHint="Start the Phase 7 FastAPI server and click Refresh Data to reconnect."
            onRetry={handleRefreshAll}
            retryLabel="Retry connection"
          />
        </div>
      ) : null}

      {!healthLoading && health && !health.model_ready ? (
        <div className="alert alert-warning" role="status" style={{ marginBottom: 20 }}>
          <AlertTriangle size={20} className="alert-icon" aria-hidden="true" />
          <div className="alert-body">
            <p className="alert-title">Inference model checkpoint is not yet loaded</p>
            <p className="alert-text">
              {health.model_status_detail ||
                'The FastAPI backend is online, but no trained AgriVision-CNN checkpoint is present yet. History, health, and system metadata endpoints remain active.'}
            </p>
          </div>
        </div>
      ) : null}

      {!healthError && summaryError ? (
        <div style={{ marginBottom: 20 }}>
          <ErrorMessage
            title="Unable to load dashboard statistics"
            message={summaryError}
            onRetry={() => loadSummary()}
          />
        </div>
      ) : null}

      <DashboardStats summary={summary} loading={summaryLoading} />

      <div className="two-col-grid">
        <RecentScans
          scans={summary?.recentScans || []}
          loading={summaryLoading}
          error={summaryError || healthError}
        />
        <QuickActions health={health} systemInfo={systemInfo} loading={healthLoading} />
      </div>
    </PageContainer>
  );
}

export default Dashboard;
