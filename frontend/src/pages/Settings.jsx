import { Cpu, Moon, Palette, RefreshCw, Server, Sun } from 'lucide-react';
import PageContainer from '../components/layout/PageContainer';
import StatusIndicator from '../components/common/StatusIndicator';
import ErrorMessage from '../components/common/ErrorMessage';
import { API_BASE_URL } from '../services/api';
import { useTheme } from '../context/ThemeContext';
import { formatBytes, formatTimestamp } from '../utils/formatters';

export function Settings({ healthState }) {
  const { theme, setTheme } = useTheme();
  const { health, systemInfo, loading, error, lastCheckedAt, refresh } = healthState;

  const isOnline = Boolean(health?.api_status === 'online');
  const isModelReady = Boolean(health?.model_ready);

  return (
    <PageContainer
      title="Application Settings"
      subtitle="Inspect backend connectivity, safe runtime model capabilities, upload constraints, and appearance preferences."
      actions={
        <button type="button" className="btn btn-secondary btn-sm" onClick={refresh} disabled={loading}>
          <RefreshCw size={14} aria-hidden="true" />
          {loading ? 'Refreshing…' : 'Refresh Diagnostics'}
        </button>
      }
    >
      {error ? (
        <div style={{ marginBottom: 20 }}>
          <ErrorMessage
            title="Backend unreachable"
            message={error}
            recoveryHint={`Confirm the Phase 7 FastAPI server is listening at ${API_BASE_URL} and retry.`}
            onRetry={refresh}
          />
        </div>
      ) : null}

      <div className="equal-two-col">
        <section className="card" aria-label="Backend connection and upload configuration">
          <div className="card-header">
            <div>
              <h2 className="card-title">
                <Server size={17} aria-hidden="true" />
                Backend Connection
              </h2>
              <p className="card-subtitle">Live API health and public upload constraints.</p>
            </div>
          </div>

          <dl className="kv-list">
            <div className="kv-row">
              <dt className="kv-label">Connection Status</dt>
              <dd className="kv-value" style={{ margin: 0 }}>
                <StatusIndicator
                  label={loading ? 'Checking…' : isOnline ? 'Connected' : 'Offline'}
                  tone={loading ? 'neutral' : isOnline ? 'healthy' : 'danger'}
                />
              </dd>
            </div>
            <div className="kv-row">
              <dt className="kv-label">Configured API Base URL</dt>
              <dd className="kv-value mono" style={{ margin: 0, fontSize: '0.84rem' }}>
                {API_BASE_URL}
              </dd>
            </div>
            <div className="kv-row">
              <dt className="kv-label">Database Status</dt>
              <dd className="kv-value" style={{ margin: 0 }}>
                {health?.database_status || '—'}
              </dd>
            </div>
            <div className="kv-row">
              <dt className="kv-label">Supported Image Formats</dt>
              <dd className="kv-value" style={{ margin: 0 }}>
                {systemInfo?.supported_image_formats?.length
                  ? systemInfo.supported_image_formats.join(', ')
                  : '—'}
              </dd>
            </div>
            <div className="kv-row">
              <dt className="kv-label">Maximum Upload Size</dt>
              <dd className="kv-value tabular" style={{ margin: 0 }}>
                {typeof systemInfo?.max_upload_size_bytes === 'number'
                  ? formatBytes(systemInfo.max_upload_size_bytes)
                  : '—'}
              </dd>
            </div>
            <div className="kv-row">
              <dt className="kv-label">Last Checked</dt>
              <dd className="kv-value" style={{ margin: 0 }}>
                {lastCheckedAt ? formatTimestamp(lastCheckedAt) : '—'}
              </dd>
            </div>
          </dl>
        </section>

        <section className="card" aria-label="AI model and explainability runtime">
          <div className="card-header">
            <div>
              <h2 className="card-title">
                <Cpu size={17} aria-hidden="true" />
                AI Model Metadata
              </h2>
              <p className="card-subtitle">Safe technical metadata exposed by GET /api/v1/system/info.</p>
            </div>
          </div>

          <dl className="kv-list">
            <div className="kv-row">
              <dt className="kv-label">Model Availability</dt>
              <dd className="kv-value" style={{ margin: 0 }}>
                <StatusIndicator
                  label={
                    loading
                      ? 'Checking…'
                      : !isOnline
                        ? 'Unavailable'
                        : isModelReady
                          ? 'Loaded & Ready'
                          : 'Checkpoint Missing'
                  }
                  tone={loading ? 'neutral' : !isOnline ? 'danger' : isModelReady ? 'healthy' : 'warning'}
                />
              </dd>
            </div>
            <div className="kv-row">
              <dt className="kv-label">Model Name</dt>
              <dd className="kv-value" style={{ margin: 0 }}>
                {systemInfo?.model_name || '—'}
              </dd>
            </div>
            <div className="kv-row">
              <dt className="kv-label">Number of Classes</dt>
              <dd className="kv-value tabular" style={{ margin: 0 }}>
                {typeof systemInfo?.num_classes === 'number' ? systemInfo.num_classes : '—'}
              </dd>
            </div>
            <div className="kv-row">
              <dt className="kv-label">Inference Device</dt>
              <dd className="kv-value mono" style={{ margin: 0, fontSize: '0.84rem' }}>
                {systemInfo?.device_type || health?.inference_device || '—'}
              </dd>
            </div>
            <div className="kv-row">
              <dt className="kv-label">Explainability Method</dt>
              <dd className="kv-value" style={{ margin: 0 }}>
                {systemInfo?.explanation_method || '—'}
              </dd>
            </div>
            <div className="kv-row">
              <dt className="kv-label">API Version</dt>
              <dd className="kv-value mono" style={{ margin: 0, fontSize: '0.84rem' }}>
                {systemInfo?.app_version || health?.app_version || '—'}
              </dd>
            </div>
          </dl>

          {health?.model_status_detail ? (
            <div className="xai-caption" style={{ marginTop: 14 }}>
              <strong>Status detail:</strong> {health.model_status_detail}
            </div>
          ) : null}
        </section>
      </div>

      <section className="card" style={{ marginTop: 20 }} aria-label="Appearance preferences">
        <div className="card-header">
          <div>
            <h2 className="card-title">
              <Palette size={17} aria-hidden="true" />
              Theme Preference
            </h2>
            <p className="card-subtitle">
              Choose between the signature deep botanical dark interface and high-contrast light mode.
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }} role="radiogroup" aria-label="Theme mode">
          <button
            type="button"
            role="radio"
            aria-checked={theme === 'dark'}
            className={`btn ${theme === 'dark' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setTheme('dark')}
          >
            <Moon size={16} aria-hidden="true" />
            Dark Mode (Default)
          </button>

          <button
            type="button"
            role="radio"
            aria-checked={theme === 'light'}
            className={`btn ${theme === 'light' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setTheme('light')}
          >
            <Sun size={16} aria-hidden="true" />
            Light Mode
          </button>
        </div>
      </section>
    </PageContainer>
  );
}

export default Settings;
