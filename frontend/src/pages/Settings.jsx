import { useState } from 'react';
import { Cpu, Moon, Palette, RefreshCw, Server, Sun } from 'lucide-react';
import PageContainer from '../components/layout/PageContainer';
import StatusIndicator from '../components/common/StatusIndicator';
import ErrorMessage from '../components/common/ErrorMessage';
import { getApiBaseUrl, setCustomApiBaseUrl } from '../services/api';
import { useTheme } from '../context/ThemeContext';
import { formatBytes, formatTimestamp } from '../utils/formatters';

export function Settings({ healthState }) {
  const { theme, setTheme } = useTheme();
  const { health, systemInfo, loading, error, lastCheckedAt, refresh } = healthState;

  const [currentUrl, setCurrentUrl] = useState(getApiBaseUrl());
  const [isEditingUrl, setIsEditingUrl] = useState(false);
  const [urlInput, setUrlInput] = useState(getApiBaseUrl());

  const handleSaveUrl = (urlToSet) => {
    const trimmed = (urlToSet || '').trim();
    setCustomApiBaseUrl(trimmed);
    const resolved = getApiBaseUrl();
    setCurrentUrl(resolved);
    setUrlInput(resolved);
    setIsEditingUrl(false);
    refresh();
  };

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
            recoveryHint={`Confirm the Phase 7 FastAPI server is listening at ${currentUrl} and retry.`}
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
            <div className="kv-row" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '8px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
                <dt className="kv-label">Configured API Base URL</dt>
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  style={{ fontSize: '0.75rem', padding: '2px 8px' }}
                  onClick={() => setIsEditingUrl(!isEditingUrl)}
                >
                  {isEditingUrl ? 'Cancel' : 'Change URL'}
                </button>
              </div>
              {isEditingUrl ? (
                <div style={{ display: 'flex', gap: '8px', width: '100%', marginTop: '4px' }}>
                  <input
                    type="text"
                    value={urlInput}
                    onChange={(e) => setUrlInput(e.target.value)}
                    placeholder="https://floraguard-ai.onrender.com"
                    style={{ flex: 1, padding: '4px 8px', fontSize: '0.85rem', borderRadius: '4px', border: '1px solid var(--border-color, #ccc)' }}
                  />
                  <button
                    type="button"
                    className="btn btn-primary btn-sm"
                    onClick={() => handleSaveUrl(urlInput)}
                  >
                    Save & Connect
                  </button>
                </div>
              ) : (
                <dd className="kv-value mono" style={{ margin: 0, fontSize: '0.84rem', wordBreak: 'break-all' }}>
                  {currentUrl}
                </dd>
              )}
              {currentUrl !== 'https://floraguard-ai.onrender.com' && (
                <button
                  type="button"
                  style={{ fontSize: '0.75rem', background: 'none', border: 'none', color: '#10b981', cursor: 'pointer', padding: 0, textDecoration: 'underline' }}
                  onClick={() => handleSaveUrl('https://floraguard-ai.onrender.com')}
                >
                  Use Live Render URL (https://floraguard-ai.onrender.com)
                </button>
              )}
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

        <div className="theme-picker-grid" role="radiogroup" aria-label="Theme mode">
          <button
            type="button"
            role="radio"
            aria-checked={theme === 'dark'}
            className={`theme-card-option${theme === 'dark' ? ' active' : ''}`}
            onClick={() => setTheme('dark')}
          >
            <div className="theme-preview-box theme-preview-dark" aria-hidden="true">
              <div style={{ display: 'flex', gap: 5, marginBottom: 4 }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#ff5f56' }} />
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#ffbd2e' }} />
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#27c93f' }} />
              </div>
              <div style={{ height: 8, width: '55%', background: 'rgba(255,255,255,0.15)', borderRadius: 4 }} />
              <div style={{ display: 'flex', gap: 6, marginTop: 2 }}>
                <div style={{ flex: 1, height: 28, background: 'rgba(52,199,89,0.22)', border: '1px solid rgba(52,199,89,0.35)', borderRadius: 6 }} />
                <div style={{ flex: 1, height: 28, background: 'rgba(255,255,255,0.06)', borderRadius: 6 }} />
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 7, fontWeight: 600, fontSize: '0.90rem', color: 'var(--text-main)' }}>
                <Moon size={16} aria-hidden="true" />
                Dark Mode (Default)
              </span>
              <span className={`radio-dot${theme === 'dark' ? ' active' : ''}`} aria-hidden="true" />
            </div>
          </button>

          <button
            type="button"
            role="radio"
            aria-checked={theme === 'light'}
            className={`theme-card-option${theme === 'light' ? ' active' : ''}`}
            onClick={() => setTheme('light')}
          >
            <div className="theme-preview-box theme-preview-light" aria-hidden="true">
              <div style={{ display: 'flex', gap: 5, marginBottom: 4 }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#ff5f56' }} />
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#ffbd2e' }} />
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#27c93f' }} />
              </div>
              <div style={{ height: 8, width: '55%', background: 'rgba(0,0,0,0.12)', borderRadius: 4 }} />
              <div style={{ display: 'flex', gap: 6, marginTop: 2 }}>
                <div style={{ flex: 1, height: 28, background: 'rgba(21,128,61,0.15)', border: '1px solid rgba(21,128,61,0.28)', borderRadius: 6 }} />
                <div style={{ flex: 1, height: 28, background: 'rgba(0,0,0,0.04)', borderRadius: 6 }} />
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 7, fontWeight: 600, fontSize: '0.90rem', color: 'var(--text-main)' }}>
                <Sun size={16} aria-hidden="true" />
                Light Mode
              </span>
              <span className={`radio-dot${theme === 'light' ? ' active' : ''}`} aria-hidden="true" />
            </div>
          </button>
        </div>
      </section>
    </PageContainer>
  );
}

export default Settings;
