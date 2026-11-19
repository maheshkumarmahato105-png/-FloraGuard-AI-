import { Link } from 'react-router-dom';
import { Camera, Cpu, History as HistoryIcon, Settings as SettingsIcon, Upload } from 'lucide-react';
import StatusIndicator from '../common/StatusIndicator';

export function QuickActions({ health, systemInfo, loading }) {
  const isOnline = Boolean(health?.api_status === 'online');
  const isModelReady = Boolean(health?.model_ready);

  return (
    <section className="card" aria-label="Quick actions and system readiness">
      <div className="card-header">
        <div>
          <h2 className="card-title">
            <Cpu size={17} aria-hidden="true" />
            Quick Actions &amp; System Status
          </h2>
          <p className="card-subtitle">Live diagnostics reported by the FastAPI backend.</p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 10, marginBottom: 18 }}>
        <Link to="/detection" className="btn btn-primary" style={{ justifyContent: 'space-between' }}>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
            <Upload size={16} aria-hidden="true" />
            Upload Leaf Image
          </span>
          <span style={{ fontSize: '0.78rem', opacity: 0.85 }}>Scan</span>
        </Link>

        <Link
          to="/detection?mode=camera"
          className="btn btn-secondary"
          style={{ justifyContent: 'space-between' }}
        >
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
            <Camera size={16} aria-hidden="true" />
            Open Live Camera Capture
          </span>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>Opt-in</span>
        </Link>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          <Link to="/history" className="btn btn-secondary btn-sm">
            <HistoryIcon size={15} aria-hidden="true" />
            History
          </Link>
          <Link to="/settings" className="btn btn-secondary btn-sm">
            <SettingsIcon size={15} aria-hidden="true" />
            Settings
          </Link>
        </div>
      </div>

      <dl className="kv-list">
        <div className="kv-row">
          <dt className="kv-label">Backend API</dt>
          <dd className="kv-value" style={{ margin: 0 }}>
            <StatusIndicator
              label={loading ? 'Checking…' : isOnline ? 'Connected' : 'Offline'}
              tone={loading ? 'neutral' : isOnline ? 'healthy' : 'danger'}
            />
          </dd>
        </div>
        <div className="kv-row">
          <dt className="kv-label">Model Availability</dt>
          <dd className="kv-value" style={{ margin: 0 }}>
            <StatusIndicator
              label={
                loading
                  ? 'Checking…'
                  : !isOnline
                    ? 'Unknown'
                    : isModelReady
                      ? 'Ready'
                      : 'Checkpoint Missing'
              }
              tone={loading ? 'neutral' : !isOnline ? 'danger' : isModelReady ? 'healthy' : 'warning'}
            />
          </dd>
        </div>
        <div className="kv-row">
          <dt className="kv-label">Architecture</dt>
          <dd className="kv-value" style={{ margin: 0 }}>
            {systemInfo?.model_name || '—'}
          </dd>
        </div>
        <div className="kv-row">
          <dt className="kv-label">Inference Device</dt>
          <dd className="kv-value mono" style={{ margin: 0, fontSize: '0.84rem' }}>
            {health?.inference_device || systemInfo?.device_type || '—'}
          </dd>
        </div>
      </dl>
    </section>
  );
}

export default QuickActions;
