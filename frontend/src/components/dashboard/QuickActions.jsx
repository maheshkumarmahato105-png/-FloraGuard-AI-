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
            Quick Actions &amp; Readiness
          </h2>
          <p className="card-subtitle">Live system telemetry reported by the backend.</p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 10, marginBottom: 20 }}>
        <Link to="/detection" className="btn btn-primary" style={{ justifyContent: 'space-between', padding: '12px 18px' }}>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 10, fontWeight: 600 }}>
            <Upload size={17} aria-hidden="true" />
            Upload Leaf Image
          </span>
          <span style={{ fontSize: '0.78rem', opacity: 0.9, background: 'rgba(0,0,0,0.15)', padding: '2px 8px', borderRadius: 'var(--radius-sm)' }}>
            Instant Scan
          </span>
        </Link>

        <Link
          to="/detection?mode=camera"
          className="btn btn-secondary"
          style={{ justifyContent: 'space-between', padding: '12px 18px' }}
        >
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 10 }}>
            <Camera size={17} aria-hidden="true" />
            Open Camera Capture
          </span>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>Live View</span>
        </Link>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          <Link to="/history" className="btn btn-secondary btn-sm" style={{ padding: '9px 12px' }}>
            <HistoryIcon size={15} aria-hidden="true" />
            History
          </Link>
          <Link to="/settings" className="btn btn-secondary btn-sm" style={{ padding: '9px 12px' }}>
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
          <dt className="kv-label">Inference Device</dt>
          <dd className="kv-value mono" style={{ margin: 0 }}>
            {loading ? '…' : (health?.inference_device || 'CPU').toUpperCase()}
          </dd>
        </div>
        <div className="kv-row">
          <dt className="kv-label">Active Architecture</dt>
          <dd className="kv-value mono" style={{ margin: 0 }}>
            {systemInfo?.model_name || 'agrivision_cnn'}
          </dd>
        </div>
      </dl>
    </section>
  );
}

export default QuickActions;
