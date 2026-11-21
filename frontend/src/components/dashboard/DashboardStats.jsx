import { Activity, AlertTriangle, Bug, CheckCircle2 } from 'lucide-react';

export function DashboardStats({ summary, loading }) {
  const cards = [
    {
      key: 'total',
      label: 'Total Scans',
      value: summary?.totalScans ?? 0,
      caption: 'Persisted in backend SQLite history',
      icon: Activity,
      toneColor: 'var(--accent-primary)',
      badgeBg: 'var(--bg-hover)',
    },
    {
      key: 'healthy',
      label: 'Healthy Plants',
      value: summary?.healthyCount ?? 0,
      caption: summary?.statsSampled
        ? 'Among latest 100 accepted scans'
        : 'Accepted healthy leaf predictions',
      icon: CheckCircle2,
      toneColor: 'var(--status-healthy)',
      badgeBg: 'var(--status-healthy-bg)',
    },
    {
      key: 'diseased',
      label: 'Disease Detections',
      value: summary?.diseasedCount ?? 0,
      caption: summary?.statsSampled
        ? 'Among latest 100 accepted scans'
        : 'Accepted diseased leaf predictions',
      icon: Bug,
      toneColor: 'var(--status-danger)',
      badgeBg: 'var(--status-danger-bg)',
    },
    {
      key: 'uncertain',
      label: 'Uncertain Predictions',
      value: summary?.uncertainCount ?? 0,
      caption: 'Rejected by uncertainty safety policy',
      icon: AlertTriangle,
      toneColor: 'var(--status-warning)',
      badgeBg: 'var(--status-warning-bg)',
    },
  ];

  return (
    <section className="stats-grid" aria-label="Prediction statistics">
      {cards.map(({ key, label, value, caption, icon: Icon, toneColor, badgeBg }) => (
        <article key={key} className="card stat-card card-interactive">
          <div className="stat-top">
            <span className="stat-label">{label}</span>
            <span
              className="stat-icon"
              style={{ color: toneColor, background: badgeBg }}
              aria-hidden="true"
            >
              <Icon size={16} />
            </span>
          </div>
          {loading ? (
            <div className="skeleton" style={{ height: 36, width: '48%', margin: '4px 0 8px' }} aria-hidden="true" />
          ) : (
            <div className="stat-value tabular">{value}</div>
          )}
          <div className="stat-caption">{caption}</div>
        </article>
      ))}
    </section>
  );
}

export default DashboardStats;
