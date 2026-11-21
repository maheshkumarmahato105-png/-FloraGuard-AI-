import { Link } from 'react-router-dom';
import { ArrowRight, ChevronRight, Clock, Leaf, ScanSearch } from 'lucide-react';
import { formatClassLabel, formatConfidence, formatTimestamp } from '../../utils/formatters';
import EmptyState from '../common/EmptyState';
import StatusIndicator from '../common/StatusIndicator';

export function RecentScans({ scans = [], loading, error }) {
  return (
    <section className="card" aria-label="Recent plant leaf scans">
      <div className="card-header">
        <div>
          <h2 className="card-title">
            <Clock size={17} aria-hidden="true" />
            Recent Scans
          </h2>
          <p className="card-subtitle">Latest records returned by the backend history database.</p>
        </div>
        <Link to="/history" className="btn btn-ghost btn-sm">
          <span>View all</span>
          <ArrowRight size={14} aria-hidden="true" />
        </Link>
      </div>

      {loading ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {[1, 2, 3].map((idx) => (
            <div key={idx} className="skeleton" style={{ height: 60, width: '100%', borderRadius: 'var(--radius-md)' }} />
          ))}
        </div>
      ) : error ? (
        <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
          Unable to load recent scans while the backend is unreachable.
        </p>
      ) : scans.length === 0 ? (
        <EmptyState
          icon={ScanSearch}
          title="No plant scans recorded yet"
          description="Submit a leaf image from the Detection workspace to analyze plant health and build your real scan history."
          action={
            <Link to="/detection" className="btn btn-primary btn-sm">
              Start First Scan
            </Link>
          }
        />
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {scans.map((scan) => {
            const isUncertain = scan.status === 'uncertain' || scan.is_uncertain;
            const tone = isUncertain
              ? 'warning'
              : scan.is_healthy
                ? 'healthy'
                : 'danger';
            const statusLabel = isUncertain
              ? 'Uncertain'
              : scan.is_healthy
                ? 'Healthy'
                : 'Diseased';

            const displayName = isUncertain
              ? 'Uncertain / Rejected Sample'
              : scan.top_predictions?.[0]?.display_name || formatClassLabel(scan.predicted_class);

            return (
              <Link
                key={scan.request_id}
                to={`/results/${encodeURIComponent(scan.request_id)}`}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: 14,
                  padding: '12px 16px',
                  borderRadius: 'var(--radius-md)',
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border-subtle)',
                  transition: 'background-color var(--transition-fast), border-color var(--transition-fast), transform var(--transition-fast)',
                }}
                className="card-interactive"
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, minWidth: 0, flex: 1 }}>
                  <div
                    style={{
                      width: 36,
                      height: 36,
                      borderRadius: 'var(--radius-sm)',
                      background: isUncertain
                        ? 'var(--status-warning-bg)'
                        : scan.is_healthy
                          ? 'var(--status-healthy-bg)'
                          : 'var(--status-danger-bg)',
                      color: isUncertain
                        ? 'var(--status-warning)'
                        : scan.is_healthy
                          ? 'var(--status-healthy)'
                          : 'var(--status-danger)',
                      display: 'inline-flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexShrink: 0,
                    }}
                    aria-hidden="true"
                  >
                    <Leaf size={18} />
                  </div>
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <div
                      style={{
                        fontWeight: 600,
                        fontSize: '0.91rem',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        color: 'var(--text-main)',
                      }}
                    >
                      {displayName}
                    </div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: 2 }}>
                      {formatTimestamp(scan.created_at)}
                    </div>
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexShrink: 0 }}>
                  {!isUncertain && typeof scan.confidence === 'number' ? (
                    <span
                      className="tabular"
                      style={{
                        fontSize: '0.84rem',
                        fontWeight: 600,
                        padding: '3px 8px',
                        borderRadius: 'var(--radius-sm)',
                        background: 'var(--bg-hover)',
                        color: 'var(--text-main)',
                      }}
                    >
                      {formatConfidence(scan.confidence)}
                    </span>
                  ) : null}
                  <StatusIndicator label={statusLabel} tone={tone} />
                  <ChevronRight size={16} style={{ color: 'var(--text-muted)' }} aria-hidden="true" />
                </div>
              </Link>
            );
          })}
        </div>
      )}
    </section>
  );
}

export default RecentScans;
