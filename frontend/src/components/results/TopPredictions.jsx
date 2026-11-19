import { ListOrdered } from 'lucide-react';
import { confidenceToPercent, formatClassLabel, formatConfidence } from '../../utils/formatters';
import StatusIndicator from '../common/StatusIndicator';

export function TopPredictions({ predictions = [], isUncertain = false }) {
  if (isUncertain) {
    return null;
  }

  const topThree = predictions.slice(0, 3);
  if (topThree.length === 0) {
    return null;
  }

  return (
    <section className="card" aria-label="Top model predictions">
      <div className="card-header">
        <div>
          <h3 className="card-title">
            <ListOrdered size={17} aria-hidden="true" />
            Top {topThree.length} Predictions
          </h3>
          <p className="card-subtitle">Highest-ranked classes returned by the backend model.</p>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        {topThree.map((item, idx) => {
          const pct = confidenceToPercent(item.confidence);
          const label = item.display_name || formatClassLabel(item.class_name);
          return (
            <div key={item.class_name || idx}>
              <div
                style={{
                  display: 'flex',
                  flexWrap: 'wrap',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: 8,
                  marginBottom: 6,
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
                  <span
                    className="mono"
                    style={{
                      fontSize: '0.75rem',
                      color: 'var(--text-muted)',
                      width: 20,
                    }}
                  >
                    #{idx + 1}
                  </span>
                  <span style={{ fontWeight: idx === 0 ? 600 : 500, fontSize: '0.9rem' }}>
                    {label}
                  </span>
                  {typeof item.is_healthy === 'boolean' ? (
                    <StatusIndicator
                      label={item.is_healthy ? 'Healthy' : 'Disease'}
                      tone={item.is_healthy ? 'healthy' : 'danger'}
                    />
                  ) : null}
                </div>
                <span className="tabular" style={{ fontWeight: 600, fontSize: '0.88rem' }}>
                  {formatConfidence(item.confidence)}
                </span>
              </div>
              <div className="progress-track" style={{ height: 7 }}>
                <div
                  className="progress-fill"
                  style={{
                    width: `${pct}%`,
                    opacity: idx === 0 ? 1 : 0.65,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

export default TopPredictions;
