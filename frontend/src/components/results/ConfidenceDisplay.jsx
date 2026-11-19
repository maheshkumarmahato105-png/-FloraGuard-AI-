import { Gauge } from 'lucide-react';
import { confidenceToPercent, formatConfidence, formatMetric } from '../../utils/formatters';

export function ConfidenceDisplay({ confidence, uncertainty, isUncertain = false }) {
  const effectiveScore =
    typeof confidence === 'number'
      ? confidence
      : typeof uncertainty?.max_probability === 'number'
        ? uncertainty.max_probability
        : null;

  const pct = confidenceToPercent(effectiveScore);
  const barTone = isUncertain ? 'warning' : pct >= 80 ? '' : 'warning';

  return (
    <section className="card" aria-label="Confidence and uncertainty metrics">
      <div className="card-header">
        <div>
          <h3 className="card-title">
            <Gauge size={17} aria-hidden="true" />
            Confidence &amp; Uncertainty Telemetry
          </h3>
          <p className="card-subtitle">
            {isUncertain
              ? 'Peak softmax probability did not pass reliability thresholds.'
              : 'Model confidence calibrated alongside Shannon entropy and top-2 margin.'}
          </p>
        </div>
        <span className="tabular" style={{ fontSize: '1.35rem', fontWeight: 700 }}>
          {formatConfidence(effectiveScore)}
        </span>
      </div>

      <div
        className="progress-track"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(pct)}
        aria-label="Prediction confidence percentage"
        style={{ height: 10, marginBottom: 16 }}
      >
        <div className={`progress-fill ${barTone}`} style={{ width: `${pct}%` }} />
      </div>

      {uncertainty ? (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
            gap: 10,
          }}
        >
          <div
            style={{
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-sm)',
              padding: '10px 12px',
            }}
          >
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Max Prob.</div>
            <div className="tabular" style={{ fontWeight: 700, fontSize: '0.96rem', marginTop: 2 }}>
              {formatConfidence(uncertainty.max_probability ?? effectiveScore)}
            </div>
          </div>
          <div
            style={{
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-sm)',
              padding: '10px 12px',
            }}
          >
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Entropy</div>
            <div className="tabular" style={{ fontWeight: 700, fontSize: '0.96rem', marginTop: 2 }}>
              {formatMetric(uncertainty.entropy, 3)}
            </div>
          </div>
          <div
            style={{
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-sm)',
              padding: '10px 12px',
            }}
          >
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Top-2 Margin</div>
            <div className="tabular" style={{ fontWeight: 700, fontSize: '0.96rem', marginTop: 2 }}>
              {formatMetric(uncertainty.margin, 3)}
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}

export default ConfidenceDisplay;
