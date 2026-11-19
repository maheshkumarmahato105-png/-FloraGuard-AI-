import { AlertTriangle, CheckCircle2, Cpu, ShieldAlert } from 'lucide-react';
import { formatClassLabel, formatConfidence, formatTimestamp } from '../../utils/formatters';
import StatusIndicator from '../common/StatusIndicator';

export function PredictionCard({ result }) {
  if (!result) return null;

  const isUncertain = result.status === 'uncertain' || result.uncertainty?.is_uncertain;

  if (isUncertain) {
    const reasons = result.uncertainty?.rejection_reasons?.length
      ? result.uncertainty.rejection_reasons
      : result.rejection_reason
        ? [result.rejection_reason]
        : [];

    return (
      <section className="card" aria-label="Uncertain prediction notice">
        <div className="alert alert-warning" role="alert">
          <AlertTriangle size={22} className="alert-icon" aria-hidden="true" />
          <div className="alert-body">
            <h2 className="alert-title" style={{ fontSize: '1.08rem' }}>
              Uncertain Prediction — Manual Review Recommended
            </h2>
            <p className="alert-text" style={{ marginTop: 6 }}>
              {result.message ||
                'The AI model could not make a reliable diagnosis for this image. No unverified disease diagnosis or treatment guidance is displayed.'}
            </p>
            <p className="alert-text" style={{ marginTop: 8 }}>
              <strong>Recommended next steps:</strong> Capture a sharper, well-lit close-up photo of
              a single leaf against a clean background, or consult a qualified agricultural
              extension specialist.
            </p>
            {reasons.length > 0 ? (
              <div style={{ marginTop: 12 }}>
                <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-main)' }}>
                  Uncertainty checks triggered:
                </div>
                <ul className="guidance-list" style={{ marginTop: 4 }}>
                  {reasons.map((reason, idx) => (
                    <li key={idx}>{reason}</li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        </div>

        <dl className="kv-list" style={{ marginTop: 18 }}>
          <div className="kv-row">
            <dt className="kv-label">Status</dt>
            <dd className="kv-value" style={{ margin: 0 }}>
              <StatusIndicator label="Uncertain (Rejected)" tone="warning" />
            </dd>
          </div>
          <div className="kv-row">
            <dt className="kv-label">Timestamp</dt>
            <dd className="kv-value" style={{ margin: 0 }}>
              {formatTimestamp(result.created_at)}
            </dd>
          </div>
          <div className="kv-row">
            <dt className="kv-label">Model</dt>
            <dd className="kv-value" style={{ margin: 0 }}>
              {result.model_name || '—'}
            </dd>
          </div>
          {result.inference_device ? (
            <div className="kv-row">
              <dt className="kv-label">Inference Device</dt>
              <dd className="kv-value mono" style={{ margin: 0, fontSize: '0.84rem' }}>
                {result.inference_device}
              </dd>
            </div>
          ) : null}
        </dl>
      </section>
    );
  }

  const pred = result.prediction || {};
  const isHealthy = Boolean(pred.is_healthy);
  const displayTitle =
    pred.display_name || formatClassLabel(pred.class_name || result.predicted_class);
  const plantName = pred.plant_name || null;
  const diseaseName = pred.disease_name || null;

  return (
    <section className="card" aria-label="Prediction summary">
      <div className="card-header" style={{ alignItems: 'flex-start' }}>
        <div>
          <div style={{ marginBottom: 8 }}>
            <StatusIndicator
              label={isHealthy ? 'Healthy Plant' : 'Disease Detected'}
              tone={isHealthy ? 'healthy' : 'danger'}
            />
          </div>
          <h2 className="page-title" style={{ fontSize: '1.4rem' }}>
            {displayTitle}
          </h2>
          {(plantName || diseaseName) ? (
            <p className="card-subtitle">
              {plantName ? `Crop: ${plantName}` : ''}
              {plantName && diseaseName ? ' · ' : ''}
              {diseaseName ? `Condition: ${diseaseName}` : ''}
            </p>
          ) : null}
        </div>

        <span
          className="stat-icon"
          style={{
            width: 44,
            height: 44,
            color: isHealthy ? 'var(--status-healthy)' : 'var(--status-danger)',
          }}
          aria-hidden="true"
        >
          {isHealthy ? <CheckCircle2 size={22} /> : <ShieldAlert size={22} />}
        </span>
      </div>

      <dl className="kv-list" style={{ marginTop: 14 }}>
        <div className="kv-row">
          <dt className="kv-label">Prediction Confidence</dt>
          <dd className="kv-value tabular" style={{ margin: 0, color: 'var(--accent-primary)' }}>
            {formatConfidence(pred.confidence)}
          </dd>
        </div>
        <div className="kv-row">
          <dt className="kv-label">Analyzed At</dt>
          <dd className="kv-value" style={{ margin: 0 }}>
            {formatTimestamp(result.created_at)}
          </dd>
        </div>
        <div className="kv-row">
          <dt className="kv-label">Model Architecture</dt>
          <dd className="kv-value" style={{ margin: 0, display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <Cpu size={14} aria-hidden="true" />
            {result.model_name || '—'}
          </dd>
        </div>
        {result.checkpoint_id ? (
          <div className="kv-row">
            <dt className="kv-label">Checkpoint</dt>
            <dd className="kv-value mono" style={{ margin: 0, fontSize: '0.82rem' }}>
              {result.checkpoint_id}
            </dd>
          </div>
        ) : null}
        {result.inference_device ? (
          <div className="kv-row">
            <dt className="kv-label">Inference Device</dt>
            <dd className="kv-value mono" style={{ margin: 0, fontSize: '0.82rem' }}>
              {result.inference_device}
            </dd>
          </div>
        ) : null}
      </dl>
    </section>
  );
}

export default PredictionCard;
