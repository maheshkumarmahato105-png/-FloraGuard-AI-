import { Link } from 'react-router-dom';
import { Eye, Trash2 } from 'lucide-react';
import {
  formatClassLabel,
  formatConfidence,
  formatTimestamp,
  truncateId,
} from '../../utils/formatters';
import StatusIndicator from '../common/StatusIndicator';

function getRecordBadge(record) {
  const isUncertain = record.status === 'uncertain' || record.is_uncertain;
  if (isUncertain) return { label: 'Uncertain', tone: 'warning' };
  if (record.is_healthy) return { label: 'Healthy', tone: 'healthy' };
  return { label: 'Diseased', tone: 'danger' };
}

function getRecordLabel(record, isUncertain) {
  if (isUncertain) return 'Uncertain / Rejected Sample';
  const topDisplay = record.top_predictions?.[0]?.display_name;
  return topDisplay || formatClassLabel(record.predicted_class);
}

export function HistoryTable({ records = [], onRequestDelete }) {
  return (
    <div>
      {/* Desktop / Tablet Table */}
      <div className="history-desktop-table table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              <th>Prediction</th>
              <th>Status</th>
              <th>Confidence</th>
              <th>Model</th>
              <th>Timestamp</th>
              <th>Request ID</th>
              <th style={{ textAlign: 'right' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {records.map((rec) => {
              const isUncertain = rec.status === 'uncertain' || rec.is_uncertain;
              const badge = getRecordBadge(rec);
              const label = getRecordLabel(rec, isUncertain);

              return (
                <tr key={rec.request_id}>
                  <td style={{ fontWeight: 600 }}>{label}</td>
                  <td>
                    <StatusIndicator label={badge.label} tone={badge.tone} />
                  </td>
                  <td className="tabular">
                    {isUncertain ? '—' : formatConfidence(rec.confidence)}
                  </td>
                  <td style={{ color: 'var(--text-secondary)' }}>{rec.model_name || '—'}</td>
                  <td style={{ color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>
                    {formatTimestamp(rec.created_at)}
                  </td>
                  <td className="mono" style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    {truncateId(rec.request_id, 12)}
                  </td>
                  <td style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                    <div style={{ display: 'inline-flex', gap: 6 }}>
                      <Link
                        to={`/results/${encodeURIComponent(rec.request_id)}`}
                        className="btn btn-secondary btn-sm"
                        aria-label={`View prediction details for ${label}`}
                      >
                        <Eye size={14} aria-hidden="true" />
                        Details
                      </Link>
                      <button
                        type="button"
                        className="btn btn-danger btn-sm"
                        onClick={() => onRequestDelete?.(rec)}
                        aria-label={`Delete prediction record ${rec.request_id}`}
                      >
                        <Trash2 size={14} aria-hidden="true" />
                        Delete
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Mobile Card List */}
      <div className="history-mobile-list">
        {records.map((rec) => {
          const isUncertain = rec.status === 'uncertain' || rec.is_uncertain;
          const badge = getRecordBadge(rec);
          const label = getRecordLabel(rec, isUncertain);

          return (
            <article
              key={rec.request_id}
              style={{
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border-color)',
                borderRadius: 'var(--radius-md)',
                padding: 14,
                display: 'flex',
                flexDirection: 'column',
                gap: 10,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 10 }}>
                <div style={{ fontWeight: 600, fontSize: '0.94rem' }}>{label}</div>
                <StatusIndicator label={badge.label} tone={badge.tone} />
              </div>

              <div
                style={{
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: '6px 14px',
                  fontSize: '0.82rem',
                  color: 'var(--text-secondary)',
                }}
              >
                {!isUncertain ? (
                  <span className="tabular">Confidence: {formatConfidence(rec.confidence)}</span>
                ) : null}
                <span>{formatTimestamp(rec.created_at)}</span>
                <span className="mono">{truncateId(rec.request_id, 10)}</span>
              </div>

              <div style={{ display: 'flex', gap: 8, paddingTop: 4 }}>
                <Link
                  to={`/results/${encodeURIComponent(rec.request_id)}`}
                  className="btn btn-secondary btn-sm"
                  style={{ flex: 1 }}
                >
                  <Eye size={14} aria-hidden="true" />
                  View Details
                </Link>
                <button
                  type="button"
                  className="btn btn-danger btn-sm"
                  onClick={() => onRequestDelete?.(rec)}
                  aria-label={`Delete prediction record ${rec.request_id}`}
                >
                  <Trash2 size={14} aria-hidden="true" />
                  Delete
                </button>
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}

export default HistoryTable;
