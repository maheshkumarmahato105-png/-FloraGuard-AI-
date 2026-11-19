import { AlertCircle, RefreshCw } from 'lucide-react';

export function ErrorMessage({
  title = 'Something went wrong',
  message,
  recoveryHint,
  onRetry,
  retryLabel = 'Try again',
  variant = 'danger',
}) {
  if (!message) return null;

  return (
    <div
      className={`alert alert-${variant}`}
      role="alert"
      aria-live="assertive"
    >
      <AlertCircle size={20} className="alert-icon" aria-hidden="true" />
      <div className="alert-body">
        <p className="alert-title">{title}</p>
        <p className="alert-text">{message}</p>
        {recoveryHint ? (
          <p className="alert-text" style={{ marginTop: '6px', opacity: 0.9 }}>
            {recoveryHint}
          </p>
        ) : null}
        {typeof onRetry === 'function' ? (
          <div style={{ marginTop: '12px' }}>
            <button type="button" className="btn btn-secondary btn-sm" onClick={onRetry}>
              <RefreshCw size={14} aria-hidden="true" />
              {retryLabel}
            </button>
          </div>
        ) : null}
      </div>
    </div>
  );
}

export default ErrorMessage;
