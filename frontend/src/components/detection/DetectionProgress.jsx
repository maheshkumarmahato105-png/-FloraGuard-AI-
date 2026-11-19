import { X } from 'lucide-react';

export function DetectionProgress({ stage, uploadProgress = 0, onCancel }) {
  if (stage !== 'uploading' && stage !== 'analyzing') {
    return null;
  }

  const isUploading = stage === 'uploading';
  const pct = isUploading ? Math.min(99, Math.max(4, uploadProgress)) : 100;

  return (
    <div
      className="card"
      role="status"
      aria-live="polite"
      style={{
        marginTop: 16,
        background: 'var(--bg-secondary)',
        borderColor: 'var(--accent-border)',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 12,
          marginBottom: 10,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span className="spinner" style={{ width: 18, height: 18 }} aria-hidden="true" />
          <div>
            <div style={{ fontWeight: 600, fontSize: '0.92rem' }}>
              {isUploading
                ? `Uploading leaf image (${uploadProgress}%)…`
                : 'Running AgriVision-CNN inference on backend…'}
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              {isUploading
                ? 'Transferring image payload via multipart FormData.'
                : 'Waiting for real model prediction and uncertainty checks from the FastAPI server.'}
            </div>
          </div>
        </div>

        {typeof onCancel === 'function' ? (
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={onCancel}
            aria-label="Cancel ongoing analysis"
          >
            <X size={15} aria-hidden="true" />
            Cancel
          </button>
        ) : null}
      </div>

      <div
        className="progress-track"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct}
        aria-label={isUploading ? 'Image upload progress' : 'Model inference in progress'}
      >
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default DetectionProgress;
