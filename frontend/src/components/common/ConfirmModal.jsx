import { useEffect } from 'react';
import { X } from 'lucide-react';

export function ConfirmModal({
  open,
  title,
  description,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  confirmVariant = 'danger',
  busy = false,
  onConfirm,
  onCancel,
}) {
  useEffect(() => {
    if (!open) return undefined;
    const onKeyDown = (e) => {
      if (e.key === 'Escape' && !busy) {
        onCancel?.();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [open, busy, onCancel]);

  if (!open) return null;

  return (
    <div
      className="modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-labelledby="confirm-modal-title"
      onClick={(e) => {
        if (e.target === e.currentTarget && !busy) {
          onCancel?.();
        }
      }}
    >
      <div className="modal-panel">
        <div className="modal-header">
          <h2 id="confirm-modal-title" className="modal-title">
            {title}
          </h2>
          <button
            type="button"
            className="btn btn-ghost btn-icon"
            onClick={onCancel}
            disabled={busy}
            aria-label="Close dialog"
          >
            <X size={18} />
          </button>
        </div>

        <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.92rem' }}>
          {description}
        </p>

        <div className="modal-footer">
          <button
            type="button"
            className="btn btn-secondary"
            onClick={onCancel}
            disabled={busy}
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            className={`btn btn-${confirmVariant}`}
            onClick={onConfirm}
            disabled={busy}
          >
            {busy ? 'Working…' : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

export default ConfirmModal;
