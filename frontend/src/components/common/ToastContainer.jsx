import { AlertTriangle, CheckCircle2, Info, X, XCircle } from 'lucide-react';
import { useToast } from '../../context/ToastContext';

const ICONS = {
  success: CheckCircle2,
  error: XCircle,
  warning: AlertTriangle,
  info: Info,
};

export function ToastContainer() {
  const { toasts, dismissToast } = useToast();

  if (!toasts.length) return null;

  return (
    <div className="toast-region" role="region" aria-label="Notifications" aria-live="polite">
      {toasts.map((toast) => {
        const Icon = ICONS[toast.type] || Info;
        return (
          <div key={toast.id} className={`toast toast-${toast.type}`} role="status">
            <Icon
              size={18}
              style={{
                flexShrink: 0,
                marginTop: 2,
                color:
                  toast.type === 'success'
                    ? 'var(--status-healthy)'
                    : toast.type === 'error'
                      ? 'var(--status-danger)'
                      : toast.type === 'warning'
                        ? 'var(--status-warning)'
                        : 'var(--status-info)',
              }}
              aria-hidden="true"
            />
            <div style={{ flex: 1, minWidth: 0 }}>
              {toast.title ? (
                <div style={{ fontWeight: 600, marginBottom: toast.message ? 2 : 0 }}>
                  {toast.title}
                </div>
              ) : null}
              {toast.message ? (
                <div style={{ color: 'var(--text-secondary)', fontSize: '0.84rem' }}>
                  {toast.message}
                </div>
              ) : null}
            </div>
            <button
              type="button"
              className="btn btn-ghost btn-icon"
              style={{ width: 26, height: 26 }}
              onClick={() => dismissToast(toast.id)}
              aria-label="Dismiss notification"
            >
              <X size={14} />
            </button>
          </div>
        );
      })}
    </div>
  );
}

export default ToastContainer;
