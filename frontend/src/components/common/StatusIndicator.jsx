/**
 * Compact semantic status badge with dot indicator.
 * tone: 'healthy' | 'warning' | 'danger' | 'info' | 'neutral'
 */
export function StatusIndicator({ label, tone = 'neutral', title }) {
  const toneClass =
    tone === 'healthy'
      ? 'badge-healthy'
      : tone === 'warning'
        ? 'badge-warning'
        : tone === 'danger'
          ? 'badge-danger'
          : tone === 'info'
            ? 'badge-info'
            : '';

  return (
    <span className={`badge ${toneClass}`} title={title || label}>
      <span className="status-dot" aria-hidden="true" />
      <span>{label}</span>
    </span>
  );
}

export default StatusIndicator;
