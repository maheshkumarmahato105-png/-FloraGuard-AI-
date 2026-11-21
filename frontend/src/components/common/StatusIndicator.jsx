/**
 * Compact semantic status badge with Apple-style dot indicator.
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
            : 'badge-neutral';

  const dotTone =
    tone === 'healthy' || tone === 'warning' || tone === 'danger'
      ? tone
      : 'neutral';

  return (
    <span className={`badge ${toneClass}`} title={title || label}>
      <span className={`status-dot ${dotTone}`} aria-hidden="true" />
      <span>{label}</span>
    </span>
  );
}

export default StatusIndicator;
