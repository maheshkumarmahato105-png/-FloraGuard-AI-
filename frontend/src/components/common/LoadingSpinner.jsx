export function LoadingSpinner({ label = 'Loading…', size = 'md', inline = false }) {
  const dim = size === 'sm' ? 16 : size === 'lg' ? 28 : 20;

  if (inline) {
    return (
      <span
        role="status"
        aria-live="polite"
        style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}
      >
        <span
          className="spinner"
          style={{ width: dim, height: dim }}
          aria-hidden="true"
        />
        {label ? <span>{label}</span> : null}
      </span>
    );
  }

  return (
    <div
      role="status"
      aria-live="polite"
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '12px',
        padding: '32px 16px',
        color: 'var(--text-secondary)',
      }}
    >
      <span
        className="spinner"
        style={{ width: dim, height: dim }}
        aria-hidden="true"
      />
      {label ? <span style={{ fontSize: '0.9rem' }}>{label}</span> : null}
    </div>
  );
}

export default LoadingSpinner;
