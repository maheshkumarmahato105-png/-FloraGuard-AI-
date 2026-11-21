import React from 'react';

/**
 * Production ErrorBoundary preventing white-screen crashes from unexpected UI exceptions.
 * Safeguards application state and provides clean, humanized recovery pathways.
 */
export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    // Log exception for telemetry while keeping UI gracefully interactive
    console.error('FloraGuard AI Caught UI Exception:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
    window.location.href = '/';
  };

  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            minHeight: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '24px',
            background: 'var(--bg-main, #0b110e)',
            color: 'var(--text-main, #e8f5e9)',
            fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
          }}
        >
          <div
            style={{
              maxWidth: '520px',
              width: '100%',
              background: 'var(--bg-card, #131d17)',
              border: '1px solid var(--border-subtle, #1e3325)',
              borderRadius: '16px',
              padding: '36px 32px',
              textAlign: 'center',
              boxShadow: '0 20px 40px rgba(0,0,0,0.5)',
            }}
          >
            <div
              style={{
                width: '60px',
                height: '60px',
                borderRadius: '50%',
                background: 'rgba(239, 68, 68, 0.12)',
                color: '#ef4444',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 20px auto',
                fontSize: '28px',
                border: '1px solid rgba(239, 68, 68, 0.25)',
              }}
            >
              🛡️
            </div>
            <h2 style={{ fontSize: '1.35rem', fontWeight: 700, marginBottom: '12px', letterSpacing: '-0.02em' }}>
              FloraGuard Resilience Safeguard
            </h2>
            <p
              style={{
                fontSize: '0.92rem',
                color: 'var(--text-muted, #94a3b8)',
                lineHeight: 1.6,
                marginBottom: '26px',
              }}
            >
              An unexpected component render exception was safely caught. Your analysis models, local history, and data remain secure.
            </p>
            <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>
              <button
                type="button"
                onClick={() => window.location.reload()}
                style={{
                  padding: '11px 22px',
                  borderRadius: '10px',
                  border: '1px solid var(--border-subtle, #2d4a36)',
                  background: 'transparent',
                  color: 'var(--text-main, #e8f5e9)',
                  cursor: 'pointer',
                  fontWeight: 600,
                  fontSize: '0.9rem',
                  transition: 'all 0.2s ease',
                }}
              >
                Reload View
              </button>
              <button
                type="button"
                onClick={this.handleReset}
                style={{
                  padding: '11px 22px',
                  borderRadius: '10px',
                  border: 'none',
                  background: 'var(--brand-primary, #10b981)',
                  color: '#04160c',
                  cursor: 'pointer',
                  fontWeight: 700,
                  fontSize: '0.9rem',
                  boxShadow: '0 4px 14px rgba(16, 185, 129, 0.3)',
                  transition: 'all 0.2s ease',
                }}
              >
                Return to AI Scanner
              </button>
            </div>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
