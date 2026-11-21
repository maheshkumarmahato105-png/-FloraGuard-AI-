import { Link, NavLink } from 'react-router-dom';
import { FileText, History as HistoryIcon, Moon, RefreshCw, Settings as SettingsIcon, Sun } from 'lucide-react';
import { useTheme } from '../../context/ThemeContext';
import StatusIndicator from '../common/StatusIndicator';

export function Navbar({ health, loading, onRefreshHealth }) {
  const { theme, toggleTheme } = useTheme();

  const statusTone = loading
    ? 'neutral'
    : !health
      ? 'danger'
      : health.status === 'healthy'
        ? 'healthy'
        : health.status === 'degraded'
          ? 'warning'
          : 'danger';

  const statusText = loading
    ? 'Connecting…'
    : !health
      ? 'Backend Offline'
      : health.status === 'healthy'
        ? 'System Ready'
        : health.status === 'degraded'
          ? 'Model Unavailable'
          : 'System Unhealthy';

  return (
    <header className="app-header top-navbar">
      <div className="navbar-left">
        <Link to="/" className="brand-logo" aria-label="FloraGuard AI Home">
          <div className="brand-icon" aria-hidden="true">🌿</div>
          <span>Flora<span style={{ color: 'var(--accent-primary)' }}>Guard</span> AI</span>
        </Link>
      </div>

      <nav className="nav-menu" aria-label="Main application tabs">
        <NavLink
          to="/"
          end
          className={({ isActive }) => `nav-link-pill${isActive ? ' active' : ''}`}
        >
          <span className="nav-pill-dot" aria-hidden="true" />
          <span>AI Scanner</span>
        </NavLink>
        <NavLink
          to="/sample-report"
          className={({ isActive }) => `nav-link-pill${isActive ? ' active' : ''}`}
        >
          <FileText size={15} aria-hidden="true" />
          <span>Sample A4 Diagnostic Report</span>
        </NavLink>
        <NavLink
          to="/history"
          className={({ isActive }) => `nav-link-pill${isActive ? ' active' : ''}`}
        >
          <HistoryIcon size={15} aria-hidden="true" />
          <span>History</span>
        </NavLink>
        <NavLink
          to="/settings"
          className={({ isActive }) => `nav-link-pill${isActive ? ' active' : ''}`}
        >
          <SettingsIcon size={15} aria-hidden="true" />
          <span>Settings</span>
        </NavLink>
      </nav>

      <div className="navbar-right" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <StatusIndicator
          label={statusText}
          tone={statusTone}
          title={health?.model_status_detail || statusText}
        />

        {typeof onRefreshHealth === 'function' ? (
          <button
            type="button"
            className="btn btn-secondary btn-icon"
            onClick={onRefreshHealth}
            aria-label="Refresh backend connection status"
            title="Refresh backend status"
          >
            <RefreshCw size={15} aria-hidden="true" className={loading ? 'spin' : ''} />
          </button>
        ) : null}

        <button
          type="button"
          className="btn btn-secondary btn-icon"
          onClick={toggleTheme}
          aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
          title={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
        >
          {theme === 'dark' ? (
            <Sun size={15} aria-hidden="true" />
          ) : (
            <Moon size={15} aria-hidden="true" />
          )}
        </button>
      </div>
    </header>
  );
}

export default Navbar;
