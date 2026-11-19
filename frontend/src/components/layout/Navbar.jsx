import { useLocation } from 'react-router-dom';
import { Leaf, Moon, RefreshCw, Sun } from 'lucide-react';
import { useTheme } from '../../context/ThemeContext';
import StatusIndicator from '../common/StatusIndicator';

function resolveRouteLabel(pathname) {
  if (pathname === '/') return 'Dashboard Overview';
  if (pathname.startsWith('/detection')) return 'Plant Disease Detection';
  if (pathname.startsWith('/results')) return 'Prediction Analysis';
  if (pathname.startsWith('/history')) return 'Prediction History';
  if (pathname.startsWith('/settings')) return 'Application Settings';
  return 'PlantDiseaseAI';
}

export function Navbar({ health, loading, onRefreshHealth }) {
  const location = useLocation();
  const { theme, toggleTheme } = useTheme();
  const routeLabel = resolveRouteLabel(location.pathname);

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
    <header className="top-navbar">
      <div className="navbar-left">
        <div className="navbar-mobile-brand">
          <span className="brand-mark" style={{ width: 32, height: 32 }} aria-hidden="true">
            <Leaf size={17} />
          </span>
          <span>PlantDiseaseAI</span>
        </div>
        <span className="navbar-breadcrumb">{routeLabel}</span>
      </div>

      <div className="navbar-right">
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
            <RefreshCw size={16} aria-hidden="true" />
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
            <Sun size={16} aria-hidden="true" />
          ) : (
            <Moon size={16} aria-hidden="true" />
          )}
        </button>
      </div>
    </header>
  );
}

export default Navbar;
