import { NavLink } from 'react-router-dom';
import {
  History as HistoryIcon,
  LayoutDashboard,
  Leaf,
  ScanSearch,
  Settings as SettingsIcon,
} from 'lucide-react';
import StatusIndicator from '../common/StatusIndicator';

export const NAV_ITEMS = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/detection', label: 'Detection', icon: ScanSearch, end: false },
  { to: '/history', label: 'History', icon: HistoryIcon, end: false },
  { to: '/settings', label: 'Settings', icon: SettingsIcon, end: false },
];

export function Sidebar({ health, systemInfo, loading }) {
  const apiTone = loading
    ? 'neutral'
    : health?.api_status === 'online'
      ? 'healthy'
      : 'danger';
  const apiLabel = loading
    ? 'Checking…'
    : health?.api_status === 'online'
      ? 'API Online'
      : 'API Offline';

  const modelTone = loading
    ? 'neutral'
    : !health
      ? 'danger'
      : health.model_ready
        ? 'healthy'
        : 'warning';
  const modelLabel = loading
    ? 'Checking…'
    : !health
      ? 'Unavailable'
      : health.model_ready
        ? 'Model Ready'
        : 'No Checkpoint';

  return (
    <aside className="sidebar" aria-label="Primary sidebar navigation">
      <div className="sidebar-brand">
        <div className="brand-mark" aria-hidden="true">
          <Leaf size={20} />
        </div>
        <div>
          <div className="brand-title">PlantDiseaseAI</div>
          <div className="brand-subtitle">
            {systemInfo?.model_name || 'AgriVision-CNN'}
          </div>
        </div>
      </div>

      <nav className="sidebar-nav" aria-label="Main navigation">
        {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
          >
            <Icon size={18} aria-hidden="true" />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="sidebar-meta">
          <div className="sidebar-meta-row">
            <span>Backend</span>
            <StatusIndicator label={apiLabel} tone={apiTone} />
          </div>
          <div className="sidebar-meta-row">
            <span>Inference</span>
            <StatusIndicator label={modelLabel} tone={modelTone} />
          </div>
        </div>
      </div>
    </aside>
  );
}

export default Sidebar;
