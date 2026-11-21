import { NavLink, useLocation } from 'react-router-dom';
import { NAV_ITEMS } from './Sidebar';

export function MobileNavigation() {
  const location = useLocation();

  if (location.pathname.startsWith('/results') || location.pathname === '/sample-report') {
    return null;
  }
  return (
    <nav className="mobile-nav" aria-label="Mobile bottom navigation">
      <ul className="mobile-nav-list">
        {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
          <li key={to} className="mobile-nav-item">
            <NavLink
              to={to}
              end={end}
              className={({ isActive }) => `mobile-nav-link${isActive ? ' active' : ''}`}
            >
              <Icon size={19} aria-hidden="true" className="mobile-nav-icon" />
              <span className="mobile-nav-label">{label}</span>
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  );
}

export default MobileNavigation;
