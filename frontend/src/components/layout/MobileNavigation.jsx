import { NavLink } from 'react-router-dom';
import { NAV_ITEMS } from './Sidebar';

export function MobileNavigation() {
  return (
    <nav className="mobile-nav" aria-label="Mobile bottom navigation">
      <ul className="mobile-nav-list">
        {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
          <li key={to}>
            <NavLink
              to={to}
              end={end}
              className={({ isActive }) => `mobile-nav-link${isActive ? ' active' : ''}`}
            >
              <Icon size={19} aria-hidden="true" />
              <span>{label}</span>
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  );
}

export default MobileNavigation;
