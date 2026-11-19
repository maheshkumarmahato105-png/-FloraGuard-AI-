import { Link } from 'react-router-dom';
import { Compass, Home, ScanSearch } from 'lucide-react';
import PageContainer from '../components/layout/PageContainer';

export function NotFound() {
  return (
    <PageContainer>
      <div className="empty-state" style={{ maxWidth: 540, margin: '40px auto', padding: '44px 24px' }}>
        <div className="empty-state-icon" aria-hidden="true">
          <Compass size={26} />
        </div>
        <div className="mono" style={{ fontSize: '0.82rem', color: 'var(--accent-primary)', marginBottom: 6 }}>
          ERROR 404
        </div>
        <h1 className="page-title" style={{ fontSize: '1.45rem', marginBottom: 8 }}>
          Page Not Found
        </h1>
        <p className="empty-state-desc">
          The route you navigated to does not exist in the PlantDiseaseAI workspace.
        </p>
        <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: 10 }}>
          <Link to="/" className="btn btn-primary">
            <Home size={16} aria-hidden="true" />
            Return to Dashboard
          </Link>
          <Link to="/detection" className="btn btn-secondary">
            <ScanSearch size={16} aria-hidden="true" />
            Go to Detection
          </Link>
        </div>
      </div>
    </PageContainer>
  );
}

export default NotFound;
