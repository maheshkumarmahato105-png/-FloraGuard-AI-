import { useNavigate } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';

export function PageContainer({
  title,
  subtitle,
  actions,
  showBack = false,
  backTo = null,
  backLabel = 'Back',
  children,
}) {
  const navigate = useNavigate();

  const handleBack = () => {
    if (backTo) {
      navigate(backTo);
    } else {
      navigate(-1);
    }
  };

  return (
    <main id="main-content" className="page-container" tabIndex={-1}>
      {(title || subtitle || actions || showBack) ? (
        <div className="page-header">
          <div>
            {showBack ? (
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                style={{ marginBottom: 8, paddingLeft: 4 }}
                onClick={handleBack}
              >
                <ArrowLeft size={15} aria-hidden="true" />
                <span>{backLabel}</span>
              </button>
            ) : null}
            {title ? <h1 className="page-title">{title}</h1> : null}
            {subtitle ? <p className="page-subtitle">{subtitle}</p> : null}
          </div>
          {actions ? <div className="page-actions">{actions}</div> : null}
        </div>
      ) : null}
      {children}
    </main>
  );
}

export default PageContainer;
