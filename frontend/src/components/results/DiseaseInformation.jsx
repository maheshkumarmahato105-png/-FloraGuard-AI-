import { BookOpen, Info, ShieldCheck, Stethoscope, Wrench } from 'lucide-react';

export function DiseaseInformation({ diseaseInfo, isUncertain = false }) {
  // STRICT RULE: Never display treatment or disease guidance for uncertain predictions
  if (isUncertain) {
    return null;
  }

  if (!diseaseInfo) {
    return (
      <section className="card" aria-label="Disease information">
        <div className="card-header">
          <h3 className="card-title">
            <BookOpen size={17} aria-hidden="true" />
            Disease &amp; Agronomic Guidance
          </h3>
        </div>
        <div className="alert alert-info">
          <Info size={18} className="alert-icon" aria-hidden="true" />
          <div className="alert-body">
            <p className="alert-title">Detailed guidance not stored in history record</p>
            <p className="alert-text">
              The Phase 7 backend returns detailed symptoms, prevention, and treatment guidance in
              the live prediction response, while SQLite history stores compact telemetry. Run a
              fresh scan to view complete educational disease metadata.
            </p>
          </div>
        </div>
      </section>
    );
  }

  const sections = [
    {
      key: 'symptoms',
      title: 'Observed Symptoms',
      icon: Stethoscope,
      items: diseaseInfo.symptoms || [],
    },
    {
      key: 'prevention',
      title: 'Prevention Practices',
      icon: ShieldCheck,
      items: diseaseInfo.prevention || [],
    },
    {
      key: 'treatment',
      title: 'Treatment Guidance',
      icon: Wrench,
      items: diseaseInfo.treatment || [],
    },
  ];

  return (
    <section className="card" aria-label="Disease information and treatment guidance">
      <div className="card-header">
        <div>
          <h3 className="card-title">
            <BookOpen size={17} aria-hidden="true" />
            {diseaseInfo.disease_name || 'Agronomic Reference'}
          </h3>
          {diseaseInfo.affected_plant ? (
            <p className="card-subtitle">Host crop: {diseaseInfo.affected_plant}</p>
          ) : null}
        </div>
      </div>

      {diseaseInfo.summary ? (
        <p style={{ margin: '0 0 18px', color: 'var(--text-main)', fontSize: '0.93rem' }}>
          {diseaseInfo.summary}
        </p>
      ) : null}

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: 16,
        }}
      >
        {sections.map(({ key, title, icon: Icon, items }) => (
          <div
            key={key}
            style={{
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-md)',
              padding: 14,
            }}
          >
            <h4
              style={{
                margin: 0,
                fontSize: '0.9rem',
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                gap: 7,
              }}
            >
              <Icon size={15} style={{ color: 'var(--accent-primary)' }} aria-hidden="true" />
              {title}
            </h4>
            {items.length > 0 ? (
              <ul className="guidance-list">
                {items.map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            ) : (
              <p style={{ margin: '8px 0 0', fontSize: '0.84rem', color: 'var(--text-muted)' }}>
                No specific guidance entries returned.
              </p>
            )}
          </div>
        ))}
      </div>

      {diseaseInfo.disclaimer ? (
        <div className="xai-caption" style={{ marginTop: 16 }}>
          <strong>Disclaimer:</strong> {diseaseInfo.disclaimer}
        </div>
      ) : null}
    </section>
  );
}

export default DiseaseInformation;
