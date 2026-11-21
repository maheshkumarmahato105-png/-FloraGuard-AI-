import { Link, useNavigate } from 'react-router-dom';
import { ArrowLeft, CheckCircle2, Download, FileText, Leaf, Printer, RefreshCw, ShieldAlert, Sparkles } from 'lucide-react';

export function SampleReport() {
  const navigate = useNavigate();

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="report-view-wrapper">
      {/* Top Action Control Bar (Hidden when printing) */}
      <div className="report-top-bar no-print">
        <div className="top-bar-left">
          <button type="button" className="btn-report-action" onClick={() => navigate('/')}>
            <ArrowLeft size={16} aria-hidden="true" />
            <span>Back to Dashboard</span>
          </button>
        </div>
        <div className="top-bar-right">
          <button type="button" className="btn-report-action btn-report-primary" onClick={handlePrint}>
            <Printer size={15} aria-hidden="true" />
            <span>Print Report</span>
          </button>
          <button type="button" className="btn-report-action" onClick={handlePrint}>
            <Download size={15} aria-hidden="true" />
            <span>Download PDF</span>
          </button>
          <Link to="/detection" className="btn-report-action">
            <RefreshCw size={15} aria-hidden="true" />
            <span>New Live Scan</span>
          </Link>
        </div>
      </div>

      {/* Printable A4 Canvas Container */}
      <div className="report-paper-container">
        <article className="report-paper" id="printableReportPaper">
          {/* 1. Header */}
          <header className="report-header-section">
            <div className="brand-header-box">
              <div className="brand-header-icon" aria-hidden="true">
                <Leaf size={32} />
              </div>
              <div className="brand-header-text">
                <h1>FLORAGUARD AI</h1>
                <p>Plant Disease Detection &amp; Crop Care Pathology System</p>
                <span style={{ fontSize: 11, color: '#059669', fontWeight: 600 }}>
                  AI-Assisted Plant Pathology Diagnostic Report
                </span>
              </div>
            </div>

            <div className="report-header-meta">
              <span className="badge-ai-diagnosis">AI-ASSISTED DIAGNOSIS</span>
              <div className="meta-line">REPORT ID: <strong>FLORA-DIAG-716808</strong></div>
              <div className="meta-line">MODEL: <strong>AgriVision-CNN</strong></div>
              <div className="meta-line">DATE: <strong>21/11/2026</strong></div>
              <div className="meta-line">TIME: <strong>07:30 PM</strong></div>
            </div>
          </header>

          {/* 2. Top Summary Banner */}
          <section className="report-title-banner">
            <h2 className="report-main-title">PLANT DISEASE DIAGNOSTIC REPORT</h2>
            <div className="report-main-subtitle">Automated Leaf Image Analysis &amp; Crop Care Recommendations</div>

            <div className="title-summary-grid">
              <div className="summary-item">
                <span className="summary-label">Target Plant</span>
                <span className="summary-val">Tomato (Solanum lycopersicum)</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Predicted Pathology</span>
                <span className="summary-val">Tomato Early Blight</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Diagnosis Status</span>
                <span className="summary-val" style={{ color: '#dc2626' }}>DISEASE DETECTED</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Calibrated Confidence</span>
                <span className="summary-val">94.6%</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Estimated Severity</span>
                <span className="summary-val" style={{ color: '#b45309' }}>Moderate</span>
              </div>
            </div>
          </section>

          {/* 3. Executive Summary */}
          <section className="report-section">
            <h3 className="section-heading">
              <FileText size={16} aria-hidden="true" />
              Executive Summary
            </h3>
            <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 14, fontSize: 13, lineHeight: 1.6, color: '#334155' }}>
              FloraGuard AI processed the submitted leaf photograph with the calibrated AgriVision-CNN computer vision network.
              The network localized distinctive concentric rings and chlorotic yellowing margins characteristic of Alternaria solani (Early Blight).
              Uncertainty telemetry passed all distribution reliability thresholds with high confidence (94.6%), Shannon entropy 0.28, and top-2 margin 0.81.
            </div>
          </section>

          {/* 4. Top-3 Model Probability Breakdown */}
          <section className="report-section">
            <h3 className="section-heading">Top Model Class Probabilities</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {[
                { label: 'Tomato Early Blight (Alternaria solani)', pct: 94.6, color: '#059669' },
                { label: 'Tomato Leaf Mold (Passalora fulva)', pct: 2.8, color: '#94a3b8' },
                { label: 'Tomato Septoria Leaf Spot', pct: 1.7, color: '#94a3b8' },
                { label: 'Healthy Tomato Foliage', pct: 0.9, color: '#94a3b8' },
              ].map((item) => (
                <div key={item.label}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12.5, fontWeight: 600, color: '#1e293b', marginBottom: 4 }}>
                    <span>{item.label}</span>
                    <span className="tabular">{item.pct}%</span>
                  </div>
                  <div style={{ width: '100%', height: 10, background: '#e2e8f0', borderRadius: 5, overflow: 'hidden' }}>
                    <div style={{ width: `${item.pct}%`, height: '100%', background: item.color, borderRadius: 5 }} />
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* 5. Observed Symptoms & Severity Gauge */}
          <div className="report-symptoms-severity-grid">
            <section className="report-section" style={{ margin: 0 }}>
              <h3 className="section-heading">Observed Pathology Symptoms</h3>
              <ul style={{ margin: '8px 0 0 18px', padding: 0, fontSize: 12.5, color: '#334155', lineHeight: 1.6 }}>
                <li>Small brown-to-black necrotic spots developing on older lower foliage.</li>
                <li>Concentric ring target-like patterns forming within lesions.</li>
                <li>Chlorotic yellow haloes surrounding diseased leaf margins.</li>
                <li>Premature defoliation exposing developing fruit to sunscald.</li>
              </ul>
            </section>

            <section className="report-section" style={{ margin: 0 }}>
              <h3 className="section-heading">Severity Index (Moderate)</h3>
              <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 14 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, fontWeight: 700 }}>
                  <span style={{ color: '#059669' }}>MILD</span>
                  <span style={{ color: '#d97706' }}>MODERATE</span>
                  <span style={{ color: '#dc2626' }}>SEVERE</span>
                </div>
                <div style={{ width: '100%', height: 12, background: '#e2e8f0', borderRadius: 6, margin: '8px 0', overflow: 'hidden' }}>
                  <div style={{ width: '58%', height: '100%', background: 'linear-gradient(90deg, #10b981, #f59e0b)', borderRadius: 6 }} />
                </div>
                <p style={{ margin: 0, fontSize: 11.5, color: '#64748b' }}>
                  Lesions currently encompass 15–25% of lower canopy foliage. Prompt cultural and fungicidal intervention will safeguard fruit yield.
                </p>
              </div>
            </section>
          </div>

          {/* 6. Agronomic Care & Treatment Guidance */}
          <section className="report-section">
            <h3 className="section-heading">Recommended Agronomic Management</h3>
            <div className="report-management-grid">
              <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 12, fontSize: 12, color: '#334155' }}>
                <strong style={{ color: '#064e3b', display: 'block', marginBottom: 6 }}>1. Cultural Sanitation</strong>
                Prune and burn or bag infected lower leaves. Maintain 24-inch row spacing to facilitate air circulation and rapid canopy drying.
              </div>
              <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 12, fontSize: 12, color: '#334155' }}>
                <strong style={{ color: '#064e3b', display: 'block', marginBottom: 6 }}>2. Irrigation Protocol</strong>
                Switch from overhead sprinklers to soil-level drip irrigation. Water exclusively early in the morning so sun evaporates any splashed water.
              </div>
              <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 12, fontSize: 12, color: '#334155' }}>
                <strong style={{ color: '#064e3b', display: 'block', marginBottom: 6 }}>3. Organic / Chemical Control</strong>
                Apply copper octanoate or chlorothalonil fungicide every 7 to 10 days until dry conditions return. Rotate with biofungicides (Bacillus amyloliquefaciens).
              </div>
            </div>
          </section>

          {/* 7. Disclaimer */}
          <div className="disclaimer-box">
            <strong>DISCLAIMER:</strong> FloraGuard AI provides AI-assisted image-based plant disease assessment for educational, research, and preliminary agronomic decision-support purposes. The output does not replace certified laboratory diagnosis by a licensed agricultural extension plant pathologist.
          </div>

          {/* 8. Footer */}
          <footer className="report-footer-section">
            <div><strong>FloraGuard AI</strong> | Plant Disease Detection &amp; Crop Care System</div>
            <div>Report ID: <strong>FLORA-DIAG-716808</strong> | Generated: 21/11/2026 07:30 PM</div>
            <div><em>AI-Assisted Agricultural Intelligence</em> | Page 1 of 1</div>
          </footer>
        </article>
      </div>
    </div>
  );
}

export default SampleReport;
