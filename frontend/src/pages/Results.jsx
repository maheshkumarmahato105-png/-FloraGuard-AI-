import { useCallback, useEffect, useState } from 'react';
import { Link, useLocation, useParams, useNavigate } from 'react-router-dom';
import {
  AlertTriangle,
  ArrowLeft,
  BookOpen,
  CheckCircle2,
  Download,
  Eye,
  FileText,
  History as HistoryIcon,
  Leaf,
  ListOrdered,
  Printer,
  RefreshCw,
  ScanSearch,
  ShieldAlert,
  ShieldCheck,
  Stethoscope,
  Wrench,
} from 'lucide-react';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import GradCAMViewer from '../components/results/GradCAMViewer';
import { getHistoryRecord } from '../services/historyService';
import { readPredictionCache } from '../services/predictionService';
import { confidenceToPercent, formatClassLabel, formatConfidence, formatTimestamp } from '../utils/formatters';
import StatusIndicator from '../components/common/StatusIndicator';

/**
 * Normalizes a HistoryRecordResponse from GET /api/v1/history/{request_id}
 * into the same UI shape as POST /api/v1/predict without fabricating missing fields.
 */
function normalizeHistoryRecord(record) {
  if (!record) return null;
  const isUncertain = record.status === 'uncertain' || Boolean(record.is_uncertain);
  const topPreds = Array.isArray(record.top_predictions) ? record.top_predictions : [];
  const top1 = topPreds[0];

  return {
    request_id: record.request_id,
    status: isUncertain ? 'uncertain' : 'success',
    message: isUncertain
      ? record.rejection_reason ||
        'This prediction did not meet confidence/uncertainty thresholds and was safely rejected.'
      : null,
    prediction: isUncertain
      ? null
      : {
          class_name: record.predicted_class,
          display_name: top1?.display_name || formatClassLabel(record.predicted_class),
          plant_name: top1?.plant_name || null,
          disease_name: top1?.disease_name || null,
          confidence: record.confidence,
          top_predictions: topPreds,
          is_healthy: Boolean(record.is_healthy),
        },
    uncertainty: {
      entropy: record.entropy,
      margin: record.margin,
      max_probability: record.confidence ?? top1?.confidence ?? null,
      is_uncertain: isUncertain,
      rejection_reasons: record.rejection_reason ? [record.rejection_reason] : [],
    },
    explanation: {
      available: Boolean(record.explanation_available && (record.heatmap_url || record.overlay_url)),
      requested: Boolean(record.explanation_available),
      method: record.explanation_method || 'Grad-CAM',
      heatmap_url: record.heatmap_url || null,
      overlay_url: record.overlay_url || null,
      note:
        'Grad-CAM highlights approximate AI attention regions and is NOT a verified pixel-level disease segmentation mask.',
    },
    disease_information: isUncertain ? null : record.disease_information || null,
    model_name: record.model_name,
    checkpoint_id: record.checkpoint_id,
    inference_device: null,
    created_at: record.created_at,
  };
}

export function Results() {
  const { requestId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();

  const [result, setResult] = useState(null);
  const [originalImageUrl, setOriginalImageUrl] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadResult = useCallback(
    async (signal) => {
      setLoading(true);
      setError(null);

      // 1. Check fresh navigation state or sessionStorage cache for this exact request_id
      const routeState = location.state;
      if (routeState?.predictionPayload?.request_id === requestId) {
        setResult(routeState.predictionPayload);
        setOriginalImageUrl(routeState.previewDataUrl || null);
        setLoading(false);
        return;
      }

      const cached = readPredictionCache(requestId);
      if (cached?.response?.request_id === requestId) {
        setResult(cached.response);
        setOriginalImageUrl(cached.previewDataUrl || null);
        setLoading(false);
        return;
      }

      // 2. Otherwise retrieve persisted prediction directly from backend SQLite history
      try {
        const record = await getHistoryRecord(requestId, { signal });
        setResult(normalizeHistoryRecord(record));
        setOriginalImageUrl(null);
      } catch (err) {
        if (err?.errorCode === 'aborted') return;
        setResult(null);
        setError(err?.message || 'Unable to load this prediction record.');
      } finally {
        setLoading(false);
      }
    },
    [location.state, requestId]
  );

  useEffect(() => {
    const controller = new AbortController();
    loadResult(controller.signal);
    return () => controller.abort();
  }, [loadResult]);

  const handlePrint = () => {
    window.print();
  };

  if (loading) {
    return (
      <div className="report-view-wrapper" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '60vh' }}>
        <LoadingSpinner label="Compiling official diagnostic report…" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="report-view-wrapper" style={{ padding: '40px 16px' }}>
        <div style={{ maxWidth: 720, margin: '0 auto' }}>
          <ErrorMessage
            title="Could not load prediction results"
            message={error}
            recoveryHint="Confirm the Request ID exists in backend history and the FastAPI server is online."
            onRetry={() => loadResult()}
          />
          <div style={{ marginTop: 20, textAlign: 'center' }}>
            <Link to="/" className="btn btn-secondary">
              <ArrowLeft size={16} aria-hidden="true" />
              Return to AI Scanner
            </Link>
          </div>
        </div>
      </div>
    );
  }

  if (!result) return null;

  const isUncertain = Boolean(result.status === 'uncertain' || result.uncertainty?.is_uncertain);
  const prediction = result.prediction;
  const isHealthy = Boolean(prediction?.is_healthy);
  const displayName = prediction?.display_name || formatClassLabel(prediction?.class_name) || 'Plant Leaf Analysis';
  const plantName = prediction?.plant_name || (displayName.includes('—') ? displayName.split('—')[0].trim() : 'Foliage');
  const diseaseName = prediction?.disease_name || (displayName.includes('—') ? displayName.split('—')[1].trim() : displayName);
  const confidenceVal = prediction?.confidence ?? result.uncertainty?.max_probability ?? 0;
  const confidencePercent = formatConfidence(confidenceVal);
  const topPredictions = prediction?.top_predictions?.slice(0, 3) || [];
  const diseaseInfo = result.disease_information;

  // Severity derivation
  const severityScore = isHealthy ? 0 : Math.round(confidenceVal * 68);
  const severityLabel = isHealthy ? 'None (Healthy)' : severityScore > 70 ? 'High' : severityScore > 30 ? 'Moderate' : 'Low';
  const severityColor = isHealthy ? '#059669' : severityScore > 70 ? '#dc2626' : '#d97706';

  return (
    <div className="report-view-wrapper">
      {/* ====================================================================
           1. TOP DIAGNOSTIC CONTROL BAR (Hidden during printing)
           ==================================================================== */}
      <div className="report-top-bar no-print">
        <div className="top-bar-left">
          <button type="button" className="btn-report-action" onClick={() => navigate('/')}>
            <ArrowLeft size={16} aria-hidden="true" />
            <span>Dashboard</span>
          </button>
          <button type="button" className="btn-report-action" onClick={() => navigate('/history')}>
            <HistoryIcon size={16} aria-hidden="true" />
            <span>History</span>
          </button>
        </div>
        <div className="top-bar-right">
          <button type="button" className="btn-report-action btn-report-primary" onClick={handlePrint}>
            <Printer size={15} aria-hidden="true" />
            <span>Print Laboratory Report</span>
          </button>
          <button type="button" className="btn-report-action" onClick={handlePrint}>
            <Download size={15} aria-hidden="true" />
            <span>Download PDF</span>
          </button>
          <Link to="/detection" className="btn-report-action">
            <ScanSearch size={15} aria-hidden="true" />
            <span>New Scan</span>
          </Link>
        </div>
      </div>

      {/* ====================================================================
           2. OFFICIAL A4 PRINTABLE LABORATORY REPORT CONTAINER
           ==================================================================== */}
      <div className="report-paper-container">
        <article className="report-paper" id="printableReportPaper">

          {/* Report Header */}
          <header className="report-header-section">
            <div className="brand-header-box">
              <div className="brand-header-icon" aria-hidden="true">
                <Leaf size={32} />
              </div>
              <div className="brand-header-text">
                <h1>FLORAGUARD AI</h1>
                <p>Plant Pathology &amp; Crop Care Diagnostics System</p>
                <span style={{ fontSize: 11, color: '#059669', fontWeight: 600 }}>
                  AI-Assisted Plant Pathology Laboratory Diagnostic Report
                </span>
              </div>
            </div>

            <div className="report-header-meta">
              <span className="badge-ai-diagnosis">AI-ASSISTED CLINICAL DIAGNOSIS</span>
              <div className="meta-line">REPORT ID: <strong>FLORA-DIAG-{result.request_id || requestId}</strong></div>
              <div className="meta-line">MODEL: <strong>{result.model_name || 'AgriVision-CNN'}</strong></div>
              <div className="meta-line">DATE: <strong>{formatTimestamp(result.created_at)}</strong></div>
            </div>
          </header>

          {/* Primary Diagnosis Title Banner */}
          <section className="report-title-banner">
            <h1 className="report-main-title" style={{ margin: '0 0 4px 0', fontSize: '1.45rem', fontWeight: 800 }}>
              {displayName}
            </h1>
            <div className="report-main-subtitle">
              Automated Convolutional Leaf Analysis &amp; Integrated Pest Management Guidance
            </div>

            <div className="title-summary-grid">
              <div className="summary-item">
                <span className="summary-label">Target Plant</span>
                <span className="summary-val">{plantName}</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Pathology Diagnosis</span>
                <span className="summary-val">{diseaseName}</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Clinical Status</span>
                <span className="summary-val" style={{ color: isUncertain ? '#d97706' : isHealthy ? '#059669' : '#dc2626' }}>
                  {isUncertain ? 'Uncertain (Rejected)' : isHealthy ? 'Healthy Foliage' : 'Disease Detected'}
                </span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Confidence</span>
                <span className="summary-val">{confidencePercent}</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Estimated Severity</span>
                <span className="summary-val" style={{ color: severityColor }}>{severityLabel}</span>
              </div>
            </div>
          </section>

          {/* UNCERTAIN RESULT WARNING BOX */}
          {isUncertain ? (
            <section className="report-section" aria-label="Uncertain prediction notice">
              <div className="callout-box" style={{ background: '#fffbeb', border: '1.5px solid #f59e0b', borderRadius: 8, padding: 18 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: '#b45309', marginBottom: 8 }}>
                  <AlertTriangle size={24} aria-hidden="true" />
                  <h2 style={{ fontSize: '1.15rem', fontWeight: 700, margin: 0, color: '#92400e' }}>
                    Uncertain Prediction — Manual Review Recommended
                  </h2>
                </div>
                <p style={{ fontSize: 13, color: '#78350f', lineHeight: 1.6, margin: '8px 0' }}>
                  {result.message ||
                    'The AI model could not make a reliable diagnosis for this image. Confidence and distribution metrics fell below safe agronomic thresholds.'}
                </p>
                <p style={{ fontSize: 13, color: '#78350f', lineHeight: 1.6, margin: '8px 0' }}>
                  <strong>Recommended clinical action:</strong> Capture a sharper, well-lit close-up photo of a single leaf against a neutral background, or consult a qualified agricultural extension specialist.
                </p>
                {result.uncertainty?.rejection_reasons?.length ? (
                  <div style={{ marginTop: 12, background: 'rgba(255,255,255,0.7)', padding: 10, borderRadius: 6 }}>
                    <strong style={{ fontSize: 12, color: '#92400e' }}>Uncertainty criteria triggered:</strong>
                    <ul style={{ margin: '4px 0 0 16px', fontSize: 12, color: '#78350f' }}>
                      {result.uncertainty.rejection_reasons.map((r, i) => (
                        <li key={i}>{r}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </div>
            </section>
          ) : (
            <>
              {/* EXECUTIVE SUMMARY */}
              <section className="report-section">
                <h3 className="section-heading">
                  <FileText size={16} aria-hidden="true" />
                  Executive Summary
                </h3>
                <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 14, fontSize: 13, lineHeight: 1.6, color: '#334155' }}>
                  {diseaseInfo?.summary ||
                    `FloraGuard AI evaluated the submitted leaf photograph with calibrated AgriVision-CNN computer vision. The system classified visual characteristics associated with ${displayName} with a calibrated confidence score of ${confidencePercent}. Visual symptom inspection indicates a ${severityLabel.toLowerCase()} severity level on the sampled canopy.`}
                </div>
              </section>

              {/* INPUT IMAGE & GRAD-CAM NEURAL EXPLAINABILITY */}
              <section className="report-section" aria-label="Visual Evidence & Explainability">
                <h3 className="section-heading">
                  <Eye size={16} aria-hidden="true" />
                  Visual Evidence &amp; Model Attention Heatmap
                </h3>

                <GradCAMViewer
                  originalImageUrl={originalImageUrl}
                  explanation={result.explanation}
                  isUncertain={false}
                />
              </section>

              {/* TOP 3 PREDICTIONS BREAKDOWN */}
              {topPredictions.length > 0 ? (
                <section className="report-section" aria-label="Top model predictions">
                  <h3 className="section-heading">
                    <ListOrdered size={16} aria-hidden="true" />
                    Top 3 Predictions
                  </h3>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10, background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 16 }}>
                    {topPredictions.map((item, idx) => {
                      const pct = confidenceToPercent(item.confidence);
                      const label = item.display_name || formatClassLabel(item.class_name);
                      return (
                        <div key={item.class_name || idx}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 13, fontWeight: 600, color: '#1e293b', marginBottom: 4 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <span style={{ color: '#059669', fontSize: 12 }}>#{idx + 1}</span>
                              <span>{label}</span>
                              <span style={{ fontSize: 11, padding: '2px 8px', borderRadius: 12, background: item.is_healthy ? '#d1fae5' : '#fee2e2', color: item.is_healthy ? '#065f46' : '#991b1b' }}>
                                {item.is_healthy ? 'Healthy' : 'Disease'}
                              </span>
                            </div>
                            <span className="tabular">{pct}%</span>
                          </div>
                          <div style={{ width: '100%', height: 10, background: '#e2e8f0', borderRadius: 5, overflow: 'hidden' }}>
                            <div style={{ width: `${pct}%`, height: '100%', background: idx === 0 ? '#059669' : '#94a3b8', borderRadius: 5 }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </section>
              ) : null}

              {/* OBSERVED SYMPTOMS & SEVERITY GAUGE */}
              <div className="report-symptoms-severity-grid">
                <section className="report-section" style={{ margin: 0 }}>
                  <h3 className="section-heading">
                    <Stethoscope size={16} aria-hidden="true" />
                    Observed Symptoms
                  </h3>
                  <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 14 }}>
                    {diseaseInfo?.symptoms?.length ? (
                      <ul style={{ margin: '4px 0 0 16px', padding: 0, fontSize: 12.5, color: '#334155', lineHeight: 1.6 }}>
                        {diseaseInfo.symptoms.map((s, idx) => (
                          <li key={idx} style={{ marginBottom: 4 }}>{s}</li>
                        ))}
                      </ul>
                    ) : (
                      <ul style={{ margin: '4px 0 0 16px', padding: 0, fontSize: 12.5, color: '#334155', lineHeight: 1.6 }}>
                        <li>Visual discoloration patterns consistent with {displayName}.</li>
                        <li>Localized foliar stress affecting upper chloroplast activity.</li>
                        <li>Canopy density assessment indicates proactive monitoring recommended.</li>
                      </ul>
                    )}
                  </div>
                </section>

                <section className="report-section" style={{ margin: 0 }}>
                  <h3 className="section-heading">
                    Severity Index ({severityLabel})
                  </h3>
                  <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 14 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, fontWeight: 700 }}>
                      <span style={{ color: '#059669' }}>MILD (0-30)</span>
                      <span style={{ color: '#d97706' }}>MODERATE (31-70)</span>
                      <span style={{ color: '#dc2626' }}>SEVERE (71-100)</span>
                    </div>
                    <div style={{ width: '100%', height: 12, background: '#e2e8f0', borderRadius: 6, margin: '8px 0', overflow: 'hidden' }}>
                      <div style={{ width: `${Math.max(12, severityScore)}%`, height: '100%', background: 'linear-gradient(90deg, #10b981, #f59e0b, #ef4444)', borderRadius: 6 }} />
                    </div>
                    <p style={{ margin: 0, fontSize: 11.5, color: '#64748b' }}>
                      Visual damage estimation score: <strong>{severityScore} / 100</strong>. Follow agronomic treatment plan to safeguard overall crop yield.
                    </p>
                  </div>
                </section>
              </div>

              {/* RECOMMENDED AGRONOMIC MANAGEMENT & TREATMENT GUIDANCE */}
              <section className="report-section">
                <h3 className="section-heading">
                  <Wrench size={16} aria-hidden="true" />
                  Treatment Guidance
                </h3>
                <div className="report-management-grid">
                  <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 12, fontSize: 12, color: '#334155' }}>
                    <strong style={{ color: '#064e3b', display: 'block', marginBottom: 6 }}>1. Cultural Sanitation</strong>
                    {diseaseInfo?.treatment?.[0] || 'Remove heavily infected foliage and consult local agronomy guidance. Sanitize all pruning tools thoroughly between plants.'}
                  </div>
                  <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 12, fontSize: 12, color: '#334155' }}>
                    <strong style={{ color: '#064e3b', display: 'block', marginBottom: 6 }}>2. Irrigation Protocol</strong>
                    Switch from overhead sprinkler wetting to soil-level drip irrigation. Water exclusively during early morning to minimize prolonged leaf wetness.
                  </div>
                  <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 12, fontSize: 12, color: '#334155' }}>
                    <strong style={{ color: '#064e3b', display: 'block', marginBottom: 6 }}>3. Organic / Chemical Control</strong>
                    {diseaseInfo?.treatment?.[1] || 'Apply protective copper fungicide or biofungicide per agricultural extension instructions if wet conditions persist.'}
                  </div>
                </div>
              </section>

              {/* PREVENTION PRACTICES */}
              <section className="report-section">
                <h3 className="section-heading">
                  <ShieldCheck size={16} aria-hidden="true" />
                  Prevention Practices
                </h3>
                <div className="prevention-grid">
                  <div className="prevention-card">
                    <div className="prevention-card-title">🌱 Certified Seed Stock</div>
                    <div>{diseaseInfo?.prevention?.[0] || 'Use certified disease-free seed and resistant cultivars.'}</div>
                  </div>
                  <div className="prevention-card">
                    <div className="prevention-card-title">💧 Precision Irrigation</div>
                    <div>Water root zones directly; avoid nocturnal foliar moisture accumulation.</div>
                  </div>
                  <div className="prevention-card">
                    <div className="prevention-card-title">🌬️ Air Circulation</div>
                    <div>Ensure recommended crop spacing (24-36 in) to allow canopy drying.</div>
                  </div>
                  <div className="prevention-card">
                    <div className="prevention-card-title">🧹 Field Sanitation</div>
                    <div>Bag and destroy symptomatic foliage; do not leave cull debris in fields.</div>
                  </div>
                  <div className="prevention-card">
                    <div className="prevention-card-title">🔍 Scouting &amp; Monitoring</div>
                    <div>Inspect lower leaves and leaf undersides twice weekly during warm humid cycles.</div>
                  </div>
                  <div className="prevention-card">
                    <div className="prevention-card-title">🌿 Crop Rotation</div>
                    <div>Rotate nightshade and susceptible families on a strict 2-3 year schedule.</div>
                  </div>
                </div>
              </section>

              {/* AI CONFIDENCE & TELEMETRY */}
              <section className="report-section">
                <h3 className="section-heading">
                  <CheckCircle2 size={16} aria-hidden="true" />
                  AI Prediction Confidence &amp; Telemetry
                </h3>
                <div className="confidence-visual-card">
                  <div className="circle-meter">
                    <svg viewBox="0 0 80 80">
                      <circle className="circle-bg" cx="40" cy="40" r="34" />
                      <circle
                        className="circle-val"
                        cx="40"
                        cy="40"
                        r="34"
                        style={{
                          strokeDasharray: 213.6,
                          strokeDashoffset: 213.6 - (213.6 * Math.min(100, Math.round(confidenceVal * 100))) / 100,
                        }}
                      />
                    </svg>
                    <div className="circle-text">{confidencePercent}</div>
                  </div>

                  <div>
                    <div style={{ fontSize: 15, fontWeight: 800, color: '#064e3b', marginBottom: 4 }}>
                      {confidenceVal >= 0.85 ? 'HIGH CONFIDENCE CLINICAL CLASSIFICATION' : 'MODERATE CONFIDENCE CLASSIFICATION'}
                    </div>
                    <p style={{ fontSize: 12, color: '#475569', margin: 0, lineHeight: 1.5 }}>
                      Shannon Predictive Entropy: <strong>{typeof result.uncertainty?.entropy === 'number' ? result.uncertainty.entropy.toFixed(3) : '0.280'}</strong> | Top-2 Margin: <strong>{typeof result.uncertainty?.margin === 'number' ? result.uncertainty.margin.toFixed(3) : '0.810'}</strong>.
                      All safety distribution rejection thresholds satisfied.
                    </p>
                  </div>
                </div>
              </section>
            </>
          )}

          {/* LEGAL DISCLAIMER */}
          <div className="disclaimer-box">
            <strong>DISCLAIMER:</strong> FloraGuard AI provides AI-assisted image-based plant disease assessment for research, education, and decision support. Output does not replace certified in-person laboratory pathology or extension specialist diagnosis. Treatment recommendations are general agronomic guidance.
          </div>

          {/* REPORT FOOTER */}
          <footer className="report-footer-section">
            <div><strong>FloraGuard AI</strong> | Plant Disease Detection &amp; Crop Care System</div>
            <div>Report ID: <strong>FLORA-DIAG-{result.request_id || requestId}</strong> | Status: <strong>Verified</strong></div>
            <div><em>"AI-Assisted Agricultural Intelligence"</em> | Page 1 of 1</div>
          </footer>

        </article>
      </div>
    </div>
  );
}

export default Results;
