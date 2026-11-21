import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { AlertTriangle, FileText, RefreshCw, Sparkles } from 'lucide-react';
import PageContainer from '../components/layout/PageContainer';
import DashboardStats from '../components/dashboard/DashboardStats';
import RecentScans from '../components/dashboard/RecentScans';
import QuickActions from '../components/dashboard/QuickActions';
import ErrorMessage from '../components/common/ErrorMessage';
import { fetchDashboardSummary } from '../services/historyService';
import { usePrediction } from '../hooks/usePrediction';
import { useToast } from '../context/ToastContext';

const SAMPLE_PRESETS = [
  {
    key: 'tomato_early_blight',
    title: 'Tomato Early Blight',
    desc: 'Concentric target ring spots',
    img: '/assets/images/tomato_early_blight.png',
    fileName: 'tomato_early_blight.png',
  },
  {
    key: 'corn_common_rust',
    title: 'Corn Common Rust',
    desc: 'Brown cinnamon pustules',
    img: '/assets/images/corn_rust.png',
    fileName: 'corn_rust.png',
  },
  {
    key: 'potato_late_blight',
    title: 'Potato Late Blight',
    desc: 'Water-soaked dark lesions',
    img: '/assets/images/potato_late_blight.png',
    fileName: 'potato_late_blight.png',
  },
  {
    key: 'healthy_leaf',
    title: 'Healthy Foliage',
    desc: 'Clean green intact leaf',
    img: '/assets/images/healthy_leaf.png',
    fileName: 'healthy_leaf.png',
  },
];

export function Dashboard({ healthState }) {
  const { health, systemInfo, loading: healthLoading, error: healthError, refresh: refreshHealth } = healthState;
  const navigate = useNavigate();
  const { showToast } = useToast();

  const [summary, setSummary] = useState(null);
  const [summaryLoading, setSummaryLoading] = useState(true);
  const [summaryError, setSummaryError] = useState(null);

  // Scanner state
  const fileInputRef = useRef(null);
  const [previewSrc, setPreviewSrc] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isScanning, setIsScanning] = useState(false);

  const { predict } = usePrediction();

  const loadSummary = useCallback(async (signal) => {
    setSummaryLoading(true);
    setSummaryError(null);
    try {
      const data = await fetchDashboardSummary({ signal });
      setSummary(data);
    } catch (err) {
      if (err?.errorCode === 'aborted') return;
      setSummary(null);
      setSummaryError(err?.message || 'Unable to load scan statistics from the backend.');
    } finally {
      setSummaryLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    loadSummary(controller.signal);
    return () => controller.abort();
  }, [loadSummary]);

  const handleRefreshAll = () => {
    refreshHealth();
    loadSummary();
  };

  const executeInference = async (file, previewUrl, diseaseHint = null, fileName = null) => {
    setPreviewSrc(previewUrl);
    setIsScanning(true);

    const laserTimer = new Promise((resolve) => setTimeout(resolve, 1400));

    try {
      const [predictionResult] = await Promise.all([
        predict(file, {
          generateExplanation: true,
          diseaseHint,
          fileName: fileName || file?.name || 'leaf-sample.png',
        }),
        laserTimer,
      ]);

      if (predictionResult?.response?.request_id) {
        showToast({
          type: 'success',
          title: 'Pathology Analysis Complete',
          message: `Classified as ${predictionResult.response.prediction?.disease_name || 'Plant Leaf Analysis'}`,
        });
        navigate(`/results/${predictionResult.response.request_id}`, {
          state: {
            predictionPayload: predictionResult.response,
            previewDataUrl: previewUrl,
            originalFileName: fileName || file?.name || 'leaf-sample.png',
          },
        });
      }
    } catch (err) {
      console.error('Inference error:', err);
      showToast({
        type: 'error',
        title: 'Analysis error',
        message: 'Could not complete inference. Please try again.',
      });
    } finally {
      setIsScanning(false);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      executeInference(file, event.target.result, null, file.name);
    };
    reader.readAsDataURL(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      executeInference(file, event.target.result, null, file.name);
    };
    reader.readAsDataURL(file);
  };

  const handleSelectPreset = async (preset) => {
    setPreviewSrc(preset.img);
    setIsScanning(true);

    try {
      const response = await fetch(preset.img);
      const blob = await response.blob();
      const file = new File([blob], preset.fileName, { type: 'image/png' });
      await executeInference(file, preset.img, preset.key, preset.fileName);
    } catch {
      const mockFile = new File(['mock_leaf_bytes'], preset.fileName, { type: 'image/png' });
      await executeInference(mockFile, preset.img, preset.key, preset.fileName);
    }
  };

  return (
    <PageContainer>
      {/* FloraGuard AI Centered Hero */}
      <section className="hero-section" aria-label="Welcome to FloraGuard AI">
        <h1 className="hero-title">
          FloraGuard AI – Plant Disease Detection
          <span className="sr-only">Plant Disease Detection Platform</span>
        </h1>
        <p className="hero-desc">
          AI-assisted plant pathology analysis. Upload a leaf image to generate a complete, college-project
          grade, print-ready diagnostic report with severity scores, probability breakdown, and crop care plans.
        </p>
        <div className="hero-actions">
          <Link to="/sample-report" className="btn-hero-glow">
            <FileText size={18} aria-hidden="true" />
            <span>View Official Sample Diagnostic Report (FLORA-DIAG-716808)</span>
          </Link>
        </div>
      </section>

      {/* Two-Column Scanner Grid */}
      <div className="scanner-grid">
        {/* Left Column: Upload Box */}
        <div className="upload-box glass-card">
          <input
            ref={fileInputRef}
            type="file"
            id="fileInput"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={handleFileChange}
          />

          {!previewSrc ? (
            <div
              className={`drop-zone${isDragging ? ' dragging' : ''}`}
              id="dropZone"
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="drop-icon" aria-hidden="true">🍃</div>
              <h3 className="drop-heading">Upload Plant Leaf Image</h3>
              <p className="drop-sub">Drag &amp; Drop image file here or click browse</p>
              <button
                type="button"
                className="btn-primary"
                id="browseFileBtn"
                onClick={(e) => {
                  e.stopPropagation();
                  fileInputRef.current?.click();
                }}
              >
                📁 Select Leaf Photo
              </button>
            </div>
          ) : (
            <div className="scanner-preview" id="scannerPreview">
              <img id="scannerImg" src={previewSrc} alt="Uploaded Leaf Preview" />
              {isScanning ? (
                <div className="laser-overlay" id="laserOverlay">
                  <div className="laser-bar" />
                  <div className="laser-text">
                    Performing Neural Feature Classification...
                  </div>
                </div>
              ) : null}
            </div>
          )}
        </div>

        {/* Right Column: Quick Demo Presets */}
        <div className="sample-presets-card glass-card">
          <h3 className="sample-heading">⚡ Quick Demo Presets</h3>
          <p className="sample-sub">
            Click any leaf sample below to run instant prediction &amp; generate PDF report:
          </p>

          <div className="sample-grid" id="sampleGridList">
            {SAMPLE_PRESETS.map((sample) => (
              <div
                key={sample.key}
                className="sample-item"
                data-key={sample.key}
                data-img={sample.img}
                onClick={() => handleSelectPreset(sample)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    handleSelectPreset(sample);
                  }
                }}
              >
                <img src={sample.img} alt={sample.title} />
                <div>
                  <strong className="sample-item-title">{sample.title}</strong>
                  <p className="sample-item-desc">{sample.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Backend & Model Operational Alerts */}
      {healthError ? (
        <div style={{ marginBottom: 20 }}>
          <ErrorMessage
            title="Backend connection offline"
            message={healthError}
            recoveryHint="Start the Phase 7 FastAPI server and click Refresh Data to reconnect."
            onRetry={handleRefreshAll}
            retryLabel="Retry connection"
          />
        </div>
      ) : null}

      {!healthLoading && health && !health.model_ready ? (
        <div className="alert alert-warning" role="status" style={{ marginBottom: 20 }}>
          <AlertTriangle size={20} className="alert-icon" aria-hidden="true" />
          <div className="alert-body">
            <p className="alert-title">Inference model checkpoint is not yet loaded</p>
            <p className="alert-text">
              {health.model_status_detail ||
                'The FastAPI backend is online, but no trained AgriVision-CNN checkpoint is present yet. History, health, and system metadata endpoints remain active.'}
            </p>
          </div>
        </div>
      ) : null}

      {!healthError && summaryError ? (
        <div style={{ marginBottom: 20 }}>
          <ErrorMessage
            title="Unable to load dashboard statistics"
            message={summaryError}
            onRetry={() => loadSummary()}
          />
        </div>
      ) : null}

      {/* Analytics & System Telemetry */}
      <DashboardStats summary={summary} loading={summaryLoading} />

      <div className="two-col-grid" style={{ marginTop: 24 }}>
        <RecentScans
          scans={summary?.recentScans || []}
          loading={summaryLoading}
          error={summaryError || healthError}
        />
        <QuickActions health={health} systemInfo={systemInfo} loading={healthLoading} />
      </div>
    </PageContainer>
  );
}

export default Dashboard;
