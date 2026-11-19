import { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { AlertTriangle, Camera, Eye, ScanSearch, Upload } from 'lucide-react';
import PageContainer from '../components/layout/PageContainer';
import ImageUploader from '../components/detection/ImageUploader';
import CameraCapture from '../components/detection/CameraCapture';
import ImagePreview from '../components/detection/ImagePreview';
import DetectionProgress from '../components/detection/DetectionProgress';
import ErrorMessage from '../components/common/ErrorMessage';
import { usePrediction } from '../hooks/usePrediction';
import { useToast } from '../context/ToastContext';
import { DEFAULT_ALLOWED_FORMATS, DEFAULT_MAX_UPLOAD_BYTES } from '../utils/validators';

export function Detection({ healthState }) {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const { addToast } = useToast();
  const { health, systemInfo, error: healthError, refresh: refreshHealth } = healthState;

  const initialMode = searchParams.get('mode') === 'camera' ? 'camera' : 'upload';
  const [inputMode, setInputMode] = useState(initialMode);
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileSource, setFileSource] = useState('upload');
  const [generateExplanation, setGenerateExplanation] = useState(true);

  const {
    stage,
    uploadProgress,
    isSubmitting,
    error: predictionError,
    predict,
    cancel,
    reset: resetPrediction,
  } = usePrediction();

  useEffect(() => {
    const paramMode = searchParams.get('mode');
    if (paramMode === 'camera' || paramMode === 'upload') {
      setInputMode(paramMode);
    }
  }, [searchParams]);

  const maxBytes = systemInfo?.max_upload_size_bytes || DEFAULT_MAX_UPLOAD_BYTES;
  const supportedFormats = systemInfo?.supported_image_formats || DEFAULT_ALLOWED_FORMATS;

  const handleModeChange = (nextMode) => {
    setInputMode(nextMode);
    setSearchParams(nextMode === 'camera' ? { mode: 'camera' } : {});
  };

  const handleFileSelected = (file, source) => {
    resetPrediction();
    setSelectedFile(file);
    setFileSource(source);
  };

  const handleRemoveFile = () => {
    resetPrediction();
    setSelectedFile(null);
  };

  const handleRetake = () => {
    resetPrediction();
    setSelectedFile(null);
    handleModeChange('camera');
  };

  const handleAnalyze = async () => {
    if (!selectedFile || isSubmitting) return;

    const enriched = await predict(selectedFile, { generateExplanation });
    if (!enriched?.response?.request_id) {
      return;
    }

    const isUncertain = enriched.response.status === 'uncertain';
    addToast({
      type: isUncertain ? 'warning' : 'success',
      title: isUncertain ? 'Uncertain prediction returned' : 'Analysis complete',
      message: isUncertain
        ? 'The model rejected this sample due to uncertainty thresholds.'
        : 'Redirecting to detailed prediction and explainability results.',
    });

    navigate(`/results/${encodeURIComponent(enriched.response.request_id)}`, {
      state: {
        predictionPayload: enriched.response,
        previewDataUrl: enriched.previewDataUrl,
        originalFileName: enriched.originalFileName,
      },
    });
  };

  return (
    <PageContainer
      title="Plant Disease Detection"
      subtitle="Upload a plant leaf photograph or use opt-in live camera capture to run AgriVision-CNN inference."
    >
      {healthError ? (
        <div style={{ marginBottom: 20 }}>
          <ErrorMessage
            title="Backend offline"
            message={healthError}
            recoveryHint="Start the Phase 7 FastAPI backend before running an analysis."
            onRetry={refreshHealth}
            retryLabel="Check connection"
          />
        </div>
      ) : null}

      {health && !health.model_ready ? (
        <div className="alert alert-warning" role="status" style={{ marginBottom: 20 }}>
          <AlertTriangle size={20} className="alert-icon" aria-hidden="true" />
          <div className="alert-body">
            <p className="alert-title">Trained model checkpoint unavailable on backend</p>
            <p className="alert-text">
              {health.model_status_detail} Submitting an image now will return an explicit HTTP 503
              error from the API rather than a fabricated prediction.
            </p>
          </div>
        </div>
      ) : null}

      <div className="two-col-grid">
        <section className="card" aria-label="Leaf image input">
          <div className="card-header" style={{ flexWrap: 'wrap' }}>
            <div>
              <h2 className="card-title">1. Provide Plant Leaf Image</h2>
              <p className="card-subtitle">Choose file upload or opt-in live camera capture.</p>
            </div>

            <div className="tabs" role="tablist" aria-label="Image input method">
              <button
                type="button"
                role="tab"
                aria-selected={inputMode === 'upload'}
                className={`tab-btn${inputMode === 'upload' ? ' active' : ''}`}
                disabled={isSubmitting}
                onClick={() => handleModeChange('upload')}
              >
                <Upload size={15} aria-hidden="true" />
                Upload File
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={inputMode === 'camera'}
                className={`tab-btn${inputMode === 'camera' ? ' active' : ''}`}
                disabled={isSubmitting}
                onClick={() => handleModeChange('camera')}
              >
                <Camera size={15} aria-hidden="true" />
                Live Camera
              </button>
            </div>
          </div>

          {selectedFile ? (
            <ImagePreview
              file={selectedFile}
              source={fileSource}
              maxBytes={maxBytes}
              supportedFormats={supportedFormats}
              onReplace={handleFileSelected}
              onRemove={handleRemoveFile}
              onRetake={handleRetake}
              disabled={isSubmitting}
            />
          ) : inputMode === 'upload' ? (
            <ImageUploader
              onFileSelected={handleFileSelected}
              maxBytes={maxBytes}
              supportedFormats={supportedFormats}
              disabled={isSubmitting}
            />
          ) : (
            <CameraCapture
              onPhotoCaptured={handleFileSelected}
              onSwitchToUpload={() => handleModeChange('upload')}
              disabled={isSubmitting}
            />
          )}

          <DetectionProgress
            stage={stage}
            uploadProgress={uploadProgress}
            onCancel={cancel}
          />

          {predictionError ? (
            <div style={{ marginTop: 16 }}>
              <ErrorMessage
                title={
                  predictionError.status === 503
                    ? 'AI model unavailable (HTTP 503)'
                    : predictionError.isTimeout
                      ? 'Prediction request timed out'
                      : predictionError.isNetworkError
                        ? 'Backend unreachable'
                        : 'Prediction failed'
                }
                message={predictionError.message}
                recoveryHint={
                  predictionError.status === 503
                    ? 'Ensure a trained AgriVision-CNN checkpoint is available in the backend configuration, then retry.'
                    : 'Verify your image format and backend availability before trying again.'
                }
                onRetry={selectedFile ? handleAnalyze : undefined}
                retryLabel="Retry analysis"
              />
            </div>
          ) : null}
        </section>

        <section className="card" aria-label="Analysis options and submission">
          <div className="card-header">
            <div>
              <h2 className="card-title">2. Analysis Options</h2>
              <p className="card-subtitle">Configure explainability and run live inference.</p>
            </div>
          </div>

          <div
            style={{
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-md)',
              padding: 14,
              marginBottom: 18,
            }}
          >
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={generateExplanation}
                onChange={(e) => setGenerateExplanation(e.target.checked)}
                disabled={isSubmitting}
              />
              <span>
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontWeight: 600 }}>
                  <Eye size={15} style={{ color: 'var(--accent-primary)' }} aria-hidden="true" />
                  Generate Grad-CAM Attention Map
                </span>
                <span
                  style={{
                    display: 'block',
                    fontSize: '0.82rem',
                    color: 'var(--text-secondary)',
                    marginTop: 4,
                  }}
                >
                  Requests heatmap and overlay artifacts showing approximate model attention regions
                  when a prediction passes uncertainty checks.
                </span>
              </span>
            </label>
          </div>

          <button
            type="button"
            className="btn btn-primary"
            style={{ width: '100%', padding: '12px 18px' }}
            disabled={!selectedFile || isSubmitting}
            onClick={handleAnalyze}
          >
            <ScanSearch size={18} aria-hidden="true" />
            {isSubmitting
              ? stage === 'uploading'
                ? `Uploading (${uploadProgress}%)…`
                : 'Analyzing Leaf…'
              : 'Analyze Plant Leaf'}
          </button>

          <div style={{ marginTop: 20 }}>
            <h3 style={{ margin: '0 0 8px', fontSize: '0.88rem', fontWeight: 600 }}>
              Image Capture Guidelines
            </h3>
            <ul className="guidance-list" style={{ marginTop: 0 }}>
              <li>Center a single plant leaf in sharp focus with even natural lighting.</li>
              <li>Avoid blurry motion shots or extreme shadows across lesions.</li>
              <li>Low-confidence or out-of-distribution images are safely flagged as uncertain.</li>
            </ul>
          </div>
        </section>
      </div>
    </PageContainer>
  );
}

export default Detection;
