import { useEffect, useState } from 'react';
import { Eye, ImageOff, Maximize2, X } from 'lucide-react';
import { resolveApiUrl } from '../../services/api';

const DEFAULT_DISCLAIMER =
  'Highlighted regions indicate approximate AgriVision-CNN model attention (Grad-CAM) and are NOT a verified ground-truth disease segmentation mask.';

export function GradCAMViewer({
  originalImageUrl,
  explanation,
  isUncertain = false,
}) {
  const heatmapUrl = explanation?.heatmap_url ? resolveApiUrl(explanation.heatmap_url) : null;
  const overlayUrl = explanation?.overlay_url ? resolveApiUrl(explanation.overlay_url) : null;
  const explanationAvailable = Boolean(explanation?.available && (heatmapUrl || overlayUrl));

  const defaultMode = overlayUrl ? 'overlay' : heatmapUrl ? 'heatmap' : 'original';
  const [activeMode, setActiveMode] = useState(defaultMode);
  const [failedUrls, setFailedUrls] = useState({});
  const [zoomOpen, setZoomOpen] = useState(false);

  useEffect(() => {
    setActiveMode(overlayUrl ? 'overlay' : heatmapUrl ? 'heatmap' : 'original');
  }, [overlayUrl, heatmapUrl, originalImageUrl]);

  useEffect(() => {
    if (!zoomOpen) return undefined;
    const onKeyDown = (e) => {
      if (e.key === 'Escape') setZoomOpen(false);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [zoomOpen]);

  const modes = [
    {
      id: 'original',
      label: 'Original',
      url: originalImageUrl,
      caption: 'Original leaf photograph submitted for analysis.',
    },
    {
      id: 'overlay',
      label: 'Grad-CAM Overlay',
      url: overlayUrl,
      caption:
        'Grad-CAM attention overlay blended onto the preprocessed leaf image. Warmer areas received stronger model activation.',
    },
    {
      id: 'heatmap',
      label: 'Grad-CAM Heatmap',
      url: heatmapUrl,
      caption:
        'Raw Grad-CAM spatial attention map from AgriVision-CNN. Brighter regions contributed more strongly to the predicted class.',
    },
  ];

  const current = modes.find((m) => m.id === activeMode) || modes[0];
  const currentUrl = current?.url && !failedUrls[current.url] ? current.url : null;

  return (
    <section className="card" aria-label="Grad-CAM Explainable AI viewer">
      <div className="card-header" style={{ flexWrap: 'wrap' }}>
        <div>
          <h3 className="card-title">
            <Eye size={17} aria-hidden="true" />
            Grad-CAM Explainability
          </h3>
          <p className="card-subtitle">
            Inspect model attention maps alongside the original leaf photograph.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <div className="tabs" role="tablist" aria-label="Grad-CAM visualization modes">
            {modes.map((mode) => (
              <button
                key={mode.id}
                type="button"
                role="tab"
                aria-selected={activeMode === mode.id}
                className={`tab-btn${activeMode === mode.id ? ' active' : ''}`}
                disabled={!mode.url}
                onClick={() => setActiveMode(mode.id)}
              >
                {mode.label}
              </button>
            ))}
          </div>

          {currentUrl ? (
            <button
              type="button"
              className="btn btn-secondary btn-icon"
              onClick={() => setZoomOpen(true)}
              aria-label={`Zoom ${current.label} visualization`}
              title="Zoom image"
            >
              <Maximize2 size={15} aria-hidden="true" />
            </button>
          ) : null}
        </div>
      </div>

      {!explanationAvailable && !isUncertain ? (
        <div className="alert alert-info" style={{ marginBottom: 14 }}>
          <ImageOff size={18} className="alert-icon" aria-hidden="true" />
          <div className="alert-body">
            <p className="alert-title">Grad-CAM explanation unavailable</p>
            <p className="alert-text">
              {explanation?.requested
                ? 'The prediction succeeded, but Grad-CAM explanation artifacts were not generated or could not be retrieved.'
                : 'Grad-CAM visualization was not requested for this scan.'}
            </p>
          </div>
        </div>
      ) : null}

      <div className="xai-stage">
        {currentUrl ? (
          <img
            src={currentUrl}
            alt={`${current.label} view of plant leaf analysis`}
            className="xai-image fade-in-image"
            onError={() => {
              setFailedUrls((prev) => ({ ...prev, [currentUrl]: true }));
            }}
          />
        ) : (
          <div style={{ textAlign: 'center', padding: 28, color: 'var(--text-secondary)' }}>
            <ImageOff size={26} style={{ marginBottom: 8, opacity: 0.7 }} aria-hidden="true" />
            <div style={{ fontSize: '0.9rem', fontWeight: 500 }}>
              {activeMode === 'original'
                ? 'Original uploaded image was not retained by the backend.'
                : 'Explanation image is unavailable for this record.'}
            </div>
          </div>
        )}
      </div>

      <div className="xai-caption">
        <div>{current.caption}</div>
        <div style={{ marginTop: 6, color: 'var(--text-muted)', fontSize: '0.8rem' }}>
          <strong>Scientific note:</strong> {explanation?.note || DEFAULT_DISCLAIMER}
        </div>
      </div>

      {zoomOpen && currentUrl ? (
        <div
          className="modal-backdrop"
          role="dialog"
          aria-modal="true"
          aria-label={`Zoomed ${current.label} image`}
          onClick={(e) => {
            if (e.target === e.currentTarget) setZoomOpen(false);
          }}
        >
          <div className="modal-panel modal-lg">
            <div className="modal-header">
              <h3 className="modal-title">{current.label}</h3>
              <button
                type="button"
                className="btn btn-ghost btn-icon"
                onClick={() => setZoomOpen(false)}
                aria-label="Close zoomed image"
              >
                <X size={18} />
              </button>
            </div>
            <div
              style={{
                background: 'var(--bg-primary)',
                borderRadius: 'var(--radius-md)',
                padding: 8,
                textAlign: 'center',
              }}
            >
              <img
                src={currentUrl}
                alt={`Zoomed ${current.label} view`}
                style={{ maxHeight: '72vh', width: 'auto', margin: '0 auto', display: 'block' }}
              />
            </div>
            <p className="xai-caption" style={{ marginBottom: 0 }}>
              {explanation?.note || DEFAULT_DISCLAIMER}
            </p>
          </div>
        </div>
      ) : null}
    </section>
  );
}

export default GradCAMViewer;
