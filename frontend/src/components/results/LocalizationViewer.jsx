import { useEffect, useState } from 'react';
import { Crosshair, ImageOff } from 'lucide-react';
import { resolveApiUrl } from '../../services/api';

/**
 * Computes an approximate bounding box around peak Grad-CAM heatmap activations
 * (high red-channel / intensity pixels) purely as a client-side attention aid.
 * Never presented as ground-truth pixel-level disease segmentation.
 */
function extractAttentionBoxFromHeatmap(img) {
  const width = img.naturalWidth || 224;
  const height = img.naturalHeight || 224;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  ctx.drawImage(img, 0, 0, width, height);
  const { data } = ctx.getImageData(0, 0, width, height);

  let maxScore = 0;
  const scores = new Float32Array(width * height);
  for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    // Jet/hot colormap high activation correlates with high red relative to blue
    const score = Math.max(0, r * 0.75 + g * 0.25 - b * 0.45);
    scores[p] = score;
    if (score > maxScore) maxScore = score;
  }

  if (maxScore < 40) return null;
  const threshold = maxScore * 0.62;

  let minX = width;
  let minY = height;
  let maxX = 0;
  let maxY = 0;
  let count = 0;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (scores[y * width + x] >= threshold) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        count += 1;
      }
    }
  }

  if (count < 16 || maxX <= minX || maxY <= minY) return null;

  return {
    leftPct: (minX / width) * 100,
    topPct: (minY / height) * 100,
    widthPct: ((maxX - minX) / width) * 100,
    heightPct: ((maxY - minY) / height) * 100,
  };
}

export function LocalizationViewer({ originalImageUrl, explanation, isUncertain = false }) {
  const heatmapUrl = explanation?.heatmap_url ? resolveApiUrl(explanation.heatmap_url) : null;
  const overlayUrl = explanation?.overlay_url ? resolveApiUrl(explanation.overlay_url) : null;
  const baseImageUrl = overlayUrl || originalImageUrl;

  const [bbox, setBbox] = useState(null);
  const [statusNote, setStatusNote] = useState('');
  const [imageError, setImageError] = useState(false);

  useEffect(() => {
    setBbox(null);
    setStatusNote('');
    setImageError(false);

    if (!explanation?.available || !heatmapUrl || isUncertain) {
      return undefined;
    }

    let cancelled = false;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      if (cancelled) return;
      try {
        const computed = extractAttentionBoxFromHeatmap(img);
        setBbox(computed);
        if (!computed) {
          setStatusNote('Attention map is diffuse; no single peak attention region exceeded threshold.');
        }
      } catch {
        setStatusNote(
          'Showing Grad-CAM overlay directly. Client-side attention box derivation requires CORS-readable heatmap pixels.'
        );
      }
    };
    img.onerror = () => {
      if (!cancelled) {
        setStatusNote('Heatmap pixels could not be loaded for approximate attention bounding box.');
      }
    };
    img.src = heatmapUrl;

    return () => {
      cancelled = true;
    };
  }, [explanation?.available, heatmapUrl, isUncertain]);

  return (
    <section className="card" aria-label="Approximate AI attention localization">
      <div className="card-header">
        <div>
          <h3 className="card-title">
            <Crosshair size={17} aria-hidden="true" />
            Approximate AI Attention Region
          </h3>
          <p className="card-subtitle">
            Bounding indicator derived from peak Grad-CAM activation — not ground-truth segmentation.
          </p>
        </div>
      </div>

      {!explanation?.available || !baseImageUrl || isUncertain || imageError ? (
        <div className="xai-stage">
          <div style={{ textAlign: 'center', padding: 28, color: 'var(--text-secondary)' }}>
            <ImageOff size={26} style={{ marginBottom: 8, opacity: 0.7 }} aria-hidden="true" />
            <div style={{ fontSize: '0.9rem', fontWeight: 500 }}>
              Approximate attention localization is unavailable for this prediction.
            </div>
          </div>
        </div>
      ) : (
        <div className="xai-stage" style={{ padding: 8 }}>
          <div style={{ position: 'relative', display: 'inline-block', maxWidth: '100%' }}>
            <img
              src={baseImageUrl}
              alt="Leaf image with approximate AI attention region indicator"
              className="xai-image fade-in-image"
              onError={() => setImageError(true)}
            />
            {bbox ? (
              <div
                aria-label="Approximate AI attention region bounding box"
                style={{
                  position: 'absolute',
                  left: `${bbox.leftPct}%`,
                  top: `${bbox.topPct}%`,
                  width: `${bbox.widthPct}%`,
                  height: `${bbox.heightPct}%`,
                  border: '2px solid #4ADE80',
                  borderRadius: '6px',
                  boxShadow: '0 0 0 1px rgba(10, 15, 13, 0.75), inset 0 0 16px rgba(74, 222, 128, 0.16)',
                  pointerEvents: 'none',
                }}
              >
                <span
                  style={{
                    position: 'absolute',
                    top: -22,
                    left: 0,
                    background: 'rgba(10, 15, 13, 0.88)',
                    color: '#86EFAC',
                    border: '1px solid rgba(74, 222, 128, 0.45)',
                    borderRadius: '4px',
                    padding: '1px 6px',
                    fontSize: '0.7rem',
                    fontWeight: 600,
                    whiteSpace: 'nowrap',
                  }}
                >
                  Approximate AI Attention Region
                </span>
              </div>
            ) : null}
          </div>
        </div>
      )}

      <div className="xai-caption">
        <div>
          <strong>Labeling policy:</strong> Any box highlighted above represents an{' '}
          <strong>approximate AI attention region</strong> computed from peak Grad-CAM heatmap
          intensity. It is <strong>not</strong> a pixel-level disease segmentation mask.
        </div>
        {statusNote ? (
          <div style={{ marginTop: 6, color: 'var(--text-muted)', fontSize: '0.8rem' }}>
            {statusNote}
          </div>
        ) : null}
      </div>
    </section>
  );
}

export default LocalizationViewer;
