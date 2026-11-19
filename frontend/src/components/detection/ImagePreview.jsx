import { useEffect, useRef, useState } from 'react';
import { RefreshCw, RotateCcw, Trash2 } from 'lucide-react';
import { formatBytes } from '../../utils/formatters';
import {
  buildAcceptAttribute,
  DEFAULT_ALLOWED_FORMATS,
  DEFAULT_MAX_UPLOAD_BYTES,
  validateImageFile,
} from '../../utils/validators';
import ErrorMessage from '../common/ErrorMessage';

export function ImagePreview({
  file,
  source = 'upload',
  maxBytes = DEFAULT_MAX_UPLOAD_BYTES,
  supportedFormats = DEFAULT_ALLOWED_FORMATS,
  onReplace,
  onRemove,
  onRetake,
  disabled = false,
}) {
  const replaceInputRef = useRef(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [dimensions, setDimensions] = useState(null);
  const [replaceError, setReplaceError] = useState(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      setDimensions(null);
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file]);

  if (!file) return null;

  const handleReplaceInput = (e) => {
    const nextFile = e.target.files?.[0];
    e.target.value = '';
    if (!nextFile) return;
    const check = validateImageFile(nextFile, { maxBytes, supportedFormats });
    if (!check.valid) {
      setReplaceError(check.error);
      return;
    }
    setReplaceError(null);
    onReplace?.(nextFile, 'upload');
  };

  return (
    <div>
      <input
        ref={replaceInputRef}
        type="file"
        accept={buildAcceptAttribute(supportedFormats)}
        onChange={handleReplaceInput}
        className="sr-only"
        aria-label="Replace selected plant leaf image"
        disabled={disabled}
      />

      <div className="preview-frame">
        {previewUrl ? (
          <img
            src={previewUrl}
            alt={`Selected plant leaf preview: ${file.name}`}
            className="preview-img fade-in-image"
            onLoad={(e) => {
              setDimensions({
                width: e.currentTarget.naturalWidth,
                height: e.currentTarget.naturalHeight,
              });
            }}
          />
        ) : null}
      </div>

      <div className="preview-meta">
        <div style={{ minWidth: 0 }}>
          <div
            style={{
              fontWeight: 600,
              color: 'var(--text-main)',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {file.name}
          </div>
          <div className="tabular" style={{ fontSize: '0.8rem', marginTop: 2 }}>
            {formatBytes(file.size)}
            {dimensions ? ` · ${dimensions.width} × ${dimensions.height}px` : ''}
            {source === 'camera' ? ' · Captured via camera' : ' · Uploaded file'}
          </div>
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {source === 'camera' && typeof onRetake === 'function' ? (
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={onRetake}
              disabled={disabled}
            >
              <RotateCcw size={14} aria-hidden="true" />
              Retake Photo
            </button>
          ) : (
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => replaceInputRef.current?.click()}
              disabled={disabled}
            >
              <RefreshCw size={14} aria-hidden="true" />
              Replace Image
            </button>
          )}
          <button
            type="button"
            className="btn btn-danger btn-sm"
            onClick={onRemove}
            disabled={disabled}
          >
            <Trash2 size={14} aria-hidden="true" />
            Remove
          </button>
        </div>
      </div>

      {replaceError ? (
        <div style={{ marginTop: 12 }}>
          <ErrorMessage title="Replacement rejected" message={replaceError} variant="warning" />
        </div>
      ) : null}
    </div>
  );
}

export default ImagePreview;
