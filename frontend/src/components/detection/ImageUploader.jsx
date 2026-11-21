import { useRef, useState } from 'react';
import { ImagePlus, UploadCloud } from 'lucide-react';
import { formatBytes } from '../../utils/formatters';
import {
  buildAcceptAttribute,
  DEFAULT_ALLOWED_FORMATS,
  DEFAULT_MAX_UPLOAD_BYTES,
  validateImageFile,
} from '../../utils/validators';
import ErrorMessage from '../common/ErrorMessage';

export function ImageUploader({
  onFileSelected,
  maxBytes = DEFAULT_MAX_UPLOAD_BYTES,
  supportedFormats = DEFAULT_ALLOWED_FORMATS,
  disabled = false,
}) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [validationError, setValidationError] = useState(null);

  const acceptAttr = buildAcceptAttribute(supportedFormats);

  const handleCandidate = (file) => {
    if (!file) return;
    const check = validateImageFile(file, { maxBytes, supportedFormats });
    if (!check.valid) {
      setValidationError(check.error);
      return;
    }
    setValidationError(null);
    onFileSelected?.(file, 'upload');
  };

  const handleInputChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      handleCandidate(file);
    }
    e.target.value = '';
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
    if (disabled) return;
    const file = e.dataTransfer?.files?.[0];
    if (file) {
      handleCandidate(file);
    }
  };

  const openPicker = () => {
    if (!disabled) {
      inputRef.current?.click();
    }
  };

  return (
    <div>
      <input
        ref={inputRef}
        type="file"
        accept={acceptAttr}
        onChange={handleInputChange}
        className="sr-only"
        aria-label="Choose a plant leaf image file"
        disabled={disabled}
      />

      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-label="Upload plant leaf image by clicking or dragging and dropping a file"
        aria-disabled={disabled}
        className={`dropzone${dragging ? ' dragging' : ''}`}
        onClick={openPicker}
        onKeyDown={(e) => {
          if ((e.key === 'Enter' || e.key === ' ') && !disabled) {
            e.preventDefault();
            openPicker();
          }
        }}
        onDragEnter={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          setDragging(false);
        }}
        onDrop={handleDrop}
      >
        <div className="dropzone-icon" aria-hidden="true">
          <UploadCloud size={26} />
        </div>
        <h3 className="dropzone-title">
          {dragging ? 'Drop leaf image here' : 'Drag and drop a plant leaf photo'}
        </h3>
        <p className="dropzone-hint">
          Capture a single, clear, well-lit leaf against a clean background for optimal AgriVision-CNN diagnostic accuracy.
        </p>
        <div style={{ display: 'flex', justifyContent: 'center', gap: 6, margin: '12px 0 16px', flexWrap: 'wrap' }}>
          {(supportedFormats?.length ? supportedFormats : DEFAULT_ALLOWED_FORMATS).map((fmt) => (
            <span
              key={fmt}
              style={{
                fontSize: '0.72rem',
                fontWeight: 600,
                padding: '2px 8px',
                borderRadius: 'var(--radius-xs)',
                background: 'var(--bg-card-elevated)',
                border: '1px solid var(--border-subtle)',
                color: 'var(--text-secondary)',
                letterSpacing: '0.04em',
              }}
            >
              {fmt}
            </span>
          ))}
          <span
            style={{
              fontSize: '0.72rem',
              fontWeight: 600,
              padding: '2px 8px',
              borderRadius: 'var(--radius-xs)',
              background: 'var(--accent-subtle)',
              border: '1px solid var(--accent-border)',
              color: 'var(--accent-primary)',
            }}
          >
            Max {formatBytes(maxBytes)}
          </span>
        </div>
        <button
          type="button"
          className="btn btn-secondary btn-sm"
          disabled={disabled}
          onClick={(e) => {
            e.stopPropagation();
            openPicker();
          }}
        >
          <ImagePlus size={15} aria-hidden="true" />
          Browse Files
        </button>
      </div>

      {validationError ? (
        <div style={{ marginTop: 14 }}>
          <ErrorMessage
            title="Invalid image selection"
            message={validationError}
            recoveryHint={`Select a valid ${(supportedFormats || DEFAULT_ALLOWED_FORMATS).join(', ')} image under ${formatBytes(maxBytes)}.`}
            variant="warning"
          />
        </div>
      ) : null}
    </div>
  );
}

export default ImageUploader;
