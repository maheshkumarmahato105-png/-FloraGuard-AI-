import { useCallback, useEffect, useRef, useState } from 'react';
import { Camera, CameraOff, RefreshCw, RotateCcw, SwitchCamera, Upload } from 'lucide-react';
import ErrorMessage from '../common/ErrorMessage';

function mapCameraError(err) {
  const name = err?.name || '';
  if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
    return {
      title: 'Camera permission denied',
      message: 'Browser access to your camera was blocked.',
      hint: 'Allow camera access in your browser site settings and click Retry, or switch to manual image upload.',
    };
  }
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError' || name === 'OverconstrainedError') {
    return {
      title: 'No compatible camera detected',
      message: 'No active webcam or mobile camera was found for the requested mode.',
      hint: 'Connect a camera, try switching camera facing mode, or upload a photo from your device instead.',
    };
  }
  if (name === 'NotReadableError' || name === 'TrackStartError') {
    return {
      title: 'Camera is currently in use',
      message: 'Another application or browser tab is holding exclusive access to your camera.',
      hint: 'Close other apps using the camera and try again, or upload a leaf image manually.',
    };
  }
  return {
    title: 'Unable to start camera',
    message: err?.message || 'Camera initialization failed in this browser context.',
    hint: 'Ensure you are using HTTPS or localhost, or switch to manual file upload.',
  };
}

/**
 * Opt-in Live Camera Capture component.
 * STRICT RULE: Never accesses navigator.mediaDevices until the user clicks "Start Camera".
 * Always stops all media tracks on close, capture, or unmount.
 */
export function CameraCapture({ onPhotoCaptured, onSwitchToUpload, disabled = false }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const [cameraState, setCameraState] = useState('idle'); // idle | starting | active | error
  const [facingMode, setFacingMode] = useState('environment'); // Rear camera preference for mobile
  const [devices, setDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');
  const [cameraError, setCameraError] = useState(null);

  const stopTracks = useCallback(() => {
    const stream = streamRef.current;
    if (stream) {
      stream.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch {
          // Ignore track stop errors
        }
      });
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  // Ensure tracks are always stopped when unmounting or navigating away
  useEffect(() => {
    return () => {
      stopTracks();
    };
  }, [stopTracks]);

  // Attach active stream once video element mounts
  useEffect(() => {
    if (cameraState === 'active' && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
      const playResult = videoRef.current.play?.();
      if (playResult && typeof playResult.catch === 'function') {
        playResult.catch(() => {});
      }
    }
  }, [cameraState]);

  const startCamera = useCallback(
    async ({ nextFacingMode = facingMode, nextDeviceId = selectedDeviceId } = {}) => {
      if (disabled) return;
      stopTracks();
      setCameraError(null);

      if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== 'function') {
        setCameraError({
          title: 'Camera API unavailable',
          message: 'This browser does not support live camera capture (MediaDevices API).',
          hint: 'Use a modern browser over HTTPS or localhost, or switch to manual image upload.',
        });
        setCameraState('error');
        return;
      }

      setCameraState('starting');

      const primaryVideoConstraint = nextDeviceId
        ? { deviceId: { exact: nextDeviceId }, width: { ideal: 1280 }, height: { ideal: 960 } }
        : { facingMode: { ideal: nextFacingMode }, width: { ideal: 1280 }, height: { ideal: 960 } };

      try {
        let stream;
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: primaryVideoConstraint,
            audio: false,
          });
        } catch (firstErr) {
          // Fallback to generic video constraint if ideal facingMode isn't supported on desktop webcams
          if (
            !nextDeviceId &&
            (firstErr?.name === 'OverconstrainedError' || firstErr?.name === 'ConstraintNotSatisfiedError')
          ) {
            stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
          } else {
            throw firstErr;
          }
        }

        streamRef.current = stream;
        setCameraState('active');

        if (typeof navigator.mediaDevices.enumerateDevices === 'function') {
          try {
            const allDevices = await navigator.mediaDevices.enumerateDevices();
            const videoInputs = allDevices.filter((d) => d.kind === 'videoinput');
            setDevices(videoInputs);
          } catch {
            // Optional enumeration failure is non-fatal
          }
        }
      } catch (err) {
        stopTracks();
        setCameraError(mapCameraError(err));
        setCameraState('error');
      }
    },
    [disabled, facingMode, selectedDeviceId, stopTracks]
  );

  const handleStopCamera = () => {
    stopTracks();
    setCameraState('idle');
    setCameraError(null);
  };

  const handleToggleFacingMode = () => {
    const toggled = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(toggled);
    setSelectedDeviceId('');
    if (cameraState === 'active') {
      startCamera({ nextFacingMode: toggled, nextDeviceId: '' });
    }
  };

  const handleDeviceChange = (e) => {
    const devId = e.target.value;
    setSelectedDeviceId(devId);
    if (cameraState === 'active') {
      startCamera({ nextFacingMode: facingMode, nextDeviceId: devId });
    }
  };

  const handleCapturePhoto = () => {
    const video = videoRef.current;
    if (!video || !streamRef.current) return;

    const width = video.videoWidth || 1280;
    const height = video.videoHeight || 720;
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, width, height);

    const finalizeBlob = (blob) => {
      if (!blob) return;
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const capturedFile = new File([blob], `camera-leaf-${timestamp}.jpg`, {
        type: 'image/jpeg',
      });
      stopTracks();
      setCameraState('idle');
      onPhotoCaptured?.(capturedFile, 'camera');
    };

    if (typeof canvas.toBlob === 'function') {
      canvas.toBlob(finalizeBlob, 'image/jpeg', 0.92);
    }
  };

  return (
    <div>
      {cameraState === 'idle' ? (
        <div className="empty-state" style={{ padding: '32px 20px' }}>
          <div className="dropzone-icon" aria-hidden="true">
            <Camera size={24} />
          </div>
          <h3 className="dropzone-title">Live Camera Capture</h3>
          <p className="dropzone-hint" style={{ maxWidth: '48ch', margin: '0 auto 18px' }}>
            Your camera stays off until you explicitly start it. Defaults to rear camera on mobile
            devices with front/rear switching supported.
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: 10 }}>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => startCamera()}
              disabled={disabled}
            >
              <Camera size={16} aria-hidden="true" />
              Start Camera
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={handleToggleFacingMode}
              disabled={disabled}
            >
              <SwitchCamera size={16} aria-hidden="true" />
              Mode: {facingMode === 'environment' ? 'Rear (Environment)' : 'Front (User)'}
            </button>
          </div>
        </div>
      ) : null}

      {cameraState === 'starting' ? (
        <div className="camera-viewport">
          <div style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>
            <span className="spinner" style={{ width: 26, height: 26 }} aria-hidden="true" />
            <p style={{ margin: '12px 0 0', fontSize: '0.9rem' }}>
              Requesting camera permission…
            </p>
          </div>
        </div>
      ) : null}

      {cameraState === 'active' ? (
        <div>
          <div className="camera-viewport">
            <video
              ref={videoRef}
              className="camera-video"
              autoPlay
              playsInline
              muted
              aria-label="Live camera preview of plant leaf"
            />
          </div>

          <div className="camera-controls">
            <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 8 }}>
              <button
                type="button"
                className="btn btn-primary"
                onClick={handleCapturePhoto}
                disabled={disabled}
              >
                <Camera size={16} aria-hidden="true" />
                Capture Leaf Photo
              </button>

              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={handleToggleFacingMode}
                disabled={disabled}
                aria-label="Switch between front and rear camera"
              >
                <SwitchCamera size={15} aria-hidden="true" />
                {facingMode === 'environment' ? 'Switch to Front' : 'Switch to Rear'}
              </button>

              {devices.length > 1 ? (
                <label className="sr-only" htmlFor="camera-device-select">
                  Select camera device
                </label>
              ) : null}
              {devices.length > 1 ? (
                <select
                  id="camera-device-select"
                  className="select"
                  style={{ width: 'auto', minWidth: 160, padding: '7px 10px', fontSize: '0.82rem' }}
                  value={selectedDeviceId}
                  onChange={handleDeviceChange}
                >
                  <option value="">Auto camera</option>
                  {devices.map((dev, index) => (
                    <option key={dev.deviceId || index} value={dev.deviceId}>
                      {dev.label || `Camera ${index + 1}`}
                    </option>
                  ))}
                </select>
              ) : null}
            </div>

            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={handleStopCamera}
            >
              <CameraOff size={15} aria-hidden="true" />
              Stop Camera
            </button>
          </div>
        </div>
      ) : null}

      {cameraState === 'error' && cameraError ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <ErrorMessage
            title={cameraError.title}
            message={cameraError.message}
            recoveryHint={cameraError.hint}
            variant="warning"
          />
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => startCamera()}
            >
              <RefreshCw size={14} aria-hidden="true" />
              Retry Camera
            </button>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={handleToggleFacingMode}
            >
              <RotateCcw size={14} aria-hidden="true" />
              Switch to {facingMode === 'environment' ? 'Front' : 'Rear'} Camera
            </button>
            {typeof onSwitchToUpload === 'function' ? (
              <button
                type="button"
                className="btn btn-primary btn-sm"
                onClick={onSwitchToUpload}
              >
                <Upload size={14} aria-hidden="true" />
                Upload Image Instead
              </button>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default CameraCapture;
