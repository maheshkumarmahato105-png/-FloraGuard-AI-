/**
 * Camera controller supporting desktop webcams and mobile front/rear cameras.
 * Defaults to rear ('environment') camera on mobile devices and allows switching
 * between available cameras or facing modes cleanly.
 */

export class CameraController {
  constructor(videoElement, canvasElement) {
    this.video = videoElement;
    this.canvas = canvasElement;
    this.stream = null;
    this.devices = [];
    this.currentDeviceIndex = -1;
    this.facingMode = "environment"; // Rear camera preferred by default on mobile
    this.isActive = false;
  }

  static isSupported() {
    return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }

  async refreshDevices() {
    if (!navigator.mediaDevices?.enumerateDevices) return [];
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      this.devices = all.filter((d) => d.kind === "videoinput");
      return this.devices;
    } catch {
      return [];
    }
  }

  async start({ facingMode = this.facingMode, deviceId = null } = {}) {
    if (!CameraController.isSupported()) {
      throw new Error(
        "Camera access is not supported in this browser or requires HTTPS / localhost. Use the Image Upload or Mobile Camera File button instead."
      );
    }

    this.stop();
    this.facingMode = facingMode;

    const videoConstraints = deviceId
      ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 960 } }
      : { facingMode: { ideal: facingMode }, width: { ideal: 1280 }, height: { ideal: 960 } };

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: false,
      });
    } catch (firstErr) {
      // Fallback to generic video stream if specific facingMode/deviceId fails on desktop webcam
      if (firstErr.name === "OverconstrainedError" || firstErr.name === "NotFoundError") {
        try {
          this.stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false,
          });
        } catch (fallbackErr) {
          throw this._formatCameraError(fallbackErr);
        }
      } else {
        throw this._formatCameraError(firstErr);
      }
    }

    this.video.srcObject = this.stream;
    this.video.setAttribute("playsinline", "true");
    this.video.muted = true;
    await this.video.play();
    this.isActive = true;

    await this.refreshDevices();
    const activeTrack = this.stream.getVideoTracks()[0];
    const activeDeviceId = activeTrack?.getSettings?.()?.deviceId;
    if (activeDeviceId && this.devices.length > 0) {
      this.currentDeviceIndex = this.devices.findIndex((d) => d.deviceId === activeDeviceId);
    }

    return {
      cameraCount: this.devices.length,
      label: activeTrack?.label || (this.facingMode === "environment" ? "Rear Camera" : "Front Camera"),
      facingMode: activeTrack?.getSettings?.()?.facingMode || this.facingMode,
    };
  }

  async switchCamera() {
    await this.refreshDevices();
    if (this.devices.length > 1) {
      const nextIndex = (this.currentDeviceIndex + 1) % this.devices.length;
      const nextDevice = this.devices[nextIndex];
      this.currentDeviceIndex = nextIndex;
      return this.start({ deviceId: nextDevice.deviceId });
    }
    // Toggle facingMode on mobile devices where enumerateDevices may hide labels/IDs
    const nextFacing = this.facingMode === "environment" ? "user" : "environment";
    return this.start({ facingMode: nextFacing });
  }

  async captureImage({ maxDimension = 1280, quality = 0.92 } = {}) {
    if (!this.isActive || !this.video.videoWidth || !this.video.videoHeight) {
      throw new Error("Camera is not ready yet. Please wait a moment and try capturing again.");
    }

    let width = this.video.videoWidth;
    let height = this.video.videoHeight;

    if (Math.max(width, height) > maxDimension) {
      const ratio = maxDimension / Math.max(width, height);
      width = Math.round(width * ratio);
      height = Math.round(height * ratio);
    }

    this.canvas.width = width;
    this.canvas.height = height;
    const ctx = this.canvas.getContext("2d");
    ctx.drawImage(this.video, 0, 0, width, height);

    const blob = await new Promise((resolve, reject) => {
      this.canvas.toBlob(
        (b) => (b ? resolve(b) : reject(new Error("Failed to capture image from camera."))),
        "image/jpeg",
        quality
      );
    });

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const file = new File([blob], `camera-leaf-${timestamp}.jpg`, { type: "image/jpeg" });
    const dataUrl = this.canvas.toDataURL("image/jpeg", quality);
    return { file, dataUrl, width, height };
  }

  stop() {
    if (this.stream) {
      for (const track of this.stream.getTracks()) {
        track.stop();
      }
      this.stream = null;
    }
    if (this.video) {
      this.video.srcObject = null;
    }
    this.isActive = false;
  }

  _formatCameraError(err) {
    if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
      return new Error(
        "Camera permission was denied. Please allow camera access in your browser settings and try again."
      );
    }
    if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
      return new Error(
        "No camera was found on this device. Please use the Image Upload tab instead."
      );
    }
    if (err.name === "NotReadableError" || err.name === "TrackStartError") {
      return new Error(
        "Your camera is currently in use by another application. Close the other app and try again."
      );
    }
    return new Error(err.message || "Unable to access device camera.");
  }
}

/**
 * Client-side optimization for large mobile camera photos (>1600px).
 * Shrinks oversized images smoothly before uploading to speed up mobile inference.
 */
export async function optimizeImageForUpload(file, maxDimension = 1400, quality = 0.9) {
  if (!file || !file.type.startsWith("image/")) {
    return file;
  }
  // Keep small/medium files untouched to preserve original bytes
  if (file.size <= 1.2 * 1024 * 1024) {
    return file;
  }

  return new Promise((resolve) => {
    const img = new Image();
    const objectUrl = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(objectUrl);
      const maxSide = Math.max(img.width, img.height);
      if (maxSide <= maxDimension) {
        resolve(file);
        return;
      }

      const scale = maxDimension / maxSide;
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(
        (blob) => {
          if (!blob) {
            resolve(file);
            return;
          }
          const baseName = (file.name || "leaf").replace(/\.[^.]+$/, "");
          resolve(new File([blob], `${baseName}.jpg`, { type: "image/jpeg" }));
        },
        "image/jpeg",
        quality
      );
    };
    img.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      resolve(file);
    };
    img.src = objectUrl;
  });
}
