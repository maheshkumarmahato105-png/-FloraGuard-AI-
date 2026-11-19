import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import CameraCapture from '../components/detection/CameraCapture';

describe('CameraCapture opt-in behavior, error states, and track cleanup', () => {
  const originalMediaDevices = navigator.mediaDevices;

  afterEach(() => {
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: originalMediaDevices,
    });
    vi.restoreAllMocks();
  });

  it('does NOT access the camera automatically until Start Camera is clicked', () => {
    const getUserMedia = vi.fn();
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia },
    });

    render(<CameraCapture onPhotoCaptured={vi.fn()} onSwitchToUpload={vi.fn()} />);

    expect(getUserMedia).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: /start camera/i })).toBeInTheDocument();
  });

  it('handles denied camera permissions and provides upload fallback', async () => {
    const user = userEvent.setup();
    const onSwitchToUpload = vi.fn();
    const deniedErr = new Error('Permission denied');
    deniedErr.name = 'NotAllowedError';

    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn().mockRejectedValue(deniedErr) },
    });

    render(<CameraCapture onPhotoCaptured={vi.fn()} onSwitchToUpload={onSwitchToUpload} />);
    await user.click(screen.getByRole('button', { name: /start camera/i }));

    expect(await screen.findByText('Camera permission denied')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /upload image instead/i }));
    expect(onSwitchToUpload).toHaveBeenCalledTimes(1);
  });

  it('handles missing camera hardware gracefully', async () => {
    const user = userEvent.setup();
    const notFoundErr = new Error('No camera');
    notFoundErr.name = 'NotFoundError';

    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn().mockRejectedValue(notFoundErr) },
    });

    render(<CameraCapture onPhotoCaptured={vi.fn()} onSwitchToUpload={vi.fn()} />);
    await user.click(screen.getByRole('button', { name: /start camera/i }));

    expect(await screen.findByText('No compatible camera detected')).toBeInTheDocument();
  });

  it('stops all active media tracks when unmounting or clicking Stop Camera', async () => {
    const user = userEvent.setup();
    const stopTrack = vi.fn();
    const mockStream = { getTracks: () => [{ stop: stopTrack }] };

    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: {
        getUserMedia: vi.fn().mockResolvedValue(mockStream),
        enumerateDevices: vi.fn().mockResolvedValue([]),
      },
    });

    const { unmount } = render(
      <CameraCapture onPhotoCaptured={vi.fn()} onSwitchToUpload={vi.fn()} />
    );

    await user.click(screen.getByRole('button', { name: /start camera/i }));
    expect(await screen.findByLabelText('Live camera preview of plant leaf')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /stop camera/i }));
    await waitFor(() => {
      expect(stopTrack).toHaveBeenCalledTimes(1);
    });

    unmount();
  });
});
