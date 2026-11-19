import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { ThemeProvider } from '../context/ThemeContext';
import { ToastProvider } from '../context/ToastContext';
import { AppShell } from '../App';
import * as systemService from '../services/systemService';
import * as historyService from '../services/historyService';

vi.mock('../services/systemService');
vi.mock('../services/historyService');

function renderWithRoute(initialRoute = '/') {
  return render(
    <ThemeProvider>
      <ToastProvider>
        <MemoryRouter initialEntries={[initialRoute]}>
          <AppShell />
        </MemoryRouter>
      </ToastProvider>
    </ThemeProvider>
  );
}

describe('Application rendering, routing, and responsive shell', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    systemService.getHealth.mockResolvedValue({
      status: 'degraded',
      api_status: 'online',
      database_status: 'connected',
      model_ready: false,
      model_status_detail: 'Checkpoint not found.',
      inference_device: 'cuda',
      app_version: '1.0.0',
    });
    systemService.getSystemInfo.mockResolvedValue({
      app_name: 'PlantDiseaseAI',
      app_version: '1.0.0',
      model_name: 'AgriVision-CNN',
      num_classes: 38,
      device_type: 'cuda',
      checkpoint_available: false,
      supported_image_formats: ['JPEG', 'JPG', 'PNG', 'WEBP'],
      max_upload_size_bytes: 10485760,
      explanation_available: false,
      explanation_method: 'Grad-CAM',
      explanation_disclaimer: 'Approximate attention only.',
      image_retention_enabled: false,
    });
    historyService.fetchDashboardSummary.mockResolvedValue({
      totalScans: 0,
      healthyCount: 0,
      diseasedCount: 0,
      uncertainCount: 0,
      successTotal: 0,
      statsSampled: false,
      recentScans: [],
    });
    historyService.listHistory.mockResolvedValue({
      total: 0,
      limit: 10,
      offset: 0,
      items: [],
    });
  });

  it('renders Dashboard with empty real history state and both desktop & mobile navigation', async () => {
    renderWithRoute('/');

    expect(screen.getByText('Plant Disease Detection Platform')).toBeInTheDocument();
    expect(screen.getByLabelText('Primary sidebar navigation')).toBeInTheDocument();
    expect(screen.getByLabelText('Mobile bottom navigation')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('No plant scans recorded yet')).toBeInTheDocument();
    });
    expect(screen.getByText(/inference model checkpoint is not yet loaded/i)).toBeInTheDocument();
  });

  it('navigates between Dashboard, Detection, History, and Settings routes', async () => {
    const user = userEvent.setup();
    renderWithRoute('/');

    const detectionLinks = screen.getAllByRole('link', { name: /detection/i });
    await user.click(detectionLinks[0]);
    expect(await screen.findByRole('heading', { name: 'Plant Disease Detection' })).toBeInTheDocument();

    const historyLinks = screen.getAllByRole('link', { name: /history/i });
    await user.click(historyLinks[0]);
    expect(await screen.findByRole('heading', { name: 'Prediction History' })).toBeInTheDocument();

    const settingsLinks = screen.getAllByRole('link', { name: /settings/i });
    await user.click(settingsLinks[0]);
    expect(await screen.findByRole('heading', { name: 'Application Settings' })).toBeInTheDocument();
  });

  it('renders NotFound page for unknown routes', async () => {
    renderWithRoute('/non-existent-route');
    expect(await screen.findByRole('heading', { name: 'Page Not Found' })).toBeInTheDocument();
    await waitFor(() => {
      expect(systemService.getHealth).toHaveBeenCalled();
    });
  });

  it('shows backend offline alert when health check fails', async () => {
    systemService.getHealth.mockRejectedValue(new Error('Unable to reach backend'));
    renderWithRoute('/');

    expect(await screen.findByText('Backend connection offline')).toBeInTheDocument();
  });
});
