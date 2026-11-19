import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { ThemeProvider } from '../context/ThemeContext';
import { ToastProvider } from '../context/ToastContext';
import History from '../pages/History';
import Settings from '../pages/Settings';
import * as historyService from '../services/historyService';

vi.mock('../services/historyService');

const sampleRecords = [
  {
    request_id: 'req-hist-001',
    created_at: '2026-11-18T10:00:00Z',
    status: 'success',
    predicted_class: 'tomato__early_blight',
    is_healthy: false,
    confidence: 0.92,
    top_predictions: [
      {
        class_name: 'tomato__early_blight',
        display_name: 'Tomato — Early Blight',
        plant_name: 'Tomato',
        disease_name: 'Early Blight',
        is_healthy: false,
        confidence: 0.92,
      },
    ],
    entropy: 0.28,
    margin: 0.81,
    is_uncertain: false,
    rejection_reason: null,
    model_name: 'AgriVision-CNN',
    checkpoint_id: 'ckpt-1',
    image_id: 'img-1',
    explanation_available: false,
  },
  {
    request_id: 'req-hist-002',
    created_at: '2026-11-18T11:30:00Z',
    status: 'uncertain',
    predicted_class: null,
    is_healthy: null,
    confidence: null,
    top_predictions: [],
    entropy: 2.31,
    margin: 0.03,
    is_uncertain: true,
    rejection_reason: 'Low confidence',
    model_name: 'AgriVision-CNN',
    checkpoint_id: 'ckpt-1',
    image_id: 'img-2',
    explanation_available: false,
  },
];

describe('History loading, empty state, and confirmed deletion (mocked API)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('loads records and deletes an entry only after user confirmation', async () => {
    const user = userEvent.setup();
    historyService.listHistory
      .mockResolvedValueOnce({ total: 2, limit: 10, offset: 0, items: sampleRecords })
      .mockResolvedValueOnce({ total: 1, limit: 10, offset: 0, items: [sampleRecords[1]] });
    historyService.deleteHistoryRecord.mockResolvedValue({ request_id: 'req-hist-001', deleted: true });

    render(
      <ThemeProvider>
        <ToastProvider>
          <MemoryRouter>
            <History />
          </MemoryRouter>
        </ToastProvider>
      </ThemeProvider>
    );

    expect(await screen.findAllByText('Tomato — Early Blight')).toHaveLength(2); // Desktop table + mobile card

    const deleteButtons = screen.getAllByRole('button', { name: /delete prediction record req-hist-001/i });
    await user.click(deleteButtons[0]);

    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByText('Delete prediction record?')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Delete Record' }));

    await waitFor(() => {
      expect(historyService.deleteHistoryRecord).toHaveBeenCalledWith('req-hist-001');
    });
  });

  it('renders clean empty state when backend returns zero history records', async () => {
    historyService.listHistory.mockResolvedValue({ total: 0, limit: 10, offset: 0, items: [] });

    render(
      <ThemeProvider>
        <ToastProvider>
          <MemoryRouter>
            <History />
          </MemoryRouter>
        </ToastProvider>
      </ThemeProvider>
    );

    expect(await screen.findByText('No prediction history yet')).toBeInTheDocument();
  });

  it('renders error state with retry button when history request fails', async () => {
    historyService.listHistory.mockRejectedValue(new Error('Backend unreachable'));

    render(
      <ThemeProvider>
        <ToastProvider>
          <MemoryRouter>
            <History />
          </MemoryRouter>
        </ToastProvider>
      </ThemeProvider>
    );

    expect(await screen.findByText('Unable to load history')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /try again/i })).toBeInTheDocument();
  });
});

describe('Settings page metadata and theme toggle', () => {
  it('displays backend model metadata and switches dark/light themes', async () => {
    const user = userEvent.setup();
    const healthState = {
      health: {
        status: 'degraded',
        api_status: 'online',
        database_status: 'connected',
        model_ready: false,
        model_status_detail: 'Trained checkpoint not found.',
        inference_device: 'cuda',
        app_version: '1.0.0',
      },
      systemInfo: {
        app_version: '1.0.0',
        model_name: 'AgriVision-CNN',
        num_classes: 38,
        device_type: 'cuda',
        supported_image_formats: ['JPEG', 'JPG', 'PNG', 'WEBP'],
        max_upload_size_bytes: 10485760,
        explanation_method: 'Grad-CAM',
      },
      loading: false,
      error: null,
      lastCheckedAt: '2026-11-18T12:00:00Z',
      refresh: vi.fn(),
    };

    render(
      <ThemeProvider>
        <ToastProvider>
          <MemoryRouter>
            <Settings healthState={healthState} />
          </MemoryRouter>
        </ToastProvider>
      </ThemeProvider>
    );

    expect(screen.getByText('AgriVision-CNN')).toBeInTheDocument();
    expect(screen.getByText('38')).toBeInTheDocument();
    expect(screen.getByText('JPEG, JPG, PNG, WEBP')).toBeInTheDocument();
    expect(screen.getByText('10 MB')).toBeInTheDocument();

    const lightBtn = screen.getByRole('radio', { name: /light mode/i });
    await user.click(lightBtn);
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
  });
});
