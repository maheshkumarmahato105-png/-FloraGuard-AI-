import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { ThemeProvider } from '../context/ThemeContext';
import { ToastProvider } from '../context/ToastContext';
import Detection from '../pages/Detection';
import Results from '../pages/Results';
import * as predictionService from '../services/predictionService';
import { ApiError } from '../services/api';

vi.mock('../services/predictionService', async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    submitPrediction: vi.fn(),
  };
});

const readyHealthState = {
  health: {
    status: 'healthy',
    api_status: 'online',
    database_status: 'connected',
    model_ready: true,
    model_status_detail: 'Model loaded.',
    inference_device: 'cuda',
    app_version: '1.0.0',
  },
  systemInfo: {
    model_name: 'AgriVision-CNN',
    supported_image_formats: ['JPEG', 'JPG', 'PNG', 'WEBP'],
    max_upload_size_bytes: 10 * 1024 * 1024,
  },
  loading: false,
  error: null,
  refresh: vi.fn(),
};

function renderDetection() {
  return render(
    <ThemeProvider>
      <ToastProvider>
        <MemoryRouter initialEntries={['/detection']}>
          <Routes>
            <Route path="/detection" element={<Detection healthState={readyHealthState} />} />
            <Route path="/results/:requestId" element={<Results />} />
          </Routes>
        </MemoryRouter>
      </ToastProvider>
    </ThemeProvider>
  );
}

describe('Detection upload, preview, and prediction submission (mocked API)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('rejects invalid image files and previews valid images with remove option', async () => {
    const user = userEvent.setup({ applyAccept: false });
    renderDetection();

    const fileInput = screen.getByLabelText('Choose a plant leaf image file');

    // Invalid PDF file
    const badFile = new File(['%PDF'], 'notes.pdf', { type: 'application/pdf' });
    await user.upload(fileInput, badFile);
    expect(screen.getByText(/unsupported image format/i)).toBeInTheDocument();

    // Valid JPEG file
    const goodFile = new File([new Uint8Array(2048)], 'tomato-leaf.jpg', { type: 'image/jpeg' });
    await user.upload(fileInput, goodFile);

    expect(screen.getByAltText(/selected plant leaf preview: tomato-leaf\.jpg/i)).toBeInTheDocument();
    expect(screen.getByText('tomato-leaf.jpg')).toBeInTheDocument();

    // Remove selected image
    await user.click(screen.getByRole('button', { name: /remove/i }));
    expect(screen.getByText(/drag and drop a plant leaf photo/i)).toBeInTheDocument();
  });

  it('submits selected image to prediction API and navigates to results page', async () => {
    const user = userEvent.setup();
    predictionService.submitPrediction.mockResolvedValue({
      request_id: 'mock-req-101',
      status: 'success',
      message: null,
      prediction: {
        class_name: 'Tomato___Early_blight',
        display_name: 'Tomato — Early blight',
        plant_name: 'Tomato',
        disease_name: 'Early blight',
        confidence: 0.934,
        is_healthy: false,
        top_predictions: [
          {
            class_name: 'Tomato___Early_blight',
            display_name: 'Tomato — Early blight',
            plant_name: 'Tomato',
            disease_name: 'Early blight',
            is_healthy: false,
            confidence: 0.934,
          },
        ],
      },
      uncertainty: {
        entropy: 0.21,
        margin: 0.88,
        max_probability: 0.934,
        is_uncertain: false,
        rejection_reasons: [],
      },
      explanation: {
        available: false,
        requested: true,
        method: 'Grad-CAM',
        heatmap_url: null,
        overlay_url: null,
        note: 'Approximate attention only.',
      },
      disease_information: {
        disease_name: 'Early blight',
        affected_plant: 'Tomato',
        is_healthy: false,
        summary: 'Fungal leaf spot disease.',
        symptoms: ['Dark concentric lesions on older leaves'],
        prevention: ['Rotate crops and avoid overhead irrigation'],
        treatment: ['Apply appropriate protectant fungicide per local guidance'],
        disclaimer: 'Educational guidance only.',
      },
      model_name: 'AgriVision-CNN',
      checkpoint_id: 'mock-ckpt',
      inference_device: 'cuda',
      created_at: '2026-11-18T12:00:00Z',
    });

    renderDetection();
    const fileInput = screen.getByLabelText('Choose a plant leaf image file');
    const goodFile = new File([new Uint8Array(2048)], 'leaf.jpg', { type: 'image/jpeg' });
    await user.upload(fileInput, goodFile);

    await user.click(screen.getByRole('button', { name: /analyze plant leaf/i }));

    await waitFor(() => {
      expect(predictionService.submitPrediction).toHaveBeenCalledTimes(1);
    });
    expect(await screen.findByRole('heading', { name: 'Tomato — Early blight' })).toBeInTheDocument();
  });

  it('displays clear HTTP 503 error recovery instructions when backend model is unavailable', async () => {
    const user = userEvent.setup();
    predictionService.submitPrediction.mockRejectedValue(
      new ApiError({
        message: 'Trained AgriVision-CNN checkpoint is missing.',
        status: 503,
        errorCode: 'model_unavailable',
      })
    );

    renderDetection();
    const fileInput = screen.getByLabelText('Choose a plant leaf image file');
    const goodFile = new File([new Uint8Array(2048)], 'leaf.jpg', { type: 'image/jpeg' });
    await user.upload(fileInput, goodFile);

    await user.click(screen.getByRole('button', { name: /analyze plant leaf/i }));

    expect(await screen.findByText('AI model unavailable (HTTP 503)')).toBeInTheDocument();
    expect(screen.getByText('Trained AgriVision-CNN checkpoint is missing.')).toBeInTheDocument();
  });
});
