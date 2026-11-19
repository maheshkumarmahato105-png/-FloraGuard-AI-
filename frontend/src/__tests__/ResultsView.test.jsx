import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { ThemeProvider } from '../context/ThemeContext';
import { ToastProvider } from '../context/ToastContext';
import Results from '../pages/Results';
import * as historyService from '../services/historyService';

vi.mock('../services/historyService');

function renderResultsWithState(requestId, state) {
  return render(
    <ThemeProvider>
      <ToastProvider>
        <MemoryRouter initialEntries={[{ pathname: `/results/${requestId}`, state }]}>
          <Routes>
            <Route path="/results/:requestId" element={<Results />} />
          </Routes>
        </MemoryRouter>
      </ToastProvider>
    </ThemeProvider>
  );
}

describe('Results page — accepted vs uncertain predictions & XAI disclaimers (mocked data)', () => {
  it('renders successful prediction with top-3, disease guidance, and Grad-CAM attention disclaimers', async () => {
    const user = userEvent.setup();
    const mockSuccess = {
      request_id: 'req-success-1',
      status: 'success',
      message: null,
      prediction: {
        class_name: 'Potato___Late_blight',
        display_name: 'Potato — Late blight',
        plant_name: 'Potato',
        disease_name: 'Late blight',
        confidence: 0.912,
        is_healthy: false,
        top_predictions: [
          {
            class_name: 'Potato___Late_blight',
            display_name: 'Potato — Late blight',
            plant_name: 'Potato',
            disease_name: 'Late blight',
            is_healthy: false,
            confidence: 0.912,
          },
          {
            class_name: 'Potato___Early_blight',
            display_name: 'Potato — Early blight',
            plant_name: 'Potato',
            disease_name: 'Early blight',
            is_healthy: false,
            confidence: 0.064,
          },
          {
            class_name: 'Potato___healthy',
            display_name: 'Potato — Healthy',
            plant_name: 'Potato',
            disease_name: 'Healthy',
            is_healthy: true,
            confidence: 0.024,
          },
        ],
      },
      uncertainty: {
        entropy: 0.342,
        margin: 0.848,
        max_probability: 0.912,
        is_uncertain: false,
        rejection_reasons: [],
      },
      explanation: {
        available: true,
        requested: true,
        method: 'Grad-CAM',
        heatmap_url: '/api/v1/explanations/req-success-1_heatmap.png',
        overlay_url: '/api/v1/explanations/req-success-1_overlay.png',
        note: 'Grad-CAM highlights approximate AI attention regions and is NOT a verified pixel-level disease segmentation mask.',
      },
      disease_information: {
        disease_name: 'Late blight',
        affected_plant: 'Potato',
        is_healthy: false,
        summary: 'Water-soaked dark lesions caused by Phytophthora infestans.',
        symptoms: ['Dark water-soaked lesions along leaf margins'],
        prevention: ['Use certified disease-free seed tubers'],
        treatment: ['Remove heavily infected foliage and consult local agronomy guidance'],
        disclaimer: 'General educational information only.',
      },
      model_name: 'AgriVision-CNN',
      checkpoint_id: 'best_model',
      inference_device: 'cuda',
      created_at: '2026-11-18T12:15:00Z',
    };

    renderResultsWithState('req-success-1', {
      predictionPayload: mockSuccess,
      previewDataUrl: 'data:image/png;base64,iVBORw0KGgo=',
    });

    expect(screen.getByRole('heading', { name: 'Potato — Late blight' })).toBeInTheDocument();
    expect(screen.getByText('Disease Detected')).toBeInTheDocument();
    expect(screen.getAllByText('91.2%').length).toBeGreaterThan(0);
    expect(screen.getByText('Top 3 Predictions')).toBeInTheDocument();
    expect(screen.getByText('Dark water-soaked lesions along leaf margins')).toBeInTheDocument();
    expect(screen.getByText('Remove heavily infected foliage and consult local agronomy guidance')).toBeInTheDocument();
    expect(screen.getByText(/NOT a verified pixel-level disease segmentation mask/i)).toBeInTheDocument();
    expect(screen.getAllByText(/approximate AI attention region/i).length).toBeGreaterThan(0);

    // Switch Grad-CAM tab to Heatmap
    await user.click(screen.getByRole('tab', { name: 'Grad-CAM Heatmap' }));
    expect(screen.getByAltText('Grad-CAM Heatmap view of plant leaf analysis')).toBeInTheDocument();
  });

  it('renders uncertain prediction warning WITHOUT unverified diagnosis or treatment guidance', () => {
    const mockUncertain = {
      request_id: 'req-uncertain-2',
      status: 'uncertain',
      message: 'Confidence below safety threshold (0.41 < 0.60).',
      prediction: null,
      uncertainty: {
        entropy: 2.15,
        margin: 0.04,
        max_probability: 0.41,
        is_uncertain: true,
        rejection_reasons: ['Confidence below threshold', 'High predictive entropy'],
      },
      explanation: null,
      disease_information: null,
      model_name: 'AgriVision-CNN',
      checkpoint_id: 'best_model',
      inference_device: 'cuda',
      created_at: '2026-11-18T12:20:00Z',
    };

    renderResultsWithState('req-uncertain-2', {
      predictionPayload: mockUncertain,
      previewDataUrl: 'data:image/png;base64,iVBORw0KGgo=',
    });

    expect(screen.getByText(/Uncertain Prediction — Manual Review Recommended/i)).toBeInTheDocument();
    expect(screen.getByText(/consult a qualified agricultural extension specialist/i)).toBeInTheDocument();
    expect(screen.queryByText('Treatment Guidance')).not.toBeInTheDocument();
    expect(screen.queryByText(/Top \d+ Predictions/i)).not.toBeInTheDocument();
  });

  it('keeps prediction intact and reports Grad-CAM unavailable when explanation fails', () => {
    const mockNoExplanation = {
      request_id: 'req-no-xai-3',
      status: 'success',
      message: null,
      prediction: {
        class_name: 'Corn___healthy',
        display_name: 'Corn — Healthy',
        plant_name: 'Corn',
        disease_name: 'Healthy',
        confidence: 0.981,
        is_healthy: true,
        top_predictions: [],
      },
      uncertainty: {
        entropy: 0.08,
        margin: 0.96,
        max_probability: 0.981,
        is_uncertain: false,
        rejection_reasons: [],
      },
      explanation: {
        available: false,
        requested: true,
        method: 'Grad-CAM',
        heatmap_url: null,
        overlay_url: null,
      },
      disease_information: null,
      model_name: 'AgriVision-CNN',
      checkpoint_id: 'best_model',
      inference_device: 'cpu',
      created_at: '2026-11-18T12:22:00Z',
    };

    renderResultsWithState('req-no-xai-3', { predictionPayload: mockNoExplanation });

    expect(screen.getByRole('heading', { name: 'Corn — Healthy' })).toBeInTheDocument();
    expect(screen.getByText('Grad-CAM explanation unavailable')).toBeInTheDocument();
  });

  it('loads persisted record and disease_information directly via history API when reopened', async () => {
    historyService.getHistoryRecord.mockResolvedValue({
      request_id: 'req-reopen-4',
      created_at: '2026-11-18T12:30:00Z',
      status: 'success',
      predicted_class: 'tomato__early_blight',
      is_healthy: false,
      confidence: 0.94,
      top_predictions: [
        {
          class_name: 'tomato__early_blight',
          display_name: 'Tomato — Early Blight',
          plant_name: 'Tomato',
          disease_name: 'Early Blight',
          is_healthy: false,
          confidence: 0.94,
        },
      ],
      entropy: 0.21,
      margin: 0.87,
      is_uncertain: false,
      rejection_reason: null,
      model_name: 'agrivision_cnn',
      checkpoint_id: 'ckpt-test',
      image_id: 'img-test',
      explanation_available: false,
      explanation_method: null,
      heatmap_url: null,
      overlay_url: null,
      disease_information: {
        disease_name: 'Early Blight',
        affected_plant: 'Tomato',
        is_healthy: false,
        summary: 'Common fungal disease caused by Alternaria solani.',
        symptoms: ['Brown-to-black circular spots showing concentric target rings.'],
        prevention: ['Stake/prune lower leaves near soil.'],
        treatment: ['Remove severely infected lower leaves promptly.'],
        disclaimer: 'General educational information only.',
      },
    });

    renderResultsWithState('req-reopen-4', null);

    expect(await screen.findByRole('heading', { name: 'Tomato — Early Blight' })).toBeInTheDocument();
    expect(screen.getByText('Brown-to-black circular spots showing concentric target rings.')).toBeInTheDocument();
    expect(screen.getByText('Remove severely infected lower leaves promptly.')).toBeInTheDocument();
  });
});
