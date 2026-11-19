import { useCallback, useEffect, useState } from 'react';
import { Link, useLocation, useParams } from 'react-router-dom';
import { ScanSearch } from 'lucide-react';
import PageContainer from '../components/layout/PageContainer';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import PredictionCard from '../components/results/PredictionCard';
import ConfidenceDisplay from '../components/results/ConfidenceDisplay';
import TopPredictions from '../components/results/TopPredictions';
import DiseaseInformation from '../components/results/DiseaseInformation';
import GradCAMViewer from '../components/results/GradCAMViewer';
import LocalizationViewer from '../components/results/LocalizationViewer';
import { getHistoryRecord } from '../services/historyService';
import { readPredictionCache } from '../services/predictionService';
import { formatClassLabel } from '../utils/formatters';

/**
 * Normalizes a HistoryRecordResponse from GET /api/v1/history/{request_id}
 * into the same UI shape as POST /api/v1/predict without fabricating missing fields.
 */
function normalizeHistoryRecord(record) {
  if (!record) return null;
  const isUncertain = record.status === 'uncertain' || Boolean(record.is_uncertain);
  const topPreds = Array.isArray(record.top_predictions) ? record.top_predictions : [];
  const top1 = topPreds[0];

  return {
    request_id: record.request_id,
    status: isUncertain ? 'uncertain' : 'success',
    message: isUncertain
      ? record.rejection_reason ||
        'This prediction did not meet confidence/uncertainty thresholds and was safely rejected.'
      : null,
    prediction: isUncertain
      ? null
      : {
          class_name: record.predicted_class,
          display_name: top1?.display_name || formatClassLabel(record.predicted_class),
          plant_name: top1?.plant_name || null,
          disease_name: top1?.disease_name || null,
          confidence: record.confidence,
          top_predictions: topPreds,
          is_healthy: Boolean(record.is_healthy),
        },
    uncertainty: {
      entropy: record.entropy,
      margin: record.margin,
      max_probability: record.confidence ?? top1?.confidence ?? null,
      is_uncertain: isUncertain,
      rejection_reasons: record.rejection_reason ? [record.rejection_reason] : [],
    },
    explanation: {
      available: Boolean(record.explanation_available && (record.heatmap_url || record.overlay_url)),
      requested: Boolean(record.explanation_available),
      method: record.explanation_method || 'Grad-CAM',
      heatmap_url: record.heatmap_url || null,
      overlay_url: record.overlay_url || null,
      note:
        'Grad-CAM highlights approximate AI attention regions and is NOT a verified pixel-level disease segmentation mask.',
    },
    disease_information: isUncertain ? null : record.disease_information || null,
    model_name: record.model_name,
    checkpoint_id: record.checkpoint_id,
    inference_device: null,
    created_at: record.created_at,
  };
}

export function Results() {
  const { requestId } = useParams();
  const location = useLocation();

  const [result, setResult] = useState(null);
  const [originalImageUrl, setOriginalImageUrl] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadResult = useCallback(
    async (signal) => {
      setLoading(true);
      setError(null);

      // 1. Check fresh navigation state or sessionStorage cache for this exact request_id
      const routeState = location.state;
      if (routeState?.predictionPayload?.request_id === requestId) {
        setResult(routeState.predictionPayload);
        setOriginalImageUrl(routeState.previewDataUrl || null);
        setLoading(false);
        return;
      }

      const cached = readPredictionCache(requestId);
      if (cached?.response?.request_id === requestId) {
        setResult(cached.response);
        setOriginalImageUrl(cached.previewDataUrl || null);
        setLoading(false);
        return;
      }

      // 2. Otherwise retrieve persisted prediction directly from backend SQLite history
      try {
        const record = await getHistoryRecord(requestId, { signal });
        setResult(normalizeHistoryRecord(record));
        setOriginalImageUrl(null);
      } catch (err) {
        if (err?.errorCode === 'aborted') return;
        setResult(null);
        setError(err?.message || 'Unable to load this prediction record.');
      } finally {
        setLoading(false);
      }
    },
    [location.state, requestId]
  );

  useEffect(() => {
    const controller = new AbortController();
    loadResult(controller.signal);
    return () => controller.abort();
  }, [loadResult]);

  const isUncertain = Boolean(result && (result.status === 'uncertain' || result.uncertainty?.is_uncertain));

  return (
    <PageContainer
      title="Prediction Results"
      subtitle={requestId ? `Request ID: ${requestId}` : undefined}
      showBack
      backTo="/history"
      backLabel="Back to History"
      actions={
        <Link to="/detection" className="btn btn-primary btn-sm">
          <ScanSearch size={15} aria-hidden="true" />
          New Scan
        </Link>
      }
    >
      {loading ? (
        <LoadingSpinner label="Loading prediction results…" />
      ) : error ? (
        <ErrorMessage
          title="Could not load prediction results"
          message={error}
          recoveryHint="Confirm the Request ID exists in backend history and the FastAPI server is online."
          onRetry={() => loadResult()}
        />
      ) : result ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div className="two-col-grid">
            <PredictionCard result={result} />
            <ConfidenceDisplay
              confidence={result.prediction?.confidence}
              uncertainty={result.uncertainty}
              isUncertain={isUncertain}
            />
          </div>

          {!isUncertain ? (
            <TopPredictions
              predictions={result.prediction?.top_predictions || []}
              isUncertain={isUncertain}
            />
          ) : null}

          <div className="equal-two-col">
            <GradCAMViewer
              originalImageUrl={originalImageUrl}
              explanation={result.explanation}
              isUncertain={isUncertain}
            />
            <LocalizationViewer
              originalImageUrl={originalImageUrl}
              explanation={result.explanation}
              isUncertain={isUncertain}
            />
          </div>

          <DiseaseInformation
            diseaseInfo={result.disease_information}
            isUncertain={isUncertain}
          />
        </div>
      ) : null}
    </PageContainer>
  );
}

export default Results;
