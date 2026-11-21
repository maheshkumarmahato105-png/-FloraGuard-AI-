import { useCallback, useEffect, useRef, useState } from 'react';
import { savePredictionCache, submitPrediction } from '../services/predictionService';

function readFileAsDataUrl(file) {
  return new Promise((resolve) => {
    if (!file || typeof FileReader === 'undefined') {
      resolve(null);
      return;
    }
    const reader = new FileReader();
    reader.onload = () => resolve(typeof reader.result === 'string' ? reader.result : null);
    reader.onerror = () => resolve(null);
    reader.readAsDataURL(file);
  });
}

/**
 * Manages plant disease prediction submission with:
 * - Duplicate-submission prevention
 * - Real upload progress tracking
 * - Clean stage transitions ('idle' -> 'uploading' -> 'analyzing' -> 'done' | 'error')
 * - Abort cleanup on unmount
 */
export function usePrediction() {
  const [stage, setStage] = useState('idle');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const inFlightRef = useRef(false);
  const abortRef = useRef(null);

  useEffect(() => {
    return () => {
      if (abortRef.current) {
        abortRef.current.abort();
      }
    };
  }, []);

  const reset = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    inFlightRef.current = false;
    setStage('idle');
    setUploadProgress(0);
    setError(null);
    setResult(null);
  }, []);

  const cancel = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    inFlightRef.current = false;
    setStage('idle');
    setUploadProgress(0);
  }, []);

  const predict = useCallback(async (file, { generateExplanation = false } = {}) => {
    if (inFlightRef.current) {
      return null;
    }

    inFlightRef.current = true;
    setError(null);
    setResult(null);
    setUploadProgress(0);
    setStage('uploading');

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const previewPromise = readFileAsDataUrl(file);
      const response = await submitPrediction(file, {
        generateExplanation,
        signal: controller.signal,
        onProgress: (pct) => {
          setUploadProgress(pct);
          if (pct >= 100) {
            setStage('analyzing');
          }
        },
      });

      const previewDataUrl = await previewPromise;
      const enriched = {
        response,
        previewDataUrl,
        originalFileName: file?.name || 'leaf-image.jpg',
      };

      if (response?.request_id) {
        savePredictionCache(response.request_id, enriched);
      }

      setResult(enriched);
      setUploadProgress(100);
      setStage('done');
      return enriched;
    } catch (err) {
      if (err?.errorCode === 'aborted') {
        setStage('idle');
        return null;
      }
      setError({
        message: err?.message || 'Prediction failed. Please try again.',
        status: err?.status || 0,
        errorCode: err?.errorCode || 'prediction_error',
        isNetworkError: Boolean(err?.isNetworkError),
        isTimeout: Boolean(err?.isTimeout),
      });
      setStage('error');
      return null;
    } finally {
      inFlightRef.current = false;
      abortRef.current = null;
    }
  }, []);

  return {
    stage,
    uploadProgress,
    isSubmitting: stage === 'uploading' || stage === 'analyzing',
    error,
    result,
    predict,
    cancel,
    reset,
  };
}
