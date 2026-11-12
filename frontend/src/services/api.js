import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

/**
 * Standardizes backend error messages into a predictable client-side structure.
 */
function parseApiError(error) {
  if (!error.response) {
    // Network failure, offline, or request timeout
    if (error.code === 'ECONNABORTED') {
      return {
        status: 408,
        error: 'Request timed out. The backend server took too long to respond.',
        actionRequired: 'Ensure the backend server is running and your GPU/CPU is not under extreme load.',
        details: null,
      };
    }
    return {
      status: 0,
      error: 'Network error: Unable to communicate with the FastAPI backend service.',
      actionRequired: 'Check that the FastAPI server is running on http://127.0.0.1:8000 and accessible.',
      details: error.message,
    };
  }

  const status = error.response.status;
  const data = error.response.data;
  const detail = data?.detail;

  let errorMessage = 'An unexpected server error occurred.';
  let actionRequired = null;
  let rawDetails = detail || data;

  if (typeof detail === 'string') {
    errorMessage = detail;
  } else if (typeof detail === 'object' && detail !== null) {
    if (Array.isArray(detail)) {
      // HTTP 422 Unprocessable Entity (FastAPI / Pydantic validation error)
      const fieldErrors = detail.map((err) => `${err.loc?.join('.')} - ${err.msg}`).join('; ');
      errorMessage = `Input validation error: ${fieldErrors}`;
      actionRequired = 'Verify the form data fields and data types meet API requirements.';
    } else {
      errorMessage = detail.message || detail.error || JSON.stringify(detail);
      actionRequired = detail.action_required || null;
    }
  } else if (data?.message) {
    errorMessage = data.message;
  }

  // HTTP Status Specific Context
  if (status === 400 && !errorMessage) {
    errorMessage = 'Invalid request parameters or corrupt image payload.';
  } else if (status === 413) {
    errorMessage = errorMessage || 'The uploaded file exceeds the maximum allowed size (25MB).';
    actionRequired = actionRequired || 'Compress the image or select a file under 25MB.';
  } else if (status === 500) {
    errorMessage = errorMessage || 'Inference pipeline failure on the backend.';
    actionRequired = actionRequired || 'Review server console logs for PyTorch/CUDA runtime errors.';
  } else if (status === 503) {
    errorMessage = errorMessage || 'Model weights are uninitialized or not found on disk.';
    actionRequired = actionRequired || 'Ensure ml/weights/component_detector_best.pt exists.';
  }

  return {
    status,
    error: errorMessage,
    actionRequired,
    details: rawDetails,
  };
}

export const apiService = {
  /**
   * Retrieves overall system health and model checkpoint readiness.
   * Path: GET /api/v1/health
   */
  async getHealthStatus() {
    try {
      const response = await apiClient.get('/health');
      return { success: true, data: response.data };
    } catch (error) {
      const parsed = parseApiError(error);
      return {
        success: false,
        status: parsed.status,
        error: parsed.error,
        actionRequired: parsed.actionRequired,
        details: parsed.details,
      };
    }
  },

  /**
   * Retrieves component detector checkpoint status and active class policies.
   * Path: GET /api/v1/detect/status
   */
  async getComponentDetectorStatus() {
    try {
      const response = await apiClient.get('/detect/status');
      return { success: true, data: response.data };
    } catch (error) {
      const parsed = parseApiError(error);
      return {
        success: false,
        status: parsed.status,
        error: parsed.error,
        actionRequired: parsed.actionRequired,
        details: parsed.details,
      };
    }
  },

  /**
   * Submits a vehicle inspection image for 3-class component localization.
   * Path: POST /api/v1/detect/components
   *
   * @param {File} file - High-resolution vehicle image (JPEG, PNG, WEBP)
   * @param {number} [confidenceThreshold=0.25] - Confidence cutoff (0.01 to 1.0)
   * @returns {Promise<{success: boolean, data?: object, status?: number, error?: string, actionRequired?: string, details?: any}>}
   */
  async detectComponents(file, confidenceThreshold = 0.25) {
    if (!file) {
      return {
        success: false,
        status: 400,
        error: 'No image file provided for component detection.',
        actionRequired: 'Select an image file before triggering detection.',
      };
    }

    const formData = new FormData();
    formData.append('file', file);
    if (confidenceThreshold !== undefined && confidenceThreshold !== null) {
      formData.append('confidence_threshold', confidenceThreshold.toString());
    }

    try {
      const response = await apiClient.post('/detect/components', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const data = response.data;

      // Validate core schema fields returned by backend
      if (!data || typeof data !== 'object') {
        throw new Error('Malformed API response: Expected JSON object.');
      }
      if (!Array.isArray(data.detections)) {
        throw new Error('Malformed API response: "detections" must be an array.');
      }
      if (!data.summary || typeof data.summary !== 'object') {
        throw new Error('Malformed API response: "summary" object missing.');
      }

      return {
        success: true,
        data,
      };
    } catch (error) {
      const parsed = parseApiError(error);
      return {
        success: false,
        status: parsed.status,
        error: parsed.error,
        actionRequired: parsed.actionRequired,
        details: parsed.details,
      };
    }
  },

  /**
   * Backward-compatible end-to-end diagnostic pipeline endpoint.
   * Path: POST /api/v1/diagnose/image
   */
  async uploadDiagnosisImage(file, confidenceThreshold = 0.35) {
    const formData = new FormData();
    formData.append('file', file);
    if (confidenceThreshold) {
      formData.append('confidence_threshold', confidenceThreshold);
    }

    try {
      const response = await apiClient.post('/diagnose/image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return { success: true, data: response.data };
    } catch (error) {
      const parsed = parseApiError(error);
      return {
        success: false,
        status: parsed.status,
        error: parsed.error,
        actionRequired: parsed.actionRequired,
        details: parsed.details,
      };
    }
  },
};

export default apiService;
