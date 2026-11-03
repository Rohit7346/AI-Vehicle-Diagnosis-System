import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export const apiService = {
  /**
   * Retrieves system health and model readiness
   */
  async getHealthStatus() {
    try {
      const response = await apiClient.get('/health');
      return { success: true, data: response.data };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || error.message || 'Failed to connect to backend',
      };
    }
  },

  /**
   * Submits an inspection image for component and visible fault diagnosis
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
      const errorData = error.response?.data?.detail;
      return {
        success: false,
        status: error.response?.status,
        error: typeof errorData === 'object' ? errorData.message || errorData.error : errorData || error.message,
        details: typeof errorData === 'object' ? errorData : null,
      };
    }
  },
};

export default apiService;
