import axios from 'axios';
import type { 
  LLMProviderConfigResponse, 
  IdentifyAppRequest, IdentifyAppResponse, 
  ContextUploadResponse,
  GenerateTestCasesRequest, GenerateTestCasesResponse
} from '../types'; // Use type-only import

// Vite exposes env variables prefixed with VITE_ on import.meta.env
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getLlmProviderConfigs = async (): Promise<LLMProviderConfigResponse> => {
  try {
    const response = await apiClient.get<LLMProviderConfigResponse>('/config/llm-providers');
    return response.data;
  } catch (error) {
    console.error('Error fetching LLM provider configs:', error);
    // In a real app, you might want to throw a more specific error or handle it differently
    throw error; 
  }
};

// Add other API service functions here later, e.g.:
// export const uploadRequirements = async (file: File): Promise<any> => { ... }

export const identifyApplicationsApi = async (data: IdentifyAppRequest): Promise<IdentifyAppResponse> => {
  try {
    const response = await apiClient.post<IdentifyAppResponse>('/actions/identify-applications', data);
    return response.data;
  } catch (error) {
    console.error('Error identifying applications:', error);
    // Consider how to propagate error details for UI display
    if (axios.isAxiosError(error) && error.response) {
      throw error.response.data; // Throw backend error details if available
    }
    throw error; // Fallback
  }
};

export const uploadContextFileApi = async (appName: string, file: File): Promise<ContextUploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await apiClient.post<ContextUploadResponse>(`/upload/context/${appName}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error(`Error uploading context file for ${appName}:`, error);
    if (axios.isAxiosError(error) && error.response) {
      throw error.response.data; 
    }
    throw error;
  }
};

export const generateTestCasesApi = async (data: GenerateTestCasesRequest): Promise<GenerateTestCasesResponse> => {
  try {
    const response = await apiClient.post<GenerateTestCasesResponse>('/actions/generate-test-cases', data);
    return response.data;
  } catch (error) {
    console.error('Error generating test cases:', error);
    if (axios.isAxiosError(error) && error.response) {
      throw error.response.data; 
    }
    throw error;
  }
};

// Add other API service functions here later

export default apiClient;
