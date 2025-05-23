import axios from 'axios';
import type { 
  LLMProviderConfigResponse, 
  IdentifyAppRequest, IdentifyAppResponse, 
  ContextUploadResponse,
  GenerateTestCasesRequest, GenerateTestCasesResponse,
  RefactorSingleTestCaseRequest, RefactorSingleTestCaseResponse,
  RefactorAllTestCasesRequest, RefactorAllTestCasesResponse,
  AIReviewRequest, AIReviewResponse,
  ApplyAIReviewRequest, ApplyAIReviewResponse,
  ExportRequest
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

export const refactorSingleTestCaseApi = async (data: RefactorSingleTestCaseRequest): Promise<RefactorSingleTestCaseResponse> => {
  try {
    const response = await apiClient.post<RefactorSingleTestCaseResponse>('/actions/refactor/single', data);
    return response.data;
  } catch (error) {
    console.error('Error refactoring single test case:', error);
    if (axios.isAxiosError(error) && error.response) {
      throw error.response.data; 
    }
    throw error;
  }
};

export const refactorAllTestCasesApi = async (appName: string, data: RefactorAllTestCasesRequest): Promise<RefactorAllTestCasesResponse> => {
  try {
    // Note: appName is part of the URL path, data is the request body
    const response = await apiClient.post<RefactorAllTestCasesResponse>(`/actions/refactor/all/${appName}`, data);
    return response.data;
  } catch (error) {
    console.error(`Error refactoring all test cases for ${appName}:`, error);
    if (axios.isAxiosError(error) && error.response) {
      throw error.response.data; 
    }
    throw error;
  }
};

export const performAiReviewApi = async (appName: string, data: AIReviewRequest): Promise<AIReviewResponse> => {
  try {
    const response = await apiClient.post<AIReviewResponse>(`/actions/review/${appName}`, data);
    return response.data;
  } catch (error) {
    console.error(`Error performing AI review for ${appName}:`, error);
    if (axios.isAxiosError(error) && error.response) {
      throw error.response.data; 
    }
    throw error;
  }
};

export const applyAiReviewApi = async (appName: string, data: ApplyAIReviewRequest): Promise<ApplyAIReviewResponse> => {
  try {
    const response = await apiClient.post<ApplyAIReviewResponse>(`/actions/review/apply/${appName}`, data);
    return response.data;
  } catch (error) {
    console.error(`Error applying AI review for ${appName}:`, error);
    if (axios.isAxiosError(error) && error.response) {
      throw error.response.data;
    }
    throw error;
  }
};

export const exportToExcelApi = async (data: ExportRequest): Promise<Blob> => {
  try {
    const response = await apiClient.post<Blob>('/export/excel', data, {
      responseType: 'blob', // Important for file downloads
    });
    return response.data;
  } catch (error) {
    console.error('Error exporting to Excel:', error);
    // Handle error appropriately, maybe try to parse error if it's JSON
    // For now, just rethrow. The caller might need to try to read error from blob.
    throw error;
  }
};

// Add other API service functions here later

export default apiClient;
