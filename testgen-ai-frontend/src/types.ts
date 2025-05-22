// src/types.ts

// Mirrors the Pydantic model LLMProviderDetail from the backend
export interface LLMProviderDetail {
  models: string[];
  credentials: string[];
  llm_module: string;
  llm_class: string;
  embeddings_module?: string | null;
  embeddings_class?: string | null;
  notes?: string | null;
  prompt_templates?: Record<string, string> | null;
  embedding_model_ids?: string[] | null; // For Bedrock
  embeddings_model_id?: string | null;   // For Gemini
}

// For the API response from /api/config/llm-providers
export interface LLMProviderConfigResponse {
  providers: Record<string, LLMProviderDetail>;
}

// You can add other frontend-specific types or interfaces here as needed.
// For example, for API request bodies if they differ slightly or for UI state.

export interface ApiCredentials {
  [key: string]: string;
}

export interface IdentifyAppRequest {
  extracted_text: string;
  llm_provider: string;
  model_name: string;
  api_credentials: Record<string, string>;
  openai_fallback_api_key?: string | null;
}

export interface IdentifyAppResponse {
  identified_applications: string[];
  error?: string | null;
}

export interface ContextUploadResponse {
  app_name: string;
  filename: string;
  content_type?: string | null;
  extracted_text?: string | null;
  error?: string | null;
}

export interface GenerateTestCasesRequest {
  main_requirements_text: string;
  selected_applications: string[];
  application_contexts: Record<string, string[]>; // appName to list of context strings
  llm_provider: string;
  model_name: string;
  api_credentials: Record<string, string>;
  openai_fallback_api_key?: string | null;
}

export interface GenerateTestCasesResponse {
  // results maps appName to either List<TestCase> (actually List<Record<string, any>> for now) or an error string
  results: Record<string, any>; 
  error?: string | null; // For overall errors
}

// Example for a test case structure, if you plan to manage them in frontend state
export interface TestCase {
  id: string; // Or "Test Case ID": string;
  name: string; // Or "Test Case Name": string;
  description?: string;
  steps?: string; // Or an array of step objects
  expectedResults?: string;
  // Add other fields as they are defined in your backend/Excel
  [key: string]: any; // Allow other dynamic fields
}
