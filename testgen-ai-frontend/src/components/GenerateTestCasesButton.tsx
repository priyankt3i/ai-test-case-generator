import type React from 'react';
import { Box, Button, CircularProgress, Alert } from '@mui/material';
import { useAppStore } from '../store';
import { generateTestCasesApi } from '../services/api';
import type { GenerateTestCasesRequest } from '../types';

const GenerateTestCasesButton: React.FC = () => {
  const {
    extractedRequirementsText,
    selectedProvider,
    selectedModel,
    apiCredentials,
    openaiFallbackApiKey,
    selectedApplications,
    applicationContexts,
    isGeneratingTestCases,
    generateTestCasesError,
    setGeneratedTestCases,
    setIsGeneratingTestCases,
    setGenerateTestCasesError,
  } = useAppStore();

  const handleGenerateTestCases = async () => {
    if (!extractedRequirementsText || !selectedProvider || !selectedModel || selectedApplications.length === 0) {
      setGenerateTestCasesError("Ensure requirements are uploaded, LLM is configured, and at least one application is selected.");
      return;
    }

    setIsGeneratingTestCases(true);
    setGenerateTestCasesError(null);

    const requestData: GenerateTestCasesRequest = {
      main_requirements_text: extractedRequirementsText,
      selected_applications: selectedApplications,
      application_contexts: applicationContexts,
      llm_provider: selectedProvider,
      model_name: selectedModel,
      api_credentials: apiCredentials,
      openai_fallback_api_key: openaiFallbackApiKey,
    };

    try {
      const response = await generateTestCasesApi(requestData);
      if (response.error) {
        setGenerateTestCasesError(response.error);
      } else {
        setGeneratedTestCases(response.results || {});
        // Optionally, show a success snackbar or simple message here
      }
    } catch (error: any) {
      console.error("Generate test cases error:", error);
      const errorMsg = error?.error || error?.detail || 'Failed to generate test cases.';
      setGenerateTestCasesError(errorMsg);
    } finally {
      setIsGeneratingTestCases(false);
    }
  };

  const canGenerate = extractedRequirementsText && selectedProvider && selectedModel && selectedApplications.length > 0;

  return (
    <Box sx={{ mt: 2 }}>
      <Button
        variant="contained"
        color="primary"
        onClick={handleGenerateTestCases}
        disabled={!canGenerate || isGeneratingTestCases}
        fullWidth
        size="large"
      >
        {isGeneratingTestCases ? <CircularProgress size={24} color="inherit" /> : '3. Generate Test Cases'}
      </Button>
      
      {generateTestCasesError && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {generateTestCasesError}
        </Alert>
      )}
      {/* Display of generatedTestCases will be handled by another component or section */}
    </Box>
  );
};

export default GenerateTestCasesButton;
