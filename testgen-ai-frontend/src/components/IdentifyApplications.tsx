import React from 'react';
import { Box, Button, Typography, CircularProgress, Alert, List, ListItem, ListItemText, Paper } from '@mui/material';
import { useAppStore } from '../store';
import { identifyApplicationsApi } from '../services/api';
import type { IdentifyAppRequest } from '../types'; // Use type-only import

const IdentifyApplications: React.FC = () => {
  const {
    extractedRequirementsText,
    selectedProvider,
    selectedModel,
    apiCredentials,
    openaiFallbackApiKey,
    identifiedApplications,
    isIdentifyingApps,
    identifyAppsError,
    setIdentifiedApplications,
    setIsIdentifyingApps,
    setIdentifyAppsError,
  } = useAppStore();

  const handleIdentifyApplications = async () => {
    if (!extractedRequirementsText) {
      setIdentifyAppsError("Please upload and extract text from a requirements document first.");
      return;
    }
    if (!selectedProvider || !selectedModel) {
      setIdentifyAppsError("Please select an LLM provider and model first.");
      return;
    }

    setIsIdentifyingApps(true);
    setIdentifyAppsError(null);

    const requestData: IdentifyAppRequest = {
      extracted_text: extractedRequirementsText,
      llm_provider: selectedProvider,
      model_name: selectedModel,
      api_credentials: apiCredentials,
      openai_fallback_api_key: openaiFallbackApiKey,
    };

    try {
      const response = await identifyApplicationsApi(requestData);
      if (response.error) {
        setIdentifyAppsError(response.error);
      } else {
        setIdentifiedApplications(response.identified_applications || []);
      }
    } catch (error: any) {
      console.error("Identify applications error:", error);
      const errorMsg = error?.error || error?.detail || 'Failed to identify applications.';
      setIdentifyAppsError(errorMsg);
    } finally {
      setIsIdentifyingApps(false);
    }
  };

  return (
    <Box sx={{ mt: 2 }}>
      <Button
        variant="contained"
        onClick={handleIdentifyApplications}
        disabled={!extractedRequirementsText || !selectedProvider || !selectedModel || isIdentifyingApps}
        fullWidth
      >
        {isIdentifyingApps ? <CircularProgress size={24} color="inherit" /> : '1. Identify Applications from Requirements'}
      </Button>

      {isIdentifyingApps && <CircularProgress sx={{ display: 'block', margin: '20px auto' }} />}
      
      {identifyAppsError && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {identifyAppsError}
        </Alert>
      )}

      {identifiedApplications.length > 0 && !identifyAppsError && (
        <Paper elevation={2} sx={{ mt: 2, p: 2 }}>
          <Typography variant="h6" gutterBottom>Identified Applications:</Typography>
          <List dense>
            {identifiedApplications.map((app, index) => (
              <ListItem key={index}>
                <ListItemText primary={app} />
              </ListItem>
            ))}
          </List>
        </Paper>
      )}
    </Box>
  );
};

export default IdentifyApplications;
