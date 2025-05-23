import type React from 'react';
import { Box, Button, Typography, CircularProgress, Alert, List, ListItem, ListItemText, Paper, Backdrop } from '@mui/material'; // Added Backdrop
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

  console.log('[IdentifyApplications] Render - isIdentifyingApps from store:', isIdentifyingApps);

  const handleIdentifyApplications = async () => {
    if (!extractedRequirementsText) {
      setIdentifyAppsError("Please upload and extract text from a requirements document first.");
      return;
    }
    if (!selectedProvider || !selectedModel) {
      setIdentifyAppsError("Please select an LLM provider and model first.");
      return;
    }

    console.log('[IdentifyApplications] handleIdentifyApplications - Setting isIdentifyingApps to true');
    setIsIdentifyingApps(true);
    setIdentifyAppsError(null);

    // const requestData: IdentifyAppRequest = { // Removed artificial delay and the setTimeout
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
      console.log('[IdentifyApplications] handleIdentifyApplications - Setting isIdentifyingApps to false in finally block');
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

      {/* Diagnostic text for loading state */}
      <Typography variant="caption" sx={{ display: 'block', mt: 1, color: isIdentifyingApps ? 'red' : 'green' }}>
        {isIdentifyingApps ? "DEBUG: Loading state is TRUE" : "DEBUG: Loading state is FALSE"}
      </Typography>

      <Backdrop
        sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 100 }} // Ensure it's above other elements
        open={isIdentifyingApps}
      >
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', color: 'white' }}>
          <CircularProgress color="inherit" sx={{ mb: 2 }} />
          <Typography variant="h6">Identifying Applications...</Typography>
        </Box>
      </Backdrop>
      
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
