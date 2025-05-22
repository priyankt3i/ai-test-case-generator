import React, { useEffect, useMemo } from 'react';
import { Box, Typography, Select, MenuItem, TextField, FormControl, InputLabel, CircularProgress, Alert } from '@mui/material';
import type { SelectChangeEvent } from '@mui/material'; // Type-only import
import { useAppStore } from '../store';
import { getLlmProviderConfigs } from '../services/api';
import type { LLMProviderDetail } from '../types'; // Type-only import

const LLMConfigSidebar: React.FC = () => {
  const {
    llmProviders,
    selectedProvider,
    selectedModel,
    availableModels,
    apiCredentials,
    openaiFallbackApiKey,
    setLlmProviders,
    setSelectedProvider,
    setSelectedModel,
    setApiCredential,
    setOpenAIFallbackApiKey,
  } = useAppStore();

  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  useEffect(() => {
    const fetchConfigs = async () => {
      try {
        setLoading(true);
        setError(null);
        const configResponse = await getLlmProviderConfigs();
        setLlmProviders(configResponse.providers);
      } catch (err) {
        console.error("Failed to fetch LLM configs:", err);
        setError('Failed to load LLM configurations from backend.');
      } finally {
        setLoading(false);
      }
    };
    fetchConfigs();
  }, [setLlmProviders]);

  const handleProviderChange = (event: SelectChangeEvent<string | null>) => {
    setSelectedProvider(event.target.value as string | null);
  };

  const handleModelChange = (event: SelectChangeEvent<string | null>) => {
    setSelectedModel(event.target.value as string | null);
  };

  const handleCredentialChange = (provider: string, key: string, value: string) => {
    setApiCredential(provider, key, value);
  };
  
  const currentProviderDetails: LLMProviderDetail | null = useMemo(() => {
    if (selectedProvider && llmProviders[selectedProvider]) {
      return llmProviders[selectedProvider];
    }
    return null;
  }, [selectedProvider, llmProviders]);

  if (loading) {
    return <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', p: 2 }}><CircularProgress /></Box>;
  }

  if (error) {
    return <Alert severity="error" sx={{ m: 2 }}>{error}</Alert>;
  }

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>LLM Configuration</Typography>
      
      <FormControl fullWidth margin="normal">
        <InputLabel id="llm-provider-select-label">LLM Provider</InputLabel>
        <Select
          labelId="llm-provider-select-label"
          id="llm-provider-select"
          value={selectedProvider || ''}
          label="LLM Provider"
          onChange={handleProviderChange}
        >
          <MenuItem value=""><em>None</em></MenuItem>
          {Object.keys(llmProviders).map((providerName) => (
            <MenuItem key={providerName} value={providerName}>
              {providerName}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {selectedProvider && currentProviderDetails && (
        <>
          <FormControl fullWidth margin="normal" disabled={availableModels.length === 0}>
            <InputLabel id="llm-model-select-label">Model</InputLabel>
            <Select
              labelId="llm-model-select-label"
              id="llm-model-select"
              value={selectedModel || ''}
              label="Model"
              onChange={handleModelChange}
            >
              <MenuItem value=""><em>None</em></MenuItem>
              {availableModels.map((modelName) => (
                <MenuItem key={modelName} value={modelName}>
                  {modelName}
                </MenuItem>
              ))}
            </Select>
            {availableModels.length === 0 && <Typography variant="caption" color="textSecondary">No models available for this provider or provider not fully selected.</Typography>}
          </FormControl>

          {currentProviderDetails.notes && (
            <Typography variant="caption" display="block" gutterBottom sx={{mt: 1, color: 'text.secondary'}}>
              {currentProviderDetails.notes}
            </Typography>
          )}

          <Typography variant="subtitle1" gutterBottom sx={{ mt: 2 }}>API Credentials</Typography>
          {currentProviderDetails.credentials.map((credKey) => {
            // Special handling for Bedrock Embedding Model ID (already handled by backend config, not user input here)
            // Or Ollama 'model' which is the dropdown.
            if ((selectedProvider === "AWS Bedrock" && credKey === "embedding_model_id") || (selectedProvider === "Ollama" && credKey === "model")) {
              return null;
            }
            const isSecret = credKey.toLowerCase().includes('key') || credKey.toLowerCase().includes('secret') || credKey.toLowerCase().includes('token');
            return (
              <TextField
                key={credKey}
                fullWidth
                margin="normal"
                label={credKey.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                type={isSecret ? 'password' : 'text'}
                value={apiCredentials[credKey] || ''}
                onChange={(e) => handleCredentialChange(selectedProvider, credKey, e.target.value)}
                variant="outlined"
                size="small"
              />
            );
          })}

          {/* Fallback Key for certain providers */}
          {(currentProviderDetails.embeddings_module === null || selectedProvider === "Claude" || selectedProvider === "Groq") && (
             <TextField
                fullWidth
                margin="normal"
                label="OpenAI API Key (for RAG Fallback)"
                type="password"
                value={openaiFallbackApiKey}
                onChange={(e) => setOpenAIFallbackApiKey(e.target.value)}
                helperText={`${selectedProvider} may require OpenAI embeddings for RAG features.`}
                variant="outlined"
                size="small"
              />
          )}
        </>
      )}
    </Box>
  );
};

export default LLMConfigSidebar;
