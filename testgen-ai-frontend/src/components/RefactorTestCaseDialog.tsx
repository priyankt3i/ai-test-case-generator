import React from 'react';
import {
  Dialog, DialogTitle, DialogContent, DialogActions, Button, TextField,
  CircularProgress, Alert, Typography, Paper // Removed Box
} from '@mui/material';
import { useAppStore } from '../store';
import { refactorSingleTestCaseApi } from '../services/api';
import type { RefactorSingleTestCaseRequest } from '../types'; // Removed TestCase

const RefactorTestCaseDialog: React.FC = () => {
  const {
    isRefactorDialogOpen,
    refactoringTestCase,
    refactorInstructions,
    isRefactoringSingleTc,
    refactorSingleTcError,
    selectedProvider, // Needed for API call
    selectedModel,    // Needed for API call
    apiCredentials,   // Needed for API call
    openaiFallbackApiKey, // Needed for API call
    closeRefactorDialog,
    setRefactorInstructions,
    updateSingleTestCase,
    setIsRefactoringSingleTc,
    setRefactorSingleTcError,
  } = useAppStore();

  const handleRefactor = async () => {
    if (!refactoringTestCase || !selectedProvider || !selectedModel || !refactorInstructions.trim()) {
      setRefactorSingleTcError("Missing data: Ensure LLM is configured, a test case is selected, and instructions are provided.");
      return;
    }

    setIsRefactoringSingleTc(true);
    setRefactorSingleTcError(null);

    const requestData: RefactorSingleTestCaseRequest = {
      app_name: refactoringTestCase.appName,
      tc_id: refactoringTestCase.tcId,
      instructions: refactorInstructions,
      original_tc_data: refactoringTestCase.originalData,
      llm_provider: selectedProvider,
      model_name: selectedModel,
      api_credentials: apiCredentials,
      openai_fallback_api_key: openaiFallbackApiKey,
    };

    try {
      const response = await refactorSingleTestCaseApi(requestData);
      if (response.error) {
        setRefactorSingleTcError(response.error);
      } else if (response.refactored_tc_data) {
        updateSingleTestCase(refactoringTestCase.appName, refactoringTestCase.tcId, response.refactored_tc_data);
        // closeRefactorDialog(); // updateSingleTestCase now handles closing dialog on success
      } else {
        setRefactorSingleTcError("Refactoring failed: No data returned from server.");
      }
    } catch (error: any) {
      console.error("Refactor single TC error:", error);
      const errorMsg = error?.error || error?.detail || 'Failed to refactor test case.';
      setRefactorSingleTcError(errorMsg);
    } finally {
      // setIsRefactoringSingleTc(false); // Handled by updateSingleTestCase or setRefactorSingleTcError
    }
  };

  if (!isRefactorDialogOpen || !refactoringTestCase) {
    return null;
  }

  // Display a snippet of the original test case for context
  const originalTcSnippet = JSON.stringify(refactoringTestCase.originalData, null, 2);


  return (
    <Dialog open={isRefactorDialogOpen} onClose={closeRefactorDialog} maxWidth="md" fullWidth>
      <DialogTitle>Refactor Test Case: {refactoringTestCase.tcId} (App: {refactoringTestCase.appName})</DialogTitle>
      <DialogContent dividers>
        <Typography variant="subtitle2" gutterBottom>Original Test Case Data:</Typography>
        <Paper variant="outlined" sx={{ p: 1, maxHeight: '150px', overflowY: 'auto', whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '0.8rem', mb: 2 }}>
          {originalTcSnippet}
        </Paper>
        
        <TextField
          label="Refactoring Instructions"
          multiline
          rows={4}
          fullWidth
          value={refactorInstructions}
          onChange={(e) => setRefactorInstructions(e.target.value)}
          variant="outlined"
          margin="normal"
          helperText="Provide clear instructions on how to refactor this test case."
        />
        {refactorSingleTcError && (
          <Alert severity="error" sx={{ mt: 1 }}>{refactorSingleTcError}</Alert>
        )}
      </DialogContent>
      <DialogActions sx={{p: '16px 24px'}}>
        <Button onClick={closeRefactorDialog} color="secondary" disabled={isRefactoringSingleTc}>
          Cancel
        </Button>
        <Button 
          onClick={handleRefactor} 
          variant="contained" 
          color="primary" 
          disabled={isRefactoringSingleTc || !refactorInstructions.trim()}
          startIcon={isRefactoringSingleTc ? <CircularProgress size={20} color="inherit" /> : null}
        >
          {isRefactoringSingleTc ? 'Refactoring...' : 'Refactor Test Case'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default RefactorTestCaseDialog;
