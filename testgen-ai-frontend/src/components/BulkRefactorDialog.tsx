import React from 'react';
import {
  Dialog, DialogTitle, DialogContent, DialogActions, Button, TextField,
  CircularProgress, Alert, Typography
} from '@mui/material';
import { useAppStore } from '../store';
import { refactorAllTestCasesApi } from '../services/api';
import type { RefactorAllTestCasesRequest } from '../types'; // Removed TestCase

const BulkRefactorDialog: React.FC = () => {
  const {
    isBulkRefactorDialogOpen,
    bulkRefactoringAppName,
    bulkRefactorInstructions,
    isRefactoringBulkTc,
    refactorBulkTcError,
    generatedTestCases, // To get the original list for the app
    selectedProvider, 
    selectedModel,    
    apiCredentials,   
    openaiFallbackApiKey,
    closeBulkRefactorDialog,
    setBulkRefactorInstructions,
    updateAllTestCasesForApp,
    setIsRefactoringBulkTc,
    setRefactorBulkTcError,
  } = useAppStore();

  const handleBulkRefactor = async () => {
    if (!bulkRefactoringAppName || !selectedProvider || !selectedModel || !bulkRefactorInstructions.trim()) {
      setRefactorBulkTcError("Missing data: Ensure LLM is configured, an app is selected, and instructions are provided.");
      return;
    }

    const originalTcListForApp = generatedTestCases[bulkRefactoringAppName];
    if (!Array.isArray(originalTcListForApp) || originalTcListForApp.length === 0) {
      setRefactorBulkTcError(`No test cases found for application "${bulkRefactoringAppName}" to refactor.`);
      return;
    }

    setIsRefactoringBulkTc(true);
    setRefactorBulkTcError(null);

    const requestData: RefactorAllTestCasesRequest = {
      instructions: bulkRefactorInstructions,
      original_tc_list: originalTcListForApp,
      llm_provider: selectedProvider,
      model_name: selectedModel,
      api_credentials: apiCredentials,
      openai_fallback_api_key: openaiFallbackApiKey,
    };

    try {
      const response = await refactorAllTestCasesApi(bulkRefactoringAppName, requestData);
      if (response.error) {
        setRefactorBulkTcError(response.error);
      } else if (response.refactored_test_cases) {
        updateAllTestCasesForApp(bulkRefactoringAppName, response.refactored_test_cases);
        // closeBulkRefactorDialog(); // updateAllTestCasesForApp now handles closing
      } else {
        setRefactorBulkTcError("Bulk refactoring failed: No data returned from server.");
      }
    } catch (error: any) {
      console.error("Bulk refactor TC error:", error);
      const errorMsg = error?.error || error?.detail || 'Failed to bulk refactor test cases.';
      setRefactorBulkTcError(errorMsg);
    } finally {
      // setIsRefactoringBulkTc(false); // Handled by updateAllTestCasesForApp or setRefactorBulkTcError
    }
  };

  if (!isBulkRefactorDialogOpen || !bulkRefactoringAppName) {
    return null;
  }

  return (
    <Dialog open={isBulkRefactorDialogOpen} onClose={closeBulkRefactorDialog} maxWidth="md" fullWidth>
      <DialogTitle>Bulk Refactor Test Cases for: {bulkRefactoringAppName}</DialogTitle>
      <DialogContent dividers>
        <Typography variant="body2" gutterBottom>
          Enter instructions to apply to all test cases for the application "{bulkRefactoringAppName}".
          The original test cases will be sent to the LLM along with your instructions.
        </Typography>
        <TextField
          label="Bulk Refactoring Instructions"
          multiline
          rows={4}
          fullWidth
          value={bulkRefactorInstructions}
          onChange={(e) => setBulkRefactorInstructions(e.target.value)}
          variant="outlined"
          margin="normal"
        />
        {refactorBulkTcError && (
          <Alert severity="error" sx={{ mt: 1 }}>{refactorBulkTcError}</Alert>
        )}
      </DialogContent>
      <DialogActions sx={{p: '16px 24px'}}>
        <Button onClick={closeBulkRefactorDialog} color="secondary" disabled={isRefactoringBulkTc}>
          Cancel
        </Button>
        <Button 
          onClick={handleBulkRefactor} 
          variant="contained" 
          color="primary" 
          disabled={isRefactoringBulkTc || !bulkRefactorInstructions.trim()}
          startIcon={isRefactoringBulkTc ? <CircularProgress size={20} color="inherit" /> : null}
        >
          {isRefactoringBulkTc ? 'Refactoring All...' : 'Refactor All Test Cases'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default BulkRefactorDialog;
