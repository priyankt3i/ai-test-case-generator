import type React from 'react';
import {
  Box, Typography, Paper, Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, Alert, Accordion, AccordionSummary, AccordionDetails,
  IconButton, Tooltip, Button, CircularProgress
} from '@mui/material'; // Added CircularProgress
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import EditIcon from '@mui/icons-material/Edit';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import RateReviewIcon from '@mui/icons-material/RateReview'; 
import { useAppStore } from '../store';
import type { TestCase, AIReviewRequest } from '../types';
import { performAiReviewApi } from '../services/api'; 
import AIReviewDisplay from './AIReviewDisplay'; // Import the new component

// Helper to get a consistent set of columns, prioritizing EXCEL_EXPECTED_COLUMNS
// This should ideally come from config or be more robustly defined.
const getTableColumns = (testCasesForApp: TestCase[]): string[] => {
  if (!testCasesForApp || testCasesForApp.length === 0) {
    return ["Test Case ID", "Test Case Name", "Description", "Steps", "Expected Results"]; // Default
  }
  // Extract all unique keys from all test cases for this app
  const allKeys = new Set<string>();
  testCasesForApp.forEach(tc => {
    if (typeof tc === 'object' && tc !== null) {
      Object.keys(tc).forEach(key => allKeys.add(key));
    }
  });
  
  // Prioritize standard columns if they exist, then add others
  const standardColumns = ["Test Case ID", "Test Case Name", "Description", "Steps", "Expected Results"];
  const prioritizedKeys = standardColumns.filter(key => allKeys.has(key));
  const otherKeys = Array.from(allKeys).filter(key => !standardColumns.includes(key)).sort();
  
  return [...prioritizedKeys, ...otherKeys];
};

const TestCasesDisplay: React.FC = () => {
  const {
    generatedTestCases,
    generateTestCasesError,
    isGeneratingTestCases,
    openRefactorDialog,
    openBulkRefactorDialog,
    // State for AI Review
    extractedRequirementsText, // Needed for AI Review API
    applicationContexts,     // Needed for AI Review API
    selectedProvider,        // Needed for AI Review API
    selectedModel,           // Needed for AI Review API
    apiCredentials,          // Needed for AI Review API
    openaiFallbackApiKey,    // Needed for AI Review API
    isPerformingAiReview,
    performAiReviewError,
    setAiReviewData,
    setIsPerformingAiReview,
    setPerformAiReviewError,
    clearAiReviewStateForApp,
    aiReviewData 
  } = useAppStore();

  const handleRefactorClick = (appName: string, tc: TestCase) => {
    const tcId = tc['Test Case ID'] || tc.id; // Prefer 'Test Case ID' if available
    if (tcId) {
      openRefactorDialog(appName, tcId, tc);
    } else {
      console.error("Cannot refactor: Test case ID is missing.", tc);
      // Optionally, show an alert to the user
    }
  };

  const handleAiReviewClick = async (appName: string) => {
    if (!extractedRequirementsText || !selectedProvider || !selectedModel) {
      setPerformAiReviewError(appName, "Please ensure requirements text is loaded and LLM is configured.");
      return;
    }

    const currentTestCasesForApp = generatedTestCases[appName];
    if (!Array.isArray(currentTestCasesForApp) || currentTestCasesForApp.length === 0) {
      setPerformAiReviewError(appName, "No generated test cases available for this app to review.");
      return;
    }
    
    clearAiReviewStateForApp(appName); // Clear previous review data/errors for this app
    setIsPerformingAiReview(appName, true);

    const contextStrings = applicationContexts[appName] || [];
    const additionalContextStr = contextStrings.join("\n\n---\n\n");

    const requestData: AIReviewRequest = {
      main_requirements_text: extractedRequirementsText,
      additional_context_str: additionalContextStr,
      existing_test_cases: currentTestCasesForApp,
      llm_provider: selectedProvider,
      model_name: selectedModel,
      api_credentials: apiCredentials,
      openai_fallback_api_key: openaiFallbackApiKey,
    };

    try {
      const response = await performAiReviewApi(appName, requestData);
      if (response.error) {
        setPerformAiReviewError(appName, response.error);
      } else if (response.review_results) {
        setAiReviewData(appName, response.review_results);
      } else {
        setPerformAiReviewError(appName, "AI Review failed: No review data returned.");
      }
    } catch (error: any) {
      console.error(`AI Review error for ${appName}:`, error);
      const errorMsg = error?.error || error?.detail || `Failed to perform AI review for ${appName}.`;
      setPerformAiReviewError(appName, errorMsg);
    } finally {
      // setIsPerformingAiReview(appName, false); // Handled by setAiReviewData or setPerformAiReviewError
    }
  };


  if (isGeneratingTestCases) {
    // Optionally show a loading indicator here too, or rely on the button's indicator
    return null; 
  }

  if (generateTestCasesError) {
    return (
      <Alert severity="error" sx={{ mt: 2 }}>
        Error generating test cases: {generateTestCasesError}
      </Alert>
    );
  }

  if (Object.keys(generatedTestCases).length === 0) {
    return (
      <Typography sx={{ mt: 2, fontStyle: 'italic' }} color="textSecondary">
        No test cases generated yet. Click "Generate Test Cases" after configuring.
      </Typography>
    );
  }

  return (
    <Box sx={{ mt: 3 }}>
      <Typography variant="h5" gutterBottom>Generated Test Cases</Typography>
      {Object.entries(generatedTestCases).map(([appName, result]) => {
        const testCasesForApp = Array.isArray(result) ? result as TestCase[] : null;
        const errorForApp = typeof result === 'string' ? result : null;
        const columns = testCasesForApp ? getTableColumns(testCasesForApp) : [];

        return (
          <Accordion key={appName} defaultExpanded sx={{ mb: 2 }}>
            <AccordionSummary 
              expandIcon={<ExpandMoreIcon />}
              sx={{ 
                '& .MuiAccordionSummary-content': { 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  alignItems: 'center',
                  width: '100%' 
                } 
              }}
            >
              <Typography variant="h6">{appName}</Typography>
              {testCasesForApp && testCasesForApp.length > 0 && ( // Only show if there are TCs to refactor
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<AutoFixHighIcon />}
                  onClick={(event) => {
                    event.stopPropagation(); // Prevent accordion from toggling
                    openBulkRefactorDialog(appName);
                  }}
                  sx={{ ml: 2 }}
                >
                  Bulk Refactor
                </Button>
              )}
              {testCasesForApp && testCasesForApp.length > 0 && (
                 <Button
                    variant="outlined"
                    size="small"
                    color="secondary"
                    startIcon={isPerformingAiReview[appName] ? <CircularProgress size={16} color="inherit" /> : <RateReviewIcon />}
                    onClick={(event) => {
                      event.stopPropagation();
                      handleAiReviewClick(appName);
                    }}
                    disabled={isPerformingAiReview[appName]}
                    sx={{ ml: 1 }}
                  >
                    AI Review
                  </Button>
              )}
            </AccordionSummary>
            <AccordionDetails sx={{ display: 'flex', flexDirection: 'column' }}>
              {performAiReviewError[appName] && (
                <Alert severity="error" sx={{ mb: 1 }}>{performAiReviewError[appName]}</Alert>
              )}
              {/* Replace placeholder with AIReviewDisplay component */}
              {aiReviewData[appName] && (
                <AIReviewDisplay appName={appName} />
              )}
              {errorForApp && (
                <Alert severity="warning" sx={{ width: '100%' }}>
                  Could not generate test cases for {appName}: {errorForApp}
                </Alert>
              )}
              {testCasesForApp && testCasesForApp.length > 0 && (
                <TableContainer component={Paper} elevation={2}>
                  <Table stickyHeader size="small">
                    <TableHead>
                      <TableRow>
                        {columns.map((colName) => (
                          <TableCell key={colName} sx={{ fontWeight: 'bold', backgroundColor: 'grey.200' }}>
                            {colName}
                          </TableCell>
                        ))}
                        <TableCell sx={{ fontWeight: 'bold', backgroundColor: 'grey.200', width: '80px' }}>Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {testCasesForApp.map((tc, index) => (
                        <TableRow key={tc['Test Case ID'] || `tc-${appName}-${index}`}>
                          {columns.map((colName) => (
                            <TableCell key={`${colName}-${index}`}>
                              {typeof tc[colName] === 'object' ? JSON.stringify(tc[colName]) : tc[colName]?.toString() || ''}
                            </TableCell>
                          ))}
                          <TableCell>
                            <Tooltip title="Refactor Test Case">
                              <IconButton size="small" onClick={() => handleRefactorClick(appName, tc)}>
                                <EditIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
              {testCasesForApp && testCasesForApp.length === 0 && (
                <Typography>No test cases were generated for {appName}.</Typography>
              )}
            </AccordionDetails>
          </Accordion>
        );
      })}
    </Box>
  );
};

export default TestCasesDisplay;
