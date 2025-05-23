import type React from 'react'; 
import { 
  Box, Typography, Paper, Button, CircularProgress, Alert,
  RadioGroup, FormControlLabel, Radio, Divider 
  // Removed Fragment, Grid, Tooltip, IconButton, CheckCircleIcon, CancelIcon as they are not used yet
} from '@mui/material';
import { useAppStore } from '../store';
import { applyAiReviewApi } from '../services/api';
import type { ApplyAIReviewRequest, TestCase } from '../types'; // Added TestCase

interface AIReviewDisplayProps {
  appName: string;
}

const AIReviewDisplay: React.FC<AIReviewDisplayProps> = ({ appName }) => {
  const {
    aiReviewData,
    aiReviewUserDecisions,
    setAiReviewUserDecision, // To be used by decision inputs
    // For Apply Changes button:
    generatedTestCases,
    isApplyingAiReview,
    applyAiReviewError,
    setIsApplyingAiReview,
    setApplyAiReviewError,
    updateTestCasesAfterAiReview,
    // clearAiReviewUserDecisionsForApp, // This action is not used in the current simplified version
  } = useAppStore();

  const reviewDataForApp = aiReviewData[appName];
  const decisionsForApp = aiReviewUserDecisions[appName] || {};

  const handleApplyChanges = async () => {
    if (!reviewDataForApp || Object.keys(decisionsForApp).length === 0) {
      // Or if no decisions are "accept" etc. - more complex validation needed here
      setApplyAiReviewError(appName, "No review data or decisions to apply.");
      return;
    }
    const currentTestCasesForApp = generatedTestCases[appName];
    if (!Array.isArray(currentTestCasesForApp)) {
        setApplyAiReviewError(appName, "Original test cases not found for this app.");
        return;
    }

    setIsApplyingAiReview(appName, true);
    setApplyAiReviewError(appName, null);

    const requestData: ApplyAIReviewRequest = {
      existing_test_cases: currentTestCasesForApp,
      ai_review_suggestions_processed: reviewDataForApp,
      user_decisions: decisionsForApp,
    };

    try {
      const response = await applyAiReviewApi(appName, requestData);
      if (response.error) {
        setApplyAiReviewError(appName, response.error);
      } else {
        updateTestCasesAfterAiReview(appName, response.updated_test_cases);
        // clearAiReviewUserDecisionsForApp(appName); // Clear decisions after successful apply
        // Optionally, also clear aiReviewData[appName] or hide this component
      }
    } catch (error: any) {
      console.error(`Error applying AI review for ${appName}:`, error);
      const errorMsg = error?.error || error?.detail || `Failed to apply AI review changes for ${appName}.`;
      setApplyAiReviewError(appName, errorMsg);
    } finally {
      // setIsApplyingAiReview(appName, false); // Handled by store actions
    }
  };


  if (!reviewDataForApp) {
    return <Typography sx={{ fontStyle: 'italic', my: 1 }}>No AI review data available for {appName}.</Typography>;
  }

  // TODO: Implement detailed UI for modified_test_cases_suggestions and identified_duplicates

  const renderDecisionControls = (suggestionType: string, suggestionId: string, currentValue?: string) => {
    const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
      setAiReviewUserDecision(appName, suggestionId, (event.target as HTMLInputElement).value);
    };

    return (
      <RadioGroup
        row
        aria-labelledby={`decision-group-label-${suggestionId}`}
        name={`decision-group-${suggestionId}`}
        value={currentValue || 'pending'}
        onChange={handleChange}
      >
        <FormControlLabel value="accept" control={<Radio size="small" color="success" />} label={<Typography variant="caption">Accept</Typography>} />
        <FormControlLabel value="reject" control={<Radio size="small" color="error" />} label={<Typography variant="caption">Reject</Typography>} />
        <FormControlLabel value="pending" control={<Radio size="small" />} label={<Typography variant="caption">Pending</Typography>} />
      </RadioGroup>
    );
  };


  return (
    <Paper elevation={1} sx={{ p: 2, my: 2, backgroundColor: 'aliceblue' }}>
      <Typography variant="h6" gutterBottom>AI Review Suggestions for {appName}</Typography>
      
      <Box sx={{ mb: 2 }}>
        <Typography variant="subtitle1">Coverage Summary:</Typography>
        <Typography variant="body2" sx={{whiteSpace: 'pre-wrap'}}>{reviewDataForApp.coverage_summary || "Not provided."}</Typography>
      </Box>
      <Divider sx={{my:2}}/>

      {/* UI for newly_suggested_test_cases */}
      {reviewDataForApp.newly_suggested_test_cases?.length > 0 && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle1" gutterBottom>
            Newly Suggested Test Cases ({reviewDataForApp.newly_suggested_test_cases.length})
          </Typography>
          {reviewDataForApp.newly_suggested_test_cases.map((tc_data: any, index: number) => {
            const suggestionId = `new_${index}`; // Simple ID for new suggestions
            return (
              <Paper key={suggestionId} variant="outlined" sx={{ p: 1.5, mb: 1.5 }}>
                <Typography variant="body2" gutterBottom component="div">
                  <strong>Suggested TC {index + 1}:</strong> 
                  <em>{tc_data['Test Case Name'] || 'N/A'}</em>
                </Typography>
                <pre style={{whiteSpace: 'pre-wrap', wordBreak: 'break-all', maxHeight: '150px', overflowY: 'auto', fontSize: '0.75rem', backgroundColor: '#f9f9f9', padding: '8px', borderRadius: '4px'}}>
                  {JSON.stringify(tc_data, null, 2)}
                </pre>
                {renderDecisionControls('new', suggestionId, decisionsForApp[suggestionId])}
              </Paper>
            );
          })}
        </Box>
      )}
      <Divider sx={{my:2}}/>

      {/* UI for modified_test_cases_suggestions */}
      {reviewDataForApp.modified_test_cases_suggestions?.length > 0 && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle1" gutterBottom>
            Modification Suggestions ({reviewDataForApp.modified_test_cases_suggestions.length})
          </Typography>
          {reviewDataForApp.modified_test_cases_suggestions.map((mod_suggestion: any, index: number) => {
            const originalTcId = mod_suggestion.original_test_case_id;
            const suggestionId = `mod_${originalTcId || index}`; // Use original TC ID if available
            return (
              <Paper key={suggestionId} variant="outlined" sx={{ p: 1.5, mb: 1.5 }}>
                <Typography variant="body2" gutterBottom component="div">
                  <strong>Suggestion for TC ID:</strong> {originalTcId || 'N/A'}
                </Typography>
                {mod_suggestion.reason_for_change && (
                  <Typography variant="caption" display="block" sx={{ fontStyle: 'italic', mb: 1 }}>
                    Reason: {mod_suggestion.reason_for_change}
                  </Typography>
                )}
                <Typography variant="caption" display="block">Suggested Change:</Typography>
                <pre style={{whiteSpace: 'pre-wrap', wordBreak: 'break-all', maxHeight: '150px', overflowY: 'auto', fontSize: '0.75rem', backgroundColor: '#f9f9f9', padding: '8px', borderRadius: '4px'}}>
                  {JSON.stringify(mod_suggestion.suggested_test_case_data, null, 2)}
                </pre>
                {renderDecisionControls('modified', suggestionId, decisionsForApp[suggestionId])}
              </Paper>
            );
          })}
        </Box>
      )}
      <Divider sx={{my:2}}/>

      {/* UI for identified_duplicates */}
      {reviewDataForApp.identified_duplicates?.length > 0 && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle1" gutterBottom>
            Identified Duplicates ({reviewDataForApp.identified_duplicates.length} groups)
          </Typography>
          {reviewDataForApp.identified_duplicates.map((dup_group: any, groupIndex: number) => {
            const groupId = dup_group.duplicate_group_id || `group_${groupIndex}`;
            const suggestionId = `dup_${groupId}`;
            const tcIdsInGroup = dup_group.test_case_ids || [];

            // Function to get a snippet of a TC for display
            const getTcSummary = (tcId: string) => {
                const appTcs = generatedTestCases[appName] as TestCase[];
                if (!Array.isArray(appTcs)) return `TC ID: ${tcId} (Data not found)`;
                const tcData = appTcs.find(tc => (tc['Test Case ID'] || tc.id) === tcId);
                return tcData ? `${tcId}: ${tcData['Test Case Name'] || 'Unnamed Test Case'}`.substring(0, 100) + '...' : `TC ID: ${tcId} (Data not found)`;
            };
            
            const handleDuplicateDecisionChange = (event: React.ChangeEvent<HTMLInputElement>) => {
              setAiReviewUserDecision(appName, suggestionId, (event.target as HTMLInputElement).value);
            };

            return (
              <Paper key={suggestionId} variant="outlined" sx={{ p: 1.5, mb: 1.5 }}>
                <Typography variant="body2" gutterBottom component="div">
                  <strong>Duplicate Group:</strong> {groupId}
                </Typography>
                {dup_group.reason_for_duplication && (
                  <Typography variant="caption" display="block" sx={{ fontStyle: 'italic', mb: 1 }}>
                    Reason: {dup_group.reason_for_duplication}
                  </Typography>
                )}
                <Typography variant="caption" display="block" sx={{mb:1}}>Select which Test Case to keep:</Typography>
                <RadioGroup
                  aria-labelledby={`dup-decision-group-label-${suggestionId}`}
                  name={`dup-decision-group-${suggestionId}`}
                  value={decisionsForApp[suggestionId] || ''} // Default to empty if no decision made
                  onChange={handleDuplicateDecisionChange}
                >
                  {tcIdsInGroup.map((tcId: string) => (
                    <FormControlLabel 
                      key={tcId} 
                      value={tcId} 
                      control={<Radio size="small" />} 
                      label={<Typography variant="caption" title={getTcSummary(tcId)}>{getTcSummary(tcId)}</Typography>} 
                    />
                  ))}
                  <FormControlLabel 
                    value="resolve_later" // A value to indicate no action or resolve later
                    control={<Radio size="small" />} 
                    label={<Typography variant="caption" sx={{fontStyle: 'italic'}}>Resolve Later / No Action</Typography>} 
                  />
                </RadioGroup>
              </Paper>
            );
          })}
        </Box>
      )}
      
      <Button
        variant="contained"
        color="primary"
        onClick={handleApplyChanges}
        disabled={isApplyingAiReview[appName] || Object.keys(decisionsForApp).length === 0} // Enable only if decisions made
        sx={{ mt: 2 }}
        startIcon={isApplyingAiReview[appName] ? <CircularProgress size={20} color="inherit" /> : null}
      >
        {isApplyingAiReview[appName] ? 'Applying...' : 'Apply Decided Changes'}
      </Button>
      {applyAiReviewError[appName] && (
        <Alert severity="error" sx={{ mt: 1 }}>{applyAiReviewError[appName]}</Alert>
      )}
    </Paper>
  );
};

export default AIReviewDisplay;
