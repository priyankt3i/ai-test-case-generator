import React from 'react';
import { Box, CssBaseline, AppBar, Toolbar, Typography, Drawer, Divider, ThemeProvider, createTheme, IconButton, Tooltip } from '@mui/material'; 
import DescriptionIcon from '@mui/icons-material/Description';
// It's better to use MUI icons if available for consistency, or ensure SVGs are handled correctly.
// For now, using img tags for external SVGs as per svg.yaml.
// import LinkedInIcon from '@mui/icons-material/LinkedIn';
// import GitHubIcon from '@mui/icons-material/GitHub';
// import LanguageIcon from '@mui/icons-material/Language';
import LLMConfigSidebar from './components/LLMConfigSidebar';
import RequirementsUploader from './components/RequirementsUploader';
import IdentifyApplications from './components/IdentifyApplications';
import AppContextSelection from './components/AppContextSelection';
import GenerateTestCasesButton from './components/GenerateTestCasesButton';
import TestCasesDisplay from './components/TestCasesDisplay';
import RefactorTestCaseDialog from './components/RefactorTestCaseDialog';
import BulkRefactorDialog from './components/BulkRefactorDialog';
import { useAppStore } from './store'; 
import { exportToExcelApi } from './services/api'; // For export
import type { ExportRequest } from './types'; // For export
import FileDownloadIcon from '@mui/icons-material/FileDownload'; // For export button
import { Alert as MuiAlert, CircularProgress as MuiCircularProgress, Button as MuiButton } from '@mui/material'; // Aliasing for clarity if needed

const drawerWidth = 300; 

// A simple theme for now
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2', // A standard blue
    },
    secondary: {
      main: '#dc004e', // A standard pink
    },
    background: {
      default: '#f4f6f8', // Light grey background
      paper: '#ffffff',   // White for paper elements like Drawer
    }
  },
  typography: {
    fontFamily: 'Roboto, Arial, sans-serif',
    h6: {
      fontWeight: 600,
    }
  },
});

// Placeholder for Sidebar content is now LLMConfigSidebar, other sidebar items can be added around it or within it.

function App() {
  const extractedRequirementsText = useAppStore((state) => state.extractedRequirementsText);
  const generatedTestCases = useAppStore((state) => state.generatedTestCases);
  const isExportingToExcel = useAppStore((state) => state.isExportingToExcel);
  const exportToExcelError = useAppStore((state) => state.exportToExcelError);
  const setIsExportingToExcel = useAppStore((state) => state.setIsExportingToExcel);
  const setExportToExcelError = useAppStore((state) => state.setExportToExcelError);

  const handleExportToExcel = async () => {
    setIsExportingToExcel(true);
    setExportToExcelError(null);

    const validTestCasesData: Record<string, Array<Record<string, any>>> = {};
    Object.entries(generatedTestCases).forEach(([appName, tcs]) => {
      if (Array.isArray(tcs) && tcs.length > 0) {
        validTestCasesData[appName] = tcs;
      }
    });

    if (Object.keys(validTestCasesData).length === 0) {
      setExportToExcelError("No valid test cases available to export.");
      setIsExportingToExcel(false);
      return;
    }

    const requestData: ExportRequest = {
      test_cases_data: validTestCasesData,
      filename: `test_cases_export_${new Date().toISOString().split('T')[0]}.xlsx`
    };

    try {
      const blob = await exportToExcelApi(requestData);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', requestData.filename || 'test_cases.xlsx');
      document.body.appendChild(link);
      link.click();
      link.parentNode?.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error: any) {
      console.error("Export to Excel error:", error);
      // If the error is a blob, it might be a JSON error from the backend
      if (error instanceof Blob && error.type === "application/json") {
        const errText = await error.text();
        try {
          const errJson = JSON.parse(errText);
          setExportToExcelError(errJson.detail || "Failed to export to Excel. Backend error.");
        } catch (parseError) {
          setExportToExcelError("Failed to export to Excel and could not parse error response.");
        }
      } else if (error.detail) { // From Axios error rethrow
         setExportToExcelError(error.detail);
      }
      else {
        setExportToExcelError("Failed to export to Excel. An unexpected error occurred.");
      }
    } finally {
      setIsExportingToExcel(false);
    }
  };


  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ display: 'flex' }}>
        <CssBaseline />
        <AppBar
          position="fixed"
          sx={{ width: `calc(100% - ${drawerWidth}px)`, ml: `${drawerWidth}px`, zIndex: (theme) => theme.zIndex.drawer + 1 }}
        >
          <Toolbar>
            <DescriptionIcon sx={{mr: 1}}/>
            <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
              TestGen AI 🧪
            </Typography>
            
            {/* Social Icons */}
            <Tooltip title="GitHub Profile">
              <IconButton 
                color="inherit" 
                href="https://github.com/priyankt3i" 
                target="_blank" 
                rel="noopener noreferrer"
                aria-label="GitHub Profile"
              >
                <img src="https://www.svgrepo.com/show/512317/github-142.svg" alt="GitHub" style={{ height: 24, width: 24, filter: 'invert(100%)' }} />
              </IconButton>
            </Tooltip>
            <Tooltip title="LinkedIn Profile">
              <IconButton 
                color="inherit" 
                href="https://www.linkedin.com/in/priyankt3i/" 
                target="_blank" 
                rel="noopener noreferrer"
                aria-label="LinkedIn Profile"
              >
                 <img src="https://www.svgrepo.com/show/521725/linkedin.svg" alt="LinkedIn" style={{ height: 24, width: 24, filter: 'invert(100%)' }} />
              </IconButton>
            </Tooltip>
            <Tooltip title="Personal Website">
              <IconButton 
                color="inherit" 
                href="https://kumarpriyank.com/" 
                target="_blank" 
                rel="noopener noreferrer"
                aria-label="Personal Website"
              >
                 <img src="https://www.svgrepo.com/show/512318/website-142.svg" alt="Website" style={{ height: 24, width: 24, filter: 'invert(100%)' }} />
              </IconButton>
            </Tooltip>

          </Toolbar>
        </AppBar>
        <Drawer
          sx={{
            width: drawerWidth,
            flexShrink: 0,
            '& .MuiDrawer-paper': {
              width: drawerWidth,
              boxSizing: 'border-box',
              backgroundColor: theme.palette.background.paper, // Ensure drawer bg
            },
          }}
          variant="permanent"
          anchor="left"
        >
          <Toolbar /> {/* For spacing under AppBar */}
          <Box sx={{ display: 'flex', alignItems: 'center', padding: theme.spacing(0, 1), ...theme.mixins.toolbar, justifyContent: 'center' }}>
             {/* You can put a logo here if you have one */}
             <Typography variant="h5" component="div" sx={{fontWeight: 'bold'}}>
                📄 TestGen AI
             </Typography>
          </Box>
          <Divider />
          {/* Replace SidebarContent with LLMConfigSidebar */}
          {/* Other sidebar elements like file upload will be added here or integrated into a main sidebar component */}
          <LLMConfigSidebar /> 
          {/* Placeholder for other sidebar sections like file upload */}
          <Divider sx={{ my: 1 }} />
          <Box sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Document Processing
            </Typography>
            {/* Replace placeholder with the actual uploader component */}
            <RequirementsUploader />
          </Box>
          <Divider sx={{ my: 1 }} />
           <Box sx={{p:2, mt: 'auto'}}> {/* Push to bottom */}
            <Typography variant="caption" color="textSecondary">
            ℹ️ AI results may require review. Always validate generated test cases.
            </Typography>
          </Box>
        </Drawer>
        <Box
          component="main"
          sx={{ flexGrow: 1, bgcolor: 'background.default', p: 3 }}
        >
          <Toolbar /> {/* Necessary for content to be below app bar */}
          <Typography variant="h4" gutterBottom>
            Convert Business Requirements to Test Cases
          </Typography>
          <Typography paragraph>
            Welcome to the TestGen AI application. Use the sidebar to configure your LLM provider,
            upload your requirements document, and then proceed with generating test cases.
          </Typography>
          
          {extractedRequirementsText && (
            <Box sx={{ mt: 2, p: 2, border: '1px solid #ccc', borderRadius: 1, whiteSpace: 'pre-wrap', maxHeight: '400px', overflowY: 'auto', backgroundColor: '#fff' }}>
              <Typography variant="h6" gutterBottom>Extracted Requirements Text:</Typography>
              <Typography component="pre" sx={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                {extractedRequirementsText}
              </Typography>
            </Box>
          )}

          {/* Add IdentifyApplications component here */}
          {extractedRequirementsText && ( 
            <IdentifyApplications />
          )}

          {/* Add AppContextSelection component here */}
          {/* This should likely be shown if identifiedApplications has items */}
          <AppContextSelection />

          {/* Add GenerateTestCasesButton component here */}
          {/* This should be shown if apps are selected, etc. The button itself handles its disabled state. */}
          <GenerateTestCasesButton />

          {/* Add TestCasesDisplay component here */}
          <TestCasesDisplay />

          {Object.keys(generatedTestCases).length > 0 && (
            <Box sx={{ mt: 3, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <MuiButton
                variant="contained"
                color="success"
                startIcon={isExportingToExcel ? <MuiCircularProgress size={20} color="inherit" /> : <FileDownloadIcon />}
                onClick={handleExportToExcel}
                disabled={isExportingToExcel || Object.values(generatedTestCases).every(val => typeof val === 'string' || val.length === 0)}
                sx={{minWidth: '200px'}}
              >
                {isExportingToExcel ? 'Exporting...' : 'Export All to Excel'}
              </MuiButton>
              {exportToExcelError && (
                <MuiAlert severity="error" sx={{ mt: 1, width: '100%', maxWidth: '600px' }}>
                  {exportToExcelError}
                </MuiAlert>
              )}
            </Box>
          )}
          
          {/* Further main content will go here - e.g., results display, etc. */}
        </Box>
        {/* Globally rendered dialogs */}
        <RefactorTestCaseDialog />
        <BulkRefactorDialog />
      </Box>
    </ThemeProvider>
  );
}

export default App;
