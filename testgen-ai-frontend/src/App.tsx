import React from 'react';
// Removed List, SettingsIcon, ListItemButton, ListItemText, ListItemIcon, UploadFileIcon
import { Box, CssBaseline, AppBar, Toolbar, Typography, Drawer, Divider, ThemeProvider, createTheme } from '@mui/material'; 
import DescriptionIcon from '@mui/icons-material/Description';
import LLMConfigSidebar from './components/LLMConfigSidebar';
import RequirementsUploader from './components/RequirementsUploader';
import IdentifyApplications from './components/IdentifyApplications';
import AppContextSelection from './components/AppContextSelection';
import GenerateTestCasesButton from './components/GenerateTestCasesButton'; // Import the new component
import { useAppStore } from './store'; // Import the store

const drawerWidth = 300; // Increased drawer width

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
            <Typography variant="h6" noWrap component="div">
              TestGen AI 🧪
            </Typography>
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
          
          {/* Further main content will go here - e.g., results display, etc. */}
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
