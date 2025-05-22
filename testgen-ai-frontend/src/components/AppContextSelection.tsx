import React, { useState, useCallback } from 'react';
import {
  Box, Typography, Select, MenuItem, Chip, OutlinedInput, Checkbox, ListItemText,
  Button, LinearProgress, Alert, Paper, Grid, FormControl, InputLabel
} from '@mui/material';
import type { SelectChangeEvent } from '@mui/material'; // Type-only import
import UploadFileIcon from '@mui/icons-material/UploadFile';
import { useAppStore } from '../store';
import { uploadContextFileApi } from '../services/api';

interface AppContextUploaderProps {
  appName: string;
}

const SingleAppContextUploader: React.FC<AppContextUploaderProps> = ({ appName }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const addApplicationContext = useAppStore((state) => state.addApplicationContext);
  const applicationContexts = useAppStore((state) => state.applicationContexts);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
      setError(null);
      setSuccessMessage(null);
    }
  };

  const handleUpload = useCallback(async () => {
    if (!selectedFile) {
      setError('Please select a file first.');
      return;
    }
    setUploading(true);
    setError(null);
    setSuccessMessage(null);
    try {
      const response = await uploadContextFileApi(appName, selectedFile);
      if (response.error) {
        setError(response.error);
      } else if (response.extracted_text) {
        addApplicationContext(appName, response.extracted_text);
        setSuccessMessage(`Uploaded ${response.filename} (${Math.round(response.extracted_text.length / 1024)} KB)`);
        setSelectedFile(null); // Clear selection after successful upload
      } else {
        setError("Upload failed: No text extracted or error returned.");
      }
    } catch (err: any) {
      console.error(`Error uploading context for ${appName}:`, err);
      const errorMsg = err?.error || err?.detail || `Failed to upload context file for ${appName}.`;
      setError(errorMsg);
    } finally {
      setUploading(false);
    }
  }, [appName, selectedFile, addApplicationContext]);

  const currentAppContexts = applicationContexts[appName] || [];

  return (
    <Paper elevation={1} sx={{ p: 2, mb: 2 }}>
      <Typography variant="subtitle1" gutterBottom>Context for: {appName}</Typography>
      <Button
        variant="outlined"
        component="label"
        size="small"
        startIcon={<UploadFileIcon />}
        fullWidth
        disabled={uploading}
      >
        Select Context File
        <input type="file" hidden onChange={handleFileChange} accept=".txt,.md,.docx,.xlsx,.json,.yaml,.yml" />
      </Button>
      {selectedFile && (
        <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
          Selected: {selectedFile.name}
        </Typography>
      )}
      <Button
        variant="contained"
        onClick={handleUpload}
        disabled={!selectedFile || uploading}
        fullWidth
        size="small"
        sx={{ mt: 1 }}
      >
        {uploading ? 'Uploading...' : 'Upload Context'}
      </Button>
      {uploading && <LinearProgress sx={{ mt: 1 }} />}
      {error && <Alert severity="error" sx={{ mt: 1, fontSize: '0.8rem', p: '0px 8px' }}>{error}</Alert>}
      {successMessage && <Alert severity="success" sx={{ mt: 1, fontSize: '0.8rem', p: '0px 8px' }}>{successMessage}</Alert>}
      {currentAppContexts.length > 0 && (
        <Box sx={{mt:1}}>
          <Typography variant="caption">{currentAppContexts.length} context document(s) added.</Typography>
        </Box>
      )}
    </Paper>
  );
};


const AppContextSelection: React.FC = () => {
  const {
    identifiedApplications,
    selectedApplications,
    setSelectedApplications,
  } = useAppStore();

  const handleSelectionChange = (event: SelectChangeEvent<string[]>) => {
    const { target: { value } } = event;
    setSelectedApplications(typeof value === 'string' ? value.split(',') : value);
  };

  if (identifiedApplications.length === 0) {
    return (
      <Typography sx={{mt: 2, fontStyle: 'italic'}} color="textSecondary">
        No applications identified yet. Please run "Identify Applications" first.
      </Typography>
    );
  }

  return (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h6" gutterBottom>2. Select Applications & Upload Context (Optional)</Typography>
      <FormControl fullWidth margin="normal">
        <InputLabel id="select-apps-label">Select Applications</InputLabel>
        <Select
          labelId="select-apps-label"
          multiple
          value={selectedApplications}
          onChange={handleSelectionChange}
          input={<OutlinedInput label="Select Applications" />}
          renderValue={(selected) => (
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
              {(selected as string[]).map((value) => (
                <Chip key={value} label={value} size="small" />
              ))}
            </Box>
          )}
        >
          {identifiedApplications.map((name) => (
            <MenuItem key={name} value={name}>
              <Checkbox checked={selectedApplications.indexOf(name) > -1} />
              <ListItemText primary={name} />
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {selectedApplications.length > 0 && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle1" gutterBottom>Upload Context Files for Selected Applications:</Typography>
          <Grid container spacing={2}>
            {selectedApplications.map((appName) => (
              <Grid item xs={12} sm={6} md={4} key={appName}>
                <SingleAppContextUploader appName={appName} />
              </Grid>
            ))}
          </Grid>
        </Box>
      )}
    </Box>
  );
};

export default AppContextSelection;
