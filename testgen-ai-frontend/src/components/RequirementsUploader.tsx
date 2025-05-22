import React, { useState, useCallback } from 'react';
import { Box, Button, Typography, LinearProgress, Alert } from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import { useAppStore } from '../store'; // Assuming store will hold extracted text
import apiClient from '../services/api'; // Using the configured apiClient

interface RequirementsUploaderProps {
  // Props if needed, e.g., onUploadSuccess callback
}

const RequirementsUploader: React.FC<RequirementsUploaderProps> = () => {
  const setExtractedRequirementsText = useAppStore((state) => state.setExtractedRequirementsText);
  
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadSuccessMessage, setUploadSuccessMessage] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      if (file.name.toLowerCase().endsWith('.docx')) {
        setSelectedFile(file);
        setError(null);
        setUploadSuccessMessage(null);
      } else {
        setSelectedFile(null);
        setError('Invalid file type. Only .docx files are accepted.');
        setUploadSuccessMessage(null);
      }
    }
  };

  const handleUpload = useCallback(async () => {
    if (!selectedFile) {
      setError('Please select a .docx file first.');
      return;
    }

    setUploading(true);
    setError(null);
    setUploadSuccessMessage(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await apiClient.post('/upload/requirements', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      if (response.data && response.data.extracted_text) {
        setExtractedRequirementsText(response.data.extracted_text); 
        console.log('Extracted Text (first 200 chars):', response.data.extracted_text.substring(0, 200) + '...');
        setUploadSuccessMessage(`Successfully uploaded and extracted text from ${response.data.filename}.`);
      } else if (response.data && response.data.error) {
        setError(`Upload failed: ${response.data.error}`);
        setExtractedRequirementsText(null);
      } else {
        setError('Upload failed: No extracted text or error message received.');
        setExtractedRequirementsText(null);
      }
    } catch (err: any) {
      console.error('Upload error:', err);
      if (err.response && err.response.data && err.response.data.detail) {
        setError(`Upload error: ${err.response.data.detail}`);
      } else {
        setError('Upload error: Could not connect to the server or an unexpected error occurred.');
      }
      setExtractedRequirementsText(null);
    } finally {
      setUploading(false);
    }
  }, [selectedFile, setExtractedRequirementsText]);

  return (
    <Box sx={{ width: '100%'}}>
      <Button
        variant="contained"
        component="label"
        startIcon={<UploadFileIcon />}
        fullWidth
        disabled={uploading}
      >
        Select .DOCX File
        <input type="file" hidden accept=".docx" onChange={handleFileChange} />
      </Button>
      {selectedFile && (
        <Typography variant="body2" sx={{ mt: 1 }}>
          Selected: {selectedFile.name}
        </Typography>
      )}
      <Button
        variant="outlined"
        onClick={handleUpload}
        disabled={!selectedFile || uploading}
        fullWidth
        sx={{ mt: 1 }}
      >
        {uploading ? 'Uploading...' : 'Upload & Extract Text'}
      </Button>
      {uploading && <LinearProgress sx={{ mt: 1 }} />}
      {error && <Alert severity="error" sx={{ mt: 1 }}>{error}</Alert>}
      {uploadSuccessMessage && <Alert severity="success" sx={{ mt: 1 }}>{uploadSuccessMessage}</Alert>}
    </Box>
  );
};

export default RequirementsUploader;
