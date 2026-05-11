import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  Slider,
  Card,
  CardMedia,
  CardContent,
  CircularProgress,
  Alert,
  Chip,
  Container,
  Switch,
  FormControlLabel,
  Fade,
  LinearProgress,
  Avatar,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Rating,
  } from '@mui/material';
import {
  CloudUpload,
  AutoFixHigh,
  Download,
  Brightness4,
  Brightness7,
  Timeline,
  Speed,
  PhotoCamera,
  Analytics,
  CheckCircle
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { createTheme, ThemeProvider } from '@mui/material/styles';

const ImageProcessor: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [originalImage, setOriginalImage] = useState<string>('');
  const [processedImage, setProcessedImage] = useState<string>('');
  const [saturationLevel, setSaturationLevel] = useState<number>(5); // Start from 5
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [metadata, setMetadata] = useState<any>(null);
  const [error, setError] = useState<string>('');
  const [darkMode, setDarkMode] = useState<boolean>(false);
  const [kpiData, setKpiData] = useState<any>(null);
  const [showSatisfactionDialog, setShowSatisfactionDialog] = useState<boolean>(false);
  const [satisfactionRating, setSatisfactionRating] = useState<number>(5);
  const [lastProcessedImage, setLastProcessedImage] = useState<string>('');
  const [predictedLevel, setPredictedLevel] = useState<number>(5);

  // Theme configuration
  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: '#1976d2',
      },
      secondary: {
        main: '#dc004e',
      },
      background: {
        default: darkMode ? '#121212' : '#f5f5f5',
        paper: darkMode ? '#1e1e1e' : '#ffffff',
      },
    },
  });

  // Fetch KPI data
  useEffect(() => {
    const fetchKPI = async () => {
      try {
        const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';
        const response = await axios.get(`${API_BASE_URL}/api/kpi-data`);
        setKpiData(response.data);
      } catch (err) {
        console.error('Failed to fetch KPI data:', err);
      }
    };
    fetchKPI();
    
    // Refresh KPI data every 5 seconds for real-time updates
    const interval = setInterval(fetchKPI, 5000);
    return () => clearInterval(interval);
  }, []);

  // Handle satisfaction submission
  const handleSatisfactionSubmit = async () => {
    try {
      const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';
      await axios.post(`${API_BASE_URL}/api/satisfaction`, {
        rating: satisfactionRating,
        timestamp: new Date().toISOString()
      });
      setShowSatisfactionDialog(false);
      setSatisfactionRating(5);
    } catch (err) {
      console.error('Failed to submit satisfaction:', err);
    }
  };

  const handleAutoEnhance = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setError('');
    
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';
      const response = await axios.post(`${API_BASE_URL}/api/auto-enhance-dng`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.success) {
        setOriginalImage(response.data.original_image);
        setProcessedImage(response.data.enhanced_image);
        setLastProcessedImage(response.data.enhanced_image);
        setMetadata(response.data);
        setPredictedLevel(response.data.predicted_level);
        setSaturationLevel(response.data.predicted_level); // Update slider dynamically
        setShowSatisfactionDialog(true);
      } else {
        setError('Auto-enhancement failed');
      }
    } catch (err: any) {
      setError(err.response?.data?.error || 'Auto-enhancement failed');
    } finally {
      setIsProcessing(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'image/dng': ['.dng'],
      'image/tiff': ['.tiff', '.tif'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png']
    },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      setError('');
      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0];
        setSelectedFile(file);
      }
    }
  });

  const handleProcessImage = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setError('');
    
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('level', saturationLevel.toString());

      const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';
      const response = await axios.post(`${API_BASE_URL}/api/process-dng`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.success) {
        setOriginalImage(response.data.original_image);
        setProcessedImage(response.data.enhanced_image);
        setLastProcessedImage(response.data.enhanced_image);
        setMetadata(response.data);
        setShowSatisfactionDialog(true);
      } else {
        setError('Processing failed');
      }
    } catch (err: any) {
      setError(err.response?.data?.error || 'Processing failed');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <Container maxWidth="xl">
        {/* Header with Dark Mode Toggle */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Box>
            <Typography variant="h4" gutterBottom>
              PrismVI25 DNG Processor
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Precision Saturation Enhancement System
            </Typography>
          </Box>
          <FormControlLabel
            control={
              <Switch
                checked={darkMode}
                onChange={(e) => setDarkMode(e.target.checked)}
                icon={<Brightness7 />}
                checkedIcon={<Brightness4 />}
              />
            }
            label={darkMode ? "Dark Mode" : "Light Mode"}
          />
        </Box>

        {/* KPI Dashboard */}
        {kpiData && (
          <Fade in={!!kpiData}>
            <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Analytics />
                Performance Dashboard
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                <Card sx={{ flex: '1 1 200px', textAlign: 'center', p: 2, minWidth: '200px' }}>
                  <Avatar sx={{ bgcolor: 'primary.main', mx: 'auto', mb: 1 }}>
                    <Speed />
                  </Avatar>
                  <Typography variant="h6">
                    {kpiData.saturation_accuracy}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Saturation Accuracy
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={kpiData.saturation_accuracy} 
                    sx={{ mt: 1 }}
                  />
                </Card>
                <Card sx={{ flex: '1 1 200px', textAlign: 'center', p: 2, minWidth: '200px' }}>
                  <Avatar sx={{ bgcolor: 'secondary.main', mx: 'auto', mb: 1 }}>
                    <Timeline />
                  </Avatar>
                  <Typography variant="h6">
                    {kpiData.processing_time}s
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Avg Processing Time
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={(1 - kpiData.processing_time / 2) * 100} 
                    sx={{ mt: 1 }}
                  />
                </Card>
                <Card sx={{ flex: '1 1 200px', textAlign: 'center', p: 2, minWidth: '200px' }}>
                  <Avatar sx={{ bgcolor: 'success.main', mx: 'auto', mb: 1 }}>
                    <CheckCircle />
                  </Avatar>
                  <Typography variant="h6">
                    {kpiData.user_satisfaction}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    User Satisfaction
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={kpiData.user_satisfaction} 
                    sx={{ mt: 1 }}
                  />
                </Card>
                <Card sx={{ flex: '1 1 200px', textAlign: 'center', p: 2, minWidth: '200px' }}>
                  <Avatar sx={{ bgcolor: 'info.main', mx: 'auto', mb: 1 }}>
                    <PhotoCamera />
                  </Avatar>
                  <Typography variant="h6">
                    {kpiData.images_processed}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Images Processed
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={Math.min(kpiData.images_processed / 20, 1) * 100} 
                    sx={{ mt: 1 }}
                  />
                </Card>
              </Box>
            </Paper>
          </Fade>
        )}

      <Box sx={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap', gap: 3 }}>
        {/* Upload Section */}
        <Box sx={{ flex: '1 1 300px', minWidth: '300px' }}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Upload DNG Image
            </Typography>
            
            <Box
              {...getRootProps()}
              sx={{
                border: '2px dashed',
                borderColor: isDragActive ? 'primary.main' : 'grey.300',
                borderRadius: 2,
                p: 4,
                textAlign: 'center',
                cursor: 'pointer',
                bgcolor: isDragActive ? 'action.hover' : 'background.paper',
                transition: 'all 0.2s ease-in-out'
              }}
            >
              <input {...getInputProps()} />
              <CloudUpload sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6">
                {isDragActive ? 'Drop the file here' : 'Drag & drop a DNG file here'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                or click to browse
              </Typography>
            </Box>

            {selectedFile && (
              <Box sx={{ mt: 2 }}>
                <Chip 
                  label={`Selected: ${selectedFile.name}`} 
                  color="primary" 
                  variant="outlined"
                />
              </Box>
            )}

            {error && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {error}
              </Alert>
            )}
          </Paper>
        </Box>

        {/* Controls Section */}
        <Box sx={{ flex: '1 1 300px', minWidth: '300px' }}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Saturation Enhancement Controls
            </Typography>
            
            <Box sx={{ mb: 3 }}>
              <Typography variant="body2" gutterBottom>
                Enhancement Level: {saturationLevel}/10
              </Typography>
              <Slider
                value={saturationLevel}
                onChange={(e, value) => setSaturationLevel(value as number)}
                min={5}
                max={10}
                marks={[
                  { value: 5, label: '5' },
                  { value: 6, label: '6' },
                  { value: 7, label: '7' },
                  { value: 8, label: '8' },
                  { value: 9, label: '9' },
                  { value: 10, label: '10' }
                ]}
                valueLabelDisplay="auto"
                sx={{ mb: 2 }}
              />
              <Typography variant="caption" color="text.secondary">
                * 5-10 levels of saturation enhancement as per project requirements
              </Typography>
            </Box>

            <Button
              variant="contained"
              size="large"
              startIcon={<AutoFixHigh />}
              onClick={handleProcessImage}
              disabled={!selectedFile || isProcessing}
              fullWidth
              sx={{ mb: 2 }}
            >
              {isProcessing ? 'Processing...' : 'Apply Enhancement'}
            </Button>

            <Button
              variant="outlined"
              size="large"
              startIcon={<AutoFixHigh />}
              onClick={handleAutoEnhance}
              disabled={!selectedFile || isProcessing}
              fullWidth
              color="secondary"
            >
              {isProcessing ? 'Processing...' : 'Auto Enhancement (AI)'}
            </Button>

            {isProcessing && (
              <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
                <CircularProgress />
              </Box>
            )}
          </Paper>
        </Box>
      </Box>

      {/* Image Comparison */}
      <Box sx={{ mt: 3 }}>
        <Paper elevation={3} sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Image Comparison
          </Typography>
          
          <Box sx={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap', gap: 2 }}>
            <Box sx={{ flex: '1 1 300px', minWidth: '300px' }}>
              <Card>
                <CardContent>
                  <Typography variant="subtitle1" gutterBottom>
                    Original
                  </Typography>
                </CardContent>
                {originalImage ? (
                  <CardMedia
                    component="img"
                    image={originalImage}
                    alt="Original"
                    sx={{ height: 300, objectFit: 'contain' }}
                  />
                ) : (
                  <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: 'grey.100' }}>
                    <Typography color="text.secondary">Upload and process a DNG image</Typography>
                  </Box>
                )}
              </Card>
            </Box>
            
            <Box sx={{ flex: '1 1 300px', minWidth: '300px' }}>
              <Card>
                <CardContent>
                  <Typography variant="subtitle1" gutterBottom>
                    Enhanced
                  </Typography>
                </CardContent>
                {processedImage ? (
                  <CardMedia
                    component="img"
                    image={processedImage}
                    alt="Enhanced"
                    sx={{ height: 300, objectFit: 'contain' }}
                  />
                ) : (
                  <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: 'grey.100' }}>
                    <Typography color="text.secondary">Process an image to see results</Typography>
                  </Box>
                )}
              </Card>
            </Box>
          </Box>
        </Paper>
      </Box>

      {/* Metadata Display */}
      {metadata && (
        <Box sx={{ mt: 3 }}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Processing Results & Metadata
            </Typography>
            
            <Box sx={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap', gap: 2 }}>
              <Box sx={{ flex: '1 1 200px', minWidth: '200px' }}>
                <Typography variant="body2" color="text.secondary">
                  Original Saturation
                </Typography>
                <Typography variant="body1">
                  {metadata.original_saturation?.toFixed(4)}
                </Typography>
              </Box>
              <Box sx={{ flex: '1 1 200px', minWidth: '200px' }}>
                <Typography variant="body2" color="text.secondary">
                  Enhanced Saturation
                </Typography>
                <Typography variant="body1" color="success.main">
                  {metadata.enhanced_saturation?.toFixed(4)}
                </Typography>
              </Box>
              <Box sx={{ flex: '1 1 200px', minWidth: '200px' }}>
                <Typography variant="body2" color="text.secondary">
                  Saturation Improvement
                </Typography>
                <Typography variant="body1" color="primary.main">
                  +{metadata.saturation_improvement?.toFixed(4)}
                </Typography>
              </Box>
              <Box sx={{ flex: '1 1 200px', minWidth: '200px' }}>
                <Typography variant="body2" color="text.secondary">
                  Processing Time
                </Typography>
                <Typography variant="body1">
                  {metadata.processing_time?.toFixed(2)}s
                </Typography>
              </Box>
              <Box sx={{ flex: '1 1 200px', minWidth: '200px' }}>
                <Typography variant="body2" color="text.secondary">
                  Camera
                </Typography>
                <Typography variant="body1">
                  {metadata.metadata?.['Image Make'] || 'Unknown'}
                </Typography>
              </Box>
              <Box sx={{ flex: '1 1 200px', minWidth: '200px' }}>
                <Typography variant="body2" color="text.secondary">
                  Model
                </Typography>
                <Typography variant="body1">
                  {metadata.metadata?.['Image Model'] || 'Unknown'}
                </Typography>
              </Box>
              <Box sx={{ flex: '1 1 200px', minWidth: '200px' }}>
                <Typography variant="body2" color="text.secondary">
                  Enhancement Level
                </Typography>
                <Typography variant="body1">
                  {metadata.enhancement_level}
                </Typography>
              </Box>
              {metadata.predicted_level && (
                <Box sx={{ flex: '1 1 200px', minWidth: '200px' }}>
                  <Typography variant="body2" color="text.secondary">
                    AI Predicted Level
                  </Typography>
                  <Typography variant="body1" color="secondary.main" sx={{ fontWeight: 'bold' }}>
                    {metadata.predicted_level}
                  </Typography>
                </Box>
              )}
              {metadata.message && (
                <Box sx={{ flex: '1 1 200px', minWidth: '200px' }}>
                  <Typography variant="body2" color="success.main" sx={{ fontWeight: 'bold' }}>
                    {metadata.message}
                  </Typography>
                </Box>
              )}
            </Box>
            
            <Box sx={{ mt: 3, textAlign: 'center' }}>
              <Button
                variant="contained"
                startIcon={<Download />}
                onClick={() => {
                  const link = document.createElement('a');
                  link.href = processedImage;
                  link.download = `enhanced_${selectedFile?.name}`;
                  link.click();
                }}
              >
                Download Enhanced Image
              </Button>
            </Box>
          </Paper>
        </Box>
      )}

      {/* Satisfaction Dialog */}
      <Dialog 
        open={showSatisfactionDialog} 
        onClose={() => setShowSatisfactionDialog(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <CheckCircle color="primary" />
            How satisfied are you with the enhancement?
          </Typography>
        </DialogTitle>
        <DialogContent>
          {lastProcessedImage && (
            <Box sx={{ mb: 3, textAlign: 'center' }}>
              <img 
                src={lastProcessedImage} 
                alt="Enhanced preview" 
                style={{ maxWidth: '100%', maxHeight: '200px', borderRadius: '8px' }}
              />
            </Box>
          )}
          <Typography variant="body1" gutterBottom>
            Please rate your satisfaction with the enhanced image:
          </Typography>
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
            <Rating
              value={satisfactionRating}
              onChange={(event, newValue) => {
                setSatisfactionRating(newValue || 5);
              }}
              size="large"
              precision={1}
            />
          </Box>
          <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center' }}>
            {satisfactionRating === 1 && "Very Dissatisfied"}
            {satisfactionRating === 2 && "Dissatisfied"}
            {satisfactionRating === 3 && "Neutral"}
            {satisfactionRating === 4 && "Satisfied"}
            {satisfactionRating === 5 && "Very Satisfied"}
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowSatisfactionDialog(false)}>
            Skip
          </Button>
          <Button onClick={handleSatisfactionSubmit} variant="contained">
            Submit Rating
          </Button>
        </DialogActions>
      </Dialog>
      </Container>
    </ThemeProvider>
  );
};

export default ImageProcessor;
