# Slide 11: Backend API

## Purpose
Provide scalable backend services for DNG image upload, saturation processing, and enhanced image delivery for the PrismVI25 system.

## Backend Flow
```
Client Request
      ↓
Flask API Endpoint
      ↓
DNG File Validation
      ↓
RAW Image Processing (rawpy)
      ↓
Saturation Enhancement Engine
      ↓
JSON Response + Enhanced Base64 Output
```

## Features
- **RESTful API integration** with Flask framework
- **DNG image processing** using rawpy and OpenCV
- **Saturation enhancement** with adjustable levels (5-10 scale)
- **EXIF metadata extraction** from DNG files
- **Secure file upload handling** with temporary file management
- **CORS enabled** for cross-origin requests
- **Error handling and logging** for robust operation
- **Base64 image encoding** for frontend delivery
- **KPI data endpoints** for dashboard metrics
- **Health check endpoints** for system monitoring
