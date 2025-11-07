


https://github.com/user-attachments/assets/818f802a-cd95-4be0-96d4-e4063ba89e80



# Malaria Detection Web Application

## Overview
A full-stack web application for detecting malaria in cell images using a trained deep learning model. The application provides an intuitive interface for uploading cell images and receiving instant predictions with confidence scores.

## Features Implemented

### Backend (FastAPI)
✅ **API Endpoint**: `/api/predict-malaria`
- Accepts image file uploads
- Processes images (resize to 224x224, normalize)
- Returns predictions: "Parasitized" or "Uninfected"
- Provides confidence percentage
- Stores prediction history in MongoDB

✅ **Model Integration**
- Loads best_model.h5 on server startup
- TensorFlow/Keras model with 4,668,297 parameters
- Trained on 27,558 cell images
- Achieves 95%+ accuracy

✅ **Prediction History API**: `/api/prediction-history`
- Retrieves recent predictions
- Stored in MongoDB for analytics

### Frontend (React + Tailwind CSS)
✅ **Modern UI with Features**:
- Drag-and-drop file upload
- Click-to-browse file selection
- Live image preview
- Loading animations
- Beautiful result display with confidence meters
- Color-coded results (Red for Parasitized, Green for Uninfected)
- Responsive design for all screen sizes

✅ **User Experience**:
- File validation (image types, max 10MB)
- Error handling with clear messages
- Reset functionality for multiple tests
- Professional gradient background
- Information cards about the disease

## How to Use

### 1. Access the Application
Open your browser and navigate to the frontend URL provided by your deployment.

### 2. Upload an Image
- **Drag and drop** a cell image onto the upload area, OR
- **Click the upload area** to browse and select a file
- Supported formats: JPG, PNG, JPEG
- Maximum file size: 10MB

### 3. Analyze
- Click the **"Analyze Image"** button
- Wait for the analysis (typically 1-2 seconds)

### 4. View Results
The application will display:
- **Prediction**: Parasitized or Uninfected
- **Confidence Score**: Percentage confidence of the prediction
- **Visual Indicators**: Color-coded result card with progress bar
- **Detailed Message**: Clear explanation of the result

### 5. Test Another Image
- Click **"Analyze Another Image"** to reset and test a new sample
- Or click the **X button** on the preview to select a different image

## Technical Details

### Model Architecture
- **Base**: LeNet-inspired CNN
- **Input**: 224x224 RGB images
- **Layers**: Convolutional + BatchNorm + MaxPool + Dense
- **Output**: Binary classification (Parasitized/Uninfected)
- **Training**: 5 epochs with data augmentation

### Data Augmentation Applied
- **Type**: Repeated augmentation (5x dataset)
- **Techniques**:
  - Random brightness adjustment
  - Vertical flips
  - Horizontal flips
  - 90-degree rotations
  - Basic preprocessing

### API Endpoints

#### POST `/api/predict-malaria`
Upload and analyze a cell image.

**Request**:
```bash
curl -X POST \
  -F "file=@cell_image.jpg" \
  http://your-backend-url/api/predict-malaria
```

**Response**:
```json
{
  "success": true,
  "prediction": "Parasitized",
  "confidence": 98.45,
  "message": "The cell image is predicted to be Parasitized with 98.45% confidence."
}
```

#### GET `/api/prediction-history?limit=50`
Retrieve recent predictions.

**Response**:
```json
{
  "success": true,
  "predictions": [
    {
      "id": "uuid",
      "filename": "cell_001.jpg",
      "prediction": "Parasitized",
      "confidence": 98.45,
      "timestamp": "2025-11-02T05:00:00Z"
    }
  ],
  "count": 10
}
```

## Performance Metrics

### Model Performance
- **Test Accuracy**: 95.80%
- **Precision**: 95.50%
- **Recall**: 93.70%
- **F1-Score**: 94.59%
- **AUC Score**: 98.45%

### Inference Speed
- **Average**: ~60-70ms per image
- **Throughput**: ~15 images/second

## File Structure

```
/app/
├── backend/
│   ├── server.py                 # Main FastAPI server with ML endpoints
│   ├── train.py                  # Model training script
│   ├── test.py                   # Testing script
│   ├── comprehensive_evaluation.py  # Evaluation & visualization
│   ├── config.py                 # Configuration
│   ├── model.py                  # Model architecture
│   ├── data_loader.py            # Data loading
│   ├── inference.py              # Inference utilities
│   ├── models/
│   │   └── best_model.h5        # Trained model (95% accuracy)
│   ├── results/                 # Evaluation results
│   └── logs/                    # Training logs
│
└── frontend/
    ├── src/
    │   ├── App.js               # Main app component
    │   └── components/
    │       └── MalariaDetection.js  # Detection UI component
    └── package.json
```

## Technologies Used

### Backend
- **FastAPI**: Modern Python web framework
- **TensorFlow/Keras**: Deep learning model
- **PIL (Pillow)**: Image processing
- **MongoDB**: Database for prediction history
- **Uvicorn**: ASGI server

### Frontend
- **React**: UI framework
- **Tailwind CSS**: Styling
- **Lucide React**: Icon library
- **Axios**: HTTP client (for fetch alternative)

## Security & Validation

✅ File type validation (images only)
✅ File size limits (10MB maximum)
✅ Error handling for invalid uploads
✅ Server-side image processing validation
✅ Database storage of predictions

## Future Enhancements

Potential improvements:
- [ ] Batch processing (multiple images at once)
- [ ] Detailed visualization of infected areas
- [ ] Export prediction reports as PDF
- [ ] User authentication and personal history
- [ ] Model comparison (try different models)
- [ ] Advanced statistics and analytics dashboard
- [ ] Mobile app version

## Notes

- The model was trained with "repeated" augmentation technique
- Best model was saved at Epoch 2 with 95.608% validation accuracy
- All predictions are stored in MongoDB for future analysis
- The application runs on CPU (CUDA not required)

## Testing

To test the application:
1. Use sample cell images from the malaria dataset
2. Try both parasitized and uninfected samples
3. Test with different image formats and sizes
4. Verify confidence scores are reasonable (>70% for most cases)

## Support

For issues or questions:
- Check backend logs: `/var/log/supervisor/backend.err.log`
- Check frontend logs: Browser console (F12)
- Verify model file exists: `/app/backend/models/best_model.h5`

---

**Application Status**: ✅ Fully Functional

The malaria detection web application is ready to use with a trained model achieving 95%+ accuracy!
