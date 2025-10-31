# Malaria Detection Model Training

This project trains a Convolutional Neural Network (CNN) model based on the LeNet architecture to detect malaria from cell images.

## Project Structure

```
/app/backend/
├── config.py           # Configuration and hyperparameters
├── data_loader.py      # Data loading and preprocessing
├── model.py            # Model architecture definition
├── train.py            # Main training script
├── requirements.txt    # Python dependencies
└── MALARIA_DETECTION_README.md  # This file
```

## Setup Instructions for Local VS Code

### 1. Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### 2. Installation Steps

**Step 1: Navigate to the backend directory**
```bash
cd /path/to/your/project/backend
```

**Step 2: Create a virtual environment (recommended)**
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**Step 3: Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- TensorFlow 2.19+ (Deep learning framework)
- TensorFlow Datasets (For malaria dataset)
- Weights & Biases (Experiment tracking)
- OpenCV, Matplotlib, Seaborn (Visualization)
- Scikit-learn (Metrics)

### 3. Running the Training

**To start training the model, simply run:**
```bash
python train.py
```

### 4. What Happens During Training

1. **Dataset Download**: The malaria dataset (~337 MB) will be automatically downloaded from TensorFlow Datasets on first run
2. **Data Preprocessing**: Images are resized to 224x224 and normalized
3. **Data Augmentation**: Random rotations and flips are applied to training data
4. **Model Training**: The LeNet-based CNN is trained for 5 epochs (configurable)
5. **WandB Logging**: Training metrics are logged to Weights & Biases
6. **Model Saving**: Best model is saved in `./models/` directory

### 5. Output Files

After training, you'll find:
- `./models/best_model.h5` - Best model based on validation accuracy
- `./models/malaria_model_final.h5` - Final trained model
- `./logs/training_log.csv` - Training history in CSV format
- `./weights/` - Model checkpoint files

### 6. Customizing Training

Edit `config.py` to modify:
- **Learning Rate**: `LEARNING_RATE = 0.001`
- **Number of Epochs**: `N_EPOCHS = 5`
- **Batch Size**: `BATCH_SIZE = 32`
- **Image Size**: `IM_SIZE = 224`
- **Dropout Rate**: `DROPOUT_RATE = 0.0`
- **Regularization**: `REGULARIZATION_RATE = 0.0`

### 7. Model Architecture

The model uses a LeNet-inspired architecture:
- **Input**: 224x224x3 RGB images
- **Conv Layer 1**: 6 filters, 3x3 kernel + BatchNorm + MaxPool + Dropout
- **Conv Layer 2**: 16 filters, 3x3 kernel + BatchNorm + MaxPool
- **Dense Layer 1**: 100 neurons + BatchNorm + Dropout
- **Dense Layer 2**: 10 neurons + BatchNorm
- **Output**: 1 neuron with sigmoid (binary classification)

### 8. Monitoring Training

View your training progress at:
- **WandB Dashboard**: https://wandb.ai/sangamnirala2004-d-d-beyond/Malaria-Detection
- **Local Logs**: Check `./logs/training_log.csv`

### 9. Dataset Information

- **Source**: TensorFlow Datasets - Malaria Cell Images
- **Total Images**: 27,558 cell images
- **Classes**: 
  - Parasitized cells
  - Uninfected cells
- **Split**: 80% train, 10% validation, 10% test

### 10. Troubleshooting

**If you encounter "CUDA not found" error:**
- The code will automatically run on CPU if CUDA is not available
- For GPU training, install CUDA and cuDNN compatible with your TensorFlow version

**If dataset download is slow:**
- The malaria dataset is ~337 MB and will be downloaded to `~/tensorflow_datasets/`
- This is a one-time download

**If WandB login fails:**
- Check that your API key in `config.py` is correct
- You can also set it via environment variable: `export WANDB_API_KEY=your_key`

### 11. Expected Training Time

- **CPU**: ~30-45 minutes for 5 epochs
- **GPU**: ~5-10 minutes for 5 epochs

### 12. Expected Performance

After training, you should see:
- **Training Accuracy**: ~75-85%
- **Validation Accuracy**: ~70-80%
- **Test Accuracy**: ~70-80%

## Additional Notes

- The model is configured with WandB API key: `980ca7615589d93a06f69caac98b8e2ab15721da`
- All paths are relative and will be created automatically
- The trained model can be used for inference on new cell images
- Model checkpoints are saved during training to prevent data loss

## Questions or Issues?

If you encounter any issues, check:
1. Python version is 3.8+
2. All dependencies are installed correctly
3. Sufficient disk space (~2GB for dataset + models)
4. Internet connection for first-time dataset download

Happy Training! 🚀
