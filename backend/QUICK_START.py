"""
QUICK START GUIDE - Malaria Detection Training
===============================================

STEP-BY-STEP COMMANDS FOR VS CODE TERMINAL:

1. OPEN TERMINAL IN VS CODE
   - Press: Ctrl + ` (Windows/Linux) or Cmd + ` (Mac)

2. NAVIGATE TO BACKEND FOLDER
   cd /path/to/your/project/backend
   
   # Or if already in project root:
   cd backend

3. CREATE VIRTUAL ENVIRONMENT (First time only)
   # Windows:
   python -m venv venv
   
   # Mac/Linux:
   python3 -m venv venv

4. ACTIVATE VIRTUAL ENVIRONMENT
   # Windows:
   venv\\Scripts\\activate
   
   # Mac/Linux:
   source venv/bin/activate
   
   # You should see (venv) in your terminal prompt

5. INSTALL DEPENDENCIES (First time only)
   pip install --upgrade pip
   pip install -r requirements.txt
   
   # This takes 5-10 minutes

6. START TRAINING
   python train.py
   
   # Training will start automatically!
   # Expected time: 30-45 minutes on CPU, 5-10 minutes on GPU

7. MONITOR PROGRESS
   - Watch terminal output for epoch progress
   - Visit: https://wandb.ai/sangamnirala2004-d-d-beyond/Malaria-Detection
   - Check: ./logs/training_log.csv

8. AFTER TRAINING - USE THE MODEL
   python inference.py --image path/to/cell_image.jpg
   
   # Example:
   python inference.py --image sample_cell.jpg

9. DEACTIVATE ENVIRONMENT (When done)
   deactivate

===============================================
COMMON ISSUES & SOLUTIONS:

Issue: "python: command not found"
Solution: Use python3 instead of python

Issue: "Permission denied"
Solution: Run with: sudo pip install -r requirements.txt (Linux/Mac)

Issue: "Module not found"
Solution: Make sure virtual environment is activated

Issue: "Dataset download slow"
Solution: First download is ~337MB, be patient!

===============================================
FILE STRUCTURE AFTER TRAINING:

backend/
├── config.py                    # Your settings
├── data_loader.py               # Data handling
├── model.py                     # Model architecture
├── train.py                     # Training script
├── inference.py                 # Prediction script
├── requirements.txt             # Dependencies
├── models/                      # Saved models ✓
│   ├── best_model.h5
│   └── malaria_model_final.h5
├── logs/                        # Training logs ✓
│   └── training_log.csv
└── weights/                     # Checkpoints ✓

===============================================
WHAT YOU NEED TO KNOW:

✓ Dataset downloads automatically (one-time, ~337MB)
✓ WandB API key already configured
✓ All paths created automatically
✓ Model saves automatically during training
✓ Best model kept based on validation accuracy

===============================================
CUSTOMIZATION (in config.py):

Want faster training?
→ Reduce N_EPOCHS from 5 to 2

Want better accuracy?
→ Increase N_EPOCHS to 10-20
→ Increase BATCH_SIZE to 64 (if you have enough RAM)

Want different image size?
→ Change IM_SIZE (warning: affects memory usage)

===============================================
"""

if __name__ == "__main__":
    print(__doc__)
