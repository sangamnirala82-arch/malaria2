# Test Script Usage Guide

## 🧪 Testing Your Malaria Detection Model

The `test.py` script allows you to thoroughly test your trained model with random images from the dataset.

## 📋 Usage Options

### Option 1: Test with Fully Trained Model (Recommended)

After training your model with `python train.py`, test it:

```bash
python test.py
```

This will:
- Load your trained model from `models/malaria_model_final.h5`
- Test on 20 random images from the dataset
- Show predictions, accuracy, and confidence scores
- Display confusion matrix
- Measure inference performance (speed)
- Generate visualization image

### Option 2: Quick Test Mode (No Training Required)

If you haven't trained a model yet, use quick mode:

```bash
python test.py --quick-train
```

This will:
- Train a small model on 20 batches (takes ~10 seconds)
- Use this model for testing immediately
- **Note**: Accuracy will be low (~40-60%) because it's minimally trained
- Purpose: To verify the code works before full training

### Option 3: Test with Custom Model

```bash
python test.py --model path/to/your/model.h5
```

## 📊 What the Test Shows

### 1. Individual Predictions Table
```
Image | True Label    | Predicted     | Confidence | Time (ms) | Correct
----------------------------------------------------------------------
    1 | Parasitized   | Parasitized   |  92.34%   |   42.15  | ✅
    2 | Uninfected    | Uninfected    |  87.56%   |   38.92  | ✅
    3 | Parasitized   | Uninfected    |  65.23%   |   41.03  | ❌
```

### 2. Overall Metrics
- **Accuracy**: Overall percentage of correct predictions
- **Precision**: How many predicted positives were actually positive
- **Recall**: How many actual positives were correctly identified
- **F1-Score**: Harmonic mean of precision and recall

### 3. Confusion Matrix
Shows true vs predicted classifications:
```
           Predicted
         Parasit. | Uninfect.
Actual  ------------------------------
Parasit. |    145   |     15
Uninfect.|     12   |    148
```

### 4. Inference Performance
- **Average inference time**: How long each prediction takes
- **Throughput**: Images processed per second
- **Min/Max times**: Range of prediction speeds

### 5. Visual Predictions
Saves an image showing:
- 6 random test images
- True labels vs predicted labels
- Confidence scores
- Color-coded results (green=correct, red=wrong)

## 🎯 Expected Results

### After Quick Training (20 batches):
- Accuracy: 40-60%
- Speed: ~40-50 ms per image
- Purpose: Code verification only

### After Full Training (5 epochs):
- Accuracy: 70-85%
- Parasitized detection: 75-90%
- Uninfected detection: 70-85%
- Speed: ~40-50 ms per image (CPU)
- Speed: ~5-15 ms per image (GPU)

## 📁 Output Files

After running the test:

```
backend/
├── test_predictions_visualization.png   # Visual predictions
└── models/
    └── quick_test_model.h5              # If using --quick-train
```

## 🔧 Customization

Edit `test.py` to change:

1. **Number of test samples**: Change `num_random_samples=20` in `main()`
2. **Visualization images**: Change `visualize=True` or number of images
3. **Model path**: Modify `model_path` parameter

## ⚡ Performance Tips

**Faster Testing:**
- Use GPU if available (automatically detected)
- Reduce number of test samples
- Skip visualization: `tester.run_comprehensive_test(num_random_samples=10, visualize=False)`

**More Thorough Testing:**
- Increase test samples: `num_random_samples=100`
- Enable full dataset test when prompted
- Save results to file for comparison

## 🐛 Troubleshooting

**Error: "Model not found"**
- Solution: Train model first with `python train.py` or use `--quick-train`

**Error: "Dataset not found"**
- Solution: Dataset downloads automatically on first run (takes 2-5 minutes)

**Low Accuracy (~40-60%)**
- Cause: Using quick-train mode or insufficient training
- Solution: Run full training with `python train.py`

**Slow Inference**
- Cause: Running on CPU
- Solution: Use GPU for faster inference (10x speedup)

## 📝 Example Complete Workflow

```bash
# 1. Train your model (30-45 minutes on CPU)
python train.py

# 2. Test the trained model
python test.py

# 3. Check results
# - View terminal output for metrics
# - Open test_predictions_visualization.png
# - Check models/ and logs/ directories

# 4. Optional: Test with more samples
# Edit test.py to increase num_random_samples
python test.py
```

## 🎓 Understanding the Results

### Good Model Indicators:
✅ Accuracy > 75%
✅ Precision and Recall balanced (difference < 10%)
✅ Both classes detected well (not biased to one class)
✅ High confidence on correct predictions
✅ Consistent inference times

### Model Needs Improvement:
⚠️ Accuracy < 70%
⚠️ Only predicting one class
⚠️ Low confidence even on correct predictions
⚠️ Large precision-recall gap

### If Model Performance is Low:
1. Train for more epochs (increase `N_EPOCHS` in `config.py`)
2. Adjust learning rate
3. Add more data augmentation
4. Try different model architectures
5. Check for data leakage or preprocessing issues

## 💡 Pro Tips

1. **Run test after each training session** to track improvements
2. **Save test results** to compare different model versions
3. **Check visualization images** to understand where model fails
4. **Monitor inference time** if deploying in production
5. **Test on your own cell images** after confirming dataset performance

---

**Ready to test?** Run `python test.py --quick-train` now to see it in action! 🚀
