# Model Files Explanation

## Why Two Models Are Saved During Training?

When you run `train.py`, **two different model files are saved**:

### 1. `best_model.h5` ✅ (Recommended)
**Location**: `./models/best_model.h5`

**How it's saved**: Automatically during training by the `ModelCheckpoint` callback

**When it's saved**: Whenever the validation accuracy improves during training

**What it contains**: The model weights from the epoch with the **highest validation accuracy**

**Why it's better**: 
- Guaranteed to have the best performance on validation data
- Protected against overfitting (won't save if performance degrades)
- This is the standard industry practice

**Code in train.py (Line 79-84)**:
```python
ModelCheckpoint(
    filepath=os.path.join(MODEL_SAVE_PATH, 'best_model.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
```

---

### 2. `malaria_model_final.h5`
**Location**: `./models/malaria_model_final.h5`

**How it's saved**: Manually at the end of training

**When it's saved**: After all epochs complete (regardless of performance)

**What it contains**: The model weights from the **last epoch**

**When to use**:
- For debugging purposes
- To resume training from the last checkpoint
- To compare final state vs best state

**Code in train.py (Line 157-159)**:
```python
final_model_path = os.path.join(MODEL_SAVE_PATH, 'malaria_model_final.h5')
model.save(final_model_path)
```

---

## Which Model Should You Use?

### For Testing and Production: `best_model.h5` ✅

**Updated `test.py` behavior**:
- Now automatically uses `best_model.h5` if it exists
- Falls back to `malaria_model_final.h5` if best_model doesn't exist
- Shows which model is being loaded

Example output:
```
📂 Loading model from: ./models/best_model.h5
   ℹ️  Using best_model.h5 (best validation accuracy during training)
✅ Model loaded successfully!
```

---

## Training Workflow

```
Start Training (train.py)
         ↓
    Epoch 1
         ↓
    Check val_accuracy
         ↓
    Save to best_model.h5 ✓
         ↓
    Epoch 2
         ↓
    val_accuracy improved?
         ↓
    Yes → Update best_model.h5 ✓
         ↓
    Epoch 3
         ↓
    val_accuracy improved?
         ↓
    No → Keep old best_model.h5
         ↓
    ... (more epochs)
         ↓
    Training Complete
         ↓
    Save to malaria_model_final.h5
         ↓
    Done! ✓
```

**Result**: 
- `best_model.h5` = Weights from Epoch 2 (or whichever had best val_accuracy)
- `malaria_model_final.h5` = Weights from last epoch

---

## Example Scenario

Let's say training runs for 5 epochs with these validation accuracies:

| Epoch | Val Accuracy | Action |
|-------|-------------|--------|
| 1     | 85.0%       | Save to best_model.h5 |
| 2     | 92.0%       | Update best_model.h5 ✓ |
| 3     | **94.5%**   | **Update best_model.h5 ✓** |
| 4     | 93.0%       | No update (worse than 94.5%) |
| 5     | 92.5%       | No update (worse than 94.5%) |

**After training**:
- ✅ `best_model.h5` = Model from **Epoch 3** (94.5% accuracy)
- ⚠️ `malaria_model_final.h5` = Model from **Epoch 5** (92.5% accuracy)

**For testing/production, use `best_model.h5`** (94.5% > 92.5%)

---

## How to Specify Which Model to Use

### 1. Default Behavior (Recommended)
```bash
python test.py
# Automatically uses best_model.h5 if available
```

### 2. Force Use of Final Model
```bash
python test.py --model-path ./models/malaria_model_final.h5
```

### 3. Use Custom Model
```bash
python test.py --model-path ./models/your_custom_model.h5
```

---

## Do I Need Both Models?

### Keep Both If:
- You want to compare best vs final performance
- You're debugging training behavior
- You want to resume training from the final state

### Keep Only `best_model.h5` If:
- You only care about the best performing model
- You want to save disk space
- You're deploying to production

### To Save Only One Model

Edit `train.py` and comment out lines 157-159:

```python
# Save final model (optional - best_model.h5 is already saved)
# final_model_path = os.path.join(MODEL_SAVE_PATH, 'malaria_model_final.h5')
# model.save(final_model_path)
# print(f"\nFinal model saved to: {final_model_path}")
```

---

## Summary

| Feature | best_model.h5 | malaria_model_final.h5 |
|---------|---------------|------------------------|
| **Performance** | ✅ Best validation accuracy | ⚠️ Last epoch (may be worse) |
| **Recommended for** | Production, Testing, Deployment | Debugging, Analysis |
| **When saved** | During training (when improved) | After training completes |
| **Disk space** | ~54 MB | ~54 MB |
| **Default in test.py** | ✅ Yes (after update) | No |

**Recommendation**: Always use `best_model.h5` for production and testing.

---

## Current Status

✅ `test.py` has been updated to:
1. Prioritize `best_model.h5` over `malaria_model_final.h5`
2. Show which model is being loaded
3. List available models if the default is not found
4. Provide helpful error messages

Run `python test.py` to automatically use the best model!
