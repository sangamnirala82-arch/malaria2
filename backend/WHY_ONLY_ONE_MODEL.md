# Why We Only Save `best_model.h5` Now

## The Problem

Previously, `train.py` was saving **TWO models**:
1. `best_model.h5` - Model with best validation accuracy ✅
2. `malaria_model_final.h5` - Model from the last epoch

**But we only ever used `best_model.h5`!**

So saving `malaria_model_final.h5` was:
- ❌ Wasting disk space (~54 MB)
- ❌ Wasting time during training
- ❌ Creating confusion about which model to use
- ❌ Redundant since we have best_model.h5

---

## The Solution

**Now `train.py` only saves ONE model: `best_model.h5`**

### How It Works

During training, the `ModelCheckpoint` callback:
1. Monitors validation accuracy after each epoch
2. Saves the model **only when validation accuracy improves**
3. Overwrites the previous best_model.h5 if a better one is found

```python
ModelCheckpoint(
    filepath='./models/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,  # ← Only saves when accuracy improves!
    verbose=1
)
```

---

## Training Flow (Simplified)

```
Start Training
    ↓
Epoch 1: val_acc = 85% → Save best_model.h5 ✓
    ↓
Epoch 2: val_acc = 92% → Update best_model.h5 ✓
    ↓
Epoch 3: val_acc = 94% → Update best_model.h5 ✓
    ↓
Epoch 4: val_acc = 93% → No save (worse than 94%)
    ↓
Epoch 5: val_acc = 92% → No save (worse than 94%)
    ↓
Training Complete
    ↓
Result: ONE model saved (from Epoch 3 with 94% accuracy) ✓
```

---

## Benefits of This Approach

✅ **Saves disk space** - Only one 54MB file instead of two

✅ **No confusion** - Only one model to use

✅ **Best performance** - Guaranteed to have the best validation accuracy

✅ **Prevents overfitting** - Won't save if the model starts overfitting

✅ **Standard practice** - This is how most ML projects work

---

## What Happens to test.py?

**Before**:
```python
# Had to check which model exists and choose one
if os.path.exists('best_model.h5'):
    use best_model
elif os.path.exists('malaria_model_final.h5'):
    use final_model
```

**Now**:
```python
# Simple! Just use best_model.h5
model_path = './models/best_model.h5'
```

---

## When You Train

```bash
python train.py
```

**Output**:
```
Training completed successfully!

✅ Best model saved to: ./models/best_model.h5
   (Saved automatically by ModelCheckpoint during training)
```

---

## When You Test

```bash
python test.py
```

**Output**:
```
📂 Loading model from: ./models/best_model.h5
   ℹ️  This is the best model (highest validation accuracy during training)
✅ Model loaded successfully!
```

---

## FAQ

**Q: What if I want to keep the final model for some reason?**

A: You can still manually save it by adding this code at the end of `train.py`:
```python
# Optional: Save final model
final_model_path = os.path.join(MODEL_SAVE_PATH, 'malaria_model_final.h5')
model.save(final_model_path)
```

**Q: How do I know which epoch the best_model.h5 is from?**

A: Check the `training_log.csv` file:
```bash
cat logs/training_log.csv
```
Look for the row with the highest `val_accuracy`.

**Q: Can I save multiple checkpoints?**

A: Yes! Modify the ModelCheckpoint filename to include the epoch:
```python
ModelCheckpoint(
    filepath='./models/model_epoch_{epoch:02d}_acc_{val_accuracy:.2f}.h5',
    monitor='val_accuracy',
    save_best_only=False  # Save every epoch
)
```

---

## Summary

| Before | After |
|--------|-------|
| 2 model files saved | 1 model file saved ✓ |
| ~108 MB disk space | ~54 MB disk space ✓ |
| Confusion about which to use | Clear: use best_model.h5 ✓ |
| Complex logic in test.py | Simple logic ✓ |

**Result**: Cleaner, simpler, more efficient! 🎉
