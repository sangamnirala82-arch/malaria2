# Data Augmentation Implementation Guide

## Overview
This implementation includes 5 advanced data augmentation techniques to improve model accuracy:

1. **Basic Augmentation** (default)
2. **MixUp Augmentation**
3. **CutMix Augmentation**
4. **Albumentations Transforms**
5. **Repeated Dataset (5x with variations)**

## Files Created

### `/app/backend/data_augmentation/`
- `__init__.py` - Package initialization
- `mixup.py` - MixUp augmentation implementation
- `cutmix.py` - CutMix augmentation implementation
- `albumentations_transforms.py` - Albumentations-based transforms
- `repeating_the_dataset.py` - Dataset repetition with 5 different augmentations

## How to Use

### 1. Configure Augmentation Type

Edit `/app/backend/config.py` and set the `AUGMENTATION_TYPE`:

```python
CONFIG = {
    # ... other settings ...
    "USE_AUGMENTATION": True,
    "AUGMENTATION_TYPE": "mixup",  # Choose: "basic", "mixup", "cutmix", "albumentations", "repeated"
    "MIXUP_ALPHA": 0.2,
    "CUTMIX_ALPHA": 0.2,
}
```

### 2. Run Training

```bash
cd /app/backend
python train.py
```

## Augmentation Techniques Explained

### 1. Basic Augmentation (default)
- Random rotation (0°, 90°, 180°, 270°)
- Random horizontal flip
- Resize and normalize to [0, 1]

**When to use**: Baseline training, quick experiments

### 2. MixUp Augmentation
Creates virtual training examples by mixing pairs of images and labels.

**Formula**: 
- `mixed_image = λ × image1 + (1-λ) × image2`
- `mixed_label = λ × label1 + (1-λ) × label2`
- `λ ~ Beta(α, α)` where α = 0.2

**Benefits**:
- Reduces overfitting
- Improves generalization
- Smooths decision boundaries

**When to use**: When you have limited training data or want better generalization

### 3. CutMix Augmentation
Cuts and pastes patches between training images.

**How it works**:
- Randomly select a rectangular region
- Replace that region with a patch from another image
- Mix labels proportional to the patch size

**Benefits**:
- Encourages model to localize features
- Better than MixUp for spatial information
- Improves robustness

**When to use**: When spatial information is important for classification

### 4. Albumentations Transforms
Professional-grade augmentation library with optimized transforms.

**Transforms applied**:
- Resize to target size
- Random horizontal/vertical flips (30% probability)
- Random 90° rotation
- Random brightness and contrast adjustment
- Image sharpening

**Benefits**:
- Fast and efficient
- Industry-standard augmentations
- Wide variety of transforms

**When to use**: For production-quality augmentation pipeline

### 5. Repeated Dataset (5x)
Creates 5 copies of the dataset, each with different augmentation:
- Copy 1: Random brightness
- Copy 2: Vertical flip
- Copy 3: Horizontal flip
- Copy 4: 90° rotation
- Copy 5: Basic resize/rescale

**Benefits**:
- 5x more training data
- Diverse augmentation strategies
- Helps with small datasets

**When to use**: When you have very limited training data (< 1000 samples)

## Performance Recommendations

### For Small Datasets (< 5000 samples)
```python
"AUGMENTATION_TYPE": "repeated"  # 5x data multiplication
```

### For Medium Datasets (5000-20000 samples)
```python
"AUGMENTATION_TYPE": "mixup"  # or "cutmix"
"MIXUP_ALPHA": 0.2
```

### For Large Datasets (> 20000 samples)
```python
"AUGMENTATION_TYPE": "albumentations"  # Fast and efficient
```

### For Maximum Accuracy (regardless of dataset size)
Try each augmentation type and compare results. Different datasets respond better to different augmentations.

## Memory Considerations

Training with augmentation requires significant memory. If you encounter memory issues:

1. **Reduce batch size**:
   ```python
   "BATCH_SIZE": 16  # instead of 32
   ```

2. **Reduce epochs** (for testing):
   ```python
   "N_EPOCHS": 3  # instead of 5
   ```

3. **Use basic augmentation first**:
   ```python
   "AUGMENTATION_TYPE": "basic"
   ```

## Dependencies

All required dependencies are already installed:
- `tensorflow >= 2.19.0`
- `tensorflow-probability >= 0.24.0`
- `albumentations >= 1.4.0`
- `tf-keras == 2.19.0`

## Verification

To verify the implementation is working:

```bash
cd /app/backend
python verify_implementation.py
```

This will test all augmentation techniques on dummy data without loading the full dataset.

## Example Results

Expected improvements with augmentation:
- **Basic**: Baseline accuracy
- **MixUp**: +2-5% accuracy improvement
- **CutMix**: +2-5% accuracy improvement
- **Albumentations**: +3-6% accuracy improvement
- **Repeated**: +5-10% accuracy improvement (especially on small datasets)

## Troubleshooting

### Import Errors
If you get import errors, verify dependencies:
```bash
pip install tensorflow-probability>=0.24.0 albumentations>=1.4.0 tf-keras==2.19.0
```

### Memory Issues
- Reduce batch size in config.py
- Use simpler augmentation (basic or albumentations)
- Close other running processes

### Training Not Starting
- Check WandB API key in config.py
- Verify dataset is downloading correctly
- Check logs: `tail -f training_output.log`

## References

- **MixUp**: Zhang et al., "mixup: Beyond Empirical Risk Minimization" (2018)
- **CutMix**: Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers" (2019)
- **Albumentations**: Buslaev et al., "Albumentations: Fast and Flexible Image Augmentations" (2020)
