"""
Albumentations Advanced Augmentation

Albumentations is a powerful image augmentation library with many transforms.
This module provides pre-configured transforms for medical imaging.

Website: https://albumentations.ai/
"""

import tensorflow as tf
import numpy as np

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("⚠️  Albumentations not installed. Run: pip install albumentations")


def get_medical_transforms(image_size=224):
    """
    Get augmentation pipeline optimized for medical images.
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations Compose object or None
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return None
    
    transforms = A.Compose([
        # Resize to target size
        A.Resize(image_size, image_size),
        
        # Geometric transforms (with probability)
        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
        ], p=0.3),
        
        # Rotation
        A.RandomRotate90(p=0.5),
        
        # Color/Contrast adjustments (important for microscopy images)
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        
        # Sharpening (helps with microscopy images)
        A.Sharpen(
            alpha=(0.2, 0.5),
            lightness=(0.5, 1.0),
            p=0.5
        ),
        
        # Slight blur (simulates focus variations)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        
        # Noise (simulates camera noise)
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(p=1.0),
        ], p=0.2),
    ])
    
    return transforms


def get_aggressive_transforms(image_size=224):
    """
    Get more aggressive augmentation pipeline for higher accuracy.
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations Compose object or None
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return None
    
    transforms = A.Compose([
        A.Resize(image_size, image_size),
        
        # All geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=30, p=0.3),
        
        # Grid shuffle (creates different spatial patterns)
        A.RandomGridShuffle(grid=(2, 2), p=0.3),
        
        # Color adjustments
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(p=0.3),
        
        # Sharpening and blur
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        
        # Noise and compression artifacts
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.ImageCompression(quality_lower=75, quality_upper=100, p=0.2),
        
        # Cutout (random erasing)
        A.CoarseDropout(
            max_holes=8,
            max_height=16,
            max_width=16,
            fill_value=0,
            p=0.3
        ),
    ])
    
    return transforms


def apply_albumentations(image, label, transforms):
    """
    Apply Albumentations transforms to a single image.
    
    Args:
        image: Image tensor or numpy array
        label: Image label
        transforms: Albumentations Compose object
        
    Returns:
        Augmented image and label
    """
    if not ALBUMENTATIONS_AVAILABLE or transforms is None:
        # Fallback to basic preprocessing
        if isinstance(image, tf.Tensor):
            image = tf.image.resize(image, (224, 224)) / 255.0
        return image, label
    
    # Convert tensor to numpy if needed
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    
    # Ensure uint8 format for Albumentations
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Apply transforms
    augmented = transforms(image=image)
    augmented_image = augmented['image']
    
    # Convert back to float32 and normalize
    augmented_image = augmented_image.astype(np.float32) / 255.0
    
    return augmented_image, label


def create_albumentations_dataset(dataset, transforms=None, image_size=224):
    """
    Create a dataset with Albumentations augmentation.
    
    Args:
        dataset: TensorFlow dataset with (image, label) pairs
        transforms: Albumentations Compose object (if None, uses medical transforms)
        image_size: Target image size
        
    Returns:
        Dataset with Albumentations applied
    """
    if not ALBUMENTATIONS_AVAILABLE:
        print("⚠️  Albumentations not available. Falling back to basic augmentation.")
        return dataset.map(
            lambda img, lbl: (tf.image.resize(img, (image_size, image_size)) / 255.0, lbl),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    if transforms is None:
        transforms = get_medical_transforms(image_size)
    
    def augment_fn(image, label):
        aug_img, aug_label = tf.numpy_function(
            func=lambda img, lbl: apply_albumentations(img, lbl, transforms),
            inp=[image, label],
            Tout=[tf.float32, tf.int64]
        )
        # Set shape explicitly (TensorFlow needs this)
        aug_img.set_shape([image_size, image_size, 3])
        aug_label.set_shape([])
        return aug_img, aug_label
    
    return dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)


def get_albumentations_info():
    """Return information about Albumentations."""
    return {
        'name': 'Albumentations',
        'description': 'Fast and flexible image augmentation library',
        'benefits': [
            '70+ transforms available',
            'Optimized for speed',
            'Medical imaging support',
            'Pixel-perfect transformations',
            'Easy to compose pipelines'
        ],
        'website': 'https://albumentations.ai/',
        'available': ALBUMENTATIONS_AVAILABLE,
        'recommended_mode': 'medical_transforms'
    }


if __name__ == "__main__":
    # Test Albumentations
    print("Testing Albumentations...")
    info = get_albumentations_info()
    print(f"\n{info['name']}: {info['description']}")
    print(f"Available: {info['available']}")
    if not info['available']:
        print("\nTo install: pip install albumentations")
    else:
        print("\nBenefits:")
        for benefit in info['benefits']:
            print(f"  - {benefit}")
