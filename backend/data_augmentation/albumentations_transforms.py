"""Albumentations-based augmentation transforms"""

import tensorflow as tf
import albumentations as A
import numpy as np
from config import CONFIG


def get_albumentations_transforms():
    """Create Albumentations transform pipeline"""
    IM_SIZE = CONFIG['IM_SIZE']
    
    transforms = A.Compose([
        A.Resize(IM_SIZE, IM_SIZE),
        
        # Random flips
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ], p=0.3),
        
        # Random rotation
        A.RandomRotate90(),
        
        # Brightness and contrast adjustments
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        
        # Sharpening
        A.Sharpen(
            alpha=(0.2, 0.5),
            lightness=(0.5, 1.0),
            p=0.5
        ),
    ])
    
    return transforms


def apply_albumentations_augmentation(image):
    """Apply Albumentations transforms to a single image
    
    Args:
        image: Input image tensor
        
    Returns:
        Augmented and normalized image
    """
    transforms = get_albumentations_transforms()
    
    # Convert to numpy for albumentations
    data = {"image": image}
    augmented = transforms(**data)
    image = augmented["image"]
    
    # Normalize to [0, 1]
    image = tf.cast(image / 255.0, tf.float32)
    return image


def process_with_albumentations(image, label):
    """Process image-label pair with albumentations
    
    Args:
        image: Input image tensor
        label: Image label
        
    Returns:
        Tuple of (augmented_image, label)
    """
    # Use tf.numpy_function to apply albumentations
    aug_img = tf.numpy_function(
        func=apply_albumentations_augmentation,
        inp=[image],
        Tout=tf.float32
    )
    
    # Set shape to avoid shape inference issues
    aug_img.set_shape([CONFIG['IM_SIZE'], CONFIG['IM_SIZE'], 3])
    
    return aug_img, label


def create_albumentations_dataset(dataset, batch_size=32):
    """Create dataset with Albumentations augmentation
    
    Args:
        dataset: Input TensorFlow dataset
        batch_size: Batch size for dataset
        
    Returns:
        Augmented dataset ready for training
    """
    augmented_dataset = (
        dataset
        .shuffle(buffer_size=1024, reshuffle_each_iteration=True)
        .map(process_with_albumentations, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    return augmented_dataset
