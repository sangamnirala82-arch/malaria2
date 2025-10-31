"""Dataset Repetition with Different Augmentation Variations

This module creates multiple copies of the dataset with different augmentations
to increase training data diversity.
"""

import tensorflow as tf
from config import CONFIG


def resize_rescale(image, label):
    """Resize and rescale images"""
    IM_SIZE = CONFIG['IM_SIZE']
    image = tf.image.resize(image, (IM_SIZE, IM_SIZE)) / 255.0
    return image, label


def augment_1(image, label):
    """Augmentation variation 1: Random brightness
    
    Args:
        image: Input image
        label: Image label
        
    Returns:
        Augmented image and label
    """
    # Ensure image has proper shape
    image.set_shape([None, None, 3])
    image, label = resize_rescale(image, label)
    
    # Apply random brightness
    image = tf.image.random_brightness(image, 0.2)
    return image, label


def augment_2(image, label):
    """Augmentation variation 2: Vertical flip
    
    Args:
        image: Input image
        label: Image label
        
    Returns:
        Augmented image and label
    """
    # Ensure image has proper shape
    image.set_shape([None, None, 3])
    image, label = resize_rescale(image, label)
    
    # Apply random vertical flip
    image = tf.image.random_flip_up_down(image)
    return image, label


def augment_3(image, label):
    """Augmentation variation 3: Horizontal flip
    
    Args:
        image: Input image
        label: Image label
        
    Returns:
        Augmented image and label
    """
    # Ensure image has proper shape
    image.set_shape([None, None, 3])
    image, label = resize_rescale(image, label)
    
    # Apply horizontal flip
    image = tf.image.flip_left_right(image)
    return image, label


def augment_4(image, label):
    """Augmentation variation 4: 90-degree rotation
    
    Args:
        image: Input image
        label: Image label
        
    Returns:
        Augmented image and label
    """
    # Ensure image has proper shape
    image.set_shape([None, None, 3])
    image, label = resize_rescale(image, label)
    
    # Apply 90-degree rotation
    image = tf.image.rot90(image)
    return image, label


def augment_5(image, label):
    """Augmentation variation 5: Basic resize and rescale
    
    Args:
        image: Input image
        label: Image label
        
    Returns:
        Processed image and label
    """
    # Ensure image has proper shape
    image.set_shape([None, None, 3])
    image, label = resize_rescale(image, label)
    
    return image, label


def augment_variations():
    """Get all augmentation variation functions
    
    Returns:
        List of augmentation functions
    """
    return [augment_1, augment_2, augment_3, augment_4, augment_5]


def create_repeated_dataset(dataset, batch_size=32):
    """Create dataset with 5x repetition using different augmentations
    
    Args:
        dataset: Input TensorFlow dataset
        batch_size: Batch size for the final dataset
        
    Returns:
        Repeated and augmented dataset
    """
    # Create 5 versions of the dataset with different augmentations
    train_dataset_1 = (
        dataset
        .shuffle(buffer_size=1024, reshuffle_each_iteration=True)
        .map(augment_1, num_parallel_calls=tf.data.AUTOTUNE)
    )
    
    train_dataset_2 = (
        dataset
        .shuffle(buffer_size=1024, reshuffle_each_iteration=True)
        .map(augment_2, num_parallel_calls=tf.data.AUTOTUNE)
    )
    
    train_dataset_3 = (
        dataset
        .shuffle(buffer_size=1024, reshuffle_each_iteration=True)
        .map(augment_3, num_parallel_calls=tf.data.AUTOTUNE)
    )
    
    train_dataset_4 = (
        dataset
        .shuffle(buffer_size=1024, reshuffle_each_iteration=True)
        .map(augment_4, num_parallel_calls=tf.data.AUTOTUNE)
    )
    
    train_dataset_5 = (
        dataset
        .shuffle(buffer_size=1024, reshuffle_each_iteration=True)
        .map(augment_5, num_parallel_calls=tf.data.AUTOTUNE)
    )
    
    # Concatenate all datasets
    full_dataset = (
        train_dataset_1
        .concatenate(train_dataset_2)
        .concatenate(train_dataset_3)
        .concatenate(train_dataset_4)
        .concatenate(train_dataset_5)
    )
    
    # Shuffle, batch, and prefetch
    full_dataset = (
        full_dataset
        .shuffle(buffer_size=1024, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    return full_dataset
