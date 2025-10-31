"""CutMix Data Augmentation

CutMix cuts and pastes patches between training images.
Reference: Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers" (2019)
"""

import tensorflow as tf
import tensorflow_probability as tfp
from config import CONFIG


def get_box(lamda, im_size):
    """Generate random bounding box for CutMix
    
    Args:
        lamda: Lambda value from Beta distribution
        im_size: Image size
        
    Returns:
        Bounding box coordinates (r_y, r_x, r_h, r_w)
    """
    # Random center point
    r_x = tf.cast(
        tfp.distributions.Uniform(0, im_size).sample(1)[0],
        dtype=tf.int32
    )
    r_y = tf.cast(
        tfp.distributions.Uniform(0, im_size).sample(1)[0],
        dtype=tf.int32
    )
    
    # Box width and height based on lambda
    r_w = tf.cast(im_size * tf.math.sqrt(1 - lamda), dtype=tf.int32)
    r_h = tf.cast(im_size * tf.math.sqrt(1 - lamda), dtype=tf.int32)
    
    # Clip to image boundaries
    r_x = tf.clip_by_value(r_x - r_w // 2, 0, im_size)
    r_y = tf.clip_by_value(r_y - r_h // 2, 0, im_size)
    
    x_b_r = tf.clip_by_value(r_x + r_w // 2, 0, im_size)
    y_b_r = tf.clip_by_value(r_y + r_h // 2, 0, im_size)
    
    # Calculate actual width and height
    r_w = x_b_r - r_x
    if r_w == 0:
        r_w = 1
    
    r_h = y_b_r - r_y
    if r_h == 0:
        r_h = 1
    
    return r_y, r_x, r_h, r_w


def cutmix_augmentation(dataset_1_sample, dataset_2_sample, alpha=0.2):
    """Apply CutMix augmentation to two samples
    
    Args:
        dataset_1_sample: Tuple of (image_1, label_1)
        dataset_2_sample: Tuple of (image_2, label_2)
        alpha: Beta distribution parameter for mixing
        
    Returns:
        CutMix augmented image and label
    """
    (image_1, label_1), (image_2, label_2) = dataset_1_sample, dataset_2_sample
    
    IM_SIZE = CONFIG['IM_SIZE']
    
    # Sample lambda from Beta distribution
    beta_dist = tfp.distributions.Beta(alpha, alpha)
    lamda = beta_dist.sample(1)[0]
    
    # Get random box
    r_y, r_x, r_h, r_w = get_box(lamda, IM_SIZE)
    
    # Crop and pad from second image
    crop_2 = tf.image.crop_to_bounding_box(image_2, r_y, r_x, r_h, r_w)
    pad_2 = tf.image.pad_to_bounding_box(crop_2, r_y, r_x, IM_SIZE, IM_SIZE)
    
    # Crop and pad from first image
    crop_1 = tf.image.crop_to_bounding_box(image_1, r_y, r_x, r_h, r_w)
    pad_1 = tf.image.pad_to_bounding_box(crop_1, r_y, r_x, IM_SIZE, IM_SIZE)
    
    # Create mixed image
    mixed_image = image_1 - pad_1 + pad_2
    
    # Adjust lambda based on actual box size
    lamda = tf.cast(
        1 - (r_w * r_h) / (IM_SIZE * IM_SIZE),
        dtype=tf.float32
    )
    
    # Mix labels
    mixed_label = lamda * tf.cast(label_1, dtype=tf.float32) + \
                  (1 - lamda) * tf.cast(label_2, dtype=tf.float32)
    
    return mixed_image, mixed_label


def resize_rescale(image, label):
    """Resize and rescale images"""
    IM_SIZE = CONFIG['IM_SIZE']
    image = tf.image.resize(image, (IM_SIZE, IM_SIZE)) / 255.0
    return image, label


def create_cutmix_dataset(dataset, batch_size=32, alpha=0.2):
    """Create dataset with CutMix augmentation
    
    Args:
        dataset: Input TensorFlow dataset
        batch_size: Batch size for dataset
        alpha: Beta distribution parameter for mixing
        
    Returns:
        CutMix augmented dataset
    """
    # Create two shuffled versions of the dataset
    train_dataset_1 = (
        dataset
        .shuffle(buffer_size=4096)
        .map(resize_rescale, num_parallel_calls=tf.data.AUTOTUNE)
    )
    
    train_dataset_2 = (
        dataset
        .shuffle(buffer_size=4096)
        .map(resize_rescale, num_parallel_calls=tf.data.AUTOTUNE)
    )
    
    # Zip the two datasets
    mixed_dataset = tf.data.Dataset.zip((train_dataset_1, train_dataset_2))
    
    # Apply CutMix augmentation
    mixed_dataset = (
        mixed_dataset
        .shuffle(buffer_size=4096, reshuffle_each_iteration=True)
        .map(
            lambda x, y: cutmix_augmentation(x, y, alpha=alpha),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    return mixed_dataset
