"""MixUp Data Augmentation

MixUp creates virtual training examples by mixing pairs of examples and their labels.
Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization" (2018)
"""

import tensorflow as tf
import tensorflow_probability as tfp
from config import CONFIG


def mixup_augmentation(dataset_1_sample, dataset_2_sample, alpha=0.2):
    """Apply MixUp augmentation to two samples
    
    Args:
        dataset_1_sample: Tuple of (image_1, label_1)
        dataset_2_sample: Tuple of (image_2, label_2)
        alpha: Beta distribution parameter for mixing
        
    Returns:
        Mixed image and label
    """
    (image_1, label_1), (image_2, label_2) = dataset_1_sample, dataset_2_sample
    
    # Sample lambda from Beta distribution
    beta_dist = tfp.distributions.Beta(alpha, alpha)
    lamda = beta_dist.sample(1)[0]
    
    # Mix images
    mixed_image = lamda * image_1 + (1 - lamda) * image_2
    
    # Mix labels
    mixed_label = lamda * tf.cast(label_1, dtype=tf.float32) + \
                  (1 - lamda) * tf.cast(label_2, dtype=tf.float32)
    
    return mixed_image, mixed_label


def resize_rescale(image, label):
    """Resize and rescale images"""
    IM_SIZE = CONFIG['IM_SIZE']
    image = tf.image.resize(image, (IM_SIZE, IM_SIZE)) / 255.0
    return image, label


def create_mixup_dataset(dataset, batch_size=32, alpha=0.2):
    """Create dataset with MixUp augmentation
    
    Args:
        dataset: Input TensorFlow dataset
        batch_size: Batch size for dataset
        alpha: Beta distribution parameter for mixing
        
    Returns:
        MixUp augmented dataset
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
    
    # Apply MixUp augmentation
    mixed_dataset = (
        mixed_dataset
        .shuffle(buffer_size=4096, reshuffle_each_iteration=True)
        .map(
            lambda x, y: mixup_augmentation(x, y, alpha=alpha),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    return mixed_dataset
