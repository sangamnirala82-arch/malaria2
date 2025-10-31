"""
CutMix Data Augmentation

CutMix cuts and pastes patches between training images.
The labels are mixed proportionally to the area of the patches.

Paper: "CutMix: Regularization Strategy to Train Strong Classifiers" (Yun et al., 2019)
"""

import tensorflow as tf
try:
    import tensorflow_probability as tfp
    TFP_AVAILABLE = True
except ImportError:
    TFP_AVAILABLE = False
    print("⚠️  TensorFlow Probability not installed. CutMix will use uniform mixing.")


def get_random_box(image_size, lamda):
    """
    Generate random box coordinates for CutMix.
    
    Args:
        image_size: Size of the image (assuming square)
        lamda: Mixing ratio
        
    Returns:
        Box coordinates (r_y, r_x, r_h, r_w)
    """
    # Calculate box dimensions
    r_w = tf.cast(image_size * tf.math.sqrt(1 - lamda), dtype=tf.int32)
    r_h = tf.cast(image_size * tf.math.sqrt(1 - lamda), dtype=tf.int32)
    
    # Get random center point
    r_x = tf.cast(tf.random.uniform([], 0, image_size), dtype=tf.int32)
    r_y = tf.cast(tf.random.uniform([], 0, image_size), dtype=tf.int32)
    
    # Clip coordinates to image boundaries
    r_x = tf.clip_by_value(r_x - r_w // 2, 0, image_size)
    r_y = tf.clip_by_value(r_y - r_h // 2, 0, image_size)
    
    x_b_r = tf.clip_by_value(r_x + r_w // 2, 0, image_size)
    y_b_r = tf.clip_by_value(r_y + r_h // 2, 0, image_size)
    
    # Calculate actual width and height (after clipping)
    r_w = x_b_r - r_x
    r_h = y_b_r - r_y
    
    # Ensure minimum size of 1
    r_w = tf.maximum(r_w, 1)
    r_h = tf.maximum(r_h, 1)
    
    return r_y, r_x, r_h, r_w


def cutmix_augmentation(image1, label1, image2, label2, image_size=224, alpha=1.0):
    """
    Apply CutMix augmentation to two images and labels.
    
    Args:
        image1: First image tensor
        label1: First image label
        image2: Second image tensor
        label2: Second image label
        image_size: Size of images (assuming square)
        alpha: Beta distribution parameter (default: 1.0)
        
    Returns:
        Mixed image and label
    """
    if TFP_AVAILABLE:
        # Use Beta distribution
        beta_dist = tfp.distributions.Beta(alpha, alpha)
        lamda = beta_dist.sample(1)[0]
    else:
        # Fallback to uniform distribution
        lamda = tf.random.uniform(shape=[], minval=0.3, maxval=0.7)
    
    # Get random box coordinates
    r_y, r_x, r_h, r_w = get_random_box(image_size, lamda)
    
    # Crop patch from image2
    crop_2 = tf.image.crop_to_bounding_box(image2, r_y, r_x, r_h, r_w)
    
    # Pad to original size
    pad_2 = tf.image.pad_to_bounding_box(crop_2, r_y, r_x, image_size, image_size)
    
    # Crop same area from image1 (to be replaced)
    crop_1 = tf.image.crop_to_bounding_box(image1, r_y, r_x, r_h, r_w)
    pad_1 = tf.image.pad_to_bounding_box(crop_1, r_y, r_x, image_size, image_size)
    
    # Create mixed image: keep image1 except in the box area
    mixed_image = image1 - pad_1 + pad_2
    
    # Adjust lambda based on actual area
    lamda = tf.cast(1 - (r_w * r_h) / (image_size * image_size), dtype=tf.float32)
    
    # Mix labels proportionally
    mixed_label = lamda * tf.cast(label1, tf.float32) + (1 - lamda) * tf.cast(label2, tf.float32)
    
    return mixed_image, mixed_label


def create_cutmix_dataset(dataset, image_size=224, alpha=1.0):
    """
    Create a dataset with CutMix augmentation.
    
    Args:
        dataset: TensorFlow dataset with (image, label) pairs
        image_size: Size of images (assuming square)
        alpha: Beta distribution parameter
        
    Returns:
        Dataset with CutMix applied
    """
    # Create two shuffled copies of the dataset
    dataset1 = dataset.shuffle(buffer_size=4096)
    dataset2 = dataset.shuffle(buffer_size=4096)
    
    # Zip them together
    mixed_dataset = tf.data.Dataset.zip((dataset1, dataset2))
    
    # Apply CutMix
    def apply_cutmix(ds1, ds2):
        (image1, label1), (image2, label2) = ds1, ds2
        return cutmix_augmentation(image1, label1, image2, label2, image_size, alpha)
    
    return mixed_dataset.map(apply_cutmix, num_parallel_calls=tf.data.AUTOTUNE)


def get_cutmix_info():
    """Return information about CutMix augmentation."""
    return {
        'name': 'CutMix',
        'description': 'Cuts and pastes patches between images with proportional label mixing',
        'benefits': [
            'Better localization ability',
            'Improved robustness',
            'Efficient use of training data',
            'Handles occlusions better'
        ],
        'paper': 'CutMix: Regularization Strategy to Train Strong Classifiers (Yun et al., 2019)',
        'recommended_alpha': 1.0,
        'requires_tfp': TFP_AVAILABLE
    }


if __name__ == "__main__":
    # Test CutMix
    print("Testing CutMix Augmentation...")
    info = get_cutmix_info()
    print(f"\n{info['name']}: {info['description']}")
    print(f"TensorFlow Probability Available: {info['requires_tfp']}")
    print("\nBenefits:")
    for benefit in info['benefits']:
        print(f"  - {benefit}")
