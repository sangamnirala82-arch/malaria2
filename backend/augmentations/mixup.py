"""
MixUp Data Augmentation

MixUp creates new training samples by mixing two images and their labels.
This helps the model generalize better and prevents overfitting.

Paper: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2017)
"""

import tensorflow as tf
try:
    import tensorflow_probability as tfp
    TFP_AVAILABLE = True
except ImportError:
    TFP_AVAILABLE = False
    print("⚠️  TensorFlow Probability not installed. MixUp will use uniform mixing.")


def mixup_augmentation(image1, label1, image2, label2, alpha=0.2):
    """
    Apply MixUp augmentation to two images and labels.
    
    Args:
        image1: First image tensor
        label1: First image label
        image2: Second image tensor  
        label2: Second image label
        alpha: Beta distribution parameter (default: 0.2)
        
    Returns:
        Mixed image and label
    """
    if TFP_AVAILABLE:
        # Use Beta distribution for better mixing
        beta_dist = tfp.distributions.Beta(alpha, alpha)
        lamda = beta_dist.sample(1)[0]
    else:
        # Fallback to uniform distribution
        lamda = tf.random.uniform(shape=[], minval=0.0, maxval=1.0)
    
    # Mix images
    mixed_image = lamda * image1 + (1 - lamda) * image2
    
    # Mix labels (for binary classification, convert to float)
    mixed_label = lamda * tf.cast(label1, tf.float32) + (1 - lamda) * tf.cast(label2, tf.float32)
    
    return mixed_image, mixed_label


def create_mixup_dataset(dataset, alpha=0.2):
    """
    Create a dataset with MixUp augmentation.
    
    Args:
        dataset: TensorFlow dataset with (image, label) pairs
        alpha: Beta distribution parameter
        
    Returns:
        Dataset with MixUp applied
    """
    # Create two shuffled copies of the dataset
    dataset1 = dataset.shuffle(buffer_size=4096)
    dataset2 = dataset.shuffle(buffer_size=4096)
    
    # Zip them together
    mixed_dataset = tf.data.Dataset.zip((dataset1, dataset2))
    
    # Apply MixUp
    def apply_mixup(ds1, ds2):
        (image1, label1), (image2, label2) = ds1, ds2
        return mixup_augmentation(image1, label1, image2, label2, alpha)
    
    return mixed_dataset.map(apply_mixup, num_parallel_calls=tf.data.AUTOTUNE)


def get_mixup_info():
    """Return information about MixUp augmentation."""
    return {
        'name': 'MixUp',
        'description': 'Mixes two images and their labels with random weight',
        'benefits': [
            'Better generalization',
            'Reduced overfitting',
            'Smoother decision boundaries',
            'Improved calibration'
        ],
        'paper': 'mixup: Beyond Empirical Risk Minimization (Zhang et al., 2017)',
        'recommended_alpha': 0.2,
        'requires_tfp': TFP_AVAILABLE
    }


if __name__ == "__main__":
    # Test MixUp
    print("Testing MixUp Augmentation...")
    info = get_mixup_info()
    print(f"\n{info['name']}: {info['description']}")
    print(f"TensorFlow Probability Available: {info['requires_tfp']}")
    print("\nBenefits:")
    for benefit in info['benefits']:
        print(f"  - {benefit}")
