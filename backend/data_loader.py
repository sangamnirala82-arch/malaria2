"""Data loading and preprocessing for Malaria Detection"""

import tensorflow as tf
import tensorflow_datasets as tfds
from config import (
    CONFIG, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    USE_ADVANCED_AUGMENTATION, AUGMENTATION_METHOD,
    MIXUP_ALPHA, CUTMIX_ALPHA, ALBUMENTATIONS_MODE
)


def load_dataset():
    """Load malaria dataset from TensorFlow Datasets"""
    print("Loading malaria dataset from TensorFlow Datasets...")
    dataset, dataset_info = tfds.load(
        'malaria',
        with_info=True,
        as_supervised=True,
        shuffle_files=True,
        split=['train']
    )
    print(f"Dataset loaded successfully. Total samples: {dataset_info.splits['train'].num_examples}")
    return dataset[0], dataset_info


def splits(dataset, train_ratio, val_ratio, test_ratio):
    """Split dataset into train, validation and test sets"""
    dataset_size = len(dataset)
    
    train_dataset = dataset.take(int(train_ratio * dataset_size))
    val_test_dataset = dataset.skip(int(train_ratio * dataset_size))
    val_dataset = val_test_dataset.take(int(val_ratio * dataset_size))
    test_dataset = val_test_dataset.skip(int(val_ratio * dataset_size))
    
    return train_dataset, val_dataset, test_dataset


@tf.function
def resize_rescale(image, label):
    """Resize and rescale images to standard size and [0,1] range"""
    return tf.image.resize(image, (CONFIG['IM_SIZE'], CONFIG['IM_SIZE'])) / 255.0, label


@tf.function
def augment_layer(image, label):
    """Apply data augmentation to images"""
    # Resize and rescale
    image, label = resize_rescale(image, label)
    
    # Random rotation
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)
    
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    
    return image, label


def prepare_datasets_basic(train_dataset, val_dataset, test_dataset, batch_size):
    """Prepare datasets with basic augmentation"""
    print("Preparing datasets with BASIC augmentation...")
    
    # Training dataset with augmentation
    train_dataset = (
        train_dataset
        .shuffle(buffer_size=1024, reshuffle_each_iteration=True)
        .map(augment_layer, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    # Validation dataset without augmentation
    val_dataset = (
        val_dataset
        .shuffle(buffer_size=32)
        .map(resize_rescale, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
    )
    
    # Test dataset without augmentation
    test_dataset = (
        test_dataset
        .map(resize_rescale, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(1)
    )
    
    print("✓ Basic augmentation applied")
    return train_dataset, val_dataset, test_dataset


def prepare_datasets_advanced(train_dataset, val_dataset, test_dataset, batch_size):
    """Prepare datasets with advanced augmentation"""
    print(f"Preparing datasets with ADVANCED augmentation: {AUGMENTATION_METHOD.upper()}")
    
    # Import advanced augmentation modules
    try:
        from augmentations import (
            create_mixup_dataset,
            create_cutmix_dataset,
            create_albumentations_dataset,
            get_medical_transforms,
            get_aggressive_transforms
        )
        
        # Preprocess first
        train_dataset_preprocessed = train_dataset.map(
            resize_rescale, 
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Apply selected augmentation
        if AUGMENTATION_METHOD == 'mixup':
            print(f"  → Applying MixUp (alpha={MIXUP_ALPHA})")
            train_dataset = create_mixup_dataset(train_dataset_preprocessed, alpha=MIXUP_ALPHA)
            
        elif AUGMENTATION_METHOD == 'cutmix':
            print(f"  → Applying CutMix (alpha={CUTMIX_ALPHA})")
            train_dataset = create_cutmix_dataset(
                train_dataset_preprocessed, 
                image_size=CONFIG['IM_SIZE'],
                alpha=CUTMIX_ALPHA
            )
            
        elif AUGMENTATION_METHOD == 'albumentations':
            print(f"  → Applying Albumentations ({ALBUMENTATIONS_MODE} mode)")
            if ALBUMENTATIONS_MODE == 'aggressive':
                transforms = get_aggressive_transforms(CONFIG['IM_SIZE'])
            else:
                transforms = get_medical_transforms(CONFIG['IM_SIZE'])
            train_dataset = create_albumentations_dataset(
                train_dataset,  # Use raw dataset (albumentations handles preprocessing)
                transforms=transforms,
                image_size=CONFIG['IM_SIZE']
            )
            
        elif AUGMENTATION_METHOD == 'all':
            print("  → Applying ALL augmentations (MixUp + Albumentations)")
            # First apply Albumentations
            transforms = get_medical_transforms(CONFIG['IM_SIZE'])
            train_dataset = create_albumentations_dataset(
                train_dataset,
                transforms=transforms,
                image_size=CONFIG['IM_SIZE']
            )
            # Then apply MixUp
            train_dataset = create_mixup_dataset(train_dataset, alpha=MIXUP_ALPHA)
        
        else:
            print(f"  ⚠️  Unknown augmentation method: {AUGMENTATION_METHOD}")
            print("  → Falling back to basic augmentation")
            return prepare_datasets_basic(train_dataset, val_dataset, test_dataset, batch_size)
        
        # Batch and prefetch
        train_dataset = (
            train_dataset
            .shuffle(buffer_size=1024, reshuffle_each_iteration=True)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        
        print("✓ Advanced augmentation applied successfully!")
        
    except ImportError as e:
        print(f"  ⚠️  Error importing augmentation modules: {e}")
        print("  → Falling back to basic augmentation")
        return prepare_datasets_basic(train_dataset, val_dataset, test_dataset, batch_size)
    
    except Exception as e:
        print(f"  ⚠️  Error applying advanced augmentation: {e}")
        print("  → Falling back to basic augmentation")
        return prepare_datasets_basic(train_dataset, val_dataset, test_dataset, batch_size)
    
    # Val and test datasets (no augmentation)
    val_dataset = (
        val_dataset
        .shuffle(buffer_size=32)
        .map(resize_rescale, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
    )
    
    test_dataset = (
        test_dataset
        .map(resize_rescale, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(1)
    )
    
    return train_dataset, val_dataset, test_dataset


def prepare_datasets(train_dataset, val_dataset, test_dataset, batch_size):
    """Prepare datasets with batching and prefetching"""
    
    if USE_ADVANCED_AUGMENTATION:
        return prepare_datasets_advanced(train_dataset, val_dataset, test_dataset, batch_size)
    else:
        return prepare_datasets_basic(train_dataset, val_dataset, test_dataset, batch_size)


def get_prepared_datasets():
    """Main function to load and prepare all datasets"""
    # Load dataset
    dataset, dataset_info = load_dataset()
    
    # Split into train/val/test
    train_dataset, val_dataset, test_dataset = splits(
        dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )
    
    # Prepare datasets with preprocessing
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        train_dataset, val_dataset, test_dataset, CONFIG['BATCH_SIZE']
    )
    
    print("Datasets prepared successfully!")
    return train_dataset, val_dataset, test_dataset, dataset_info
