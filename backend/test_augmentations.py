"""Test script to verify all augmentation techniques work correctly"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import tensorflow as tf
from config import CONFIG
from data_loader import load_dataset, splits, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
from data_augmentation.mixup import create_mixup_dataset
from data_augmentation.cutmix import create_cutmix_dataset
from data_augmentation.albumentations_transforms import create_albumentations_dataset
from data_augmentation.repeating_the_dataset import create_repeated_dataset

def test_augmentation(name, dataset_fn, train_dataset):
    """Test a single augmentation technique"""
    print(f"\n{'='*60}")
    print(f"Testing {name} augmentation...")
    print('='*60)
    
    try:
        # Create augmented dataset
        aug_dataset = dataset_fn(train_dataset)
        
        # Try to get a batch
        batch = next(iter(aug_dataset))
        images, labels = batch
        
        print(f"✓ {name} augmentation working!")
        print(f"  Batch shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Image range: [{tf.reduce_min(images).numpy():.3f}, {tf.reduce_max(images).numpy():.3f}]")
        
        return True
    except Exception as e:
        print(f"✗ {name} augmentation failed!")
        print(f"  Error: {str(e)}")
        return False

def main():
    """Test all augmentation techniques"""
    print("="*60)
    print("Augmentation Techniques Test")
    print("="*60)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset, dataset_info = load_dataset()
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = splits(
        dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )
    
    batch_size = CONFIG['BATCH_SIZE']
    results = {}
    
    # Test MixUp
    results['MixUp'] = test_augmentation(
        'MixUp',
        lambda ds: create_mixup_dataset(ds, batch_size, alpha=0.2),
        train_dataset
    )
    
    # Test CutMix
    results['CutMix'] = test_augmentation(
        'CutMix',
        lambda ds: create_cutmix_dataset(ds, batch_size, alpha=0.2),
        train_dataset
    )
    
    # Test Albumentations
    results['Albumentations'] = test_augmentation(
        'Albumentations',
        lambda ds: create_albumentations_dataset(ds, batch_size),
        train_dataset
    )
    
    # Test Repeated Dataset
    results['Repeated Dataset'] = test_augmentation(
        'Repeated Dataset',
        lambda ds: create_repeated_dataset(ds, batch_size),
        train_dataset
    )
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{name:20s}: {status}")
    
    all_passed = all(results.values())
    print("="*60)
    if all_passed:
        print("✓ All augmentation techniques are working correctly!")
    else:
        print("✗ Some augmentation techniques failed. Check errors above.")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
