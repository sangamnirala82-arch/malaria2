"""
TEST RUN - Quick verification script
This runs a minimal version of the training to verify everything works
"""

import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall

from config import CONFIG
from data_loader import get_prepared_datasets
from model import create_lenet_model

# Override config for quick test
CONFIG['N_EPOCHS'] = 1  # Just 1 epoch for testing
CONFIG['BATCH_SIZE'] = 64  # Larger batch for faster execution

def quick_test():
    """Quick test to verify everything works"""
    print("="*60)
    print("QUICK TEST - Verifying Malaria Detection Setup")
    print("="*60)
    
    # Load datasets
    print("\n1. Loading datasets...")
    try:
        train_dataset, val_dataset, test_dataset, dataset_info = get_prepared_datasets()
        print("   ✓ Datasets loaded successfully!")
    except Exception as e:
        print(f"   ✗ Error loading datasets: {e}")
        return False
    
    # Create model
    print("\n2. Creating model...")
    try:
        model = create_lenet_model()
        print("   ✓ Model created successfully!")
        print(f"   Model has {model.count_params():,} parameters")
    except Exception as e:
        print(f"   ✗ Error creating model: {e}")
        return False
    
    # Compile model
    print("\n3. Compiling model...")
    try:
        model.compile(
            optimizer=Adam(learning_rate=CONFIG['LEARNING_RATE']),
            loss=BinaryCrossentropy(),
            metrics=[BinaryAccuracy(name='accuracy'), Precision(), Recall()]
        )
        print("   ✓ Model compiled successfully!")
    except Exception as e:
        print(f"   ✗ Error compiling model: {e}")
        return False
    
    # Test training on a few batches
    print("\n4. Testing training (5 batches only)...")
    try:
        # Take only 5 batches for quick test
        test_train_dataset = train_dataset.take(5)
        history = model.fit(
            test_train_dataset,
            epochs=1,
            verbose=1
        )
        print("   ✓ Training test successful!")
        print(f"   Final loss: {history.history['loss'][-1]:.4f}")
        print(f"   Final accuracy: {history.history['accuracy'][-1]:.4f}")
    except Exception as e:
        print(f"   ✗ Error during training: {e}")
        return False
    
    # Test inference
    print("\n5. Testing inference...")
    try:
        sample_batch = next(iter(test_dataset))
        predictions = model.predict(sample_batch[0], verbose=0)
        print("   ✓ Inference test successful!")
        print(f"   Sample prediction: {predictions[0][0]:.4f}")
    except Exception as e:
        print(f"   ✗ Error during inference: {e}")
        return False
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nYour setup is working correctly!")
    print("To run full training, execute: python train.py")
    print("="*60)
    
    return True

if __name__ == "__main__":
    # Set random seeds
    tf.random.set_seed(42)
    
    # Disable WandB for quick test
    os.environ['WANDB_MODE'] = 'disabled'
    
    # Run test
    success = quick_test()
    
    if not success:
        print("\n❌ Tests failed. Please check the errors above.")
        exit(1)
