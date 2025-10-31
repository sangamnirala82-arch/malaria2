"""Main training script for Malaria Detection Model"""

import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import (
    BinaryAccuracy, Precision, Recall, AUC,
    TruePositives, FalsePositives, TrueNegatives, FalseNegatives
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
)
import wandb
from wandb.integration.keras import WandbMetricsLogger

from config import (
    CONFIG, WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY,
    MODEL_SAVE_PATH, LOGS_PATH
)
from data_loader import get_prepared_datasets
from model import create_lenet_model


def setup_wandb():
    """Initialize Weights & Biases for experiment tracking"""
    print("Setting up Weights & Biases...")
    os.environ['WANDB_API_KEY'] = WANDB_API_KEY
    
    # Login to wandb
    wandb.login(key=WANDB_API_KEY)
    
    # Initialize wandb run
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config=CONFIG
    )
    print("WandB initialized successfully!")


def compile_model(model):
    """Compile the model with optimizer, loss, and metrics"""
    metrics = [
        TruePositives(name='tp'),
        FalsePositives(name='fp'),
        TrueNegatives(name='tn'),
        FalseNegatives(name='fn'),
        BinaryAccuracy(name='accuracy'),
        Precision(name='precision'),
        Recall(name='recall'),
        AUC(name='auc')
    ]
    
    model.compile(
        optimizer=Adam(learning_rate=CONFIG['LEARNING_RATE']),
        loss=BinaryCrossentropy(),
        metrics=metrics
    )
    print("Model compiled successfully!")
    return model


def get_callbacks():
    """Setup training callbacks"""
    callbacks = [
        # WandB logging
        WandbMetricsLogger(),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            restore_best_weights=True
        ),
        
        # Model checkpoint
        ModelCheckpoint(
            filepath=os.path.join(MODEL_SAVE_PATH, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-7
        ),
        
        # CSV Logger
        CSVLogger(
            os.path.join(LOGS_PATH, 'training_log.csv'),
            append=True
        )
    ]
    return callbacks


def train_model():
    """Main training function"""
    print("="*60)
    print("Starting Malaria Detection Model Training")
    print("="*60)
    
    # Setup WandB
    setup_wandb()
    
    # Load and prepare datasets
    print("\nLoading datasets...")
    train_dataset, val_dataset, test_dataset, dataset_info = get_prepared_datasets()
    print(f"Train dataset: {train_dataset}")
    print(f"Val dataset: {val_dataset}")
    print(f"Test dataset: {test_dataset}")
    
    # Create model
    print("\nCreating model...")
    model = create_lenet_model()
    model.summary()
    
    # Compile model
    print("\nCompiling model...")
    model = compile_model(model)
    
    # Get callbacks
    callbacks = get_callbacks()
    
    # Train model
    print("\nStarting training...")
    print(f"Training for {CONFIG['N_EPOCHS']} epochs")
    print(f"Batch size: {CONFIG['BATCH_SIZE']}")
    print(f"Learning rate: {CONFIG['LEARNING_RATE']}")
    print("="*60)
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=CONFIG['N_EPOCHS'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating model on test set...")
    test_results = model.evaluate(test_dataset, verbose=1)
    
    print("\nTest Results:")
    for metric_name, value in zip(model.metrics_names, test_results):
        print(f"{metric_name}: {value:.4f}")
    
    # Note: best_model.h5 is already saved by ModelCheckpoint callback
    # It contains the model with the best validation accuracy
    best_model_path = os.path.join(MODEL_SAVE_PATH, 'best_model.h5')
    print(f"\n✅ Best model saved to: {best_model_path}")
    print("   (Saved automatically by ModelCheckpoint during training)")
    
    # Save best model to wandb
    if os.path.exists(best_model_path):
        wandb.save(best_model_path)
    
    # Finish wandb run
    wandb.finish()
    
    print("="*60)
    print("Training completed successfully!")
    print("="*60)
    
    return model, history


if __name__ == "__main__":
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    
    # Train the model
    model, history = train_model()
