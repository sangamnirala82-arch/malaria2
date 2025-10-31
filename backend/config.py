"""Configuration file for Malaria Detection Model Training"""

import os

# WandB Configuration
WANDB_API_KEY = "980ca7615589d93a06f69caac98b8e2ab15721da"
WANDB_PROJECT = "Malaria-Detection"
WANDB_ENTITY = "sangamnirala2004-d-d-beyond"

# Model Hyperparameters
CONFIG = {
    "LEARNING_RATE": 0.001,
    "N_EPOCHS": 5,
    "BATCH_SIZE": 32,
    "DROPOUT_RATE": 0.0,
    "IM_SIZE": 224,
    "REGULARIZATION_RATE": 0.0,
    "N_FILTERS": 6,
    "KERNEL_SIZE": 3,
    "N_STRIDES": 1,
    "POOL_SIZE": 2,
    "N_DENSE_1": 100,
    "N_DENSE_2": 10,
}

# Data Split Ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Paths
MODEL_SAVE_PATH = "./models/"
LOGS_PATH = "./logs/"
WEIGHTS_PATH = "./weights/"

# Create directories if they don't exist
for path in [MODEL_SAVE_PATH, LOGS_PATH, WEIGHTS_PATH]:
    os.makedirs(path, exist_ok=True)
