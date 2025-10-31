"""Model architecture for Malaria Detection"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, MaxPool2D, Dense, Flatten, 
    BatchNormalization, Dropout, Input
)
from tensorflow.keras.regularizers import L2
from config import CONFIG


def create_lenet_model():
    """Create LeNet-based CNN model for malaria detection"""
    
    IM_SIZE = CONFIG['IM_SIZE']
    DROPOUT_RATE = CONFIG['DROPOUT_RATE']
    REGULARIZATION_RATE = CONFIG['REGULARIZATION_RATE']
    N_FILTERS = CONFIG['N_FILTERS']
    KERNEL_SIZE = CONFIG['KERNEL_SIZE']
    POOL_SIZE = CONFIG['POOL_SIZE']
    N_STRIDES = CONFIG['N_STRIDES']
    N_DENSE_1 = CONFIG['N_DENSE_1']
    N_DENSE_2 = CONFIG['N_DENSE_2']
    
    model = tf.keras.Sequential([
        # Input layer
        Input(shape=(IM_SIZE, IM_SIZE, 3)),
        
        # First Convolutional Block
        Conv2D(
            filters=N_FILTERS,
            kernel_size=KERNEL_SIZE,
            strides=N_STRIDES,
            padding='valid',
            activation='relu',
            kernel_regularizer=L2(REGULARIZATION_RATE)
        ),
        BatchNormalization(),
        MaxPool2D(pool_size=POOL_SIZE, strides=N_STRIDES*2),
        Dropout(rate=DROPOUT_RATE),
        
        # Second Convolutional Block
        Conv2D(
            filters=N_FILTERS*2 + 4,
            kernel_size=KERNEL_SIZE,
            strides=N_STRIDES,
            padding='valid',
            activation='relu',
            kernel_regularizer=L2(REGULARIZATION_RATE)
        ),
        BatchNormalization(),
        MaxPool2D(pool_size=POOL_SIZE, strides=N_STRIDES*2),
        
        # Flatten and Dense Layers
        Flatten(),
        
        Dense(
            N_DENSE_1,
            activation='relu',
            kernel_regularizer=L2(REGULARIZATION_RATE)
        ),
        BatchNormalization(),
        Dropout(rate=DROPOUT_RATE),
        
        Dense(
            N_DENSE_2,
            activation='relu',
            kernel_regularizer=L2(REGULARIZATION_RATE)
        ),
        BatchNormalization(),
        
        # Output layer (binary classification)
        Dense(1, activation='sigmoid'),
    ], name='LeNet_Malaria_Detector')
    
    return model


def get_model_summary():
    """Print model architecture summary"""
    model = create_lenet_model()
    model.summary()
    return model
