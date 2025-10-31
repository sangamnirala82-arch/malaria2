"""Inference script for Malaria Detection Model"""

import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import argparse


def load_model(model_path):
    """Load trained model"""
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    return model


def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess a single image for inference"""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, target_size)
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image


def predict(model, image_path):
    """Make prediction on a single image"""
    # Preprocess image
    image = preprocess_image(image_path)
    
    # Make prediction
    prediction = model.predict(image, verbose=0)[0][0]
    
    # Interpret result
    if prediction < 0.5:
        label = "Parasitized"
        confidence = (1 - prediction) * 100
    else:
        label = "Uninfected"
        confidence = prediction * 100
    
    return label, confidence, prediction


def main():
    parser = argparse.ArgumentParser(description='Malaria Detection Inference')
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to cell image for prediction'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='./models/malaria_model_final.h5',
        help='Path to trained model (default: ./models/malaria_model_final.h5)'
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        return
    
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        print("Please train the model first by running: python train.py")
        return
    
    # Load model
    model = load_model(args.model)
    
    # Make prediction
    print(f"\nAnalyzing image: {args.image}")
    print("-" * 50)
    
    label, confidence, raw_score = predict(model, args.image)
    
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Raw Score: {raw_score:.4f}")
    print("-" * 50)
    
    if label == "Parasitized":
        print("⚠️  WARNING: This cell appears to be infected with malaria.")
    else:
        print("✓ This cell appears to be healthy (uninfected).")


if __name__ == "__main__":
    main()
