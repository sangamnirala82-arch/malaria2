"""
Test Script for Malaria Detection Model
Tests model accuracy and efficiency on random images from the dataset
"""

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from config import CONFIG, MODEL_SAVE_PATH
from data_loader import get_prepared_datasets


class MalariaModelTester:
    def __init__(self, model_path=None):
        """Initialize the tester with optional model path"""
        # Default to best_model.h5 (best validation accuracy during training)
        self.model_path = model_path or os.path.join(MODEL_SAVE_PATH, 'best_model.h5')
        self.model = None
        self.dataset_info = None
        self.test_dataset = None
        self.class_names = ['Parasitized', 'Uninfected']
        
    def load_model(self):
        """Load the trained model"""
        print("=" * 70)
        print("Loading Trained Model")
        print("=" * 70)
        
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found at: {self.model_path}")
            print("\nAvailable models in ./models/:")
            models_dir = MODEL_SAVE_PATH
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
                if model_files:
                    for model_file in model_files:
                        print(f"  - {model_file}")
                else:
                    print("  (No .h5 model files found)")
            print("\nOptions:")
            print("1. Train the model first: python train.py")
            print("2. Or run quick training test: python test.py --quick-train")
            print("3. Or specify a model: python test.py --model-path ./models/your_model.h5")
            return False
        
        print(f"📂 Loading model from: {self.model_path}")
        model_name = os.path.basename(self.model_path)
        if 'best_model' in model_name:
            print("   ℹ️  This is the best model (highest validation accuracy during training)")
        
        self.model = tf.keras.models.load_model(self.model_path)
        print("✅ Model loaded successfully!")
        print(f"   Model has {self.model.count_params():,} parameters")
        return True
    
    def load_dataset(self):
        """Load the test dataset"""
        print("\n" + "=" * 70)
        print("Loading Dataset")
        print("=" * 70)
        
        # Load datasets
        _, _, test_dataset, dataset_info = get_prepared_datasets()
        
        # Convert test dataset to unbatched for random sampling
        self.test_dataset = test_dataset.unbatch()
        self.dataset_info = dataset_info
        
        print("✅ Dataset loaded successfully!")
        return True
    
    def test_random_samples(self, num_samples=10):
        """Test model on random samples and display results"""
        print("\n" + "=" * 70)
        print(f"Testing on {num_samples} Random Images")
        print("=" * 70)
        
        # Get random samples
        samples = list(self.test_dataset.shuffle(1000).take(num_samples))
        
        predictions_list = []
        true_labels = []
        inference_times = []
        
        print("\nImage | True Label    | Predicted     | Confidence | Time (ms) | Correct")
        print("-" * 70)
        
        for idx, (image, label) in enumerate(samples, 1):
            # Measure inference time
            start_time = time.time()
            
            # Make prediction
            image_batch = tf.expand_dims(image, 0)
            prediction = self.model.predict(image_batch, verbose=0)[0][0]
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            inference_times.append(inference_time)
            
            # Get predicted class
            predicted_class = 1 if prediction > 0.5 else 0
            confidence = prediction if predicted_class == 1 else (1 - prediction)
            
            # Store for metrics
            predictions_list.append(predicted_class)
            true_labels.append(label.numpy())
            
            # Determine if correct
            is_correct = "✅" if predicted_class == label.numpy() else "❌"
            
            # Print result
            print(f"{idx:5d} | {self.class_names[label.numpy()]:13s} | "
                  f"{self.class_names[predicted_class]:13s} | "
                  f"{confidence*100:6.2f}%   | {inference_time:7.2f} | {is_correct}")
        
        return np.array(predictions_list), np.array(true_labels), inference_times
    
    def calculate_metrics(self, predictions, true_labels):
        """Calculate and display accuracy metrics"""
        print("\n" + "=" * 70)
        print("Performance Metrics")
        print("=" * 70)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == true_labels) * 100
        
        print(f"\n📊 Overall Accuracy: {accuracy:.2f}%")
        print(f"   Correct predictions: {np.sum(predictions == true_labels)}/{len(true_labels)}")
        
        # Detailed classification report
        print("\n📈 Detailed Classification Report:")
        print("-" * 70)
        report = classification_report(
            true_labels, 
            predictions, 
            target_names=self.class_names,
            digits=4
        )
        print(report)
        
        return accuracy
    
    def show_confusion_matrix(self, predictions, true_labels):
        """Display confusion matrix"""
        print("\n" + "=" * 70)
        print("Confusion Matrix")
        print("=" * 70)
        
        cm = confusion_matrix(true_labels, predictions)
        
        print("\n           Predicted")
        print("         Parasit. | Uninfect.")
        print("Actual  " + "-" * 30)
        print(f"Parasit. | {cm[0][0]:6d}   | {cm[0][1]:6d}")
        print(f"Uninfect.| {cm[1][0]:6d}   | {cm[1][1]:6d}")
        
        # Calculate per-class metrics
        if cm[0][0] + cm[0][1] > 0:
            parasitized_acc = cm[0][0] / (cm[0][0] + cm[0][1]) * 100
            print(f"\n✓ Parasitized Detection Accuracy: {parasitized_acc:.2f}%")
        
        if cm[1][0] + cm[1][1] > 0:
            uninfected_acc = cm[1][1] / (cm[1][0] + cm[1][1]) * 100
            print(f"✓ Uninfected Detection Accuracy: {uninfected_acc:.2f}%")
    
    def show_inference_performance(self, inference_times):
        """Display inference performance metrics"""
        print("\n" + "=" * 70)
        print("Inference Performance")
        print("=" * 70)
        
        avg_time = np.mean(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        
        print(f"\n⚡ Average Inference Time: {avg_time:.2f} ms")
        print(f"   Fastest: {min_time:.2f} ms")
        print(f"   Slowest: {max_time:.2f} ms")
        print(f"   Throughput: {1000/avg_time:.2f} images/second")
    
    def visualize_predictions(self, num_images=6):
        """Visualize predictions on random images"""
        print("\n" + "=" * 70)
        print(f"Generating Visualization for {num_images} Images")
        print("=" * 70)
        
        # Get random samples
        samples = list(self.test_dataset.shuffle(1000).take(num_images))
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, (image, label) in enumerate(samples):
            # Make prediction
            image_batch = tf.expand_dims(image, 0)
            prediction = self.model.predict(image_batch, verbose=0)[0][0]
            
            predicted_class = 1 if prediction > 0.5 else 0
            confidence = prediction if predicted_class == 1 else (1 - prediction)
            
            # Plot image
            axes[idx].imshow(image)
            axes[idx].axis('off')
            
            # Create title with color coding
            true_label = self.class_names[label.numpy()]
            pred_label = self.class_names[predicted_class]
            
            if predicted_class == label.numpy():
                color = 'green'
                status = '✓ Correct'
            else:
                color = 'red'
                status = '✗ Wrong'
            
            title = f"True: {true_label}\nPred: {pred_label} ({confidence*100:.1f}%)\n{status}"
            axes[idx].set_title(title, color=color, fontsize=10, weight='bold')
        
        plt.tight_layout()
        
        # Save figure
        output_path = 'test_predictions_visualization.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"\n✅ Visualization saved to: {output_path}")
        plt.close()
    
    def test_full_dataset(self, max_samples=1000):
        """Test on full test dataset (or subset)"""
        print("\n" + "=" * 70)
        print(f"Testing on Full Dataset (max {max_samples} samples)")
        print("=" * 70)
        
        all_predictions = []
        all_labels = []
        
        # Batch the dataset for faster processing
        batched_dataset = self.test_dataset.batch(32).take(max_samples // 32)
        
        print("\nProcessing batches...")
        for batch_images, batch_labels in batched_dataset:
            predictions = self.model.predict(batch_images, verbose=0)
            predicted_classes = (predictions > 0.5).astype(int).flatten()
            
            all_predictions.extend(predicted_classes)
            all_labels.extend(batch_labels.numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        print(f"✅ Tested on {len(all_predictions)} images")
        
        return all_predictions, all_labels
    
    def run_comprehensive_test(self, num_random_samples=20, visualize=True):
        """Run comprehensive testing suite"""
        print("\n" + "=" * 70)
        print("🧪 MALARIA DETECTION MODEL - COMPREHENSIVE TEST")
        print("=" * 70)
        
        # Load model
        if not self.load_model():
            return False
        
        # Load dataset
        if not self.load_dataset():
            return False
        
        # Test on random samples
        predictions, true_labels, inference_times = self.test_random_samples(num_random_samples)
        
        # Calculate metrics
        accuracy = self.calculate_metrics(predictions, true_labels)
        
        # Show confusion matrix
        self.show_confusion_matrix(predictions, true_labels)
        
        # Show inference performance
        self.show_inference_performance(inference_times)
        
        # Visualize predictions
        if visualize:
            self.visualize_predictions(6)
        
        # Optional: Test on larger dataset
        print("\n" + "=" * 70)
        user_input = input("Test on full dataset (up to 1000 samples)? (y/n): ")
        if user_input.lower() == 'y':
            full_predictions, full_labels = self.test_full_dataset(1000)
            print("\n📊 Full Dataset Results:")
            self.calculate_metrics(full_predictions, full_labels)
            self.show_confusion_matrix(full_predictions, full_labels)
        
        # Final summary
        print("\n" + "=" * 70)
        print("✅ TESTING COMPLETE")
        print("=" * 70)
        print(f"\n🎯 Summary:")
        print(f"   • Tested Samples: {num_random_samples}")
        print(f"   • Accuracy: {accuracy:.2f}%")
        print(f"   • Avg Inference Time: {np.mean(inference_times):.2f} ms")
        print(f"   • Model: {self.model_path}")
        print("\n" + "=" * 70)
        
        return True


def quick_train_and_test():
    """Quick training for testing purposes (1 epoch, small subset)"""
    print("\n" + "=" * 70)
    print("⚡ QUICK TRAINING MODE")
    print("=" * 70)
    print("\nTraining a small model for testing purposes...")
    print("(This is NOT the full training - just for demonstration)")
    
    from model import create_lenet_model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import BinaryCrossentropy
    from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
    
    # Load dataset
    train_dataset, val_dataset, test_dataset, _ = get_prepared_datasets()
    
    # Create model
    model = create_lenet_model()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=BinaryCrossentropy(),
        metrics=[BinaryAccuracy(name='accuracy'), Precision(), Recall()]
    )
    
    # Train on small subset (just 20 batches for quick test)
    print("\nTraining on 20 batches...")
    train_small = train_dataset.take(20)
    model.fit(train_small, epochs=1, verbose=1)
    
    # Save model
    quick_model_path = os.path.join(MODEL_SAVE_PATH, 'quick_test_model.h5')
    model.save(quick_model_path)
    print(f"\n✅ Quick model saved to: {quick_model_path}")
    
    return quick_model_path


def main():
    import sys
    
    # Check command line arguments
    if '--quick-train' in sys.argv:
        # Quick training mode
        model_path = quick_train_and_test()
        tester = MalariaModelTester(model_path)
    else:
        # Use trained model
        tester = MalariaModelTester()
    
    # Run comprehensive test
    success = tester.run_comprehensive_test(
        num_random_samples=20,
        visualize=True
    )
    
    if not success:
        print("\n💡 Tip: If you don't have a trained model yet, run:")
        print("   python test.py --quick-train")
        print("\n   This will train a small model quickly for testing purposes.")


if __name__ == "__main__":
    # Disable unnecessary TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    main()
