"""
Comprehensive Model Evaluation Script
Generates all performance metrics and visualizations for Malaria Detection Model
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score
)

from config import MODEL_SAVE_PATH, LOGS_PATH
from data_loader import get_prepared_datasets


class ComprehensiveEvaluator:
    """Comprehensive evaluation and visualization for trained models"""
    
    def __init__(self, model_path, results_dir='./results'):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to the trained model
            results_dir: Directory to save results
        """
        self.model_path = model_path
        self.results_dir = results_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Set style for plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        print("="*70)
        print("🔬 COMPREHENSIVE MODEL EVALUATION")
        print("="*70)
        
    def load_model(self):
        """Load the trained model"""
        print(f"\n📂 Loading model from: {self.model_path}")
        self.model = load_model(self.model_path)
        print("✅ Model loaded successfully!")
        print(f"   Total parameters: {self.model.count_params():,}")
        
    def load_data(self):
        """Load test dataset"""
        print("\n📊 Loading test dataset...")
        _, _, self.test_dataset, self.dataset_info = get_prepared_datasets()
        print("✅ Test dataset loaded!")
        
    def evaluate_model(self):
        """Evaluate model and collect predictions"""
        print("\n🧪 Evaluating model on test set...")
        
        # Get predictions and true labels
        y_true = []
        y_pred_proba = []
        
        for images, labels in self.test_dataset:
            predictions = self.model.predict(images, verbose=0)
            y_pred_proba.extend(predictions.flatten())
            y_true.extend(labels.numpy())
        
        self.y_true = np.array(y_true)
        self.y_pred_proba = np.array(y_pred_proba)
        self.y_pred = (self.y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        self.accuracy = np.mean(self.y_pred == self.y_true)
        self.precision = np.sum((self.y_pred == 1) & (self.y_true == 1)) / np.sum(self.y_pred == 1)
        self.recall = np.sum((self.y_pred == 1) & (self.y_true == 1)) / np.sum(self.y_true == 1)
        self.f1 = f1_score(self.y_true, self.y_pred)
        
        # Calculate ROC AUC
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba)
        self.roc_auc = auc(fpr, tpr)
        
        print("✅ Evaluation complete!")
        
    def print_metrics(self):
        """Print performance metrics"""
        print("\n" + "="*70)
        print("📊 PERFORMANCE METRICS")
        print("="*70)
        print(f"🎯 Accuracy:  {self.accuracy:.4f} ({self.accuracy*100:.2f}%)")
        print(f"🎯 Precision: {self.precision:.4f} ({self.precision*100:.2f}%)")
        print(f"🎯 Recall:    {self.recall:.4f} ({self.recall*100:.2f}%)")
        print(f"🎯 F1-Score:  {self.f1:.4f} ({self.f1*100:.2f}%)")
        print(f"🎯 AUC Score: {self.roc_auc:.4f} ({self.roc_auc*100:.2f}%)")
        print("="*70)
        
    def save_metrics(self):
        """Save metrics to JSON and text files"""
        metrics = {
            'timestamp': self.timestamp,
            'model_path': self.model_path,
            'accuracy': float(self.accuracy),
            'precision': float(self.precision),
            'recall': float(self.recall),
            'f1_score': float(self.f1),
            'auc_score': float(self.roc_auc),
            'total_samples': len(self.y_true),
            'correct_predictions': int(np.sum(self.y_pred == self.y_true)),
            'class_distribution': {
                'parasitized': int(np.sum(self.y_true == 1)),
                'uninfected': int(np.sum(self.y_true == 0))
            }
        }
        
        # Save as JSON
        json_path = os.path.join(self.results_dir, f'metrics_{self.timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"\n💾 Metrics saved to: {json_path}")
        
        # Save as text
        txt_path = os.path.join(self.results_dir, f'metrics_{self.timestamp}.txt')
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MALARIA DETECTION MODEL - EVALUATION RESULTS\n")
            f.write("="*70 + "\n\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Path: {self.model_path}\n\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Accuracy:  {self.accuracy:.4f} ({self.accuracy*100:.2f}%)\n")
            f.write(f"Precision: {self.precision:.4f} ({self.precision*100:.2f}%)\n")
            f.write(f"Recall:    {self.recall:.4f} ({self.recall*100:.2f}%)\n")
            f.write(f"F1-Score:  {self.f1:.4f} ({self.f1*100:.2f}%)\n")
            f.write(f"AUC Score: {self.roc_auc:.4f} ({self.roc_auc*100:.2f}%)\n\n")
            f.write("DATASET INFORMATION\n")
            f.write("-"*70 + "\n")
            f.write(f"Total Test Samples: {len(self.y_true)}\n")
            f.write(f"Correct Predictions: {np.sum(self.y_pred == self.y_true)}\n")
            f.write(f"Parasitized Samples: {np.sum(self.y_true == 1)}\n")
            f.write(f"Uninfected Samples: {np.sum(self.y_true == 0)}\n")
        print(f"💾 Metrics saved to: {txt_path}")
        
    def plot_confusion_matrix(self):
        """Generate and save confusion matrix"""
        print("\n📊 Generating Confusion Matrix...")
        
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Uninfected', 'Parasitized'],
                   yticklabels=['Uninfected', 'Parasitized'],
                   cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix - Malaria Detection\n', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        
        # Add metrics text
        accuracy_text = f'Accuracy: {self.accuracy:.2%}'
        plt.text(0.5, -0.15, accuracy_text, ha='center', transform=plt.gca().transAxes,
                fontsize=12, fontweight='bold', color='green')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, f'confusion_matrix_{self.timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Confusion Matrix saved to: {save_path}")
        
    def plot_roc_curve(self):
        """Generate and save ROC curve"""
        print("\n📈 Generating ROC Curve...")
        
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curve - Malaria Detection\n', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, f'roc_curve_{self.timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ ROC Curve saved to: {save_path}")
        
    def plot_precision_recall_curve(self):
        """Generate and save Precision-Recall curve"""
        print("\n📈 Generating Precision-Recall Curve...")
        
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (F1 = {self.f1:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12, fontweight='bold')
        plt.ylabel('Precision', fontsize=12, fontweight='bold')
        plt.title('Precision-Recall Curve - Malaria Detection\n', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, f'precision_recall_curve_{self.timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Precision-Recall Curve saved to: {save_path}")
        
    def plot_training_history(self):
        """Plot training history from logs"""
        print("\n📈 Generating Training History Plots...")
        
        # Load training history from CSV
        log_file = os.path.join(LOGS_PATH, 'training_log.csv')
        
        if not os.path.exists(log_file):
            print(f"⚠️  Training log not found at: {log_file}")
            return
        
        history_df = pd.read_csv(log_file)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Accuracy
        axes[0, 0].plot(history_df['epoch'], history_df['accuracy'], 
                       label='Training Accuracy', marker='o', linewidth=2)
        axes[0, 0].plot(history_df['epoch'], history_df['val_accuracy'], 
                       label='Validation Accuracy', marker='s', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Model Accuracy Over Epochs', fontsize=13, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(alpha=0.3)
        
        # Plot 2: Loss
        axes[0, 1].plot(history_df['epoch'], history_df['loss'], 
                       label='Training Loss', marker='o', linewidth=2, color='red')
        axes[0, 1].plot(history_df['epoch'], history_df['val_loss'], 
                       label='Validation Loss', marker='s', linewidth=2, color='orange')
        axes[0, 1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Loss', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Model Loss Over Epochs', fontsize=13, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(alpha=0.3)
        
        # Plot 3: AUC
        if 'auc' in history_df.columns and 'val_auc' in history_df.columns:
            axes[1, 0].plot(history_df['epoch'], history_df['auc'], 
                           label='Training AUC', marker='o', linewidth=2, color='green')
            axes[1, 0].plot(history_df['epoch'], history_df['val_auc'], 
                           label='Validation AUC', marker='s', linewidth=2, color='lightgreen')
            axes[1, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
            axes[1, 0].set_ylabel('AUC', fontsize=11, fontweight='bold')
            axes[1, 0].set_title('Model AUC Over Epochs', fontsize=13, fontweight='bold')
            axes[1, 0].legend(fontsize=10)
            axes[1, 0].grid(alpha=0.3)
        
        # Plot 4: Precision and Recall
        if 'precision' in history_df.columns and 'recall' in history_df.columns:
            axes[1, 1].plot(history_df['epoch'], history_df['precision'], 
                           label='Training Precision', marker='o', linewidth=2, color='purple')
            axes[1, 1].plot(history_df['epoch'], history_df['recall'], 
                           label='Training Recall', marker='s', linewidth=2, color='pink')
            if 'val_precision' in history_df.columns:
                axes[1, 1].plot(history_df['epoch'], history_df['val_precision'], 
                               label='Val Precision', marker='^', linewidth=2, color='darkviolet')
            if 'val_recall' in history_df.columns:
                axes[1, 1].plot(history_df['epoch'], history_df['val_recall'], 
                               label='Val Recall', marker='v', linewidth=2, color='hotpink')
            axes[1, 1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
            axes[1, 1].set_ylabel('Score', fontsize=11, fontweight='bold')
            axes[1, 1].set_title('Precision & Recall Over Epochs', fontsize=13, fontweight='bold')
            axes[1, 1].legend(fontsize=9)
            axes[1, 1].grid(alpha=0.3)
        
        plt.suptitle('Training History - Malaria Detection Model', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, f'training_history_{self.timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Training History saved to: {save_path}")
        
    def plot_metrics_comparison(self):
        """Create a bar plot comparing all metrics"""
        print("\n📊 Generating Metrics Comparison Plot...")
        
        metrics = {
            'Accuracy': self.accuracy,
            'Precision': self.precision,
            'Recall': self.recall,
            'F1-Score': self.f1,
            'AUC': self.roc_auc
        }
        
        plt.figure(figsize=(12, 7))
        bars = plt.bar(metrics.keys(), metrics.values(), 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}\n({height*100:.2f}%)',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.ylim([0, 1.1])
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.title('Performance Metrics Comparison - Malaria Detection\n', 
                 fontsize=16, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, f'metrics_comparison_{self.timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Metrics Comparison saved to: {save_path}")
        
    def generate_classification_report(self):
        """Generate and save detailed classification report"""
        print("\n📋 Generating Classification Report...")
        
        report = classification_report(self.y_true, self.y_pred, 
                                      target_names=['Uninfected', 'Parasitized'],
                                      digits=4)
        
        report_path = os.path.join(self.results_dir, f'classification_report_{self.timestamp}.txt')
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CLASSIFICATION REPORT - MALARIA DETECTION\n")
            f.write("="*70 + "\n\n")
            f.write(report)
        
        print(f"✅ Classification Report saved to: {report_path}")
        
    def run_comprehensive_evaluation(self):
        """Run complete evaluation pipeline"""
        # Load model and data
        self.load_model()
        self.load_data()
        
        # Evaluate
        self.evaluate_model()
        self.print_metrics()
        
        # Save metrics
        self.save_metrics()
        
        # Generate all visualizations
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        self.plot_training_history()
        self.plot_metrics_comparison()
        self.generate_classification_report()
        
        print("\n" + "="*70)
        print("✅ COMPREHENSIVE EVALUATION COMPLETE!")
        print("="*70)
        print(f"\n📁 All results saved to: {self.results_dir}")
        print("\nGenerated files:")
        print(f"  • Metrics (JSON & TXT)")
        print(f"  • Confusion Matrix")
        print(f"  • ROC Curve")
        print(f"  • Precision-Recall Curve")
        print(f"  • Training History Plots")
        print(f"  • Metrics Comparison Chart")
        print(f"  • Classification Report")
        print("\n" + "="*70)


def main():
    """Main execution function"""
    model_path = os.path.join(MODEL_SAVE_PATH, 'best_model.h5')
    results_dir = './results'
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(model_path, results_dir)
    
    # Run comprehensive evaluation
    evaluator.run_comprehensive_evaluation()


if __name__ == "__main__":
    main()
