"""
Evaluation metrics and visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import torch


EMOTION_LABELS = [
    'Neutral', 'Happy', 'Sad', 'Surprise', 
    'Fear', 'Disgust', 'Anger', 'Contempt'
]


def compute_metrics(y_true, y_pred, average='macro'):
    """
    Compute classification metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method ('macro', 'weighted', 'micro')
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    return metrics


def compute_per_class_metrics(y_true, y_pred, labels=EMOTION_LABELS):
    """
    Compute per-class metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Class labels
    
    Returns:
        Dictionary with per-class metrics
    """
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    per_class_metrics = {}
    for i, label in enumerate(labels):
        per_class_metrics[label] = {
            'precision': precision[i] if i < len(precision) else 0.0,
            'recall': recall[i] if i < len(recall) else 0.0,
            'f1': f1[i] if i < len(f1) else 0.0
        }
    
    return per_class_metrics


def plot_confusion_matrix(y_true, y_pred, labels=EMOTION_LABELS, 
                          save_path=None, normalize=False):
    """
    Plot confusion matrix
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Class labels
        save_path: Path to save figure
        normalize: Whether to normalize confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and accuracy curves)
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.close()


def plot_roc_curves(y_true, y_probs, labels=EMOTION_LABELS, save_path=None):
    """
    Plot ROC curves for multi-class classification
    
    Args:
        y_true: Ground truth labels (one-hot encoded)
        y_probs: Predicted probabilities
        labels: Class labels
        save_path: Path to save figure
    """
    num_classes = len(labels)
    
    plt.figure(figsize=(12, 10))
    
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
        auc_score = roc_auc_score(y_true[:, i], y_probs[:, i])
        plt.plot(fpr, tpr, linewidth=2, label=f'{labels[i]} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Multi-class Emotion Recognition', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    plt.close()


def print_classification_report(y_true, y_pred, labels=EMOTION_LABELS):
    """Print detailed classification report"""
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(y_true, y_pred, target_names=labels, digits=4))
    print("="*80)


def save_metrics_to_csv(metrics_dict, save_path):
    """
    Save metrics to CSV file
    
    Args:
        metrics_dict: Dictionary of metrics
        save_path: Path to save CSV
    """
    import pandas as pd
    
    df = pd.DataFrame([metrics_dict])
    df.to_csv(save_path, index=False)
    print(f"Metrics saved to {save_path}")


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
