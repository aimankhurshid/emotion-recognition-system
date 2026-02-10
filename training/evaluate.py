"""
Evaluation script for emotion recognition model
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model, load_checkpoint, EMOTION_LABELS
from utils import (
    get_data_loaders,
    compute_metrics,
    compute_per_class_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    print_classification_report
)


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    return all_labels, all_preds, all_probs


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    train_loader, val_loader, test_loader, _ = get_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print("\nCreating model...")
    model = get_model(
        model_type=args.model_type,
        num_classes=args.num_classes,
        backbone=args.backbone,
        pretrained=False
    )
    model = model.to(device)
    
    print(f"\nLoading checkpoint from {args.checkpoint_path}...")
    load_checkpoint(model, None, args.checkpoint_path, device=device)
    
    y_true, y_pred, y_probs = evaluate_model(model, test_loader, device)
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    macro_metrics = compute_metrics(y_true, y_pred, average='macro')
    weighted_metrics = compute_metrics(y_true, y_pred, average='weighted')
    
    print(f"\nMacro-averaged Metrics:")
    print(f"  Accuracy:  {macro_metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {macro_metrics['precision']:.4f}")
    print(f"  Recall:    {macro_metrics['recall']:.4f}")
    print(f"  F1-Score:  {macro_metrics['f1']:.4f}")
    
    print(f"\nWeighted-averaged Metrics:")
    print(f"  Precision: {weighted_metrics['precision']:.4f}")
    print(f"  Recall:    {weighted_metrics['recall']:.4f}")
    print(f"  F1-Score:  {weighted_metrics['f1']:.4f}")
    
    per_class_metrics = compute_per_class_metrics(y_true, y_pred, EMOTION_LABELS)
    
    print("\nPer-class Metrics:")
    print(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 50)
    for emotion, metrics in per_class_metrics.items():
        print(f"{emotion:<12} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
    
    print_classification_report(y_true, y_pred, EMOTION_LABELS)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, EMOTION_LABELS, save_path=cm_path, normalize=False)
    
    cm_norm_path = os.path.join(args.output_dir, 'confusion_matrix_normalized.png')
    plot_confusion_matrix(y_true, y_pred, EMOTION_LABELS, save_path=cm_norm_path, normalize=True)
    
    y_true_onehot = np.eye(args.num_classes)[y_true]
    roc_path = os.path.join(args.output_dir, 'roc_curves.png')
    plot_roc_curves(y_true_onehot, y_probs, EMOTION_LABELS, save_path=roc_path)
    
    results = {
        'macro_accuracy': macro_metrics['accuracy'],
        'macro_precision': macro_metrics['precision'],
        'macro_recall': macro_metrics['recall'],
        'macro_f1': macro_metrics['f1'],
        'weighted_precision': weighted_metrics['precision'],
        'weighted_recall': weighted_metrics['recall'],
        'weighted_f1': weighted_metrics['f1']
    }
    
    for emotion, metrics in per_class_metrics.items():
        results[f'{emotion}_precision'] = metrics['precision']
        results[f'{emotion}_recall'] = metrics['recall']
        results[f'{emotion}_f1'] = metrics['f1']
    
    results_df = pd.DataFrame([results])
    results_path = os.path.join(args.output_dir, 'evaluation_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    print("\n" + "="*80)
    print("Evaluation completed!")
    print(f"Visualizations saved in: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Emotion Recognition Model')
    
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='../results/visualizations',
                        help='Path to save evaluation results')
    
    parser.add_argument('--model_type', type=str, default='full',
                        choices=['full', 'simple_cnn', 'cnn_attention'],
                        help='Model architecture type')
    parser.add_argument('--backbone', type=str, default='efficientnet_b4',
                        choices=['efficientnet_b4', 'resnet50'],
                        help='CNN backbone')
    parser.add_argument('--num_classes', type=int, default=8,
                        help='Number of emotion classes')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    main(args)
