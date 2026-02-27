"""
Ablation study to validate architecture choices
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import json
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model, WeightedCrossEntropyLoss, save_checkpoint
from utils import get_data_loaders, compute_metrics, AverageMeter
from train import train_epoch, validate


ABLATION_CONFIGS = {
    'baseline_cnn': {
        'model_type': 'simple_cnn',
        'description': 'Baseline CNN (EfficientNetB4 only)'
    },
    'cnn_attention': {
        'model_type': 'cnn_attention',
        'description': 'CNN + Dual Attention (no BiLSTM)'
    },
    'full_model': {
        'model_type': 'full',
        'description': 'Full Model (CNN + Dual Attention + BiLSTM)'
    }
}


def run_ablation_experiment(config_name, config, args, train_loader, val_loader, class_weights, device):
    """Run single ablation experiment"""
    print("\n" + "="*80)
    print(f"Running: {config_name}")
    print(f"Description: {config['description']}")
    print("="*80)
    
    model = get_model(
        model_type=config['model_type'],
        num_classes=args.num_classes,
        backbone=args.backbone,
        pretrained=True,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout
    )
    model = model.to(device)
    
    if args.use_class_weights and class_weights is not None:
        criterion = WeightedCrossEntropyLoss(class_weights, device=device)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    best_acc = 0.0
    best_metrics = None
    history = []
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        val_loss, val_acc, val_metrics = validate(
            model, val_loader, criterion, device, epoch
        )
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1']
        })
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_metrics = val_metrics
            
            checkpoint_path = os.path.join(args.output_dir, f'best_{config_name}.pth')
            save_checkpoint(model, optimizer, epoch, best_acc, checkpoint_path)
    
    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")
    print(f"Best Metrics - Precision: {best_metrics['precision']:.4f}, Recall: {best_metrics['recall']:.4f}, F1: {best_metrics['f1']:.4f}")
    
    return {
        'config_name': config_name,
        'description': config['description'],
        'best_accuracy': best_acc,
        'best_precision': best_metrics['precision'],
        'best_recall': best_metrics['recall'],
        'best_f1': best_metrics['f1'],
        'history': history
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading datasets...")
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    results = []
    
    for config_name, config in ABLATION_CONFIGS.items():
        result = run_ablation_experiment(
            config_name, config, args,
            train_loader, val_loader, class_weights, device
        )
        results.append(result)
        
        history_path = os.path.join(args.output_dir, f'history_{config_name}.json')
        with open(history_path, 'w') as f:
            json.dump(result['history'], f, indent=4)
    
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    
    comparison_data = []
    for result in results:
        comparison_data.append({
            'Configuration': result['description'],
            'Accuracy (%)': f"{result['best_accuracy']:.2f}",
            'Precision': f"{result['best_precision']:.4f}",
            'Recall': f"{result['best_recall']:.4f}",
            'F1-Score': f"{result['best_f1']:.4f}"
        })
    
    df = pd.DataFrame(comparison_data)
    print("\n" + df.to_string(index=False))
    
    csv_path = os.path.join(args.output_dir, 'ablation_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    print("\n" + "="*80)
    print("Ablation study completed!")
    print(f"Results saved in: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Ablation Study')
    
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='../results/ablation',
                        help='Path to save ablation results')
    
    parser.add_argument('--backbone', type=str, default='efficientnet_b4',
                        choices=['efficientnet_b4', 'resnet50'],
                        help='CNN backbone')
    parser.add_argument('--num_classes', type=int, default=8,
                        help='Number of emotion classes')
    
    parser.add_argument('--lstm_hidden', type=int, default=256,
                        help='LSTM hidden size')
    parser.add_argument('--lstm_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs (reduced for ablation)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                        help='Use class weights for imbalanced data')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    main(args)
