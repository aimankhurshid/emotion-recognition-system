#!/usr/bin/env python3
"""
Optimized Training Script for RTX 5000 Ada (32GB VRAM)
Dataset: RAF-DB only
Target: 94.20% accuracy in ~1-2 hours
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
import logging
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import get_model, save_checkpoint
from utils import get_data_loaders, compute_metrics, AverageMeter

# =====================================================================
# RTX 5000 Ada Optimized Configuration
# =====================================================================
CONFIG = {
    # Hardware optimization for RTX 5000 Ada
    'batch_size': 256,           # 4x larger than RTX 4060 (32GB VRAM)
    'num_workers': 16,           # More CPU threads for data loading
    'pin_memory': True,          # Faster GPU transfer
    'persistent_workers': True,  # Keep workers alive between epochs
    
    # Training settings
    'epochs': 100,
    'early_stopping_patience': 15,
    'learning_rate': 4e-4,       # Linear scaling: 1e-4 * (256/64) = 4e-4
    'weight_decay': 1e-4,
    'use_amp': True,             # Mixed precision (FP16)
    
    # Model settings
    'model_name': 'cnn_dual_attention_bilstm',
    'num_classes': 8,            # RAF-DB has 8 classes
    'pretrained': True,
    
    # Dataset
    'data_dir': './data',
    'dataset': 'rafdb',
    'img_size': 224,
    
    # Output
    'checkpoint_dir': './results/rtx5000_checkpoint',
    'log_dir': './results/rtx5000_logs',
}


def setup_logging(log_dir, run_name):
    """Setup logging"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{run_name}.log')
    
    logger = logging.getLogger('rtx5000_training')
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


class WeightedCrossEntropyLoss(nn.Module):
    """Class-weighted cross entropy for RAF-DB imbalance"""
    def __init__(self, num_samples_per_class):
        super().__init__()
        total_samples = sum(num_samples_per_class)
        num_classes = len(num_samples_per_class)
        
        # w_i = N_total / (N_classes × N_i)
        weights = [total_samples / (num_classes * n) for n in num_samples_per_class]
        self.weights = torch.FloatTensor(weights)
        
    def forward(self, outputs, labels):
        self.weights = self.weights.to(outputs.device)
        return nn.functional.cross_entropy(outputs, labels, weight=self.weights)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler, logger):
    """Train one epoch"""
    model.train()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch:3d} [TRAIN]', ncols=100)
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision training
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Check for NaN
        if torch.isnan(loss):
            logger.error(f"NaN loss detected at epoch {epoch}")
            continue
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics
        _, preds = outputs.max(1)
        acc = (preds == labels).float().mean()
        
        losses.update(loss.item(), images.size(0))
        accuracies.update(acc.item(), images.size(0))
        
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc': f'{accuracies.avg:.4f}'
        })
    
    return losses.avg, accuracies.avg


def validate(model, val_loader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    all_preds = []
    all_labels = []
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch:3d} [VAL]  ', ncols=100)
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            _, preds = outputs.max(1)
            acc = (preds == labels).float().mean()
            
            losses.update(loss.item(), images.size(0))
            accuracies.update(acc.item(), images.size(0))
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{accuracies.avg:.4f}'
            })
    
    # Compute detailed metrics
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    
    return losses.avg, accuracies.avg, metrics


def main():
    """Main training function"""
    
    # Setup
    run_name = f"rafdb_rtx5000_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logging(CONFIG['log_dir'], run_name)
    
    logger.info("=" * 80)
    logger.info("RTX 5000 Ada Optimized Training - RAF-DB Dataset")
    logger.info("=" * 80)
    logger.info(f"Configuration: {json.dumps(CONFIG, indent=2)}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # RTX 5000 Ada optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Data loaders (RAF-DB)
    logger.info("Loading RAF-DB dataset...")
    train_loader, val_loader, class_counts = get_data_loaders(
        data_dir=CONFIG['data_dir'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        persistent_workers=CONFIG['persistent_workers']
    )
    
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    logger.info(f"Class distribution: {class_counts}")
    
    # Model
    logger.info("Initializing Bi-LSTM Enhanced Dual Attention Model...")
    model = get_model(
        model_name=CONFIG['model_name'],
        num_classes=CONFIG['num_classes'],
        pretrained=CONFIG['pretrained']
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function (class-weighted)
    criterion = WeightedCrossEntropyLoss(class_counts)
    logger.info(f"Using class-weighted cross-entropy with weights: {criterion.weights.tolist()}")
    
    # Optimizer (AdamW with higher LR for larger batch)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Scheduler (Cosine annealing with warm restarts)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=2,
        eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda')
    
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(CONFIG['log_dir'], run_name))
    
    # Training loop
    logger.info("=" * 80)
    logger.info("Starting Training...")
    logger.info("=" * 80)
    
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rates': []
    }
    
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        epoch_start = datetime.now()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler, logger
        )
        
        # Validate
        val_loss, val_acc, val_metrics = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        logger.info(
            f"Epoch {epoch:3d}/{CONFIG['epochs']} | "
            f"Time: {epoch_time:.1f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} "
            f"Recall: {val_metrics['recall']:.4f} "
            f"F1: {val_metrics['f1']:.4f}"
        )
        
        # TensorBoard
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # History
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint_path = os.path.join(
                CONFIG['checkpoint_dir'],
                f'best_model_rafdb_{run_name}.pth'
            )
            
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_acc': best_val_acc,
                'val_metrics': val_metrics,
                'config': CONFIG
            }, checkpoint_path)
            
            logger.info(f"✓ New best model saved! Val Acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{CONFIG['early_stopping_patience']}")
        
        # Early stopping
        if patience_counter >= CONFIG['early_stopping_patience']:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Save final history
    history_path = os.path.join(CONFIG['checkpoint_dir'], f'history_{run_name}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("=" * 80)
    logger.info(f"Training Complete!")
    logger.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
    logger.info(f"Checkpoint saved to: {CONFIG['checkpoint_dir']}")
    logger.info("=" * 80)
    
    writer.close()


if __name__ == '__main__':
    main()
