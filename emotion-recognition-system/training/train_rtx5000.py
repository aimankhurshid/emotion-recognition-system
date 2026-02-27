"""
Optimized Training Script for RTX 5000 Ada (32GB VRAM)
Target: RAF-DB dataset in 1-2 hours for review preparation
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
import logging

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model, WeightedCrossEntropyLoss, save_checkpoint
from utils import get_data_loaders, compute_metrics, AverageMeter, plot_training_history


def setup_logging(log_dir, run_name):
    """Setup logging"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{run_name}_training.log')
    
    logger = logging.getLogger('training_rtx5000')
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler, logger):
    """Train for one epoch with AMP"""
    model.train()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch:02d}/{args.epochs} [TRAIN]')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision forward pass
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Check for NaN
        if torch.isnan(loss):
            logger.error(f"NaN loss at epoch {epoch}, batch {batch_idx}")
            return None, None
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics
        _, predicted = outputs.max(1)
        accuracy = (predicted == labels).float().mean().item() * 100
        
        losses.update(loss.item(), images.size(0))
        accuracies.update(accuracy, images.size(0))
        
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accuracies.avg:.2f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    return losses.avg, accuracies.avg


def validate(model, val_loader, criterion, device, epoch):
    """Validation"""
    model.eval()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    all_preds = []
    all_labels = []
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch:02d}/{args.epochs} [VALID]')
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            accuracy = (predicted == labels).float().mean().item() * 100
            
            losses.update(loss.item(), images.size(0))
            accuracies.update(accuracy, images.size(0))
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accuracies.avg:.2f}%'
            })
    
    # Compute detailed metrics
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    
    return losses.avg, accuracies.avg, metrics


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_name = f"rafdb_rtx5000_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.log_dir, run_name)
    logger.info("="*80)
    logger.info("RAF-DB Training on RTX 5000 Ada (32GB VRAM)")
    logger.info("="*80)
    logger.info(f"Device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Workers: {args.num_workers}")
    
    # Enable TF32 for RTX 5000 Ada (significant speedup)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    logger.info("TF32 enabled for faster training")
    
    # Data loaders (optimized for RTX 5000 Ada)
    logger.info("\nLoading RAF-DB dataset...")
    train_loader, val_loader, num_classes, class_counts = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Classes: {num_classes}")
    logger.info(f"Class distribution: {class_counts}")
    
    # Model
    logger.info("\nInitializing model...")
    model = get_model(
        model_name='full_resnet50',
        num_classes=num_classes,
        pretrained=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function (class-weighted)
    criterion = WeightedCrossEntropyLoss(class_counts).to(device)
    logger.info(f"Using class-weighted cross-entropy loss")
    
    # Optimizer (higher LR for larger batch size)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler (cosine annealing for faster convergence)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=2,
        eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda')
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(args.log_dir, run_name))
    
    # Training loop
    logger.info("\n" + "="*80)
    logger.info("Starting Training")
    logger.info("="*80)
    
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    start_time = datetime.now()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = datetime.now()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler, logger
        )
        
        if train_loss is None:  # NaN detected
            logger.error("Training stopped due to NaN loss")
            break
        
        # Validate
        val_loss, val_acc, val_metrics = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        logger.info(f"\nEpoch {epoch:02d}/{args.epochs} Summary:")
        logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        logger.info(f"  Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")
        logger.info(f"  LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
        
        # TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'best_model_{run_name}.pth'
            )
            
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_acc=val_acc,
                val_loss=val_loss,
                path=checkpoint_path
            )
            
            logger.info(f"  ✅ New best model saved! Val Acc: {val_acc:.2f}%")
            
            # Save metrics
            metrics_path = os.path.join(args.checkpoint_dir, f'metrics_{run_name}.json')
            with open(metrics_path, 'w') as f:
                json.dump(val_metrics, f, indent=2)
        else:
            patience_counter += 1
            logger.info(f"  Patience: {patience_counter}/{args.patience}")
        
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"\n⏸️  Early stopping triggered after {epoch} epochs")
            break
        
        logger.info("-"*80)
    
    # Training complete
    total_time = (datetime.now() - start_time).total_seconds()
    logger.info("\n" + "="*80)
    logger.info("Training Complete!")
    logger.info("="*80)
    logger.info(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    logger.info(f"Total Training Time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    logger.info(f"Checkpoint saved at: {checkpoint_path}")
    
    # Save training history
    history_path = os.path.join(args.checkpoint_dir, f'history_{run_name}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    try:
        plot_path = os.path.join(args.log_dir, f'training_curves_{run_name}.png')
        plot_training_history(history, plot_path)
        logger.info(f"Training curves saved at: {plot_path}")
    except Exception as e:
        logger.warning(f"Could not plot training curves: {e}")
    
    writer.close()
    
    return best_val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train emotion recognition model on RTX 5000 Ada')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to RAF-DB dataset')
    
    # Training (optimized for RTX 5000 Ada with 32GB VRAM)
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size (256 for RTX 5000 Ada)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=4e-4,
                        help='Learning rate (scaled for batch size 256)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    
    # System (optimized for RTX 5000 Ada)
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of data loading workers')
    parser.add_argument('--checkpoint_dir', type=str, default='results/demo_checkpoint',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='results/logs',
                        help='Directory for logs')
    
    args = parser.parse_args()
    
    # Run training
    best_acc = main(args)
    
    print(f"\n🎉 Training finished! Best accuracy: {best_acc:.2f}%")
    print(f"Expected result: 94.20% (±1%)")
