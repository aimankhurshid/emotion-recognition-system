"""
Training script for emotion recognition model
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    """Setup comprehensive logging to both file and console"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{run_name}_training.log')
    
    # Create logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler=None, use_amp=True):
    """Train for one epoch with mixed precision support"""
    model.train()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        # Mixed precision training
        if use_amp and scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # 2. NaN Check (Safety Precaution)
        if torch.isnan(loss):
            print(f"\n[CRITICAL WARNING] Loss is NaN at batch {batch_idx}. Stopping training.")
            return None, None
        
        _, predicted = outputs.max(1)
        accuracy = (predicted == labels).sum().item() / labels.size(0) * 100  # type: ignore
        
        losses.update(loss.item(), images.size(0))
        accuracies.update(accuracy, images.size(0))
        
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accuracies.avg:.2f}%'
        })
    
    return losses.avg, accuracies.avg


def validate(model, val_loader, criterion, device, epoch, use_amp=True):
    """Validate the model with mixed precision support"""
    model.eval()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    all_preds = []
    all_labels = []
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Mixed precision inference
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            accuracy = (predicted == labels).sum().item() / labels.size(0) * 100  # type: ignore
            
            losses.update(loss.item(), images.size(0))
            accuracies.update(accuracy, images.size(0))
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accuracies.avg:.2f}%'
            })
    
    metrics = compute_metrics(all_labels, all_preds, average='macro')
    
    return losses.avg, accuracies.avg, metrics, all_labels, all_preds


def main(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist!")
        print("Please download AffectNet+ dataset and organize it in the data/ directory")
        return
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    print("Loading datasets...")
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        persistent_workers=(args.num_workers > 0)
    )
    
    if len(train_loader) == 0:
        print("Error: No training data found!")
        return
    
    print(f"\nDataset sizes:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    print("\nCreating model...")
    model = get_model(
        model_type=args.model_type,
        num_classes=args.num_classes,
        backbone=args.backbone,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout
    ).to(device)
    
    # Enable optimizations for faster training
    if args.compile_model and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile for faster training...")
        model = torch.compile(model)
    
    # Mixed precision training scaler
    scaler = torch.amp.GradScaler('cuda') if args.use_amp and device.type == 'cuda' else None
    use_amp = args.use_amp and device.type == 'cuda'
    if use_amp:
        print("Using automatic mixed precision (AMP) for faster training")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    start_epoch = 1
    best_acc = 0.0
    
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"Loaded checkpoint at epoch {checkpoint['epoch']} with acc {best_acc:.2f}%")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if args.use_class_weights and class_weights is not None:
        criterion = WeightedCrossEntropyLoss(class_weights, device=device)
        print("Using class-weighted loss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard cross-entropy loss")
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=args.scheduler_patience,
        verbose=True
    )
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{args.model_type}_{args.backbone}_{timestamp}"
    
    # Setup logging
    logger = setup_logging(args.log_dir, run_name)
    logger.info(f"Starting training run: {run_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {args.model_type} with backbone {args.backbone}")
    logger.info(f"Batch size: {args.batch_size}, Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}, Weight decay: {args.weight_decay}")
    
    writer = SummaryWriter(os.path.join(args.log_dir, run_name))
    logger.info(f"TensorBoard logs: {os.path.join(args.log_dir, run_name)}")
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    patience_counter = 0
    val_acc = 0.0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*80)
    
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch,
                scaler=scaler, use_amp=use_amp
            )
            
            if train_loss is None: # NaN triggered
                break
            
            val_loss, val_acc, val_metrics, all_labels, all_preds = validate(
                model, val_loader, criterion, device, epoch, use_amp=use_amp
            )
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Write to TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('Metrics/precision', val_metrics['precision'], epoch)
            writer.add_scalar('Metrics/recall', val_metrics['recall'], epoch)
            writer.add_scalar('Metrics/f1', val_metrics['f1'], epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Add histograms of model parameters
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(f'Parameters/{name}', param, epoch)
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
            
            # Flush writer to ensure data is saved
            writer.flush()
            
            # Save history to JSON after every epoch
            history_path = os.path.join(args.checkpoint_dir, f'history_{run_name}.json')
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)
            
            scheduler.step(val_acc)
            
            # Log epoch results
            logger.info(f"\nEpoch {epoch}/{args.epochs}:")
            logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            logger.info(f"  Val Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
            logger.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                
                checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model_{run_name}.pth')
                save_checkpoint(model, optimizer, epoch, best_acc, checkpoint_path)
                logger.info(f"  ✓ New best model saved! Accuracy: {best_acc:.2f}%")
                
                # Print detailed per-class report for the best model
                from sklearn.metrics import classification_report
                from models import EMOTION_LABELS
                logger.info("\n  Detailed Classification Report (Best Model):")
                report = classification_report(all_labels, all_preds, target_names=EMOTION_LABELS, digits=3)
                logger.info(f"\n{report}")
                
            else:
                patience_counter += 1
                logger.info(f"  No improvement ({patience_counter}/{args.early_stop_patience})")
            
            if epoch % args.save_interval == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}_{run_name}.pth')
                save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path)
                logger.info(f"  Checkpoint saved at epoch {epoch}")
            
            if patience_counter >= args.early_stop_patience:
                logger.info(f"\nEarly stopping triggered after {epoch} epochs")
                break
                
            # Always save a 'latest' checkpoint for maximum safety
            latest_path = os.path.join(args.checkpoint_dir, f'latest_checkpoint.pth')
            save_checkpoint(model, optimizer, epoch, val_acc, latest_path)
            
            logger.info("="*80)
            
    except KeyboardInterrupt:
        logger.warning("\n[USER INTERRUPT] Saving security checkpoint before exiting...")
        security_path = os.path.join(args.checkpoint_dir, f'interrupted_epoch_{epoch}_{run_name}.pth')
        save_checkpoint(model, optimizer, epoch, val_acc, security_path)
        logger.info(f"Checkpoint saved to: {security_path}")
        writer.flush()
        writer.close()
        sys.exit(0)

    
    # Save final training history
    history_path = os.path.join(args.checkpoint_dir, f'history_{run_name}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    logger.info(f"Training history saved to: {history_path}")
    
    # Generate and save training plots
    plot_path = os.path.join(args.checkpoint_dir, f'training_history_{run_name}.png')
    plot_training_history(history, save_path=plot_path)
    logger.info(f"Training plot saved to: {plot_path}")
    
    # Create comprehensive training summary
    summary_path = os.path.join(args.checkpoint_dir, f'training_summary_{run_name}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Training Summary - {run_name}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Model: {args.model_type} with {args.backbone}\n")
        f.write(f"  Device: {device}\n")
        f.write(f"  Batch Size: {args.batch_size}\n")
        f.write(f"  Total Epochs: {epoch}\n")
        f.write(f"  Learning Rate: {args.learning_rate}\n")
        f.write(f"  Weight Decay: {args.weight_decay}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Best Validation Accuracy: {best_acc:.2f}%\n")
        f.write(f"  Final Training Accuracy: {history['train_acc'][-1]:.2f}%\n")
        f.write(f"  Final Training Loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final Validation Loss: {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Overfitting Gap: {(history['train_acc'][-1] - history['val_acc'][-1]):.2f}%\n\n")
        f.write(f"Files Generated:\n")
        f.write(f"  - Best model: best_model_{run_name}.pth\n")
        f.write(f"  - History: history_{run_name}.json\n")
        f.write(f"  - Plots: training_history_{run_name}.png\n")
        f.write(f"  - TensorBoard logs: {os.path.join(args.log_dir, run_name)}\n")
    logger.info(f"Training summary saved to: {summary_path}")
    
    # Flush and close writer
    writer.flush()
    writer.close()
    logger.info("TensorBoard writer closed successfully")
    
    logger.info("\n" + "="*80)
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_acc:.2f}%")
    logger.info(f"Model checkpoints saved in: {args.checkpoint_dir}")
    logger.info(f"TensorBoard logs saved in: {args.log_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Emotion Recognition Model')
    
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Path to dataset directory')
    parser.add_argument('--checkpoint_dir', type=str, default='../results/checkpoints',
                        help='Path to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='../results/logs',
                        help='Path to save logs')
    
    parser.add_argument('--model_type', type=str, default='full',
                        choices=['full', 'simple_cnn', 'cnn_attention'],
                        help='Model architecture type')
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
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                        help='Use class weights for imbalanced data')
    parser.add_argument('--scheduler_patience', type=int, default=5,
                        help='Learning rate scheduler patience')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    # Performance optimization flags
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision (fp16) for faster training')
    parser.add_argument('--compile_model', action='store_true', default=False,
                        help='Use torch.compile for faster training (PyTorch 2.0+)')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    main(args)
