#!/usr/bin/env python3
"""
Final Optimized Training Script for RTX 4060 (Laptop/Desktop)
Dataset: RAF-DB
Target: 94%+ accuracy
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
import logging
import sys
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import get_model, save_checkpoint
from utils import get_transforms, compute_metrics, AverageMeter

# =====================================================================
# RTX 4060 Optimized Configuration
# =====================================================================
CONFIG = {
    'batch_size': 64,            # Optimized for 8GB VRAM
    'num_workers': 8,            # Responsive for typical laptop/desktop CPUs
    'pin_memory': True,
    'persistent_workers': True,
    
    'epochs': 60,                # 60 epochs is usually enough for RAF-DB
    'early_stopping_patience': 10,
    'learning_rate': 2e-4,       # Tuned for BS 64
    'weight_decay': 5e-4,        # Slightly higher weight decay for better regularization
    'use_amp': True,             # Mixed precision (FP16)
    
    'model_name': 'full',
    'num_classes': 8,            # We keep 8 to be compatible with AffectNet model
    'pretrained': True,
    
    'data_dir': r'c:\Users\Admin\Documents\ayem\RAF-DB\DATASET',
    'checkpoint_dir': './results/rafdb_final',
    'log_dir': './results/rafdb_logs',
}

# RAF-DB Mapping: 
# RAF: 1:Surprise, 2:Fear, 3:Disgust, 4:Happy, 5:Sad, 6:Anger, 7:Neutral
# MODEL: 0:Neutral, 1:Happy, 2:Sad, 3:Surprise, 4:Fear, 5:Disgust, 6:Anger, 7:Contempt
RAF_TO_MODEL = {
    '7': 0, # Neutral
    '4': 1, # Happy
    '5': 2, # Sad
    '1': 3, # Surprise
    '2': 4, # Fear
    '3': 5, # Disgust
    '6': 6, # Anger
}

class RAFDBDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.data = []
        self.labels = []
        
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")

        for raf_id, model_id in RAF_TO_MODEL.items():
            class_dir = os.path.join(self.root_dir, raf_id)
            if not os.path.exists(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append(os.path.join(class_dir, img_name))
                    self.labels.append(model_id)
        
        print(f"Loaded {len(self.data)} images for {split} split")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def setup_logging(log_dir, run_name):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{run_name}.log')
    logger = logging.getLogger('rafdb_training')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler, logger):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch:3d} [TRAIN]', ncols=100)
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        _, preds = outputs.max(1)
        acc = (preds == labels).float().mean()
        losses.update(loss.item(), images.size(0))
        accuracies.update(acc.item(), images.size(0))
        
        pbar.set_postfix({'Loss': f'{losses.avg:.4f}', 'Acc': f'{accuracies.avg:.4f}'})
    return losses.avg, accuracies.avg

def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    all_preds = []
    all_labels = []
    pbar = tqdm(val_loader, desc=f'Epoch {epoch:3d} [VAL]', ncols=100)
    
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
    
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    return losses.avg, accuracies.avg, metrics

def main():
    run_name = f"rafdb_4060_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logging(CONFIG['log_dir'], run_name)
    logger.info("Starting RAF-DB Training on RTX 4060")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    train_transform = get_transforms('train', 224)
    val_transform = get_transforms('val', 224)
    
    train_dataset = RAFDBDataset(CONFIG['data_dir'], split='train', transform=train_transform)
    val_dataset = RAFDBDataset(CONFIG['data_dir'], split='test', transform=val_transform)
    
    loader_args = {'batch_size': CONFIG['batch_size'], 'num_workers': CONFIG['num_workers'], 'pin_memory': True, 'persistent_workers': True}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    
    # Calculate class weights for imbalance
    labels = np.array(train_dataset.labels)
    class_counts = np.bincount(labels, minlength=8)
    # Avoid division by zero for class 7 (Contempt) which is empty in RAF-DB
    class_counts[7] = 1 
    total = sum(class_counts)
    weights = torch.FloatTensor([total / (7 * count if count > 0 else 1) for count in class_counts]).to(device)
    # Set weight for 7 to 0 so it doesn't affect training if somehow predicted
    weights[7] = 0.0
    
    model = get_model(CONFIG['model_name'], num_classes=8, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.amp.GradScaler('cuda')
    writer = SummaryWriter(log_dir=os.path.join(CONFIG['log_dir'], run_name))
    
    best_val_acc = 0.0
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler, logger)
        v_loss, v_acc, metrics = validate(model, val_loader, criterion, device, epoch)
        scheduler.step()
        
        logger.info(f"Epoch {epoch} | Val Acc: {v_acc:.4f} | F1: {metrics['f1']:.4f}")
        writer.add_scalars('Acc', {'train': t_acc, 'val': v_acc}, epoch)
        
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            save_checkpoint({'model_state_dict': model.state_dict(), 'acc': v_acc}, os.path.join(CONFIG['checkpoint_dir'], 'best_rafdb_model.pth'))
            logger.info(f"✓ Saved Best Model: {v_acc:.4f}")
            
    logger.info(f"Training Complete. Best Val Acc: {best_val_acc:.4f}")
    writer.close()

if __name__ == '__main__':
    main()
