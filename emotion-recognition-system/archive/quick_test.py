#!/usr/bin/env python3
"""
Quick test script to verify the system works
Uses a small model without pretrained weights to avoid download issues
"""

import os
import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models import get_model, WeightedCrossEntropyLoss
from utils import get_data_loaders, AverageMeter

print("="*60)
print("Quick System Verification Test")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✓ Device: {device}")

print("\n✓ Loading sample dataset...")
train_loader, val_loader, test_loader, class_weights = get_data_loaders(
    'data', batch_size=4, num_workers=0, img_size=224
)

print(f"  Training batches: {len(train_loader)}")
print(f"  Validation batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")

print("\n✓ Creating model (without pretrained weights for speed)...")
model = get_model(
    model_type='simple_cnn',  # Simpler model for quick test
    num_classes=8,
    backbone='efficientnet_b4',
    pretrained=False  # Skip pretrained weights
)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,}")

print("\n✓ Setting up training...")
criterion = WeightedCrossEntropyLoss(class_weights, device=device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

print("\n✓ Running quick training test (2 epochs)...")
for epoch in range(1, 3):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/2 [Train]')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = outputs.max(1)
        accuracy = (predicted == labels).float().mean().item() * 100
        
        losses.update(loss.item(), images.size(0))
        accuracies.update(accuracy, images.size(0))
        
        pbar.set_postfix({'loss': f'{losses.avg:.4f}', 'acc': f'{accuracies.avg:.2f}%'})
    
    # Validation
    model.eval()
    val_losses = AverageMeter()
    val_accuracies = AverageMeter()
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            accuracy = (predicted == labels).float().mean().item() * 100
            
            val_losses.update(loss.item(), images.size(0))
            val_accuracies.update(accuracy, images.size(0))
    
    print(f"\nEpoch {epoch}/2:")
    print(f"  Train Loss: {losses.avg:.4f}, Train Acc: {accuracies.avg:.2f}%")
    print(f"  Val Loss: {val_losses.avg:.4f}, Val Acc: {val_accuracies.avg:.2f}%")

print("\n" + "="*60)
print("✅ QUICK TEST COMPLETE - SYSTEM IS WORKING!")
print("="*60)
print("\nNext steps:")
print("1. Download real AffectNet+ dataset when you have time")
print("2. Run full training: python train.py --epochs 50")
print("3. Use pretrained weights for better performance")
print("\nNote: This test used synthetic data and no pretrained weights")
print("="*60)
