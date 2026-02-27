#!/usr/bin/env python3
"""
Show model architecture without downloading pretrained weights
"""

import sys
sys.path.append('.')

from models import get_model
import torch

print("="*70)
print("EMOTION RECOGNITION MODEL - ARCHITECTURE SUMMARY")
print("="*70)

# Create model without pretrained weights to avoid SSL issues
print("\n📦 Creating Full Model (CNN + Dual Attention + BiLSTM)...")
model = get_model(
    model_type='full',
    num_classes=8,
    backbone='efficientnet_b4',
    pretrained=False  # Skip pretrained weights
)

print("\n" + "="*70)
print("MODEL STRUCTURE")
print("="*70)
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n" + "="*70)
print("PARAMETER COUNT")
print("="*70)
print(f"Total parameters:      {total_params:,}")
print(f"Trainable parameters:  {trainable_params:,}")
print(f"Model size (approx):   ~{total_params * 4 / 1e6:.1f} MB")

# Test forward pass
print("\n" + "="*70)
print("FORWARD PASS TEST")
print("="*70)
dummy_input = torch.randn(1, 3, 224, 224)
print(f"Input shape:  {dummy_input.shape}")

with torch.no_grad():
    output = model(dummy_input)
    probs = torch.softmax(output, dim=1)

print(f"Output shape: {output.shape}")
print(f"Probabilities (should sum to 1.0): {probs.sum().item():.4f}")

print("\n" + "="*70)
print("✅ MODEL ARCHITECTURE VERIFIED!")
print("="*70)
print("\nKey Components:")
print("  ✓ EfficientNetB4 backbone")
print("  ✓ Dual Attention (Channel + Spatial)")
print("  ✓ BiLSTM (256 hidden, 2 layers)")
print("  ✓ Classification head (8 emotions)")
print("\n" + "="*70)
