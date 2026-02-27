"""Test optimal batch size for your GPU"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import get_model

print("="*80)
print("GPU MEMORY TEST - Finding Optimal Batch Size")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Free Memory: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

print("\nCreating model...")
model = get_model(
    model_type='full',
    num_classes=8,
    backbone='efficientnet_b4',
    lstm_hidden=256,
    lstm_layers=2,
    dropout=0.6
).to(device)

print(f"Model created successfully!")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test different batch sizes
test_sizes = [64, 80, 96, 112, 128]
optimal_batch = 64

print("\n" + "="*80)
print("Testing batch sizes with mixed precision (AMP)...")
print("="*80)

for batch_size in test_sizes:
    try:
        torch.cuda.empty_cache()
        
        # Simulate forward and backward pass
        dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
        
        with torch.cuda.amp.autocast():
            output = model(dummy_input)
            loss = output.sum()
        
        loss.backward()
        
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        
        print(f"✓ Batch size {batch_size:3d}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        optimal_batch = batch_size
        
        del dummy_input, output, loss
        torch.cuda.empty_cache()
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"✗ Batch size {batch_size:3d}: OUT OF MEMORY")
            break
        else:
            raise e

print("\n" + "="*80)
print(f"RECOMMENDED BATCH SIZE: {optimal_batch}")
print("="*80)
print(f"\nUse this in your training command:")
print(f"  --batch_size {optimal_batch}")
