"""Quick script to verify what dataset is being loaded"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import get_data_loaders

print("=" * 80)
print("DATASET LOADING VERIFICATION")
print("=" * 80)

data_dir = "data"
print(f"\nData directory: {os.path.abspath(data_dir)}")
print(f"Exists: {os.path.exists(data_dir)}")

print("\nLoading data loaders...")
train_loader, val_loader, test_loader, class_weights = get_data_loaders(
    data_dir,
    batch_size=32,
    num_workers=0,  # Set to 0 for quick test
    img_size=224
)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Training samples: {len(train_loader.dataset)}")
print(f"Validation samples: {len(val_loader.dataset)}")
print(f"Test samples: {len(test_loader.dataset)}")
print(f"\nTraining batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

print("\nClass weights:")
if class_weights is not None:
    for i, weight in enumerate(class_weights):
        print(f"  Class {i}: {weight:.4f}")
else:
    print("  No class weights computed")

print("\n" + "=" * 80)
print("✓ Dataset loading verification complete!")
print("=" * 80)
