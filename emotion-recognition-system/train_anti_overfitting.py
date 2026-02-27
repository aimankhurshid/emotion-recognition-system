"""
Anti-Overfitting Training Script
Trains model with aggressive regularization to reduce overfitting gap
Saves logs in separate directory to preserve original training results
"""

import subprocess
import sys
import os
from datetime import datetime

# Create separate directories for anti-overfitting experiments
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
experiment_name = f"anti_overfitting_{timestamp}"

checkpoint_dir = f"results/experiments/{experiment_name}/checkpoints"
log_dir = f"results/experiments/{experiment_name}/logs"

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

print("="*80)
print("ANTI-OVERFITTING TRAINING EXPERIMENT")
print("="*80)
print(f"Experiment: {experiment_name}")
print(f"Checkpoints: {checkpoint_dir}")
print(f"Logs: {log_dir}")
print(f"\nYour original training results are preserved in:")
print(f"  - results/phase1_laptop_benchmark/")
print(f"  - results/optimized_run/")
print("="*80)
print()

# Training with anti-overfitting measures:
# 1. Higher dropout (0.6 instead of 0.5)
# 2. Stronger weight decay (5e-5 instead of 1e-5)
# 3. Data augmentation (already in data_loader)
# 4. Early stopping with patience 7
# 5. Smaller batch size for better generalization

training_args = [
    sys.executable, "-m", "training.train",
    
    # Data
    "--data_dir", "data",
    
    # Save directories (separate from original)
    "--checkpoint_dir", checkpoint_dir,
    "--log_dir", log_dir,
    
    # Model architecture
    "--model_type", "full",
    "--backbone", "efficientnet_b4",
    "--num_classes", "8",
    
    # Anti-overfitting hyperparameters
    "--dropout", "0.6",  # Higher dropout
    "--weight_decay", "5e-5",  # Stronger L2 regularization
    "--batch_size", "16",  # Smaller batch for better generalization
    "--learning_rate", "5e-5",  # Lower learning rate for smoother training
    
    # Training settings
    "--epochs", "50",
    "--early_stop_patience", "7",
    "--scheduler_patience", "3",
    "--save_interval", "10",
    
    # Architecture settings
    "--lstm_hidden", "256",
    "--lstm_layers", "2",
    
    # Performance
    "--num_workers", "4",
    "--use_amp",
    "--use_class_weights",
    
    # Image size
    "--img_size", "224",
    
    "--seed", "42"
]

print("Starting training with anti-overfitting configuration...")
print("\nKey differences from original training:")
print("  ✓ Dropout: 0.5 → 0.6 (more aggressive)")
print("  ✓ Weight Decay: 1e-5 → 5e-5 (5x stronger)")
print("  ✓ Batch Size: 32 → 16 (better generalization)")
print("  ✓ Learning Rate: 1e-4 → 5e-5 (more stable)")
print("  ✓ Early Stopping Patience: 10 → 7 (stop sooner)")
print()
print("="*80)
print()

# Run training
result = subprocess.run(training_args)

if result.returncode == 0:
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nResults saved in: results/experiments/{experiment_name}/")
    print(f"\nTo view TensorBoard logs:")
    print(f"  tensorboard --logdir {log_dir}")
    print(f"\nTo visualize results:")
    print(f"  python visualize_experiment.py {experiment_name}")
    print("="*80)
else:
    print("\n" + "="*80)
    print("TRAINING INTERRUPTED OR FAILED")
    print("="*80)
    print(f"Check logs in: {log_dir}")
    print("="*80)

sys.exit(result.returncode)
