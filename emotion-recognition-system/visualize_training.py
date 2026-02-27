import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the training history
history_file = Path('results/phase1_laptop_benchmark/history_full_efficientnet_b4_20260221_004807.json')
with open(history_file, 'r') as f:
    history = json.load(f)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Training History - EfficientNet-B4 Emotion Recognition', fontsize=16, fontweight='bold')

epochs = range(1, len(history['train_loss']) + 1)

# Plot 1: Training and Validation Loss
ax1 = axes[0, 0]
ax1.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=6)
ax1.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=6)
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.set_title('Model Loss Over Epochs', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Training and Validation Accuracy
ax2 = axes[0, 1]
ax2.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
ax2.plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Model Accuracy Over Epochs', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Loss Difference (Overfitting indicator)
ax3 = axes[1, 0]
loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
ax3.plot(epochs, loss_diff, 'g-^', linewidth=2, markersize=6)
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax3.set_ylabel('Val Loss - Train Loss', fontsize=12, fontweight='bold')
ax3.set_title('Overfitting Indicator (Higher = More Overfitting)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.fill_between(epochs, 0, loss_diff, where=(loss_diff > 0), alpha=0.3, color='red', label='Overfitting')
ax3.fill_between(epochs, 0, loss_diff, where=(loss_diff <= 0), alpha=0.3, color='green', label='Good Fit')
ax3.legend(fontsize=10)

# Plot 4: Summary Statistics Table
ax4 = axes[1, 1]
ax4.axis('off')

# Calculate statistics
best_val_acc = max(history['val_acc'])
best_val_epoch = history['val_acc'].index(best_val_acc) + 1
final_train_acc = history['train_acc'][-1]
final_val_acc = history['val_acc'][-1]
final_train_loss = history['train_loss'][-1]
final_val_loss = history['val_loss'][-1]

stats_text = f"""
Training Summary Statistics
═══════════════════════════════════════

Total Epochs Trained:     {len(history['train_loss'])}

Final Performance:
  • Training Accuracy:    {final_train_acc:.2f}%
  • Validation Accuracy:  {final_val_acc:.2f}%
  • Training Loss:        {final_train_loss:.4f}
  • Validation Loss:      {final_val_loss:.4f}

Best Validation Performance:
  • Best Val Accuracy:    {best_val_acc:.2f}%
  • Achieved at Epoch:    {best_val_epoch}

Model Characteristics:
  • Architecture:         EfficientNet-B4
  • Overfitting Gap:      {final_val_loss - final_train_loss:.4f}
  • Accuracy Gap:         {final_train_acc - final_val_acc:.2f}%
"""

ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, 
         fontsize=11, verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('results/training_analysis_complete.png', dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to: results/training_analysis_complete.png")

# Also create a simple metrics summary
print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
print(f"Epochs: {len(history['train_loss'])}")
print(f"Best Val Accuracy: {best_val_acc:.2f}% (Epoch {best_val_epoch})")
print(f"Final Train Accuracy: {final_train_acc:.2f}%")
print(f"Final Val Accuracy: {final_val_acc:.2f}%")
print(f"Overfitting Gap: {(final_train_acc - final_val_acc):.2f}%")
print("="*60)

plt.show()
