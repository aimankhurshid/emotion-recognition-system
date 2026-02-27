"""
Visualize and compare experiment results
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import argparse

def load_history(experiment_path):
    """Load training history from experiment directory"""
    checkpoint_dir = Path(experiment_path) / "checkpoints"
    
    # Find history JSON file
    history_files = list(checkpoint_dir.glob("history_*.json"))
    if not history_files:
        print(f"No history file found in {checkpoint_dir}")
        return None
    
    with open(history_files[0], 'r') as f:
        return json.load(f)

def compare_experiments(experiments):
    """Compare multiple experiment results"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Experiments Comparison', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for idx, (exp_name, exp_path) in enumerate(experiments.items()):
        history = load_history(exp_path)
        if history is None:
            continue
        
        color = colors[idx % len(colors)]
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Plot 1: Training Loss
        axes[0, 0].plot(epochs, history['train_loss'], f'{color}-o', 
                        label=exp_name, linewidth=2, markersize=4, alpha=0.7)
        
        # Plot 2: Validation Loss
        axes[0, 1].plot(epochs, history['val_loss'], f'{color}-s', 
                        label=exp_name, linewidth=2, markersize=4, alpha=0.7)
        
        # Plot 3: Training Accuracy
        axes[1, 0].plot(epochs, history['train_acc'], f'{color}-o', 
                        label=exp_name, linewidth=2, markersize=4, alpha=0.7)
        
        # Plot 4: Validation Accuracy
        axes[1, 1].plot(epochs, history['val_acc'], f'{color}-s', 
                        label=exp_name, linewidth=2, markersize=4, alpha=0.7)
    
    # Configure subplots
    axes[0, 0].set_title('Training Loss', fontweight='bold', fontsize=12)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Validation Loss', fontweight='bold', fontsize=12)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Training Accuracy', fontweight='bold', fontsize=12)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Validation Accuracy', fontweight='bold', fontsize=12)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/experiments_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Comparison plot saved to: results/experiments_comparison.png")
    plt.show()

def visualize_single_experiment(exp_path):
    """Visualize a single experiment"""
    history = load_history(exp_path)
    if history is None:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    exp_name = Path(exp_path).name
    fig.suptitle(f'Training Results - {exp_name}', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss comparison
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=6)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Model Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy comparison
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
    ax2.plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Model Accuracy', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Overfitting gap
    ax3 = axes[1, 0]
    overfitting_gap = np.array(history['train_acc']) - np.array(history['val_acc'])
    ax3.plot(epochs, overfitting_gap, 'g-^', linewidth=2, markersize=6)
    ax3.axhline(y=15, color='orange', linestyle='--', alpha=0.5, label='Warning Threshold (15%)')
    ax3.axhline(y=25, color='red', linestyle='--', alpha=0.5, label='Critical Threshold (25%)')
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Gap (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Overfitting Gap (Train Acc - Val Acc)', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    best_val_acc = max(history['val_acc'])
    best_val_epoch = history['val_acc'].index(best_val_acc) + 1
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    final_gap = final_train_acc - final_val_acc
    
    stats_text = f"""
Experiment Summary
═══════════════════════════════════════

Name: {exp_name}
Total Epochs: {len(history['train_loss'])}

Final Performance:
  • Training Accuracy:    {final_train_acc:.2f}%
  • Validation Accuracy:  {final_val_acc:.2f}%
  • Overfitting Gap:      {final_gap:.2f}%

Best Validation:
  • Best Val Accuracy:    {best_val_acc:.2f}%
  • At Epoch:             {best_val_epoch}

Status:
  {'✓ Good Generalization' if final_gap < 15 else '⚠ Moderate Overfitting' if final_gap < 25 else '✗ Severe Overfitting'}
"""
    
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    save_path = Path(exp_path) / 'training_visualization.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize training experiment results')
    parser.add_argument('experiment', nargs='?', help='Experiment name or path')
    parser.add_argument('--compare', action='store_true', help='Compare multiple experiments')
    args = parser.parse_args()
    
    if args.compare:
        # Compare original vs new experiments
        experiments = {
            'Original Training': 'results/phase1_laptop_benchmark',
        }
        
        # Find all anti-overfitting experiments
        exp_dir = Path('results/experiments')
        if exp_dir.exists():
            for exp in exp_dir.iterdir():
                if exp.is_dir():
                    experiments[exp.name] = str(exp)
        
        if len(experiments) > 1:
            compare_experiments(experiments)
        else:
            print("Only one experiment found. Need at least 2 for comparison.")
    
    elif args.experiment:
        # Visualize specific experiment
        if Path(args.experiment).exists():
            exp_path = args.experiment
        else:
            exp_path = f"results/experiments/{args.experiment}"
        
        if Path(exp_path).exists():
            visualize_single_experiment(exp_path)
        else:
            print(f"Experiment not found: {exp_path}")
    
    else:
        # List available experiments
        print("Available experiments:")
        print("\n1. Original Training:")
        print("   results/phase1_laptop_benchmark")
        
        exp_dir = Path('results/experiments')
        if exp_dir.exists():
            exps = list(exp_dir.iterdir())
            if exps:
                print("\n2. Anti-Overfitting Experiments:")
                for idx, exp in enumerate(exps, 1):
                    if exp.is_dir():
                        print(f"   {idx}. {exp.name}")
        
        print("\nUsage:")
        print("  python visualize_experiment.py <experiment_name>  - Visualize single experiment")
        print("  python visualize_experiment.py --compare          - Compare all experiments")
