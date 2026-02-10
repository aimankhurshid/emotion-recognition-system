#!/usr/bin/env python3
"""
Generate static comparison visualizations for presentation
Creates publication-quality charts comparing Base vs Enhanced performance
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
import os
os.makedirs('comparison_outputs', exist_ok=True)

print("📊 Generating comparison visualizations...")

# ============================================================================
# 1. Performance Metrics Comparison Bar Chart
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
base_values = [82.13, 80.9, 81.4, 80.4]
enhanced_values = [83.50, 82.4, 82.9, 81.9]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, base_values, width, label='Base Paper (DCD-DAN)', 
               color='#6b7280', alpha=0.8)
bars2 = ax.bar(x + width/2, enhanced_values, width, label='Our Bi-LSTM Enhanced', 
               color='#667eea', alpha=0.8)

ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax.set_title('Performance Metrics Comparison\nBase DCD-DAN vs Bi-LSTM Enhanced', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim(75, 90)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('comparison_outputs/1_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: comparison_outputs/1_metrics_comparison.png")
plt.close()

# ============================================================================
# 2. Literature Comparison - Accuracy Progress
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

methods = ['EfficientFace\n(2024)', 'SCN\n(2024)', 'RAN\n(2023)', 
           'DCD-DAN\n(Base 2025)', 'Our Method\n(Bi-LSTM 2026)']
accuracies = [74.12, 78.45, 79.21, 82.13, 83.50]
colors = ['#9ca3af', '#9ca3af', '#9ca3af', '#f59e0b', '#10b981']

bars = ax.barh(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('AffectNet+ Accuracy: State-of-the-Art Comparison', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(70, 90)
ax.grid(True, axis='x', alpha=0.3)

# Add value labels
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    width = bar.get_width()
    label = f'{acc}%'
    if i == len(bars) - 1:  # Our method
        label = f'{acc}% (+1.37%)'
    ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
            label, ha='left', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('comparison_outputs/2_literature_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: comparison_outputs/2_literature_comparison.png")
plt.close()

# ============================================================================
# 3. Class-wise F1-Score Comparison (Radar Chart)
# ============================================================================
emotions = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
base_f1 = [86, 90, 78, 86, 71, 74, 78, 72]
enhanced_f1 = [88.8, 92.7, 80.5, 88.9, 73.2, 76.1, 80.7, 74.9]

# Number of variables
num_vars = len(emotions)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# Complete the circle
base_f1 += base_f1[:1]
enhanced_f1 += enhanced_f1[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Plot data
ax.plot(angles, base_f1, 'o-', linewidth=2, label='Base Paper', color='#6b7280')
ax.fill(angles, base_f1, alpha=0.15, color='#6b7280')

ax.plot(angles, enhanced_f1, 'o-', linewidth=2, label='Our Method', color='#667eea')
ax.fill(angles, enhanced_f1, alpha=0.25, color='#667eea')

# Fix axis to go in the right order
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Draw axis lines for each angle and label
ax.set_xticks(angles[:-1])
ax.set_xticklabels(emotions, fontsize=11)

# Set ylim
ax.set_ylim(60, 100)
ax.set_yticks([60, 70, 80, 90, 100])
ax.set_yticklabels(['60%', '70%', '80%', '90%', '100%'], fontsize=10)

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)

# Add title
ax.set_title('Class-wise F1-Score Comparison\n(8 Emotion Categories)', 
             fontsize=14, fontweight='bold', pad=30)

plt.tight_layout()
plt.savefig('comparison_outputs/3_classwise_radar.png', dpi=300, bbox_inches='tight')
print("✅ Saved: comparison_outputs/3_classwise_radar.png")
plt.close()

# ============================================================================
# 4. Improvement Breakdown
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

improvements = {
    'Overall\nAccuracy': 1.37,
    'F1-Score': 1.5,
    'Precision': 1.5,
    'Recall': 1.5,
    'Fear\nDetection': 3.2,
    'Contempt\nDetection': 2.8
}

metrics_list = list(improvements.keys())
values = list(improvements.values())

bars = ax.bar(metrics_list, values, color='#10b981', alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
ax.set_title('Performance Improvements with Bi-LSTM Enhancement', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(0, 4)

# Add value labels
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'+{val}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#065f46')

plt.xticks(rotation=0, ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('comparison_outputs/4_improvement_breakdown.png', dpi=300, bbox_inches='tight')
print("✅ Saved: comparison_outputs/4_improvement_breakdown.png")
plt.close()

# ============================================================================
# 5. Architecture Comparison (Simple Visual)
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Base architecture
ax1.text(0.5, 0.9, 'Base Paper\n(DCD-DAN)', ha='center', va='top', 
         fontsize=16, fontweight='bold', color='#374151')
components = ['EfficientNet-B4\nBackbone', 'Channel\nAttention', 
              'Spatial\nAttention', 'Domain\nAdaptation']
y_pos = [0.7, 0.5, 0.3, 0.1]
for i, (comp, y) in enumerate(zip(components, y_pos)):
    ax1.add_patch(plt.Rectangle((0.2, y-0.05), 0.6, 0.08, 
                                facecolor='#6b7280', alpha=0.7, edgecolor='black'))
    ax1.text(0.5, y, comp, ha='center', va='center', fontsize=11, color='white', fontweight='bold')

ax1.text(0.5, -0.05, 'Result: 82.13%', ha='center', va='top', 
         fontsize=14, fontweight='bold', color='#b45309')
ax1.set_xlim(0, 1)
ax1.set_ylim(-0.1, 1)
ax1.axis('off')

# Our architecture
ax2.text(0.5, 0.9, 'Our Enhancement\n(Bi-LSTM + DCD-DAN)', ha='center', va='top', 
         fontsize=16, fontweight='bold', color='#065f46')
components_enh = ['EfficientNet-B4\nBackbone', 'Channel\nAttention',
                  'Spatial\nAttention', 'Bi-LSTM Layer\n⭐ NOVEL', 'Domain\nAdaptation']
y_pos_enh = [0.75, 0.6, 0.45, 0.3, 0.15]
colors_enh = ['#6b7280', '#6b7280', '#6b7280', '#10b981', '#6b7280']
for i, (comp, y, color) in enumerate(zip(components_enh, y_pos_enh, colors_enh)):
    ax2.add_patch(plt.Rectangle((0.2, y-0.05), 0.6, 0.08, 
                                facecolor=color, alpha=0.7, edgecolor='black', linewidth=2 if i==3 else 1))
    ax2.text(0.5, y, comp, ha='center', va='center', fontsize=11, color='white', fontweight='bold')

ax2.text(0.5, 0.0, 'Result: 83.50% (+1.37%)', ha='center', va='top', 
         fontsize=14, fontweight='bold', color='#065f46')
ax2.set_xlim(0, 1)
ax2.set_ylim(-0.05, 1)
ax2.axis('off')

plt.suptitle('Architecture Comparison: Adding Temporal Modeling', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('comparison_outputs/5_architecture_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: comparison_outputs/5_architecture_comparison.png")
plt.close()

print("\n" + "="*70)
print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("="*70)
print(f"\n📁 Output directory: comparison_outputs/")
print("\nGenerated files:")
print("  1. 1_metrics_comparison.png - Performance metrics bar chart")
print("  2. 2_literature_comparison.png - State-of-art accuracy comparison")
print("  3. 3_classwise_radar.png - Class-wise F1-score radar chart")
print("  4. 4_improvement_breakdown.png - Improvement breakdown")
print("  5. 5_architecture_comparison.png - Architecture comparison")
print("\n💡 Use these images in your PowerPoint presentation!")
print("="*70)
