# Plan to Beat AffectNet+ Baseline (82.13%)

This plan outlines the optimized training strategy to ensure the Bi-LSTM enhanced model outperforms the DCD-DAN base paper on the AffectNet+ dataset.

## Proposed Strategy

To maximize performance, we will use the most powerful configuration available in our codebase:

### 1. Architecture Optimization
- **Backbone**: **EfficientNet-B4** (Default). It captures finer details than ResNet50 with fewer parameters.
- **Bi-LSTM Hidden Size**: Increase from 256 to **512**. This allows the temporal modeling layer to capture more complex relationships between the 49 spatial tokens (from the 7x7 attention map).
- **Dual Attention**: Ensure both Channel and Spatial attention are active to weigh "what" and "where" features are important.

### 2. Training Hyperparameters
- **Optimizer**: **AdamW** with a learning rate of **1e-4**. AdamW is superior for transformer-like and attention architectures.
- **Scheduler**: **ReduceLROnPlateau** (Patience 5). This will drop the learning rate when validation accuracy plateaus, helping the model find the local minima.
- **Loss Function**: **Weighted Cross Entropy**. This is critical as AffectNet+ often has fewer samples for emotions like "Contempt" and "Disgust".
- **Batch Size**: **32** (or **16** for laptop).

### 3. Storage & Savepoints (Fault-Tolerance)
- **Model Checkpoints**: Each `.pth` file is ~250MB. 
- **Resume System**: Use `--resume results/checkpoints/latest_checkpoint.pth` to restart training if the laptop shuts down.
- **Safety Precautions**: 
    - **Gradient Clipping**: Prevents exploding gradients during long laptop runs.
    - **Latest Checkpoint**: Automatically saves `latest.pth` every epoch to minimize data loss.
    - **NaN Detection**: Automatically stops and saves if numerical errors occur.
    - **Graceful Shutdown**: Pressing `Ctrl+C` will now save a special `interrupted.pth` before exiting.

## Execution Steps

### [Component] Training Execution

#### [MODIFY] [FINAL_TRAINING_ROADMAP.md](file:///Users/ayeman/Downloads/minorproject_facial/emotion_recognition_system/FINAL_TRAINING_ROADMAP.md)
Update the roadmap to include the specific "Winning Command" optimized for AffectNet+.

## Verification Plan

### Automated Verification
After the training run completes, we will verify the results using:
1. **Validation Accuracy**: Must exceed **82.13%**.
2. **Metrics Report**: Run `python training/evaluate.py` to generate the F1-score and Precision/Recall for all 8 classes.
3. **Confusion Matrix**: Visually inspect the matrix to ensure "Disgust" and "Contempt" (the hardest classes) are being classified correctly.

### Manual Verification
1. **Webcam Test**: Run `python webcam_demo_ultra.py` using the new `best_model.pth`.
2. **Visual Check**: Ensure the circuit logic correctly identifies features for the hardest emotions (e.g., Brows Down + Jaw Tight = Anger).
