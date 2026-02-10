# Training Guide

This guide covers all aspects of training the emotion recognition model.

## Quick Start

```bash
cd training
python train.py --data_dir ../data --epochs 50 --batch_size 32
```

## Training Arguments

### Essential Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `../data` | Path to dataset directory |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `32` | Batch size for training |
| `--learning_rate` | `1e-4` | Initial learning rate |

### Model Architecture

| Argument | Default | Options | Description |
|----------|---------|---------|-------------|
| `--model_type` | `full` | `full`, `simple_cnn`, `cnn_attention` | Model architecture |
| `--backbone` | `efficientnet_b4` | `efficientnet_b4`, `resnet50` | CNN backbone |
| `--lstm_hidden` | `256` | - | LSTM hidden size |
| `--lstm_layers` | `2` | - | Number of LSTM layers |
| `--dropout` | `0.5` | - | Dropout rate |

### Training Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--weight_decay` | `1e-5` | Weight decay for regularization |
| `--use_class_weights` | `True` | Use class weights for imbalanced data |
| `--scheduler_patience` | `5` | LR scheduler patience |
| `--early_stop_patience` | `10` | Early stopping patience |
| `--save_interval` | `10` | Save checkpoint every N epochs |

## Training Examples

### 1. Full Training (Recommended)
```bash
python train.py \
    --data_dir ../data \
    --model_type full \
    --backbone efficientnet_b4 \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --use_class_weights
```

### 2. Quick Demo (5 epochs)
```bash
python train.py \
    --data_dir ../data \
    --epochs 5 \
    --batch_size 16
```

### 3. Training with ResNet50
```bash
python train.py \
    --data_dir ../data \
    --backbone resnet50 \
    --epochs 50 \
    --batch_size 24  # Smaller batch for larger model
```

### 4. CPU Training (if no GPU)
```bash
python train.py \
    --data_dir ../data \
    --epochs 30 \
    --batch_size 8 \
    --num_workers 2
```

## Monitoring Training

### TensorBoard

Launch TensorBoard to monitor training in real-time:

```bash
tensorboard --logdir ../results/logs --port 6006
```

Open browser: http://localhost:6006

**Available Metrics:**
- Training/Validation Loss
- Training/Validation Accuracy
- Precision, Recall, F1-Score
- Learning Rate

### Training Logs

Logs are saved in:
- **Console output**: Real-time training progress
- **TensorBoard logs**: `results/logs/`
- **History JSON**: `results/checkpoints/history_*.json`

## Checkpoints

Checkpoints are automatically saved in `results/checkpoints/`:

- `best_model_*.pth`: Best model based on validation accuracy
- `checkpoint_epoch_*.pth`: Periodic checkpoints (every 10 epochs)

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Current epoch
- Best accuracy

## Resume Training

To resume from a checkpoint:

```python
from models import load_checkpoint

model = get_model(...)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

start_epoch, best_acc = load_checkpoint(
    model, 
    optimizer, 
    'results/checkpoints/checkpoint_epoch_20.pth'
)

# Continue training from start_epoch
```

## Expected Training Time

**Hardware**: NVIDIA RTX 3090 (24GB)
- **50 epochs**: ~8-10 hours
- **Per epoch**: ~10-12 minutes

**Hardware**: NVIDIA GTX 1080 Ti (11GB)
- **50 epochs**: ~12-15 hours
- **Per epoch**: ~15-18 minutes

**CPU Only** (not recommended):
- **50 epochs**: ~3-4 days
- **Per epoch**: ~1.5-2 hours

## Troubleshooting

### Out of Memory Error

**Symptom:** `CUDA out of memory`

**Solutions:**
1. Reduce batch size: `--batch_size 16`
2. Reduce image size: `--img_size 128`
3. Use gradient accumulation (modify code)
4. Use smaller backbone: `--backbone resnet50`

### Training Not Converging

**Symptoms:** Loss not decreasing, accuracy stuck

**Solutions:**
1. Check dataset is loaded correctly
2. Verify class weights: `--use_class_weights`
3. Adjust learning rate: `--learning_rate 5e-5`
4. Increase batch size: `--batch_size 64`
5. Check data augmentation isn't too aggressive

### Validation Accuracy Fluctuating

**Symptoms:** Val accuracy goes up and down

**Solutions:**
1. This is normal, early stopping will save best model
2. Increase scheduler patience: `--scheduler_patience 7`
3. Use learning rate warmup (modify code)

### Training Too Slow

**Solutions:**
1. Increase `--num_workers 8` for faster data loading
2. Enable mixed precision training (modify code)
3. Use smaller model: `--model_type simple_cnn`
4. Reduce dataset size for testing

## Best Practices

1. **Always start with a demo run** (5 epochs) to verify everything works
2. **Monitor TensorBoard** to catch issues early
3. **Use class weights** for imbalanced datasets
4. **Save checkpoints frequently** in case of crashes
5. **Validate on test set** only after training is complete

## Next Steps

After training:
1. Evaluate on test set: `python evaluate.py`
2. Run ablation study: `python ablation_study.py`
3. Test inference: `python ../inference/predict_single.py`
4. Deploy webcam demo: `python ../inference/webcam_demo.py`
