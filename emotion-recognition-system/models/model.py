"""
Model utilities and factory functions
"""

import torch
import torch.nn as nn
from models.cnn_dual_attention_bilstm import (
    EmotionRecognitionModel, 
    SimpleCNN, 
    CNNWithDualAttention
)


EMOTION_LABELS = [
    'Neutral', 'Happy', 'Sad', 'Surprise', 
    'Fear', 'Disgust', 'Anger', 'Contempt'
]


def get_model(model_type='full', num_classes=8, backbone='efficientnet_b4', 
              pretrained=True, **kwargs):
    """
    Factory function to create different model variants
    
    Args:
        model_type: 'full', 'simple_cnn', 'cnn_attention'
        num_classes: Number of emotion classes (default: 8)
        backbone: CNN backbone ('efficientnet_b4' or 'resnet50')
        pretrained: Use pretrained weights
        **kwargs: Additional model parameters
    
    Returns:
        PyTorch model
    """
    if model_type == 'full':
        return EmotionRecognitionModel(
            num_classes=num_classes, 
            backbone=backbone, 
            pretrained=pretrained,
            **kwargs
        )
    elif model_type == 'simple_cnn':
        return SimpleCNN(
            num_classes=num_classes, 
            backbone=backbone, 
            pretrained=pretrained
        )
    elif model_type == 'cnn_attention':
        return CNNWithDualAttention(
            num_classes=num_classes, 
            backbone=backbone, 
            pretrained=pretrained
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with class weights for handling imbalanced data"""
    def __init__(self, class_weights=None, device='cuda'):
        super(WeightedCrossEntropyLoss, self).__init__()
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        else:
            self.class_weights = None
    
    def forward(self, outputs, targets):
        return nn.functional.cross_entropy(outputs, targets, weight=self.class_weights)


def save_checkpoint(model, optimizer, epoch, best_acc, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device='cuda'):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Epoch: {epoch}, Best Accuracy: {best_acc:.4f}")
    
    return epoch, best_acc


def count_parameters(model):
    """Count trainable parameters in model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params


def get_model_summary(model, input_size=(1, 3, 224, 224)):
    """Print model architecture summary"""
    print("=" * 80)
    print(f"Model Architecture Summary")
    print("=" * 80)
    print(model)
    print("=" * 80)
    
    count_parameters(model)
    
    try:
        from torchsummary import summary
        summary(model, input_size[1:])
    except ImportError:
        print("\nInstall torchsummary for detailed layer-wise summary:")
        print("pip install torchsummary")
