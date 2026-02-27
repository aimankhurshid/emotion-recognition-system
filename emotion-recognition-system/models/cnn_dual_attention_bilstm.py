"""
Hybrid CNN + Dual Attention + BiLSTM Model for Emotion Recognition
"""

import torch
import torch.nn as nn
import torchvision.models as models
from utils.dual_attention import DualAttention


class BiLSTMDualAttention(nn.Module):
    """
    Hybrid architecture combining:
    - CNN backbone (EfficientNetB4 or ResNet50)
    - Dual Attention (Channel + Spatial)
    - BiLSTM for temporal modeling
    - Classification head
    """
    def __init__(self, num_classes=8, backbone='efficientnet_b4', pretrained=True, 
                 lstm_hidden=256, lstm_layers=2, dropout=0.5):
        super(BiLSTMDualAttention, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        if backbone == 'efficientnet_b4':
            efficientnet = models.efficientnet_b4(pretrained=pretrained)
            self.features = efficientnet.features
            feature_dim = 1792
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.dual_attention = DualAttention(in_channels=feature_dim)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # 4x4=16 timesteps (was 7x7=49)
        
        # Project high-dim features down before LSTM (1792 -> 512)
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5)
        )
        
        lstm_input_size = 512  # Projected dimension
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        lstm_output_size = lstm_hidden * 2
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        features = self.features(x)
        
        features = self.dual_attention(features)
        
        features = self.adaptive_pool(features)
        
        b, c, h, w = features.size()
        features_reshaped = features.view(b, c, h * w).permute(0, 2, 1)  # (B, 16, 1792)
        
        # Project to lower dimension for LSTM efficiency
        features_reshaped = self.feature_projection(features_reshaped)  # (B, 16, 512)
        
        lstm_out, (h_n, c_n) = self.lstm(features_reshaped)
        
        lstm_final = lstm_out[:, -1, :]
        
        output = self.classifier(lstm_final)
        
        return output
    
    def get_feature_maps(self, x):
        """Extract intermediate feature maps for visualization"""
        features = self.features(x)
        features_after_attention = self.dual_attention(features)
        return features, features_after_attention


class SimpleCNN(nn.Module):
    """Baseline CNN without attention or LSTM (for ablation study)"""
    def __init__(self, num_classes=8, backbone='efficientnet_b4', pretrained=True, dropout=0.5):
        super(SimpleCNN, self).__init__()
        
        if backbone == 'efficientnet_b4':
            efficientnet = models.efficientnet_b4(pretrained=pretrained)
            self.features = efficientnet.features
            feature_dim = 1792
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        features = self.avgpool(features)
        output = self.classifier(features)
        return output


class CNNWithDualAttention(nn.Module):
    """CNN + Dual Attention without BiLSTM (for ablation study)"""
    def __init__(self, num_classes=8, backbone='efficientnet_b4', pretrained=True, dropout=0.5):
        super(CNNWithDualAttention, self).__init__()
        
        if backbone == 'efficientnet_b4':
            efficientnet = models.efficientnet_b4(pretrained=pretrained)
            self.features = efficientnet.features
            feature_dim = 1792
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.dual_attention = DualAttention(in_channels=feature_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        features = self.dual_attention(features)
        features = self.avgpool(features)
        output = self.classifier(features)
        return output

# Alias for backward compatibility
EmotionRecognitionModel = BiLSTMDualAttention
