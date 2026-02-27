"""
Dual Attention Mechanism Implementation
Combines Channel Attention and Spatial Attention for enhanced feature learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    Learns inter-channel relationships using global pooling and MLP
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        out = avg_out + max_out
        attention = self.sigmoid(out).view(b, c, 1, 1)
        
        return x * attention.expand_as(x)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    Focuses on important spatial regions using convolution
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention_input))
        
        return x * attention


class DualAttention(nn.Module):
    """
    Dual Attention Module
    Combines Channel Attention and Spatial Attention in parallel
    """
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(DualAttention, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
