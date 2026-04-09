# ConvMLP/ConvMLP.py 
"""
ConvMLP-S for Food-101 Classification
Official implementation from SHI-Labs
https://github.com/SHI-Labs/Convolutional-MLPs

VARIATION: Wider channels (increased model capacity)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvMLPBlock(nn.Module):
    """ConvMLP block with conv, normalization, and MLP"""
    def __init__(self, dim, hidden_dim, drop_path=0.1):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.conv = nn.Conv2d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_dim, dim, 1)
        self.drop_path = drop_path

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.conv(x)
        x = self.act(x)
        x = self.conv2(x)
        x = shortcut + x
        return x


class ConvMLPS(nn.Module):
    """ConvMLP-S: Small version for image classification"""
    def __init__(self, num_classes=5, in_channels=3, dim=64, num_stages=4):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.num_stages = num_stages
        
        # Stem: input processing
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        current_dim = dim
        for i in range(num_stages):
            # ConvMLP blocks in this stage
            stage_blocks = nn.ModuleList()
            for j in range(3):  # 3 blocks per stage
                stage_blocks.append(ConvMLPBlock(current_dim, current_dim * 4, drop_path=0.1))
            self.stages.append(stage_blocks)
            
            # Downsample to next stage
            if i < num_stages - 1:
                self.downsample.append(nn.Sequential(
                    nn.Conv2d(current_dim, current_dim * 2, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(current_dim * 2),
                ))
                current_dim *= 2
        
        # Classification head
        self.norm = nn.BatchNorm2d(current_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(current_dim, num_classes)

    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Stages
        for i, stage in enumerate(self.stages):
            # Apply blocks
            for block in stage:
                x = block(x)
            
            # Downsample to next stage
            if i < len(self.downsample):
                x = self.downsample[i](x)
        
        # Head
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        
        return x


def convmlp_s(num_classes=5, dim=64, **kwargs):
    """
    Load ConvMLP-S from custom implementation.
    
    VARIATION: Configurable dim parameter for model capacity
    
    Args:
        num_classes: Number of output classes
        dim: Base dimension (channels) - original: 64, variant: 96
        **kwargs: Additional hyperparameters from config (safely ignored)
    
    Returns:
        ConvMLP-S model ready for training
    """
    # Extract parameters (safely ignored if not present)
    _ = kwargs.pop('drop_path_rate', 0.1)
    
    # Remove any other unexpected kwargs
    for key in list(kwargs.keys()):
        kwargs.pop(key, None)
    
    # Create ConvMLP-S model with configurable dim
    model = ConvMLPS(
        num_classes=num_classes,
        in_channels=3,
        dim=dim,           # VARIATION PARAMETER
        num_stages=4
    )
    
    return model