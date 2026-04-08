import torch
import torch.nn as nn
import torchvision.models as models

def swin_transformer(num_classes=5, pretrained=True, freeze_backbone=True, **kwargs):
    model = models.swin_t(
        weights=models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
    )
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    num_features = model.head.in_features
    model.head = nn.Linear(num_features, num_classes)
    
    return model