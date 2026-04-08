import torch
import torch.nn as nn
import torchvision.models as models

def efficientnet_b0(num_classes=5, pretrained=True, freeze_backbone=True, **kwargs):
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    )

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    
    return model