import torch
import torch.nn as nn
import timm

def MLP_Mixer(num_classes=5, pretrained=True, freeze_backbone=True):
    model = timm.create_model("mixer_b16_224", pretrained=pretrained)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    num_features = model.head.in_features
    model.head = nn.Linear(num_features, num_classes)

    return model