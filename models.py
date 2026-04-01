from AS_MLP.AS_MLP import as_mlp_tiny
from DeiT.DeiT import local_deit_tiny_patch16_224
from ResNeXt.ResNeXt import resnext50_32x4d
import torch
import torchvision.models as models
import torch.nn as nn

def get_model_config(model_name):
    if model_name == 'as_mlp_tiny':
        from AS_MLP.config import get_config
    elif model_name == 'deit_tiny':
        from DeiT.config import get_config
    elif model_name == 'resnext50_local':
        from ResNeXt.config import get_config
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return get_config()

def build_model(config):
    model_name = config.MODEL.NAME
    
    # Extract all parameters from config.MODEL to pass to the model builder
    # We lowercase the keys to match Python argument naming conventions
    model_kwargs = {k.lower(): v for k, v in config.MODEL.__dict__.items() if k != 'NAME' and not k.startswith('_')}

    # 1. Local Models
    if model_name == 'as_mlp_tiny':
        return as_mlp_tiny(**model_kwargs)
    elif model_name == 'deit_tiny':
        return local_deit_tiny_patch16_224(**model_kwargs)
    elif model_name == 'resnext50_local':
        return resnext50_32x4d(**model_kwargs)

    # continue defining other models
