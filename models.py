from AS_MLP.AS_MLP import as_mlp_tiny, as_mlp_deep
from DeiT.DeiT import local_deit_tiny_patch16_224, local_deit_tiny_distilled_patch16_224
from ResNeXt.ResNeXt import resnext101_32x8d, resnext50_32x4d
from ConvMLP.ConvMLP import convmlp_s
from DenseNet.DenseNet import densenet121
from PVT.pvt import pvt_tiny 
from ResNet18.ResNet18 import resnet18
from gMLP.gMLP import gmlp_tiny
from ViT.ViT import vit_base


# Add imports for other models as needed

import torch
import torchvision.models as models
import torch.nn as nn

def get_model_config(model_name):
    if model_name == 'as_mlp_tiny':
        from AS_MLP.config import get_config
        return get_config(variant='as_mlp_tiny')
    elif model_name == 'as_mlp_deep':
        from AS_MLP.config import get_config
        return get_config(variant='as_mlp_deep')
    elif model_name == 'deit_tiny':
        from DeiT.config import get_config
        return get_config(variant='deit_tiny')
    elif model_name == 'deit_tiny_distilled':
        from DeiT.config import get_config
        return get_config(variant='deit_tiny_distilled')
    elif model_name == 'resnext50_local':
        from ResNeXt.config import get_config
        return get_config(variant='resnext50_32x4d')
    elif model_name == 'resnext101_local':
        from ResNeXt.config import get_config
        return get_config(variant='resnext101_32x8d')
    elif model_name == 'densenet121':
        from DenseNet.config import get_config
        return get_config()
    elif model_name == 'convmlp_s':
        from ConvMLP.config import get_config
        return get_config()
    elif model_name == 'pvt_v2_b0':
        from PVT.config import get_config
        return get_config()
    elif model_name == 'resnet18':          
        from ResNet18.config import get_config
    elif model_name == 'gmlp_tiny':         
        from gMLP.config import get_config
    elif model_name == 'vit_base':          
        from ViT.config import get_config
    else:
        raise ValueError(f"Unknown model: {model_name}")


def build_model(config):
    model_name = config.MODEL.NAME
    
    # Extract all parameters from config.MODEL to pass to the model builder
    # We lowercase the keys to match Python argument naming conventions
    model_kwargs = {k.lower(): v for k, v in config.MODEL.__dict__.items() if k != 'NAME' and not k.startswith('_')}

    # 1. Local Models
    if model_name == 'as_mlp_tiny':
        return as_mlp_tiny(**model_kwargs)
    elif model_name == 'as_mlp_deep':
        return as_mlp_deep(**model_kwargs)
    elif model_name == 'deit_tiny':
        return local_deit_tiny_patch16_224(**model_kwargs)
    elif model_name == 'deit_tiny_distilled':
        return local_deit_tiny_distilled_patch16_224(**model_kwargs)
    elif model_name == 'resnext50_local':
        return resnext50_32x4d(**model_kwargs)
    elif model_name == 'resnext101_local':
        return resnext101_32x8d(**model_kwargs)
    elif model_name == 'densenet121':
        return densenet121(**model_kwargs)
    elif model_name == 'convmlp_s':
        return convmlp_s(**model_kwargs)
    elif model_name == 'pvt_v2_b0':
        return pvt_tiny(**model_kwargs)
    elif model_name == 'resnet18':          
        return resnet18(**model_kwargs)
    elif model_name == 'gmlp_tiny':         
        return gmlp_tiny(**model_kwargs)
    elif model_name == 'vit_base':          
        return vit_base(**model_kwargs)



    # continue defining other models
