"""
gMLP (Gated MLP) for Food-101 Classification
Paper: Pay Attention to MLPs
https://arxiv.org/abs/2105.08050
"""
import torch
import torch.nn as nn
import numpy as np

class gMLPBlock(nn.Module):
    """
    gMLP block with spatial gating mechanism.
    Replaces self-attention with gated linear projections.
    """
    def __init__(self, dim, seq_len, mlp_ratio=4., **kwargs):
        super(gMLPBlock, self).__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.gate = nn.Linear(seq_len, seq_len)
        self.activation = nn.GELU()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, dim)
        shortcut = x
        x = self.norm(x)
        
        # MLP projection
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        
        # Spatial gating
        gates = self.gate(x.transpose(1, 2)).transpose(1, 2)
        x = x * gates
        
        return x + shortcut


class gMLP(nn.Module):
    """
    gMLP model for image classification.
    Divides image into patches and applies MLP with gating.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=5,
                 embed_dim=256, depth=12, mlp_ratio=4., drop_path_rate=0.1, **kwargs):
        super(gMLP, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)
        
        # gMLP blocks
        self.blocks = nn.ModuleList([
            gMLPBlock(dim=embed_dim, seq_len=self.num_patches + 1, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (B, C, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # gMLP blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        x = x[:, 0]  # Use class token
        x = self.head(x)
        
        return x


def gmlp_tiny(num_classes=5, **kwargs):
    """
    gMLP-Tiny variant for Food-101 classification.
    """
    model = gMLP(
        img_size=kwargs.pop('img_size', 224),
        patch_size=kwargs.pop('patch_size', 16),
        in_chans=kwargs.pop('in_chans', 3),
        num_classes=num_classes,
        embed_dim=kwargs.pop('embed_dim', 256),
        depth=kwargs.pop('depth', 12),
        mlp_ratio=kwargs.pop('mlp_ratio', 4.),
        drop_path_rate=kwargs.pop('drop_path_rate', 0.1),
        **kwargs
    )
    return model