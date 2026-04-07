"""
Vision Transformer (ViT) for Food-101 Classification
Paper: An Image is Worth 16x16 Words
https://arxiv.org/abs/2010.11929
"""
import torch
import torch.nn as nn
import numpy as np

class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.proj(x)  # (B, C, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with multi-head self-attention and MLP."""
    def __init__(self, dim, num_heads=12, mlp_ratio=4., **kwargs):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x):
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual connection
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for image classification.
    Treats image as a sequence of patches and applies transformer encoder.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=5,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., 
                 drop_path_rate=0.1, **kwargs):
        super(VisionTransformer, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        
        # Class token (learnable parameter for classification)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
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
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification: use class token
        x = self.norm(x)
        x = x[:, 0]  # Extract class token
        x = self.head(x)
        
        return x


def vit_base(num_classes=5, **kwargs):
    """
    Vision Transformer Base variant for Food-101.
    """
    model = VisionTransformer(
        img_size=kwargs.pop('img_size', 224),
        patch_size=kwargs.pop('patch_size', 16),
        in_chans=kwargs.pop('in_chans', 3),
        num_classes=num_classes,
        embed_dim=kwargs.pop('embed_dim', 768),
        depth=kwargs.pop('depth', 12),
        num_heads=kwargs.pop('num_heads', 12),
        mlp_ratio=kwargs.pop('mlp_ratio', 4.),
        drop_path_rate=kwargs.pop('drop_path_rate', 0.1),
        **kwargs
    )
    return model