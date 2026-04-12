import torch
import torch.nn as nn

# DropPath (Stochastic Depth)
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob


    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# Patch Embedding Layer
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)


    def forward(self, x):
        x = self.proj(x)  
        x = x.flatten(2).transpose(1, 2) 
        return x

# Residual MLP Block
class ResidualMLPBlock(nn.Module):
    def __init__(self, num_patches, embed_dim, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)


        # Cross-patch mixing (spatial)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.fc_patch = nn.Linear(num_patches, num_patches)

        # Cross-channel MLP
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc_channel = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )


        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        # Spatial mixing
        x = x + self.drop_path(self.fc_patch(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        # Channel mixing
        x = x + self.drop_path(self.fc_channel(self.norm2(x)))
        return x

# ResMLP-S12 Model
class ResMLP_S12(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=5,
                 embed_dim=384, depth=12, mlp_ratio=4.0, drop_path=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches


        self.blocks = nn.ModuleList([
            ResidualMLPBlock(num_patches, embed_dim, mlp_ratio, drop_path)
            for _ in range(depth)
        ])


        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)


    def forward(self, x):
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x).mean(dim=1)  
        x = self.head(x)
        return x

# Helper function
def resmlp_s12(num_classes=5):
    return ResMLP_S12(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        mlp_ratio=4.0,
        drop_path=0.1,
    )
