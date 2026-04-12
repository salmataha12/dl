import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# Convolutional Token Embedding
class ConvTokenEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, kernel_size, stride, padding):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  
        B, E, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  
        x = self.norm(x)
        return x, H, W


# Convolutional Attention (Q/K/V via depthwise conv)
class ConvAttention(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=3, stride=1,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # Depthwise conv for qkv
        self.qkv = nn.Conv2d(dim, dim * 3,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=kernel_size//2,
                             groups=dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        qkv = self.qkv(x_2d).reshape(B, 3, self.num_heads, C // self.num_heads, H*W)
        q, k, v = qkv[:,0], qkv[:,1], qkv[:,2]  # [B, heads, dim_head, HW]

        q = q.permute(0,1,3,2)  # [B, heads, HW, dim_head]
        k = k.permute(0,1,3,2)
        v = v.permute(0,1,3,2)

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(2,3).reshape(B, H*W, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ConvAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
      

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# CvT-13 Model
class CvT13_local(nn.Module):
    def __init__(self, num_classes=5, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        # Stage 1: 64 dim, 1 block, 1 head
        self.stage1 = ConvTokenEmbedding(3, 64, kernel_size=7, stride=4, padding=3)
        self.blocks1 = nn.ModuleList([TransformerBlock(64, 1, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate) for _ in range(1)])

        # Stage 2: 192 dim, 2 blocks, 3 heads
        self.stage2 = ConvTokenEmbedding(64, 192, kernel_size=3, stride=2, padding=1)
        self.blocks2 = nn.ModuleList([TransformerBlock(192, 3, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate) for _ in range(2)])

        # Stage 3: 384 dim, 10 blocks, 6 heads
        self.stage3 = ConvTokenEmbedding(192, 384, kernel_size=3, stride=2, padding=1)
        self.blocks3 = nn.ModuleList([TransformerBlock(384, 6, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate) for _ in range(10)])

        self.norm = nn.LayerNorm(384)
        self.head = nn.Linear(384, num_classes)

    def forward(self, x):
        B = x.size(0)

        # Stage 1
        x, H, W = self.stage1(x)
        for blk in self.blocks1:
            x = blk(x, H, W)

        # Stage 2
        x, H, W = self.stage2(x.transpose(1,2).reshape(B, -1, H, W))
        for blk in self.blocks2:
            x = blk(x, H, W)

        # Stage 3
        x, H, W = self.stage3(x.transpose(1,2).reshape(B, -1, H, W))
        for blk in self.blocks3:
            x = blk(x, H, W)

        x = self.norm(x)
        x = x.mean(dim=1)  
        return self.head(x)


def create_cvt13_local(num_classes=5, **kwargs):
    return CvT13_local(num_classes=num_classes, **kwargs)