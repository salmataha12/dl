import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Shift(nn.Module):
    def __init__(self, shift_size, dim):
        super().__init__()
        self.shift_size = shift_size
        self.dim = dim # 2 for height, 3 for width

    def forward(self, x):
        B, C, H, W = x.shape
        pad = self.shift_size // 2
        x = F.pad(x, (pad, pad, pad, pad), "constant", 0)
        xs = torch.chunk(x, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, self.dim) for x_c, shift in zip(xs, range(-pad, pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, pad, H)
        x_cat = torch.narrow(x_cat, 3, pad, W)
        return x_cat

def MyNorm(dim):
    return nn.GroupNorm(1, dim)

class AxialShift(nn.Module):
    def __init__(self, dim, shift_size, as_bias=True, proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.shift_size = shift_size
        self.conv1 = nn.Conv2d(dim, dim, 1, 1, 0, bias=as_bias)
        self.conv2_1 = nn.Conv2d(dim, dim, 1, 1, 0, bias=as_bias)
        self.conv2_2 = nn.Conv2d(dim, dim, 1, 1, 0, bias=as_bias)
        self.conv3 = nn.Conv2d(dim, dim, 1, 1, 0, bias=as_bias)
        self.actn = nn.GELU()
        self.norm1 = MyNorm(dim)
        self.norm2 = MyNorm(dim)
        self.shift_dim2 = Shift(self.shift_size, 2)
        self.shift_dim3 = Shift(self.shift_size, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.actn(x)
        x_shift_lr = self.shift_dim3(x)
        x_shift_td = self.shift_dim2(x)
        x_lr = self.conv2_1(x_shift_lr)
        x_td = self.conv2_2(x_shift_td)
        x = self.actn(x_lr) + self.actn(x_td)
        x = self.norm2(x)
        x = self.conv3(x)
        return x

class AxialShiftedBlock(nn.Module):
    def __init__(self, dim, input_resolution, shift_size=7,
                 mlp_ratio=4., as_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=MyNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.axial_shift = AxialShift(dim, shift_size=shift_size, as_bias=as_bias, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.axial_shift(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=MyNorm):
        super().__init__()
        self.reduction = nn.Conv2d(4 * dim, 2 * dim, 1, 1, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], 1)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, shift_size,
                 mlp_ratio=4., as_bias=True, drop=0.,
                 drop_path=0., norm_layer=MyNorm, downsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            AxialShiftedBlock(dim=dim, input_resolution=input_resolution,
                              shift_size=shift_size, mlp_ratio=mlp_ratio,
                              as_bias=as_bias, drop=drop,
                              drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                              norm_layer=norm_layer)
            for i in range(depth)])
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample:
            x = self.downsample(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)
        if self.norm:
            x = self.norm(x)
        return x

class AS_MLP(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=5,
                 embed_dim=96, depths=[2, 2, 6, 2], shift_size=5, mlp_ratio=4.,
                 as_bias=True, drop_rate=0., drop_path_rate=0.1,
                 norm_layer=MyNorm, patch_norm=True, **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, 
                                      in_chans=in_chans, embed_dim=embed_dim,
                                      norm_layer=norm_layer if patch_norm else None)
        patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer], shift_size=shift_size,
                               mlp_ratio=mlp_ratio, as_bias=as_bias, drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

def as_mlp_tiny(num_classes=5, **kwargs):
    depths = kwargs.pop('depths', [2, 2, 6, 2])
    embed_dim = kwargs.pop('embed_dim', 96)
    shift_size = kwargs.pop('shift_size', 5)
    return AS_MLP(depths=depths, embed_dim=embed_dim, shift_size=shift_size, num_classes=num_classes, **kwargs)
