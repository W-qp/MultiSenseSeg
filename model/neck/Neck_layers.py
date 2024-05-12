import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


def creat_norm_layer(norm_layer, channel):
    """
    Args:
        norm_layer (str): Normalization layer type, use 'BN' or 'LN'.
        channel (int): Input channels.
    """
    if norm_layer == 'LN':
        norm = nn.Sequential(
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(channel),
            Rearrange('b h w c -> b c h w')
        )
    elif norm_layer == 'BN':
        norm = nn.BatchNorm2d(channel)
    else:
        raise NotImplementedError(f"norm layer type does not exist, please check the 'norm_layer' arg!")
    return norm


class PPM(nn.Module):
    """
    Args:
        ppm_in_chans (int): Number of input image channels.
        out_chans (int): Number of output image channels. Default: 512
        pool_sizes (list(int) | tuple(int)): Pool size. Default: (1, 2, 3, 6)
        norm_layer (str): Normalization layer type, use 'BN' or 'LN'. Default: 'BN'
        act_layer (optional): Act layer. Default: nn.ReLU

    Return shape: (b c H W)
    """

    def __init__(self, ppm_in_chans, out_chans=512, pool_sizes=(1, 2, 3, 6), norm_layer='BN', act_layer=nn.ReLU):
        super().__init__()
        self.pool_projs = nn.ModuleList(
            nn.Sequential(
                nn.AdaptiveMaxPool2d(pool_size),
                nn.Conv2d(ppm_in_chans, out_chans, kernel_size=1, bias=False),
                creat_norm_layer(norm_layer, out_chans),
                act_layer(inplace=True) if act_layer != nn.GELU else act_layer()
            )for pool_size in pool_sizes)

        self.bottom = nn.Sequential(
            nn.Conv2d(ppm_in_chans + len(pool_sizes) * out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            creat_norm_layer(norm_layer, out_chans),
            act_layer(inplace=True) if act_layer != nn.GELU else act_layer()
        )

    def forward(self, x):
        xs = [x]
        for pool_proj in self.pool_projs:
            pool_x = F.interpolate(pool_proj(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
            xs.append(pool_x)

        x = torch.cat(xs, dim=1)
        x = self.bottom(x)

        return x


class FPN_neck(nn.Module):
    """
    FPN neck

    Args:
        in_chans (int): Number of input image channels.
        depth (int): Total stages.
        out_chans (int): Number of output image channels. Default: 512
        norm_layer (str): Normalization layer type, use 'BN' or 'LN'. Default: 'BN'
        act_layer (optional): Act layer. Default: nn.ReLU

    Return shape: (b c H W)
    """

    def __init__(self, in_chans, depth, out_chans=512, norm_layer='BN', act_layer=nn.ReLU):
        super().__init__()
        self.depth = depth
        stage = [i for i in range(depth)]

        self.conv_ = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_chans * 2 ** stage[::-1][i + 1], out_chans, kernel_size=1, bias=False),
                creat_norm_layer(norm_layer, out_chans),
                act_layer(inplace=True) if act_layer != nn.GELU else act_layer()
            )for i in range(depth - 1))

        self.fpn_conv = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
                creat_norm_layer(norm_layer, out_chans),
                act_layer(inplace=True) if act_layer != nn.GELU else act_layer()
            )for _ in range(depth - 1))

        self.out = nn.Sequential(
            nn.Conv2d(out_chans * depth, out_chans, kernel_size=3, padding=1, bias=False),
            creat_norm_layer(norm_layer, out_chans),
            act_layer(inplace=True) if act_layer != nn.GELU else act_layer()
        )

    def forward(self, x):
        fpn_x = x[0]
        out = [fpn_x]
        for i in range(self.depth - 1):
            fpn_x = F.interpolate(x[i], scale_factor=2, mode='bilinear', align_corners=True)
            fpn_x = self.fpn_conv[i](fpn_x) + self.conv_[i](x[i + 1])
            x[i + 1] = fpn_x
            out.append(fpn_x)
        out = out[::-1]
        _, _, H, W = out[0].shape
        for i in range(1, len(out)):
            out[i] = F.interpolate(out[i], size=(H, W), mode='bilinear', align_corners=True)
        x = torch.cat(out, dim=1)

        return self.out(x)
