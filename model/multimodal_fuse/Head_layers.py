import torch
import torch.nn.functional as F
from einops import repeat
from timm.models.layers import to_2tuple
from torch import nn
from einops.layers.torch import Rearrange
import numpy as np


def creat_norm_layer(norm_layer, channel, is_token=False):
    """
    Args:
        norm_layer (str): Normalization layer type, use 'BN' or 'LN'.
        channel (int): Input channels.
        is_token (bool): Whether to process token. Default: False
    """
    if not is_token:
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
    else:
        if norm_layer == 'LN':
            norm = nn.LayerNorm(channel)
        elif norm_layer == 'BN':
            norm = nn.Sequential(
                Rearrange('b d n -> b n d'),
                nn.BatchNorm1d(channel)
            )
        else:
            raise NotImplementedError(f"norm layer type does not exist, please check the 'norm_layer' arg!")

    return norm


class MSE(nn.Module):
    """
    Build MSE

    Args:
        in_chans (int): Number of input image channels. Default: 3
        out_chans (int): Number of output image channels. Default: 24

    Return shape: (b c H W)
    """

    def __init__(self, in_chans, out_chans, n_group=4, use_pos=True, channel_attn_type='SE', ratio=16):
        super().__init__()
        self.use_pos = use_pos

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Conv2d(out_chans, out_chans // 2, kernel_size=1, bias=False)
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_chans // 2, out_chans // 2, kernel_size=3, padding=1, groups=n_group),
            nn.BatchNorm2d(out_chans // 2),
            nn.Conv2d(out_chans // 2, out_chans, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        if channel_attn_type == 'SE':
            self.attn = SE_channel_attention(out_chans, ratio)
        else:
            self.attn = CBAM_channel_attention(out_chans, ratio)

    def forward(self, x, pos):
        x = self.conv1(x)
        short_cut = x
        x = self.conv2(x)
        if self.use_pos:
            b, c, H, W = x.shape
            pos = repeat(pos, '1 -> b c H W', b=b, c=c, H=H, W=W)
            x = x + pos
        x = self.conv3(x)
        x = x + short_cut
        x = self.attn(x)

        return x


class AMM(nn.Module):
    """
    Creat AMM module

    Args:
        in_chans (int): Number of input image channels.
        out_chans (int): Number of output image channels.
        n_heads (int): Number of attention heads. Default: 4
        n_branch (int): Number of branches.
        patch_size (int | tuple[int]): Patch size. Default: 4
        n_heads (int): Number of attention heads.
        fuse_drop (float): Dropout rate.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True

    Return shape: (b c h w)
    """

    def __init__(self, in_chans,
                 out_chans,
                 n_branch,
                 offset_scale=16,
                 patch_size=4,
                 n_heads=4,
                 fuse_drop=0.,
                 qkv_bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.patch_size = to_2tuple(patch_size)

        self.short_cut_conv = nn.Sequential(nn.Conv2d(in_chans, out_chans, kernel_size=patch_size, stride=patch_size),
                                            creat_norm_layer('LN', out_chans))

        self.q = nn.Conv2d(in_chans, in_chans, kernel_size=1, bias=qkv_bias, groups=n_branch)
        self.k = nn.Conv2d(in_chans, in_chans, kernel_size=1, bias=qkv_bias, groups=n_branch)
        self.v = nn.Conv2d(in_chans, in_chans, kernel_size=1, bias=qkv_bias, groups=n_branch)
        self.q_proj = nn.Sequential(nn.MaxPool2d(offset_scale, stride=offset_scale),
                                    nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=1, groups=in_chans))
        self.k_proj = nn.Sequential(nn.MaxPool2d(offset_scale, stride=offset_scale),
                                    nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=1, groups=in_chans))
        self.v_proj = nn.Conv2d(in_chans, in_chans, kernel_size=patch_size, stride=patch_size, groups=in_chans)
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((n_heads, 1, 1))), requires_grad=True)

        self.cpb_mlp = nn.Sequential(nn.Linear(1, 16 * n_branch, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(16 * n_branch, n_heads, bias=False))

        coords = torch.zeros([in_chans, in_chans], dtype=torch.int64)
        for idx in range(in_chans):
            coords[idx] = torch.arange(in_chans) - idx
        relative_position_bias = coords / coords.max()
        relative_position_bias *= 8  # normalize to -8, 8
        relative_position_bias = torch.sign(relative_position_bias) * torch.log2(torch.abs(relative_position_bias) + 1.0) / np.log2(8)
        self.register_buffer("relative_position_bias", relative_position_bias.unsqueeze(-1))

        self.dropout = nn.Dropout(fuse_drop)
        self.norm = creat_norm_layer('LN', out_chans)
        self.softmax = nn.Softmax(dim=-1)
        self.softmax1 = nn.Softmax(dim=-1)
        self.proj = nn.Sequential(nn.Conv2d(in_chans, in_chans, kernel_size=1),
                                  nn.GELU(),
                                  nn.Conv2d(in_chans, out_chans, kernel_size=1))

    def forward(self, x):
        short_cut = x
        b, c, H, W = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)  # b, c, h, w
        q, k, v = self.q_proj(q).flatten(2), self.k_proj(k).flatten(2), self.v_proj(v).flatten(2)  # b, c, h*w
        q = q.reshape(b, c, self.n_heads, -1).permute(0, 2, 1, 3)  # b, n, c, h*w//n
        k = k.reshape(b, c, self.n_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(b, c, self.n_heads, -1).permute(0, 2, 1, 3)

        # cosine attention
        sim = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01).to(sim.device))).exp()
        sim = sim * logit_scale

        relative_position_bias = self.cpb_mlp(self.relative_position_bias).view(-1, self.n_heads).view(c, c, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = torch.sigmoid(relative_position_bias)
        sim = sim + relative_position_bias.unsqueeze(0)

        sim = self.softmax1(1 - self.softmax(sim))
        sim = self.dropout(sim)
        x = (sim @ v).transpose(1, 2).reshape(b, c, -1)  # b, c, h*w
        x = x.view(b, -1, H // self.patch_size[0], W // self.patch_size[1])  # b, c, h, w
        x = self.proj(x)
        x = self.dropout(x)
        x = self.norm(x) + self.short_cut_conv(short_cut)

        return x, short_cut


class SE_channel_attention(nn.Module):
    """
    Build channel attention based on SE

    Args:
        in_chans (int): Number of input image channels.
        ratio (int): Scaling ratio. Default: 4
        act_layer (nn.Module): Act layer. Default: nn.ReLU6

    Return shape: (b c h w)
    """

    def __init__(self, in_chans, ratio=4, act_layer=nn.ReLU6):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_chans, in_chans // ratio, kernel_size=1, bias=False),
            act_layer(inplace=True) if act_layer != nn.GELU else act_layer(),
            nn.Conv2d(in_chans // ratio, in_chans, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attn(x)


class CBAM_channel_attention(nn.Module):
    """
    Build channel attention based on CBAM

    Args:
        in_chans (int): Number of input image channels.
        ratio (int): Scaling ratio. Default: 4

    Return shape: (b c h w)
    """

    def __init__(self, in_chans, ratio=4, act_layer=nn.ReLU6):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(in_chans, in_chans // ratio, kernel_size=1, bias=False)
        self.act_layer = act_layer(inplace=True) if act_layer != nn.GELU else act_layer()
        self.conv2 = nn.Conv2d(in_chans // ratio, in_chans, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv2(self.act_layer(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.act_layer(self.conv1(self.max_pool(x))))
        weight = self.sigmoid(avg_out + max_out)
        return x * weight
