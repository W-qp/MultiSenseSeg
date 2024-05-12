import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from torch import nn


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
                Rearrange('b n d -> b d n'),
                nn.BatchNorm1d(channel),
                Rearrange('b d n -> b n d'),
            )
        else:
            raise NotImplementedError(f"norm layer type does not exist, please check the 'norm_layer' arg!")

    return norm


def sa_window_partition(x, window_size):
    """
    Args:
        x: (b, H, W, c)
        window_size (int): window size

    Return shape:  (num_windows*b, window_size, window_size, c)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def sa_window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Return shape:  (b, H, W, c)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class MLP(nn.Module):
    """
    Multilayer perceptron

    Args:
        d (int): Dim of Input and output tokens. Default: 96
        hidden_dim (int): Dim of hidden MSEs. Default: None
        out_dim (int): Dim of output MSEs. Default: None
        act_layer (optional): Act layer. Default: nn.GELU
        drop (float, optional): Dropout ratio of MLP. Default: 0.0

    Return shape: (b n d)
    """

    def __init__(self, d=96, hidden_dim=None, out_dim=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_dim = out_dim or d
        hidden_dim = hidden_dim or d
        self.fc1 = nn.Linear(d, hidden_dim)
        self.act_layer = act_layer(inplace=True) if act_layer != nn.GELU else act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, *_):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class CNNMlp(nn.Module):
    """
    Multilayer perceptron based on CNN

    Args:
        in_chans (int): Number of input image channels.
        hidden_chans (int): Number of hidden layer image channels.
        group_dim (int | None):  Group dimension.
        drop (float, optional): Dropout ratio of MLP. Default: 0.0
        norm_layer (str): Normalization layer type, use 'BN' or 'LN'. Default: 'BN'
        act_layer (optional): Act layer. Default: nn.ReLU

    Return shape: (b c H W)
    """

    def __init__(self, in_chans, hidden_chans, group_dim, drop=0., norm_layer='BN', act_layer=nn.ReLU):
        super().__init__()
        if group_dim is None:
            n_group = 1
        else:
            assert in_chans % group_dim == 0, f"The total number of channels is {in_chans}, while the group dimension is {group_dim}."
            n_group = in_chans // group_dim

        self.convup = nn.Sequential(
            nn.Conv2d(in_chans, hidden_chans, kernel_size=1, groups=n_group),
            act_layer(inplace=True) if act_layer != nn.GELU else act_layer()
        )
        self.dw_conv = nn.Sequential(
            nn.Conv2d(hidden_chans, hidden_chans, kernel_size=3, padding=1, bias=False, groups=hidden_chans),
            creat_norm_layer(norm_layer, hidden_chans),
            act_layer(inplace=True) if act_layer != nn.GELU else act_layer(),
        )
        self.convdown = nn.Conv2d(hidden_chans, in_chans, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x, C, H, W):
        x = x.transpose(1, 2).contiguous().view(-1, C, H, W)
        short_cut = x
        x = self.convup(x)
        x = self.drop(x)
        x = self.dw_conv(x)
        x = self.drop(x)
        x = self.convdown(x)
        x = self.drop(x)
        x = short_cut + x

        return x.flatten(2).transpose(1, 2)


class FC_window_self_attention(nn.Module):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias and fully connection.
    It supports both of shifted and non-shifted window.

    Args:
        d (int): Dim of input tokens.
        window_size (tuple[int]): The height and width of the window.
        n_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        ratio (float | None):  Scaling ratio of q, k. Default: None

    Return shape: (num_windows*b n d)
    """

    def __init__(self, d, window_size, n_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., ratio=None):
        super().__init__()
        self.dim = d
        self.window_size = window_size
        self.n_heads = n_heads
        head_dim = d // n_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), n_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        qkv_dim = d * 3 if ratio is None else d + 2 * (d // ratio // n_heads) * n_heads
        self.qkv = nn.Linear(d, int(qkv_dim), bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d, d)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*b, n, d)   d=c  n=WindowH*WindowW
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, n, d = x.shape
        q_k_v = self.qkv(x)
        q, k = q_k_v[..., :-d].chunk(2, dim=-1)
        v = q_k_v[..., -d:]
        q = q.reshape(B_, n, self.n_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B_, n, self.n_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B_, n, self.n_heads, -1).permute(0, 2, 1, 3)

        qk = (q @ k.transpose(-2, -1)) * self.scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        qk = qk + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            qk = qk.view(B_ // nW, nW, self.n_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            qk = qk.view(-1, self.n_heads, n, n)
            qk = self.softmax(qk)
        else:
            qk = self.softmax(qk)

        qk = self.attn_drop(qk)

        x = (qk @ v).transpose(1, 2).reshape(B_, n, d)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CNN_window_self_attention(nn.Module):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias and convolution.
    It supports both of shifted and non-shifted window.

    Args:
        d (int): Dim of input tokens.
        window_size (tuple[int]): The height and width of the window.
        n_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        ratio (float | None):  Scaling ratio of q, k. Default: None

    Return shape: (num_windows*b n d)
    """

    def __init__(self, d, window_size, n_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., ratio=None):
        super().__init__()
        self.dim = d
        self.window_size = window_size
        self.Wh, self.Ww = window_size
        self.n_heads = n_heads
        head_dim = d // n_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), n_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        qkv_dim = d * 3 if ratio is None else d + 2 * (d // ratio // n_heads) * n_heads
        self.qkv = nn.Conv2d(d, int(qkv_dim), kernel_size=1, bias=qkv_bias)  # change kernel_size and padding to extract local context
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(d, d, kernel_size=1, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*b, n, d)   d=c  n=WindowH*WindowW
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        x = rearrange(x, 'B (Wh Ww) d -> B d Wh Ww', Wh=self.Wh, Ww=self.Ww, d=self.dim)
        q_k_v = self.qkv(x)
        q, k = q_k_v[:, :-self.dim, ...].flatten(2).permute(0, 2, 1).chunk(2, dim=-1)
        v = q_k_v[:, -self.dim:, ...].flatten(2).permute(0, 2, 1)
        B_, n, d = v.shape
        q = q.reshape(B_, n, self.n_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B_, n, self.n_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B_, n, self.n_heads, -1).permute(0, 2, 1, 3)

        qk = (q @ k.transpose(-2, -1)) * self.scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        qk = qk + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            qk = qk.view(B_ // nW, nW, self.n_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            qk = qk.view(-1, self.n_heads, n, n)
            qk = self.softmax(qk)
        else:
            qk = self.softmax(qk)

        qk = self.attn_drop(qk)

        x = (qk @ v).transpose(1, 2).reshape(B_, n, d)
        x = rearrange(x, 'B (Wh Ww) d -> B d Wh Ww', Wh=self.Wh, Ww=self.Ww, d=self.dim)
        x = self.proj(x)
        x = rearrange(x, 'B d Wh Ww -> B (Wh Ww) d', Wh=self.Wh, Ww=self.Ww, d=self.dim)
        x = self.proj_drop(x)
        return x


class PatchEmbed_block(nn.Module):
    """
    Image to Patch Embedding

    Args:
        patch_size (int): Size of input patches. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        out_chans (int): Number of output image channels. Default: 96

    Return shape: (b c H W)
    """

    def __init__(self, patch_size=4, in_chans=3, out_chans=96):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.embed = nn.Conv2d(in_chans, out_chans, kernel_size=patch_size, stride=patch_size)
        self.ln = nn.LayerNorm(out_chans)

    def forward(self, x):
        _, _, H, W = x.shape
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.embed(x)
        _, c, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = x.transpose(1, 2).view(-1, c, H, W)

        return x


class Downsampling_block(nn.Module):
    """
    Patch Merging Layer

    Args:
        in_chans (int): Number of input image channels.
        out_chans (int): Number of output layer image channels.
    """
    def __init__(self, in_chans, out_chans):
        super().__init__()
        dim = in_chans * 4
        self.reduction = nn.Linear(dim, out_chans, bias=False)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.ln(x)
        x = self.reduction(x)

        return x


class Basic_block(nn.Module):
    """
    Extract the local-information and long-distance dependencies of the image

    Args:
        d (int): Number of input channels
        n_heads (int): Number of attention heads
        window_size (int): Window size. Default: 8
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qk_ratio (float | None):  Scaling ratio of q, k. Default: None
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        qkv_type (str): Type of qkv, use 'CNN' or 'FC'.  Default: CNN
        ffn_type (str): Type of feed forword net, use 'CNN' or 'FC'.  Default: CNN
        norm_layer (str): Normalization layer type, use 'BN' or 'LN'. Default: 'LN'
        act_layer (optional): Act layer. Default: nn.GELU
        group_dim (int | None):  Group dimension. Default: 16
        idx2group (int): Use '0', or '1' to control local-distance extraction block.

    Return shape: (b n d)
    """

    def __init__(self, d, n_heads, window_size=8, shift_size=0, mlp_ratio=4., qk_ratio=None,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., qkv_type='CNN',
                 ffn_type='CNN', norm_layer='LN', act_layer=nn.GELU, group_dim=16, idx2group=None):
        super().__init__()
        self.dim = d
        self.num_heads = n_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.ffn_type = ffn_type
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        assert qkv_type in {'CNN', 'FC'}, "qkv type must be either 'CNN' (cnn feed forward) or 'FC' (mlp feed forward)"
        assert ffn_type in {'CNN', 'FC'}, "ffd type must be either 'CNN' (cnn feed forward) or 'FC' (mlp feed forward)"
        self.H, self.W = None, None
        group_dims = [group_dim, None]

        if qkv_type == 'CNN':
            self.attn = CNN_window_self_attention(
                d, window_size=to_2tuple(self.window_size), n_heads=n_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, ratio=qk_ratio)
        elif qkv_type == 'FC':
            self.attn = FC_window_self_attention(
                d, window_size=to_2tuple(self.window_size), n_heads=n_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, ratio=qk_ratio)

        self.norm1 = nn.LayerNorm(d)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(d * mlp_ratio)

        if ffn_type == 'CNN':
            self.mlp = CNNMlp(in_chans=d, hidden_chans=mlp_hidden_dim, group_dim=group_dims[idx2group], drop=drop, act_layer=act_layer)
        elif ffn_type == 'FC':
            self.mlp = MLP(d=d, hidden_dim=mlp_hidden_dim, drop=drop, act_layer=nn.GELU)

        self.norm2 = creat_norm_layer(norm_layer, d, True)

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "Input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = sa_window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = sa_window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        x = self.drop_path(self.mlp(self.norm2(x), C, H, W)) + x

        return x


class BasicLayer(nn.Module):
    """
    A basic layer for one stage.

    Args:
        d (int): Number of feature channels.
        group_dim (int):  Group dimension.
        depth (int): Depths of this stage.
        n_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 8.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qk_ratio (float | None):  Scaling ratio of q, k. Default: 3
        down_ratio (int): Amplification ratio of the number of dim after downsampling. Default: 2.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        qkv_type (str): Type of qkv, use 'CNN' or 'FC'.  Default: CNN
        ffn_type (str): Type of feed forword net, use 'CNN' or 'FC'.  Default: CNN
        norm_layer (str): Normalization layer type, use 'BN' or 'LN'. Default: 'LN'
        act_layer (optional): Act layer. Default: nn.GELU
    """

    def __init__(self,
                 d,
                 group_dim,
                 depth,
                 n_heads,
                 window_size=8,
                 mlp_ratio=4.,
                 qk_ratio=3,
                 down_ratio=2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 downsample=None,
                 use_checkpoint=False,
                 qkv_type='CNN',
                 ffn_type='CNN',
                 norm_layer='LN',
                 act_layer=nn.GELU):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.long_blocks = nn.ModuleList([
            Basic_block(
                d=d,
                n_heads=n_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qk_ratio=qk_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                qkv_type=qkv_type,
                ffn_type=ffn_type,
                norm_layer=norm_layer,
                act_layer=act_layer,
                group_dim=group_dim,
                idx2group=i % 2)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(in_chans=d, out_chans=d * down_ratio)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H: Spatial resolution of the input feature height.
            W: Spatial resolution of the input feature width.
        """
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = sa_window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for i, blk in enumerate(self.long_blocks):
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W
