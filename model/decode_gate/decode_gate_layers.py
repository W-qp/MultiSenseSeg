from einops.layers.torch import Rearrange
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
                Rearrange('b d n -> b n d'),
                nn.BatchNorm1d(channel)
            )
        else:
            raise NotImplementedError(f"norm layer type does not exist, please check the 'norm_layer' arg!")

    return norm


class Spatial_attention(nn.Module):
    """
    Build spatial attention

    Args:
        encoder_chans (int): Input channels from encoder.
        decoder_chans (int): Input channels from decoder.
        act_layer (nn.Module): Act layer. Default: nn.ReLU

    Return shape: (b c h w)
    """

    def __init__(self, encoder_chans, decoder_chans, attn_chans=None, act_layer=nn.ReLU):
        super().__init__()
        attn_chans = attn_chans or decoder_chans
        self.conv1 = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Conv2d(encoder_chans, attn_chans, kernel_size=1),
            nn.BatchNorm2d(attn_chans)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(decoder_chans, attn_chans, kernel_size=1),
            nn.BatchNorm2d(attn_chans)
        )
        self.attn = nn.Sequential(
            act_layer(inplace=True) if act_layer != nn.GELU else act_layer(),
            nn.Conv2d(attn_chans, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x_en, x_de):
        """
        x_en: feature map from encoder
        x_de: Feature map from decoder
        """
        x_en = self.conv1(x_en)
        x_de = self.conv2(x_de)

        return x_de * self.attn(x_en + x_de)


class Dw_spatial_attention(nn.Module):
    """
    Build spatial attention by downscaling convolution

    Args:
        in_chans (int): Input channels.

    Return shape: (b c h w)
    """

    def __init__(self, in_chans):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_chans, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, _, x):
        return x * self.attn(x)


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
