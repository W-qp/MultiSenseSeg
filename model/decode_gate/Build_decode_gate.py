import torch.nn.functional as F
from model.decode_gate.decode_gate_layers import *


class Build_decode_gate(nn.Module):
    """
    Build decode gate

    Args:
        in_chans (int): Number of input image channels.
        n_classes (int): Number of probabilities you want to get per pixel.
        norm_layer (str): Normalization layer.
        act_layer (optional): Act layer.
        head_chans (int | None): Number of decoder head image channels.
        chan_ratio (int): Scaling ratio of 'CBAM' or 'SE' channel attention. Default: 16
        chan_attn_type (str): Channnel attention method, using 'CBAM' or 'SE'. Default: 'SE'
        dw_spac_attn (bool): Whether to use 'dw' spatial attention.
        en_chans (int | None): Number of encoder feature channels.

    Return shape: (b c H W)
    """

    def __init__(self, in_chans, n_classes, norm_layer, act_layer, head_chans=None,
                 chan_ratio=16, chan_attn_type='SE', dw_spac_attn=False, en_chans=None):
        super().__init__()
        head_chans = head_chans or in_chans // 2

        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, head_chans, kernel_size=3, padding=1, bias=False),
            creat_norm_layer(norm_layer, head_chans)
        )

        self.spat_attn = Spatial_attention(encoder_chans=en_chans,
                                           decoder_chans=head_chans,
                                           attn_chans=None,
                                           act_layer=act_layer
                                           ) if not dw_spac_attn else Dw_spatial_attention(head_chans)

        self.dwconv = nn.Sequential(
            nn.Conv2d(head_chans, head_chans, kernel_size=3, padding=1, groups=head_chans),
            creat_norm_layer(norm_layer, head_chans),
            nn.Conv2d(head_chans, in_chans, kernel_size=1, bias=False)
        )

        self.out = nn.Sequential(
            act_layer(inplace=True) if act_layer != nn.GELU else act_layer(),
            nn.Conv2d(in_chans, n_classes, kernel_size=1)
        )

        if chan_attn_type == 'CBAM':
            self.chan_attn = CBAM_channel_attention(in_chans=head_chans, ratio=chan_ratio)
        elif chan_attn_type == 'SE':
            self.chan_attn = SE_channel_attention(in_chans=head_chans, ratio=chan_ratio)
        else:
            raise NotImplementedError(f"Build channel attention does not support {chan_attn_type}")


    def forward(self, x, x1):
        """
        x: Feature map from encoder
        x1: Feature map from decoder
        """
        short_cut = x1
        x1 = self.conv(x1)

        spat_x = self.spat_attn(x, x1)
        chan_x = self.chan_attn(x1)
        fuse_attn_x = self.dwconv(spat_x + chan_x)

        x = short_cut + fuse_attn_x
        x = self.out(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

        return x
