from .Neck_layers import *


class Build_neck(nn.Module):
    """
    Build neck based on UperNet

    Args:
        in_chans (int): Number of input image channels.
        out_chans (int): Number of FPN output image channels.
        depth (int): Total stages.
        pool_sizes (list(int) | tuple(int)): Pool size. Default: (1, 2, 3, 6)
        norm_layer (str): Normalization layer type, use 'BN' or 'LN'. Default: 'BN'
        act_layer (optional): Act layer. Default: nn.ReLU

    Return shape: (b c H W)
    """

    def __init__(self, in_chans, out_chans, depth, pool_sizes=(1, 2, 3, 6), norm_layer='BN', act_layer=nn.ReLU):
        super().__init__()
        self.ppm_head = PPM(ppm_in_chans=in_chans * 2 ** (depth - 1),
                            out_chans=out_chans,
                            pool_sizes=pool_sizes,
                            norm_layer=norm_layer,
                            act_layer=act_layer)
        self.fpn_neck = FPN_neck(in_chans=in_chans,
                                 out_chans=out_chans,
                                 depth=depth,
                                 norm_layer=norm_layer,
                                 act_layer=act_layer)

    def forward(self, x):
        x = list(x)[::-1]
        x[0] = self.ppm_head(x[0])
        x = self.fpn_neck(x)

        return x
