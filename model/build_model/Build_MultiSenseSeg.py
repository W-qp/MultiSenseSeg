from model.multimodal_fuse.Build_multimodal_fuse_head import *
from model.pipeline.Build_pipeline import *
from model.neck.Build_neck import *
from model.decode_gate.Build_decode_gate import *


class Build_MultiSenseSeg(nn.Module):
    """
    Build MultiSenseSeg model

    Args:
        n_classes (int): Number of probabilities you want to get per pixel.
        n_branch (int): Number of branches. Default: None
        decoder_chans (int): Channel numbers of decoder.
        patch_size (int | tuple[int]): Patch size. Default: 4
        in_chans (int | tuple[int]): Number of input image channels. Default: 3
        head_out_chans (int): Number of feature refinement head output image channels. Default: 32
        group_dim (int): Dim of feature refinement head each groups. Default: 8
        use_pos (bool): Whether to add a position embedding to each branch. Default: False
        embed_dim (int): Number of output channels. Default: 96
        offset_scale (int): Scale of offset alignment.
        depths (tuple[int]): Depths of each stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 8
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qk_ratio (float | None):  Scaling ratio of q, k.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate.
        drop_path_rate (float): Stochastic depth rate.
        norm_layer (str): Normalization layer type, use 'BN' or 'LN'. Default: 'BN'
        act_layer (optional): Act layer. Default: nn.GELU
        fpn_norm_layer (str): Normalization layer type, use 'BN' or 'LN'. Default: 'BN'
        fpn_act_layer (optional): Act layer. Default: nn.ReLU
        patch_norm (bool): If True, add normalization after patch embedding.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode). -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        qkv_type (str): Type of qkv, use 'CNN' or 'FC'.
        ffn_type (str): Type of feed forword net, use 'CNN' or 'FC'.
        pool_sizes (list(int) | tuple(int)): Pool size. Default: (1, 2, 3, 6)
        chan_ratio (int): Scaling ratio of 'CBAM' or 'SE' channel attention.
        chan_attn_type (str): Channnel attention method, using 'CBAM' or 'SE'. Default: 'SE'
        aux (bool): Whether to use auxiliary classification head. Default: True
        use_faster (bool): Whether to use faster CNN version. Default: True
    """

    def __init__(self,
                 n_classes,
                 n_branch=None,
                 decoder_chans=512,
                 patch_size=4,
                 in_chans=3,
                 head_out_chans=32,
                 group_dim=8,
                 use_pos=True,
                 embed_dim=96,
                 offset_scale=8,
                 fuse_proj=None,
                 depths=(2, 2, 8, 2),
                 num_heads=(3, 6, 12, 24),
                 window_size=8,
                 mlp_ratio=4.,
                 qk_ratio=1.5,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.1,
                 attn_drop_rate=0.1,
                 drop_path_rate=0.1,
                 norm_layer='BN',
                 act_layer=nn.GELU,
                 fpn_norm_layer='BN',
                 fpn_act_layer=nn.ReLU,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 qkv_type='FC',
                 ffn_type='CNN',
                 pool_sizes=(1, 2, 3, 6),
                 chan_attn_num_head=4,
                 chan_ratio=8,
                 chan_attn_type='SE',
                 aux=True,
                 use_faster=False):
        super().__init__()
        n_branch = len(in_chans) if isinstance(in_chans, tuple) else 1 if n_branch is None else n_branch
        use_pos = False if n_branch == 1 else use_pos
        self.n_classes = n_classes
        self.depths = depths
        patch_size = to_2tuple(patch_size)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.aux = aux

        self.build_MSEs_AMM = Build_multimodal_fuse_head(
            n_branch=n_branch,
            in_chans=in_chans,
            out_chans=head_out_chans,
            n_group=head_out_chans // 2 // group_dim,
            use_pos=use_pos,
            patch_size=patch_size,
            offset_scale=offset_scale,
            attn_drop=attn_drop_rate,
            qkv_bias=qkv_bias,
            chan_ratio=chan_ratio,
            n_heads=chan_attn_num_head,
            chan_attn_type=chan_attn_type,
            fuse_type=fuse_proj,
            embed_dim=embed_dim
        )

        self.build_pipeline = Build_backbone(
            patch_size=None if fuse_proj is None else patch_size,
            in_chans=head_out_chans,
            embed_dim=embed_dim,
            group_dim=group_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qk_ratio=qk_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            act_layer=act_layer,
            patch_norm=patch_norm,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            use_checkpoint=use_checkpoint,
            qkv_type=qkv_type,
            ffn_type=ffn_type) if not use_faster else CNN_backbone()

        self.build_neck = Build_neck(
            in_chans=embed_dim,
            out_chans=decoder_chans,
            depth=len(depths),
            pool_sizes=pool_sizes,
            norm_layer=fpn_norm_layer,
            act_layer=fpn_act_layer)

        self.build_decode_head = Build_decode_gate(
            in_chans=decoder_chans,
            head_chans=None,
            n_classes=n_classes,
            norm_layer=fpn_norm_layer,
            act_layer=fpn_act_layer,
            chan_ratio=chan_ratio,
            chan_attn_type=chan_attn_type,
            en_chans=head_out_chans)

        if self.aux:
            self.aux_out = nn.Sequential(
                nn.Conv2d(embed_dim * 2 ** (len(depths) - 2), decoder_chans // 2, kernel_size=3, padding=1, bias=False),
                creat_norm_layer(fpn_norm_layer, decoder_chans // 2),
                fpn_act_layer(inplace=True) if fpn_act_layer != nn.GELU else fpn_act_layer(),
                nn.Conv2d(decoder_chans // 2, n_classes, kernel_size=1)
            )
        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.build_MSEs_AMM(x)
        x = x if isinstance(x, tuple) else (x, x)
        x, back_bone_input = x
        x = self.build_pipeline(x)
        aux_x = x[-2] if self.aux else None
        x = self.build_neck(x)
        x = self.build_decode_head(back_bone_input, x)
        # x = x if not self.use_uper else F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

        aux = self.aux if self.training else False
        if aux:
            aux_x = self.aux_out(aux_x)
            aux_x = F.interpolate(aux_x, scale_factor=2 ** (len(self.depths)), mode='bilinear', align_corners=True)
            return x, aux_x
        else:
            return x
